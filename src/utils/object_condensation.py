import torch

from fastgraphcompute.extensions.oc_helper import oc_helper_matrices, select_with_default
from utils.torch_geometric_interface import strict_batch_from_row_splits

def arctanhsq(x):
    return torch.arctanh(x)**2

class ObjectCondensation(torch.nn.Module):

    def __init__(self,
                 q_min = 0.1,
                 s_B = 1.,
                 norm_payload = True,
                 weighted_obj_coordinates = 0.,
                 fixed_repulsive_norm = None,
                 beta_scaling_epsilon = 1e-3,
                 v_beta_scaling = arctanhsq,
                 p_beta_scaling = arctanhsq,
                 repulsive_chunk_size = 32,
                 repulsive_distance_cutoff = None,
                 use_checkpointing = False,
                 **kwargs) -> None:
        '''
        Initializes the ObjectCondensation loss module.
        
        This module implements the object condensation loss as described in the paper:
        "Object condensation: one-stage grid-free multi-object reconstruction
        in physics detectors, graph and image data" [arXiv:2002.03605].
        
        Parameters:
            q_min (float): 
                Minimum charge for object condensation potentials. 
                
            s_B (float): 
                Scaling factor for the noise term of the beta potential.
        
            norm_payload (bool): 
                Whether to normalize the payload loss per object.
                If True, the payload loss is divided by the total object contribution
                to normalize it.
        
            weighted_obj_coordinates (float): 
                Weighting factor for object coordinates. 
                Defines the ratio of object-average coordinates to alpha-selected coordinates.
                Must be in the range [0, 1]. For non-zero values, this feature goes beyond
                what is described in the paper.
        
            fixed_repulsive_norm (float, optional): 
                Fixed normalization value for the repulsive loss. 
                If None, the normalization will be calculated dynamically based on 
                the number of points per object.
        
            beta_scaling_epsilon (float): 
                Small epsilon value to stabilize beta scaling calculations.
        
            v_beta_scaling (callable): 
                Function to scale the beta values for the potential loss.
                Default is `arctanhsq`, which computes the square of the hyperbolic arctangent.
        
            p_beta_scaling (callable): 
                Function to scale the beta values for the payload loss.
                Default is `arctanhsq`, which computes the square of the hyperbolic arctangent.
        
            repulsive_chunk_size (int):
                Number of objects to process at once in repulsive potential calculation.
                Lower values use less memory but may be slower. Default: 32.
        
            repulsive_distance_cutoff (float, optional):
                Maximum distance for repulsive potential. Objects farther apart are ignored.
                This can dramatically reduce memory and compute. Default: None (no cutoff).
        
            use_checkpointing (bool):
                Whether to use gradient checkpointing for memory efficiency during backprop.
                Trades compute for memory. Default: False.
        
            **kwargs: 
                Additional keyword arguments passed to the parent `torch.nn.Module`.
        
        Raises:
            AssertionError: 
                If `weighted_obj_coordinates` is not in the range [0, 1].
        
        Forward Method:
            Inputs:
                beta (torch.Tensor): 
                    Tensor of shape (N, 1) containing the beta values for each point. 
                    Beta values determine the point's contribution to object formation 
                    and are limited to the range [0, 1].
                
                coords (torch.Tensor): 
                    Tensor of shape (N, C) containing the spatial coordinates of the points. 
                    These coordinates represent the cluster positions in a C-dimensional space.
                
                asso_idx (torch.Tensor): 
                    Tensor of shape (N, 1) containing association indices for each point. 
                    Positive integers correspond to the object a point belongs to, while -1 
                    marks noise points that do not belong to any object.
                
                row_splits (torch.Tensor): 
                    Tensor of shape (N_rs,) defining the row splits for batching. Each entry in 
                    `row_splits` indicates the start and end indices of a batch in the data.
            
            What It Does:
                - Constructs the object condensation loss components:
                    1. Attractive potential loss (`L_V`): Encourages points to group into objects.
                    2. Repulsive potential loss (`L_rep`): Discourages overlapping objects.
                    3. Beta loss (`L_b`): Enforces a continuous maximum on beta and limits noise.
                - Scales and normalizes these losses per object and batch.
                - Returns scaling factors for payloads and losses for further downstream use.
            
            Returns:
                tuple: 
                    - L_V (torch.Tensor): Scalar attractive potential loss.
                    - L_rep (torch.Tensor): Scalar repulsive potential loss.
                    - L_b (torch.Tensor): Scalar beta loss, including noise penalty.
                    - pl_scaling (torch.Tensor): Tensor of shape (N, 1), scaling factor for payload loss. 
                                                 All normalisation is already applied. Just use as scaling 
                                                 factor on a per-point basis and sum all points in the 
                                                 batch to get the total payload loss.
                    - L_V_rep (torch.Tensor): Tensor of shape (N, 1), combined per-point potential loss.
                                              Can be used, e.g. for visualization or monitoring purposes.
        '''

        self.q_min = q_min
        self.s_B = s_B
        self.beta_scaling_epsilon = beta_scaling_epsilon
        self.v_beta_scaling = v_beta_scaling
        self.p_beta_scaling = p_beta_scaling
        self.repulsive_chunk_size = repulsive_chunk_size
        self.repulsive_distance_cutoff = repulsive_distance_cutoff
        self.use_checkpointing = use_checkpointing

        self.pl_norm = self._no_op if not norm_payload else self._norm_payload
        self.rep_norm = self._norm_repulsive_fixed if fixed_repulsive_norm is not None else self._norm_repulsive
        self.fixed_repulsive_norm = fixed_repulsive_norm

        assert 0. <= weighted_obj_coordinates <= 1.
        self.weighted_obj_coordinates = weighted_obj_coordinates

        super().__init__(**kwargs)

    def _no_op(self, x, *args):
        return x

    def _norm_payload(self, payload_loss_k_m, M):
        # clip payload_loss_k_m to avoid NaNs, do not add an epsilon, as this would bias the result
        payload_loss_k_m = torch.clamp(payload_loss_k_m, min=1e-6)
        return payload_loss_k_m / (torch.sum(payload_loss_k_m, dim=1, keepdim=True))

    def _norm_repulsive_fixed(self, rep_loss, N_k):
        # rep_loss as K x 1, N_prs as K x 1
        return rep_loss / self.fixed_repulsive_norm

    def _norm_repulsive(self, rep_loss, N_k):
        return rep_loss / N_k
    
    def _scatter_to_N_indices(self, x_k_m, asso_indices, M):
        '''
        Inputs: 
            x_k_1 (torch.Tensor): The values to scatter, shape (K, M, 1)
        '''
        # Step 1: Use select_with_default to get indices
        M2 = select_with_default(M, torch.arange(asso_indices.size(0), device=asso_indices.device, dtype=torch.int64).view(-1, 1), -1)
        
        # Step 3: Flatten valid entries in M2
        valid_mask = M2 >= 0
        n_flat = x_k_m[valid_mask].view(-1)  # Flatten valid points
        m_flat = M2[valid_mask].view(-1)  # Flatten corresponding indices

        assert torch.max(m_flat) < asso_indices.size(0), "m_flat contains out-of-bounds indices."
        
        # Step 4: Scatter the values back to all points
        x = torch.zeros_like(asso_indices, dtype=x_k_m.dtype)
        x[m_flat] = n_flat
        return x
    
    def _mean_per_row_split(self, x, row_splits):
        '''
        x : N x 1
        row_splits : N_rs
        '''
        # Calculate lengths of each row split
        lengths = row_splits[1:] - row_splits[:-1]
        
        # Create indices for each row split
        row_indices = torch.repeat_interleave(torch.arange(len(lengths), device=row_splits.device, dtype=torch.int64), lengths)
        
        # Calculate sum per row split - use x.dtype instead of hardcoded float32
        x_squeezed = x.squeeze(1)
        sum_per_split = torch.zeros(len(lengths), dtype=x_squeezed.dtype, device=row_splits.device).scatter_add(0, row_indices, x_squeezed)
        return sum_per_split / lengths
    
    def _beta_loss(self, beta_k_m):
        """
        Calculate the beta penalty using a continuous max approximation via LogSumExp
        and an additional penalty term for faster convergence.
    
        Args:
            beta_k_m (torch.Tensor): Tensor of shape (K, M, 1) containing the beta values.
    
        Returns:
            torch.Tensor: Tensor of shape (K, 1) containing the beta penalties.
        """
        eps = 1e-3
        # Continuous max approximation using LogSumExp
        beta_pen = 1. - eps * torch.logsumexp(beta_k_m / eps, dim=1)  # Sum over M

        # Add penalty for faster convergence
        beta_pen += 1. - torch.clamp(torch.sum(beta_k_m, dim=1), min=0., max=1.)
        return beta_pen


    def get_alpha_indices(self, beta_k_m, M):
        '''
        Calculates the arg max of beta_k_m in dimension 1.
        '''
        m_idxs = torch.argmax(beta_k_m, dim=1).squeeze(1)
        return M[torch.arange(M.size(0), dtype=torch.int64), m_idxs]

    def V_repulsive_func(self, distsq):
        '''
        Calculates the repulsive potential function.
        It is in a dedicated function to allow for easy replacement in inherited classes.
        '''
        # Use a saturating hinge function to prevent unbounded growth
        dist = torch.sqrt(distsq + 1e-6)
        return torch.relu(1. - dist)
    
    def V_attractive_func(self, distsq):
        '''
        Calculates the attractive potential function.
        It is in a dedicated function to allow for easy replacement in inherited classes.
        '''
        return distsq
    
    def alpha_features(self, x, x_k_m, alpha_indices):
        '''
        Returns the features of the alphas.
        In other implementations, this can be a weighted function.

        Returns K, 1, F where F is the number of features.
        '''
        x_a = (1. - self.weighted_obj_coordinates ) * x[alpha_indices]
        x_a = x_a + self.weighted_obj_coordinates * torch.mean(x_k_m, dim=1)
        return x_a.view(-1, 1, x.size(1))
    
    def _attractive_potential(self, beta_scale, beta_scale_k, coords_k, coords_k_m, M):
        '''
        returns the attractive potential loss for each object. # K x 1
        Memory-optimized version.
        '''
        beta_scale_k_m = select_with_default(M, beta_scale, 0.) # K x M x 1
        # mask
        mask_k_m = select_with_default(M, torch.ones_like(beta_scale), 0.) # K x M x 1
        
        ## Attractive potential
        # get the distances - compute in-place where possible
        distsq_k_m = coords_k - coords_k_m  # K x M x C
        distsq_k_m = distsq_k_m.pow_(2)  # In-place square
        distsq_k_m = torch.sum(distsq_k_m, dim=2, keepdim=True) # K x M x 1
        
        # get the attractive potential
        V_attractive = self.V_attractive_func(distsq_k_m)  # K x M x 1
        V_attractive.mul_(beta_scale_k_m)  # In-place multiply
        V_attractive.mul_(beta_scale_k.view(-1, 1, 1))  # In-place multiply
        V_attractive.mul_(mask_k_m)  # In-place multiply
        
        return torch.sum(V_attractive, dim=1) / (torch.sum(mask_k_m, dim=1) + 1e-6) # K x 1

    def _repulsive_potential(self, beta_scale, beta_scale_k, coords, coords_k, M_not):
        '''
        returns the repulsive potential loss for each object. # K x 1
        Memory-optimized version with chunking.
        '''
        K = coords_k.size(0)
        
        # Process in chunks to reduce memory
        rep_loss_chunks = []
        mask_sum_chunks = []
        
        for chunk_start in range(0, K, self.repulsive_chunk_size):
            chunk_size = min(self.repulsive_chunk_size, K - chunk_start)
            
            rep_loss_chunk, mask_sum_chunk = self._repulsive_potential_chunk(
                beta_scale, beta_scale_k, coords, coords_k, M_not,
                chunk_start, chunk_size
            )
            
            rep_loss_chunks.append(rep_loss_chunk)
            mask_sum_chunks.append(mask_sum_chunk)
        
        rep_loss = torch.cat(rep_loss_chunks, dim=0)
        mask_sum = torch.cat(mask_sum_chunks, dim=0)
        
        return self.rep_norm(rep_loss, mask_sum + 1e-6)
    
    def _repulsive_potential_chunk(self, beta_scale, beta_scale_k, coords, coords_k, M_not,
                                   chunk_start, chunk_size):
        '''
        Process a chunk of objects for repulsive potential.
        Returns unnormalized loss and mask sum for the chunk.
        Uses vectorized operations when possible, falls back to per-object processing for memory safety.
        '''
        chunk_end = chunk_start + chunk_size
        
        # Get chunk of M_not and coords_k
        M_not_chunk = M_not[chunk_start:chunk_end]
        coords_k_chunk = coords_k[chunk_start:chunk_end]
        beta_scale_k_chunk = beta_scale_k[chunk_start:chunk_end]
        
        # Try vectorized approach first (fast path)
        # This creates (chunk_size × N' × features) tensors
        # Only use if we estimate memory will be reasonable
        N_total = coords.size(0)
        C = coords.size(1)
        
        # More conservative memory estimate
        estimated_mb = (chunk_size * N_total * C * 4) / (1024 * 1024)
        
        # Dynamically determine memory threshold based on available GPU memory
        if torch.cuda.is_available() and coords.is_cuda:
            # Get available GPU memory (free memory)
            device = coords.device
            free_memory_mb = torch.cuda.mem_get_info(device)[0] / (1024 * 1024)
            # Use only 30% of free memory to be conservative
            memory_threshold_mb = free_memory_mb * 0.3
        else:
            # CPU fallback: use conservative 200 MB threshold
            memory_threshold_mb = 200
        
        # Use vectorized path if memory estimate is reasonable
        if estimated_mb < memory_threshold_mb:
            try:
                # Vectorized computation (original approach, but safer with smaller chunks)
                coords_k_n = select_with_default(M_not_chunk, coords, 0.)
                beta_scale_k_n = select_with_default(M_not_chunk, beta_scale, 0.)
                mask_k_n = select_with_default(M_not_chunk, torch.ones_like(beta_scale), 0.)
                
                # Compute distances
                distsq_k_n = torch.sum((coords_k_n - coords_k_chunk)**2, dim=2, keepdim=True)
                
                # Apply distance cutoff if specified
                if self.repulsive_distance_cutoff is not None:
                    cutoff_mask = distsq_k_n < (self.repulsive_distance_cutoff ** 2)
                    mask_k_n = mask_k_n * cutoff_mask.float()
                
                V_repulsive = mask_k_n * beta_scale_k_chunk.view(-1, 1, 1) * self.V_repulsive_func(distsq_k_n) * beta_scale_k_n
                
                # Return sum over N' for this chunk
                rep_loss_chunk = torch.sum(V_repulsive, dim=1)
                mask_sum_chunk = torch.sum(mask_k_n, dim=1)
                
                # Clean up immediately
                del coords_k_n, beta_scale_k_n, mask_k_n, distsq_k_n, V_repulsive
                
                return rep_loss_chunk, mask_sum_chunk
                
            except RuntimeError as e:
                # If we get OOM, fall back to slower but safer per-object processing
                if "out of memory" not in str(e).lower():
                    raise e
                # Clear memory and continue to fallback
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Fallback: per-object processing (slower but memory-safe)
        chunk_rep_loss = torch.zeros(chunk_size, 1, device=coords.device, dtype=coords.dtype)
        chunk_mask_sum = torch.zeros(chunk_size, 1, device=coords.device, dtype=coords.dtype)
        
        max_points_per_batch = 5000  # Reduced back to 5000 for safety
        
        # Process each object individually
        for i in range(chunk_size):
            indices_not = M_not_chunk[i]
            
            # Filter valid indices (remove padding -1s)
            valid_mask = indices_not >= 0
            if not valid_mask.any():
                continue
                
            valid_indices = indices_not[valid_mask]
            num_valid = len(valid_indices)
            
            # If too many points, process in sub-batches
            if num_valid > max_points_per_batch:
                total_loss = 0.0
                total_count = 0
                
                for batch_start in range(0, num_valid, max_points_per_batch):
                    batch_end = min(batch_start + max_points_per_batch, num_valid)
                    batch_indices = valid_indices[batch_start:batch_end]
                    
                    coords_batch = coords[batch_indices]
                    beta_batch = beta_scale[batch_indices]
                    
                    distsq = torch.sum((coords_batch - coords_k_chunk[i])**2, dim=1, keepdim=True)
                    
                    if self.repulsive_distance_cutoff is not None:
                        cutoff_mask = distsq < (self.repulsive_distance_cutoff ** 2)
                        distsq = distsq[cutoff_mask]
                        beta_batch = beta_batch[cutoff_mask]
                        batch_count = cutoff_mask.sum().item()
                    else:
                        batch_count = len(batch_indices)
                    
                    if batch_count > 0:
                        V_rep = beta_scale_k_chunk[i] * self.V_repulsive_func(distsq) * beta_batch
                        total_loss += V_rep.sum().item()
                        total_count += batch_count
                    
                    # Clean up batch tensors
                    del coords_batch, beta_batch, distsq
                    if self.repulsive_distance_cutoff is not None:
                        del cutoff_mask
                
                chunk_rep_loss[i] = total_loss
                chunk_mask_sum[i] = total_count
                
            else:
                # Process all points at once if small enough
                coords_not_obj = coords[valid_indices]
                beta_not_obj = beta_scale[valid_indices]
                
                distsq = torch.sum((coords_not_obj - coords_k_chunk[i])**2, dim=1, keepdim=True)
                
                if self.repulsive_distance_cutoff is not None:
                    cutoff_mask = distsq < (self.repulsive_distance_cutoff ** 2)
                    if not cutoff_mask.any():
                        continue
                    distsq = distsq[cutoff_mask]
                    beta_not_obj = beta_not_obj[cutoff_mask]
                    num_valid = cutoff_mask.sum().item()
                
                V_rep = beta_scale_k_chunk[i] * self.V_repulsive_func(distsq) * beta_not_obj
                
                chunk_rep_loss[i] = V_rep.sum()
                chunk_mask_sum[i] = num_valid
                
                # Clean up
                del coords_not_obj, beta_not_obj, distsq, V_rep
        
        return chunk_rep_loss, chunk_mask_sum
        
    def _payload_scaling(self, beta, asso_idx, K_k, M):
        '''
        returns payload scaling for each POINT. # N x 1
        All normalisation is in there.
        '''
        ## Payload scaling
        pl_scaling = self.p_beta_scaling(beta / (1. + self.beta_scaling_epsilon))
        pl_scaling_k_m = select_with_default(M, pl_scaling, 0.) # K x M x 1
        pl_scaling_k_m = self.pl_norm(pl_scaling_k_m, M) # K x M x 1
        #normalise w.r.t K
        pl_scaling_k_m = pl_scaling_k_m / K_k.view(-1, 1, 1)

        return self._scatter_to_N_indices(pl_scaling_k_m, asso_idx, M) # N x 1


    def _compute_potentials(self, beta_scale, beta_scale_k, coords, coords_k, coords_k_m, M, M_not):
        '''
        Helper function for computing attractive and repulsive potentials.
        Can be used with gradient checkpointing.
        '''
        L_V_k = self._attractive_potential(beta_scale, beta_scale_k, coords_k, coords_k_m, M)
        L_rep_k = self._repulsive_potential(beta_scale, beta_scale_k, coords, coords_k, M_not)
        return L_V_k, L_rep_k

    def forward(self, beta, coords, asso_idx, row_splits):
        '''
        Inputs:
            beta (torch.Tensor): The beta values for each object, shape (N, 1)
            coords (torch.Tensor): The cluster coordinates of the objects, shape (N, C)
            asso_idx (torch.Tensor): The association indices of the objects, shape (N, 1)
                                        By convention, noise is marked by and index of -1.
                                        All objects have indices >= 0.
            row_splits (torch.Tensor): The row splits tensor, shape (N_rs)
        '''
        #check inputs
        assert beta.dim() == 2 and beta.size(1) == 1
        assert coords.dim() == 2 and coords.size(1) >= 1
        assert asso_idx.dim() == 2 and asso_idx.size(1) == 1

        asso_idx = asso_idx.squeeze(1)

        beta_scale = self.v_beta_scaling(beta / (1. + self.beta_scaling_epsilon)) + self.q_min # K x 1

        # get the matrices, row splits will be encoded in M and M_not
        # and are not needed after this
        M, M_not, obj_per_rs = oc_helper_matrices(asso_idx, row_splits) # M is (K, M), M_not is (K, N_prs)

        beta_k_m = select_with_default(M, beta, 0.) # K x M x 1
        # get argmax in dim 1 of beta_scale_k_m
        alpha_indices = self.get_alpha_indices(beta_k_m, M)

        # get the coordinates of the alphas
        coords_k_m = select_with_default(M, coords, 0.)
        coords_k = self.alpha_features(coords, coords_k_m, alpha_indices) # K x 1 x C
        beta_scale_k = beta_scale[alpha_indices] # K x 1

        # Compute potentials with optional gradient checkpointing
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            L_V_k, L_rep_k = checkpoint(
                self._compute_potentials,
                beta_scale, beta_scale_k, coords, coords_k, coords_k_m, M, M_not,
                use_reentrant=False
            )
        else:
            L_V_k, L_rep_k = self._compute_potentials(
                beta_scale, beta_scale_k, coords, coords_k, coords_k_m, M, M_not
            )

        # Use repeat_interleave to assign batch indices
        batch_idx = strict_batch_from_row_splits(row_splits)
        K_k = select_with_default(M, obj_per_rs[batch_idx].view(-1,1), 1)[:,0] #for normalisation, (K, 1)

        # for proper normalisation, we also need to adjust for the number of batch elements
        K_k = K_k * float(row_splits.shape[0] - 1) #never zero

        # mean over V' and mean over K
        L_V = torch.sum(L_V_k / K_k ) # scalar

        # mean over N and mean over K
        L_rep = torch.sum(L_rep_k / K_k) # scalar

        #scatter back V = V_attractive + V_repulsive and pl_scaling to (N, 1)
        L_k_m = (L_V_k + L_rep_k).view(-1, 1, 1).repeat(1, M.size(1), 1)
        L_V_rep_N = self._scatter_to_N_indices(L_k_m, asso_idx, M).view(-1,1) # N x 1

        # create the beta loss
        L_b_k = self._beta_loss(beta_k_m) # K x 1
        L_b = torch.sum(L_b_k / K_k) # scalar

        # add noise term
        L_noise = self.s_B *  beta * (asso_idx < 0).view(-1,1)  #norm per rs here too
        L_noise = torch.mean(self._mean_per_row_split(L_noise, row_splits))

        L_b = L_b + L_noise

        pl_scaling = self._payload_scaling(beta, asso_idx, K_k, M).view(-1,1) # N x 1

        return L_V, L_rep, L_b, pl_scaling, L_V_rep_N

