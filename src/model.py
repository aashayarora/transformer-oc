import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from utils.object_condensation import ObjectCondensation
from utils.torch_geometric_interface import row_splits_from_strict_batch as batch_to_rowsplits

import pytorch_lightning as pl


class SparseNeighborAttention(nn.Module):
    """
    Sparse multi-head self-attention where each node attends only to its ΔR neighbors.

    Instead of a dense [N, N] matrix, we build a neighbor index of shape [N, k] where k is
    the (padded) maximum number of neighbors. Memory is O(N * k) — for N=60k and k~100
    this is ~100x smaller than the dense case.
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, neighbor_idx, neighbor_mask):
        """
        Args:
            x:             [N, hidden_dim]
            neighbor_idx:  [N, k]  — indices of neighbors (padded with 0)
            neighbor_mask: [N, k]  — True where the neighbor is a PAD (should be ignored)
        Returns:
            [N, hidden_dim]
        """
        N, k = neighbor_idx.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(N, H, D)                   # [N, H, D]
        k_all = self.k_proj(x)                              # [N, hidden_dim]
        v_all = self.v_proj(x)                              # [N, hidden_dim]

        # Gather neighbor keys and values: [N, k, hidden_dim]
        flat_idx = neighbor_idx.reshape(-1)                 # [N*k]
        k_neigh = k_all[flat_idx].view(N, k, H, D)         # [N, k, H, D]
        v_neigh = v_all[flat_idx].view(N, k, H, D)         # [N, k, H, D]

        # Attention scores: [N, H, k]
        q_exp = q.unsqueeze(2)                              # [N, H, 1, D]
        k_t   = k_neigh.permute(0, 2, 3, 1)                # [N, H, D, k]
        scores = torch.matmul(q_exp, k_t).squeeze(2) * self.scale  # [N, H, k]

        # Mask padding positions
        scores = scores.masked_fill(neighbor_mask.unsqueeze(1), float('-inf'))  # [N, H, k]

        attn = torch.softmax(scores, dim=-1)
        # Replace NaN rows (nodes with zero valid neighbors) with 0
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)                         # [N, H, k]

        # Weighted sum of values: [N, H, D]
        v_t = v_neigh.permute(0, 2, 1, 3)                  # [N, H, k, D]
        out = torch.matmul(attn.unsqueeze(2), v_t).squeeze(2)  # [N, H, D]
        out = out.reshape(N, self.hidden_dim)

        return self.out_proj(out)


class SparseTransformerLayer(nn.Module):
    """Pre-norm transformer layer using SparseNeighborAttention."""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = SparseNeighborAttention(hidden_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, neighbor_idx, neighbor_mask):
        x = x + self.attn(self.norm1(x), neighbor_idx, neighbor_mask)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerOCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, latent_dim, dropout=0.1,
                 dr_threshold=0.2, max_neighbors=64, attention_chunk_size=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dr_threshold = dr_threshold
        self.max_neighbors = max_neighbors

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.transformer_layers = nn.ModuleList([
            SparseTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.latent_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )

        self.beta_head = nn.Sequential(
            nn.Linear(self.hidden_dim, int(self.hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.hidden_dim // 2), 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.input_proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.input_proj.bias)

        with torch.no_grad():
            self.beta_head[-1].weight.mul_(0.1)
            self.beta_head[-1].bias.fill_(-3.0)

        nn.init.xavier_uniform_(self.latent_head[0].weight)
        nn.init.zeros_(self.latent_head[0].bias)
        nn.init.xavier_uniform_(self.latent_head[-1].weight)
        nn.init.zeros_(self.latent_head[-1].bias)

    # Normalisation constants matching dataset.py (49-feature case)
    # col 1: eta, col 2: phi
    # MD slots 0,1,3 → col offsets 13,22,40; each block 9 cols: anchor_x(+0), anchor_y(+1), anchor_z(+2)
    _MIN_VALS = torch.tensor([-3.4627, -2.4507, -3.1416, -0.2362, -0.7272, -0.8202, -0.8768, -0.5988,
                               -0.2704, -0.8894, -1.0590, -1.3935, -0.7862, -73.8012, -72.4212, -187.5750,
                               -73.8019, -77.1076, -187.8705, -0.0231, -0.5393, -5.0250, -93.8255, -93.9099,
                               -223.8540, -97.9344, -93.9060, -224.1495, -0.0111, -0.8231, -5.0255, -93.8255,
                               -93.9099, -223.8540, -97.9344, -93.9060, -224.1495, -0.0111, -0.8231, -5.0255,
                               -110.0160, -110.0150, -267.2350, -110.1943, -110.1950, -267.6350, -0.0046,
                               -0.9416, -5.0259])
    _MAX_VALS = torch.tensor([7.5964, 2.4535, 3.1416, 0.2434, 0.7304, 0.7744, 0.8728, 0.5682,
                               0.2755, 0.8922, 1.1500, 1.5453, 0.8093, 73.7524, 70.4150, 187.5750,
                               73.7676, 71.4665, 187.8705, 0.0228, 0.5575, 5.0250, 93.8869, 93.8406,
                               223.8540, 96.1440, 96.9953, 224.1495, 0.0097, 0.8098, 5.0253, 93.8869,
                               93.8406, 223.8540, 96.1440, 96.9953, 224.1495, 0.0097, 0.8098, 5.0253,
                               110.0128, 110.0150, 267.2350, 110.1814, 110.1950, 267.6350, 0.0050,
                               0.9323, 5.0260])
    _MD_OFFSETS = [13, 22, 40]   # inner, middle, outer anchor col offsets
    _N_MDS = 3

    def _denorm(self, x_raw, col):
        mn = self._MIN_VALS[col].to(x_raw.device)
        mx = self._MAX_VALS[col].to(x_raw.device)
        return x_raw[:, col] * (mx - mn + 1e-8) + mn

    def _md_coords(self, x_raw):
        """
        Extract per-MD anchor (r, eta, phi) for the 3 unique layers of each T3.
        Returns r, eta, phi each of shape [N, 3].
        """
        ax = torch.stack([self._denorm(x_raw, off)     for off in self._MD_OFFSETS], dim=1)
        ay = torch.stack([self._denorm(x_raw, off + 1) for off in self._MD_OFFSETS], dim=1)
        az = torch.stack([self._denorm(x_raw, off + 2) for off in self._MD_OFFSETS], dim=1)
        r   = torch.sqrt(ax**2 + ay**2)
        phi = torch.atan2(ay, ax)
        theta = torch.atan2(r, az)
        eta = -torch.log(torch.clamp(torch.tan(theta / 2.0), min=1e-6))
        return r, eta, phi

    @torch.no_grad()
    def build_neighbor_index(self, x_raw, batch):
        """
        Build neighbor index using closest-radius MD anchor ΔR.

        For each pair of T3s (i, j) we compare all 3×3 = 9 combinations of
        anchor hits (inner/middle/outer from each T3), select the pair with
        the smallest |Δr| (closest in transverse radius), and use the (η, φ)
        of those two anchors to compute ΔR.

        Returns:
            neighbor_idx:  [N, k]  int64 — neighbor indices, padded with 0
            neighbor_mask: [N, k]  bool  — True = padding (ignore in attention)
        """
        device = x_raw.device
        N = x_raw.shape[0]
        k = self.max_neighbors
        CROSS_GRAPH_PENALTY = 1e6

        # MD anchor coords: each [N, 3]
        r_md, eta_md, phi_md = self._md_coords(x_raw)

        neighbor_idx  = torch.zeros(N, k, dtype=torch.long,  device=device)
        neighbor_mask = torch.ones (N, k, dtype=torch.bool,   device=device)

        chunk_size = 512  # keep [chunk, N, 3, 3] tensors manageable
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            C = i_end - i_start

            # All 3x3 |Δr| combinations: [C, N, 3, 3]
            r_i    = r_md[i_start:i_end].view(C, 1, self._N_MDS, 1)   # [C,1,3,1]
            r_j    = r_md.view(1, N, 1, self._N_MDS)                   # [1,N,1,3]
            dr_r   = torch.abs(r_i - r_j)                              # [C,N,3,3]

            # Best (closest-radius) MD pair per T3-T3 combination
            dr_r_flat = dr_r.view(C, N, self._N_MDS * self._N_MDS)     # [C,N,9]
            best_idx  = dr_r_flat.argmin(dim=-1)                        # [C,N]
            md_i_best = best_idx // self._N_MDS                         # [C,N]
            md_j_best = best_idx %  self._N_MDS                         # [C,N]

            # Gather η and φ for the chosen MD on each side
            row_i = torch.arange(C, device=device)
            row_j = torch.arange(N, device=device)

            # eta/phi for query T3s: index [C,N] into [C,3] → [C,N]
            eta_i = eta_md[i_start:i_end][row_i.unsqueeze(1), md_i_best]  # [C,N]
            phi_i = phi_md[i_start:i_end][row_i.unsqueeze(1), md_i_best]
            eta_j = eta_md[row_j.unsqueeze(0), md_j_best]                 # [C,N]
            phi_j = phi_md[row_j.unsqueeze(0), md_j_best]

            deta = eta_i - eta_j
            dphi = torch.atan2(torch.sin(phi_i - phi_j), torch.cos(phi_i - phi_j))
            dr   = torch.sqrt(deta**2 + dphi**2)                           # [C,N]

            # Penalize cross-graph pairs
            cross = (batch[i_start:i_end].unsqueeze(1) != batch.unsqueeze(0))
            dr    = dr + cross.float() * CROSS_GRAPH_PENALTY

            actual_k = min(k, N)
            topk_dr, topk_col = torch.topk(dr, actual_k, dim=1, largest=False)

            valid = topk_dr <= self.dr_threshold

            no_valid = ~valid.any(dim=1)
            if no_valid.any():
                topk_col[no_valid, 0] = torch.arange(i_start, i_end, device=device)[no_valid]
                valid[no_valid, 0] = True

            neighbor_idx [i_start:i_end, :actual_k] = topk_col
            neighbor_mask[i_start:i_end, :actual_k] = ~valid

        return neighbor_idx, neighbor_mask

    def forward(self, x_raw, batch):
        neighbor_idx, neighbor_mask = self.build_neighbor_index(x_raw, batch)

        x = self.input_proj(x_raw)

        for layer in self.transformer_layers:
            x = layer(x, neighbor_idx, neighbor_mask)

        coords_latent = self.latent_head(x)
        coords_latent = F.normalize(coords_latent, p=2, dim=-1)

        eps = 1e-6
        beta = self.beta_head(x).sigmoid()
        beta = torch.clamp(beta, eps, 1 - eps)

        return {"B": beta, "H": coords_latent}
    

class TransformerLightningModule(pl.LightningModule):
    def __init__(self, config, input_dim):
        super().__init__()
        
        hparams = {
            'input_dim': input_dim,
            'hidden_dim': config.get('hidden_dim', config.get('d_model', 128)),
            'num_layers': config['num_layers'],
            'num_heads': config['nhead'],
            'latent_dim': config['latent_dim'],
            'dropout': config.get('dropout', 0.1),
            'dr_threshold': config.get('dr_threshold', 0.2),
            'max_neighbors': config.get('max_neighbors', 64),
            
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'scheduler_gamma': config['scheduler_gamma'],
            'scheduler_milestones': config.get('scheduler_milestones', [10]),
            
            'loss_weight_attractive': config.get('loss_weight_attractive', 1.0),
            'loss_weight_repulsive': config.get('loss_weight_repulsive', 1.0),
            'loss_weight_beta': config.get('loss_weight_beta', 1.0),
            
            'oc_q_min': config.get('oc_q_min', 0.1),
            'oc_s_B': config.get('oc_s_B', 1.0),
            'oc_repulsive_chunk_size': config.get('oc_repulsive_chunk_size', 32),
            
            'epochs': config.get('epochs', 100),
            'gradient_clip_val': config.get('gradient_clip_val', 1.0),
            'accumulate_grad_batches': config.get('accumulate_grad_batches', 1),
            'precision': config.get('precision', 32),
            
            'train_subset': config.get('train_subset', None),
            'val_subset': config.get('val_subset', None),
            
            'seed': config.get('seed', 42),
        }
        self.save_hyperparameters(hparams)
        
        self.config = config
        
        self.model = TransformerOCModel(
            input_dim=input_dim,
            hidden_dim=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            latent_dim=self.hparams.latent_dim,
            dropout=self.hparams.dropout,
            dr_threshold=self.hparams.dr_threshold,
            max_neighbors=self.hparams.max_neighbors,
        )
        
        self.criterion = ObjectCondensation(
            q_min=self.hparams.oc_q_min,
            s_B=self.hparams.oc_s_B,
            repulsive_chunk_size=self.hparams.oc_repulsive_chunk_size,
        )
    
    def training_step(self, data):
        x, sim_index, batch = data.x.to(self.device), data.sim_index.to(self.device), data.batch.to(self.device)
        sim_index[sim_index < 0] = -1
        row_splits = batch_to_rowsplits(batch)
        
        model_out = self.model(x, batch)
        beta = model_out["B"].float()
        coords = model_out["H"].float()

        L_att, L_rep, L_beta, _, _ = self.criterion(
            beta=beta,
            coords=coords,
            asso_idx = sim_index.unsqueeze(-1).to(torch.int64),
            row_splits = row_splits
        )
        
        tot_loss_batch = (self.hparams.loss_weight_attractive * L_att + 
                         self.hparams.loss_weight_repulsive * L_rep + 
                         self.hparams.loss_weight_beta * L_beta)

        self.log('train_loss', tot_loss_batch, on_step=True, on_epoch=True, prog_bar=True, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_attractive', L_att, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_repulsive', L_rep, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_beta', L_beta, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        
        self.log('train_beta_mean', beta.mean(), on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_beta_std', beta.std(), on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        
        del model_out, L_att, L_rep, L_beta
        
        return tot_loss_batch

    def validation_step(self, data):
        x, sim_index, batch = data.x.to(self.device), data.sim_index.to(self.device), data.batch.to(self.device)
        sim_index[sim_index < 0] = -1
        row_splits = batch_to_rowsplits(batch)

        model_out = self.model(x, batch)
        beta = model_out["B"].float()
        coords = model_out["H"].float()

        L_att, L_rep, L_beta, _, _ = self.criterion(
            beta=beta,
            coords=coords,
            asso_idx = sim_index.unsqueeze(-1).to(torch.int64),
            row_splits = row_splits
        )
        
        tot_loss_batch = (self.hparams.loss_weight_attractive * L_att + 
                         self.hparams.loss_weight_repulsive * L_rep + 
                         self.hparams.loss_weight_beta * L_beta)

        self.log('val_loss', tot_loss_batch, on_step=False, on_epoch=True, prog_bar=True, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_attractive', L_att, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_repulsive', L_rep, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_beta', L_beta, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        
        self.log('val_beta_mean', beta.mean(), on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_beta_std', beta.std(), on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)

        del model_out, L_att, L_rep, L_beta
        
        return tot_loss_batch
    
    def on_before_optimizer_step(self, optimizer):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.log('grad_norm', total_norm, on_step=True, on_epoch=False, prog_bar=False)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.hparams.scheduler_milestones,
            gamma=self.hparams.scheduler_gamma
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss',
            }
        }