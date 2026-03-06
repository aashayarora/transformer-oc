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

    @torch.no_grad()
    def build_neighbor_index(self, x_raw, batch):
        """
        Fully vectorized ΔR neighbor index construction — no Python loops over nodes.

        Processes nodes in chunks of `chunk_size` rows at a time to keep peak
        memory at O(N * chunk_size) instead of O(N²).

        Returns:
            neighbor_idx:  [N, k]  int64 — neighbor indices, padded with 0
            neighbor_mask: [N, k]  bool  — True = padding (ignore in attention)
        """
        device = x_raw.device
        N = x_raw.shape[0]
        k = self.max_neighbors

        eta_norm = x_raw[:, 1]
        phi_norm = x_raw[:, 2]
        eta_min, eta_max = -2.62, 2.62
        phi_min, phi_max = -3.1416, 3.1416
        eta = eta_norm * (eta_max - eta_min) + eta_min  # [N]
        phi = phi_norm * (phi_max - phi_min) + phi_min  # [N]

        neighbor_idx  = torch.zeros(N, k, dtype=torch.long, device=device)
        neighbor_mask = torch.ones (N, k, dtype=torch.bool, device=device)

        # Use same-graph offset: two nodes are in the same graph iff batch[i]==batch[j].
        # Encode as a large additive penalty so cross-graph pairs always exceed threshold.
        # We add this to dr so cross-graph pairs are never selected.
        CROSS_GRAPH_PENALTY = 1e6

        chunk_size = 2048  # rows processed at once — tune for your GPU VRAM
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            C = i_end - i_start  # current chunk size

            eta_chunk = eta[i_start:i_end]  # [C]
            phi_chunk = phi[i_start:i_end]  # [C]
            batch_chunk = batch[i_start:i_end]  # [C]

            # ΔR: [C, N]
            deta = eta_chunk.unsqueeze(1) - eta.unsqueeze(0)
            dphi = phi_chunk.unsqueeze(1) - phi.unsqueeze(0)
            dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
            dr = torch.sqrt(deta ** 2 + dphi ** 2)  # [C, N]

            # Penalize cross-graph pairs so they're never picked
            cross = (batch_chunk.unsqueeze(1) != batch.unsqueeze(0))  # [C, N]
            dr = dr + cross.float() * CROSS_GRAPH_PENALTY

            # For each query node, find the k smallest dr values
            # Clamp k to N in case the graph is tiny
            actual_k = min(k, N)
            topk_dr, topk_col = torch.topk(dr, actual_k, dim=1, largest=False)  # [C, k]

            # Valid if within threshold (cross-graph pairs have dr >> threshold)
            valid = topk_dr <= self.dr_threshold  # [C, k]

            # Self-loop fallback for nodes with zero valid neighbors
            no_valid = ~valid.any(dim=1)  # [C]
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