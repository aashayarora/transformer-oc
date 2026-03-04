import torch
import torch.backends.cuda
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True) 
torch.backends.cuda.enable_cudnn_sdp(False)

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from utils.object_condensation import ObjectCondensation
# from fastgraphcompute.object_condensation import ObjectCondensation
from utils.torch_geometric_interface import row_splits_from_strict_batch as batch_to_rowsplits

import pytorch_lightning as pl

class TransformerOCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, latent_dim, dropout=0.1, 
                 dr_threshold=0.2, attention_chunk_size=1024):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dr_threshold = dr_threshold
        self.attention_chunk_size = attention_chunk_size

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=self.hidden_dim * 2,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])

        self.input_proj = nn.Linear(input_dim, hidden_dim)

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
    # col 0: log(pt),  col 1: eta,  col 2: phi
    # cols 13-48: MD features (4 MDs × 9): anchor_x, anchor_y, anchor_z, other_x, ...
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

    def _denorm(self, x_raw, col):
        """Denormalise a single column back to its physical value."""
        mn = self._MIN_VALS[col].to(x_raw.device)
        mx = self._MAX_VALS[col].to(x_raw.device)
        return x_raw[:, col] * (mx - mn + 1e-8) + mn

    def compute_dr_mask(self, x_raw, batch):
        """
        Compute ΔR-based attention mask using helix propagation.

        All hits are propagated to a common reference radius r_target using
        the barrel helix formula, then masked by ΔR in (η, φ_propagated) space.

        Helix φ propagation (small-angle, constant-B approximation):
            Δφ = arcsin( (r_target² - r_init²) / (2 · r_init · R_helix) )
        where R_helix = pT / (0.3 · B) [pT in GeV, R in cm, B = 3.8 T].

        r_init is estimated as the mean transverse radius of the 4 MD anchor hits.
        """
        device = x_raw.device
        N = x_raw.shape[0]

        B_FIELD = 3.8        # Tesla
        R_TARGET = 60.0      # cm — roughly middle of the tracker

        # --- Denormalise physics quantities ---
        # col 0 is log(pt) normalised; denorm then exp to get pT in GeV
        log_pt = self._denorm(x_raw, 0)
        pt = torch.exp(log_pt).clamp(min=0.5)   # GeV; clamp avoids huge R for near-zero pT

        eta = self._denorm(x_raw, 1)
        phi = self._denorm(x_raw, 2)

        # --- Compute r_init from the 4 MD anchor (x,y) positions ---
        # MD feature block starts at col 13; each MD occupies 9 cols:
        #   anchor_x(+0), anchor_y(+1), anchor_z(+2), other_x(+3), ...
        # MD0: cols 13,14  MD1: cols 22,23  MD2: cols 31,32  MD3: cols 40,41
        md_offsets = [13, 22, 31, 40]
        r_vals = []
        for off in md_offsets:
            ax = self._denorm(x_raw, off)       # anchor_x [cm]
            ay = self._denorm(x_raw, off + 1)   # anchor_y [cm]
            r_vals.append(torch.sqrt(ax**2 + ay**2))
        r_init = torch.stack(r_vals, dim=1).mean(dim=1)   # [N]

        # --- Helix radius [cm] ---
        R_helix = pt / (0.3 * B_FIELD)   # R = pT[GeV] / (0.3 · B[T])  in cm

        # --- Propagate φ to r_target ---
        # arg = (r_target² - r_init²) / (2 · r_init · R_helix)
        # clamped to [-1, 1] for numerical safety
        arg = (R_TARGET**2 - r_init**2) / (2.0 * r_init * R_helix + 1e-6)
        arg = arg.clamp(-1.0, 1.0)
        delta_phi = torch.asin(arg)
        phi_prop = phi + delta_phi                            # propagated φ [rad]

        # --- η is unchanged (straight line along z for barrel) ---
        eta_prop = eta

        # --- Chunked ΔR mask in propagated (η, φ) space ---
        chunk_size = self.attention_chunk_size
        attention_mask = torch.full((N, N), float('-inf'), device=device, dtype=torch.float32)

        for i in range(0, N, chunk_size):
            i_end = min(i + chunk_size, N)

            eta_chunk = eta_prop[i:i_end].unsqueeze(1)   # [chunk, 1]
            deta_abs = torch.abs(eta_chunk - eta_prop.unsqueeze(0))   # [chunk, N]

            potential_neighbors = deta_abs <= self.dr_threshold

            if potential_neighbors.any():
                phi_chunk = phi_prop[i:i_end].unsqueeze(1)   # [chunk, 1]
                dphi = phi_chunk - phi_prop.unsqueeze(0)      # [chunk, N]
                dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))

                deta = eta_chunk - eta_prop.unsqueeze(0)
                dr = torch.sqrt(deta**2 + dphi**2)

                attention_mask[i:i_end, :] = torch.where(dr <= self.dr_threshold, 0.0, float('-inf'))

                del dphi, deta, dr

            del deta_abs, potential_neighbors

        return attention_mask

    def forward(self, x_raw, batch):
        x = self.input_proj(x_raw)
        
        batch_size = batch.max().item() + 1
        device = x.device
        
        x_input = x.unsqueeze(0)  # [1, N, hidden_dim]
        
        attention_mask = self.compute_dr_mask(x_raw, batch)
        attention_mask = attention_mask.to(device)

        x_transformed = x_input
        for layer in self.transformer_layers:
            x_transformed = layer(x_transformed, src_mask=attention_mask)
        
        x_out = x_transformed.squeeze(0)

        coords_latent = self.latent_head(x_out)
        coords_latent = torch.nn.functional.normalize(coords_latent, p=2, dim=-1)

        eps = 1e-6
        beta = self.beta_head(x_out).sigmoid()
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
            'attention_chunk_size': config.get('attention_chunk_size', 512),
            
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
            attention_chunk_size=self.hparams.attention_chunk_size
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