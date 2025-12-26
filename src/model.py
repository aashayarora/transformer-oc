import torch
import torch.backends.cuda
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True) 
torch.backends.cuda.enable_cudnn_sdp(False)

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

# from utils.object_condensation import ObjectCondensation
from fastgraphcompute.object_condensation import ObjectCondensation
from utils.torch_geometric_interface import row_splits_from_strict_batch as batch_to_rowsplits

import pytorch_lightning as pl

class TransformerOCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, latent_dim, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

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

    def forward(self, x_raw, batch):
        x = self.input_proj(x_raw)
        
        batch_size = batch.max().item() + 1
        device = x.device
        
        counts = torch.bincount(batch, minlength=batch_size)
        max_nodes = counts.max().item()
        
        x_padded = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device, dtype=x.dtype)
        
        offsets = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
        offsets[1:] = counts.cumsum(0)
        position_in_graph = torch.arange(len(batch), device=device) - offsets[batch]
        
        x_padded[batch, position_in_graph] = x
        
        padding_mask = torch.arange(max_nodes, device=device).unsqueeze(0) >= counts.unsqueeze(1)
        
        x_transformed = x_padded
        for layer in self.transformer_layers:
            x_transformed = layer(x_transformed, src_key_padding_mask=padding_mask)
        
        x_out = x_transformed[batch, position_in_graph]

        coords_latent = self.latent_head(x_out)

        eps = 1e-6
        beta = self.beta_head(x_out).sigmoid()
        beta = torch.clamp(beta, eps, 1 - eps)
        
        return {"B": beta, "H": coords_latent}
    

class TransformerLightningModule(pl.LightningModule):
    def __init__(self, config, input_dim):
        super().__init__()
        self.save_hyperparameters(config)
        
        self.model = TransformerOCModel(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', config.get('d_model', 128)),
            num_layers=config['num_layers'],
            num_heads=config['nhead'],
            latent_dim=config['latent_dim'],
            dropout=config.get('dropout', 0.1),
        )

        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.scheduler_gamma = config['scheduler_gamma']
        
        self.loss_weight_attractive = config.get('loss_weight_attractive', 1.0)
        self.loss_weight_repulsive = config.get('loss_weight_repulsive', 1.0)
        self.loss_weight_beta = config.get('loss_weight_beta', 1.0)
        self.loss_weight_chi2 = config.get('loss_weight_chi2', 0.0)
        
        self.criterion = ObjectCondensation(
            q_min=config.get('oc_q_min', 0.1),
            s_B=config.get('oc_s_B', 1.0),
            # repulsive_chunk_size=config.get('oc_repulsive_chunk_size', 32),
            # repulsive_distance_cutoff=config.get('oc_repulsive_distance_cutoff', None),
            # use_checkpointing=config.get('oc_use_checkpointing', False)
        )
    
    def compute_chi2_loss(self, x, sim_index, batch):
        pt = x[:, 0]
        eta = x[:, 1]
        phi = x[:, 2]
        
        loss = 0.0
        count = 0
        
        for graph_id in batch.unique():
            mask = batch == graph_id
            graph_pt = pt[mask]
            graph_eta = eta[mask]
            graph_phi = phi[mask]
            graph_sim = sim_index[mask]
            
            valid_mask = graph_sim >= 0
            if valid_mask.sum() == 0:
                continue
            
            for track_id in graph_sim[valid_mask].unique():
                track_mask = (graph_sim == track_id) & valid_mask
                n_hits = track_mask.sum()
                
                if n_hits <= 2:
                    continue
                
                track_pt = graph_pt[track_mask]
                track_eta = graph_eta[track_mask]
                track_phi = graph_phi[track_mask]
                
                pt_mean = track_pt.mean()
                pt_chi2 = ((track_pt - pt_mean) ** 2).sum() / (pt_mean ** 2 + 1e-6)
                
                if n_hits >= 3:
                    phi_unwrapped = track_phi.clone()
                    phi_diff = phi_unwrapped[1:] - phi_unwrapped[:-1]
                    phi_diff = torch.where(phi_diff > torch.pi, phi_diff - 2*torch.pi, phi_diff)
                    phi_diff = torch.where(phi_diff < -torch.pi, phi_diff + 2*torch.pi, phi_diff)
                    phi_unwrapped[1:] = phi_unwrapped[0] + torch.cumsum(phi_diff, dim=0)
                    
                    phi_mean = phi_unwrapped.mean()
                    eta_mean = track_eta.mean()
                    
                    numerator = ((phi_unwrapped - phi_mean) * (track_eta - eta_mean)).sum()
                    denominator = ((phi_unwrapped - phi_mean) ** 2).sum() + 1e-6
                    slope = numerator / denominator
                    intercept = eta_mean - slope * phi_mean
                    
                    eta_pred = slope * phi_unwrapped + intercept
                    eta_chi2 = ((track_eta - eta_pred) ** 2).sum() / (n_hits - 2 + 1e-6)
                else:
                    eta_chi2 = torch.tensor(0.0, device=track_eta.device)
                
                if n_hits >= 3:
                    phi_diffs = track_phi[1:] - track_phi[:-1]
                    phi_diffs = torch.where(phi_diffs > torch.pi, phi_diffs - 2*torch.pi, phi_diffs)
                    phi_diffs = torch.where(phi_diffs < -torch.pi, phi_diffs + 2*torch.pi, phi_diffs)
                    
                    phi_curvature_chi2 = phi_diffs.var() / (phi_diffs.mean().abs() + 1e-6)
                else:
                    phi_curvature_chi2 = torch.tensor(0.0, device=track_phi.device)
                
                total_chi2 = pt_chi2 + eta_chi2 + phi_curvature_chi2
                loss += total_chi2 / 3.0 
                count += 1
        
        return loss / (count + 1e-6)
    
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
        
        # Physics-informed auxiliary loss
        L_chi2 = torch.tensor(0.0, device=self.device)
        
        if self.loss_weight_chi2 > 0:
            L_chi2 = self.compute_chi2_loss(x, sim_index, batch)
        
        tot_loss_batch = (self.loss_weight_attractive * L_att + 
                         self.loss_weight_repulsive * L_rep + 
                         self.loss_weight_beta * L_beta +
                         self.loss_weight_chi2 * L_chi2)

        self.log('train_loss', tot_loss_batch, on_step=True, on_epoch=True, prog_bar=True, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_attractive', L_att, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_repulsive', L_rep, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_beta', L_beta, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        
        if self.loss_weight_chi2 > 0:
            self.log('train_loss_chi2', L_chi2, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        
        del model_out, L_att, L_rep, L_beta, L_chi2
        
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
        
        # Physics-informed auxiliary loss
        L_chi2 = torch.tensor(0.0, device=self.device)
        
        if self.loss_weight_chi2 > 0:
            L_chi2 = self.compute_chi2_loss(x, sim_index, batch)
        
        tot_loss_batch = (self.loss_weight_attractive * L_att + 
                         self.loss_weight_repulsive * L_rep + 
                         self.loss_weight_beta * L_beta +
                         self.loss_weight_chi2 * L_chi2)

        self.log('val_loss', tot_loss_batch, on_step=False, on_epoch=True, prog_bar=True, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_attractive', L_att, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_repulsive', L_rep, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_beta', L_beta, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        
        if self.loss_weight_chi2 > 0:
            self.log('val_loss_chi2', L_chi2, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)

        del model_out, L_att, L_rep, L_beta, L_chi2
        
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
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.hparams.get('scheduler_milestones', [10]),
            gamma=self.scheduler_gamma
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