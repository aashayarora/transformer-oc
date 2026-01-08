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

    def compute_dr_mask(self, x_raw, batch, dr_threshold=0.2):
        device = x_raw.device
        batch_size = batch.max().item() + 1
        
        eta_norm = x_raw[:, 1]  # Normalized eta
        phi_norm = x_raw[:, 2]  # Normalized phi
        
        eta_min, eta_max = -2.62, 2.62
        phi_min, phi_max = -3.1416, 3.1416
        
        eta = eta_norm * (eta_max - eta_min) + eta_min
        phi = phi_norm * (phi_max - phi_min) + phi_min
        
        deta = eta.unsqueeze(1) - eta.unsqueeze(0)  # [N, N]
        dphi = phi.unsqueeze(1) - phi.unsqueeze(0)  # [N, N]
        
        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        
        dr = torch.sqrt(deta**2 + dphi**2)
        
        attention_mask = torch.zeros_like(dr)
        attention_mask[dr > dr_threshold] = float('-inf')
        
        del deta, dphi, dr, eta, phi
        
        return attention_mask

    def forward(self, x_raw, batch):
        x = self.input_proj(x_raw)
        
        batch_size = batch.max().item() + 1
        device = x.device
        
        x_input = x.unsqueeze(0)  # [1, N, hidden_dim]
        
        attention_mask = self.compute_dr_mask(x_raw, batch, dr_threshold=0.2)
        attention_mask = attention_mask.to(device)

        x_transformed = x_input
        for layer in self.transformer_layers:
            x_transformed = layer(x_transformed, src_mask=attention_mask)
        
        x_out = x_transformed.squeeze(0)

        coords_latent = self.latent_head(x_out)

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
            
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'scheduler_gamma': config['scheduler_gamma'],
            'scheduler_milestones': config.get('scheduler_milestones', [10]),
            
            'loss_weight_attractive': config.get('loss_weight_attractive', 1.0),
            'loss_weight_repulsive': config.get('loss_weight_repulsive', 1.0),
            'loss_weight_beta': config.get('loss_weight_beta', 1.0),
            
            'oc_q_min': config.get('oc_q_min', 0.1),
            'oc_s_B': config.get('oc_s_B', 1.0),
            
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
        )
        
        self.criterion = ObjectCondensation(
            q_min=self.hparams.oc_q_min,
            s_B=self.hparams.oc_s_B,
            # repulsive_chunk_size=config.get('oc_repulsive_chunk_size', 32),
            # repulsive_distance_cutoff=config.get('oc_repulsive_distance_cutoff', None),
            # use_checkpointing=config.get('oc_use_checkpointing', False)
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