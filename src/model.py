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
                batch_first=True
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
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')
        
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
            x_transformed = layer(x_transformed, src_key_padding_mask=padding_mask) + x_padded
        
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
        self.loss_weight_repulsive = config.get('loss_weight_repulsive', 0.1)
        self.loss_weight_beta = config.get('loss_weight_beta', 1.0)
        self.loss_weight_aux = config.get('loss_weight_aux', 1.0)  # Auxiliary noise/signal supervision
        
        self.criterion = ObjectCondensation(
            q_min=config.get('oc_q_min', 0.1),
            s_B=config.get('oc_s_B', 1.0),
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
        
        is_noise = (sim_index < 0).float().unsqueeze(-1)  # 1 for noise, 0 for signal
        L_aux = (is_noise * beta.pow(2)).mean()  # noise beta → 0
    
        tot_loss_batch = (self.loss_weight_attractive * L_att + 
                         self.loss_weight_repulsive * L_rep + 
                         self.loss_weight_beta * L_beta +
                         self.loss_weight_aux * L_aux)

        self.log('train_loss', tot_loss_batch, on_step=True, on_epoch=True, prog_bar=True, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_attractive', L_att, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_repulsive', L_rep, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_beta', L_beta, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_aux', L_aux, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        
        del model_out, L_att, L_rep, L_beta, L_aux
        
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
        
        is_noise = (sim_index < 0).float().unsqueeze(-1)  # 1 for noise, 0 for signal
        L_aux = (is_noise * beta.pow(2)).mean()  # noise beta → 0
    
        tot_loss_batch = (self.loss_weight_attractive * L_att + 
                         self.loss_weight_repulsive * L_rep + 
                         self.loss_weight_beta * L_beta +
                         self.loss_weight_aux * L_aux)

        self.log('val_loss', tot_loss_batch, on_step=False, on_epoch=True, prog_bar=True, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_attractive', L_att, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_repulsive', L_rep, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_beta', L_beta, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_aux', L_aux, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)

        del model_out, L_att, L_rep, L_beta, L_aux
        
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