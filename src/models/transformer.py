import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from utils.object_condensation import ObjectCondensation
# from fastgraphcompute.object_condensation import ObjectCondensation
from fastgraphcompute.torch_geometric_interface import row_splits_from_strict_batch as batch_to_rowsplits

import pytorch_lightning as pl

class TransformerOCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, latent_dim, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.latent_head = nn.Linear(self.hidden_dim, self.latent_dim)

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
                nn.init.xavier_uniform_(p)

    def forward(self, x, coords, batch):
        x = self.input_proj(x)
        
        batch_size = batch.max().item() + 1
        device = x.device
        
        # Process graphs with varying sequence lengths more memory-efficiently
        # Group by batch and process, but avoid creating full dense tensor
        outputs = []
        
        for i in range(batch_size):
            mask = batch == i
            x_graph = x[mask]  # Shape: (num_nodes_i, hidden_dim)
            
            # Add sequence dimension and process through transformer
            # Shape: (seq_len, 1, hidden_dim)
            x_graph = x_graph.unsqueeze(1)
            
            # Process this single graph through transformer
            # No padding mask needed for single graph
            x_transformed = self.transformer_encoder(x_graph)
            
            # Remove batch dimension
            x_transformed = x_transformed.squeeze(1)
            
            outputs.append(x_transformed)
        
        # Concatenate all outputs back together
        x_out = torch.cat(outputs, dim=0)

        coords_latent = self.latent_head(x_out)
        
        # Normalize latent coordinates to prevent them from growing unbounded
        coords_latent = torch.nn.functional.normalize(coords_latent, p=2, dim=-1) * (self.latent_dim ** 0.5)

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
        
        self.criterion = ObjectCondensation(
            q_min=config.get('oc_q_min', 0.1),
            s_B=config.get('oc_s_B', 1.0),
            repulsive_chunk_size=config.get('oc_repulsive_chunk_size', 32),
            repulsive_distance_cutoff=config.get('oc_repulsive_distance_cutoff', None),
            use_checkpointing=config.get('oc_use_checkpointing', False)
        )
    
    def training_step(self, data):
        x, sim_index, batch = data.x.to(self.device), data.sim_index.to(self.device), data.batch.to(self.device)
        sim_index[sim_index < 0] = -1
        row_splits = batch_to_rowsplits(batch)
        
        model_out = self.model(x, None, batch)

        L_att, L_rep, L_beta, _, _ = self.criterion(
            beta = model_out["B"],
            coords = model_out["H"],
            asso_idx = sim_index.unsqueeze(-1).to(torch.int64),
            row_splits = row_splits
        )
    
        tot_loss_batch = L_att + L_rep + L_beta

        self.log('train_loss', tot_loss_batch, on_step=True, on_epoch=True, prog_bar=True, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_attractive', L_att, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_repulsive', L_rep, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('train_loss_noise', L_beta, on_step=True, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        
        # Clean up intermediate tensors
        del model_out, L_att, L_rep, L_beta
        
        return tot_loss_batch

    def validation_step(self, data):
        x, sim_index, batch = data.x.to(self.device), data.sim_index.to(self.device), data.batch.to(self.device)
        sim_index[sim_index < 0] = -1
        row_splits = batch_to_rowsplits(batch)

        model_out = self.model(x, None, batch)

        L_att, L_rep, L_beta, _, _ = self.criterion(
            beta = model_out["B"],
            coords = model_out["H"],
            asso_idx = sim_index.unsqueeze(-1).to(torch.int64),
            row_splits = row_splits
        )
    
        tot_loss_batch = L_att + L_rep + L_beta

        self.log('val_loss', tot_loss_batch, on_step=False, on_epoch=True, prog_bar=True, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_attractive', L_att, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_repulsive', L_rep, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)
        self.log('val_loss_noise', L_beta, on_step=False, on_epoch=True, prog_bar=False, batch_size=data.num_graphs, sync_dist=True)

        # Clean up intermediate tensors
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
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=[10, 15], gamma=self.scheduler_gamma),
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}