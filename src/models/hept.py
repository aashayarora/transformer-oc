import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.object_condensation import ObjectCondensation
from fastgraphcompute.torch_geometric_interface import row_splits_from_strict_batch as batch_to_rowsplits

import pytorch_lightning as pl

from src.utils.hept_utils import HEPTAttention, get_regions, prepare_input

class Attn(nn.Module):
    def __init__(self, coords_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]

        self.w_q = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_k = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_v = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)

        self.attn = HEPTAttention(self.dim_per_head + coords_dim, **kwargs)

        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(self.dim_per_head)
        self.norm2 = nn.LayerNorm(self.dim_per_head)
        self.ff = nn.Sequential(
            nn.Linear(self.dim_per_head, self.dim_per_head),
            nn.ReLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        self.w_rpe = nn.Linear(kwargs["num_w_per_dist"] * (coords_dim - 1), self.num_heads * self.dim_per_head)

    def forward(self, x, kwargs):
        x_normed = self.norm1(x)
        q, k, v = self.w_q(x_normed), self.w_k(x_normed), self.w_v(x_normed)
        aggr_out = self.attn(q, k, v, w_rpe=self.w_rpe, **kwargs)

        x = x + self.dropout(aggr_out)
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x
    
class HEPTOCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, latent_dim, 
                 dropout=0.1, coords_dim=2, block_size=32, n_hashes=4, 
                 num_regions=8, num_w_per_dist=8):
        super().__init__()
        
        self.n_layers = num_layers
        self.h_dim = hidden_dim
        self.latent_dim = latent_dim
        self.coords_dim = coords_dim

        self.feat_encoder = nn.Sequential(
            nn.Linear(input_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        attn_kwargs = {
            "h_dim": self.h_dim,
            "num_heads": num_heads,
            "block_size": block_size,
            "n_hashes": n_hashes,
            "num_regions": num_regions,
            "num_w_per_dist": num_w_per_dist,
        }
        
        self.attns = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attns.append(Attn(coords_dim, **attn_kwargs))

        self.dropout = nn.Dropout(dropout)
        
        self.W = nn.Linear(self.h_dim * (self.n_layers + 1), int(self.h_dim // 2), bias=False)

        self.mlp_out = nn.Sequential(
            nn.Linear(int(self.h_dim // 2), int(self.h_dim // 2)),
            nn.ReLU(),
            nn.Linear(int(self.h_dim // 2), int(self.h_dim // 2)),
        )

        self.beta_head = nn.Sequential(
            nn.Linear(int(self.h_dim // 2), int(self.h_dim // 4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.h_dim // 4), 1)
        )
        
        self.coord_head = nn.Sequential(
            nn.Linear(int(self.h_dim // 2), int(self.h_dim // 4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.h_dim // 4), self.latent_dim)
        )

        self.helper_params = {}
        self.helper_params["block_size"] = block_size
        self.regions = nn.Parameter(
            get_regions(num_regions, n_hashes, num_heads), requires_grad=False
        )
        self.helper_params["regions"] = self.regions
        self.helper_params["num_heads"] = num_heads

        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, coords, batch):
        x, kwargs, unpad_seq = prepare_input(x, coords, batch, self.helper_params)

        encoded_x = self.feat_encoder(x)
        all_encoded_x = [encoded_x]
                
        for i in range(self.n_layers):
            encoded_x = self.attns[i](encoded_x, kwargs)
            all_encoded_x.append(encoded_x)

        encoded_x = self.W(torch.cat(all_encoded_x, dim=-1))
        out = encoded_x + self.dropout(self.mlp_out(encoded_x))
        
        out = out[unpad_seq]

        eps = 1e-6
        beta = self.beta_head(out).sigmoid()
        beta = torch.clamp(beta, eps, 1 - eps)
        
        coords_latent = self.coord_head(out)

        return {"B": beta, "H": coords_latent}
        

class HEPTOCLightningModule(pl.LightningModule):
    def __init__(self, config, input_dim):
        super().__init__()
        self.save_hyperparameters(config)
        
        self.model = HEPTOCModel(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', config.get('d_model', 128)),
            num_layers=config['num_layers'],
            num_heads=config['nhead'],
            latent_dim=config['latent_dim'],
            dropout=config.get('dropout', 0.1),
            coords_dim=2,
            block_size=config.get('block_size', 32),
            n_hashes=config.get('n_hashes', 4),
            num_regions=config.get('num_regions', 8),
            num_w_per_dist=config.get('num_w_per_dist', 8)
        )

        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.scheduler_patience = config['scheduler_patience']
        
        # Object Condensation with memory optimization settings
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
        
        coords = data.x[:, 1:3].to(self.device)  # eta, phi

        model_out = self.model(x, coords, batch)

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
        
        return tot_loss_batch

    def validation_step(self, data):
        x, sim_index, batch = data.x.to(self.device), data.sim_index.to(self.device), data.batch.to(self.device)
        sim_index[sim_index < 0] = -1
        row_splits = batch_to_rowsplits(batch)
        
        coords = data.x[:, 1:3].to(self.device)

        model_out = self.model(x, coords, batch)

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

        return tot_loss_batch
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=self.scheduler_patience),
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}