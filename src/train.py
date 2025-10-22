import os
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torch_geometric.loader import DataLoader

from dataset import PCDataset

class Trainer:
    def __init__(self, config):
        self.config = config
        
    def setup_data(self):
        """Setup training and validation data loaders"""
        train_dataset = PCDataset(self.config['train_data_dir'], 
                                     subset=self.config.get('train_subset', None))
        val_dataset = PCDataset(self.config['val_data_dir'], 
                                   subset=self.config.get('val_subset', None))

        # Get feature dimensions
        self.feature_dims = train_dataset.get_feature_dims()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=self.config['num_workers']
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=self.config['num_workers']
        )
        
        return train_loader, val_loader
    
    def train(self):
        self.train_loader, self.val_loader = self.setup_data()
        
        self.model = self.config.get('model', 'transformer')
        if self.model == 'transformer':
            from models.transformer import TransformerLightningModule
            self.model = TransformerLightningModule(
                self.config, 
                input_dim=self.feature_dims['node_features_dim']
            )
        elif self.model == 'hept':
            from models.hept import HEPTOCLightningModule
            self.model = HEPTOCLightningModule(
                self.config, 
                input_dim=self.feature_dims['node_features_dim']
            )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.config['output_dir'], 'checkpoints'),
            filename='model-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            every_n_epochs=self.config.get('save_every', 10),
            save_on_train_epoch_end=False
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('early_stopping_patience', 20),
            mode='min',
            verbose=True
        )
        
        logger = TensorBoardLogger(
            save_dir=self.config['output_dir']
        )
        
        trainer = pl.Trainer(
            max_epochs=self.config['epochs'],
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=[1] if torch.cuda.is_available() else None,
            enable_progress_bar=True,
            log_every_n_steps=10,
            enable_checkpointing=True        
        )
        
        ckpt_path = None
        if self.config.get('resume'):
            ckpt_path = self.config['resume']
            print(f"Resuming training from {ckpt_path}")
        
        trainer.fit(self.model, self.train_loader, self.val_loader, ckpt_path=ckpt_path)
        
        if trainer.is_global_zero:
            final_model_path = os.path.join(self.config['output_dir'], 'final_model.ckpt')
            trainer.save_checkpoint(final_model_path)
            print(f"Final model saved to {final_model_path}")
            