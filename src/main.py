#!/usr/bin/env python3

import argparse
import os
import json
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from dataset import PCDataset
from helpers import set_seed, load_config

class Trainer:
    def __init__(self, config):
        self.config = config

    def setup_data(self):
        self.train_dataset = PCDataset(
            self.config['train_data_dir'],
            subset=self.config.get('train_subset', None),
        )
        val_dataset = PCDataset(
            self.config['val_data_dir'],
            subset=self.config.get('val_subset', None),
        )
        self.feature_dims = self.train_dataset.get_feature_dims()
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        return train_loader, val_loader

    def train(self):
        self.train_loader, self.val_loader = self.setup_data()
        self.model = self.config.get('model', 'transformer')
        if self.model == 'transformer':
            from model import TransformerLightningModule
            self.model = TransformerLightningModule(
                self.config,
                input_dim=self.feature_dims['node_features_dim']
            )
        logger = TensorBoardLogger(
            save_dir=self.config['output_dir']
        )
        debug_mode = self.config.get('debug', False)
        checkpoint_callback = ModelCheckpoint(
            dirpath=logger.log_dir + '/checkpoints',
            filename='ckpt-{epoch:03d}',
            monitor='train_loss' if debug_mode else 'val_loss',
            mode='min',
            save_top_k=-1 if debug_mode else 3,
            save_last=True,
            every_n_epochs=1 if debug_mode else self.config.get('save_every', 5),
            save_on_train_epoch_end=True
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('early_stopping_patience', 20),
            mode='min',
            verbose=True
        )

        num_gpus = len(self.config.get('gpus', [1])) if isinstance(self.config.get('gpus'), list) else self.config.get('gpus', 1)
        accelerator = 'gpu' if torch.cuda.is_available() and num_gpus > 0 else 'cpu'

        trainer = pl.Trainer(
            max_epochs=self.config['epochs'],
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger,
            accelerator=accelerator,
            devices=(self.config.get('gpus', [1]) if isinstance(self.config.get('gpus'), list) else self.config.get('gpus', 1)),
            strategy="auto",
            enable_progress_bar=True,
            check_val_every_n_epoch=self.config.get('check_val_every_n_epoch', 5),
            log_every_n_steps=5,
            enable_checkpointing=True,
            gradient_clip_val=self.config.get('gradient_clip_val', 1.0),
            gradient_clip_algorithm=self.config.get('gradient_clip_algorithm', 'norm'),
            accumulate_grad_batches=self.config.get('accumulate_grad_batches', 1),
            precision=self.config.get('precision', 32),
        )

        ckpt_path = None
        if self.config.get('resume'):
            ckpt_path = self.config['resume']
            print(f"Resuming training from {ckpt_path}")
        trainer.fit(self.model, self.train_loader, self.val_loader, ckpt_path=ckpt_path)
        
        if trainer.is_global_zero:
            final_model_path = os.path.join(logger.log_dir, 'final_model.ckpt')
            trainer.save_checkpoint(final_model_path)
            print(f"Final model saved to {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Graph Neural Network for Particle Tracking')
    parser.add_argument('--config', type=str,
                       help='Path to YAML config file')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError("A configuration file must be provided with --config")
    set_seed(config['seed'])
    os.makedirs(config['output_dir'], exist_ok=True)
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()