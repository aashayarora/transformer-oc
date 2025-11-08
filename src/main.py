#!/usr/bin/env python3
import argparse
import os
import json
from train import Trainer
from helpers import set_seed

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def parse_args():
    parser = argparse.ArgumentParser(description='Train Graph Neural Network for Particle Tracking')
    
    parser.add_argument('--train_data_dir', type=str,
                       help='Directory containing training graph data')
    parser.add_argument('--val_data_dir', type=str,
                       help='Directory containing validation graph data')
    
    parser.add_argument('--model_type', type=str, default=None,
                       choices=['graph_transformer'],
                       help='Type of model to use')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of layers')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads (for transformer)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (currently only supports 1)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    
    parser.add_argument('--scheduler_patience', type=int, default=5,
                       help='Scheduler patience for ReduceLROnPlateau')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value (clips gradients by norm)')
    parser.add_argument('--gradient_clip_algorithm', type=str, default='norm',
                       choices=['norm', 'value'],
                       help='Gradient clipping algorithm: "norm" clips by norm, "value" clips by value')
    
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory to save output')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--config', type=str,
                       help='Path to JSON config file (overrides command line args)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = vars(args)
    
    set_seed(config['seed'])
    
    os.makedirs(config['output_dir'], exist_ok=True)

    if os.path.exists(config['output_dir']):
        existing_runs = [d for d in os.listdir(config['output_dir']) if os.path.isdir(os.path.join(config['output_dir'], d)) and d.startswith('run_')]
        if existing_runs:
            existing_indices = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
            next_index = max(existing_indices) + 1 if existing_indices else 0
        else:
            next_index = 0
        config['output_dir'] = os.path.join(config['output_dir'], f'run_{next_index:02d}')
    
    print("Training Configuration:")
    print("-" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 50)
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()