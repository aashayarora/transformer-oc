#!/usr/bin/env python3
import argparse
import os
import json
from train import Trainer
from helpers import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Train Graph Neural Network for Particle Tracking')
    parser.add_argument('--config', type=str,
                       help='Path to JSON config file (overrides command line args)')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("A configuration file must be provided with --config")
    
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