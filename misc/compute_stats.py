#!/usr/bin/env python3
"""
Script to precompute dataset statistics (mean and std) and save them to a file.
This avoids the slow on-the-fly computation during training.

Usage:
    python compute_stats.py --input ./data/muonG/train --output stats_muonG_train.pt
    python compute_stats.py --input ./data/muonG/val --output stats_muonG_val.pt
    python compute_stats.py --input ./data/muonG/train --output stats.pt --num_workers 8
"""

import os
import sys
import argparse
import torch
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def process_single_graph(graph_path):
    """
    Process a single graph file and return local statistics.
    
    Args:
        graph_path: Path to the graph file
        
    Returns:
        Tuple of (count, sum, sum_of_squares, num_features)
    """
    try:
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)
        
        # Apply same preprocessing as dataset
        # Apply log transform to pT (feature 0)
        graph.x[:, 0] = torch.log(graph.x[:, 0] + 1e-6)
        
        # Compute local statistics
        x = graph.x.double()
        count = x.shape[0]
        local_sum = torch.sum(x, dim=0)
        local_sum_sq = torch.sum(x ** 2, dim=0)
        num_features = x.shape[1]
        
        return count, local_sum, local_sum_sq, num_features
    except Exception as e:
        print(f"Error processing {graph_path}: {e}")
        return None


def compute_statistics(input, subset=None, num_workers=None):
    """
    Compute mean and std statistics for dataset using parallel processing.
    
    Args:
        input: Directory containing graph_*.pt files
        subset: Optional limit on number of files to process
        num_workers: Number of parallel workers (default: cpu_count)
        
    Returns:
        Dictionary with 'mean' and 'std' tensors
    """
    print(f"Loading graph files from {input}...")
    graph_files = glob.glob(os.path.join(input, "graph_*.pt"))
    graph_files.sort()
    
    if len(graph_files) == 0:
        raise ValueError(f"No graph files found in {input}")
    
    if subset is not None:
        graph_files = graph_files[:subset]
    
    print(f"Found {len(graph_files)} graph files")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(graph_files))
    
    print(f"Using {num_workers} workers for parallel processing...")
    
    # Process graphs in parallel
    total_count = 0
    total_sum = None
    total_sum_sq = None
    num_features = None
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_graph, graph_files),
            total=len(graph_files),
            desc="Processing graphs"
        ))
    
    # Aggregate results
    print("Aggregating statistics...")
    for result in results:
        if result is None:
            continue
        
        count, local_sum, local_sum_sq, nf = result
        
        if num_features is None:
            num_features = nf
            total_sum = torch.zeros(num_features, dtype=torch.float64)
            total_sum_sq = torch.zeros(num_features, dtype=torch.float64)
        
        total_count += count
        total_sum += local_sum
        total_sum_sq += local_sum_sq
    
    # Compute mean and std
    mean = total_sum / total_count
    # Var = E[X^2] - E[X]^2
    variance = (total_sum_sq / total_count) - (mean ** 2)
    std = torch.sqrt(variance)
    
    # Prevent division by zero
    std[std < 1e-6] = 1.0
    
    # Convert back to float32 for storage
    mean = mean.float()
    std = std.float()
    
    print(f"Processed {total_count} total nodes")
    
    return {
        'mean': mean,
        'std': std,
        'num_nodes': total_count,
        'num_graphs': len(graph_files)
    }


def main():
    parser = argparse.ArgumentParser(description='Precompute dataset statistics with multiprocessing')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing graph_*.pt files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path for statistics (e.g., stats.pt)')
    parser.add_argument('--subset', type=int, default=None,
                        help='Optional: limit number of graphs to process')
    parser.add_argument('--num_workers', '-n', type=int, default=None,
                        help=f'Number of parallel workers (default: {cpu_count()})')
    
    args = parser.parse_args()
    
    # Compute statistics
    stats = compute_statistics(args.input, args.subset, args.num_workers)
    
    # Save to file
    print(f"\nSaving statistics to {args.output}...")
    torch.save(stats, args.output)
    
    print("\nStatistics computed:")
    print(f"  Number of graphs: {stats['num_graphs']}")
    print(f"  Number of nodes: {stats['num_nodes']}")
    print(f"  Feature dimensions: {len(stats['mean'])}")
    print(f"\nMean (first 5 features): {stats['mean'][:5].tolist()}")
    print(f"Std (first 5 features): {stats['std'][:5].tolist()}")
    print("\nDone!")


if __name__ == '__main__':
    main()
