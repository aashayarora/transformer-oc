#!/usr/bin/env python3
"""
Example script demonstrating GPU profiling for inference.

This script shows how to use the InferenceProfiler to analyze:
- GPU memory usage during model inference
- Timing for different stages (forward pass, clustering, data transfer)
- Peak memory consumption
"""

import sys
sys.path.append('/home/users/aaarora/phys/tracking/transformers/oc')
sys.path.append('/home/users/aaarora/phys/tracking/transformers/oc/src')

import torch
import argparse
import time
import numpy as np
from torch_geometric.loader import DataLoader

from src.model import TransformerOCModel
from src.dataset import PCDataset
from src.helpers import load_config
from analysis.profiler import InferenceProfiler, print_gpu_info, profile_memory
from analysis.measures import run_inference_and_clustering


def load_model(checkpoint_path, config_path, device='cuda:0'):
    """Load trained model from checkpoint."""
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint to get input dimension
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present
    if any(key.startswith('model.') for key in state_dict.keys()):
        state_dict = {key.replace('model.', '', 1): value for key, value in state_dict.items()}
    
    # Get input dimension from checkpoint
    if 'input_proj.weight' in state_dict:
        input_dim = state_dict['input_proj.weight'].shape[1]
        print(f"Detected input dimension from checkpoint: {input_dim}")
    else:
        input_dim = config.get('input_dim', 26)
        print(f"Using input dimension from config: {input_dim}")
    
    # Create model with config
    model = TransformerOCModel(
        input_dim=input_dim,
        hidden_dim=config.get('hidden_dim', config.get('d_model', 128)),
        num_layers=config['num_layers'],
        num_heads=config['nhead'],
        latent_dim=config['latent_dim'],
        dropout=config.get('dropout', 0.1),
        dr_threshold=config.get('dr_threshold', 0.2),
        attention_chunk_size=config.get('attention_chunk_size', 1024)
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def profile_inference(model, data_dir, device='cuda:0', num_samples=10, 
                     use_fp16=False, eps=0.2, min_samples=2, beta_threshold=None,
                     use_tensorboard=False, log_dir='./profiler_logs'):
    """
    Profile inference on a dataset.
    
    Args:
        model: Trained model
        data_dir: Directory containing data
        device: Device to run on
        num_samples: Number of samples to profile
        use_fp16: Whether to use FP16
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        beta_threshold: Optional beta threshold
        use_tensorboard: Whether to export TensorBoard traces
        log_dir: Directory for profiler logs
    """
    print("\n" + "="*90)
    print(f"PROFILING INFERENCE ON {num_samples} SAMPLES")
    print("="*90)
    
    # Print GPU info
    print_gpu_info()
    
    # Create dataset
    print(f"\nLoading dataset from {data_dir}")
    with profile_memory("Dataset loading"):
        dataset = PCDataset(data_dir, subset=num_samples)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Create profiler
    profiler = InferenceProfiler(
        device=device, 
        enabled=True,
        use_tensorboard=use_tensorboard,
        log_dir=log_dir
    )
    
    # Start PyTorch profiler
    profiler.start_profiling()
    
    # Run inference on all samples
    print(f"\nRunning inference on {num_samples} samples...")
    
    simple_timings = []
    for i, data in enumerate(loader):
        print(f"  Processing sample {i+1}/{num_samples}...", end='\r')
        
        # Also measure simple end-to-end time for comparison
        start = time.perf_counter()
        
        # Run inference with profiling
        cluster, beta, _ = run_inference_and_clustering(
            data=data,
            model=model,
            device=device,
            eps=eps,
            min_samples=min_samples,
            use_fp16=use_fp16,
            beta_threshold=beta_threshold,
            profiler=profiler
        )
        
        simple_timings.append(time.perf_counter() - start)
    
    print(f"\n✓ Completed {num_samples} samples")
    
    # Stop PyTorch profiler
    profiler.stop_profiling()
    
    # Print summary
    profiler.print_summary()
    
    # Export chrome trace if requested
    if use_tensorboard:
        profiler.export_chrome_trace()
        print(f"\nTensorBoard logs saved to: {log_dir}")
        print(f"To view in TensorBoard, run:")
        print(f"  tensorboard --logdir={log_dir}")
    
    # Also print simple timing comparison
    print("\n" + "="*90)
    print("SIMPLE END-TO-END TIMING (for comparison with main.py)")
    print("="*90)
    print(f"Average time per sample: {np.mean(simple_timings):.4f}s ± {np.std(simple_timings):.4f}s")
    print(f"Median time per sample:  {np.median(simple_timings):.4f}s")
    print(f"Min/Max time per sample: {np.min(simple_timings):.4f}s / {np.max(simple_timings):.4f}s")
    print(f"Total time:              {np.sum(simple_timings):.4f}s")
    print("="*90 + "\n")
    
    return profiler


def profile_single_sample(model, data, device='cuda:0', use_fp16=False, 
                         eps=0.2, min_samples=2, beta_threshold=None, verbose=True,
                         use_tensorboard=False, log_dir='./profiler_logs'):
    """
    Profile a single inference sample in detail.
    
    This is useful for detailed analysis of a specific sample.
    """
    print("\n" + "="*90)
    print("DETAILED SINGLE SAMPLE PROFILING")
    print("="*90)
    
    profiler = InferenceProfiler(
        device=device, 
        enabled=True,
        use_tensorboard=use_tensorboard,
        log_dir=log_dir
    )
    
    # Show initial memory state
    print(f"\nInitial Memory State:")
    if torch.cuda.is_available():
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.3f} GB")
    
    # Start profiling
    profiler.start_profiling()
    
    # Run inference
    print("\nRunning inference...")
    cluster, beta, _ = run_inference_and_clustering(
        data=data,
        model=model,
        device=device,
        eps=eps,
        min_samples=min_samples,
        use_fp16=use_fp16,
        beta_threshold=beta_threshold,
        profiler=profiler
    )
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Show final memory state
    print(f"\nFinal Memory State:")
    if torch.cuda.is_available():
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.3f} GB")
        print(f"  Peak:      {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
    
    # Print detailed summary
    if verbose:
        profiler.print_summary()
    
    # Export chrome trace if requested
    if use_tensorboard:
        profiler.export_chrome_trace()
        print(f"\nTensorBoard logs saved to: {log_dir}")
        print(f"To view in TensorBoard, run:")
        print(f"  tensorboard --logdir={log_dir}")
    
    return profiler, cluster, beta


def compare_fp16_vs_fp32(model, data, device='cuda:0', eps=0.2, min_samples=2):
    """
    Compare FP16 vs FP32 inference performance.
    """
    print("\n" + "="*90)
    print("COMPARING FP16 vs FP32 INFERENCE")
    print("="*90)
    
    # FP32 inference
    print("\n--- Running FP32 Inference ---")
    profiler_fp32 = InferenceProfiler(device=device)
    cluster_fp32, beta_fp32, _ = run_inference_and_clustering(
        data=data,
        model=model,
        device=device,
        eps=eps,
        min_samples=min_samples,
        use_fp16=False,
        profiler=profiler_fp32
    )
    
    # FP16 inference
    print("\n--- Running FP16 Inference ---")
    torch.cuda.empty_cache()
    profiler_fp16 = InferenceProfiler(device=device)
    cluster_fp16, beta_fp16, _ = run_inference_and_clustering(
        data=data,
        model=model,
        device=device,
        eps=eps,
        min_samples=min_samples,
        use_fp16=True,
        profiler=profiler_fp16
    )
    
    # Compare results
    print("\n" + "="*90)
    print("COMPARISON RESULTS")
    print("="*90)
    
    fp32_summary = profiler_fp32.get_timing_summary()
    fp16_summary = profiler_fp16.get_timing_summary()
    
    print(f"\n{'Metric':<30} {'FP32':>15} {'FP16':>15} {'Speedup':>12}")
    print("-" * 90)
    
    for section in ['model_forward', 'gpu_to_cpu_transfer', 'clustering']:
        if section in fp32_summary and section in fp16_summary:
            fp32_time = fp32_summary[section]['gpu_time_mean']
            fp16_time = fp16_summary[section]['gpu_time_mean']
            speedup = fp32_time / fp16_time if fp16_time > 0 else 0
            
            print(f"{section:<30} {fp32_time:>14.4f}s {fp16_time:>14.4f}s {speedup:>11.2f}x")
    
    fp32_mem = profiler_fp32.get_memory_stats()
    fp16_mem = profiler_fp16.get_memory_stats()
    mem_reduction = (fp32_mem['peak_gb'] - fp16_mem['peak_gb']) / fp32_mem['peak_gb'] * 100 if fp32_mem['peak_gb'] > 0 else 0
    
    print(f"\n{'Peak Memory Usage':<30} {fp32_mem['peak_gb']:>14.3f}GB {fp16_mem['peak_gb']:>14.3f}GB {mem_reduction:>10.1f}% reduction")
    print("="*90 + "\n")
    
    return profiler_fp32, profiler_fp16


def main():
    parser = argparse.ArgumentParser(description='Profile inference for particle tracking model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on (default: cuda:0)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to profile (default: 10)')
    parser.add_argument('--use-fp16', action='store_true', default=True,
                       help='Use FP16 for inference (default: True)')
    parser.add_argument('--compare-precision', action='store_true',
                       help='Compare FP16 vs FP32 performance')
    parser.add_argument('--eps', type=float, default=0.2,
                       help='DBSCAN eps parameter (default: 0.2)')
    parser.add_argument('--min-samples', type=int, default=2,
                       help='DBSCAN min_samples parameter (default: 2)')
    parser.add_argument('--beta-threshold', type=float, default=None,
                       help='Beta threshold for filtering (optional)')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Export TensorBoard traces')
    parser.add_argument('--log-dir', type=str, default='./profiler_logs',
                       help='Directory for profiler logs (default: ./profiler_logs)')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, args.config, device=args.device)
    
    # Load a sample for single-sample profiling
    dataset = PCDataset(args.data_dir, subset=1)
    sample_data = dataset[0]
    
    if args.compare_precision:
        # Compare FP16 vs FP32
        compare_fp16_vs_fp32(
            model=model,
            data=sample_data,
            device=args.device,
            eps=args.eps,
            min_samples=args.min_samples
        )
    else:
        # Profile multiple samples
        profile_inference(
            model=model,
            data_dir=args.data_dir,
            device=args.device,
            num_samples=args.num_samples,
            use_fp16=args.use_fp16,
            eps=args.eps,
            min_samples=args.min_samples,
            beta_threshold=args.beta_threshold,
            use_tensorboard=args.tensorboard,
            log_dir=args.log_dir
        )


if __name__ == '__main__':
    main()
