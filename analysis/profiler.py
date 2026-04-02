"""
Inference Profiler for GPU Memory and Timing Analysis

This module provides profiling tools specifically for inference/evaluation using PyTorch's built-in profiler:
- GPU memory usage tracking during inference
- Timing analysis for model forward pass and clustering
- Memory snapshots and peak tracking
- TensorBoard integration
"""

import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import numpy as np
from contextlib import contextmanager
from typing import Dict, Optional
import json
import os


class InferenceProfiler:
    """Profile GPU memory and timing during inference using PyTorch Profiler."""
    
    def __init__(self, device: str = 'cuda:0', enabled: bool = True, use_tensorboard: bool = False, log_dir: str = './profiler_logs'):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.enabled = False
            self.device = None
            self.device_id = None
            print("WARNING: CUDA not available, profiling disabled")
        else:
            self.enabled = enabled
            # Store device as torch.device object
            self.device = torch.device(device)
            # Extract device index for CUDA functions that need it
            if self.device.type == 'cuda':
                self.device_id = self.device.index if self.device.index is not None else 0
            else:
                self.device_id = 0
        
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.profiler = None
        self.section_times = {}
        self.memory_snapshots = []
        
        if self.enabled:
            # Set the current device for CUDA operations
            torch.cuda.set_device(self.device)
            # Reset peak stats (works on current device)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # Create log directory if using TensorBoard
            if self.use_tensorboard:
                os.makedirs(log_dir, exist_ok=True)
    
    def start_profiling(self):
        """Start the PyTorch profiler."""
        if not self.enabled:
            return
        
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        self.profiler = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # Set to True for more detailed stack traces (slower)
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir) if self.use_tensorboard else None,
        )
        self.profiler.__enter__()
    
    def stop_profiling(self):
        """Stop the PyTorch profiler."""
        if not self.enabled or self.profiler is None:
            return
        
        self.profiler.__exit__(None, None, None)
    
    @contextmanager
    def profile_section(self, section_name: str, record_memory: bool = True):
        """Context manager to profile a code section."""
        if not self.enabled:
            yield
            return
        
        # Record memory before
        if record_memory:
            mem_before = torch.cuda.memory_allocated() / 1024**3
        
        # Use PyTorch's record_function for better profiling
        with record_function(section_name):
            start_time = time.perf_counter()
            yield
            elapsed = time.perf_counter() - start_time
        
        # Record memory after
        if record_memory:
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_delta = mem_after - mem_before
        else:
            mem_delta = 0
        
        # Store timing info
        if section_name not in self.section_times:
            self.section_times[section_name] = []
        self.section_times[section_name].append({
            'time': elapsed,
            'memory_delta': mem_delta
        })
    
    def record_event(self, event_name: str, include_memory: bool = True):
        """Record a profiling event with optional memory snapshot."""
        if not self.enabled:
            return
        
        event = {
            'name': event_name,
            'timestamp': time.time()
        }
        
        if include_memory:
            event['memory_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
            event['memory_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
            event['memory_peak_gb'] = torch.cuda.max_memory_allocated() / 1024**3
        
        self.memory_snapshots.append(event)
        return event
    
    def get_summary(self) -> dict:
        """Get a summary of profiling results."""
        if not self.enabled:
            return {}
        
        summary = {
            'section_stats': {},
            'memory_snapshots': self.memory_snapshots,
            'profiler_table': None
        }
        
        # Calculate statistics for each section
        for section_name, measurements in self.section_times.items():
            times = [m['time'] for m in measurements]
            mem_deltas = [m['memory_delta'] for m in measurements]
            
            summary['section_stats'][section_name] = {
                'count': len(times),
                'total_time': sum(times),
                'mean_time': np.mean(times) if times else 0,
                'std_time': np.std(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'mean_memory_delta_gb': np.mean(mem_deltas) if mem_deltas else 0
            }
        
        # Get PyTorch profiler summary if available
        if self.profiler is not None:
            summary['profiler_table'] = self.profiler.key_averages().table(
                sort_by="cuda_time_total", row_limit=20
            )
        
        # Peak memory usage
        if torch.cuda.is_available():
            summary['peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1024**3
            summary['current_memory_gb'] = torch.cuda.memory_allocated() / 1024**3
            summary['reserved_memory_gb'] = torch.cuda.memory_reserved() / 1024**3
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of profiling results."""
        if not self.enabled:
            print("Profiling is disabled")
            return
        
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        
        # Print section timing stats
        print("\nSection Timing Statistics:")
        print("-" * 80)
        
        section_stats = summary['section_stats']
        if section_stats:
            total_time = sum(s['total_time'] for s in section_stats.values())
            
            # Sort by total time
            sorted_sections = sorted(
                section_stats.items(),
                key=lambda x: x[1]['total_time'],
                reverse=True
            )
            
            print(f"{'Section':<25} {'Count':>8} {'Total(s)':>10} {'Mean(s)':>10} {'Std(s)':>10} {'%':>8}")
            print("-" * 80)
            
            for section_name, stats in sorted_sections:
                pct = (stats['total_time'] / total_time * 100) if total_time > 0 else 0
                print(f"{section_name:<25} {stats['count']:>8} {stats['total_time']:>10.4f} "
                      f"{stats['mean_time']:>10.4f} {stats['std_time']:>10.4f} {pct:>7.2f}%")
            
            print("-" * 80)
            print(f"{'TOTAL':<25} {'':<8} {total_time:>10.4f}")
        
        # Print memory stats
        print("\nMemory Statistics:")
        print("-" * 80)
        print(f"Peak Memory:     {summary.get('peak_memory_gb', 0):.3f} GB")
        print(f"Current Memory:  {summary.get('current_memory_gb', 0):.3f} GB")
        print(f"Reserved Memory: {summary.get('reserved_memory_gb', 0):.3f} GB")
        
        # Print PyTorch profiler table if available
        if summary['profiler_table']:
            print("\nPyTorch Profiler Details (Top 20 operations by CUDA time):")
            print("-" * 80)
            print(summary['profiler_table'])
        
        print("="*80 + "\n")
    
    def export_chrome_trace(self, filename: str = "trace.json"):
        """Export profiler trace for viewing in chrome://tracing."""
        if not self.enabled or self.profiler is None:
            print("Profiler not available for export")
            return
        
        filepath = os.path.join(self.log_dir, filename)
        self.profiler.export_chrome_trace(filepath)
        print(f"Chrome trace exported to: {filepath}")
        print(f"View at: chrome://tracing")
    
    def reset(self):
        """Reset all profiling statistics."""
        self.section_times = {}
        self.memory_snapshots = []
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()


@contextmanager
def profile_memory(label: str = "operation"):
    """
    Simple context manager to profile memory usage.
    
    Example:
        with profile_memory("data loading"):
            data = load_data()
    """
    if not torch.cuda.is_available():
        yield
        return
    
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1024**3
    peak_before = torch.cuda.max_memory_allocated() / 1024**3
    
    start_time = time.perf_counter()
    
    yield
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    mem_after = torch.cuda.memory_allocated() / 1024**3
    peak_after = torch.cuda.max_memory_allocated() / 1024**3
    
    mem_delta = mem_after - mem_before
    peak_delta = peak_after - peak_before
    elapsed = end_time - start_time
    
    print(f"\n[{label}]")
    print(f"  Time:        {elapsed:.4f} s")
    print(f"  Memory Δ:    {mem_delta:+.3f} GB (Before: {mem_before:.3f} GB, After: {mem_after:.3f} GB)")
    print(f"  Peak Δ:      {peak_delta:+.3f} GB (Peak: {peak_after:.3f} GB)")


def print_gpu_info():
    """Print GPU device information."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("\n" + "="*80)
    print("GPU INFORMATION")
    print("="*80)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi-Processors: {props.multi_processor_count}")
        
        # Current usage
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free = (props.total_memory / 1024**3) - allocated
        
        print(f"  Current Allocated: {allocated:.3f} GB")
        print(f"  Current Reserved: {reserved:.3f} GB")
        print(f"  Available: {free:.3f} GB")
    
    print("="*80 + "\n")
