import os
import torch
from torch_geometric.data import Dataset
import glob
import gc
import hashlib

class PCDataset(Dataset):
    def __init__(self, data_dir, subset=None, compute_stats=True, cache_dir=None):
        self.data_dir = data_dir
        self.subset = subset if subset is not None else float('inf')

        self.graph_files = glob.glob(os.path.join(data_dir, "graph_*.pt"))
        self.graph_files.sort()
        
        if len(self.graph_files) == 0:
            raise ValueError(f"No graph files found in {data_dir}")
        
        # Initialize statistics as None - will compute on-the-fly on first pass
        self.mean = None
        self.std = None
        self.compute_stats = compute_stats
        
        # Welford's algorithm accumulators for on-the-fly computation
        self._stats_count = 0
        self._stats_mean = None
        self._stats_M2 = None
        self._stats_finalized = False
        
        # Initialize statistics as None - will compute on-the-fly on first pass
        self.mean = None
        self.std = None
        self.compute_stats = compute_stats
        
        # Welford's algorithm accumulators for on-the-fly computation
        self._stats_count = 0
        self._stats_mean = None
        self._stats_M2 = None
        self._stats_finalized = False
        
    def _update_stats(self, x):
        """Update running statistics using Welford's online algorithm."""
        if not self.compute_stats or self._stats_finalized:
            return
            
        # Initialize on first call
        if self._stats_mean is None:
            num_features = x.shape[1]
            self._stats_mean = torch.zeros(num_features)
            self._stats_M2 = torch.zeros(num_features)
        
        # Update for each node in the graph
        for feature_val in x:
            self._stats_count += 1
            delta = feature_val - self._stats_mean
            self._stats_mean += delta / self._stats_count
            delta2 = feature_val - self._stats_mean
            self._stats_M2 += delta * delta2
    
    def finalize_stats(self):
        """Finalize statistics computation after first pass through data."""
        if not self.compute_stats or self._stats_finalized or self._stats_mean is None:
            return
        
        # Compute final statistics
        variance = self._stats_M2 / self._stats_count
        self.std = torch.sqrt(variance)
        
        # Prevent division by zero
        self.std[self.std < 1e-6] = 1.0
        
        self.mean = self._stats_mean
        self._stats_finalized = True
        
        # Clear accumulators to free memory
        self._stats_mean = None
        self._stats_M2 = None
    
    def __len__(self):
        return min(len(self.graph_files), self.subset)

    def __getitem__(self, idx):
        graph_path = self.graph_files[idx]
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)
        
        # 0: ls_pt, 1: ls_eta, 2: ls_phi, 3: ls_dPhis, 4: ls_dPhiChanges, 5: ls_dAlphaInners, 6: ls_dAlphaOuters, 7: ls_dAlphaInnerOuters
        # 8-13: md1 features (need normalization)
        # 14-16: md1 features (don't need normalization)
        # 17-22: md2 features (need normalization)
        # 23-24: md2 features (don't need normalization)

        # Apply log transform to pT
        graph.x[:, 0] = torch.log(graph.x[:, 0] + 1e-6)
        
        # Update statistics during first pass (before finalization)
        if self.compute_stats and not self._stats_finalized:
            self._update_stats(graph.x.clone())
        
        # Standardize all features using computed statistics (after finalization)
        if self.mean is not None and self.std is not None:
            graph.x = (graph.x - self.mean) / self.std
        
        return graph
    
    def get_feature_dims(self):
        sample = self[0]
        return {
            'node_features_dim': sample.num_node_features,
            'sim_features_dim': sample.sim_features.shape[1] if hasattr(sample, 'sim_features') else None
        }
