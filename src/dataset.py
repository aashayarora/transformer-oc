import os
import torch
from torch_geometric.data import Dataset
import glob
import gc

class PCDataset(Dataset):
    def __init__(self, data_dir, subset=None):
        self.data_dir = data_dir
        self.subset = subset if subset is not None else float('inf')

        self.graph_files = glob.glob(os.path.join(data_dir, "graph_*.pt"))
        self.graph_files.sort()
        
        if len(self.graph_files) == 0:
            raise ValueError(f"No graph files found in {data_dir}")
        
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

        graph.x[:, 0] = torch.log(graph.x[:, 0] + 1e-6)
        graph.x[:, 1] = graph.x[:, 1] / 4.0
        graph.x[:, 2] = graph.x[:, 2] / 4.0
        graph.x[:, 8:14] = graph.x[:, 8:14] / 100.0
        graph.x[:, 17:23] = graph.x[:, 17:23] / 100.0
        return graph
    
    def get_feature_dims(self):
        sample = self[0]
        return {
            'node_features_dim': sample.num_node_features,
            'sim_features_dim': sample.sim_features.shape[1] if hasattr(sample, 'sim_features') else None
        }