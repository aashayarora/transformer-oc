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
        
        graph.x[:, 0] = torch.log(graph.x[:, 0] + 1e-6)
        
        if graph.x.shape[1] == 26:
            
            min_vals = torch.tensor([-1.02, -2.62, -3.15, -0.29, -0.92, -1.49, -1.60, -0.80,
                                    -99.0, -99.0, -224.0, -99.0, -99.0, -225.0, -0.018, -0.82,
                                    -5.03, -111.0, -111.0, -268.0, -111.0, -111.0, -268.0, -0.008,
                                    -0.92, -5.03])
            max_vals = torch.tensor([7, 2.62, 3.15, 0.30, 0.92, 1.58, 1.64, 0.80,
                                    99.0, 98.0, 224.0, 99.0, 98.0, 225.0, 0.015, 0.84,
                                    5.03, 111.0, 111.0, 268.0, 111.0, 111.0, 268.0, 0.009,
                                    0.92, 5.03])
            
        else:
            min_vals = torch.tensor([-3.4627, -2.4507, -3.1416, -0.2362, -0.7272, -0.8202, -0.8768, -0.5988,
                                    -0.2704, -0.8894, -1.0590, -1.3935, -0.7862, -73.8012, -72.4212, -187.5750,
                                    -73.8019, -77.1076, -187.8705, -0.0231, -0.5393, -5.0250, -93.8255, -93.9099,
                                    -223.8540, -97.9344, -93.9060, -224.1495, -0.0111, -0.8231, -5.0255, -93.8255,
                                    -93.9099, -223.8540, -97.9344, -93.9060, -224.1495, -0.0111, -0.8231, -5.0255,
                                    -110.0160, -110.0150, -267.2350, -110.1943, -110.1950, -267.6350, -0.0046,
                                    -0.9416, -5.0259])

            max_vals = torch.tensor([7.5964, 2.4535, 3.1416, 0.2434, 0.7304, 0.7744, 0.8728, 0.5682,
                                    0.2755, 0.8922, 1.1500, 1.5453, 0.8093, 73.7524, 70.4150, 187.5750,
                                    73.7676, 71.4665, 187.8705, 0.0228, 0.5575, 5.0250, 93.8869, 93.8406,
                                    223.8540, 96.1440, 96.9953, 224.1495, 0.0097, 0.8098, 5.0253, 93.8869,
                                    93.8406, 223.8540, 96.1440, 96.9953, 224.1495, 0.0097, 0.8098, 5.0253,
                                    110.0128, 110.0150, 267.2350, 110.1814, 110.1950, 267.6350, 0.0050,
                                    0.9323, 5.0260])


        graph.x = (graph.x - min_vals) / (max_vals - min_vals + 1e-8)
        
        return graph
    
    def get_feature_dims(self):
        sample = self[0]
        return {
            'node_features_dim': sample.num_node_features,
            'sim_features_dim': sample.sim_features.shape[1] if hasattr(sample, 'sim_features') else None
        }