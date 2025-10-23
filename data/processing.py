import os
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser

import uproot
import awkward as ak

import torch
from torch_geometric.data import Data

SIM_VARS = ["sim_pt", "sim_eta", "sim_phi"]
LS_VARS = ["ls_pt", "ls_eta", "ls_phi", "ls_dPhis", "ls_dPhiChanges", "ls_dAlphaInners", "ls_dAlphaOuters", "ls_dAlphaInnerOuters"]
MD_VARS = ["md_anchor_x", "md_anchor_y", "md_anchor_z", "md_other_x", "md_other_y", "md_other_z", "md_dphi", "md_dphichange", "md_dz"]

MD_INDEX = ["ls_mdIdx0", "ls_mdIdx1"]
TARGET = ["ls_simIdx"]
FAKE_TARGET = ["ls_isFake"]

ALL_COLUMNS = LS_VARS + MD_VARS + MD_INDEX + TARGET + SIM_VARS + FAKE_TARGET

class GraphBuilder:
    def __init__(self, input_path, output_path, train_split=0.8):
        self.output_path = output_path
        self.train_split = train_split
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        self.train_path = os.path.join(self.output_path, "train")
        self.val_path = os.path.join(self.output_path, "val")
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)

        self.input_tree = uproot.open(input_path)["tree"]

    def process_event(self, event_data, idx, debug=False, overwrite=False):
        """Processes a single event and saves the graph."""
        try:
            output_file = os.path.join(self.output_path, f"graph_{idx}.pt")
            if debug:
                ...
            elif ((not overwrite) and os.path.exists(output_file)):
                print(f"Graph {idx} already exists, skipping...")
                return
            
            ls_features = ak.to_dataframe(event_data[LS_VARS]).values
            md_idx = ak.to_dataframe(event_data[MD_INDEX]).values
            md_features = ak.to_dataframe(event_data[MD_VARS]).values[md_idx]

            md_features = md_features.reshape(-1, 2 * len(MD_VARS))

            fake_mask = ak.to_dataframe(event_data[FAKE_TARGET]).values.flatten() == 0

            node_features = torch.Tensor(ak.concatenate([ls_features[fake_mask], md_features[fake_mask]], axis=1))
            target = torch.Tensor(ak.to_dataframe(event_data[TARGET]).values)[fake_mask]

            target_flat = target.flatten()
            # target_flat[target_flat < 0] = -999999
            
            sim_features = torch.Tensor(ak.to_dataframe(event_data[SIM_VARS]).values)

            graph = Data(x=node_features, sim_index=target_flat, sim_features=sim_features)

            if debug:
                print(graph)
                return

            torch.save(graph, output_file)
            print(f"Processed graph {idx}")
            
        except Exception as e:
            print(f"Error processing graph {idx}: {e}")

    def process_events_in_parallel(self, n_workers, n_events, debug=False, overwrite=False):
        num_events = self.input_tree.num_entries if n_events == -1 else n_events
        if debug:
            num_events = 1
        n_workers =  min(n_workers, os.cpu_count() // 2) if not debug else 1

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for idx in range(num_events):
                executor.submit(
                    self.process_event,
                    self.input_tree.arrays(ALL_COLUMNS, entry_start=idx, entry_stop=idx + 1),
                    idx, debug, overwrite
                )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, default="./")
    argparser.add_argument("--n_workers", "-n", type=int, default=16)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--overwrite", action="store_true", help="Overwrite existing graphs")
    argparser.add_argument("--n_events", type=int, default=-1, help="Maximum number of events to process (-1 for all)")
    argparser.add_argument("--split", type=float, default=0.8, help="Train/val split ratio")
    args = argparser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    training_data = GraphBuilder(args.input, args.output, args.split)
    training_data.process_events_in_parallel(args.n_workers, args.n_events, args.debug, args.overwrite)