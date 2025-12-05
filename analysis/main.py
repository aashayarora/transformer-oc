import os
from glob import glob
import numpy as np
from argparse import ArgumentParser
import pathlib

import torch
from torch_geometric.loader import DataLoader

import sys
sys.path.append('./src')

from dataset import PCDataset
from measures import calculate_tracking_metrics, run_inference_and_clustering, plot_performance_histograms, plot_cluster_size_histogram
from validation import validate_model, make_epsilon_validation_plot

def main():
    argparser = ArgumentParser()
    argparser.add_argument('--config', type=str, default='./config.yaml', help='Path to the config file (YAML)')
    argparser.add_argument('--output', type=str, help='Path to the trained model file')
    argparser.add_argument('--epsilon-validation', action='store_true', help='Run model validation')
    argparser.add_argument('--epsilon', type=float, default=None, help='DBSCAN epsilon parameter (overrides config)')
    argparser.add_argument('--n-events', type=int, default=-1, help='Number of events to process (-1 for all)')
    args = argparser.parse_args()

    from helpers import load_config
    config = load_config(args.config)

    subset = args.n_events if args.n_events > 0 else None
    dataset = PCDataset(
        config['val_data_dir'], 
        subset=subset,
    )
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, drop_last=False, num_workers=2)
    
    print(f"Loaded dataset with {len(dataset)} graphs")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = device.type == 'cuda'  # Only use FP16 on CUDA
    print(f"Using device: {device}, FP16: {use_fp16}")

    num_node_features = dataset[0].num_node_features

    if config.get('model_type', 'transformer') == 'transformer':
        from model import TransformerOCModel
        model = TransformerOCModel(
            input_dim=num_node_features,
            hidden_dim=config.get('hidden_dim', config.get('d_model', 128)),
            num_layers=config['num_layers'],
            num_heads=config['nhead'],
            latent_dim=config['latent_dim'],
            dropout=config.get('dropout', 0.1)
        )
    else:
        from model import HEPTOCModel
        model = HEPTOCModel(
                input_dim=num_node_features,
                hidden_dim=config.get('hidden_dim', config.get('d_model', 128)),
                num_layers=config['num_layers'],
                num_heads=config['nhead'],
                latent_dim=config['latent_dim'],
                dropout=config.get('dropout', 0.1),
                coords_dim=2,
                block_size=config.get('block_size', 32),
                n_hashes=config.get('n_hashes', 4),
                num_regions=config.get('num_regions', 8),
                num_w_per_dist=config.get('num_w_per_dist', 8)
            )
        
    model.to(device)
    
    if ".pt" in args.output:
        output = args.output
    else:
        output = args.output + '/final_model.ckpt'

    args.output = args.output if not args.output.endswith('.ckpt') else str(pathlib.Path(args.output).parent)

    if not os.path.exists(output):
        output = sorted(glob(os.path.join(args.output, 'checkpoints', '*.ckpt')))[-1]

    print(f"Loading model from: {output}")

    state_dict = torch.load(output, weights_only=True)

    if any(key.startswith('model.') for key in state_dict.keys()):
        state_dict = {key.replace('model.', '', 1): value for key, value in state_dict.items()}

    try:
        model.load_state_dict(state_dict)['state_dict']
    except:
        model.load_state_dict(state_dict)

    model.eval()

    # Get DBSCAN and purity parameters from config (needed for both epsilon validation and regular inference)
    min_samples = config.get('dbscan_min_samples', 2)
    purity_threshold = config.get('purity_threshold', 0.75)

    if args.epsilon_validation:
        eps_range = config.get('epsilon_validation_values', [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0])
        rates = []
        for i, data in enumerate(data_loader):
            if i % 10 == 0:
                print(f"Processing event {i+1}/{len(data_loader)}")
            for eps in eps_range:
                cluster = run_inference_and_clustering(data, model, device, eps=eps, min_samples=min_samples, use_fp16=use_fp16)
                cluster_labels = cluster.labels_
                rates.append(validate_model(data, cluster_labels, purity_threshold=purity_threshold))

        make_epsilon_validation_plot(eps_range, rates, args.output)
        print(f"Epsilon validation plots saved to: {args.output}/")
        return

    # Get binning parameters from config
    pt_bins = np.linspace(
        config.get('pt_bins_start', 0.5),
        config.get('pt_bins_end', 2.0),
        config.get('pt_bins_count', 50)
    )
    eta_bins = np.linspace(
        config.get('eta_bins_start', -4),
        config.get('eta_bins_end', 4),
        config.get('eta_bins_count', 50)
    )

    print(f"PT bins: {len(pt_bins)-1} bins from {pt_bins[0]:.2f} to {pt_bins[-1]:.2f} GeV")
    print(f"Eta bins: {len(eta_bins)-1} bins from {eta_bins[0]:.2f} to {eta_bins[-1]:.2f}")

    hist_eff_num_pt = np.zeros(len(pt_bins) - 1)
    hist_eff_den_pt = np.zeros(len(pt_bins) - 1)
    hist_eff_num_eta = np.zeros(len(eta_bins) - 1)
    hist_eff_den_eta = np.zeros(len(eta_bins) - 1)
    
    hist_fake_num_pt = np.zeros(len(pt_bins) - 1)
    hist_fake_den_pt = np.zeros(len(pt_bins) - 1)
    hist_fake_num_eta = np.zeros(len(eta_bins) - 1)
    hist_fake_den_eta = np.zeros(len(eta_bins) - 1)
    
    hist_purity_sum_pt = np.zeros(len(pt_bins) - 1)
    hist_purity_count_pt = np.zeros(len(pt_bins) - 1)
    hist_purity_sum_eta = np.zeros(len(eta_bins) - 1)
    hist_purity_count_eta = np.zeros(len(eta_bins) - 1)
    
    hist_dup_num_pt = np.zeros(len(pt_bins) - 1)
    hist_dup_num_eta = np.zeros(len(eta_bins) - 1)

    all_cluster_sizes = []  # Collect cluster sizes from all events

    print(f"Processing {len(data_loader)} events with epsilon = {args.epsilon if args.epsilon else config.get('dbscan_eps', 0.05)}")
    
    # Get epsilon from args, or fall back to config
    epsilon = args.epsilon if args.epsilon is not None else config.get('dbscan_eps', 0.05)
    eta_cut = config.get('eta_cut', (-2.5, 2.5))
    
    for i, data in enumerate(data_loader):
        if i % 10 == 0:
            print(f"Processing event {i+1}/{len(data_loader)}")
            
        cluster = run_inference_and_clustering(data, model, device, eps=epsilon, min_samples=min_samples, use_fp16=use_fp16)
        cluster_labels = cluster.labels_
        metrics = calculate_tracking_metrics(data, cluster_labels, pt_bins, eta_bins, purity_threshold=purity_threshold, eta_cut=eta_cut)

        # Simply accumulate histograms
        hist_eff_num_pt += metrics['efficiency_numerator_pt']
        hist_eff_den_pt += metrics['efficiency_denominator_pt']
        hist_eff_num_eta += metrics['efficiency_numerator_eta']
        hist_eff_den_eta += metrics['efficiency_denominator_eta']
        
        hist_fake_num_pt += metrics['fake_rate_numerator_pt']
        hist_fake_den_pt += metrics['total_tracks_pt']
        hist_fake_num_eta += metrics['fake_rate_numerator_eta']
        hist_fake_den_eta += metrics['total_tracks_eta']
        
        hist_purity_sum_pt += metrics['purity_sum_pt']
        hist_purity_count_pt += metrics['purity_count_pt']
        hist_purity_sum_eta += metrics['purity_sum_eta']
        hist_purity_count_eta += metrics['purity_count_eta']
        
        hist_dup_num_pt += metrics['duplicate_rate_numerator_pt']
        hist_dup_num_eta += metrics['duplicate_rate_numerator_eta']
        
        # Collect cluster sizes
        all_cluster_sizes.extend(metrics['cluster_sizes'])

    def binomial_error(num, den):
        err = np.sqrt(num * (den - num) / np.power(den, 3, where=den>0))
        return np.where(den > 0, err, 0)
    
    eff_pt = np.divide(hist_eff_num_pt, hist_eff_den_pt, out=np.zeros_like(hist_eff_num_pt), where=hist_eff_den_pt!=0)
    eff_eta = np.divide(hist_eff_num_eta, hist_eff_den_eta, out=np.zeros_like(hist_eff_num_eta), where=hist_eff_den_eta!=0)
    
    fake_pt = np.divide(hist_fake_num_pt, hist_fake_den_pt, out=np.zeros_like(hist_fake_num_pt), where=hist_fake_den_pt!=0)
    fake_eta = np.divide(hist_fake_num_eta, hist_fake_den_eta, out=np.zeros_like(hist_fake_num_eta), where=hist_fake_den_eta!=0)
    
    purity_pt = np.divide(hist_purity_sum_pt, hist_purity_count_pt, out=np.zeros_like(hist_purity_sum_pt), where=hist_purity_count_pt!=0)
    purity_eta = np.divide(hist_purity_sum_eta, hist_purity_count_eta, out=np.zeros_like(hist_purity_sum_eta), where=hist_purity_count_eta!=0)
    
    dup_pt = np.divide(hist_dup_num_pt, hist_eff_den_pt, out=np.zeros_like(hist_dup_num_pt), where=hist_eff_den_pt!=0)
    dup_eta = np.divide(hist_dup_num_eta, hist_eff_den_eta, out=np.zeros_like(hist_dup_num_eta), where=hist_eff_den_eta!=0)
    
    eff_err_pt = binomial_error(hist_eff_num_pt, hist_eff_den_pt)
    eff_err_eta = binomial_error(hist_eff_num_eta, hist_eff_den_eta)
    fake_err_pt = binomial_error(hist_fake_num_pt, hist_fake_den_pt)
    fake_err_eta = binomial_error(hist_fake_num_eta, hist_fake_den_eta)
    dup_err_pt = binomial_error(hist_dup_num_pt, hist_eff_den_pt)
    dup_err_eta = binomial_error(hist_dup_num_eta, hist_eff_den_eta)
    
    purity_err_pt = np.sqrt(purity_pt * (1 - purity_pt) / hist_purity_count_pt)
    purity_err_pt = np.where(hist_purity_count_pt > 0, purity_err_pt, 0)
    purity_err_eta = np.sqrt(purity_eta * (1 - purity_eta) / hist_purity_count_eta)
    purity_err_eta = np.where(hist_purity_count_eta > 0, purity_err_eta, 0)

    metrics_pt = {
        'efficiency': eff_pt,
        'fake_rate': fake_pt,
        'purity': purity_pt,
        'duplicate_rate': dup_pt,
        'efficiency_err': eff_err_pt,
        'fake_rate_err': fake_err_pt,
        'purity_err': purity_err_pt,
        'duplicate_rate_err': dup_err_pt
    }
    
    metrics_eta = {
        'efficiency': eff_eta,
        'fake_rate': fake_eta,
        'purity': purity_eta,
        'duplicate_rate': dup_eta,
        'efficiency_err': eff_err_eta,
        'fake_rate_err': fake_err_eta,
        'purity_err': purity_err_eta,
        'duplicate_rate_err': dup_err_eta
    }

    plot_performance_histograms(pt_bins, eta_bins, metrics_pt, metrics_eta, args.output, epsilon)

    # Plot cluster size histogram
    if all_cluster_sizes:
        plot_cluster_size_histogram(all_cluster_sizes, args.output, epsilon)
        print(f"\nCluster size statistics:")
        print(f"  Total clusters: {len(all_cluster_sizes)}")
        print(f"  Mean cluster size: {np.mean(all_cluster_sizes):.2f}")
        print(f"  Median cluster size: {np.median(all_cluster_sizes):.0f}")
        print(f"  Min/Max cluster size: {min(all_cluster_sizes)}/{max(all_cluster_sizes)}")

    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Epsilon: {epsilon}")
    print(f"Purity Threshold: {purity_threshold}")
    print(f"Events processed: {len(data_loader)}")
    print(f"Output directory: {args.output}")
    print("\nOverall averages:")
    print(f"  Efficiency: {np.mean(eff_pt[eff_pt > 0]):.3f}")
    print(f"  Fake Rate: {np.mean(fake_pt[fake_pt > 0]):.3f}")
    print(f"  Purity: {np.mean(purity_pt[purity_pt > 0]):.3f}")

    print(f"Plots saved to: {args.output}/")

if __name__ == "__main__":
    main()
