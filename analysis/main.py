import os
import json
import numpy as np
from argparse import ArgumentParser

import torch
from torch_geometric.loader import DataLoader

import sys
sys.path.append('./src')

from dataset import PCDataset
from measures import calculate_tracking_metrics, run_inference_and_clustering, plot_performance_histograms
from validation import validate_model, make_epsilon_validation_plot

def main():
    argparser = ArgumentParser()
    argparser.add_argument('--config', type=str, default='./config.json', help='Path to the config file')
    argparser.add_argument('--output', type=str, help='Path to the trained model file')
    argparser.add_argument('--epsilon-validation', action='store_true', help='Run model validation')
    argparser.add_argument('--epsilon', type=float, default=0.05, help='DBSCAN epsilon parameter')
    argparser.add_argument('--n-events', type=int, default=-1, help='Number of events to process (-1 for all)')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    subset = args.n_events if args.n_events > 0 else None
    dataset = PCDataset(
        config['val_data_dir'], 
        subset=subset,
    )
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1, drop_last=False, num_workers=4)
    
    print(f"Loaded dataset with {len(dataset)} graphs")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_node_features = dataset[0].num_node_features

    if config.get('model_type', 'transformer') == 'transformer':
        from models.transformer import TransformerOCModel
        model = TransformerOCModel(
            input_dim=num_node_features,
            hidden_dim=config.get('hidden_dim', config.get('d_model', 128)),
            num_layers=config['num_layers'],
            num_heads=config['nhead'],
            latent_dim=config['latent_dim'],
            dropout=config.get('dropout', 0.1)
        )
    else:
        from models.hept import HEPTOCModel
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
    
    output = args.output + '/final_model.ckpt'

    if not os.path.exists(output):
        output = args.output + '/last.ckpt'

    state_dict = torch.load(output, weights_only=True)['state_dict']

    if any(key.startswith('model.') for key in state_dict.keys()):
        state_dict = {key.replace('model.', '', 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    if args.epsilon_validation:
        rates = []
        for i, data in enumerate(data_loader):
            if i % 10 == 0:
                print(f"Processing event {i+1}/{len(data_loader)}")
            eps_range = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
            for eps in eps_range:
                cluster = run_inference_and_clustering(data, model, device, eps=eps)
                cluster_labels = cluster.labels_
                rates.append(validate_model(data, cluster_labels))

        make_epsilon_validation_plot(eps_range, rates, args.output)
        print(f"Epsilon validation plots saved to: {args.output}/")
        return

    pt_bins = np.linspace(0.5, 2, 40)
    eta_bins = np.linspace(-4, 4, 40)

    print(f"PT bins: {len(pt_bins)-1} bins from {pt_bins[0]:.2f} to {pt_bins[-1]:.2f} GeV")
    print(f"Eta bins: {len(eta_bins)-1} bins from {eta_bins[0]:.2f} to {eta_bins[-1]:.2f}")

    # Change efficiency variables to track numerator/denominator
    total_efficiency_num_pt = np.zeros(len(pt_bins) - 1)
    total_efficiency_num_eta = np.zeros(len(eta_bins) - 1)
    total_efficiency_den_pt = np.zeros(len(pt_bins) - 1)
    total_efficiency_den_eta = np.zeros(len(eta_bins) - 1)
    total_fake_rate_num_pt = np.zeros(len(pt_bins) - 1)
    total_fake_rate_num_eta = np.zeros(len(eta_bins) - 1)
    total_fake_rate_den_pt = np.zeros(len(pt_bins) - 1)
    total_fake_rate_den_eta = np.zeros(len(eta_bins) - 1)
    total_purity_sum_pt = np.zeros(len(pt_bins) - 1)
    total_purity_sum_eta = np.zeros(len(eta_bins) - 1)
    total_purity_count_pt = np.zeros(len(pt_bins) - 1)
    total_purity_count_eta = np.zeros(len(eta_bins) - 1)
    total_duplicate_rate_num_pt = np.zeros(len(pt_bins) - 1)
    total_duplicate_rate_num_eta = np.zeros(len(eta_bins) - 1)

    print(f"Processing {len(data_loader)} events with epsilon = {args.epsilon}")
    
    for i, data in enumerate(data_loader):
        if i % 10 == 0:
            print(f"Processing event {i+1}/{len(data_loader)}")
            
        cluster = run_inference_and_clustering(data, model, device, eps=args.epsilon)
        cluster_labels = cluster.labels_

        metrics = calculate_tracking_metrics(data, cluster_labels, pt_bins, eta_bins)

        total_efficiency_num_pt += metrics['efficiency_numerator_pt']
        total_efficiency_num_eta += metrics['efficiency_numerator_eta']
        total_efficiency_den_pt += metrics['efficiency_denominator_pt']
        total_efficiency_den_eta += metrics['efficiency_denominator_eta']
        
        total_fake_rate_num_pt += metrics['fake_rate_numerator_pt']
        total_fake_rate_num_eta += metrics['fake_rate_numerator_eta']
        total_fake_rate_den_pt += metrics['total_tracks_pt']
        total_fake_rate_den_eta += metrics['total_tracks_eta']
        
        total_purity_sum_pt += metrics['purity_sum_pt']
        total_purity_sum_eta += metrics['purity_sum_eta']
        total_purity_count_pt += metrics['purity_count_pt']
        total_purity_count_eta += metrics['purity_count_eta']
        
        total_duplicate_rate_num_pt += metrics['duplicate_rate_numerator_pt']
        total_duplicate_rate_num_eta += metrics['duplicate_rate_numerator_eta']

    n_events = len(data_loader)
    
    avg_efficiency_pt = np.divide(total_efficiency_num_pt, total_efficiency_den_pt, 
                                 out=np.zeros_like(total_efficiency_num_pt), where=total_efficiency_den_pt!=0)
    avg_efficiency_eta = np.divide(total_efficiency_num_eta, total_efficiency_den_eta, 
                                  out=np.zeros_like(total_efficiency_num_eta), where=total_efficiency_den_eta!=0)
    
    eff_err_pt = np.sqrt(total_efficiency_num_pt * (total_efficiency_den_pt - total_efficiency_num_pt) / total_efficiency_den_pt**3)
    eff_err_pt = np.where(total_efficiency_den_pt > 0, eff_err_pt, 0)
    eff_err_eta = np.sqrt(total_efficiency_num_eta * (total_efficiency_den_eta - total_efficiency_num_eta) / total_efficiency_den_eta**3)
    eff_err_eta = np.where(total_efficiency_den_eta > 0, eff_err_eta, 0)
    
    avg_fake_rate_pt = np.divide(total_fake_rate_num_pt, total_fake_rate_den_pt, 
                                out=np.zeros_like(total_fake_rate_num_pt), where=total_fake_rate_den_pt!=0)
    avg_fake_rate_eta = np.divide(total_fake_rate_num_eta, total_fake_rate_den_eta, 
                                 out=np.zeros_like(total_fake_rate_num_eta), where=total_fake_rate_den_eta!=0)
    
    fake_err_pt = np.sqrt(total_fake_rate_num_pt * (total_fake_rate_den_pt - total_fake_rate_num_pt) / total_fake_rate_den_pt**3)
    fake_err_pt = np.where(total_fake_rate_den_pt > 0, fake_err_pt, 0)
    fake_err_eta = np.sqrt(total_fake_rate_num_eta * (total_fake_rate_den_eta - total_fake_rate_num_eta) / total_fake_rate_den_eta**3)
    fake_err_eta = np.where(total_fake_rate_den_eta > 0, fake_err_eta, 0)
    
    avg_purity_pt = np.divide(total_purity_sum_pt, total_purity_count_pt, 
                             out=np.zeros_like(total_purity_sum_pt), where=total_purity_count_pt!=0)
    avg_purity_eta = np.divide(total_purity_sum_eta, total_purity_count_eta, 
                              out=np.zeros_like(total_purity_sum_eta), where=total_purity_count_eta!=0)
    
    purity_err_pt = np.sqrt(avg_purity_pt * (1 - avg_purity_pt) / total_purity_count_pt)
    purity_err_pt = np.where(total_purity_count_pt > 0, purity_err_pt, 0)
    purity_err_eta = np.sqrt(avg_purity_eta * (1 - avg_purity_eta) / total_purity_count_eta)
    purity_err_eta = np.where(total_purity_count_eta > 0, purity_err_eta, 0)
    
    # Duplicate rate: number of duplicated sim particles / total sim particles
    avg_duplicate_rate_pt = np.divide(total_duplicate_rate_num_pt, total_efficiency_den_pt,
                                     out=np.zeros_like(total_duplicate_rate_num_pt), where=total_efficiency_den_pt!=0)
    avg_duplicate_rate_eta = np.divide(total_duplicate_rate_num_eta, total_efficiency_den_eta,
                                      out=np.zeros_like(total_duplicate_rate_num_eta), where=total_efficiency_den_eta!=0)
    
    dup_err_pt = np.sqrt(total_duplicate_rate_num_pt * (total_efficiency_den_pt - total_duplicate_rate_num_pt) / total_efficiency_den_pt**3)
    dup_err_pt = np.where(total_efficiency_den_pt > 0, dup_err_pt, 0)
    dup_err_eta = np.sqrt(total_duplicate_rate_num_eta * (total_efficiency_den_eta - total_duplicate_rate_num_eta) / total_efficiency_den_eta**3)
    dup_err_eta = np.where(total_efficiency_den_eta > 0, dup_err_eta, 0)

    metrics_pt = {
        'efficiency': avg_efficiency_pt,
        'fake_rate': avg_fake_rate_pt,
        'purity': avg_purity_pt,
        'duplicate_rate': avg_duplicate_rate_pt,
        'efficiency_err': eff_err_pt,
        'fake_rate_err': fake_err_pt,
        'purity_err': purity_err_pt,
        'duplicate_rate_err': dup_err_pt
    }
    
    metrics_eta = {
        'efficiency': avg_efficiency_eta,
        'fake_rate': avg_fake_rate_eta,
        'purity': avg_purity_eta,
        'duplicate_rate': avg_duplicate_rate_eta,
        'efficiency_err': eff_err_eta,
        'fake_rate_err': fake_err_eta,
        'purity_err': purity_err_eta,
        'duplicate_rate_err': dup_err_eta
    }

    plot_performance_histograms(pt_bins, eta_bins, metrics_pt, metrics_eta, args.output, args.epsilon)

    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Epsilon: {args.epsilon}")
    print(f"Events processed: {n_events}")
    print(f"Output directory: {args.output}")
    print("\nOverall averages:")
    print(f"  Efficiency: {np.mean(avg_efficiency_pt[avg_efficiency_pt > 0]):.3f}")
    print(f"  Fake Rate: {np.mean(avg_fake_rate_pt[avg_fake_rate_pt > 0]):.3f}")
    print(f"  Purity: {np.mean(avg_purity_pt[avg_purity_pt > 0]):.3f}")

    print(f"Plots saved to: {args.output}/")

if __name__ == "__main__":
    main()
