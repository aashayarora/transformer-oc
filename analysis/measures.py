import sys
sys.path.append('/home/users/aaarora/phys/tracking/transformers/oc')
sys.path.append('/home/users/aaarora/phys/tracking/transformers/oc/training')

import numpy as np
import torch

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ROOT)

@torch.no_grad()
def run_inference_and_clustering(data, model, device, eps=0.2):
    x = data.x.to(device)
    batch = data.batch.to(device)
    coords = data.x[:, 1:3].to(device)
    
    model.eval()
    out = model(x, coords, batch)
    X = out["H"].cpu().detach().numpy()
    cluster = DBSCAN(eps=eps, min_samples=2).fit(X)
    return cluster

def calculate_tracking_metrics(data, cluster_labels, pt_bins, eta_bins, purity_threshold=0.75, eta_cut=None):
    sim_features = data.sim_features
    truth_labels = data.sim_index.cpu().detach().numpy().flatten()
    
    n_bins_pt = len(pt_bins) - 1
    n_bins_eta = len(eta_bins) - 1
    
    hist_eff_num_pt = np.zeros(n_bins_pt)
    hist_eff_den_pt = np.zeros(n_bins_pt)
    hist_eff_num_eta = np.zeros(n_bins_eta)
    hist_eff_den_eta = np.zeros(n_bins_eta)
    hist_fake_num_pt = np.zeros(n_bins_pt)
    hist_fake_num_eta = np.zeros(n_bins_eta)
    hist_tracks_pt = np.zeros(n_bins_pt)
    hist_tracks_eta = np.zeros(n_bins_eta)
    hist_purity_sum_pt = np.zeros(n_bins_pt)
    hist_purity_sum_eta = np.zeros(n_bins_eta)
    hist_purity_count_pt = np.zeros(n_bins_pt)
    hist_purity_count_eta = np.zeros(n_bins_eta)
    hist_dup_num_pt = np.zeros(n_bins_pt)
    hist_dup_num_eta = np.zeros(n_bins_eta)
    
    correctly_reconstructed_sim = set()
    sim_to_pred_clusters = {}  
    
    unique_pred_labels = np.unique(cluster_labels[cluster_labels >= 0])
    
    for pred_label in unique_pred_labels:
        mask = cluster_labels == pred_label
        truth_in_cluster = truth_labels[mask]
        truth_no_fake = truth_in_cluster[truth_in_cluster >= 0]
        
        if len(truth_no_fake) == 0:
            is_fake, purity = True, 0.0
            track_pt = track_eta = -999.0
        else:
            unique_labels, counts = np.unique(truth_no_fake, return_counts=True)
            majority_label = int(unique_labels[np.argmax(counts)])
            purity = np.sum(truth_in_cluster == majority_label) / len(truth_in_cluster)
            is_fake = purity < purity_threshold
            
            if not is_fake:
                correctly_reconstructed_sim.add(majority_label)
                sim_to_pred_clusters.setdefault(majority_label, []).append(pred_label)
            
            if majority_label < len(sim_features):
                track_pt = float(sim_features[majority_label][0])
                track_eta = float(sim_features[majority_label][1])
            else:
                track_pt = track_eta = -999.0
        
        pt_bin = np.digitize(track_pt, pt_bins) - 1
        eta_bin = np.digitize(track_eta, eta_bins) - 1
        
        if 0 <= pt_bin < n_bins_pt:
            hist_tracks_pt[pt_bin] += 1
            if is_fake:
                hist_fake_num_pt[pt_bin] += 1
            else:
                hist_purity_sum_pt[pt_bin] += purity
                hist_purity_count_pt[pt_bin] += 1
                
        if 0 <= eta_bin < n_bins_eta:
            hist_tracks_eta[eta_bin] += 1
            if is_fake:
                hist_fake_num_eta[eta_bin] += 1
            else:
                hist_purity_sum_eta[eta_bin] += purity
                hist_purity_count_eta[eta_bin] += 1
    
    unique_truth_labels = np.unique(truth_labels[truth_labels >= 0])
    
    valid_sim_mask = unique_truth_labels < len(sim_features)
    valid_sim_labels = unique_truth_labels[valid_sim_mask].astype(int)
    sim_pts = np.array([float(sim_features[i][0]) for i in valid_sim_labels])
    sim_etas = np.array([float(sim_features[i][1]) for i in valid_sim_labels])
    sim_reconstructed = np.array([int(i in correctly_reconstructed_sim) for i in valid_sim_labels])
    sim_duplicated = np.array([int(len(sim_to_pred_clusters.get(i, [])) > 1) for i in valid_sim_labels])
    
    if eta_cut is not None:
        eta_min, eta_max = eta_cut
        in_acceptance = (sim_etas >= eta_min) & (sim_etas <= eta_max)
    else:
        in_acceptance = np.ones(len(sim_etas), dtype=bool)
    
    pt_bins_idx = np.digitize(sim_pts[in_acceptance], pt_bins) - 1
    valid_pt = (pt_bins_idx >= 0) & (pt_bins_idx < n_bins_pt)
    np.add.at(hist_eff_den_pt, pt_bins_idx[valid_pt], 1)
    np.add.at(hist_eff_num_pt, pt_bins_idx[valid_pt], sim_reconstructed[in_acceptance][valid_pt])
    np.add.at(hist_dup_num_pt, pt_bins_idx[valid_pt], sim_duplicated[in_acceptance][valid_pt])
    
    eta_bins_idx = np.digitize(sim_etas, eta_bins) - 1
    valid_eta = (eta_bins_idx >= 0) & (eta_bins_idx < n_bins_eta)
    np.add.at(hist_eff_den_eta, eta_bins_idx[valid_eta], 1)
    np.add.at(hist_eff_num_eta, eta_bins_idx[valid_eta], sim_reconstructed[valid_eta])
    np.add.at(hist_dup_num_eta, eta_bins_idx[valid_eta], sim_duplicated[valid_eta])
    
    return {
        'efficiency_numerator_pt': hist_eff_num_pt,
        'efficiency_numerator_eta': hist_eff_num_eta,
        'efficiency_denominator_pt': hist_eff_den_pt,
        'efficiency_denominator_eta': hist_eff_den_eta,
        'fake_rate_numerator_pt': hist_fake_num_pt,
        'fake_rate_numerator_eta': hist_fake_num_eta,
        'purity_sum_pt': hist_purity_sum_pt,
        'purity_sum_eta': hist_purity_sum_eta,
        'total_tracks_pt': hist_tracks_pt,
        'total_tracks_eta': hist_tracks_eta,
        'purity_count_pt': hist_purity_count_pt,
        'purity_count_eta': hist_purity_count_eta,
        'duplicate_rate_numerator_pt': hist_dup_num_pt,
        'duplicate_rate_numerator_eta': hist_dup_num_eta
    }

def plot_performance_histograms(pt_bins, eta_bins, metrics_pt, metrics_eta, output_path, epsilon):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2
    
    # First row: pT metrics
    axes[0, 0].errorbar(pt_centers, metrics_pt['efficiency'], yerr=metrics_pt['efficiency_err'], 
                       fmt='o', color='blue', label='Efficiency', capsize=3)
    axes[0, 0].set_xlabel('Sim $p_T$ [GeV]')
    axes[0, 0].set_ylabel('Efficiency')
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].set_title('Efficiency vs $p_T$')
    
    axes[0, 1].errorbar(pt_centers, metrics_pt['fake_rate'], yerr=metrics_pt['fake_rate_err'], 
                       fmt='o', color='red', label='Fake Rate', capsize=3)
    axes[0, 1].set_xlabel('Sim $p_T$ [GeV]')
    axes[0, 1].set_ylabel('Fake Rate')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Fake Rate vs $p_T$')
    
    axes[0, 2].errorbar(pt_centers, metrics_pt['purity'], yerr=metrics_pt['purity_err'], 
                    fmt='o', color='green', label='Purity', capsize=3)
    axes[0, 2].set_xlabel('Sim $p_T$ [GeV]')
    axes[0, 2].set_ylabel('Purity')
    axes[0, 2].set_ylim(0, 1.1)
    axes[0, 2].set_title('Purity vs $p_T$')
    
    axes[0, 3].errorbar(pt_centers, metrics_pt['duplicate_rate'], yerr=metrics_pt['duplicate_rate_err'], 
                    fmt='o', color='purple', label='Duplicate Rate', capsize=3)
    axes[0, 3].set_xlabel('Sim $p_T$ [GeV]')
    axes[0, 3].set_ylabel('Duplicate Rate')
    axes[0, 3].set_ylim(0, max(0.5, np.max(metrics_pt['duplicate_rate']) * 1.2))
    axes[0, 3].set_title('Duplicate Rate vs $p_T$')
    
    # Second row: eta metrics
    axes[1, 0].errorbar(eta_centers, metrics_eta['efficiency'], yerr=metrics_eta['efficiency_err'], 
                       fmt='o', color='blue', label='Efficiency', capsize=3)
    axes[1, 0].set_xlabel('Sim $η$')
    axes[1, 0].set_ylabel('Efficiency')
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].set_title('Efficiency vs $η$')
    
    axes[1, 1].errorbar(eta_centers, metrics_eta['fake_rate'], yerr=metrics_eta['fake_rate_err'], 
                       fmt='o', color='red', label='Fake Rate', capsize=3)
    axes[1, 1].set_xlabel('Sim $η$')
    axes[1, 1].set_ylabel('Fake Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Fake Rate vs $η$')
    
    axes[1, 2].errorbar(eta_centers, metrics_eta['purity'], yerr=metrics_eta['purity_err'], 
                    fmt='o', color='green', label='Purity', capsize=3)
    axes[1, 2].set_xlabel('Sim $η$')
    axes[1, 2].set_ylabel('Purity')
    axes[1, 2].set_ylim(0, 1.1)
    axes[1, 2].set_title('Purity vs $η$')
    
    axes[1, 3].errorbar(eta_centers, metrics_eta['duplicate_rate'], yerr=metrics_eta['duplicate_rate_err'], 
                    fmt='o', color='purple', label='Duplicate Rate', capsize=3)
    axes[1, 3].set_xlabel('Sim $η$')
    axes[1, 3].set_ylabel('Duplicate Rate')
    axes[1, 3].set_ylim(0, max(0.5, np.max(metrics_eta['duplicate_rate']) * 1.2))
    axes[1, 3].set_title('Duplicate Rate vs $η$')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/performance_metrics_eps_{epsilon:.3f}.png", dpi=300, bbox_inches='tight')
    plt.close()