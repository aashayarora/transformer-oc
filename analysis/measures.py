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
    # Extract eta and phi coordinates from features (columns 1 and 2)
    coords = data.x[:, 1:3].to(device)
    
    model.eval()
    out = model(x, coords, batch)
    X = out["H"].cpu().detach().numpy()
    cluster = DBSCAN(eps=eps, min_samples=2).fit(X)
    return cluster

def calculate_tracking_metrics(data, cluster_labels, pt_bins, eta_bins, purity_threshold=0.75):
    sim_features = data.sim_features
    truth_labels = data.sim_index.cpu().detach().numpy().flatten()
    
    unique_truth_labels = [x for x in set(truth_labels) if x >= 0]
    unique_pred_labels = [x for x in set(cluster_labels) if x >= 0]
    
    n_bins_pt = len(pt_bins) - 1
    n_bins_eta = len(eta_bins) - 1
    
    correctly_reconstructed_sim = set()
    sim_to_pred_clusters = {}  # Track which sim particles have multiple reconstructed tracks
    efficiency_numerator_pt = np.zeros(n_bins_pt)
    efficiency_numerator_eta = np.zeros(n_bins_eta)
    efficiency_denominator_pt = np.zeros(n_bins_pt)
    efficiency_denominator_eta = np.zeros(n_bins_eta)
    
    fake_rate_pt = np.zeros(n_bins_pt)
    fake_rate_eta = np.zeros(n_bins_eta)
    total_tracks_pt = np.zeros(n_bins_pt)
    total_tracks_eta = np.zeros(n_bins_eta)
    
    purity_sum_pt = np.zeros(n_bins_pt)
    purity_sum_eta = np.zeros(n_bins_eta)
    purity_count_pt = np.zeros(n_bins_pt)
    purity_count_eta = np.zeros(n_bins_eta)
    
    duplicate_rate_pt = np.zeros(n_bins_pt)
    duplicate_rate_eta = np.zeros(n_bins_eta)
    
    for pred_label in unique_pred_labels:
        pred_cluster_indices = np.where(cluster_labels == pred_label)[0]
        
        if len(pred_cluster_indices) == 0:
            continue
            
        truth_labels_in_cluster = truth_labels[pred_cluster_indices]
        truth_labels_no_fake = truth_labels_in_cluster[truth_labels_in_cluster >= 0]
        
        if len(truth_labels_no_fake) == 0:
            is_fake = True
            track_pt = 1.0
            track_eta = 0.0
            purity = 0.0
        else:
            unique_labels, counts = np.unique(truth_labels_no_fake, return_counts=True)
            majority_label = int(unique_labels[np.argmax(counts)])
            
            n_correct = np.sum(truth_labels_in_cluster == majority_label)
            n_total = len(pred_cluster_indices)
            purity = n_correct / n_total
            
            is_fake = purity < purity_threshold
            
            if purity >= purity_threshold:
                correctly_reconstructed_sim.add(majority_label)
                # Track duplicates: record which sim particles have been reconstructed
                if majority_label not in sim_to_pred_clusters:
                    sim_to_pred_clusters[majority_label] = []
                sim_to_pred_clusters[majority_label].append(pred_label)
            
            if majority_label < len(sim_features):
                track_pt = float(sim_features[majority_label][0])
                track_eta = float(sim_features[majority_label][1])
            else:
                track_pt = 1.0
                track_eta = 0.0
        
        # Bin the track
        pt_bin_idx = int(np.digitize(track_pt, pt_bins) - 1)
        eta_bin_idx = int(np.digitize(track_eta, eta_bins) - 1)
        
        if 0 <= pt_bin_idx < n_bins_pt:
            total_tracks_pt[pt_bin_idx] += 1
            if is_fake:
                fake_rate_pt[pt_bin_idx] += 1
                
        if 0 <= eta_bin_idx < n_bins_eta:
            total_tracks_eta[eta_bin_idx] += 1
            if is_fake:
                fake_rate_eta[eta_bin_idx] += 1
        
        if not is_fake and len(truth_labels_no_fake) > 0:
            if 0 <= pt_bin_idx < n_bins_pt:
                purity_sum_pt[pt_bin_idx] += purity
                purity_count_pt[pt_bin_idx] += 1
                
            if 0 <= eta_bin_idx < n_bins_eta:
                purity_sum_eta[eta_bin_idx] += purity
                purity_count_eta[eta_bin_idx] += 1
    
    for sim_label in unique_truth_labels:
        sim_label = int(sim_label)
        
        if sim_label >= len(sim_features):
            continue
            
        sim_pt = float(sim_features[sim_label][0])
        sim_eta = float(sim_features[sim_label][1])
        
        pt_bin_idx = int(np.digitize(sim_pt, pt_bins) - 1)
        eta_bin_idx = int(np.digitize(sim_eta, eta_bins) - 1)
        
        if 0 <= pt_bin_idx < n_bins_pt:
            efficiency_denominator_pt[pt_bin_idx] += 1
            
        if 0 <= eta_bin_idx < n_bins_eta:
            efficiency_denominator_eta[eta_bin_idx] += 1
    
    for sim_label in unique_truth_labels:
        sim_label = int(sim_label)
        
        if sim_label >= len(sim_features):
            continue
            
        sim_pt = float(sim_features[sim_label][0])
        sim_eta = float(sim_features[sim_label][1])
        
        is_correctly_reconstructed = int(sim_label in correctly_reconstructed_sim)
        
        pt_bin_idx = int(np.digitize(sim_pt, pt_bins) - 1)
        eta_bin_idx = int(np.digitize(sim_eta, eta_bins) - 1)
        
        if 0 <= pt_bin_idx < n_bins_pt:
            efficiency_numerator_pt[pt_bin_idx] += is_correctly_reconstructed
            
        if 0 <= eta_bin_idx < n_bins_eta:
            efficiency_numerator_eta[eta_bin_idx] += is_correctly_reconstructed
    
    # Calculate duplicate rates: count sim particles with multiple reconstructed tracks
    for sim_label, pred_clusters in sim_to_pred_clusters.items():
        if len(pred_clusters) > 1:  # This sim particle has duplicates
            if sim_label < len(sim_features):
                sim_pt = float(sim_features[sim_label][0])
                sim_eta = float(sim_features[sim_label][1])
                
                pt_bin_idx = int(np.digitize(sim_pt, pt_bins) - 1)
                eta_bin_idx = int(np.digitize(sim_eta, eta_bins) - 1)
                
                if 0 <= pt_bin_idx < n_bins_pt:
                    duplicate_rate_pt[pt_bin_idx] += 1
                    
                if 0 <= eta_bin_idx < n_bins_eta:
                    duplicate_rate_eta[eta_bin_idx] += 1
    
    return {
        'efficiency_numerator_pt': efficiency_numerator_pt,
        'efficiency_numerator_eta': efficiency_numerator_eta,
        'efficiency_denominator_pt': efficiency_denominator_pt,
        'efficiency_denominator_eta': efficiency_denominator_eta,
        'fake_rate_numerator_pt': fake_rate_pt,  # Raw fake count
        'fake_rate_numerator_eta': fake_rate_eta,  # Raw fake count
        'purity_sum_pt': purity_sum_pt,
        'purity_sum_eta': purity_sum_eta,
        'total_tracks_pt': total_tracks_pt,
        'total_tracks_eta': total_tracks_eta,
        'purity_count_pt': purity_count_pt,
        'purity_count_eta': purity_count_eta,
        'duplicate_rate_numerator_pt': duplicate_rate_pt,
        'duplicate_rate_numerator_eta': duplicate_rate_eta
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