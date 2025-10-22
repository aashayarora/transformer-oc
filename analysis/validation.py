import sys
sys.path.append('/home/users/aaarora/phys/tracking/transformers/oc')
sys.path.append('/home/users/aaarora/phys/tracking/transformers/oc/training')

import numpy as np
import torch
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ROOT)

def validate_model(data, cluster_labels, purity_threshold=0.75):
    truth_labels = data.sim_index.cpu().detach().numpy().flatten()

    uniq_truth_labels = sorted([x for x in set(truth_labels) if x >= 0])

    perfect = []
    lhc = []
    dm = []
    perf_truth = []
    perf_fakes = []

    for uniq_cl in uniq_truth_labels:
        true_cluster_indices = np.where(truth_labels == uniq_cl)[0]
        cluster_dbscan_labels = cluster_labels[true_cluster_indices]
        
        if np.all(cluster_dbscan_labels == -1):
            perfect.append(False)
            lhc.append(0)
            dm.append(0)
            perf_truth.append(0.0)
            perf_fakes.append(1.0)
            # print(f"Cluster {uniq_cl}: completely missed (all points labeled as noise)")
            continue

        non_noise_labels = cluster_dbscan_labels[cluster_dbscan_labels != -1]
        if len(non_noise_labels) == 0:
            perfect.append(False)
            lhc.append(0)
            dm.append(0)
            perf_truth.append(0.0)
            perf_fakes.append(1.0)
            continue
        
        unique_labels, counts = np.unique(non_noise_labels, return_counts=True)
        dbscan_label = unique_labels[np.argmax(counts)]

        num_elements_true = len(true_cluster_indices)
        num_elements_pred = np.sum(cluster_labels == dbscan_label)
        num_elements_correct = np.sum(truth_labels[cluster_labels == dbscan_label] == uniq_cl)
        num_elements_fake = num_elements_pred - num_elements_correct
        
        perfect.append(num_elements_correct == num_elements_true and num_elements_pred == num_elements_true)
        lhc.append(1 if num_elements_pred > 0 and num_elements_correct / num_elements_pred >= purity_threshold else 0)
        dm.append(1 if (num_elements_true > 0 and num_elements_correct / num_elements_true >= 0.5 and 
                       num_elements_pred > 0 and num_elements_fake / num_elements_pred < 0.5) else 0)
        
        perf_truth.append(num_elements_correct / num_elements_true)
        perf_fakes.append(num_elements_fake / num_elements_pred)
        
    if len(perfect) == 0:
        return 0.0, 0.0, 0.0
    
    return (np.mean(perfect), np.mean(lhc), np.mean(dm))


def make_epsilon_validation_plot(eps_range, rates, output_dir):
    rates = np.array(rates).reshape(-1, len(eps_range), 3)
    avg_rates = np.mean(rates, axis=0)
    eps_range = np.array(eps_range)

    fig, ax = plt.subplots()
    ax.plot(eps_range, avg_rates[:, 0], 'o-', label='Perfect')
    ax.plot(eps_range, avg_rates[:, 1], 's-', label='LHC')
    ax.plot(eps_range, avg_rates[:, 2], '^-', label='DM')

    ax.set_ylim(0, 1.05)
    
    ax.set_xlabel('Epsilon (Clustering Distance Threshold)')
    ax.set_ylabel('Rate')
    ax.legend()

    plt.savefig(f"{output_dir}/validation_vs_epsilon.png")