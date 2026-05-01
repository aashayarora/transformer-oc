import os
from glob import glob
import numpy as np
from argparse import ArgumentParser
import pathlib

import time

import torch
from torch_geometric.loader import DataLoader

import uproot
import awkward as ak

import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)

from statsmodels.stats.proportion import proportion_confint

import sys
sys.path.append('./src')

from dataset import PCDataset
from helpers import load_config
from measures import calculate_tracking_metrics, run_inference_and_clustering, plot_cluster_size_histogram, plot_layer_span_histogram, plot_layer_span_2d


def load_t5_data(root_file_pattern, max_events=None):
    """Load T5 data from ROOT file(s)."""

    files = glob(root_file_pattern) if '*' in root_file_pattern or '?' in root_file_pattern else [root_file_pattern]

    all_data = {
        't5_pt': [], 't5_eta': [], 'sim_pt': [], 'sim_eta': [], 'sim_q': [],
        'sim_vx': [], 'sim_vy': [], 'sim_vz': [], 'sim_t5IdxAllFrac': [],
        't5_isFake': [], 't5_isDup': []
    }

    total_loaded = 0

    for root_file in files:
        if max_events is not None and total_loaded >= max_events:
            break

        with uproot.open(root_file) as f:
            if "tree" not in f:
                continue
            tree = f["tree"]

            t5_pt = tree["t5_pt"].array()
            t5_eta = tree["t5_eta"].array()
            sim_pt = tree["sim_pt"].array()
            sim_eta = tree["sim_eta"].array()
            sim_q = tree["sim_q"].array()
            sim_vx = tree["sim_vx"].array()
            sim_vy = tree["sim_vy"].array()
            sim_vz = tree["sim_vz"].array()
            sim_t5IdxAllFrac = tree["sim_t5IdxAllFrac"].array()

            # Load arrays for calculating fake rate using MD-based matching
            t5_t3Idx0 = tree["t5_t3Idx0"].array()
            t5_t3Idx1 = tree["t5_t3Idx1"].array()
            t3_lsIdx0 = tree["t3_lsIdx0"].array()
            t3_lsIdx1 = tree["t3_lsIdx1"].array()
            ls_mdIdx0 = tree["ls_mdIdx0"].array()
            ls_mdIdx1 = tree["ls_mdIdx1"].array()
            md_simIdx = tree["md_simIdx"].array()
            t5_isDup = tree["t5_isDuplicate"].array()

            # Get all 8 MD sim indices for each T5
            md_simIdx0 = md_simIdx[ls_mdIdx0[t3_lsIdx0[t5_t3Idx0]]]
            md_simIdx1 = md_simIdx[ls_mdIdx1[t3_lsIdx0[t5_t3Idx0]]]
            md_simIdx2 = md_simIdx[ls_mdIdx0[t3_lsIdx1[t5_t3Idx0]]]
            md_simIdx3 = md_simIdx[ls_mdIdx1[t3_lsIdx1[t5_t3Idx0]]]
            md_simIdx4 = md_simIdx[ls_mdIdx0[t3_lsIdx0[t5_t3Idx1]]]
            md_simIdx5 = md_simIdx[ls_mdIdx1[t3_lsIdx0[t5_t3Idx1]]]
            md_simIdx6 = md_simIdx[ls_mdIdx0[t3_lsIdx1[t5_t3Idx1]]]
            md_simIdx7 = md_simIdx[ls_mdIdx1[t3_lsIdx1[t5_t3Idx1]]]

            n_events = len(md_simIdx0)
            if max_events is not None and total_loaded + n_events > max_events:
                n_events = max_events - total_loaded

            t5_isFake = []

            for evt in range(n_events):
                evt_md_simIdx0 = np.asarray(md_simIdx0[evt])
                evt_md_simIdx1 = np.asarray(md_simIdx1[evt])
                evt_md_simIdx2 = np.asarray(md_simIdx2[evt])
                evt_md_simIdx3 = np.asarray(md_simIdx3[evt])
                evt_md_simIdx4 = np.asarray(md_simIdx4[evt])
                evt_md_simIdx5 = np.asarray(md_simIdx5[evt])
                evt_md_simIdx6 = np.asarray(md_simIdx6[evt])
                evt_md_simIdx7 = np.asarray(md_simIdx7[evt])

                match_count = zip(evt_md_simIdx0, evt_md_simIdx1, evt_md_simIdx2, evt_md_simIdx3,
                                  evt_md_simIdx4, evt_md_simIdx5, evt_md_simIdx6, evt_md_simIdx7)

                matches = []
                for i, sims in enumerate(match_count):
                    sims = np.array(sims)
                    unique, counts = np.unique(sims, return_counts=True)
                    majority_sim = unique[np.argmax(counts)]
                    if majority_sim < 0:
                        matches.append(0)
                    else:
                        matches.append(np.sum(sims == majority_sim))

                matching_fraction = np.array(matches) / 8.0
                evt_t5_isFake = matching_fraction <= 0.75
                t5_isFake.append(evt_t5_isFake)

            t5_isFake = ak.Array(t5_isFake)

            all_data['t5_pt'].append(t5_pt[:n_events])
            all_data['t5_eta'].append(t5_eta[:n_events])
            all_data['sim_pt'].append(sim_pt[:n_events])
            all_data['sim_eta'].append(sim_eta[:n_events])
            all_data['sim_q'].append(sim_q[:n_events])
            all_data['sim_vx'].append(sim_vx[:n_events])
            all_data['sim_vy'].append(sim_vy[:n_events])
            all_data['sim_vz'].append(sim_vz[:n_events])
            all_data['sim_t5IdxAllFrac'].append(sim_t5IdxAllFrac[:n_events])
            all_data['t5_isFake'].append(t5_isFake)
            all_data['t5_isDup'].append(t5_isDup[:n_events])

            total_loaded += n_events

    return {k: ak.concatenate(v) if len(v) > 0 else ak.Array([]) for k, v in all_data.items()}


def calculate_t5_metrics(t5_data, pt_bins, eta_bins, eta_cut=None, purity_threshold=0.75):
    """Calculate efficiency, fake rate, and duplicate rate for T5 reconstruction."""
    n_bins_pt = len(pt_bins) - 1
    n_bins_eta = len(eta_bins) - 1

    hist_eff_num_pt = np.zeros(n_bins_pt)
    hist_eff_den_pt = np.zeros(n_bins_pt)
    hist_eff_num_eta = np.zeros(n_bins_eta)
    hist_eff_den_eta = np.zeros(n_bins_eta)

    hist_fake_num_pt = np.zeros(n_bins_pt)
    hist_fake_den_pt = np.zeros(n_bins_pt)
    hist_fake_num_eta = np.zeros(n_bins_eta)
    hist_fake_den_eta = np.zeros(n_bins_eta)

    hist_dup_num_pt = np.zeros(n_bins_pt)
    hist_dup_den_pt = np.zeros(n_bins_pt)
    hist_dup_num_eta = np.zeros(n_bins_eta)
    hist_dup_den_eta = np.zeros(n_bins_eta)

    n_events = len(t5_data['sim_pt'])

    for evt in range(n_events):
        sim_pt = np.asarray(t5_data['sim_pt'][evt])
        sim_eta = np.asarray(t5_data['sim_eta'][evt])
        sim_q = np.asarray(t5_data['sim_q'][evt])
        sim_vx = np.asarray(t5_data['sim_vx'][evt])
        sim_vy = np.asarray(t5_data['sim_vy'][evt])
        sim_vz = np.asarray(t5_data['sim_vz'][evt])
        sim_t5IdxAllFrac = t5_data['sim_t5IdxAllFrac'][evt]
        t5_isFake = np.asarray(t5_data['t5_isFake'][evt])
        t5_isDup = np.asarray(t5_data['t5_isDup'][evt])
        t5_pt = np.asarray(t5_data['t5_pt'][evt])
        t5_eta = np.asarray(t5_data['t5_eta'][evt])

        n_sim = len(sim_pt)
        n_t5 = len(t5_isFake)

        sim_fracs = ak.fill_none(ak.max(sim_t5IdxAllFrac, axis=1), 0)
        sim_fracs = np.asarray(sim_fracs)

        # Process sim particles for efficiency
        for sim_idx in range(n_sim):
            sim_pt_val = sim_pt[sim_idx]
            sim_eta_val = sim_eta[sim_idx]
            sim_q_val = sim_q[sim_idx]
            sim_vx_val = sim_vx[sim_idx]
            sim_vy_val = sim_vy[sim_idx]
            sim_vz_val = sim_vz[sim_idx]

            # Apply selection cuts
            if sim_q_val == 0:
                continue
            if sim_pt_val <= 0.8:
                continue
            if abs(sim_eta_val) >= 2.4:
                continue
            if abs(sim_vz_val) >= 30:
                continue
            rho = np.sqrt(sim_vx_val**2 + sim_vy_val**2)
            if rho >= 2.5:
                continue

            is_matched = sim_fracs[sim_idx] > purity_threshold

            in_acceptance_pt = True
            if eta_cut is not None:
                eta_min, eta_max = eta_cut
                in_acceptance_pt = (sim_eta_val >= eta_min) and (sim_eta_val <= eta_max)

            if in_acceptance_pt:
                pt_bin = np.digitize(sim_pt_val, pt_bins) - 1
                if 0 <= pt_bin < n_bins_pt:
                    hist_eff_den_pt[pt_bin] += 1
                    if is_matched:
                        hist_eff_num_pt[pt_bin] += 1

            eta_bin = np.digitize(sim_eta_val, eta_bins) - 1
            if 0 <= eta_bin < n_bins_eta:
                hist_eff_den_eta[eta_bin] += 1
                if is_matched:
                    hist_eff_num_eta[eta_bin] += 1

        # Process T5 tracks for fake rate and duplicate rate
        for t5_idx in range(n_t5):
            is_fake = t5_isFake[t5_idx]
            is_dup = bool(t5_isDup[t5_idx])
            track_pt = t5_pt[t5_idx]
            track_eta = t5_eta[t5_idx]

            # Apply same pT cut as transformer
            if track_pt <= 0.8 or track_pt >= 100:
                continue

            pt_bin = np.digitize(track_pt, pt_bins) - 1
            eta_bin = np.digitize(track_eta, eta_bins) - 1

            if 0 <= pt_bin < n_bins_pt:
                hist_fake_den_pt[pt_bin] += 1
                if is_fake:
                    hist_fake_num_pt[pt_bin] += 1
                hist_dup_den_pt[pt_bin] += 1
                if is_dup:
                    hist_dup_num_pt[pt_bin] += 2

            if 0 <= eta_bin < n_bins_eta:
                hist_fake_den_eta[eta_bin] += 1
                if is_fake:
                    hist_fake_num_eta[eta_bin] += 1
                hist_dup_den_eta[eta_bin] += 1
                if is_dup:
                    hist_dup_num_eta[eta_bin] += 2

    def safe_divide(num, den):
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)

    def clopper_pearson_error(k, N):
        valid = N > 0
        k = np.asarray(k)
        N = np.asarray(N)
        _k = np.where(valid, k, 0)
        _N = np.where(valid, N, 1)
        eff = np.where(valid, _k / _N, 0)
        low, high = proportion_confint(_k, _N, method="beta")
        low = np.where(valid, low, 0)
        high = np.where(valid, high, 0)
        err_low = eff - low
        err_high = high - eff
        return [err_low, err_high]

    eff_pt = safe_divide(hist_eff_num_pt, hist_eff_den_pt)
    eff_eta = safe_divide(hist_eff_num_eta, hist_eff_den_eta)
    fake_pt = safe_divide(hist_fake_num_pt, hist_fake_den_pt)
    fake_eta = safe_divide(hist_fake_num_eta, hist_fake_den_eta)
    dup_pt = safe_divide(hist_dup_num_pt, hist_dup_den_pt)
    dup_eta = safe_divide(hist_dup_num_eta, hist_dup_den_eta)

    eff_err_pt = clopper_pearson_error(hist_eff_num_pt, hist_eff_den_pt)
    eff_err_eta = clopper_pearson_error(hist_eff_num_eta, hist_eff_den_eta)
    fake_err_pt = clopper_pearson_error(hist_fake_num_pt, hist_fake_den_pt)
    fake_err_eta = clopper_pearson_error(hist_fake_num_eta, hist_fake_den_eta)
    dup_err_pt = clopper_pearson_error(hist_dup_num_pt, hist_dup_den_pt)
    dup_err_eta = clopper_pearson_error(hist_dup_num_eta, hist_dup_den_eta)

    return {
        'efficiency_pt': eff_pt,
        'efficiency_eta': eff_eta,
        'fake_rate_pt': fake_pt,
        'fake_rate_eta': fake_eta,
        'duplicate_rate_pt': dup_pt,
        'duplicate_rate_eta': dup_eta,
        'efficiency_err_pt': eff_err_pt,
        'efficiency_err_eta': eff_err_eta,
        'fake_rate_err_pt': fake_err_pt,
        'fake_rate_err_eta': fake_err_eta,
        'duplicate_rate_err_pt': dup_err_pt,
        'duplicate_rate_err_eta': dup_err_eta,
    }


def plot_efficiency_fake_rate(pt_bins, eta_bins, metrics_pt, metrics_eta, output_path, epsilon):
    """Plot efficiency and fake rate."""
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2

    plots = [
        ('efficiency', 'pt', pt_centers, metrics_pt['efficiency'], metrics_pt['efficiency_err'], 'Simulated track $p_T$ [GeV]', 'Efficiency', 'blue', (0, 1.1)),
        ('fake_rate', 'pt', pt_centers, metrics_pt['fake_rate'], metrics_pt['fake_rate_err'], 'Reconstructed track $p_T$ [GeV]', 'Fake Rate', 'red', None),
        ('efficiency', 'eta', eta_centers, metrics_eta['efficiency'], metrics_eta['efficiency_err'], 'Simulated track $\eta$', 'Efficiency', 'blue', (0, 1.1)),
        ('fake_rate', 'eta', eta_centers, metrics_eta['fake_rate'], metrics_eta['fake_rate_err'], 'Reconstructed track $\eta$', 'Fake Rate', 'red', None),
    ]

    for metric_name, var_name, centers, y_data, y_err, xlabel, ylabel, color, ylim in plots:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.errorbar(centers, y_data, yerr=y_err, fmt='o', color=color, label='Transformer OC', capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)

        hep.cms.label(llabel="Phase-2 Simulation Preliminary", data=False, rlabel=r"$t\bar{t}$ PU200, 14 TeV", ax=ax)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}/{metric_name}_{var_name}_eps_{epsilon:.3f}.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_purity_duplicate(pt_bins, eta_bins, metrics_pt, metrics_eta, output_path, epsilon):
    """Plot purity and duplicate rate."""
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2

    plots = [
        ('purity', 'pt', pt_centers, metrics_pt['purity'], metrics_pt['purity_err'], 'Simulated track $p_T$ [GeV]', 'Purity', 'green', (0.95, 1.0)),
        ('duplicate_rate', 'pt', pt_centers, metrics_pt['duplicate_rate'], metrics_pt['duplicate_rate_err'], 'Simulated track $p_T$ [GeV]', 'Duplicate Rate', 'purple', (0, 0.2)),
        ('purity', 'eta', eta_centers, metrics_eta['purity'], metrics_eta['purity_err'], 'Simulated track $\eta$', 'Purity', 'green', (0.95, 1.0)),
        ('duplicate_rate', 'eta', eta_centers, metrics_eta['duplicate_rate'], metrics_eta['duplicate_rate_err'], 'Simulated track $\eta$', 'Duplicate Rate', 'purple', (0, 0.2)),
    ]

    for metric_name, var_name, centers, y_data, y_err, xlabel, ylabel, color, ylim in plots:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.errorbar(centers, y_data, yerr=y_err, fmt='o', color=color, label='Transformer OC', capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)

        hep.cms.label(llabel="Phase-2 Simulation Preliminary", data=False, rlabel=r"$t\bar{t}$ PU200, 14 TeV", ax=ax)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}/{metric_name}_{var_name}_eps_{epsilon:.3f}.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_t5_comparison(pt_bins, eta_bins, model_metrics_pt, model_metrics_eta, t5_metrics, output_path, epsilon):
    """Plot comparison of transformer and T5 efficiency, fake rate, and duplicate rate."""
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2

    plots = [
        ('efficiency', 'pt', pt_centers, model_metrics_pt['efficiency'], model_metrics_pt['efficiency_err'],
         t5_metrics['efficiency_pt'], t5_metrics['efficiency_err_pt'], 'Simulated track $p_T$ [GeV]', 'Efficiency', (0, 1.1)),
        ('fake_rate', 'pt', pt_centers, model_metrics_pt['fake_rate'], model_metrics_pt['fake_rate_err'],
         t5_metrics['fake_rate_pt'], t5_metrics['fake_rate_err_pt'], 'Reconstructed track $p_T$ [GeV]', 'Fake Rate', None),
        ('duplicate_rate', 'pt', pt_centers, model_metrics_pt['duplicate_rate'], model_metrics_pt['duplicate_rate_err'],
         t5_metrics['duplicate_rate_pt'], t5_metrics['duplicate_rate_err_pt'], 'Simulated track $p_T$ [GeV]', 'Duplicate Rate', (0, 0.2)),
        ('efficiency', 'eta', eta_centers, model_metrics_eta['efficiency'], model_metrics_eta['efficiency_err'],
         t5_metrics['efficiency_eta'], t5_metrics['efficiency_err_eta'], 'Simulated track $\eta$', 'Efficiency', (0, 1.1)),
        ('fake_rate', 'eta', eta_centers, model_metrics_eta['fake_rate'], model_metrics_eta['fake_rate_err'],
         t5_metrics['fake_rate_eta'], t5_metrics['fake_rate_err_eta'], 'Reconstructed track $\eta$', 'Fake Rate', None),
        ('duplicate_rate', 'eta', eta_centers, model_metrics_eta['duplicate_rate'], model_metrics_eta['duplicate_rate_err'],
         t5_metrics['duplicate_rate_eta'], t5_metrics['duplicate_rate_err_eta'], 'Simulated track $\eta$', 'Duplicate Rate', (0, 0.2)),
    ]

    for metric_name, var_name, centers, m_y, m_err, t_y, t_err, xlabel, ylabel, ylim in plots:
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.errorbar(centers, m_y, yerr=m_err, fmt='o', color='blue', label='Transformer OC', capsize=3, markersize=5)
        ax.errorbar(centers, t_y, yerr=t_err, fmt='s', color='red', label='LST T5s', capsize=3, markersize=5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)

        hep.cms.label(llabel="Phase-2 Simulation Preliminary", data=False, rlabel=r"$t\bar{t}$ PU200, 14 TeV", ax=ax)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_path}/comparison_{metric_name}_{var_name}_eps_{epsilon:.3f}.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    argparser = ArgumentParser()
    argparser.add_argument('--config', type=str, default='./config.yaml', help='Path to the config file (YAML)')
    argparser.add_argument('--output', type=str, help='Path to the trained model file')
    argparser.add_argument('--epsilon', type=float, default=None, help='DBSCAN epsilon parameter (overrides config)')
    argparser.add_argument('--n-events', type=int, default=-1, help='Number of events to process (-1 for all)')
    argparser.add_argument('--t5', action='store_true', help='Include T5 comparison plots')
    argparser.add_argument('--t5-file', type=str, default="/home/users/aaarora/phys/tracking/lst/lstod/CMSSW_15_1_0_pre2/src/RecoTracker/LSTCore/standalone/output_pu200_all/*.root", help='Path or wildcard to ROOT file(s) with T5 data')
    args = argparser.parse_args()

    config = load_config(args.config)

    subset = args.n_events if args.n_events > 0 else None
    dataset = PCDataset(
        config['val_data_dir'],
        subset=subset,
    )
    data_loader = DataLoader(dataset, shuffle=False, drop_last=False, num_workers=2)

    print(f"Loaded dataset with {len(dataset)} graphs")

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
            dropout=config.get('dropout', 0.1),
            dr_threshold=config.get('dr_threshold', 0.2),
            attention_chunk_size=config.get('attention_chunk_size', 1024)
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

    checkpoint = torch.load(output, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if any(key.startswith('model.') for key in state_dict.keys()):
        state_dict = {key.replace('model.', '', 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)

    model.eval()

    min_samples = config.get('dbscan_min_samples', 2)
    purity_threshold = config.get('purity_threshold', 0.75)
    beta_threshold = config.get('beta_threshold', None)

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

    all_cluster_sizes = []
    all_cluster_data = []
    all_cluster_layer_spans = []

    print(f"Processing {len(data_loader)} events with epsilon = {args.epsilon if args.epsilon else config.get('dbscan_eps', 0.05)}")
    if beta_threshold is not None:
        print(f"Using beta threshold = {beta_threshold} (rejecting hits with beta < {beta_threshold})")

    epsilon = args.epsilon if args.epsilon is not None else config.get('dbscan_eps', 0.05)
    eta_cut = config.get('eta_cut', (-2.5, 2.5))

    timings = []

    for i, data in enumerate(data_loader):
        if i % 10 == 0:
            print(f"Processing event {i+1}/{len(data_loader)}")

        start_time = time.time()
        cluster, beta = run_inference_and_clustering(data, model, device, eps=epsilon, min_samples=min_samples, use_fp16=use_fp16, beta_threshold=beta_threshold)
        end_time = time.time()
        timings.append(end_time - start_time)

        cluster_labels = cluster.labels_
        metrics = calculate_tracking_metrics(data, cluster_labels, pt_bins, eta_bins, purity_threshold=purity_threshold, eta_cut=eta_cut, beta=beta)

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

        all_cluster_sizes.extend(metrics['cluster_sizes'])
        all_cluster_data.extend(metrics['cluster_data'])
        all_cluster_layer_spans.extend(metrics['cluster_layer_spans'])

    print(f"Average inference and clustering time per event: {np.mean(timings):.4f} seconds")

    def clopper_pearson_error(k, N):
        valid = N > 0
        k = np.asarray(k)
        N = np.asarray(N)
        _k = np.where(valid, k, 0)
        _N = np.where(valid, N, 1)
        eff = np.where(valid, _k / _N, 0)
        low, high = proportion_confint(_k, _N, method="beta")
        low = np.where(valid, low, 0)
        high = np.where(valid, high, 0)
        err_low = eff - low
        err_high = high - eff
        return [err_low, err_high]

    eff_pt = np.divide(hist_eff_num_pt, hist_eff_den_pt, out=np.zeros_like(hist_eff_num_pt), where=hist_eff_den_pt!=0)
    eff_eta = np.divide(hist_eff_num_eta, hist_eff_den_eta, out=np.zeros_like(hist_eff_num_eta), where=hist_eff_den_eta!=0)

    fake_pt = np.divide(hist_fake_num_pt, hist_fake_den_pt, out=np.zeros_like(hist_fake_num_pt), where=hist_fake_den_pt!=0)
    fake_eta = np.divide(hist_fake_num_eta, hist_fake_den_eta, out=np.zeros_like(hist_fake_num_eta), where=hist_fake_den_eta!=0)

    purity_pt = np.divide(hist_purity_sum_pt, hist_purity_count_pt, out=np.zeros_like(hist_purity_sum_pt), where=hist_purity_count_pt!=0)
    purity_eta = np.divide(hist_purity_sum_eta, hist_purity_count_eta, out=np.zeros_like(hist_purity_sum_eta), where=hist_purity_count_eta!=0)

    dup_pt = np.divide(hist_dup_num_pt, hist_eff_den_pt, out=np.zeros_like(hist_dup_num_pt), where=hist_eff_den_pt!=0)
    dup_eta = np.divide(hist_dup_num_eta, hist_eff_den_eta, out=np.zeros_like(hist_dup_num_eta), where=hist_eff_den_eta!=0)

    eff_err_pt = clopper_pearson_error(hist_eff_num_pt, hist_eff_den_pt)
    eff_err_eta = clopper_pearson_error(hist_eff_num_eta, hist_eff_den_eta)
    fake_err_pt = clopper_pearson_error(hist_fake_num_pt, hist_fake_den_pt)
    fake_err_eta = clopper_pearson_error(hist_fake_num_eta, hist_fake_den_eta)
    dup_err_pt = clopper_pearson_error(hist_dup_num_pt, hist_eff_den_pt)
    dup_err_eta = clopper_pearson_error(hist_dup_num_eta, hist_eff_den_eta)

    purity_err_pt = np.sqrt(purity_pt * (1 - purity_pt) / hist_purity_count_pt)
    purity_err_pt = np.where(hist_purity_count_pt > 0, purity_err_pt, 0)
    # Reformat symmetric error as asymmetric to match other errors in the plots
    purity_err_pt = [purity_err_pt, purity_err_pt]

    purity_err_eta = np.sqrt(purity_eta * (1 - purity_eta) / hist_purity_count_eta)
    purity_err_eta = np.where(hist_purity_count_eta > 0, purity_err_eta, 0)
    purity_err_eta = [purity_err_eta, purity_err_eta]

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

    # Plot efficiency and fake rate
    plot_efficiency_fake_rate(pt_bins, eta_bins, metrics_pt, metrics_eta, args.output, epsilon)

    # Plot purity and duplicate rate
    plot_purity_duplicate(pt_bins, eta_bins, metrics_pt, metrics_eta, args.output, epsilon)

    # Load and plot T5 comparison if requested
    if args.t5:
        n_model_events = len(data_loader)
        print(f"\nLoading T5 data from: {args.t5_file} (max {n_model_events} events)")
        t5_data = load_t5_data(args.t5_file, max_events=n_model_events)

        # Limit T5 events to match model events
        if len(t5_data['t5_pt']) > n_model_events:
            print(f"Limiting T5 data to {n_model_events} events")
            for key in t5_data:
                t5_data[key] = t5_data[key][:n_model_events]

        print("\nCalculating T5 metrics...")
        t5_metrics = calculate_t5_metrics(t5_data, pt_bins, eta_bins, eta_cut=eta_cut, purity_threshold=purity_threshold)

        # Plot comparison
        plot_t5_comparison(pt_bins, eta_bins, metrics_pt, metrics_eta, t5_metrics, args.output, epsilon)

    if all_cluster_sizes:
        plot_cluster_size_histogram(all_cluster_sizes, pt_bins, eta_bins, args.output, epsilon, cluster_data=all_cluster_data)
        print(f"\nCluster size statistics:")
        print(f"  Total clusters: {len(all_cluster_sizes)}")
        print(f"  Mean cluster size: {np.mean(all_cluster_sizes):.2f}")
        print(f"  Median cluster size: {np.median(all_cluster_sizes):.0f}")
        print(f"  Min/Max cluster size: {min(all_cluster_sizes)}/{max(all_cluster_sizes)}")

    if all_cluster_layer_spans:
        plot_layer_span_histogram(all_cluster_layer_spans, args.output, epsilon)
        plot_layer_span_2d(all_cluster_data, all_cluster_layer_spans, pt_bins, eta_bins, args.output, epsilon)
        print(f"\nLayer span statistics:")
        print(f"  Mean layer span: {np.mean(all_cluster_layer_spans):.2f}")
        print(f"  Median layer span: {np.median(all_cluster_layer_spans):.0f}")
        print(f"  Min/Max layer span: {min(all_cluster_layer_spans)}/{max(all_cluster_layer_spans)}")

    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Epsilon: {epsilon}")
    print(f"Purity Threshold: {purity_threshold}")
    if beta_threshold is not None:
        print(f"Beta Threshold: {beta_threshold}")
    print(f"Events processed: {len(data_loader)}")
    print(f"Output directory: {args.output}")
    print("\nTransformer OC Performance (mean over non-zero bins):")
    print(f"  Efficiency: {np.mean(eff_pt[eff_pt > 0]):.3f}")
    print(f"  Fake Rate: {np.mean(fake_pt[fake_pt > 0]):.3f}")
    print(f"  Purity: {np.mean(purity_pt[purity_pt > 0]):.3f}")

    if args.t5:
        print(f"\nT5 Performance (mean over non-zero bins):")
        print(f"  Efficiency: {np.mean(t5_metrics['efficiency_pt'][t5_metrics['efficiency_pt'] > 0]):.3f}")
        print(f"  Fake Rate: {np.mean(t5_metrics['fake_rate_pt'][t5_metrics['fake_rate_pt'] > 0]):.3f}")

    print(f"\nPlots saved to: {args.output}/")

if __name__ == "__main__":
    main()
