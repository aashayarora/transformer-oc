import sys

sys.path.append("/home/users/aaarora/phys/tracking/transformers/oc")
sys.path.append("/home/users/aaarora/phys/tracking/transformers/oc/training")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from sklearn.cluster import DBSCAN

hep.style.use(hep.style.CMS)

from profiler import InferenceProfiler, print_gpu_info, profile_memory


@torch.no_grad()
def run_inference_and_clustering(
    data,
    model,
    device,
    eps=0.2,
    min_samples=2,
    use_fp16=False,
    beta_threshold=None,
    profiler=None,
    enable_profiling=False,
):
    # Create profiler if requested and not provided
    if profiler is None and enable_profiling:
        profiler = InferenceProfiler(device=device if device != "cpu" else "cuda:0")

    # Data transfer
    if profiler:
        with profiler.profile_section("data_transfer"):
            x = data.x.to(device)
            batch = data.batch.to(device)
    else:
        x = data.x.to(device)
        batch = data.batch.to(device)

    model.eval()

    # Model inference
    if profiler:
        with profiler.profile_section("model_forward"):
            if use_fp16 and device != "cpu":
                with torch.amp.autocast("cuda"):
                    out = model(x, batch)
                X = out["H"].float()
                beta = out["B"].float()
            else:
                out = model(x, batch)
                X = out["H"]
                beta = out["B"]
    else:
        if use_fp16 and device != "cpu":
            with torch.amp.autocast("cuda"):
                out = model(x, batch)
            X = out["H"].float()
            beta = out["B"].float()
        else:
            out = model(x, batch)
            X = out["H"]
            beta = out["B"]

    # Transfer to CPU
    if profiler:
        with profiler.profile_section("gpu_to_cpu_transfer"):
            X_cpu = X.cpu().numpy()
            beta_cpu = beta.cpu().numpy()
    else:
        X_cpu = X.cpu().detach().numpy()
        beta_cpu = beta.cpu().detach().numpy()

    # Clustering
    if profiler:
        with profiler.profile_section("clustering"):
            if beta_threshold is not None:
                beta_mask = beta_cpu.flatten() > beta_threshold
                X_filtered = X_cpu[beta_mask]

                cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(X_filtered)

                full_labels = np.full(len(X_cpu), -1, dtype=int)
                full_labels[beta_mask] = cluster.labels_

                class ClusterResult:
                    def __init__(self, labels):
                        self.labels_ = labels

                cluster_result = ClusterResult(full_labels)
            else:
                cluster_result = DBSCAN(eps=eps, min_samples=min_samples).fit(X_cpu)
    else:
        if beta_threshold is not None:
            beta_mask = beta_cpu.flatten() > beta_threshold
            X_filtered = X_cpu[beta_mask]

            cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(X_filtered)

            full_labels = np.full(len(X_cpu), -1, dtype=int)
            full_labels[beta_mask] = cluster.labels_

            class ClusterResult:
                def __init__(self, labels):
                    self.labels_ = labels

            cluster_result = ClusterResult(full_labels)
        else:
            cluster_result = DBSCAN(eps=eps, min_samples=min_samples).fit(X_cpu)

    if profiler:
        return cluster_result, beta_cpu, profiler
    else:
        return cluster_result, beta_cpu


def calculate_tracking_metrics(
    data,
    cluster_labels,
    pt_bins,
    eta_bins,
    purity_threshold=0.75,
    eta_cut=None,
    beta=None,
    beta_stats=True,
    vtxperp_bins=None,
):
    sim_features = data.sim_features
    t3_truth_labels = data.sim_index.cpu().detach().numpy().flatten()
    t3_truth_labels[t3_truth_labels < 0] = -1

    x_features = data.x.cpu().detach().numpy()

    md_layers = data.md_layer.cpu().detach().numpy()
    md_simIdx = data.md_simIdx.cpu().detach().numpy()

    if x_features.shape[1] == 26:
        min_pt, max_pt = -1.02, 7.0
        min_eta, max_eta = -2.62, 2.62
    else:
        min_pt, max_pt = -3.4627, 7.5964
        min_eta, max_eta = -2.4507, 2.4535

    real_track_pts = []
    real_track_etas = []
    real_track_purities = []

    fake_track_pts = []
    fake_track_etas = []

    cluster_sizes = []
    cluster_data = []
    cluster_layer_spans = []

    correctly_reconstructed_sim = set()
    sim_to_pred_clusters = {}

    unique_pred_labels = np.unique(cluster_labels[cluster_labels >= 0])

    for pred_label in unique_pred_labels:
        mask = cluster_labels == pred_label
        cluster_indices = np.where(mask)[0]

        # Always use MD simIdx for purity calculation
        cluster_md_simIdx = md_simIdx[cluster_indices].flatten()
        truth_in_cluster = cluster_md_simIdx
        truth_no_fake = truth_in_cluster[truth_in_cluster >= 0]

        cluster_size = int(np.sum(mask))
        cluster_sizes.append(cluster_size)
        segment_pts_norm = x_features[cluster_indices, 0]
        segment_etas_norm = x_features[cluster_indices, 1]

        segment_pts_log = segment_pts_norm * (max_pt - min_pt) + min_pt
        segment_pts = np.exp(segment_pts_log) - 1e-6
        segment_etas = segment_etas_norm * (max_eta - min_eta) + min_eta

        t3_md0_x = x_features[cluster_indices, 13]
        t3_md0_y = x_features[cluster_indices, 14]
        t3_md1_x = x_features[cluster_indices, 22]
        t3_md1_y = x_features[cluster_indices, 23]
        t3_md2_x = x_features[cluster_indices, 31]
        t3_md2_y = x_features[cluster_indices, 32]
        t3_md3_x = x_features[cluster_indices, 40]
        t3_md3_y = x_features[cluster_indices, 41]

        t3_md0_r = np.sqrt(t3_md0_x**2 + t3_md0_y**2)
        t3_md1_r = np.sqrt(t3_md1_x**2 + t3_md1_y**2)
        t3_md2_r = np.sqrt(t3_md2_x**2 + t3_md2_y**2)
        t3_md3_r = np.sqrt(t3_md3_x**2 + t3_md3_y**2)

        t3_lowest_r = np.minimum(
            np.minimum(t3_md0_r, t3_md1_r), np.minimum(t3_md2_r, t3_md3_r)
        )
        lowest_r_idx = np.argmin(t3_lowest_r)
        reco_track_pt = float(segment_pts[lowest_r_idx])
        reco_track_eta = float(segment_etas[lowest_r_idx])

        # Always check layer span and filter out layer span == 3
        cluster_md_layers = md_layers[cluster_indices]
        all_layers = set(cluster_md_layers.flatten())
        layer_span = len(all_layers)
        if layer_span < 4:
            continue

        if reco_track_pt != -999.0 and reco_track_eta != -999.0:
            cluster_data.append((reco_track_pt, reco_track_eta, cluster_size))

        if len(truth_no_fake) == 0:
            is_fake = True
            purity = 0.0
        else:
            unique_labels, counts = np.unique(truth_no_fake, return_counts=True)
            majority_label = int(unique_labels[np.argmax(counts)])
            purity = np.sum(truth_in_cluster == majority_label) / len(truth_in_cluster)
            is_fake = purity < purity_threshold

            if not is_fake:
                cluster_layer_spans.append(layer_span)
                correctly_reconstructed_sim.add(majority_label)
                sim_to_pred_clusters.setdefault(majority_label, []).append(pred_label)

        if is_fake:
            if reco_track_pt > 0.8 and reco_track_pt < 100:
                fake_track_pts.append(reco_track_pt)
                fake_track_etas.append(reco_track_eta)
        else:
            if reco_track_pt > 0.8 and reco_track_pt < 100:
                real_track_pts.append(reco_track_pt)
                real_track_etas.append(reco_track_eta)
                real_track_purities.append(purity)

    # For efficiency calculation, we use t3_truth_labels to match with sim_features
    unique_truth_labels = np.unique(t3_truth_labels[t3_truth_labels >= 0])
    valid_sim_mask = unique_truth_labels < len(sim_features)
    valid_sim_labels = unique_truth_labels[valid_sim_mask].astype(int)

    sim_pts = np.array([float(sim_features[i][0]) for i in valid_sim_labels])
    sim_etas = np.array([float(sim_features[i][1]) for i in valid_sim_labels])
    sim_q = np.array([float(sim_features[i][3]) for i in valid_sim_labels])
    sim_vx = np.array([float(sim_features[i][4]) for i in valid_sim_labels])
    sim_vy = np.array([float(sim_features[i][5]) for i in valid_sim_labels])
    sim_vz = np.array([float(sim_features[i][6]) for i in valid_sim_labels])

    selection_mask = (
        (sim_q != 0)
        & (sim_pts > 0.8)
        & (np.abs(sim_etas) < 2.4)
        & (np.abs(sim_vz) < 30)
        & (np.sqrt(sim_vx**2 + sim_vy**2) < 2.5)
    )

    valid_sim_labels = valid_sim_labels[selection_mask]
    sim_pts = sim_pts[selection_mask]
    sim_etas = sim_etas[selection_mask]

    sim_reconstructed = np.array(
        [i in correctly_reconstructed_sim for i in valid_sim_labels]
    )
    sim_duplicated = np.array(
        [len(sim_to_pred_clusters.get(i, [])) > 1 for i in valid_sim_labels]
    )

    if eta_cut is not None:
        eta_min, eta_max = eta_cut
        in_acceptance = (sim_etas >= eta_min) & (sim_etas <= eta_max)
    else:
        in_acceptance = np.ones(len(sim_etas), dtype=bool)

    # Efficiency: reconstructed / all truth particles
    hist_eff_num_pt, _ = np.histogram(
        sim_pts[in_acceptance][sim_reconstructed[in_acceptance]], bins=pt_bins
    )
    hist_eff_den_pt, _ = np.histogram(sim_pts[in_acceptance], bins=pt_bins)
    hist_eff_num_eta, _ = np.histogram(sim_etas[sim_reconstructed], bins=eta_bins)
    hist_eff_den_eta, _ = np.histogram(sim_etas, bins=eta_bins)

    # Duplicate rate: duplicated / all truth particles
    hist_dup_num_pt, _ = np.histogram(
        sim_pts[in_acceptance][sim_duplicated[in_acceptance]], bins=pt_bins
    )
    hist_dup_num_eta, _ = np.histogram(sim_etas[sim_duplicated], bins=eta_bins)

    # Fake rate: fake tracks (by reco pT/eta) / all tracks
    real_track_pts = (
        np.array(real_track_pts) if len(real_track_pts) > 0 else np.array([])
    )
    real_track_etas = (
        np.array(real_track_etas) if len(real_track_etas) > 0 else np.array([])
    )
    fake_track_pts = (
        np.array(fake_track_pts) if len(fake_track_pts) > 0 else np.array([])
    )
    fake_track_etas = (
        np.array(fake_track_etas) if len(fake_track_etas) > 0 else np.array([])
    )

    # Bin by reconstructed kinematics (now properly inverted from preprocessing)
    hist_real_pt, _ = np.histogram(real_track_pts, bins=pt_bins)
    hist_real_eta, _ = np.histogram(real_track_etas, bins=eta_bins)
    hist_fake_pt, _ = np.histogram(fake_track_pts, bins=pt_bins)
    hist_fake_eta, _ = np.histogram(fake_track_etas, bins=eta_bins)

    hist_tracks_pt = hist_real_pt + hist_fake_pt
    hist_tracks_eta = hist_real_eta + hist_fake_eta

    # Purity: compute weighted averages per bin using np.histogram with weights
    real_track_pts = (
        np.array(real_track_pts) if len(real_track_pts) > 0 else np.array([])
    )
    real_track_purities = (
        np.array(real_track_purities) if len(real_track_purities) > 0 else np.array([])
    )

    if len(real_track_pts) > 0:
        hist_purity_sum_pt, _ = np.histogram(
            real_track_pts, bins=pt_bins, weights=real_track_purities
        )
        hist_purity_count_pt, _ = np.histogram(real_track_pts, bins=pt_bins)
        hist_purity_sum_eta, _ = np.histogram(
            real_track_etas, bins=eta_bins, weights=real_track_purities
        )
        hist_purity_count_eta, _ = np.histogram(real_track_etas, bins=eta_bins)
    else:
        hist_purity_sum_pt = np.zeros(len(pt_bins) - 1)
        hist_purity_count_pt = np.zeros(len(pt_bins) - 1)
        hist_purity_sum_eta = np.zeros(len(eta_bins) - 1)
        hist_purity_count_eta = np.zeros(len(eta_bins) - 1)

    return {
        "efficiency_numerator_pt": hist_eff_num_pt,
        "efficiency_numerator_eta": hist_eff_num_eta,
        "efficiency_denominator_pt": hist_eff_den_pt,
        "efficiency_denominator_eta": hist_eff_den_eta,
        "fake_rate_numerator_pt": hist_fake_pt,
        "fake_rate_numerator_eta": hist_fake_eta,
        "purity_sum_pt": hist_purity_sum_pt,
        "purity_sum_eta": hist_purity_sum_eta,
        "total_tracks_pt": hist_tracks_pt,
        "total_tracks_eta": hist_tracks_eta,
        "purity_count_pt": hist_purity_count_pt,
        "purity_count_eta": hist_purity_count_eta,
        "duplicate_rate_numerator_pt": hist_dup_num_pt,
        "duplicate_rate_numerator_eta": hist_dup_num_eta,
        "cluster_sizes": cluster_sizes,
        "cluster_data": cluster_data,
        "cluster_layer_spans": cluster_layer_spans,
    }


def plot_performance_histograms(
    pt_bins, eta_bins, metrics_pt, metrics_eta, output_path, epsilon
):
    pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2

    plots = [
        (
            "efficiency",
            "pt",
            pt_centers,
            metrics_pt["efficiency"],
            metrics_pt["efficiency_err"],
            "Simulated track $p_T$ [GeV]",
            "Efficiency",
            "blue",
            (0, 1.1),
        ),
        (
            "fake_rate",
            "pt",
            pt_centers,
            metrics_pt["fake_rate"],
            metrics_pt["fake_rate_err"],
            "Reconstructed track $p_T$ [GeV]",
            "Fake Rate",
            "red",
            None,
        ),
        (
            "purity",
            "pt",
            pt_centers,
            metrics_pt["purity"],
            metrics_pt["purity_err"],
            "Simulated track $p_T$ [GeV]",
            "Purity",
            "green",
            (0.95, 1.0),
        ),
        (
            "duplicate_rate",
            "pt",
            pt_centers,
            metrics_pt["duplicate_rate"],
            metrics_pt["duplicate_rate_err"],
            "Simulated track $p_T$ [GeV]",
            "Duplicate Rate",
            "purple",
            (0, max(0.5, np.max(metrics_pt["duplicate_rate"]) * 1.2)),
        ),
        (
            "efficiency",
            "eta",
            eta_centers,
            metrics_eta["efficiency"],
            metrics_eta["efficiency_err"],
            "Simulated track $η$",
            "Efficiency",
            "blue",
            (0, 1.1),
        ),
        (
            "fake_rate",
            "eta",
            eta_centers,
            metrics_eta["fake_rate"],
            metrics_eta["fake_rate_err"],
            "Reconstructed track $η$",
            "Fake Rate",
            "red",
            None,
        ),
        (
            "purity",
            "eta",
            eta_centers,
            metrics_eta["purity"],
            metrics_eta["purity_err"],
            "Simulated track $η$",
            "Purity",
            "green",
            (0.95, 1.0),
        ),
        (
            "duplicate_rate",
            "eta",
            eta_centers,
            metrics_eta["duplicate_rate"],
            metrics_eta["duplicate_rate_err"],
            "Simulated track $η$",
            "Duplicate Rate",
            "purple",
            (0, max(0.5, np.max(metrics_eta["duplicate_rate"]) * 1.2)),
        ),
    ]

    for (
        metric_name,
        var_name,
        centers,
        y_data,
        y_err,
        xlabel,
        ylabel,
        color,
        ylim,
    ) in plots:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.errorbar(
            centers,
            y_data,
            yerr=y_err,
            fmt="o",
            color=color,
            label="Transformer OC",
            capsize=3,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)

        hep.cms.label(
            llabel="Phase-2 Simulation Preliminary",
            data=False,
            rlabel=r"$t\bar{t}$ PU200, 14 TeV",
            ax=ax,
        )
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/{metric_name}_{var_name}_eps_{epsilon:.3f}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_cluster_size_histogram(
    cluster_sizes, pt_bins, eta_bins, output_path, epsilon, cluster_data=None
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    cluster_sizes = np.array(cluster_sizes)
    bins = np.arange(0, min(30, cluster_sizes.max() + 2), 1) - 0.5
    ax.hist(
        cluster_sizes,
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
        label="Transformer OC",
    )

    ax.set_xlabel("Number of T3s per Cluster")
    ax.set_ylabel("Number of Clusters")
    ax.grid(True, alpha=0.3)
    hep.cms.label(
        llabel="Phase-2 Simulation Preliminary",
        data=False,
        text="$t\bar{t}$ PU200",
        com=14,
        ax=ax,
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        f"{output_path}/cluster_size_overall_eps_{epsilon:.3f}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    if cluster_data is not None and len(cluster_data) > 0:
        cluster_data_array = np.array(cluster_data)
        pts = cluster_data_array[:, 0]
        etas = cluster_data_array[:, 1]
        sizes = cluster_data_array[:, 2]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        size_sum, xedges, yedges = np.histogram2d(
            pts, etas, bins=[pt_bins, eta_bins], weights=sizes
        )
        size_count, _, _ = np.histogram2d(pts, etas, bins=[pt_bins, eta_bins])

        mean_size = np.divide(
            size_sum, size_count, out=np.zeros_like(size_sum), where=size_count > 0
        )

        im = ax.pcolormesh(eta_bins, pt_bins, mean_size, cmap="viridis", shading="auto")
        cbar = plt.colorbar(im, ax=ax)

        ax.set_xlabel("Simulated track $η$")
        ax.set_ylabel("Simulated track $p_T$ [GeV]")
        ax.grid(True, alpha=0.3, color="white", linewidth=0.5)
        hep.cms.label(
            llabel="Phase-2 Simulation Preliminary",
            data=False,
            rlabel=r"$t\bar{t}$ PU200, 14 TeV",
            ax=ax,
        )

        plt.tight_layout()
        plt.savefig(
            f"{output_path}/cluster_size_2d_eps_{epsilon:.3f}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_layer_span_histogram(cluster_layer_spans, output_path, epsilon):
    if cluster_layer_spans is None or len(cluster_layer_spans) == 0:
        print("No layer span data available to plot.")
        return

    fig, ax = plt.subplots()

    cluster_layer_spans = np.array(cluster_layer_spans)
    bins = np.arange(cluster_layer_spans.min() - 1, cluster_layer_spans.max() + 2, 1) - 0.5
    ax.hist(
        cluster_layer_spans,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.7,
        color="coral",
        label="Transformer OC",
    )

    ax.set_xlabel("# Unique Layers per Cluster")
    ax.set_ylabel("Density")
    ax.set_ylim(0, 0.7)
    ax.grid(True, alpha=0.3)
    hep.cms.text("Phase-2 Simulation Preliminary", ax=ax)
    ax.text(
        0.98,
        0.98,
        r"$t\bar{t}$ PU200, 14 TeV",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        f"{output_path}/cluster_layer_span_eps_{epsilon:.3f}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_layer_span_2d(
    cluster_data, cluster_layer_spans, pt_bins, eta_bins, output_path, epsilon
):
    if cluster_data is None or cluster_layer_spans is None or len(cluster_data) == 0:
        print("No cluster data or layer span data available for 2D plot.")
        return

    cluster_data_array = np.array(cluster_data)
    pts = cluster_data_array[:, 0]
    etas = cluster_data_array[:, 1]
    layer_spans = np.array(cluster_layer_spans)

    if len(pts) != len(layer_spans):
        print(
            f"Warning: Mismatch in data lengths (pts: {len(pts)}, layer_spans: {len(layer_spans)})"
        )
        min_len = min(len(pts), len(layer_spans))
        pts = pts[:min_len]
        etas = etas[:min_len]
        layer_spans = layer_spans[:min_len]

    fig, ax = plt.subplots()

    layer_sum, xedges, yedges = np.histogram2d(
        pts, etas, bins=[pt_bins, eta_bins], weights=layer_spans
    )
    layer_count, _, _ = np.histogram2d(pts, etas, bins=[pt_bins, eta_bins])

    mean_layer_span = np.divide(
        layer_sum, layer_count, out=np.zeros_like(layer_sum), where=layer_count > 0
    )

    im = ax.pcolormesh(
        eta_bins, pt_bins, mean_layer_span, cmap="plasma", shading="auto"
    )
    cbar = plt.colorbar(im, ax=ax)

    ax.set_xlabel("Simulated track $η$")
    ax.set_ylabel("Simulated track $p_T$ [GeV]")
    ax.grid(True, alpha=0.3, color="white", linewidth=0.5)
    hep.cms.label(
        llabel="Phase-2 Simulation Preliminary",
        data=False,
        rlabel=r"$t\bar{t}$ PU200, 14 TeV",
        ax=ax,
    )

    plt.savefig(
        f"{output_path}/layer_span_2d_eps_{epsilon:.3f}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_path}/layer_span_2d_eps_{epsilon:.3f}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    plt.savefig(
        f"{output_path}/layer_span_2d_eps_{epsilon:.3f}.png",
        dpi=300,
        bbox_inches="tight",
    )
