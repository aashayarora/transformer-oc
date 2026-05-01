import os
import sys
from glob import glob
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from dataset import PCDataset
from helpers import load_config
from model import TransformerOCModel


def resolve_checkpoint(model_path: str) -> str:
    if model_path.endswith(".ckpt") or model_path.endswith(".pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        return model_path

    candidate = Path(model_path) / "final_model.ckpt"
    if candidate.exists():
        return str(candidate)

    ckpts = sorted(glob(str(Path(model_path) / "checkpoints" / "*.ckpt")))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint found in {model_path}. Expected final_model.ckpt or checkpoints/*.ckpt"
        )
    return ckpts[-1]


def load_model(config, num_node_features, checkpoint_path, device):
    model = TransformerOCModel(
        input_dim=num_node_features,
        hidden_dim=config.get("hidden_dim", config.get("d_model", 128)),
        num_layers=config["num_layers"],
        num_heads=config["nhead"],
        latent_dim=config["latent_dim"],
        dropout=config.get("dropout", 0.1),
        dr_threshold=config.get("dr_threshold", 0.2),
        attention_chunk_size=config.get("attention_chunk_size", 1024),
    )
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    if any(key.startswith("model.") for key in state_dict.keys()):
        state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def collect_scores_and_labels(model, data_loader, device, use_fp16=False):
    all_scores = []
    all_labels = []

    for i, data in enumerate(data_loader):
        if i % 10 == 0:
            print(f"Processing event {i + 1}/{len(data_loader)}")

        x = data.x.to(device)
        batch = data.batch.to(device)

        if use_fp16 and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(x, batch)
            beta = out["B"].float()
        else:
            out = model(x, batch)
            beta = out["B"]

        scores = beta.cpu().detach().numpy().reshape(-1)

        if hasattr(data, "t3_isFake"):
            t3_is_fake = data.t3_isFake.cpu().detach().numpy().reshape(-1)
            # true class = real T3, so label is inverse of fake flag
            labels = (t3_is_fake <= 0).astype(np.int32)
        elif hasattr(data, "sim_index"):
            truth = data.sim_index.cpu().detach().numpy().reshape(-1)
            labels = (truth >= 0).astype(np.int32)
        elif hasattr(data, "md_simIdx"):
            md_truth = data.md_simIdx.cpu().detach().numpy()
            if md_truth.ndim == 1:
                labels = (md_truth >= 0).astype(np.int32)
            else:
                # One label per T3: true if any associated MD has valid sim index.
                labels = np.any(md_truth >= 0, axis=1).astype(np.int32)
        else:
            raise AttributeError("Input graph has neither sim_index nor md_simIdx for truth labeling.")

        if labels.shape[0] != scores.shape[0]:
            raise RuntimeError(
                f"Score/label length mismatch: beta has {scores.shape[0]} entries, "
                f"labels has {labels.shape[0]} entries."
            )

        finite_mask = np.isfinite(scores)
        scores = scores[finite_mask]
        labels = labels[finite_mask]

        all_scores.append(scores)
        all_labels.append(labels)

    if not all_scores:
        raise RuntimeError("No scores were collected from the dataloader.")

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    unique_classes = np.unique(y_true)
    if unique_classes.size < 2:
        raise RuntimeError(
            f"Need both true and fake T3s for ROC, but found labels: {unique_classes.tolist()}"
        )

    return y_true, y_score


def compute_operating_point(y_true, y_score, beta_cut):
    pred_positive = y_score >= beta_cut

    true_positive = np.sum((y_true == 1) & pred_positive)
    false_positive = np.sum((y_true == 0) & pred_positive)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    tpr = true_positive / n_pos if n_pos > 0 else 0.0
    fpr = false_positive / n_neg if n_neg > 0 else 0.0
    return fpr, tpr


def plot_and_save_roc(y_true, y_score, output_path, beta_cut=None, n_events=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.plot(fpr, tpr, lw=2.5, color="tab:blue", label="Condensation likelihood ROC")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Random")

    ax.set_xlabel("Fake T3 efficiency (FPR)", fontsize=30)
    ax.set_ylabel("True T3 efficiency (TPR)", fontsize=30)
    # ax.set_title("T3 ROC from condensation likelihood ($\\beta$)", fontsize=16, pad=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)

    operating_point = None
    if beta_cut is not None:
        cut_fpr, cut_tpr = compute_operating_point(y_true, y_score, beta_cut)
        operating_point = (cut_fpr, cut_tpr)
        ax.scatter(
            [cut_fpr],
            [cut_tpr],
            marker="o",
            s=90,
            color="tab:red",
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
            label=fr"Current cut: $\beta \geq {beta_cut:.3f}$",
        )

    ax.legend(loc="lower right", fontsize=20, frameon=True)

    annotation_lines = [f"AUC = {auc:.4f}"]
    if operating_point is not None:
        annotation_lines.extend([
            fr"$\beta_{{cut}}$ = {beta_cut:.3f}",
            fr"TPR = {operating_point[1]:.3f}",
            fr"FPR = {operating_point[0]:.3f}",
        ])
    ax.text(
        0.04,
        0.08,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        fontsize=20,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.95),
    )

    hep.cms.text("Phase-2 Simulation Preliminary", ax=ax)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        str(Path(output_path).with_suffix(".npz")),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc=auc,
        beta_cut=np.nan if beta_cut is None else beta_cut,
        cut_fpr=np.nan if operating_point is None else operating_point[0],
        cut_tpr=np.nan if operating_point is None else operating_point[1],
    )

    return auc, operating_point


def plot_score_distributions(y_true, y_score, output_path, beta_cut=None):
    real_scores = y_score[y_true == 1]
    fake_scores = y_score[y_true == 0]

    bins = np.linspace(0.0, 1.0, 60)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.hist(
        fake_scores,
        bins=bins,
        density=True,
        alpha=0.45,
        color="tab:orange",
        edgecolor="none",
        label=f"Fake T3s",
    )
    ax.hist(
        real_scores,
        bins=bins,
        density=True,
        alpha=0.45,
        color="tab:blue",
        edgecolor="none",
        label=f"Real T3s",
    )

    if beta_cut is not None:
        ax.axvline(beta_cut, color="tab:red", linestyle="--", linewidth=2.0, label=fr"Current cut: $\beta={beta_cut:.3f}$")

    ax.set_xlabel("Condensation likelihood score ($\\beta$)", fontsize=30)
    ax.set_ylabel("Density", fontsize=30)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", fontsize=20, frameon=True, ncol=1)

    hep.cms.text("Phase-2 Simulation Preliminary", ax=ax)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Standalone ROC for condensation likelihood on true/fake T3s")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint (.ckpt/.pt) or output dir")
    parser.add_argument("--n-events", type=int, default=20, help="Number of validation events to use")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu or cuda")
    parser.add_argument("--output", type=str, default=None, help="Output ROC png path")
    parser.add_argument("--beta-cut", type=float, default=None, help="Beta threshold to mark on ROC/distribution (defaults to config beta_threshold)")
    args = parser.parse_args()

    config = load_config(args.config)
    val_dir = config.get("val_data_dir", None)
    if val_dir is None:
        raise KeyError("config is missing 'val_data_dir'")

    subset = args.n_events if args.n_events and args.n_events > 0 else None
    dataset = PCDataset(val_dir, subset=subset)
    data_loader = DataLoader(dataset, shuffle=False, drop_last=False, num_workers=args.num_workers)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"

    ckpt_path = resolve_checkpoint(args.model)

    print(f"Loaded dataset: {len(dataset)} events (subset={subset})")
    print(f"Using device: {device}")
    print(f"Loading model from: {ckpt_path}")

    model = load_model(config, dataset[0].num_node_features, ckpt_path, device)

    y_true, y_score = collect_scores_and_labels(model, data_loader, device, use_fp16=use_fp16)

    output_path = args.output
    if output_path is None:
        output_dir = Path(args.model) if Path(args.model).is_dir() else Path(args.model).parent
        output_path = str(output_dir / f"t3_beta_roc_nevt_{len(dataset)}.png")

    output_path = str(output_path)
    dist_output_path = str(Path(output_path).with_name(Path(output_path).stem + "_score_dist.png"))

    beta_cut = args.beta_cut
    if beta_cut is None:
        beta_cut = config.get("beta_threshold", None)
    if beta_cut is not None:
        beta_cut = float(beta_cut)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    auc, operating_point = plot_and_save_roc(
        y_true=y_true,
        y_score=y_score,
        output_path=output_path,
        beta_cut=beta_cut,
        n_events=len(dataset),
    )
    plot_score_distributions(y_true=y_true, y_score=y_score, output_path=dist_output_path, beta_cut=beta_cut)

    print("\nDone.")
    print(f"  ROC plot: {output_path}")
    print(f"  Score distribution plot: {dist_output_path}")
    print(f"  ROC data: {Path(output_path).with_suffix('.npz')}")
    print(f"  AUC: {auc:.6f}")
    if beta_cut is not None and operating_point is not None:
        print(f"  Beta cut: {beta_cut:.6f} -> FPR={operating_point[0]:.6f}, TPR={operating_point[1]:.6f}")


if __name__ == "__main__":
    main()
