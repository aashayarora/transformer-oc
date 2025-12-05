import argparse
import torch
from torch.amp import autocast
import os
import sys
import glob

sys.path.append('./src')

from dataset import PCDataset
from torch_geometric.loader import DataLoader
from model import TransformerOCModel, TransformerLightningModule
from helpers import load_config

from fastgraphcompute.object_condensation import ObjectCondensation
from utils.torch_geometric_interface import row_splits_from_strict_batch as batch_to_rowsplits

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist


def find_latest_checkpoint(config):
    output_dir = config.get('output_dir', './output')
    
    version_dirs = glob.glob(os.path.join(output_dir, 'lightning_logs', 'version_*'))
    if version_dirs:
        def get_version_num(path):
            try:
                return int(os.path.basename(path).split('_')[-1])
            except:
                return -1
        latest_version = max(version_dirs, key=get_version_num)
        
        checkpoint_patterns = [
            os.path.join(latest_version, 'checkpoints', 'model-*.ckpt'),
            os.path.join(latest_version, 'checkpoints', 'epoch*.ckpt'),
        ]
        
        for pattern in checkpoint_patterns:
            matches = glob.glob(pattern)
            if matches:
                return sorted(matches)[-1]
        
        last_ckpt = os.path.join(latest_version, 'checkpoints', 'last.ckpt')
        if os.path.exists(last_ckpt):
            return last_ckpt
    
    for pattern in [os.path.join(output_dir, '*.ckpt'), os.path.join(output_dir, 'checkpoints', '*.ckpt')]:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]
    
    return None


def extract_epoch_step_from_checkpoint(checkpoint_path):
    import re
    
    filename = os.path.basename(checkpoint_path)
    
    match = re.search(r'epoch[=_](\d+)', filename, re.IGNORECASE)
    epoch = int(match.group(1)) if match else None
    
    match = re.search(r'step[=_](\d+)', filename, re.IGNORECASE)
    step = int(match.group(1)) if match else None
    
    return epoch, step


def load_model_from_checkpoint(checkpoint_path, config, input_dim, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        
        if any(k.startswith('model.') for k in state_dict.keys()):
            model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        else:
            model_state_dict = state_dict
            
        if 'hyper_parameters' in checkpoint:
            hp = checkpoint['hyper_parameters']
            hidden_dim = hp.get('hidden_dim', config.get('hidden_dim', 128))
            num_layers = hp.get('num_layers', config['num_layers'])
            num_heads = hp.get('nhead', config['nhead'])
            latent_dim = hp.get('latent_dim', config['latent_dim'])
            dropout = hp.get('dropout', config.get('dropout', 0.1))
            print(f"  Using hyperparameters from checkpoint:")
            print(f"    hidden_dim={hidden_dim}, num_layers={num_layers}, nhead={num_heads}, latent_dim={latent_dim}")
        else:
            hidden_dim = config.get('hidden_dim', 128)
            num_layers = config['num_layers']
            num_heads = config['nhead']
            latent_dim = config['latent_dim']
            dropout = config.get('dropout', 0.1)
            print(f"  Using hyperparameters from config")
    else:
        model_state_dict = checkpoint
        hidden_dim = config.get('hidden_dim', 128)
        num_layers = config['num_layers']
        num_heads = config['nhead']
        latent_dim = config['latent_dim']
        dropout = config.get('dropout', 0.1)
    
    model = TransformerOCModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        latent_dim=latent_dim,
        dropout=dropout
    )
    
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def analyze_single_event(model, data, criterion, config, device):
    model.eval()
    
    w_att = config.get('loss_weight_attractive', 1.0)
    w_rep = config.get('loss_weight_repulsive', 1.0)
    w_beta = config.get('loss_weight_beta', 1.0)
    
    with torch.no_grad():
        x, sim_index, batch = data.x.to(device), data.sim_index.to(device), data.batch.to(device)
        sim_index[sim_index < 0] = -1
        row_splits = batch_to_rowsplits(batch)
        
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            model_out = model(x, batch)
        
        beta = model_out["B"].float()
        coords = model_out["H"].float()
        
        L_att, L_rep, L_beta, _, _ = criterion(
            beta=beta,
            coords=coords,
            asso_idx=sim_index.unsqueeze(-1).to(torch.int64),
            row_splits=row_splits
        )
        
        all_coords = coords.cpu().numpy()
        all_betas = beta.cpu().numpy().flatten()
        all_sim_indices = sim_index.cpu().numpy().flatten()
        
        if coords.shape[0] > 1000:
            idx = torch.randperm(coords.shape[0])[:1000]
            coords_sample = coords[idx]
        else:
            coords_sample = coords
        
        pairwise_dist = torch.cdist(coords_sample, coords_sample, p=2)
        n = pairwise_dist.size(0)
        triu_idx = torch.triu_indices(n, n, offset=1)
        distances = pairwise_dist[triu_idx[0], triu_idx[1]].cpu().numpy()
        
        unique_objs = np.unique(all_sim_indices[all_sim_indices >= 0])
        intra_dists = []
        object_centroids = {}
        for obj_id in unique_objs[:50]:
            mask = all_sim_indices == obj_id
            if mask.sum() > 1:
                obj_coords = all_coords[mask]
                intra_dists.extend(pdist(obj_coords))
                object_centroids[obj_id] = obj_coords.mean(axis=0)
        
        inter_dists = []
        if len(object_centroids) > 1:
            inter_dists = pdist(np.array(list(object_centroids.values())))
        
        dbscan_eps = config.get('dbscan_eps', 0.1)
        clustering = DBSCAN(eps=dbscan_eps, min_samples=config.get('dbscan_min_samples', 2)).fit(all_coords)
        pred_labels = clustering.labels_
        
        signal_mask = all_sim_indices >= 0
        if signal_mask.sum() > 0:
            true_signal = all_sim_indices[signal_mask].astype(int)
            pred_signal = pred_labels[signal_mask]
            n_pred_clusters = len(np.unique(pred_signal[pred_signal >= 0]))
            n_true_objects = len(np.unique(true_signal))
            
            purities = []
            for pred_id in np.unique(pred_signal):
                if pred_id < 0:
                    continue
                mask = pred_signal == pred_id
                true_ids = true_signal[mask]
                if len(true_ids) > 0:
                    purities.append(np.bincount(true_ids).max() / len(true_ids))
            avg_purity = np.mean(purities) if purities else 0
        else:
            n_pred_clusters = 0
            n_true_objects = 0
            avg_purity = 0
        
        return {
            'coords': all_coords,
            'betas': all_betas,
            'sim_indices': all_sim_indices,
            'pred_labels': pred_labels,
            'distances': distances,
            'intra_dists': intra_dists,
            'inter_dists': inter_dists,
            'losses': {
                'attractive': L_att.item(),
                'repulsive': L_rep.item(),
                'beta': L_beta.item(),
                'total': (w_att * L_att + w_rep * L_rep + w_beta * L_beta).item()
            },
            'clustering': {
                'n_pred': n_pred_clusters,
                'n_true': n_true_objects,
                'purity': avg_purity,
                'eps': dbscan_eps
            },
            'n_nodes': data.num_nodes
        }


def plot_event_analysis(metrics, event_idx, output_dir, epoch=None, step=None):
    coords = metrics['coords']
    betas = metrics['betas']
    sim_indices = metrics['sim_indices']
    distances = metrics['distances']
    losses = metrics['losses']
    clustering = metrics['clustering']
    
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(coords)
    
    title_parts = []
    if epoch is not None:
        title_parts.append(f'Epoch {epoch}')
    if step is not None:
        title_parts.append(f'Step {step}')
    title_parts.append(f'Event {event_idx}')
    title_parts.append(f'Nodes: {metrics["n_nodes"]}')
    title_parts.append(f'Loss: {losses["total"]:.4f}')
    title_parts.append(f'Purity: {clustering["purity"]:.3f}')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.hist(distances, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Repulsion threshold')
    ax.set_xlabel('Pairwise Distance in Latent Space')
    ax.set_ylabel('Density')
    ax.set_title(f'Pairwise Distances (mean={np.mean(distances):.3f})')
    ax.legend()
    
    ax = axes[0, 1]
    noise_mask = sim_indices < 0
    signal_mask = sim_indices >= 0
    ax.hist(betas[noise_mask], bins=30, alpha=0.6, label=f'Noise (n={noise_mask.sum()})', density=True)
    ax.hist(betas[signal_mask], bins=30, alpha=0.6, label=f'Signal (n={signal_mask.sum()})', density=True)
    ax.set_xlabel('Beta Values')
    ax.set_ylabel('Density')
    ax.set_title(f'Beta Distribution (noise={np.mean(betas[noise_mask]):.3f}, signal={np.mean(betas[signal_mask]):.3f})')
    ax.legend()
    
    ax = axes[0, 2]
    if metrics['intra_dists']:
        ax.hist(metrics['intra_dists'], bins=30, alpha=0.6, label=f'Intra-object (mean={np.mean(metrics["intra_dists"]):.3f})', density=True)
    if len(metrics['inter_dists']) > 0:
        ax.hist(metrics['inter_dists'], bins=30, alpha=0.6, label=f'Inter-object (mean={np.mean(metrics["inter_dists"]):.3f})', density=True)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Density')
    ratio = np.mean(metrics['inter_dists']) / np.mean(metrics['intra_dists']) if metrics['intra_dists'] else 0
    ax.set_title(f'Intra vs Inter Object Distances (ratio={ratio:.1f})')
    ax.legend()
    
    ax = axes[1, 0]
    unique_objects = np.unique(sim_indices[sim_indices >= 0])
    n_colors = min(20, len(unique_objects))
    
    if noise_mask.sum() > 0:
        ax.scatter(coords_2d[noise_mask, 0], coords_2d[noise_mask, 1], 
                   c='lightgray', s=10, alpha=0.3, label='Noise')
    
    cmap = plt.cm.get_cmap('tab20', n_colors)
    for idx, obj_id in enumerate(unique_objects[:n_colors]):
        mask = sim_indices == obj_id
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                   c=[cmap(idx % n_colors)], s=15, alpha=0.8)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA (colored by true object)')
    
    ax = axes[1, 1]
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=betas, 
                         cmap='viridis', s=15, alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label='Beta')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA (colored by beta)')
    
    ax = axes[1, 2]
    pred_labels = metrics['pred_labels']
    n_clusters = len(np.unique(pred_labels[pred_labels >= 0]))
    
    noise_pred_mask = pred_labels < 0
    if noise_pred_mask.sum() > 0:
        ax.scatter(coords_2d[noise_pred_mask, 0], coords_2d[noise_pred_mask, 1],
                   c='lightgray', s=10, alpha=0.3, label=f'DBSCAN noise')
    
    cmap_pred = plt.cm.get_cmap('tab20', min(20, n_clusters))
    for idx, cluster_id in enumerate(np.unique(pred_labels[pred_labels >= 0])[:20]):
        mask = pred_labels == cluster_id
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=[cmap_pred(idx % 20)], s=15, alpha=0.8)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'PCA (DBSCAN clusters: {clustering["n_pred"]}/{clustering["n_true"]}, eps={clustering["eps"]})')
    
    plt.tight_layout()
    
    filename_parts = []
    if epoch is not None:
        filename_parts.append(f'epoch_{epoch:06d}')
    if step is not None:
        filename_parts.append(f'step_{step:06d}')
    filename_parts.append(f'event_{event_idx:03d}')
    filename = '_'.join(filename_parts) + '.png'
    
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def print_summary_stats(all_metrics):
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    losses = {k: [m['losses'][k] for m in all_metrics] for k in all_metrics[0]['losses'].keys()}
    clustering = {k: [m['clustering'][k] for m in all_metrics] for k in all_metrics[0]['clustering'].keys()}
    
    print(f"\nLosses (mean ± std across {len(all_metrics)} events):")
    for k, v in losses.items():
        print(f"  {k:12s}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    
    print(f"\nClustering (eps={clustering['eps'][0]}):")
    print(f"  Predicted clusters: {np.mean(clustering['n_pred']):.1f} ± {np.std(clustering['n_pred']):.1f}")
    print(f"  True objects:       {np.mean(clustering['n_true']):.1f} ± {np.std(clustering['n_true']):.1f}")
    print(f"  Purity:             {np.mean(clustering['purity']):.3f} ± {np.std(clustering['purity']):.3f}")
    
    intra = [np.mean(m['intra_dists']) for m in all_metrics if m['intra_dists']]
    inter = [np.mean(m['inter_dists']) for m in all_metrics if len(m['inter_dists']) > 0]
    
    print(f"\nDistances:")
    print(f"  Intra-object:       {np.mean(intra):.3f} ± {np.std(intra):.3f}")
    print(f"  Inter-object:       {np.mean(inter):.3f} ± {np.std(inter):.3f}")
    print(f"  Ratio (inter/intra): {np.mean(inter)/np.mean(intra):.1f}")
    
    print("="*70 + "\n")


def plot_summary(all_metrics, output_dir):
    n_events = len(all_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Summary over {n_events} Events', fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    for loss_type in ['attractive', 'repulsive', 'beta', 'total']:
        values = [m['losses'][loss_type] for m in all_metrics]
        ax.bar(loss_type, np.mean(values), yerr=np.std(values), capsize=5, alpha=0.7)
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Components')
    
    ax = axes[0, 1]
    purities = [m['clustering']['purity'] for m in all_metrics]
    ax.hist(purities, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(purities), color='r', linestyle='--', label=f'Mean: {np.mean(purities):.3f}')
    ax.set_xlabel('Cluster Purity')
    ax.set_ylabel('Count')
    ax.set_title('Purity Distribution')
    ax.legend()
    
    ax = axes[1, 0]
    intra = [np.mean(m['intra_dists']) if m['intra_dists'] else 0 for m in all_metrics]
    inter = [np.mean(m['inter_dists']) if len(m['inter_dists']) > 0 else 0 for m in all_metrics]
    x = range(n_events)
    ax.plot(x, intra, 'o-', label='Intra-object', alpha=0.7)
    ax.plot(x, inter, 's-', label='Inter-object', alpha=0.7)
    ax.set_xlabel('Event')
    ax.set_ylabel('Mean Distance')
    ax.set_title('Intra vs Inter Object Distances')
    ax.legend()
    
    ax = axes[1, 1]
    n_pred = [m['clustering']['n_pred'] for m in all_metrics]
    n_true = [m['clustering']['n_true'] for m in all_metrics]
    ax.scatter(n_true, n_pred, alpha=0.7)
    max_val = max(max(n_true), max(n_pred))
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect')
    ax.set_xlabel('True Objects')
    ax.set_ylabel('Predicted Clusters')
    ax.set_title('Cluster Count: Predicted vs True')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate analysis plots from checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (auto-detect if not specified)')
    parser.add_argument('--n-events', type=int, default=1, help='Number of events to analyze')
    parser.add_argument('--event-idx', type=int, default=0, help='Index of event to start from')
    parser.add_argument('--output-dir', type=str, default='checkpoint_analysis', help='Output directory for plots')
    parser.add_argument('--data-split', type=str, default='val', choices=['train', 'val'], help='Data split to use')
    parser.add_argument('--subset', type=int, default=None, help='Subset of data to use')
    args = parser.parse_args()
    
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(config)
        if checkpoint_path is None:
            print("ERROR: No checkpoint found. Please specify --checkpoint")
            sys.exit(1)
    
    data_dir = config['val_data_dir'] if args.data_split == 'val' else config['train_data_dir']
    subset = args.subset or config.get(f'{args.data_split}_subset')
    
    print(f"Loading dataset from: {data_dir}")
    dataset = PCDataset(data_dir, subset=subset)
    print(f"  Dataset size: {len(dataset)} events")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    sample_data = dataset[0]
    input_dim = sample_data.num_node_features
    print(f"  Input features: {input_dim}")
    
    model = load_model_from_checkpoint(checkpoint_path, config, input_dim, device)
    
    criterion = ObjectCondensation(
        q_min=config.get('oc_q_min', 0.1),
        s_B=config.get('oc_s_B', 1.0)
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    epoch, step = extract_epoch_step_from_checkpoint(checkpoint_path)
    if epoch is not None or step is not None:
        print(f"Checkpoint info: epoch={epoch}, step={step}")
    
    print(f"\nAnalyzing {args.n_events} event(s) starting from index {args.event_idx}...")
    all_metrics = []
    
    for event_idx, data in enumerate(loader):
        if event_idx < args.event_idx:
            continue
        if event_idx >= args.event_idx + args.n_events:
            break
        
        print(f"  Event {event_idx}: {data.num_nodes} nodes...", end=' ')
        metrics = analyze_single_event(model, data, criterion, config, device)
        all_metrics.append(metrics)
        
        plot_event_analysis(metrics, event_idx, args.output_dir, epoch=epoch, step=step)
        print(f"done (purity={metrics['clustering']['purity']:.3f})")
    
    print_summary_stats(all_metrics)
    if len(all_metrics) > 1:
        plot_summary(all_metrics, args.output_dir)
    
    print(f"\nPlots saved to: {args.output_dir}/")
    print(f"  - event_XXX.png: Individual event analysis")
    if len(all_metrics) > 1:
        print(f"  - summary.png: Summary across all events")


if __name__ == '__main__':
    main()
