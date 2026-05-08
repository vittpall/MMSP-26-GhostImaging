import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import time
from tqdm import tqdm
import os
import json

from models.temporal_ghost_gpt       import TemporalGhostGPT
from models.temporal_ghost_gpt       import GhostGPT
from models.cnn_ghost                import CNNGhost
from models.unet_ghost               import UNetGhost
from models.haar_temporal_ghost_gpt  import HaarTemporalGhostGPT
from models.fista_warm_dynghost      import FISTAWarmDynGhost, compute_warm_start
from datasets import MovingMNISTGhost, MovingCIFAR10Ghost, KvasirGhost

import sys
sys.path.append('..')
from models.classical_alg import DGI_Recon, PseudoInverse_Recon, FISTA_Recon

# ============================================================================
# CONFIG
# ============================================================================

SPECKLE_PATH   = 'data/speckle_pattern.pt'
CHECKPOINT_DIR = './checkpoints'
OUTPUT_DIR     = './outputs'
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'

# DynGhost checkpoint per dataset — set to None to skip
CHECKPOINTS = {
    'mnist':   'checkpoints/temporal_ghost_gpt_final.pt',
    #'cifar10': 'checkpoints/dynghost_cifar10_final.pt',
    #'kvasir':  'checkpoints/temporal_ghost_gpt_kvasir_final.pt',
}

# CNN / U-Net checkpoints per dataset — set to None to skip
CNN_CHECKPOINTS = {
    #'mnist':   'checkpoints/cnn_mnist_final.pt',
    #'cifar10': 'checkpoints/cnn_cifar10_final.pt',
    #'kvasir': 'checkpoints/cnn_kvasir_final.pt',
}

UNET_CHECKPOINTS = {
    #'mnist':   'checkpoints/unet_mnist_final.pt',
    #'cifar10': 'checkpoints/unet_cifar10_final.pt',
    #'kvasir': 'checkpoints/unet_kvasir_final.pt',
}

GHOSTGPT_CHECKPOINTS = {
    #'mnist':   'checkpoints/ghost_gpt_mnist_final.pt',
    #'cifar10': 'checkpoints/ghostgpt_cifar10_final.pt',
    #'kvasir':  'checkpoints/ghostgpt_kvasir_final.pt',
}

HAAR_CHECKPOINTS = {
    'mnist':   'checkpoints/haarghost_mnist_final.pt',
    #'cifar10': 'checkpoints/haar_ghost_cifar10_final.pt',
    #'kvasir':  'checkpoints/haar_ghost_kvasir_final.pt',
}

FISTADYNGHOST_CHECKPOINTS = {
    'mnist':   'checkpoints/fistadynghost_mnist_final.pt',
    #'cifar10': 'checkpoints/fistadynghost_cifar10_final.pt',
    #'kvasir':  'checkpoints/fistadynghost_kvasir_final.pt',
}

DATASET_LABELS = {
    'mnist':   'Moving MNIST',
    #'cifar10': 'Moving CIFAR-10',
    #'kvasir': 'Kvasir Endoscopy',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(pred, target):
    pred_np   = pred.cpu().numpy()   if isinstance(pred,   torch.Tensor) else pred
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    pred_np   = np.clip(pred_np,   0, 1)
    target_np = np.clip(target_np, 0, 1)
    return mse(target_np, pred_np), ssim(target_np, pred_np, data_range=1.0)

# ============================================================================
# DATASET FACTORY
# ============================================================================

def get_test_dataset(name, speckle_patterns, image_size=256, seq_length=8):
    kwargs = dict(speckle_patterns=speckle_patterns,
                  seq_length=seq_length,
                  image_size=image_size,
                  train=False)
    if name == 'mnist':
        return MovingMNISTGhost(dataset_size=500, **kwargs)
    elif name == 'cifar10':
        return MovingCIFAR10Ghost(dataset_size=500, **kwargs)
    elif name == 'kvasir':
        return KvasirGhost(dataset_size=200,
                           kvasir_root='./data/kvasir-v2',
                           motion_scale=5.0, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# ============================================================================
# MODEL LOADERS
# ============================================================================

def load_temporal_model(checkpoint_path, num_patterns, device):
    model = TemporalGhostGPT(
        d_in=32, d_out=32, num_blocks=8, number_of_heads=8,
        embedding_dim=32, flattened_image_size=256*256,
        context_size=num_patterns, final_image_size=256*256,
        seq_length=8
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded epoch {checkpoint.get('epoch', '?')} "
              f"[{checkpoint.get('dataset', '?')}]")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_cnn_model(checkpoint_path, num_patterns, device):
    model = CNNGhost(num_patterns=num_patterns,
                     image_size=256, seq_length=8).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_unet_model(checkpoint_path, num_patterns, device):
    model = UNetGhost(num_patterns=num_patterns,
                      image_size=256, seq_length=8).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_haar_model(checkpoint_path, num_patterns, device):
    model = HaarTemporalGhostGPT(
        d_in=32, d_out=32, num_blocks=8, number_of_heads=8,
        embedding_dim=32, num_patterns=num_patterns,
        final_image_size=256 * 256, seq_length=8,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded epoch {checkpoint.get('epoch', '?')} "
              f"[{checkpoint.get('dataset', '?')}]")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_fistadynghost_model(checkpoint_path, num_patterns, device,
                              patch_size=32):
    model = FISTAWarmDynGhost(
        d_in=32, d_out=32, num_blocks=8, number_of_heads=8,
        embedding_dim=32, flattened_image_size=256 * 256,
        context_size=num_patterns, final_image_size=256 * 256,
        seq_length=8, patch_size=patch_size,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded epoch {checkpoint.get('epoch', '?')} "
              f"[{checkpoint.get('dataset', '?')}]")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_ghostgpt_model(checkpoint_path, num_patterns, device):
    model = GhostGPT(
        d_in=32, d_out=32, num_blocks=8, number_of_heads=8,
        embedding_dim=32, flattened_image_size=256 * 256,
        context_size=num_patterns, final_image_size=256 * 256,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded epoch {checkpoint.get('epoch', '?')} "
              f"[{checkpoint.get('dataset', '?')}]")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_temporal_model(model, dataset, patterns_flat, device,
                             num_samples=100):
    """DynGhost — uses full T-frame sequence."""
    model.eval()
    all_mse, all_ssim, times = [], [], []
    frame_mse, frame_ssim    = [], []

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))),
                      desc="  DynGhost"):
            sample      = dataset[i]
            buckets     = sample['buckets'].unsqueeze(0).to(device)
            frames_gt   = sample['frames']

            start       = time.time()
            pred_frames = model(patterns_flat, buckets).squeeze(0).cpu()
            times.append(time.time() - start)

            s_mse, s_ssim = [], []
            for t in range(frames_gt.shape[0]):
                m, s = compute_metrics(pred_frames[t], frames_gt[t])
                s_mse.append(m);  s_ssim.append(s)

            all_mse.append(np.mean(s_mse))
            all_ssim.append(np.mean(s_ssim))
            frame_mse.append(s_mse)
            frame_ssim.append(s_ssim)

    return {
        'mse_mean':     np.mean(all_mse),
        'mse_std':      np.std(all_mse),
        'ssim_mean':    np.mean(all_ssim),
        'ssim_std':     np.std(all_ssim),
        'time_mean_ms': np.mean(times) * 1000,
        'time_std_ms':  np.std(times)  * 1000,
        'frame_mse':    np.mean(frame_mse,  axis=0).tolist(),
        'frame_ssim':   np.mean(frame_ssim, axis=0).tolist(),
    }


def evaluate_dl_model(model, dataset, device, num_samples=100,
                       desc='Model'):
    """
    Generic evaluator for CNN and U-Net.
    Both take raw buckets [B, T, M] and return [B, T, H, W].
    No speckle patterns needed — they learn directly from bucket values.
    """
    model.eval()
    all_mse, all_ssim, times = [], [], []

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))),
                      desc=f"  {desc}"):
            sample    = dataset[i]
            buckets   = sample['buckets'].unsqueeze(0).to(device)  # [1,T,M]
            frames_gt = sample['frames']                            # [T,H,W]

            start       = time.time()
            pred_frames = model(buckets).squeeze(0).cpu()           # [T,H,W]
            times.append(time.time() - start)

            s_mse, s_ssim = [], []
            for t in range(frames_gt.shape[0]):
                m, s = compute_metrics(pred_frames[t], frames_gt[t])
                s_mse.append(m);  s_ssim.append(s)

            all_mse.append(np.mean(s_mse))
            all_ssim.append(np.mean(s_ssim))

    return {
        'mse_mean':     np.mean(all_mse),
        'mse_std':      np.std(all_mse),
        'ssim_mean':    np.mean(all_ssim),
        'ssim_std':     np.std(all_ssim),
        'time_mean_ms': np.mean(times) * 1000,
        'time_std_ms':  np.std(times)  * 1000,
    }


def evaluate_ghostgpt_model(model, dataset, patterns_flat, device,
                             num_samples=100):
    """GhostGPT — single-frame model evaluated on every frame in the sequence."""
    model.eval()
    all_mse, all_ssim, times = [], [], []

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))),
                      desc="  GhostGPT"):
            sample    = dataset[i]
            buckets   = sample['buckets'].to(device)   # [T, M]
            frames_gt = sample['frames']                # [T, H, W]

            s_mse, s_ssim = [], []
            t_total = 0.0
            for t in range(buckets.shape[0]):
                bucket_t = buckets[t].unsqueeze(0)      # [1, M]
                start    = time.time()
                pred_flat = model(patterns_flat, bucket_t)   # [1, H*W]
                t_total  += time.time() - start
                H = W = int(pred_flat.shape[-1] ** 0.5)
                pred = pred_flat.view(H, W).cpu()
                m, s = compute_metrics(pred, frames_gt[t])
                s_mse.append(m);  s_ssim.append(s)

            times.append(t_total / buckets.shape[0])
            all_mse.append(np.mean(s_mse))
            all_ssim.append(np.mean(s_ssim))

    return {
        'mse_mean':     np.mean(all_mse),
        'mse_std':      np.std(all_mse),
        'ssim_mean':    np.mean(all_ssim),
        'ssim_std':     np.std(all_ssim),
        'time_mean_ms': np.mean(times) * 1000,
        'time_std_ms':  np.std(times)  * 1000,
    }


def evaluate_fistadynghost_model(model, dataset, patterns_flat, device,
                                  num_samples=100):
    """FISTAWarmDynGhost — DGI warm-start computed on-the-fly per sample."""
    model.eval()
    all_mse, all_ssim, times = [], [], []
    frame_mse, frame_ssim    = [], []
    H = W = int(patterns_flat.shape[-1] ** 0.5)

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))),
                      desc="  FISTAWarmDynGhost"):
            sample    = dataset[i]
            buckets   = sample['buckets'].unsqueeze(0).to(device)   # [1, T, M]
            frames_gt = sample['frames']                             # [T, H, W]

            warm = compute_warm_start(patterns_flat, buckets, H, W) # [1, T, H, W]

            start       = time.time()
            pred_frames = model(patterns_flat, buckets, warm).squeeze(0).cpu()
            times.append(time.time() - start)

            s_mse, s_ssim = [], []
            for t in range(frames_gt.shape[0]):
                m, s = compute_metrics(pred_frames[t], frames_gt[t])
                s_mse.append(m);  s_ssim.append(s)

            all_mse.append(np.mean(s_mse))
            all_ssim.append(np.mean(s_ssim))
            frame_mse.append(s_mse)
            frame_ssim.append(s_ssim)

    return {
        'mse_mean':     np.mean(all_mse),
        'mse_std':      np.std(all_mse),
        'ssim_mean':    np.mean(all_ssim),
        'ssim_std':     np.std(all_ssim),
        'time_mean_ms': np.mean(times) * 1000,
        'time_std_ms':  np.std(times)  * 1000,
        'frame_mse':    np.mean(frame_mse,  axis=0).tolist(),
        'frame_ssim':   np.mean(frame_ssim, axis=0).tolist(),
    }


def evaluate_classical_algorithms(dataset, speckle_patterns,
                                   num_samples=30):
    """DGI, PI, FISTA on the middle frame of each sequence."""
    results = {k: {'mse': [], 'ssim': [], 'time': []}
               for k in ('DGI', 'PI', 'FISTA')}

    for i in tqdm(range(min(num_samples, len(dataset))),
                  desc="  Classical algorithms"):
        sample   = dataset[i]
        buckets  = sample['buckets'].numpy()
        frames_gt = sample['frames'].numpy()
        t        = buckets.shape[0] // 2
        bucket_t = buckets[t]
        frame_gt = frames_gt[t]

        for alg, fn, kwargs in [
            ('DGI',   DGI_Recon,          {}),
            ('PI',    PseudoInverse_Recon, {}),
            ('FISTA', FISTA_Recon,         {'eps': 50}),
        ]:
            start = time.time()
            pred  = fn(speckle_patterns, bucket_t, **kwargs)
            results[alg]['time'].append(time.time() - start)
            m, s = compute_metrics(pred, frame_gt)
            results[alg]['mse'].append(m)
            results[alg]['ssim'].append(s)

    return {
        alg: {
            'mse_mean':     np.mean(v['mse']),
            'mse_std':      np.std(v['mse']),
            'ssim_mean':    np.mean(v['ssim']),
            'ssim_std':     np.std(v['ssim']),
            'time_mean_ms': np.mean(v['time']) * 1000,
            'time_std_ms':  np.std(v['time'])  * 1000,
        }
        for alg, v in results.items()
    }

# ============================================================================
# PRINTING
# ============================================================================

def print_table(results_dict, dataset_label):
    print(f"\n{'='*80}")
    print(f"  {dataset_label}")
    print(f"{'='*80}")
    print(f"{'Method':<28} {'MSE':^22} {'SSIM':^22} {'Time (ms)':^18}")
    print(f"{'-'*80}")
    order = ['DynGhost (ours)', 'FISTAWarmDynGhost', 'GhostGPT', 'HaarGhost',
             'CNN', 'U-Net', 'DGI', 'PI', 'FISTA']
    printed = set()
    for name in order + [k for k in results_dict if k not in order]:
        if name in results_dict and name not in printed:
            r = results_dict[name]
            print(f"{name:<28} "
                  f"{r['mse_mean']:.4f} ± {r['mse_std']:.4f}     "
                  f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}     "
                  f"{r['time_mean_ms']:7.1f}")
            printed.add(name)
    print(f"{'='*80}")

# ============================================================================
# PLOTTING
# ============================================================================

def plot_dataset_comparison(all_dataset_results,
                             save_path='outputs/dataset_comparison.png'):
    datasets    = list(all_dataset_results.keys())
    method_sets = [set(all_dataset_results[d].keys()) for d in datasets]
    methods     = list(method_sets[0].intersection(*method_sets[1:]))

    # Fixed display order
    order = ['DynGhost (ours)', 'GhostGPT', 'HaarGhost', 'CNN', 'U-Net', 'DGI', 'PI', 'FISTA']
    methods = [m for m in order if m in methods] + \
              [m for m in methods if m not in order]

    x      = np.arange(len(datasets))
    width  = 0.8 / len(methods)
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics   = [
        ('mse_mean',     'MSE',       'MSE (↓ lower is better)'),
        ('ssim_mean',    'SSIM',      'SSIM (↑ higher is better)'),
        ('time_mean_ms', 'Time (ms)', 'Inference time (↓ lower is better)'),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for j, (method, color) in enumerate(zip(methods, colors)):
            vals = [all_dataset_results[d][method][key]
                    if method in all_dataset_results[d] else 0
                    for d in datasets]
            ax.bar(x + j * width - 0.4 + width / 2, vals,
                   width, label=method, color=color,
                   edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets],
                           fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        if key == 'time_mean_ms':
            ax.set_yscale('log')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(methods), fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    plt.suptitle('DynGhost — cross-dataset comparison', fontsize=13,
                 fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_ssim_bars_per_dataset(all_dataset_results,
                                save_path='outputs/ssim_per_dataset.png'):
    n_datasets = len(all_dataset_results)
    fig, axes  = plt.subplots(1, n_datasets,
                               figsize=(5 * n_datasets, 4), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    order = ['DynGhost (ours)', 'GhostGPT', 'HaarGhost', 'CNN', 'U-Net', 'DGI', 'PI', 'FISTA']

    for ax, (dname, results) in zip(axes, all_dataset_results.items()):
        methods   = [m for m in order if m in results] + \
                    [m for m in results if m not in order]
        ssim_vals = [results[m]['ssim_mean'] for m in methods]
        ssim_errs = [results[m]['ssim_std']  for m in methods]
        colors    = ['#2ecc71' if 'DynGhost' in m else
                     '#3498db' if m == 'GhostGPT'  else
                     '#e67e22' if m == 'CNN'        else
                     '#9b59b6' if m == 'U-Net'      else
                     '#95a5a6' for m in methods]

        ax.bar(range(len(methods)), ssim_vals, yerr=ssim_errs,
               color=colors, capsize=4, edgecolor='white', linewidth=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
        ax.set_title(DATASET_LABELS.get(dname, dname), fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('SSIM', fontsize=11)
    plt.suptitle('SSIM across datasets and methods', fontsize=12,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_temporal_consistency(model, dataset, patterns_flat, device,
                               dataset_name='mnist', save_path=None):
    model.eval()
    num_samples = min(50, len(dataset))
    seq_length  = dataset.seq_length
    all_ssim    = np.zeros((num_samples, seq_length))

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="  Temporal consistency"):
            sample      = dataset[i]
            buckets     = sample['buckets'].unsqueeze(0).to(device)
            frames_gt   = sample['frames']
            pred_frames = model(patterns_flat, buckets).squeeze(0).cpu()
            for t in range(seq_length):
                _, s = compute_metrics(pred_frames[t], frames_gt[t])
                all_ssim[i, t] = s

    mean_ssim = np.mean(all_ssim, axis=0)
    std_ssim  = np.std(all_ssim,  axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(range(seq_length), mean_ssim, 'o-', linewidth=2,
             markersize=6, color='#2ecc71')
    plt.fill_between(range(seq_length),
                     mean_ssim - std_ssim,
                     mean_ssim + std_ssim,
                     alpha=0.25, color='#2ecc71')
    plt.xlabel('Frame index', fontsize=11)
    plt.ylabel('SSIM', fontsize=11)
    plt.title(f'Temporal consistency — '
              f'{DATASET_LABELS.get(dataset_name, dataset_name)}',
              fontsize=12)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is None:
        save_path = f'{OUTPUT_DIR}/temporal_consistency_{dataset_name}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def visualize_reconstructions(dynghost_model, ghostgpt_model, haar_model,
                               cnn_model, unet_model, fistadynghost_model,
                               dataset, patterns_flat, speckle_patterns, device,
                               dataset_name='mnist', num_samples=3):
    """GT | DynGhost | FISTAWarm | GhostGPT | HaarGhost | CNN | U-Net | DGI | PI | FISTA."""
    fig, axes = plt.subplots(num_samples, 10,
                              figsize=(30, 3 * num_samples))
    col_titles = ['Ground truth', 'DynGhost (ours)', 'FISTAWarmDynGhost',
                  'GhostGPT', 'HaarGhost', 'CNN', 'U-Net',
                  'DGI', 'Pseudo-inv.', 'FISTA']

    H_img = W_img = int(patterns_flat.shape[-1] ** 0.5)

    with torch.no_grad():
        for i in range(num_samples):
            sample    = dataset[i]
            buckets   = sample['buckets']
            frames_gt = sample['frames']
            t         = buckets.shape[0] // 2
            frame_gt  = frames_gt[t].numpy()
            bucket_t  = buckets[t].numpy()
            buckets_dev = buckets.unsqueeze(0).to(device)   # [1, T, M]

            # DynGhost
            pred_dynghost = dynghost_model(
                patterns_flat, buckets_dev
            )[0, t].cpu().numpy()

            # FISTAWarmDynGhost
            if fistadynghost_model is not None:
                warm = compute_warm_start(patterns_flat, buckets_dev,
                                          H_img, W_img)
                pred_fistadyn = fistadynghost_model(
                    patterns_flat, buckets_dev, warm
                )[0, t].cpu().numpy()
            else:
                pred_fistadyn = np.zeros_like(frame_gt)

            # GhostGPT
            if ghostgpt_model is not None:
                bucket_t_tensor = buckets[t].unsqueeze(0).to(device)  # [1, M]
                pred_flat = ghostgpt_model(patterns_flat, bucket_t_tensor)
                H = W = int(pred_flat.shape[-1] ** 0.5)
                pred_ghostgpt = pred_flat.view(H, W).cpu().numpy()
            else:
                pred_ghostgpt = np.zeros_like(frame_gt)

            # HaarGhost
            if haar_model is not None:
                pred_haar = haar_model(
                    buckets_dev
                )[0, t].cpu().numpy()
            else:
                pred_haar = np.zeros_like(frame_gt)

            # CNN
            if cnn_model is not None:
                pred_cnn = cnn_model(
                    buckets_dev
                )[0, t].cpu().numpy()
            else:
                pred_cnn = np.zeros_like(frame_gt)

            # U-Net
            if unet_model is not None:
                pred_unet = unet_model(
                    buckets_dev
                )[0, t].cpu().numpy()
            else:
                pred_unet = np.zeros_like(frame_gt)

            # Classical
            pred_dgi   = DGI_Recon(speckle_patterns, bucket_t)
            pred_pi    = PseudoInverse_Recon(speckle_patterns, bucket_t)
            pred_fista = FISTA_Recon(speckle_patterns, bucket_t, eps=50)

            preds = [frame_gt, pred_dynghost, pred_fistadyn,
                     pred_ghostgpt, pred_haar, pred_cnn, pred_unet,
                     pred_dgi, pred_pi, pred_fista]

            for j, (pred, title) in enumerate(zip(preds, col_titles)):
                ax = axes[i, j]
                ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                if i == 0:
                    ax.set_title(title, fontsize=8)
                if j > 0:
                    _, s = compute_metrics(pred, frame_gt)
                    ax.set_xlabel(f'SSIM={s:.3f}', fontsize=7)

    plt.suptitle(f'Reconstructions — '
                 f'{DATASET_LABELS.get(dataset_name, dataset_name)}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/reconstructions_{dataset_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Device: {DEVICE}")

    # ---- Speckle patterns ----
    print("\nLoading speckle patterns...")
    speckle_patterns = torch.load(SPECKLE_PATH)
    if isinstance(speckle_patterns, torch.Tensor):
        speckle_patterns = speckle_patterns.numpy()
    num_patterns = speckle_patterns.shape[0]
    print(f"  Shape: {speckle_patterns.shape}")

    patterns_flat = (torch.tensor(speckle_patterns).float()
                     .view(num_patterns, -1).to(DEVICE))

    all_dataset_results = {}

    for dataset_name, ckpt_path in CHECKPOINTS.items():

        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"\nSkipping {dataset_name} — DynGhost checkpoint "
                  f"not found: {ckpt_path}")
            continue

        print(f"\n{'#'*60}")
        print(f"  Evaluating on: {DATASET_LABELS[dataset_name]}")
        print(f"{'#'*60}")

        try:
            dataset = get_test_dataset(dataset_name, speckle_patterns)
        except FileNotFoundError as e:
            print(f"  Dataset not available: {e}")
            continue

        results = {}

        # ---- DynGhost ----
        print("\nLoading DynGhost...")
        dynghost = load_temporal_model(ckpt_path, num_patterns, DEVICE)
        print("Evaluating DynGhost...")
        results['DynGhost (ours)'] = evaluate_temporal_model(
            dynghost, dataset, patterns_flat, DEVICE, num_samples=100
        )

        # ---- CNN ----
        cnn_ckpt = CNN_CHECKPOINTS.get(dataset_name)
        cnn_model = None
        if cnn_ckpt and os.path.exists(cnn_ckpt):
            print("\nLoading CNN...")
            cnn_model = load_cnn_model(cnn_ckpt, num_patterns, DEVICE)
            print("Evaluating CNN...")
            results['CNN'] = evaluate_dl_model(
                cnn_model, dataset, DEVICE,
                num_samples=100, desc='CNN'
            )
        else:
            print(f"\nCNN checkpoint not found for {dataset_name} "
                  f"— skipping")

        # ---- U-Net ----
        unet_ckpt = UNET_CHECKPOINTS.get(dataset_name)
        unet_model = None
        if unet_ckpt and os.path.exists(unet_ckpt):
            print("\nLoading U-Net...")
            unet_model = load_unet_model(unet_ckpt, num_patterns, DEVICE)
            print("Evaluating U-Net...")
            results['U-Net'] = evaluate_dl_model(
                unet_model, dataset, DEVICE,
                num_samples=100, desc='U-Net'
            )
        else:
            print(f"\nU-Net checkpoint not found for {dataset_name} "
                  f"— skipping")

        # ---- GhostGPT ----
        ghostgpt_ckpt = GHOSTGPT_CHECKPOINTS.get(dataset_name)
        ghostgpt_model = None
        if ghostgpt_ckpt and os.path.exists(ghostgpt_ckpt):
            print("\nLoading GhostGPT...")
            ghostgpt_model = load_ghostgpt_model(
                ghostgpt_ckpt, num_patterns, DEVICE)
            print("Evaluating GhostGPT...")
            results['GhostGPT'] = evaluate_ghostgpt_model(
                ghostgpt_model, dataset, patterns_flat, DEVICE,
                num_samples=100
            )
        else:
            print(f"\nGhostGPT checkpoint not found for {dataset_name} "
                  f"— skipping")

        # ---- HaarGhost ----
        haar_ckpt = HAAR_CHECKPOINTS.get(dataset_name)
        haar_model = None
        if haar_ckpt and os.path.exists(haar_ckpt):
            print("\nLoading HaarGhost...")
            haar_model = load_haar_model(haar_ckpt, num_patterns, DEVICE)
            print("Evaluating HaarGhost...")
            results['HaarGhost'] = evaluate_dl_model(
                haar_model, dataset, DEVICE,
                num_samples=100, desc='HaarGhost'
            )
        else:
            print(f"\nHaarGhost checkpoint not found for {dataset_name} "
                  f"— skipping")

        # ---- FISTAWarmDynGhost ----
        fistadynghost_ckpt = FISTADYNGHOST_CHECKPOINTS.get(dataset_name)
        fistadynghost_model = None
        if fistadynghost_ckpt and os.path.exists(fistadynghost_ckpt):
            print("\nLoading FISTAWarmDynGhost...")
            fistadynghost_model = load_fistadynghost_model(
                fistadynghost_ckpt, num_patterns, DEVICE)
            print("Evaluating FISTAWarmDynGhost...")
            results['FISTAWarmDynGhost'] = evaluate_fistadynghost_model(
                fistadynghost_model, dataset, patterns_flat, DEVICE,
                num_samples=100
            )
        else:
            print(f"\nFISTAWarmDynGhost checkpoint not found for "
                  f"{dataset_name} — skipping")

        # ---- Classical ----
        print("\nEvaluating classical algorithms...")
        classical = evaluate_classical_algorithms(
            dataset, speckle_patterns, num_samples=30
        )
        results.update(classical)

        all_dataset_results[dataset_name] = results
        print_table(results, DATASET_LABELS[dataset_name])

        # Per-dataset plots
        plot_temporal_consistency(dynghost, dataset, patterns_flat,
                                  DEVICE, dataset_name)
        visualize_reconstructions(dynghost, ghostgpt_model, haar_model,
                                  cnn_model, unet_model, fistadynghost_model,
                                  dataset, patterns_flat, speckle_patterns,
                                  DEVICE, dataset_name)

    # ---- Cross-dataset plots ----
    if len(all_dataset_results) > 1:
        print("\nGenerating cross-dataset comparison plots...")
        plot_dataset_comparison(all_dataset_results)
        plot_ssim_bars_per_dataset(all_dataset_results)

    # ---- Save JSON ----
    results_path = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_dataset_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY — SSIM across datasets")
    print(f"{'='*60}")
    order = ['DynGhost (ours)', 'FISTAWarmDynGhost', 'GhostGPT', 'HaarGhost',
             'CNN', 'U-Net', 'DGI', 'PI', 'FISTA']
    for dname, results in all_dataset_results.items():
        print(f"\n  {DATASET_LABELS[dname]}")
        for method in order:
            if method in results:
                r = results[method]
                print(f"    {method:<22} "
                      f"SSIM={r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}  "
                      f"MSE={r['mse_mean']:.4f} ± {r['mse_std']:.4f}")


if __name__ == "__main__":
    main()