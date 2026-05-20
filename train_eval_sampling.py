"""
Train DynGhost with randomly generated speckle patterns and evaluate
performance across multiple sampling ratios (M / N).

Typical usage
-------------
# Single run — train with 376 patterns, then evaluate
python train_eval_sampling.py --num_patterns 376 --dataset mnist

# Sampling-ratio sweep — train+eval each configuration
python train_eval_sampling.py \
    --num_patterns 94,188,376,752,1024 \
    --dataset mnist --num_epochs 30

# Use an already-trained checkpoint (skip training)
python train_eval_sampling.py --num_patterns 376 --dataset mnist \
    --eval_only --checkpoint checkpoints/dynghost_mnist_speckle_M376_final.pt

Pattern types
-------------
  speckle  — random binary {0,1} patterns, each pixel lit with probability 0.5
  hadamard — deterministic Walsh-Hadamard S-matrix rows

Sampling ratio
--------------
  ratio = M / N   where N = image_size²
  e.g. 188 / (256*256) ≈ 0.29 %,   752 / 65536 ≈ 1.15 %
"""

import argparse
import os
import json
import glob
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as ssim_loss
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import mean_squared_error as mse_metric
from tqdm import tqdm

from models.temporal_ghost_gpt import TemporalGhostGPT
from datasets import MovingMNISTGhost, MovingCIFAR10Ghost, KvasirGhost, DAVISGhost
from generate_patterns import make_speckle_patterns, make_hadamard_s_patterns

# ============================================================================
# DEFAULT CONFIG  (all values overridable via CLI)
# ============================================================================

CONFIG = {
    'model':         'dynghost',
    'dataset':       'mnist',
    'pattern_type':  'speckle',   # 'speckle' | 'hadamard'
    'speckle_seed':  42,          # RNG seed for generated patterns
    'image_size':    256,
    'seq_length':    8,
    'batch_size':    4,
    'num_epochs':    30,
    'learning_rate': 3e-4,
    'weight_decay':  1e-3,
    'num_blocks':    8,
    'num_heads':     8,
    'embedding_dim': 32,
    'num_eval':      100,         # dataset samples for evaluation
}

CHECKPOINT_DIR = './checkpoints'
OUTPUT_DIR     = './outputs/sampling_study'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,     exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# PATTERN GENERATION
# ============================================================================

def generate_patterns(pattern_type: str, num_patterns: int,
                      image_size: int, seed: int = 42) -> np.ndarray:
    """
    Generate `num_patterns` measurement patterns of shape (M, H, W).

    For 'speckle'  — random binary with seed for reproducibility.
    For 'hadamard' — deterministic S-matrix rows (seed ignored).
    """
    if pattern_type == 'hadamard':
        N = image_size ** 2
        if (N & (N - 1)) != 0:
            raise ValueError(
                f"image_size={image_size} gives N={N}, not a power of 2. "
                "Hadamard patterns require N = 32², 64², 128², or 256²."
            )
        patterns = make_hadamard_s_patterns(image_size, num_patterns)
        print(f"Generated Hadamard patterns: {patterns.shape}")
    else:
        patterns = make_speckle_patterns(image_size, num_patterns, seed=seed)
        print(f"Generated random speckle patterns: {patterns.shape}  seed={seed}")
    return patterns


# ============================================================================
# DATASET FACTORY
# ============================================================================

def get_dataset(name, speckle_patterns, image_size, seq_length, train: bool):
    kwargs = dict(
        speckle_patterns=speckle_patterns,
        seq_length=seq_length,
        image_size=image_size,
        train=train,
    )
    if name == 'mnist':
        return MovingMNISTGhost(dataset_size=5000 if train else 500, **kwargs)
    elif name == 'cifar10':
        return MovingCIFAR10Ghost(dataset_size=5000 if train else 500, **kwargs)
    elif name == 'kvasir':
        return KvasirGhost(dataset_size=2000 if train else 200,
                           kvasir_root='./data/kvasir-v2',
                           motion_scale=5.0, **kwargs)
    elif name == 'davis':
        return DAVISGhost(
            davis_root='./data/DAVIS',
            speckle_patterns=speckle_patterns,
            seq_length=seq_length,
            image_size=image_size,
            train=train,
            dataset_size=2000 if train else 200,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ============================================================================
# MODEL
# ============================================================================

def build_model(num_patterns: int, config: dict) -> TemporalGhostGPT:
    return TemporalGhostGPT(
        d_in=config['embedding_dim'],
        d_out=config['embedding_dim'],
        num_blocks=config['num_blocks'],
        number_of_heads=config['num_heads'],
        embedding_dim=config['embedding_dim'],
        flattened_image_size=config['image_size'] ** 2,
        context_size=num_patterns,
        final_image_size=config['image_size'] ** 2,
        seq_length=config['seq_length'],
    ).to(DEVICE)


# ============================================================================
# LOSS
# ============================================================================

def compute_loss(pred_frames, frames, criterion):
    mse_val  = criterion(pred_frames, frames)
    ssim_val = 1 - ssim_loss(pred_frames, frames,
                              data_range=1.0, size_average=True)
    loss = mse_val + 0.5 * ssim_val

    if pred_frames.shape[1] > 1:
        pred_diff = pred_frames[:, 1:] - pred_frames[:, :-1]
        true_diff = frames[:, 1:]      - frames[:, :-1]
        loss = loss + 0.1 * criterion(pred_diff, true_diff)

    return loss


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def _ckpt_prefix(num_patterns: int, config: dict) -> str:
    return (f"dynghost_{config['dataset']}_{config['pattern_type']}"
            f"_M{num_patterns}")


def find_latest_checkpoint(prefix: str) -> Optional[str]:
    pattern = os.path.join(CHECKPOINT_DIR, f'{prefix}_epoch*.pt')
    ckpts   = glob.glob(pattern)
    if not ckpts:
        return None

    def _epoch(p):
        base = os.path.basename(p)
        return int(base.replace(f'{prefix}_epoch', '').replace('.pt', ''))

    return max(ckpts, key=_epoch)


def save_checkpoint(path, model, optimizer, epoch,
                    train_losses, val_losses, config, num_patterns):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses':         train_losses,
        'val_losses':           val_losses,
        'config':               config,
        'num_patterns':         num_patterns,
    }, path)
    print(f"  Saved: {path}")


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch  = ckpt['epoch']
        train_losses = ckpt.get('train_losses', [])
        val_losses   = ckpt.get('val_losses',   [])
        print(f"  Resumed from epoch {start_epoch}")
    else:
        model.load_state_dict(ckpt)
        start_epoch, train_losses, val_losses = 0, [], []
        print("  Loaded weights-only checkpoint")
    return start_epoch, train_losses, val_losses


# ============================================================================
# TRAINING
# ============================================================================

def train_one_config(num_patterns: int, patterns: np.ndarray,
                     config: dict, eval_only: bool = False,
                     checkpoint_override: Optional[str] = None) -> dict:
    """
    Train (or load) a DynGhost model for `num_patterns` patterns,
    then evaluate and return a metrics dict.
    """
    PREFIX        = _ckpt_prefix(num_patterns, config)
    final_ckpt    = os.path.join(CHECKPOINT_DIR, f'{PREFIX}_final.pt')
    patterns_flat = (torch.tensor(patterns).float()
                     .view(num_patterns, -1).to(DEVICE))

    # ---- Datasets ----
    print(f"\n  Building datasets for M={num_patterns}...")
    train_ds = get_dataset(config['dataset'], patterns,
                           config['image_size'], config['seq_length'], train=True)
    val_ds   = get_dataset(config['dataset'], patterns,
                           config['image_size'], config['seq_length'], train=False)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

    # ---- Model & optimiser ----
    model     = build_model(num_patterns, config)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e6:.2f}M")

    # ---- Decide whether to train ----
    start_epoch  = 0
    train_losses = []
    val_losses   = []

    ckpt_to_load = checkpoint_override or final_ckpt

    if eval_only:
        if not os.path.exists(ckpt_to_load):
            raise FileNotFoundError(
                f"--eval_only set but no checkpoint found at {ckpt_to_load}"
            )
        print(f"  Loading checkpoint: {ckpt_to_load}")
        start_epoch, train_losses, val_losses = load_checkpoint(
            ckpt_to_load, model, optimizer
        )
    elif os.path.exists(final_ckpt):
        print(f"  Final checkpoint already exists — skipping training")
        start_epoch, train_losses, val_losses = load_checkpoint(
            final_ckpt, model, optimizer
        )
    else:
        latest = find_latest_checkpoint(PREFIX)
        if latest:
            print(f"  Resuming from: {latest}")
            start_epoch, train_losses, val_losses = load_checkpoint(
                latest, model, optimizer
            )
        else:
            print("  Starting training from scratch")

        # ---- Training loop ----
        for epoch in range(start_epoch, config['num_epochs']):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader,
                        desc=f"[M={num_patterns}|{config['dataset']}] "
                             f"Epoch {epoch+1}/{config['num_epochs']}")
            for batch in pbar:
                buckets = batch['buckets'].to(DEVICE)
                frames  = batch['frames'].to(DEVICE)

                optimizer.zero_grad()
                pred    = model(patterns_flat, buckets)
                loss    = compute_loss(pred, frames, criterion)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})

            avg_train = epoch_loss / len(train_loader)
            train_losses.append(avg_train)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    buckets = batch['buckets'].to(DEVICE)
                    frames  = batch['frames'].to(DEVICE)
                    pred    = model(patterns_flat, buckets)
                    val_loss += criterion(pred, frames).item()
            avg_val = val_loss / len(val_loader)
            val_losses.append(avg_val)

            print(f"  Epoch {epoch+1:3d}: train={avg_train:.6f}  val={avg_val:.6f}")

            if (epoch + 1) % 5 == 0:
                ckpt_path = os.path.join(
                    CHECKPOINT_DIR, f'{PREFIX}_epoch{epoch+1}.pt'
                )
                save_checkpoint(ckpt_path, model, optimizer, epoch + 1,
                                train_losses, val_losses, config, num_patterns)

        # Save final
        save_checkpoint(final_ckpt, model, optimizer, config['num_epochs'],
                        train_losses, val_losses, config, num_patterns)

        # Loss curve
        _plot_loss_curve(train_losses, val_losses, num_patterns, config)

    # ---- Evaluate ----
    metrics = evaluate(model, val_ds, patterns_flat, config)
    sampling_ratio = num_patterns / (config['image_size'] ** 2)
    metrics['num_patterns']    = num_patterns
    metrics['sampling_ratio']  = sampling_ratio
    print(f"\n  M={num_patterns}  ratio={sampling_ratio:.4f}  "
          f"SSIM={metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}  "
          f"MSE={metrics['mse_mean']:.5f}")

    # Qualitative reconstruction grid
    visualize_reconstructions(model, patterns_flat, val_ds,
                              num_patterns, config)

    return metrics


# ============================================================================
# EVALUATION
# ============================================================================

def compute_metrics(pred, target):
    pred_np   = np.clip(pred.cpu().numpy()   if isinstance(pred,   torch.Tensor) else pred,   0, 1)
    target_np = np.clip(target.cpu().numpy() if isinstance(target, torch.Tensor) else target, 0, 1)
    return mse_metric(target_np, pred_np), ssim_metric(target_np, pred_np, data_range=1.0)


def evaluate(model, dataset, patterns_flat, config, num_eval=None) -> dict:
    n = num_eval or config['num_eval']
    model.eval()
    all_mse, all_ssim, times = [], [], []

    with torch.no_grad():
        for i in tqdm(range(min(n, len(dataset))), desc="  Evaluating"):
            sample    = dataset[i]
            buckets   = sample['buckets'].unsqueeze(0).to(DEVICE)
            frames_gt = sample['frames']

            t0          = time.time()
            pred_frames = model(patterns_flat, buckets).squeeze(0).cpu()
            times.append(time.time() - t0)

            s_mse, s_ssim = [], []
            for t in range(frames_gt.shape[0]):
                m, s = compute_metrics(pred_frames[t], frames_gt[t])
                s_mse.append(m)
                s_ssim.append(s)
            all_mse.append(np.mean(s_mse))
            all_ssim.append(np.mean(s_ssim))

    return {
        'mse_mean':     float(np.mean(all_mse)),
        'mse_std':      float(np.std(all_mse)),
        'ssim_mean':    float(np.mean(all_ssim)),
        'ssim_std':     float(np.std(all_ssim)),
        'time_mean_ms': float(np.mean(times) * 1000),
        'time_std_ms':  float(np.std(times)  * 1000),
    }


# ============================================================================
# VISUALISATION
# ============================================================================

def _plot_loss_curve(train_losses, val_losses, num_patterns, config):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'DynGhost — {config["dataset"]} — M={num_patterns}')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        f'loss_{config["dataset"]}_M{num_patterns}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved loss curve: {path}")


def visualize_reconstructions(model, patterns_flat, dataset,
                               num_patterns, config, num_samples=3):
    model.eval()
    n_cols = min(config['seq_length'], 8) // 2 * 2  # even number of cols
    t_indices = list(range(0, config['seq_length'], config['seq_length'] // (n_cols // 2)))[:n_cols // 2]

    fig, axes = plt.subplots(num_samples * 2, len(t_indices),
                              figsize=(3 * len(t_indices), 3 * num_samples))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for i in range(num_samples):
            sample      = dataset[i]
            buckets     = sample['buckets'].unsqueeze(0).to(DEVICE)
            frames_gt   = sample['frames']
            pred_frames = model(patterns_flat, buckets).squeeze(0).cpu()

            for j, t in enumerate(t_indices):
                gt_row   = i * 2
                pred_row = i * 2 + 1

                axes[gt_row,   j].imshow(frames_gt[t],   cmap='gray', vmin=0, vmax=1)
                axes[pred_row, j].imshow(pred_frames[t], cmap='gray', vmin=0, vmax=1)
                axes[gt_row,   j].set_title(f't={t} GT',   fontsize=7)
                axes[pred_row, j].set_title(f't={t} Pred', fontsize=7)
                axes[gt_row,   j].axis('off')
                axes[pred_row, j].axis('off')

    ratio = num_patterns / config['image_size'] ** 2
    plt.suptitle(f'DynGhost — {config["dataset"]} — '
                 f'M={num_patterns}  ratio={ratio:.4f}', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        f'recon_{config["dataset"]}_M{num_patterns}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved reconstructions: {path}")


def plot_sampling_ratio_curve(sweep_results: List[dict], config: dict):
    """Plot SSIM and MSE vs. sampling ratio for all trained configurations."""
    ratios = [r['sampling_ratio'] for r in sweep_results]
    ssims  = [r['ssim_mean']      for r in sweep_results]
    ssim_e = [r['ssim_std']       for r in sweep_results]
    mses   = [r['mse_mean']       for r in sweep_results]
    mse_e  = [r['mse_std']        for r in sweep_results]
    ms_pat = [r['num_patterns']   for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(ratios, ssims, yerr=ssim_e, marker='o', capsize=4, linewidth=2)
    for x, y, m in zip(ratios, ssims, ms_pat):
        ax1.annotate(f'M={m}', (x, y), textcoords='offset points',
                     xytext=(4, 4), fontsize=8)
    ax1.set_xlabel('Sampling ratio  (M / N)')
    ax1.set_ylabel('SSIM')
    ax1.set_title(f'SSIM vs. Sampling Ratio — {config["dataset"]}')
    ax1.grid(alpha=0.3)

    ax2.errorbar(ratios, mses, yerr=mse_e, marker='o', capsize=4,
                 linewidth=2, color='tab:orange')
    for x, y, m in zip(ratios, mses, ms_pat):
        ax2.annotate(f'M={m}', (x, y), textcoords='offset points',
                     xytext=(4, 4), fontsize=8)
    ax2.set_xlabel('Sampling ratio  (M / N)')
    ax2.set_ylabel('MSE')
    ax2.set_title(f'MSE vs. Sampling Ratio — {config["dataset"]}')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'sampling_curve_{config["dataset"]}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved sampling-ratio curve: {path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Train+eval DynGhost over sampling ratios'
    )
    p.add_argument('--num_patterns', type=str, default='188',
                   help='Number of patterns M. Comma-separated for a sweep, '
                        'e.g. 94,188,376,752  (default: 188)')
    p.add_argument('--pattern_type', choices=['speckle', 'hadamard'],
                   default='speckle',
                   help='Pattern type to generate (default: speckle)')
    p.add_argument('--speckle_seed', type=int, default=42,
                   help='RNG seed for random speckle generation (default: 42)')
    p.add_argument('--dataset',
                   choices=['mnist', 'cifar10', 'kvasir', 'davis'],
                   default='mnist')
    p.add_argument('--image_size',    type=int, default=256)
    p.add_argument('--seq_length',    type=int, default=8)
    p.add_argument('--batch_size',    type=int, default=4)
    p.add_argument('--num_epochs',    type=int, default=30)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--num_eval',      type=int, default=100,
                   help='Validation samples for evaluation (default: 100)')
    p.add_argument('--eval_only', action='store_true',
                   help='Skip training; only evaluate using existing checkpoints')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Explicit checkpoint path (only used with --eval_only '
                        'and a single --num_patterns value)')
    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # Parse the (possibly comma-separated) pattern counts
    pattern_counts = [int(x.strip()) for x in args.num_patterns.split(',')]
    pattern_counts.sort()

    # Apply CLI overrides to CONFIG
    for key in ('dataset', 'image_size', 'seq_length', 'batch_size',
                'num_epochs', 'learning_rate', 'num_eval', 'speckle_seed'):
        val = getattr(args, key, None)
        if val is not None:
            CONFIG[key] = val
    CONFIG['pattern_type'] = args.pattern_type

    print(f"Device:        {DEVICE}")
    print(f"Pattern type:  {CONFIG['pattern_type']}")
    print(f"Dataset:       {CONFIG['dataset']}")
    print(f"Pattern counts: {pattern_counts}")
    print(f"Image size:    {CONFIG['image_size']}  "
          f"(N={CONFIG['image_size']**2} pixels)")
    print(f"Sampling ratios: "
          + ", ".join(f"{m}/{CONFIG['image_size']**2}="
                      f"{m/CONFIG['image_size']**2:.4f}"
                      for m in pattern_counts))

    sweep_results = []

    for num_patterns in pattern_counts:
        print(f"\n{'='*60}")
        print(f"  M = {num_patterns}  "
              f"(ratio = {num_patterns / CONFIG['image_size']**2:.4f})")
        print(f"{'='*60}")

        # Generate a fresh set of random patterns for this M
        patterns = generate_patterns(
            CONFIG['pattern_type'], num_patterns,
            CONFIG['image_size'],   CONFIG['speckle_seed'],
        )

        ckpt_override = args.checkpoint if len(pattern_counts) == 1 else None

        metrics = train_one_config(
            num_patterns, patterns, CONFIG,
            eval_only=args.eval_only,
            checkpoint_override=ckpt_override,
        )
        sweep_results.append(metrics)

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print(f"{'M':>8}  {'Ratio':>8}  {'SSIM':>12}  {'MSE':>12}  {'Time(ms)':>10}")
    print(f"{'-'*70}")
    for r in sweep_results:
        print(f"{r['num_patterns']:8d}  {r['sampling_ratio']:8.4f}  "
              f"{r['ssim_mean']:6.4f}±{r['ssim_std']:.4f}  "
              f"{r['mse_mean']:6.5f}±{r['mse_std']:.5f}  "
              f"{r['time_mean_ms']:10.1f}")
    print(f"{'='*70}")

    # ---- Sampling-ratio curve (only useful with multiple M values) ----
    if len(sweep_results) > 1:
        plot_sampling_ratio_curve(sweep_results, CONFIG)

    # ---- Save JSON ----
    out_path = os.path.join(
        OUTPUT_DIR,
        f'sampling_results_{CONFIG["dataset"]}_{CONFIG["pattern_type"]}.json'
    )
    with open(out_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Saved results: {out_path}")


if __name__ == '__main__':
    main()
