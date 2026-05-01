import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as ssim_loss
import glob
import os

from models.temporal_ghost_gpt      import TemporalGhostGPT
from models.haar_temporal_ghost_gpt import HaarTemporalGhostGPT
from models.cnn_ghost               import CNNGhost
from models.unet_ghost              import UNetGhost
from models.Ghost_GPT               import GhostGPT
from datasets import MovingMNISTGhost, MovingCIFAR10Ghost, KvasirGhost

CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs('./outputs',     exist_ok=True)

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # -------------------------------------------------------------------
    # MODEL — pick one:
    #   'dynghost'  — Temporal Ghost-GPT (full model, uses speckle patterns)
    #   'haarghost' — Temporal Ghost-GPT with Haar multi-scale tokenization
    #                 (no speckle patterns needed at runtime)
    #   'ghostgpt'  — Ghost-GPT          (single-frame, uses speckle patterns)
    #   'cnn'       — CNN baseline       (no speckle patterns at runtime)
    #   'unet'      — U-Net baseline     (no speckle patterns at runtime)
    # -------------------------------------------------------------------
    'model': 'haarghost',

    # -------------------------------------------------------------------
    # DATASET — pick one:
    #   'mnist'   — Moving MNIST  (default)
    #   'cifar10' — Moving CIFAR-10
    #   'kvasir'  — Kvasir endoscopy
    # -------------------------------------------------------------------
    'dataset': 'mnist',

    # Shared hyperparameters
    'image_size':    256,
    'seq_length':    8,
    'batch_size':    4,
    'num_epochs':    30,
    'learning_rate': 3e-4,
    'weight_decay':  1e-3,

    # DynGhost-only hyperparameters (ignored for CNN / U-Net)
    'num_blocks':    8,
    'num_heads':     8,
    'embedding_dim': 32,

    # CNN-only
    'cnn_hidden_dim': 512,

    # HaarGhost-only — number of Haar decomposition levels
    'num_haar_levels': 3,
}

# ============================================================================
# HELPERS — checkpoint naming
# ============================================================================

def _ckpt_prefix(model_name, dataset_name):
    return f'{model_name}_{dataset_name}'


def find_latest_checkpoint(model_name, dataset_name):
    prefix  = _ckpt_prefix(model_name, dataset_name)
    pattern = os.path.join(CHECKPOINT_DIR, f'{prefix}_epoch*.pt')
    ckpts   = glob.glob(pattern)
    if not ckpts:
        return None

    def epoch_num(path):
        base = os.path.basename(path)
        return int(base.replace(f'{prefix}_epoch', '').replace('.pt', ''))

    return max(ckpts, key=epoch_num)

# ============================================================================
# DATASET FACTORY
# ============================================================================

def get_dataset(name, speckle_patterns, config, train):
    kwargs = dict(
        speckle_patterns=speckle_patterns,
        seq_length=config['seq_length'],
        image_size=config['image_size'],
        train=train,
    )
    if name == 'mnist':
        return MovingMNISTGhost(dataset_size=5000 if train else 500,   **kwargs)
    elif name == 'cifar10':
        return MovingCIFAR10Ghost(dataset_size=5000 if train else 500, **kwargs)
    elif name == 'kvasir':
        return KvasirGhost(dataset_size=2000 if train else 200,
                           kvasir_root='./data/kvasir-v2',
                           motion_scale=5.0, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Choose from: mnist, cifar10, kvasir")

# ============================================================================
# MODEL FACTORY
# ============================================================================

def build_model(model_name, config, num_patterns, device):
    """
    Instantiate the requested model and move it to device.
    Returns (model, uses_patterns_flat).
    uses_patterns_flat = True  → forward(patterns_flat, buckets)
    uses_patterns_flat = False → forward(buckets)
    """
    if model_name == 'dynghost':
        model = TemporalGhostGPT(
            d_in=config['embedding_dim'],
            d_out=config['embedding_dim'],
            num_blocks=config['num_blocks'],
            number_of_heads=config['num_heads'],
            embedding_dim=config['embedding_dim'],
            flattened_image_size=config['image_size'] ** 2,
            context_size=num_patterns,
            final_image_size=config['image_size'] ** 2,
            seq_length=config['seq_length'],
        ).to(device)
        return model, True

    elif model_name == 'haarghost':
        model = HaarTemporalGhostGPT(
            d_in=config['embedding_dim'],
            d_out=config['embedding_dim'],
            num_blocks=config['num_blocks'],
            number_of_heads=config['num_heads'],
            embedding_dim=config['embedding_dim'],
            num_patterns=num_patterns,
            final_image_size=config['image_size'] ** 2,
            seq_length=config['seq_length'],
            num_haar_levels=config.get('num_haar_levels', 3),
        ).to(device)
        return model, False   # does not use speckle patterns at runtime

    elif model_name == 'cnn':
        model = CNNGhost(
            num_patterns=num_patterns,
            image_size=config['image_size'],
            seq_length=config['seq_length'],
            hidden_dim=config['cnn_hidden_dim'],
        ).to(device)
        return model, False

    elif model_name == 'unet':
        model = UNetGhost(
            num_patterns=num_patterns,
            image_size=config['image_size'],
            seq_length=config['seq_length'],
        ).to(device)
        return model, False

    elif model_name == 'ghostgpt':
        model = GhostGPT(
            d_in=config['embedding_dim'],
            d_out=config['embedding_dim'],
            num_blocks=config['num_blocks'],
            number_of_heads=config['num_heads'],
            embedding_dim=config['embedding_dim'],
            flattened_image_size=config['image_size'] ** 2,
            context_size=num_patterns,
            final_image_size=config['image_size'] ** 2,
        ).to(device)
        return model, True

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: dynghost, haarghost, ghostgpt, cnn, unet")

# ============================================================================
# FORWARD PASS WRAPPER
# ============================================================================

def forward(model, uses_patterns, patterns_flat, buckets, model_name=None, image_size=None):
    """
    Unified forward call regardless of model type.
    Always returns [B, T, H, W].

    GhostGPT is a single-frame model: we loop over the T dimension,
    run the model for each frame, then stack the results.
    """
    if uses_patterns:
        if model_name == 'ghostgpt':
            B, T, N = buckets.shape
            H = W = image_size
            frames = [model(patterns_flat, buckets[:, t, :]) for t in range(T)]
            return torch.stack(frames, dim=1).view(B, T, H, W)
        return model(patterns_flat, buckets)
    else:
        return model(buckets)

# ============================================================================
# LOSS
# ============================================================================

def compute_loss(pred_frames, frames, criterion, uses_temporal_loss=True):
    """
    Combined MSE + SSIM + optional temporal consistency loss.
    Identical for all three models.
    """
    mse_val  = criterion(pred_frames, frames)
    ssim_val = 1 - ssim_loss(pred_frames, frames,
                              data_range=1.0, size_average=True)
    loss = mse_val + 0.5 * ssim_val

    if uses_temporal_loss and pred_frames.shape[1] > 1:
        pred_diff = pred_frames[:, 1:] - pred_frames[:, :-1]
        true_diff = frames[:, 1:]      - frames[:, :-1]
        loss = loss + 0.1 * criterion(pred_diff, true_diff)

    return loss

# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def save_checkpoint(path, model, optimizer, epoch,
                    train_losses, val_losses, config):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses':         train_losses,
        'val_losses':           val_losses,
        'model':                config['model'],
        'dataset':              config['dataset'],
        'config':               config,
    }, path)
    print(f"Saved: {path}")


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch  = ckpt['epoch']
        train_losses = ckpt.get('train_losses', [])
        val_losses   = ckpt.get('val_losses',   [])
        print(f"Resuming from epoch {start_epoch + 1} "
              f"[model={ckpt.get('model','?')}, "
              f"dataset={ckpt.get('dataset','?')}]")
    else:
        # Legacy format — weights only
        model.load_state_dict(ckpt)
        start_epoch  = 0
        train_losses = []
        val_losses   = []
        print("Loaded legacy weights (epoch unknown), starting from epoch 1")
    return start_epoch, train_losses, val_losses

# ============================================================================
# VISUALISATION
# ============================================================================

def visualize_results(model, uses_patterns, patterns_flat,
                      dataset, device, model_name, dataset_name,
                      image_size=None, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 8, figsize=(16, 2 * num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            sample      = dataset[i]
            buckets     = sample['buckets'].unsqueeze(0).to(device)
            frames_gt   = sample['frames']
            pred_frames = forward(model, uses_patterns, patterns_flat, buckets,
                                  model_name=model_name,
                                  image_size=image_size).squeeze(0).cpu()

            for j, t in enumerate([0, 2, 4, 6]):
                axes[i, j*2].imshow(frames_gt[t],    cmap='gray')
                axes[i, j*2].set_title(f't={t} GT')
                axes[i, j*2].axis('off')

                axes[i, j*2+1].imshow(pred_frames[t], cmap='gray')
                axes[i, j*2+1].set_title(f't={t} Pred')
                axes[i, j*2+1].axis('off')

    plt.suptitle(f'{model_name} — {dataset_name}', fontsize=12)
    plt.tight_layout()
    save_path = f'outputs/reconstruction_{model_name}_{dataset_name}.png'
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train():
    MODEL_NAME   = CONFIG['model']
    DATASET_NAME = CONFIG['dataset']
    PREFIX       = _ckpt_prefix(MODEL_NAME, DATASET_NAME)
    DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Model:   {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Device:  {DEVICE}")

    # ---- Speckle patterns ----
    SPECKLE_PATH = 'data/speckle_pattern.pt'
    speckle_patterns = torch.load(SPECKLE_PATH)
    if isinstance(speckle_patterns, torch.Tensor):
        speckle_patterns = speckle_patterns.numpy()
    elif isinstance(speckle_patterns, dict):
        speckle_patterns = speckle_patterns['patterns']
        if isinstance(speckle_patterns, torch.Tensor):
            speckle_patterns = speckle_patterns.numpy()

    num_patterns  = speckle_patterns.shape[0]
    CONFIG['context_size'] = num_patterns
    print(f"Speckle patterns: {speckle_patterns.shape}")

    patterns_flat = (torch.tensor(speckle_patterns).float()
                     .view(num_patterns, -1).to(DEVICE))

    # ---- Datasets ----
    print("Creating datasets...")
    train_dataset = get_dataset(DATASET_NAME, speckle_patterns, CONFIG, train=True)
    val_dataset   = get_dataset(DATASET_NAME, speckle_patterns, CONFIG, train=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

    # ---- Model ----
    print("Building model...")
    model, uses_patterns = build_model(MODEL_NAME, CONFIG, num_patterns, DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.2f}M")

    # ---- Optimiser ----
    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG['learning_rate'],
                            weight_decay=CONFIG['weight_decay'])
    criterion = nn.MSELoss()

    # ---- Resume from checkpoint if available ----
    start_epoch  = 0
    train_losses = []
    val_losses   = []

    latest = find_latest_checkpoint(MODEL_NAME, DATASET_NAME)
    if latest:
        print(f"Found checkpoint: {latest}")
        start_epoch, train_losses, val_losses = load_checkpoint(
            latest, model, optimizer, DEVICE
        )
    else:
        print("No checkpoint found — starting from scratch")

    # ---- Training loop ----
    for epoch in range(start_epoch, CONFIG['num_epochs']):

        # --- Train ---
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"[{MODEL_NAME}|{DATASET_NAME}] "
                         f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

        for batch in pbar:
            buckets = batch['buckets'].to(DEVICE)
            frames  = batch['frames'].to(DEVICE)

            optimizer.zero_grad()
            pred_frames = forward(model, uses_patterns, patterns_flat, buckets,
                                  model_name=MODEL_NAME,
                                  image_size=CONFIG['image_size'])
            loss = compute_loss(pred_frames, frames, criterion)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.5f}'})

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                buckets = batch['buckets'].to(DEVICE)
                frames  = batch['frames'].to(DEVICE)
                pred_frames = forward(model, uses_patterns, patterns_flat, buckets,
                                      model_name=MODEL_NAME,
                                      image_size=CONFIG['image_size'])
                val_loss += criterion(pred_frames, frames).item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"Epoch {epoch+1:3d}: "
              f"train={avg_train:.6f}  val={avg_val:.6f}")

        # --- Periodic checkpoint ---
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(
                CHECKPOINT_DIR, f'{PREFIX}_epoch{epoch+1}.pt'
            )
            save_checkpoint(ckpt_path, model, optimizer, epoch + 1,
                            train_losses, val_losses, CONFIG)

    # ---- Final model ----
    final_path = os.path.join(CHECKPOINT_DIR, f'{PREFIX}_final.pt')
    save_checkpoint(final_path, model, optimizer, CONFIG['num_epochs'],
                    train_losses, val_losses, CONFIG)

    # ---- Loss curve ----
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch');  plt.ylabel('Loss')
    plt.title(f'Loss — {MODEL_NAME} on {DATASET_NAME}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'outputs/loss_{MODEL_NAME}_{DATASET_NAME}.png')
    plt.show()

    # ---- Qualitative results ----
    visualize_results(model, uses_patterns, patterns_flat,
                      val_dataset, DEVICE, MODEL_NAME, DATASET_NAME,
                      image_size=CONFIG['image_size'])


if __name__ == "__main__":
    train()