import argparse
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
from models.cnn_ghost               import CNNGhost
from models.unet_ghost              import UNetGhost
from models.Ghost_GPT               import GhostGPT
from models.dynghost_diff           import DynGhostDiff
from datasets import MovingMNISTGhost, MovingCIFAR10Ghost, KvasirGhost, DAVISGhost
from generate_patterns import make_hadamard_s_patterns

CHECKPOINT_DIR       = './checkpoints'
CHECKPOINT_DIR_S1    = './checkpoints/stage1'
CHECKPOINT_DIR_S2    = './checkpoints/stage2'
os.makedirs(CHECKPOINT_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR_S1, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_S2, exist_ok=True)
os.makedirs('./outputs',        exist_ok=True)

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    # -------------------------------------------------------------------
    # MODEL — pick one:
    #   'dynghost'      — Temporal Ghost-GPT (full model, uses speckle patterns)
    #   'haarghost'     — Temporal Ghost-GPT with Haar multi-scale tokenization
    #                     (no speckle patterns needed at runtime)
    #   'fistadynghost' — DynGhost + early-fusion DGI warm-start (patch tokens)
    #   'ghostgpt'      — Ghost-GPT          (single-frame, uses speckle patterns)
    #   'cnn'           — CNN baseline       (no speckle patterns at runtime)
    #   'unet'          — U-Net baseline     (no speckle patterns at runtime)
    # -------------------------------------------------------------------
    'model': 'dynghost',

    # -------------------------------------------------------------------
    # DATASET — pick one:
    #   'mnist'   — Moving MNIST  (default)
    #   'cifar10' — Moving CIFAR-10
    #   'kvasir'  — Kvasir endoscopy
    #   'davis'   — DAVIS video sequences
    # -------------------------------------------------------------------
    'dataset': 'kvasir',

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

    # FISTAWarmDynGhost-only — patch size for warm-start patch embedding
    # 256/32 = 8 → 64 patches; must divide image_size evenly
    'patch_size': 32,

    # DynGhostDiff-only
    'diffusion_steps': 1000,   # DDPM total timesteps T
    'ddim_steps':        20,   # DDIM inference steps
    'stage1_epochs':     15,   # epochs for deterministic pre-training
    'stage2_epochs':     15,   # epochs for diffusion head fine-tuning

    # -------------------------------------------------------------------
    # MEASUREMENT PATTERNS — pick one:
    #   'speckle'  — random binary speckle (loaded from data/speckle_pattern.pt)
    #   'hadamard' — Hadamard S-matrix rows (generated on-the-fly or from
    #                data/hadamard_pattern.pt if it exists)
    # -------------------------------------------------------------------
    'pattern_type': 'hadamard',
}

# ============================================================================
# HELPERS — checkpoint naming
# ============================================================================

def _ckpt_prefix(model_name, dataset_name, pattern_type='speckle'):
    return f'{model_name}_{dataset_name}_{pattern_type}'


def find_latest_checkpoint(model_name, dataset_name):
    prefix  = _ckpt_prefix(model_name, dataset_name, CONFIG.get('pattern_type', 'speckle'))
    pattern = os.path.join(CHECKPOINT_DIR, f'{prefix}_epoch*.pt')
    ckpts   = glob.glob(pattern)
    if not ckpts:
        return None

    def epoch_num(path):
        base = os.path.basename(path)
        return int(base.replace(f'{prefix}_epoch', '').replace('.pt', ''))

    return max(ckpts, key=epoch_num)

# ============================================================================
# PATTERN LOADING
# ============================================================================

def load_patterns(pattern_type: str, image_size: int) -> np.ndarray:
    """
    Load or generate measurement patterns.

    Returns numpy array of shape (M, image_size, image_size), values in {0,1}.
    """
    if pattern_type == 'hadamard':
        hadamard_path = 'data/hadamard_pattern.pt'
        if os.path.exists(hadamard_path):
            patterns = torch.load(hadamard_path, weights_only=False)
            if isinstance(patterns, torch.Tensor):
                patterns = patterns.numpy()
            elif isinstance(patterns, dict):
                patterns = patterns['patterns']
                if isinstance(patterns, torch.Tensor):
                    patterns = patterns.numpy()
            print(f"Loaded Hadamard patterns from {hadamard_path}: {patterns.shape}")
        else:
            # Load speckle to get M, then generate equivalent Hadamard patterns
            speckle = torch.load('data/speckle_pattern.pt', weights_only=False)
            M = (speckle.shape[0] if isinstance(speckle, torch.Tensor)
                 else speckle['patterns'].shape[0])
            print(f"Generating Hadamard S-matrix patterns: M={M}, image_size={image_size}")
            N = image_size ** 2
            if (N & (N - 1)) != 0:
                raise ValueError(
                    f"image_size={image_size} gives N={N}, which is not a power of 2. "
                    f"Hadamard patterns require N = power of 2 (32, 64, 128, 256)."
                )
            patterns = make_hadamard_s_patterns(image_size, M)
            torch.save(torch.tensor(patterns), hadamard_path)
            print(f"Saved generated patterns to {hadamard_path}")
        return patterns

    else:  # speckle (default)
        speckle = torch.load('data/speckle_pattern.pt', weights_only=False)
        if isinstance(speckle, torch.Tensor):
            patterns = speckle.numpy()
        elif isinstance(speckle, dict):
            patterns = speckle['patterns']
            if isinstance(patterns, torch.Tensor):
                patterns = patterns.numpy()
        else:
            patterns = speckle
        print(f"Loaded speckle patterns: {patterns.shape}")
        return patterns


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
    elif name == 'davis':
        return DAVISGhost(
            davis_root   = './data/DAVIS',
            speckle_patterns = speckle_patterns,
            seq_length   = config['seq_length'],
            image_size   = config['image_size'],
            train        = train,
            dataset_size = 2000 if train else 200,
        )
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

    elif model_name == 'fistadynghost':
        model = FISTAWarmDynGhost(
            d_in=config['embedding_dim'],
            d_out=config['embedding_dim'],
            num_blocks=config['num_blocks'],
            number_of_heads=config['num_heads'],
            embedding_dim=config['embedding_dim'],
            flattened_image_size=config['image_size'] ** 2,
            context_size=num_patterns,
            final_image_size=config['image_size'] ** 2,
            seq_length=config['seq_length'],
            patch_size=config.get('patch_size', 32),
        ).to(device)
        return model, True   # uses speckle patterns (via warm-start computation)

    elif model_name == 'dynghost_diff':
        model = DynGhostDiff(
            d_in=config['embedding_dim'],
            d_out=config['embedding_dim'],
            num_blocks=config['num_blocks'],
            number_of_heads=config['num_heads'],
            embedding_dim=config['embedding_dim'],
            flattened_image_size=config['image_size'] ** 2,
            context_size=num_patterns,
            final_image_size=config['image_size'] ** 2,
            seq_length=config['seq_length'],
            diffusion=False,                             # start in deterministic mode
            diffusion_steps=config.get('diffusion_steps', 1000),
        ).to(device)
        return model, True

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: dynghost, haarghost, fistadynghost, "
                         f"ghostgpt, cnn, unet, dynghost_diff")

# ============================================================================
# FORWARD PASS WRAPPER
# ============================================================================

def forward(model, uses_patterns, patterns_flat, buckets, model_name=None, image_size=None):
    """
    Unified forward call regardless of model type.
    Always returns [B, T, H, W].

    GhostGPT is a single-frame model: we loop over the T dimension,
    run the model for each frame, then stack the results.

    FISTAWarmDynGhost computes a DGI warm-start on-the-fly then calls
    model(patterns_flat, buckets, warm_start).
    """
    if uses_patterns:
        if model_name == 'ghostgpt':
            B, T, N = buckets.shape
            H = W = image_size
            frames = [model(patterns_flat, buckets[:, t, :]) for t in range(T)]
            return torch.stack(frames, dim=1).view(B, T, H, W)
        if model_name == 'fistadynghost':
            H = W = image_size
            warm = compute_warm_start(patterns_flat, buckets, H, W)
            return model(patterns_flat, buckets, warm)
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
    PATTERN_TYPE = CONFIG.get('pattern_type', 'speckle')
    PREFIX       = _ckpt_prefix(MODEL_NAME, DATASET_NAME, PATTERN_TYPE)
    DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Model:        {MODEL_NAME}")
    print(f"Dataset:      {DATASET_NAME}")
    print(f"Device:       {DEVICE}")
    print(f"Pattern type: {PATTERN_TYPE}")

    # ---- Measurement patterns ----
    speckle_patterns = load_patterns(PATTERN_TYPE, CONFIG['image_size'])
    num_patterns  = speckle_patterns.shape[0]
    CONFIG['context_size'] = num_patterns

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


# ============================================================================
# TWO-STAGE TRAINING FOR DynGhostDiff
# ============================================================================

def _x0_from_eps(model, x_taus, eps_preds, timesteps):
    """
    Compute predicted x0 per frame from the stored (x_tau, eps_pred, t) tuples
    without running the U-Net a second time.

    x_taus    : [B, T, 1, H, W]
    eps_preds : [B, T, 1, H, W]
    timesteps : [B, T]

    Returns x0_preds [B, T, H, W] in [0, 1].
    """
    B, T = timesteps.shape
    x0_list = []
    for t in range(T):
        tau_t       = timesteps[:, t]                              # [B]
        x_tau_t     = x_taus[:, t]                                # [B, 1, H, W]
        eps_pred_t  = eps_preds[:, t]                             # [B, 1, H, W]
        sqrt_recip  = model.sqrt_recip_alphas_cumprod[tau_t].view(-1, 1, 1, 1)
        sqrt_recm1  = model.sqrt_recipm1_alphas_cumprod[tau_t].view(-1, 1, 1, 1)
        x0_t        = sqrt_recip * x_tau_t - sqrt_recm1 * eps_pred_t
        x0_t        = x0_t.clamp(-1, 1).squeeze(1)               # [B, H, W]
        x0_list.append(x0_t)
    x0_preds = torch.stack(x0_list, dim=1)                       # [B, T, H, W]
    return (x0_preds + 1) / 2                                    # rescale to [0, 1]


def train_dynghost_diff():
    """Two-stage training for DynGhostDiff."""
    MODEL_NAME   = 'dynghost_diff'
    DATASET_NAME = CONFIG['dataset']
    DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

    PATTERN_TYPE = CONFIG.get('pattern_type', 'speckle')
    print(f"Model:        {MODEL_NAME} (two-stage diffusion training)")
    print(f"Dataset:      {DATASET_NAME}")
    print(f"Device:       {DEVICE}")
    print(f"Pattern type: {PATTERN_TYPE}")

    # ---- Measurement patterns ----
    speckle_patterns = load_patterns(PATTERN_TYPE, CONFIG['image_size'])
    num_patterns = speckle_patterns.shape[0]
    CONFIG['context_size'] = num_patterns

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

    # ---- Model (starts in deterministic mode) ----
    print("Building DynGhostDiff...")
    model = DynGhostDiff(
        d_in=CONFIG['embedding_dim'],
        d_out=CONFIG['embedding_dim'],
        num_blocks=CONFIG['num_blocks'],
        number_of_heads=CONFIG['num_heads'],
        embedding_dim=CONFIG['embedding_dim'],
        flattened_image_size=CONFIG['image_size'] ** 2,
        context_size=num_patterns,
        final_image_size=CONFIG['image_size'] ** 2,
        seq_length=CONFIG['seq_length'],
        diffusion=False,
        diffusion_steps=CONFIG.get('diffusion_steps', 1000),
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Stage-1 parameters: {num_params / 1e6:.2f}M")

    criterion = nn.MSELoss()

    PREFIX_S1 = f'{MODEL_NAME}_{DATASET_NAME}_{PATTERN_TYPE}'

    # =========================================================================
    # STAGE 1 — deterministic pre-training (MSE + SSIM + temporal consistency)
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 1 — deterministic head ({CONFIG['stage1_epochs']} epochs)")
    print(f"{'='*60}")

    optimizer_s1 = optim.AdamW(model.parameters(),
                                lr=CONFIG['learning_rate'],
                                weight_decay=CONFIG['weight_decay'])

    # Resume stage-1 if a checkpoint exists
    start_epoch  = 0
    train_losses = []
    val_losses   = []
    latest_s1 = find_latest_checkpoint(PREFIX_S1, DATASET_NAME)
    # find_latest_checkpoint looks in CHECKPOINT_DIR; stage1 has its own dir
    latest_s1_direct = None
    s1_pattern = os.path.join(CHECKPOINT_DIR_S1, f'{PREFIX_S1}_epoch*.pt')
    s1_ckpts   = glob.glob(s1_pattern)
    if s1_ckpts:
        def _epoch_num(p):
            base = os.path.basename(p)
            return int(base.replace(f'{PREFIX_S1}_epoch', '').replace('.pt', ''))
        latest_s1_direct = max(s1_ckpts, key=_epoch_num)
    if latest_s1_direct:
        start_epoch, train_losses, val_losses = load_checkpoint(
            latest_s1_direct, model, optimizer_s1, DEVICE
        )

    for epoch in range(start_epoch, CONFIG['stage1_epochs']):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"[S1|{DATASET_NAME}] Epoch {epoch+1}/{CONFIG['stage1_epochs']}")
        for batch in pbar:
            buckets = batch['buckets'].to(DEVICE)
            frames  = batch['frames'].to(DEVICE)

            optimizer_s1.zero_grad()
            pred_frames = model(patterns_flat, buckets)
            loss = compute_loss(pred_frames, frames, criterion)
            loss.backward()
            optimizer_s1.step()

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
                pred_frames = model(patterns_flat, buckets)
                val_loss += criterion(pred_frames, frames).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"Stage-1 Epoch {epoch+1:3d}: train={avg_train:.6f}  val={avg_val:.6f}")

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(
                CHECKPOINT_DIR_S1, f'{PREFIX_S1}_epoch{epoch+1}.pt'
            )
            save_checkpoint(ckpt_path, model, optimizer_s1, epoch + 1,
                            train_losses, val_losses, CONFIG)

    s1_final = os.path.join(CHECKPOINT_DIR_S1, f'{PREFIX_S1}_final.pt')
    save_checkpoint(s1_final, model, optimizer_s1, CONFIG['stage1_epochs'],
                    train_losses, val_losses, CONFIG)

    # =========================================================================
    # STAGE 2 — diffusion head (backbone at 10x lower LR, U-Net at full LR)
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 2 — diffusion head ({CONFIG['stage2_epochs']} epochs)")
    print(f"{'='*60}")

    # Switch to diffusion mode: register schedule and add diffusion modules.
    # Call model.to(DEVICE) afterwards so newly registered buffers land on GPU.
    model.diffusion = True
    model._register_schedule(CONFIG.get('diffusion_steps', 1000))

    from models.dynghost_diff import (SpatialConditioningProjection,
                                    ConditionalUNet)
    COND_DIM     = 256
    SPATIAL_SIZE = 16

    model.cond_proj = SpatialConditioningProjection(
        d_out=CONFIG['embedding_dim'],
        cond_dim=COND_DIM,
        spatial_size=SPATIAL_SIZE,
    )
    model.cond_proj_global = nn.Sequential(
        nn.LayerNorm(CONFIG['embedding_dim']),
        nn.Linear(CONFIG['embedding_dim'], COND_DIM),
    )
    model.unet = ConditionalUNet(
        in_ch=1, base_ch=64, cond_dim=COND_DIM,
        time_dim=128, spatial_size=SPATIAL_SIZE,
    )
    model.to(DEVICE)
    num_params_s2 = sum(p.numel() for p in model.parameters())
    print(f"Stage-2 parameters: {num_params_s2 / 1e6:.2f}M")

    # Separate parameter groups: backbone at 10x lower LR
    backbone_params  = [p for name, p in model.named_parameters()
                        if not name.startswith(('cond_proj', 'unet'))]
    diffhead_params  = [p for name, p in model.named_parameters()
                        if name.startswith(('cond_proj', 'unet'))]
    optimizer_s2 = optim.AdamW([
        {'params': backbone_params, 'lr': CONFIG['learning_rate'] / 10},
        {'params': diffhead_params, 'lr': CONFIG['learning_rate']},
    ], weight_decay=CONFIG['weight_decay'])

    # Resume stage-2 if a checkpoint exists
    start_epoch  = 0
    train_losses = []
    val_losses   = []
    s2_pattern = os.path.join(CHECKPOINT_DIR_S2, f'{PREFIX_S1}_epoch*.pt')
    s2_ckpts   = glob.glob(s2_pattern)
    if s2_ckpts:
        latest_s2 = max(s2_ckpts, key=lambda p: int(
            os.path.basename(p).replace(f'{PREFIX_S1}_epoch', '').replace('.pt', '')
        ))
        start_epoch, train_losses, val_losses = load_checkpoint(
            latest_s2, model, optimizer_s2, DEVICE
        )

    for epoch in range(start_epoch, CONFIG['stage2_epochs']):
        # --- Train ---
        model.train()
        epoch_diff  = 0.0
        epoch_ssim  = 0.0
        epoch_total = 0.0
        pbar = tqdm(train_loader,
                    desc=f"[S2|{DATASET_NAME}] Epoch {epoch+1}/{CONFIG['stage2_epochs']}")

        for batch in pbar:
            buckets = batch['buckets'].to(DEVICE)
            frames  = batch['frames'].to(DEVICE)

            optimizer_s2.zero_grad()

            # Diffusion forward: predict noise
            eps_preds, noises, x_taus, timesteps, conds = model.forward_diffusion(
                patterns_flat, buckets, frames
            )

            # L_diff = MSE between predicted and actual noise
            l_diff = criterion(eps_preds, noises)

            # Predict x0 from already-computed eps (no extra U-Net call)
            x0_preds = _x0_from_eps(model, x_taus, eps_preds, timesteps)

            # SSIM and temporal consistency on predicted x0
            ssim_val = 1 - ssim_loss(x0_preds, frames,
                                     data_range=1.0, size_average=True)
            if frames.shape[1] > 1:
                x0_diff   = x0_preds[:, 1:] - x0_preds[:, :-1]
                true_diff = frames[:, 1:]    - frames[:, :-1]
                temp_cons = criterion(x0_diff, true_diff)
            else:
                temp_cons = torch.tensor(0.0, device=DEVICE)

            loss = l_diff + 0.1 * ssim_val + 0.1 * temp_cons
            loss.backward()
            optimizer_s2.step()

            epoch_diff  += l_diff.item()
            epoch_ssim  += (1 - ssim_val.item())   # actual SSIM value
            epoch_total += loss.item()
            pbar.set_postfix({
                'L_diff': f'{l_diff.item():.5f}',
                'SSIM':   f'{1 - ssim_val.item():.4f}',
            })

        n = len(train_loader)
        avg_diff  = epoch_diff  / n
        avg_ssim  = epoch_ssim  / n
        avg_total = epoch_total / n
        train_losses.append(avg_total)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                buckets = batch['buckets'].to(DEVICE)
                frames  = batch['frames'].to(DEVICE)
                eps_preds, noises, x_taus, timesteps, _ = model.forward_diffusion(
                    patterns_flat, buckets, frames
                )
                val_loss += criterion(eps_preds, noises).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"Stage-2 Epoch {epoch+1:3d}: "
              f"L_diff={avg_diff:.6f}  SSIM_x0={avg_ssim:.4f}  "
              f"total={avg_total:.6f}  val_diff={avg_val:.6f}")

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(
                CHECKPOINT_DIR_S2, f'{PREFIX_S1}_epoch{epoch+1}.pt'
            )
            save_checkpoint(ckpt_path, model, optimizer_s2, epoch + 1,
                            train_losses, val_losses, CONFIG)

    s2_final = os.path.join(CHECKPOINT_DIR_S2, f'{PREFIX_S1}_final.pt')
    save_checkpoint(s2_final, model, optimizer_s2, CONFIG['stage2_epochs'],
                    train_losses, val_losses, CONFIG)

    # ---- Qualitative results using DDIM inference ----
    visualize_results(model, True, patterns_flat,
                      val_dataset, DEVICE, MODEL_NAME, DATASET_NAME,
                      image_size=CONFIG['image_size'])


# ============================================================================
# ARGPARSE — allows overriding CONFIG fields from the command line
# ============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(description='Train DynGhost / DynGhostDiff')
    parser.add_argument('--model',           type=str,
                        choices=['dynghost', 'haarghost', 'ghostgpt',
                                 'cnn', 'unet', 'fistadynghost', 'dynghost_diff'],
                        help='Model to train (overrides CONFIG[model])')
    parser.add_argument('--diffusion_steps', type=int,
                        help='DDPM total timesteps T (default 1000)')
    parser.add_argument('--ddim_steps',      type=int,
                        help='DDIM inference steps (default 20)')
    parser.add_argument('--stage1_epochs',   type=int,
                        help='Epochs for Stage-1 deterministic pre-training (default 15)')
    parser.add_argument('--stage2_epochs',   type=int,
                        help='Epochs for Stage-2 diffusion fine-tuning (default 15)')
    parser.add_argument('--num_epochs',      type=int,
                        help='Total epochs for single-stage training (default 30)')
    parser.add_argument('--batch_size',      type=int,
                        help='Batch size')
    parser.add_argument('--learning_rate',   type=float,
                        help='Learning rate')
    parser.add_argument('--dataset', type=str,
                        choices=['mnist', 'cifar10', 'kvasir', 'davis'],
                        help='Dataset')
    parser.add_argument('--pattern_type', type=str,
                        choices=['speckle', 'hadamard'],
                        help='Measurement pattern type (default: speckle)')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Apply CLI overrides to CONFIG
    for key in ('model', 'dataset', 'pattern_type', 'diffusion_steps', 'ddim_steps',
                'stage1_epochs', 'stage2_epochs', 'num_epochs',
                'batch_size', 'learning_rate'):
        val = getattr(args, key, None)
        if val is not None:
            CONFIG[key] = val

    if CONFIG['model'] == 'dynghost_diff':
        train_dynghost_diff()
    else:
        train()