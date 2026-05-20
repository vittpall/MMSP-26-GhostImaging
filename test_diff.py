import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import mean_squared_error as mse_fn
from models.dynghost_diff import DynGhostDiff
from datasets import KvasirGhost

# ============================================================================
# CONFIG
# ============================================================================

CKPT_S2       = 'checkpoints/stage2/dynghost_diff_kvasir_final.pt'
SPECKLE_PATH  = 'data/speckle_pattern.pt'
KVASIR_ROOT   = './data/kvasir-v2'
OUTPUT_DIR    = 'outputs/spatial_debug'
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_SAMPLES   = 5     # number of validation sequences to test
START_T       = 400   # warm-start timestep
DDIM_STEPS    = 20
SPATIAL_SIZE  = 16

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOAD
# ============================================================================

print("Loading speckle patterns...")
speckle = torch.load(SPECKLE_PATH)
if isinstance(speckle, torch.Tensor):
    speckle = speckle.numpy()
num_patterns = speckle.shape[0]
patterns_flat = torch.tensor(speckle).float().view(num_patterns, -1).to(DEVICE)

print("Loading dataset...")
dataset = KvasirGhost(
    dataset_size=50,
    kvasir_root=KVASIR_ROOT,
    speckle_patterns=speckle,
    seq_length=8, image_size=256,
    train=False, motion_scale=5.0,
)

print("Loading model...")
model = DynGhostDiff(
    d_in=32, d_out=32, num_blocks=8, number_of_heads=8,
    embedding_dim=32, flattened_image_size=256*256,
    context_size=num_patterns, final_image_size=256*256,
    seq_length=8, diffusion=True, diffusion_steps=1000,
    spatial_size=SPATIAL_SIZE,
).to(DEVICE)

ckpt = torch.load(CKPT_S2, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  Loaded epoch {ckpt.get('epoch','?')}")

# ============================================================================
# TEST 1 — artifact source: is it in the deterministic head?
# ============================================================================

print("\n[Test 1] Checking if artifact is in deterministic head...")

fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(9, 3 * NUM_SAMPLES))
det_ssims = []

with torch.no_grad():
    for i in range(NUM_SAMPLES):
        sample  = dataset[i]
        buckets = sample['buckets'].unsqueeze(0).to(DEVICE)
        gt      = sample['frames']          # [T, H, W]

        # Deterministic head only
        model.diffusion = False
        det_out = model(patterns_flat, buckets).squeeze(0).cpu()
        model.diffusion = True

        t_mid = gt.shape[0] // 2
        s = ssim_fn(det_out[t_mid].numpy(), gt[t_mid].numpy(), data_range=1.0)
        det_ssims.append(s)

        axes[i, 0].imshow(gt[t_mid].numpy(),       cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('GT',                 fontsize=8)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(det_out[t_mid].numpy(),  cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Det (SSIM={s:.3f})', fontsize=8)
        axes[i, 1].axis('off')

        # Difference map to make artifact visible
        diff = np.abs(det_out[t_mid].numpy() - gt[t_mid].numpy())
        im = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[i, 2].set_title('|Det - GT|',         fontsize=8)
        axes[i, 2].axis('off')

plt.colorbar(im, ax=axes[:, 2])
plt.suptitle('Test 1: Deterministic head output — artifact visible here?',
             fontsize=10, fontweight='bold')
plt.tight_layout()
path = f'{OUTPUT_DIR}/test1_deterministic_artifact.png'
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Mean det SSIM: {np.mean(det_ssims):.4f}")
print(f"  Saved: {path}")

# ============================================================================
# TEST 2 — spatial map energy: which cells are over-activating?
# ============================================================================

print("\n[Test 2] Visualizing spatial conditioning map energy...")

fig, axes = plt.subplots(2, NUM_SAMPLES, figsize=(3 * NUM_SAMPLES, 7))

with torch.no_grad():
    for i in range(NUM_SAMPLES):
        sample  = dataset[i]
        buckets = sample['buckets'].unsqueeze(0).to(DEVICE)
        gt      = sample['frames']

        tokens    = model._backbone(patterns_flat, buckets)   # [1, T, M, d_out]
        c_spatial = model.cond_proj(tokens[:, 0])             # [1, cond_dim, S, S]

        # Cell energy: L2 norm across cond_dim per spatial cell
        cell_energy = c_spatial.norm(dim=1).squeeze(0).cpu().numpy()  # [S, S]

        # Also show raw max activation
        cell_max = c_spatial.abs().max(dim=1).values.squeeze(0).cpu().numpy()

        im0 = axes[0, i].imshow(cell_energy, cmap='hot',
                                interpolation='nearest')
        axes[0, i].set_title(f'L2 norm\nsample {i}', fontsize=8)
        axes[0, i].axis('off')
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046)

        im1 = axes[1, i].imshow(cell_max, cmap='hot',
                                interpolation='nearest')
        axes[1, i].set_title(f'Max abs\nsample {i}', fontsize=8)
        axes[1, i].axis('off')
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046)

        # Print max cell location
        max_cell = np.unravel_index(cell_energy.argmax(), cell_energy.shape)
        print(f"  Sample {i}: max energy at cell {max_cell}, "
              f"value={cell_energy.max():.3f}, "
              f"mean={cell_energy.mean():.3f}, "
              f"ratio={cell_energy.max()/cell_energy.mean():.1f}x")

plt.suptitle(f'Test 2: Spatial map energy ({SPATIAL_SIZE}×{SPATIAL_SIZE} grid)\n'
             f'Hot cells in same location = artifact source',
             fontsize=10, fontweight='bold')
plt.tight_layout()
path = f'{OUTPUT_DIR}/test2_spatial_map_energy.png'
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")

# ============================================================================
# TEST 3 — diffusion output vs deterministic: where does artifact appear/worsen?
# ============================================================================

print("\n[Test 3] Comparing deterministic vs diffusion output...")

fig, axes = plt.subplots(NUM_SAMPLES, 5, figsize=(15, 3 * NUM_SAMPLES))
diff_ssims, det_ssims2 = [], []

with torch.no_grad():
    for i in range(NUM_SAMPLES):
        sample  = dataset[i]
        buckets = sample['buckets'].unsqueeze(0).to(DEVICE)
        gt      = sample['frames']
        t_mid   = gt.shape[0] // 2

        # Deterministic
        model.diffusion = False
        det_out = model(patterns_flat, buckets).squeeze(0).cpu()
        model.diffusion = True

        # Diffusion (warm start)
        diff_out = model(patterns_flat, buckets,
                         num_steps=DDIM_STEPS,
                         start_t=START_T).squeeze(0).cpu()

        s_det  = ssim_fn(det_out[t_mid].numpy(),  gt[t_mid].numpy(), data_range=1.0)
        s_diff = ssim_fn(diff_out[t_mid].numpy(), gt[t_mid].numpy(), data_range=1.0)
        det_ssims2.append(s_det)
        diff_ssims.append(s_diff)

        axes[i, 0].imshow(gt[t_mid].numpy(),        cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('GT',                  fontsize=8)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(det_out[t_mid].numpy(),   cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Det  SSIM={s_det:.3f}', fontsize=8)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(diff_out[t_mid].numpy(),  cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Diff SSIM={s_diff:.3f}', fontsize=8)
        axes[i, 2].axis('off')

        diff_map_det  = np.abs(det_out[t_mid].numpy()  - gt[t_mid].numpy())
        diff_map_diff = np.abs(diff_out[t_mid].numpy() - gt[t_mid].numpy())

        axes[i, 3].imshow(diff_map_det,  cmap='hot', vmin=0, vmax=0.5)
        axes[i, 3].set_title('|Det - GT|',  fontsize=8)
        axes[i, 3].axis('off')

        axes[i, 4].imshow(diff_map_diff, cmap='hot', vmin=0, vmax=0.5)
        axes[i, 4].set_title('|Diff - GT|', fontsize=8)
        axes[i, 4].axis('off')

plt.suptitle('Test 3: Deterministic vs diffusion — does artifact grow or shrink?',
             fontsize=10, fontweight='bold')
plt.tight_layout()
path = f'{OUTPUT_DIR}/test3_det_vs_diff.png'
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Mean det  SSIM: {np.mean(det_ssims2):.4f}")
print(f"  Mean diff SSIM: {np.mean(diff_ssims):.4f}")
print(f"  Saved: {path}")

# ============================================================================
# TEST 4 — start_t sweep: how much does start_t affect the artifact?
# ============================================================================

print("\n[Test 4] start_t sweep on one sample...")

sample  = dataset[0]
buckets = sample['buckets'].unsqueeze(0).to(DEVICE)
gt      = sample['frames']
t_mid   = gt.shape[0] // 2

start_ts = [50, 100, 200, 300, 400, 500]
fig, axes = plt.subplots(2, len(start_ts) + 1, figsize=(3 * (len(start_ts) + 1), 7))

axes[0, 0].imshow(gt[t_mid].numpy(), cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('GT', fontsize=8); axes[0, 0].axis('off')
axes[1, 0].axis('off')

with torch.no_grad():
    for j, st in enumerate(start_ts):
        out = model(patterns_flat, buckets,
                    num_steps=DDIM_STEPS,
                    start_t=st).squeeze(0).cpu()
        s = ssim_fn(out[t_mid].numpy(), gt[t_mid].numpy(), data_range=1.0)
        print(f"  start_t={st:4d}  SSIM={s:.4f}")

        axes[0, j+1].imshow(out[t_mid].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, j+1].set_title(f'start_t={st}\nSSIM={s:.3f}', fontsize=8)
        axes[0, j+1].axis('off')

        diff_map = np.abs(out[t_mid].numpy() - gt[t_mid].numpy())
        axes[1, j+1].imshow(diff_map, cmap='hot', vmin=0, vmax=0.5)
        axes[1, j+1].set_title('|Pred - GT|', fontsize=8)
        axes[1, j+1].axis('off')

plt.suptitle('Test 4: start_t sweep — artifact vs quality trade-off',
             fontsize=10, fontweight='bold')
plt.tight_layout()
path = f'{OUTPUT_DIR}/test4_start_t_sweep.png'
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"Deterministic head SSIM : {np.mean(det_ssims):.4f}")
print(f"Diffusion output  SSIM  : {np.mean(diff_ssims):.4f}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("\nInterpretation guide:")
print("  Test 1 bright square → artifact is in deterministic head (MLP bug)")
print("  Test 1 clean         → artifact is from spatial conditioning")
print("  Test 2 consistent hot cell in same position → spatial map over-activating")
print("  Test 2 spread energy → artifact from elsewhere")
print("  Test 4 low start_t removes artifact → spatial conditioning is source")
print("  Test 4 artifact persists at all start_t → deterministic head is source")