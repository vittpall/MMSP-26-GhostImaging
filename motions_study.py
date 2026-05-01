"""
Motion Analysis for Temporal Ghost-GPT
======================================
Tests:
1. Different motion types (linear, oscillatory, random walk)
2. Different motion speeds
3. Different sequence lengths
4. Motion tracking accuracy
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
import os
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

from models.temporal_ghost_gpt import TemporalGhostGPT
from datasets import MovingMNISTGhost
from torchvision import datasets, transforms


# ============================================================================
# CUSTOM DATASET WITH CONTROLLABLE MOTION
# ============================================================================

class ControlledMotionDataset(torch.utils.data.Dataset):
    """Dataset with precise control over motion parameters"""
    
    def __init__(self, speckle_patterns, motion_type='linear', speed=5.0,
                 seq_length=8, image_size=256, dataset_size=100, train=True):
        
        self.H = torch.tensor(speckle_patterns).float()
        self.M = self.H.shape[0]
        self.seq_length = seq_length
        self.image_size = image_size
        self.dataset_size = dataset_size
        self.motion_type = motion_type
        self.speed = speed
        
        self.H_flat = self.H.view(self.M, -1)
        
        self.mnist = datasets.MNIST(
            root='./data', train=train, download=True,
            transform=transforms.ToTensor()
        )
    
    def generate_trajectory(self):
        T = self.seq_length
        
        if self.motion_type == 'linear':
            # Constant velocity in random direction
            angle = np.random.uniform(0, 2 * np.pi)
            dx = self.speed * np.cos(angle) * np.arange(T)
            dy = self.speed * np.sin(angle) * np.arange(T)
            
        elif self.motion_type == 'oscillatory':
            # Sinusoidal motion (breathing-like)
            freq = np.random.uniform(0.5, 1.5)
            t = np.linspace(0, 2 * np.pi * freq, T)
            dx = self.speed * 3 * np.sin(t)
            dy = self.speed * np.cos(t)
            
        elif self.motion_type == 'random_walk':
            # Brownian motion
            steps = np.random.randn(T, 2) * self.speed * 0.5
            dx = np.cumsum(steps[:, 0])
            dy = np.cumsum(steps[:, 1])
            
        elif self.motion_type == 'circular':
            # Circular motion
            t = np.linspace(0, 2 * np.pi, T)
            radius = self.speed * 3
            dx = radius * np.cos(t)
            dy = radius * np.sin(t)
            
        elif self.motion_type == 'accelerating':
            # Accelerating motion
            angle = np.random.uniform(0, 2 * np.pi)
            t = np.arange(T)
            acceleration = self.speed * 0.2
            displacement = 0.5 * acceleration * t ** 2
            dx = displacement * np.cos(angle)
            dy = displacement * np.sin(angle)
            
        elif self.motion_type == 'stop_and_go':
            # Motion with pauses
            dx = np.zeros(T)
            dy = np.zeros(T)
            moving = True
            pos_x, pos_y = 0, 0
            angle = np.random.uniform(0, 2 * np.pi)
            for t in range(T):
                if t % 3 == 0:  # Change direction or pause
                    moving = np.random.rand() > 0.3
                    angle = np.random.uniform(0, 2 * np.pi)
                if moving:
                    pos_x += self.speed * np.cos(angle)
                    pos_y += self.speed * np.sin(angle)
                dx[t] = pos_x
                dy[t] = pos_y
        else:
            # Default to linear
            angle = np.random.uniform(0, 2 * np.pi)
            dx = self.speed * np.cos(angle) * np.arange(T)
            dy = self.speed * np.sin(angle) * np.arange(T)
        
        return np.stack([dx, dy], axis=1).astype(np.float32)
    
    def warp_image(self, img, displacement):
        dx, dy = displacement
        theta = torch.tensor([
            [1, 0, -2 * dx / self.image_size],
            [0, 1, -2 * dy / self.image_size]
        ]).float().unsqueeze(0)
        
        grid = F.affine_grid(theta, (1, 1, self.image_size, self.image_size), 
                            align_corners=False)
        img_4d = img.unsqueeze(0).unsqueeze(0)
        warped = F.grid_sample(img_4d, grid, align_corners=False, 
                               padding_mode='zeros')
        return warped.squeeze()
    
    def compute_buckets(self, image):
        img_flat = image.view(-1)
        buckets = self.H_flat @ img_flat
        return buckets
    
    def __getitem__(self, idx):
        mnist_idx = idx % len(self.mnist)
        img, label = self.mnist[mnist_idx]
        
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size),
                           mode='bilinear', align_corners=False).squeeze()
        
        trajectory = self.generate_trajectory()
        
        frames = []
        buckets = []
        for t in range(self.seq_length):
            frame_t = self.warp_image(img, trajectory[t])
            bucket_t = self.compute_buckets(frame_t)
            frames.append(frame_t)
            buckets.append(bucket_t)
        
        return {
            'buckets': torch.stack(buckets),
            'frames': torch.stack(frames),
            'trajectory': torch.tensor(trajectory),
            'label': label,
            'motion_type': self.motion_type,
            'speed': self.speed
        }
    
    def __len__(self):
        return self.dataset_size


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def compute_metrics(pred, target):
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    mse_val = mse(target_np, pred_np)
    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    
    return mse_val, ssim_val


def compute_temporal_consistency(pred_frames):
    if isinstance(pred_frames, torch.Tensor):
        pred_frames = pred_frames.cpu().numpy()
    
    frame_diffs = []
    for t in range(1, pred_frames.shape[0]):
        diff = np.abs(pred_frames[t] - pred_frames[t-1]).mean()
        frame_diffs.append(diff)
    
    return np.mean(frame_diffs), np.std(frame_diffs)


def estimate_centroid(frame):
    """Estimate object centroid in a frame"""
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    
    # Threshold to find object
    threshold = 0.3
    binary = (frame > threshold).astype(float)
    
    if binary.sum() == 0:
        return None, None
    
    # Compute centroid
    y_coords, x_coords = np.where(binary > 0)
    cx = np.mean(x_coords)
    cy = np.mean(y_coords)
    
    return cx, cy


def compute_motion_tracking_error(pred_frames, gt_trajectory):
    """Compare predicted object motion with ground truth trajectory"""
    if isinstance(pred_frames, torch.Tensor):
        pred_frames = pred_frames.cpu().numpy()
    if isinstance(gt_trajectory, torch.Tensor):
        gt_trajectory = gt_trajectory.cpu().numpy()
    
    T = pred_frames.shape[0]
    errors = []
    
    # Get initial centroid
    cx0, cy0 = estimate_centroid(pred_frames[0])
    if cx0 is None:
        return None
    
    for t in range(1, T):
        cx, cy = estimate_centroid(pred_frames[t])
        if cx is None:
            continue
        
        # Predicted displacement from frame 0
        pred_dx = cx - cx0
        pred_dy = cy - cy0
        
        # Ground truth displacement
        gt_dx = gt_trajectory[t, 0] - gt_trajectory[0, 0]
        gt_dy = gt_trajectory[t, 1] - gt_trajectory[0, 1]
        
        # Error
        error = np.sqrt((pred_dx - gt_dx)**2 + (pred_dy - gt_dy)**2)
        errors.append(error)
    
    return np.mean(errors) if errors else None


def evaluate_model(model, dataset, patterns_flat, device, num_samples=100):
    model.eval()
    
    all_mse = []
    all_ssim = []
    all_tc = []
    all_tracking_error = []
    frame_ssim = []
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Evaluating"):
            sample = dataset[i]
            buckets = sample['buckets'].unsqueeze(0).to(device)
            frames_gt = sample['frames']
            trajectory = sample['trajectory']
            
            pred_frames = model(patterns_flat, buckets).squeeze(0).cpu()
            
            # Per-frame metrics
            sample_mse = []
            sample_ssim = []
            for t in range(frames_gt.shape[0]):
                m, s = compute_metrics(pred_frames[t], frames_gt[t])
                sample_mse.append(m)
                sample_ssim.append(s)
            
            all_mse.append(np.mean(sample_mse))
            all_ssim.append(np.mean(sample_ssim))
            frame_ssim.append(sample_ssim)
            
            # Temporal consistency
            tc_mean, _ = compute_temporal_consistency(pred_frames)
            all_tc.append(tc_mean)
            
            # Motion tracking
            tracking_error = compute_motion_tracking_error(pred_frames, trajectory)
            if tracking_error is not None:
                all_tracking_error.append(tracking_error)
    
    return {
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'temporal_consistency': np.mean(all_tc),
        'tracking_error': np.mean(all_tracking_error) if all_tracking_error else None,
        'frame_ssim': np.mean(frame_ssim, axis=0),
    }


# ============================================================================
# ANALYSIS 1: MOTION TYPE COMPARISON
# ============================================================================

def analyze_motion_types(model, speckle_patterns, patterns_flat, device, config):
    """Compare performance across different motion types"""
    print("\n" + "="*60)
    print("ANALYSIS 1: Motion Type Comparison")
    print("="*60)
    
    motion_types = ['linear', 'oscillatory', 'random_walk', 'circular', 'accelerating', 'stop_and_go']
    results = {}
    
    for motion_type in motion_types:
        print(f"\nTesting motion type: {motion_type}")
        
        dataset = ControlledMotionDataset(
            speckle_patterns=speckle_patterns,
            motion_type=motion_type,
            speed=5.0,
            seq_length=config['seq_length'],
            image_size=config['image_size'],
            dataset_size=100,
            train=False
        )
        
        results[motion_type] = evaluate_model(model, dataset, patterns_flat, device, num_samples=100)
        print(f"  MSE: {results[motion_type]['mse_mean']:.4f}, SSIM: {results[motion_type]['ssim_mean']:.4f}")
    
    return results


def plot_motion_type_results(results, save_path='outputs/motion_type_comparison.png'):
    """Plot results for different motion types"""
    motion_types = list(results.keys())
    mse_vals = [results[m]['mse_mean'] for m in motion_types]
    ssim_vals = [results[m]['ssim_mean'] for m in motion_types]
    tc_vals = [results[m]['temporal_consistency'] for m in motion_types]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(motion_types)))
    
    # MSE
    bars1 = axes[0].bar(motion_types, mse_vals, color=colors)
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE by Motion Type (↓ lower is better)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # SSIM
    bars2 = axes[1].bar(motion_types, ssim_vals, color=colors)
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM by Motion Type (↑ higher is better)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Temporal Consistency
    bars3 = axes[2].bar(motion_types, tc_vals, color=colors)
    axes[2].set_ylabel('Frame Difference')
    axes[2].set_title('Temporal Consistency (↓ smoother)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


# ============================================================================
# ANALYSIS 2: MOTION SPEED
# ============================================================================

def analyze_motion_speeds(model, speckle_patterns, patterns_flat, device, config):
    """Test performance at different motion speeds"""
    print("\n" + "="*60)
    print("ANALYSIS 2: Motion Speed Analysis")
    print("="*60)
    
    speeds = [1, 2, 5, 10, 15, 20, 30, 50]
    results = {}
    
    for speed in speeds:
        print(f"\nTesting speed: {speed} pixels/frame")
        
        dataset = ControlledMotionDataset(
            speckle_patterns=speckle_patterns,
            motion_type='linear',
            speed=speed,
            seq_length=config['seq_length'],
            image_size=config['image_size'],
            dataset_size=100,
            train=False
        )
        
        results[speed] = evaluate_model(model, dataset, patterns_flat, device, num_samples=100)
        print(f"  MSE: {results[speed]['mse_mean']:.4f}, SSIM: {results[speed]['ssim_mean']:.4f}")
    
    return results


def plot_speed_results(results, save_path='outputs/motion_speed_analysis.png'):
    """Plot MSE and SSIM vs motion speed"""
    speeds = sorted(results.keys())
    mse_vals = [results[s]['mse_mean'] for s in speeds]
    mse_stds = [results[s]['mse_std'] for s in speeds]
    ssim_vals = [results[s]['ssim_mean'] for s in speeds]
    ssim_stds = [results[s]['ssim_std'] for s in speeds]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE vs Speed
    axes[0].errorbar(speeds, mse_vals, yerr=mse_stds, marker='o', capsize=5, 
                     linewidth=2, markersize=8, color='blue')
    axes[0].set_xlabel('Motion Speed (pixels/frame)', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('MSE vs Motion Speed', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # SSIM vs Speed
    axes[1].errorbar(speeds, ssim_vals, yerr=ssim_stds, marker='o', capsize=5,
                     linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Motion Speed (pixels/frame)', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM vs Motion Speed', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


# ============================================================================
# ANALYSIS 3: SEQUENCE LENGTH
# ============================================================================

def analyze_sequence_lengths(model_class, speckle_patterns, device, config):
    """Test different sequence lengths (requires retraining for each)"""
    print("\n" + "="*60)
    print("ANALYSIS 3: Sequence Length Analysis")
    print("="*60)
    
    # Note: This evaluates the trained model on different sequence lengths
    # For proper analysis, you'd retrain for each length
    
    seq_lengths = [2, 4, 8, 12, 16]
    results = {}
    
    num_patterns = speckle_patterns.shape[0]
    patterns_flat = torch.tensor(speckle_patterns).float()
    patterns_flat = patterns_flat.view(num_patterns, -1).to(device)
    
    # Load the model trained with seq_length=8
    model = model_class(
        d_in=config['embedding_dim'],
        d_out=config['embedding_dim'],
        num_blocks=config['num_blocks'],
        number_of_heads=config['num_heads'],
        embedding_dim=config['embedding_dim'],
        flattened_image_size=config['image_size'] * config['image_size'],
        context_size=num_patterns,
        final_image_size=config['image_size'] * config['image_size'],
        seq_length=8  # Fixed - model was trained with this
    ).to(device)
    
    checkpoint = torch.load(config['model_path'], map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        if seq_len != 8:
            print(f"  (Note: Model trained with seq_length=8, testing with {seq_len})")
            # For lengths != 8, we need to either:
            # 1. Pad/truncate sequences
            # 2. Retrain the model (ideal)
            # Here we'll use padding/truncation for demonstration
        
        dataset = ControlledMotionDataset(
            speckle_patterns=speckle_patterns,
            motion_type='linear',
            speed=5.0,
            seq_length=seq_len,
            image_size=config['image_size'],
            dataset_size=100,
            train=False
        )
        
        # Evaluate with padding/truncation to match model's expected length
        results[seq_len] = evaluate_with_length_adjustment(
            model, dataset, patterns_flat, device, 
            target_length=8, num_samples=100
        )
        print(f"  MSE: {results[seq_len]['mse_mean']:.4f}, SSIM: {results[seq_len]['ssim_mean']:.4f}")
    
    return results


def evaluate_with_length_adjustment(model, dataset, patterns_flat, device, 
                                     target_length=8, num_samples=100):
    """Evaluate by padding or truncating sequences to target length"""
    model.eval()
    
    all_mse = []
    all_ssim = []
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Evaluating"):
            sample = dataset[i]
            buckets = sample['buckets']  # [T, M]
            frames_gt = sample['frames']  # [T, H, W]
            
            T = buckets.shape[0]
            
            # Adjust to target length
            if T < target_length:
                # Pad by repeating last frame
                pad_size = target_length - T
                buckets = torch.cat([buckets, buckets[-1:].repeat(pad_size, 1)], dim=0)
                frames_gt = torch.cat([frames_gt, frames_gt[-1:].repeat(pad_size, 1, 1)], dim=0)
            elif T > target_length:
                # Truncate
                buckets = buckets[:target_length]
                frames_gt = frames_gt[:target_length]
            
            buckets = buckets.unsqueeze(0).to(device)
            pred_frames = model(patterns_flat, buckets).squeeze(0).cpu()
            
            # Only evaluate on original frames (not padded)
            eval_length = min(T, target_length)
            sample_mse = []
            sample_ssim = []
            for t in range(eval_length):
                m, s = compute_metrics(pred_frames[t], frames_gt[t])
                sample_mse.append(m)
                sample_ssim.append(s)
            
            all_mse.append(np.mean(sample_mse))
            all_ssim.append(np.mean(sample_ssim))
    
    return {
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
    }


def plot_sequence_length_results(results, save_path='outputs/sequence_length_analysis.png'):
    """Plot results for different sequence lengths"""
    seq_lengths = sorted(results.keys())
    mse_vals = [results[s]['mse_mean'] for s in seq_lengths]
    ssim_vals = [results[s]['ssim_mean'] for s in seq_lengths]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(seq_lengths, mse_vals, 'bo-', linewidth=2, markersize=10)
    axes[0].axvline(x=8, color='red', linestyle='--', label='Training length')
    axes[0].set_xlabel('Sequence Length', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('MSE vs Sequence Length', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(seq_lengths, ssim_vals, 'go-', linewidth=2, markersize=10)
    axes[1].axvline(x=8, color='red', linestyle='--', label='Training length')
    axes[1].set_xlabel('Sequence Length', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM vs Sequence Length', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


# ============================================================================
# ANALYSIS 4: FRAME-BY-FRAME SSIM FOR EACH MOTION TYPE
# ============================================================================

def plot_frame_ssim_by_motion(motion_results, save_path='outputs/frame_ssim_by_motion.png'):
    """Plot SSIM across frames for each motion type"""
    plt.figure(figsize=(12, 6))
    
    for motion_type, res in motion_results.items():
        if 'frame_ssim' in res and res['frame_ssim'] is not None:
            ssim_per_frame = res['frame_ssim']
            plt.plot(range(len(ssim_per_frame)), ssim_per_frame, 
                    'o-', label=motion_type, linewidth=2, markersize=6)
    
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title('SSIM Across Frames by Motion Type', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


# ============================================================================
# VISUALIZATION: SAMPLE RECONSTRUCTIONS FOR EACH MOTION TYPE
# ============================================================================

def visualize_motion_samples(model, speckle_patterns, patterns_flat, device, config,
                              save_path='outputs/motion_samples.png'):
    """Visualize reconstructions for each motion type"""
    motion_types = ['linear', 'oscillatory', 'random_walk', 'circular']
    
    fig, axes = plt.subplots(len(motion_types), 9, figsize=(18, 2.5 * len(motion_types)))
    
    model.eval()
    
    with torch.no_grad():
        for row, motion_type in enumerate(motion_types):
            dataset = ControlledMotionDataset(
                speckle_patterns=speckle_patterns,
                motion_type=motion_type,
                speed=5.0,
                seq_length=config['seq_length'],
                image_size=config['image_size'],
                dataset_size=10,
                train=False
            )
            
            sample = dataset[0]
            buckets = sample['buckets'].unsqueeze(0).to(device)
            frames_gt = sample['frames']
            trajectory = sample['trajectory'].numpy()
            
            pred_frames = model(patterns_flat, buckets).squeeze(0).cpu()
            
            # Show frames 0, 2, 4, 6 (GT and Pred alternating)
            for col, t in enumerate([0, 2, 4, 6]):
                # Ground truth
                axes[row, col * 2].imshow(frames_gt[t], cmap='gray')
                if row == 0:
                    axes[row, col * 2].set_title(f't={t} GT')
                axes[row, col * 2].axis('off')
                
                # Prediction
                axes[row, col * 2 + 1].imshow(pred_frames[t].numpy(), cmap='gray')
                if row == 0:
                    axes[row, col * 2 + 1].set_title(f't={t} Pred')
                axes[row, col * 2 + 1].axis('off')
            
            # Motion trajectory
            axes[row, 8].plot(trajectory[:, 0], trajectory[:, 1], 'b.-', markersize=8)
            axes[row, 8].plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
            axes[row, 8].plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
            axes[row, 8].set_title(motion_type)
            axes[row, 8].set_aspect('equal')
            axes[row, 8].set_xlabel('dx')
            axes[row, 8].set_ylabel('dy')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_motion_analysis():
    # Config
    SPECKLE_PATH = 'data/speckle_pattern.pt'
    MODEL_PATH = 'checkpoints/temporal_ghost_gpt_final.pt'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    CONFIG = {
        'image_size': 256,
        'seq_length': 8,
        'num_blocks': 8,
        'num_heads': 8,
        'embedding_dim': 32,
        'model_path': MODEL_PATH,
    }
    
    # Load speckle patterns
    print("Loading speckle patterns...")
    speckle_patterns = torch.load(SPECKLE_PATH)
    if isinstance(speckle_patterns, torch.Tensor):
        speckle_patterns = speckle_patterns.numpy()
    num_patterns = speckle_patterns.shape[0]
    print(f"Speckle patterns shape: {speckle_patterns.shape}")
    
    patterns_flat = torch.tensor(speckle_patterns).float()
    patterns_flat = patterns_flat.view(num_patterns, -1).to(DEVICE)
    
    # Load model
    print("Loading model...")
    model = TemporalGhostGPT(
        d_in=CONFIG['embedding_dim'],
        d_out=CONFIG['embedding_dim'],
        num_blocks=CONFIG['num_blocks'],
        number_of_heads=CONFIG['num_heads'],
        embedding_dim=CONFIG['embedding_dim'],
        flattened_image_size=CONFIG['image_size'] * CONFIG['image_size'],
        context_size=num_patterns,
        final_image_size=CONFIG['image_size'] * CONFIG['image_size'],
        seq_length=CONFIG['seq_length']
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    all_results = {}
    
    # Analysis 1: Motion Types
    motion_results = analyze_motion_types(model, speckle_patterns, patterns_flat, DEVICE, CONFIG)
    all_results['motion_types'] = motion_results
    plot_motion_type_results(motion_results)
    plot_frame_ssim_by_motion(motion_results)
    
    # Analysis 2: Motion Speeds
    speed_results = analyze_motion_speeds(model, speckle_patterns, patterns_flat, DEVICE, CONFIG)
    all_results['motion_speeds'] = speed_results
    plot_speed_results(speed_results)
    
    # Analysis 3: Sequence Lengths
    seq_results = analyze_sequence_lengths(TemporalGhostGPT, speckle_patterns, DEVICE, CONFIG)
    all_results['sequence_lengths'] = seq_results
    plot_sequence_length_results(seq_results)
    
    # Visualization
    visualize_motion_samples(model, speckle_patterns, patterns_flat, DEVICE, CONFIG)
    
    # Print summary table
    print("\n" + "="*80)
    print("MOTION ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nMotion Type Results:")
    print(f"{'Motion Type':<15} {'MSE':<15} {'SSIM':<15} {'Tracking Error':<15}")
    print("-"*60)
    for motion, res in motion_results.items():
        te = f"{res['tracking_error']:.2f}" if res['tracking_error'] else "N/A"
        print(f"{motion:<15} {res['mse_mean']:.4f}±{res['mse_std']:.4f}  {res['ssim_mean']:.4f}±{res['ssim_std']:.4f}  {te}")
    
    print("\nMotion Speed Results:")
    print(f"{'Speed':<10} {'MSE':<15} {'SSIM':<15}")
    print("-"*40)
    for speed in sorted(speed_results.keys()):
        res = speed_results[speed]
        print(f"{speed:<10} {res['mse_mean']:.4f}±{res['mse_std']:.4f}  {res['ssim_mean']:.4f}±{res['ssim_std']:.4f}")
    
    # Save results
    with open('outputs/motion_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print("\nSaved results to outputs/motion_analysis_results.json")
    
    return all_results


if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    results = run_motion_analysis()