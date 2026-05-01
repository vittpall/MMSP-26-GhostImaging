"""
Noise Robustness Analysis for Temporal Ghost-GPT
================================================
Tests:
1. SNR analysis (like paper's Figure 6)
2. Missing bucket measurements
3. Different noise types
4. Comparison with classical methods
5. Comparison with Original Ghost-GPT
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

from models.temporal_ghost_gpt import TemporalGhostGPT
from models.Ghost_GPT import GhostGPT
from datasets import MovingMNISTGhost
from models.classical_alg import DGI_Recon, PseudoInverse_Recon, FISTA_Recon


# ============================================================================
# NOISE FUNCTIONS
# ============================================================================

def add_gaussian_noise(buckets, snr_db):
    """
    Add Gaussian noise to bucket measurements at specified SNR.
    SNR (dB) = 10 * log10(signal_power / noise_power)
    """
    if isinstance(buckets, torch.Tensor):
        buckets_np = buckets.numpy()
    else:
        buckets_np = buckets.copy()
    
    # Calculate signal power
    signal_power = np.mean(buckets_np ** 2)
    
    # Calculate noise power from SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    
    # Generate noise
    noise = np.random.normal(0, noise_std, buckets_np.shape)
    noisy_buckets = buckets_np + noise
    
    if isinstance(buckets, torch.Tensor):
        return torch.tensor(noisy_buckets).float()
    return noisy_buckets


def add_poisson_noise(buckets, scale=1.0):
    """Add Poisson-like noise (shot noise)"""
    if isinstance(buckets, torch.Tensor):
        buckets_np = buckets.numpy()
    else:
        buckets_np = buckets.copy()
    
    # Shift to positive values
    min_val = buckets_np.min()
    shifted = buckets_np - min_val + 1e-6
    
    # Add Poisson noise
    noisy = np.random.poisson(shifted * scale) / scale
    
    # Shift back
    noisy = noisy + min_val
    
    if isinstance(buckets, torch.Tensor):
        return torch.tensor(noisy).float()
    return noisy


def add_uniform_noise(buckets, noise_level=0.1):
    """Add uniform noise as fraction of signal range"""
    if isinstance(buckets, torch.Tensor):
        buckets_np = buckets.numpy()
    else:
        buckets_np = buckets.copy()
    
    signal_range = buckets_np.max() - buckets_np.min()
    noise = np.random.uniform(-1, 1, buckets_np.shape) * signal_range * noise_level
    noisy = buckets_np + noise
    
    if isinstance(buckets, torch.Tensor):
        return torch.tensor(noisy).float()
    return noisy


def drop_measurements(buckets, drop_ratio=0.1):
    """Randomly drop (zero out) some bucket measurements"""
    if isinstance(buckets, torch.Tensor):
        buckets_np = buckets.numpy().copy()
    else:
        buckets_np = buckets.copy()
    
    # Create mask
    mask = np.random.rand(*buckets_np.shape) > drop_ratio
    buckets_np = buckets_np * mask
    
    if isinstance(buckets, torch.Tensor):
        return torch.tensor(buckets_np).float()
    return buckets_np


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


def evaluate_temporal_with_noise(model, dataset, patterns_flat, device, 
                                  noise_func, noise_param, num_samples=100):
    """Evaluate Temporal model with noisy bucket measurements"""
    model.eval()
    
    all_mse = []
    all_ssim = []
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Temporal GPT"):
            sample = dataset[i]
            buckets = sample['buckets']  # [T, M]
            frames_gt = sample['frames']  # [T, H, W]
            
            # Add noise to buckets
            noisy_buckets = noise_func(buckets, noise_param)
            noisy_buckets = noisy_buckets.unsqueeze(0).to(device)
            
            pred_frames = model(patterns_flat, noisy_buckets).squeeze(0).cpu()
            
            # Compute metrics
            sample_mse = []
            sample_ssim = []
            for t in range(frames_gt.shape[0]):
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


def evaluate_original_with_noise(model, dataset, patterns_flat, device, 
                                  noise_func, noise_param, num_samples=100):
    """
    Evaluate Original Ghost-GPT with noisy bucket measurements.
    Original model processes single frames, so we evaluate each frame independently.
    """
    model.eval()
    
    all_mse = []
    all_ssim = []
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Original GPT"):
            sample = dataset[i]
            buckets = sample['buckets']  # [T, M]
            frames_gt = sample['frames']  # [T, H, W]
            
            # Add noise to buckets
            noisy_buckets = noise_func(buckets, noise_param)
            
            # Process each frame independently with original model
            sample_mse = []
            sample_ssim = []
            for t in range(frames_gt.shape[0]):
                bucket_t = noisy_buckets[t].unsqueeze(0).to(device)  # [1, M]
                
                # Original model forward pass
                pred_frame = model(patterns_flat, bucket_t).squeeze(0).cpu()
                
                # Reshape to image
                H = W = int(np.sqrt(pred_frame.shape[0]))
                pred_frame = pred_frame.view(H, W)
                
                m, s = compute_metrics(pred_frame, frames_gt[t])
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


def evaluate_classical_with_noise(dataset, speckle_patterns, 
                                   noise_func, noise_param, num_samples=30):
    """Evaluate classical methods with noisy measurements"""
    results = {
        'DGI': {'mse': [], 'ssim': []},
        'PI': {'mse': [], 'ssim': []},
        'FISTA': {'mse': [], 'ssim': []},
    }
    
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Classical"):
        sample = dataset[i]
        buckets = sample['buckets'].numpy()  # [T, M]
        frames_gt = sample['frames'].numpy()  # [T, H, W]
        
        # Use middle frame
        t = buckets.shape[0] // 2
        bucket_t = buckets[t]
        frame_gt = frames_gt[t]
        
        # Add noise
        noisy_bucket = noise_func(bucket_t, noise_param)
        if isinstance(noisy_bucket, torch.Tensor):
            noisy_bucket = noisy_bucket.numpy()
        
        # DGI
        try:
            pred_dgi = DGI_Recon(speckle_patterns, noisy_bucket)
            m, s = compute_metrics(pred_dgi, frame_gt)
            results['DGI']['mse'].append(m)
            results['DGI']['ssim'].append(s)
        except:
            pass
        
        # PI
        try:
            pred_pi = PseudoInverse_Recon(speckle_patterns, noisy_bucket)
            m, s = compute_metrics(pred_pi, frame_gt)
            results['PI']['mse'].append(m)
            results['PI']['ssim'].append(s)
        except:
            pass
        
        # FISTA
        try:
            pred_fista = FISTA_Recon(speckle_patterns, noisy_bucket, eps=50)
            m, s = compute_metrics(pred_fista, frame_gt)
            results['FISTA']['mse'].append(m)
            results['FISTA']['ssim'].append(s)
        except:
            pass
    
    final = {}
    for alg in results:
        if results[alg]['mse']:
            final[alg] = {
                'mse_mean': np.mean(results[alg]['mse']),
                'mse_std': np.std(results[alg]['mse']),
                'ssim_mean': np.mean(results[alg]['ssim']),
                'ssim_std': np.std(results[alg]['ssim']),
            }
    return final


# ============================================================================
# ANALYSIS 1: SNR ANALYSIS
# ============================================================================

def snr_analysis(temporal_model, original_model, dataset, speckle_patterns, 
                 patterns_flat, patterns_flat_original, device, num_samples=50):
    """Test performance at different SNR levels"""
    print("\n" + "="*60)
    print("ANALYSIS 1: SNR Analysis")
    print("="*60)
    
    snr_levels = [0, 5, 10, 15, 20, 25, 30, 40]
    
    temporal_results = {}
    original_results = {}
    classical_results = {snr: {} for snr in snr_levels}
    
    for snr in snr_levels:
        print(f"\nTesting SNR: {snr} dB")
        
        noise_func = add_gaussian_noise
        noise_param = snr
        
        # Temporal GPT
        temporal_results[snr] = evaluate_temporal_with_noise(
            temporal_model, dataset, patterns_flat, device,
            noise_func, noise_param, num_samples=num_samples
        )
        print(f"  Temporal GPT - MSE: {temporal_results[snr]['mse_mean']:.4f}, SSIM: {temporal_results[snr]['ssim_mean']:.4f}")
        
        # Original GPT (if available)
        if original_model is not None:
            original_results[snr] = evaluate_original_with_noise(
                original_model, dataset, patterns_flat_original, device,
                noise_func, noise_param, num_samples=num_samples
            )
            print(f"  Original GPT - MSE: {original_results[snr]['mse_mean']:.4f}, SSIM: {original_results[snr]['ssim_mean']:.4f}")
        
        # Classical methods (slower, use fewer samples)
        classical_results[snr] = evaluate_classical_with_noise(
            dataset, speckle_patterns, noise_func, noise_param, num_samples=20
        )
        for alg, res in classical_results[snr].items():
            print(f"  {alg} - MSE: {res['mse_mean']:.4f}, SSIM: {res['ssim_mean']:.4f}")
    
    return temporal_results, original_results, classical_results


def plot_snr_results(temporal_results, original_results, classical_results, 
                     save_path='outputs/snr_analysis.png'):
    """Plot SNR analysis results (like paper's Figure 6)"""
    snr_levels = sorted(temporal_results.keys())
    
    # Temporal GPT
    temp_mse = [temporal_results[s]['mse_mean'] for s in snr_levels]
    temp_ssim = [temporal_results[s]['ssim_mean'] for s in snr_levels]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE vs SNR
    axes[0].plot(snr_levels, temp_mse, 'go-', linewidth=2, markersize=10, label='Temporal GPT (Ours)')
    
    # Original GPT
    if original_results:
        orig_mse = [original_results[s]['mse_mean'] for s in snr_levels]
        axes[0].plot(snr_levels, orig_mse, 'mo-', linewidth=2, markersize=8, label='Original GPT (Paper)')
    
    # Add classical methods
    for alg, color in [('DGI', 'red'), ('PI', 'blue'), ('FISTA', 'orange')]:
        alg_mse = []
        alg_snr = []
        for snr in snr_levels:
            if snr in classical_results and alg in classical_results[snr]:
                alg_mse.append(classical_results[snr][alg]['mse_mean'])
                alg_snr.append(snr)
        if alg_mse:
            axes[0].plot(alg_snr, alg_mse, 'o--', color=color, linewidth=1.5, markersize=6, label=alg)
    
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('MSE vs SNR (↓ lower is better)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # SSIM vs SNR
    axes[1].plot(snr_levels, temp_ssim, 'go-', linewidth=2, markersize=10, label='Temporal GPT (Ours)')
    
    # Original GPT
    if original_results:
        orig_ssim = [original_results[s]['ssim_mean'] for s in snr_levels]
        axes[1].plot(snr_levels, orig_ssim, 'mo-', linewidth=2, markersize=8, label='Original GPT (Paper)')
    
    for alg, color in [('DGI', 'red'), ('PI', 'blue'), ('FISTA', 'orange')]:
        alg_ssim = []
        alg_snr = []
        for snr in snr_levels:
            if snr in classical_results and alg in classical_results[snr]:
                alg_ssim.append(classical_results[snr][alg]['ssim_mean'])
                alg_snr.append(snr)
        if alg_ssim:
            axes[1].plot(alg_snr, alg_ssim, 'o--', color=color, linewidth=1.5, markersize=6, label=alg)
    
    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM vs SNR (↑ higher is better)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


# ============================================================================
# ANALYSIS 2: MISSING MEASUREMENTS
# ============================================================================

def missing_measurements_analysis(temporal_model, original_model, dataset, 
                                   patterns_flat, patterns_flat_original, device, num_samples=50):
    """Test performance with missing bucket measurements"""
    print("\n" + "="*60)
    print("ANALYSIS 2: Missing Measurements Analysis")
    print("="*60)
    
    drop_ratios = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    temporal_results = {}
    original_results = {}
    
    for ratio in drop_ratios:
        print(f"\nTesting drop ratio: {ratio*100:.0f}%")
        
        temporal_results[ratio] = evaluate_temporal_with_noise(
            temporal_model, dataset, patterns_flat, device,
            drop_measurements, ratio, num_samples=num_samples
        )
        print(f"  Temporal GPT - MSE: {temporal_results[ratio]['mse_mean']:.4f}, SSIM: {temporal_results[ratio]['ssim_mean']:.4f}")
        
        if original_model is not None:
            original_results[ratio] = evaluate_original_with_noise(
                original_model, dataset, patterns_flat_original, device,
                drop_measurements, ratio, num_samples=num_samples
            )
            print(f"  Original GPT - MSE: {original_results[ratio]['mse_mean']:.4f}, SSIM: {original_results[ratio]['ssim_mean']:.4f}")
    
    return temporal_results, original_results


def plot_missing_results(temporal_results, original_results, save_path='outputs/missing_measurements.png'):
    """Plot missing measurements analysis"""
    ratios = sorted(temporal_results.keys())
    
    temp_mse = [temporal_results[r]['mse_mean'] for r in ratios]
    temp_ssim = [temporal_results[r]['ssim_mean'] for r in ratios]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot([r*100 for r in ratios], temp_mse, 'go-', linewidth=2, markersize=10, label='Temporal GPT (Ours)')
    if original_results:
        orig_mse = [original_results[r]['mse_mean'] for r in ratios]
        axes[0].plot([r*100 for r in ratios], orig_mse, 'mo-', linewidth=2, markersize=8, label='Original GPT (Paper)')
    axes[0].set_xlabel('Missing Measurements (%)', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('MSE vs Missing Measurements', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot([r*100 for r in ratios], temp_ssim, 'go-', linewidth=2, markersize=10, label='Temporal GPT (Ours)')
    if original_results:
        orig_ssim = [original_results[r]['ssim_mean'] for r in ratios]
        axes[1].plot([r*100 for r in ratios], orig_ssim, 'mo-', linewidth=2, markersize=8, label='Original GPT (Paper)')
    axes[1].set_xlabel('Missing Measurements (%)', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM vs Missing Measurements', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


# ============================================================================
# ANALYSIS 3: NOISE TYPE COMPARISON
# ============================================================================

def noise_type_analysis(temporal_model, original_model, dataset, 
                        patterns_flat, patterns_flat_original, device, num_samples=50):
    """Compare different noise types"""
    print("\n" + "="*60)
    print("ANALYSIS 3: Noise Type Comparison")
    print("="*60)
    
    noise_configs = [
        ('No Noise', lambda x, p: x, None),
        ('Gaussian (20dB)', add_gaussian_noise, 20),
        ('Gaussian (10dB)', add_gaussian_noise, 10),
        ('Poisson', add_poisson_noise, 100),
        ('Uniform (10%)', add_uniform_noise, 0.1),
        ('Uniform (20%)', add_uniform_noise, 0.2),
    ]
    
    temporal_results = {}
    original_results = {}
    
    for name, noise_func, noise_param in noise_configs:
        print(f"\nTesting: {name}")
        
        temporal_results[name] = evaluate_temporal_with_noise(
            temporal_model, dataset, patterns_flat, device,
            noise_func, noise_param, num_samples=num_samples
        )
        print(f"  Temporal GPT - MSE: {temporal_results[name]['mse_mean']:.4f}, SSIM: {temporal_results[name]['ssim_mean']:.4f}")
        
        if original_model is not None:
            original_results[name] = evaluate_original_with_noise(
                original_model, dataset, patterns_flat_original, device,
                noise_func, noise_param, num_samples=num_samples
            )
            print(f"  Original GPT - MSE: {original_results[name]['mse_mean']:.4f}, SSIM: {original_results[name]['ssim_mean']:.4f}")
    
    return temporal_results, original_results


def plot_noise_type_results(temporal_results, original_results, save_path='outputs/noise_type_comparison.png'):
    """Plot noise type comparison"""
    names = list(temporal_results.keys())
    
    temp_mse = [temporal_results[n]['mse_mean'] for n in names]
    temp_ssim = [temporal_results[n]['ssim_mean'] for n in names]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(names))
    width = 0.35
    
    # MSE
    bars1 = axes[0].bar(x - width/2, temp_mse, width, label='Temporal GPT (Ours)', color='green', alpha=0.8)
    if original_results:
        orig_mse = [original_results[n]['mse_mean'] for n in names]
        bars2 = axes[0].bar(x + width/2, orig_mse, width, label='Original GPT (Paper)', color='purple', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('MSE by Noise Type (↓ lower is better)', fontsize=14)
    axes[0].legend()
    
    # SSIM
    bars3 = axes[1].bar(x - width/2, temp_ssim, width, label='Temporal GPT (Ours)', color='green', alpha=0.8)
    if original_results:
        orig_ssim = [original_results[n]['ssim_mean'] for n in names]
        bars4 = axes[1].bar(x + width/2, orig_ssim, width, label='Original GPT (Paper)', color='purple', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM by Noise Type (↑ higher is better)', fontsize=14)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


# ============================================================================
# VISUALIZATION: RECONSTRUCTIONS AT DIFFERENT SNR
# ============================================================================

def visualize_snr_reconstructions(temporal_model, original_model, dataset, 
                                   patterns_flat, patterns_flat_original, device,
                                   save_path='outputs/snr_reconstructions.png'):
    """Visualize reconstructions at different SNR levels"""
    snr_levels = [40, 30, 20, 10, 5]
    
    num_cols = len(snr_levels) + 1  # +1 for ground truth
    num_rows = 3 if original_model is not None else 2  # GT, Temporal, [Original]
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    
    temporal_model.eval()
    if original_model is not None:
        original_model.eval()
    
    sample = dataset[0]
    buckets = sample['buckets']
    frames_gt = sample['frames']
    t = frames_gt.shape[0] // 2  # Middle frame
    
    with torch.no_grad():
        for col, snr in enumerate([None] + snr_levels):
            if snr is None:
                # Ground truth column
                axes[0, col].imshow(frames_gt[t], cmap='gray')
                axes[0, col].set_title('Ground Truth', fontsize=10)
                axes[0, col].axis('off')
                
                axes[1, col].imshow(frames_gt[t], cmap='gray')
                axes[1, col].set_title('Temporal GPT', fontsize=10)
                axes[1, col].axis('off')
                axes[1, col].set_ylabel('', fontsize=10)
                
                if original_model is not None:
                    axes[2, col].imshow(frames_gt[t], cmap='gray')
                    axes[2, col].set_title('Original GPT', fontsize=10)
                    axes[2, col].axis('off')
            else:
                noisy_buckets = add_gaussian_noise(buckets, snr)
                
                # Temporal GPT
                noisy_buckets_temporal = noisy_buckets.unsqueeze(0).to(device)
                pred_temporal = temporal_model(patterns_flat, noisy_buckets_temporal).squeeze(0).cpu()
                _, ssim_temporal = compute_metrics(pred_temporal[t], frames_gt[t])
                
                axes[0, col].imshow(frames_gt[t], cmap='gray')
                axes[0, col].set_title(f'SNR={snr}dB', fontsize=10)
                axes[0, col].axis('off')
                
                axes[1, col].imshow(pred_temporal[t].numpy(), cmap='gray')
                axes[1, col].set_title(f'SSIM={ssim_temporal:.3f}', fontsize=10)
                axes[1, col].axis('off')
                
                # Original GPT
                if original_model is not None:
                    bucket_t = noisy_buckets[t].unsqueeze(0).to(device)
                    pred_original = original_model(patterns_flat_original, bucket_t).squeeze(0).cpu()
                    H = W = int(np.sqrt(pred_original.shape[0]))
                    pred_original = pred_original.view(H, W)
                    _, ssim_original = compute_metrics(pred_original, frames_gt[t])
                    
                    axes[2, col].imshow(pred_original.numpy(), cmap='gray')
                    axes[2, col].set_title(f'SSIM={ssim_original:.3f}', fontsize=10)
                    axes[2, col].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Ground Truth', fontsize=12)
    axes[1, 0].set_ylabel('Temporal GPT', fontsize=12)
    if original_model is not None:
        axes[2, 0].set_ylabel('Original GPT', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


# ============================================================================
# LOAD ORIGINAL GHOST-GPT MODEL
# ============================================================================

def load_original_ghostgpt(checkpoint_path, device):
    """
    Load the original Ghost-GPT model from the paper.
    The original model has different hyperparameters based on the checkpoint.
    """
    print(f"\nLoading Original Ghost-GPT from {checkpoint_path}...")
    
    # First, inspect the checkpoint to determine the architecture
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if it's a state dict or wrapped
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Infer architecture from state dict
    # From the error message, we know:
    # - W_query shape: [384, 32] -> num_heads * d_out = 384, embedding_dim = 32
    # - So num_heads = 384 / 32 = 12
    # - pos_embedding_layer.weight shape: [250, 32] -> context_size = 250
    # - final_projection_layer2.weight shape: [65536, 4000] -> 250 * 16 = 4000
    
    # Extract actual dimensions from checkpoint
    context_size = None
    embedding_dim = None
    total_dim = None

    for key, value in state_dict.items():
        if 'pos_embedding_layer.weight' in key:
            context_size = value.shape[0]
            embedding_dim = value.shape[1]
            print(f"  Detected context_size={context_size}, embedding_dim={embedding_dim}")
        if 'main_body.0.MultiHeadedAttention.W_query.weight' in key:
            total_dim = value.shape[0]

    num_heads = (total_dim // embedding_dim) if (total_dim and embedding_dim) else 12
    print(f"  Detected num_heads={num_heads}")
    if 'final_projection_layer2.weight' in key:
        output_size = value.shape[0]
        intermediate_size = value.shape[1]
        print(f"  Detected output_size={output_size}, intermediate_size={intermediate_size}")

    # Count number of blocks
    num_blocks = 0
    for key in state_dict.keys():
        if 'main_body.' in key:
            block_idx = int(key.split('.')[1])
            num_blocks = max(num_blocks, block_idx + 1)
    print(f"  Detected num_blocks={num_blocks}")
    
    # Create model with inferred parameters
    try:
        model = GhostGPT(
            d_in=embedding_dim,
            d_out=embedding_dim,
            num_blocks=num_blocks,
            number_of_heads=num_heads,
            embedding_dim=embedding_dim,
            flattened_image_size=256*256,
            context_size=context_size,
            final_image_size=256*256
        ).to(device)
        
        # Load weights
        model.load_state_dict(state_dict)
        model.eval()
        print("  Successfully loaded Original Ghost-GPT!")
        return model, context_size
        
    except Exception as e:
        print(f"  Warning: Could not load Original Ghost-GPT: {e}")
        print("  Continuing without Original Ghost-GPT comparison...")
        return None, None


# ============================================================================
# MAIN
# ============================================================================

def run_noise_analysis():
    # Config
    SPECKLE_PATH = 'data/speckle_pattern.pt'
    TEMPORAL_MODEL_PATH = 'checkpoints/temporal_ghost_gpt_final.pt'
    ORIGINAL_MODEL_PATH = None
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    CONFIG = {
        'image_size': 256,
        'seq_length': 8,
        'num_blocks': 8,
        'num_heads': 8,
        'embedding_dim': 32,
    }
    
    # Load speckle patterns
    print("Loading speckle patterns...")
    speckle_patterns = torch.load(SPECKLE_PATH)
    if isinstance(speckle_patterns, torch.Tensor):
        speckle_patterns = speckle_patterns.numpy()
    num_patterns = speckle_patterns.shape[0]
    print(f"Speckle patterns shape: {speckle_patterns.shape}")
    
    # Patterns for temporal model
    patterns_flat = torch.tensor(speckle_patterns).float()
    patterns_flat = patterns_flat.view(num_patterns, -1).to(DEVICE)
    
    # Load Temporal Ghost-GPT
    print("\nLoading Temporal Ghost-GPT...")
    temporal_model = TemporalGhostGPT(
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
    
    checkpoint = torch.load(TEMPORAL_MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        temporal_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        temporal_model.load_state_dict(checkpoint)
    temporal_model.eval()
    
    # Load Original Ghost-GPT (if available)
    original_model = None
    patterns_flat_original = None
    original_context_size = None
    
    if ORIGINAL_MODEL_PATH and os.path.exists(ORIGINAL_MODEL_PATH):
        original_model, original_context_size = load_original_ghostgpt(ORIGINAL_MODEL_PATH, DEVICE)
        
        if original_model is not None and original_context_size is not None:
            # Original model may need different speckle pattern size
            if original_context_size != num_patterns:
                print(f"\n  Note: Original model expects {original_context_size} patterns, but we have {num_patterns}")
                print(f"  Creating padded/truncated patterns for original model...")
                
                if original_context_size > num_patterns:
                    # Pad with repeated patterns
                    padding_needed = original_context_size - num_patterns
                    padded = np.concatenate([speckle_patterns, speckle_patterns[:padding_needed]], axis=0)
                    patterns_flat_original = torch.tensor(padded).float()
                else:
                    # Truncate
                    patterns_flat_original = torch.tensor(speckle_patterns[:original_context_size]).float()
                
                patterns_flat_original = patterns_flat_original.view(original_context_size, -1).to(DEVICE)
            else:
                patterns_flat_original = patterns_flat
    else:
        print(f"\nOriginal model not found at {ORIGINAL_MODEL_PATH}")
        print("Continuing without Original Ghost-GPT comparison...")
    
    # Create dataset
    print("\nCreating test dataset...")
    dataset = MovingMNISTGhost(
        speckle_patterns=speckle_patterns,
        seq_length=CONFIG['seq_length'],
        image_size=CONFIG['image_size'],
        dataset_size=100,
        train=False
    )
    
    all_results = {}
    
    # Analysis 1: SNR
    print("\n" + "="*80)
    snr_temp, snr_orig, snr_classical = snr_analysis(
        temporal_model, original_model, dataset, speckle_patterns,
        patterns_flat, patterns_flat_original, DEVICE, num_samples=50
    )
    all_results['snr_temporal'] = snr_temp
    all_results['snr_original'] = snr_orig
    all_results['snr_classical'] = snr_classical
    plot_snr_results(snr_temp, snr_orig, snr_classical)
    
    # Analysis 2: Missing measurements
    missing_temp, missing_orig = missing_measurements_analysis(
        temporal_model, original_model, dataset,
        patterns_flat, patterns_flat_original, DEVICE, num_samples=50
    )
    all_results['missing_temporal'] = missing_temp
    all_results['missing_original'] = missing_orig
    plot_missing_results(missing_temp, missing_orig)
    
    # Analysis 3: Noise types
    noise_temp, noise_orig = noise_type_analysis(
        temporal_model, original_model, dataset,
        patterns_flat, patterns_flat_original, DEVICE, num_samples=50
    )
    all_results['noise_type_temporal'] = noise_temp
    all_results['noise_type_original'] = noise_orig
    plot_noise_type_results(noise_temp, noise_orig)
    
    # Visualization
    visualize_snr_reconstructions(
        temporal_model, original_model, dataset,
        patterns_flat, patterns_flat_original, DEVICE
    )
    
    # Print summary
    print("\n" + "="*80)
    print("NOISE ROBUSTNESS SUMMARY")
    print("="*80)
    
    print("\nSNR Analysis:")
    print(f"{'SNR (dB)':<10} {'Temporal MSE':<15} {'Temporal SSIM':<15}", end="")
    if snr_orig:
        print(f"{'Original MSE':<15} {'Original SSIM':<15}", end="")
    print()
    print("-"*80)
    for snr in sorted(snr_temp.keys()):
        res_t = snr_temp[snr]
        print(f"{snr:<10} {res_t['mse_mean']:.4f}±{res_t['mse_std']:.4f}   {res_t['ssim_mean']:.4f}±{res_t['ssim_std']:.4f}   ", end="")
        if snr_orig and snr in snr_orig:
            res_o = snr_orig[snr]
            print(f"{res_o['mse_mean']:.4f}±{res_o['mse_std']:.4f}   {res_o['ssim_mean']:.4f}±{res_o['ssim_std']:.4f}", end="")
        print()
    
    print("\nMissing Measurements:")
    print(f"{'Missing %':<12} {'Temporal MSE':<15} {'Temporal SSIM':<15}", end="")
    if missing_orig:
        print(f"{'Original MSE':<15} {'Original SSIM':<15}", end="")
    print()
    print("-"*80)
    for ratio in sorted(missing_temp.keys()):
        res_t = missing_temp[ratio]
        print(f"{ratio*100:.0f}%{'':<9} {res_t['mse_mean']:.4f}±{res_t['mse_std']:.4f}   {res_t['ssim_mean']:.4f}±{res_t['ssim_std']:.4f}   ", end="")
        if missing_orig and ratio in missing_orig:
            res_o = missing_orig[ratio]
            print(f"{res_o['mse_mean']:.4f}±{res_o['mse_std']:.4f}   {res_o['ssim_mean']:.4f}±{res_o['ssim_std']:.4f}", end="")
        print()
    
    # Save results
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif obj == np.inf:
            return "inf"
        return obj
    
    with open('outputs/noise_robustness_results.json', 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    print("\nSaved results to outputs/noise_robustness_results.json")
    
    return all_results


if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    results = run_noise_analysis()