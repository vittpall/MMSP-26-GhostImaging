"""
Quantum Ghost Imaging Evaluation Suite
======================================
Compares classical vs quantum detection for ghost imaging.

Experiments:
1. Detector type comparison (SNSPD vs SPAD vs SiPM vs Classical)
2. Photon flux analysis (how few photons can we use?)
3. SNR comparison at different light levels
4. Dark count robustness
5. Detection efficiency impact
6. Time-resolved vs integrated detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
import os
import json
from pathlib import Path

from models.temporal_ghost_gpt import TemporalGhostGPT
from datasets import MovingMNISTGhost

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# DETECTOR SIMULATION CLASSES
# ============================================================================

class ClassicalDetector:
    """Classical analog detector (photodiode, CCD pixel, etc.)"""
    
    def __init__(self, noise_std=0.01, quantization_bits=16):
        self.noise_std = noise_std
        self.max_value = 2 ** quantization_bits - 1
        self.name = "Classical"
    
    def measure(self, true_intensity):
        """
        Simulate classical detection.
        
        Args:
            true_intensity: Normalized intensity [0, 1]
        Returns:
            measurement: Noisy intensity measurement
        """
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, true_intensity.shape)
        measured = true_intensity + noise
        
        # Clip to valid range
        measured = np.clip(measured, 0, 1)
        
        return measured.astype(np.float32)


class QuantumDetector:
    """Base class for quantum single-photon detectors."""
    
    def __init__(self, mean_photons=100, efficiency=0.9, 
                 dark_count_rate=100, integration_time=1e-3,
                 dead_time=0, afterpulse_prob=0, crosstalk_prob=0):
        
        self.mean_photons = mean_photons
        self.efficiency = efficiency
        self.dark_count_rate = dark_count_rate
        self.integration_time = integration_time
        self.dark_counts_per_measurement = dark_count_rate * integration_time
        self.dead_time = dead_time
        self.afterpulse_prob = afterpulse_prob
        self.crosstalk_prob = crosstalk_prob
        self.name = "Quantum"
    
    def measure(self, true_intensity):
        """
        Simulate quantum photon counting detection.
        
        Args:
            true_intensity: Normalized intensity [0, 1]
        Returns:
            photon_counts: Integer photon counts
        """
        # Expected photons
        expected = true_intensity * self.mean_photons * self.efficiency
        
        # Poisson sampling
        counts = np.random.poisson(expected)
        
        # Add dark counts
        dark = np.random.poisson(self.dark_counts_per_measurement, counts.shape)
        counts = counts + dark
        
        # Afterpulsing (SPAD)
        if self.afterpulse_prob > 0:
            afterpulses = np.random.binomial(counts, self.afterpulse_prob)
            counts = counts + afterpulses
        
        # Crosstalk (SiPM)
        if self.crosstalk_prob > 0:
            crosstalk = np.random.binomial(counts, self.crosstalk_prob)
            counts = counts + crosstalk
        
        return counts.astype(np.float32)
    
    def get_snr(self, true_intensity):
        """Calculate theoretical SNR for given intensity."""
        signal = true_intensity * self.mean_photons * self.efficiency
        # For Poisson: variance = mean, so SNR = sqrt(signal)
        # With dark counts: SNR = signal / sqrt(signal + dark)
        noise_var = signal + self.dark_counts_per_measurement
        snr = signal / np.sqrt(noise_var + 1e-10)
        return snr


class SNSPD(QuantumDetector):
    """Superconducting Nanowire Single-Photon Detector"""
    
    def __init__(self, mean_photons=100):
        super().__init__(
            mean_photons=mean_photons,
            efficiency=0.95,           # Very high efficiency
            dark_count_rate=10,        # Very low dark counts
            integration_time=1e-3,
            dead_time=40e-9,           # 40ns dead time
            afterpulse_prob=0,
            crosstalk_prob=0
        )
        self.name = "SNSPD"
        self.timing_jitter = 50e-12   # 50ps timing resolution


class SPAD(QuantumDetector):
    """Single-Photon Avalanche Diode"""
    
    def __init__(self, mean_photons=100):
        super().__init__(
            mean_photons=mean_photons,
            efficiency=0.70,           # Lower efficiency
            dark_count_rate=1000,      # Higher dark counts
            integration_time=1e-3,
            dead_time=50e-9,
            afterpulse_prob=0.01,      # 1% afterpulsing
            crosstalk_prob=0
        )
        self.name = "SPAD"
        self.timing_jitter = 300e-12  # 300ps


class SiPM(QuantumDetector):
    """Silicon Photomultiplier"""
    
    def __init__(self, mean_photons=100):
        super().__init__(
            mean_photons=mean_photons,
            efficiency=0.50,           # PDE ~50%
            dark_count_rate=100000,    # High dark count rate
            integration_time=1e-3,
            dead_time=20e-9,
            afterpulse_prob=0.02,
            crosstalk_prob=0.05        # 5% optical crosstalk
        )
        self.name = "SiPM"
        self.timing_jitter = 100e-12


# ============================================================================
# NORMALIZATION FOR PHOTON COUNTS
# ============================================================================

class PhotonCountNormalizer:
    """Normalize photon counts for neural network input."""
    
    def __init__(self, method='anscombe'):
        self.method = method
        self.fitted = False
        self.stats = {}
    
    def fit(self, counts):
        """Compute normalization statistics from data."""
        if isinstance(counts, torch.Tensor):
            counts = counts.numpy()
        counts = counts.flatten()
        
        self.stats = {
            'min': float(np.min(counts)),
            'max': float(np.max(counts)),
            'mean': float(np.mean(counts)),
            'std': float(np.std(counts)),
            'median': float(np.median(counts)),
        }
        self.fitted = True
        return self
    
    def transform(self, counts):
        """Apply normalization transform."""
        if isinstance(counts, np.ndarray):
            counts = torch.tensor(counts).float()
        
        if self.method == 'anscombe':
            # Variance-stabilizing transform for Poisson data
            # Makes variance approximately constant (≈1/4)
            return 2 * torch.sqrt(counts + 3/8)
        
        elif self.method == 'sqrt':
            return torch.sqrt(counts)
        
        elif self.method == 'log':
            return torch.log1p(counts)
        
        elif self.method == 'freeman_tukey':
            # Another variance-stabilizing transform
            return torch.sqrt(counts) + torch.sqrt(counts + 1)
        
        elif self.method == 'minmax':
            if not self.fitted:
                raise ValueError("Must call fit() before minmax transform")
            return (counts - self.stats['min']) / (self.stats['max'] - self.stats['min'] + 1e-8)
        
        elif self.method == 'zscore':
            if not self.fitted:
                raise ValueError("Must call fit() before zscore transform")
            return (counts - self.stats['mean']) / (self.stats['std'] + 1e-8)
        
        else:  # 'none'
            return counts
    
    def inverse_transform(self, normalized):
        """Reverse the normalization (approximate)."""
        if self.method == 'anscombe':
            return (normalized / 2) ** 2 - 3/8
        elif self.method == 'sqrt':
            return normalized ** 2
        elif self.method == 'log':
            return torch.expm1(normalized)
        else:
            return normalized


# ============================================================================
# DATASET WITH CONFIGURABLE DETECTOR
# ============================================================================

class GhostImagingWithDetector(torch.utils.data.Dataset):
    """Ghost imaging dataset with configurable detector simulation."""
    
    def __init__(self, speckle_patterns, detector, normalizer=None,
                 seq_length=8, image_size=256, dataset_size=100, train=False):
        
        from torchvision import datasets, transforms
        
        self.H = torch.tensor(speckle_patterns).float()
        self.M = self.H.shape[0]
        self.H_flat = self.H.view(self.M, -1)
        
        self.detector = detector
        self.normalizer = normalizer
        self.seq_length = seq_length
        self.image_size = image_size
        self.dataset_size = dataset_size
        
        self.mnist = datasets.MNIST(
            root='./data', train=train, download=True,
            transform=transforms.ToTensor()
        )
    
    def compute_true_intensity(self, image):
        """Compute bucket intensities before detection."""
        img_flat = image.view(-1).numpy()
        intensities = self.H_flat.numpy() @ img_flat
        
        # Normalize to [0, 1]
        intensities = (intensities - intensities.min()) / \
                      (intensities.max() - intensities.min() + 1e-8)
        return intensities
    
    def __getitem__(self, idx):
        mnist_idx = idx % len(self.mnist)
        img, label = self.mnist[mnist_idx]
        
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size),
                           mode='bilinear', align_corners=False).squeeze()
        
        # Generate trajectory
        trajectory = self._generate_trajectory()
        
        frames = []
        buckets = []
        true_intensities = []
        
        for t in range(self.seq_length):
            frame_t = self._warp_image(img, trajectory[t])
            
            # Compute true intensity
            intensity_t = self.compute_true_intensity(frame_t)
            true_intensities.append(intensity_t)
            
            # Apply detector simulation
            bucket_t = self.detector.measure(intensity_t)
            
            frames.append(frame_t)
            buckets.append(torch.tensor(bucket_t))
        
        buckets = torch.stack(buckets)
        
        # Apply normalization if provided
        if self.normalizer is not None:
            buckets = self.normalizer.transform(buckets)
        
        return {
            'buckets': buckets,
            'frames': torch.stack(frames),
            'true_intensities': torch.tensor(np.stack(true_intensities)),
            'label': label,
            'detector': self.detector.name,
        }
    
    def _generate_trajectory(self):
        T = self.seq_length
        speed = np.random.uniform(2, 8)
        angle = np.random.uniform(0, 2 * np.pi)
        dx = speed * np.cos(angle) * np.arange(T)
        dy = speed * np.sin(angle) * np.arange(T)
        return np.stack([dx, dy], axis=1).astype(np.float32)
    
    def _warp_image(self, img, displacement):
        dx, dy = displacement
        theta = torch.tensor([
            [1, 0, -2 * dx / self.image_size],
            [0, 1, -2 * dy / self.image_size]
        ]).float().unsqueeze(0)
        
        grid = F.affine_grid(theta, (1, 1, self.image_size, self.image_size),
                            align_corners=False)
        warped = F.grid_sample(img.unsqueeze(0).unsqueeze(0), grid, 
                               align_corners=False, padding_mode='zeros')
        return warped.squeeze()
    
    def __len__(self):
        return self.dataset_size


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def compute_metrics(pred, target):
    """Compute MSE and SSIM metrics."""
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    mse_val = mse(target_np, pred_np)
    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    
    return mse_val, ssim_val


def evaluate_model_with_detector(model, speckle_patterns, patterns_flat, 
                                  detector, normalizer, device, 
                                  num_samples=100, seq_length=8, image_size=256):
    """Evaluate model with a specific detector type."""
    
    dataset = GhostImagingWithDetector(
        speckle_patterns=speckle_patterns,
        detector=detector,
        normalizer=normalizer,
        seq_length=seq_length,
        image_size=image_size,
        dataset_size=num_samples,
        train=False
    )
    
    model.eval()
    all_mse = []
    all_ssim = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=f"Evaluating {detector.name}"):
            sample = dataset[i]
            buckets = sample['buckets'].unsqueeze(0).to(device)
            frames_gt = sample['frames']
            
            pred_frames = model(patterns_flat, buckets).squeeze(0).cpu()
            
            for t in range(frames_gt.shape[0]):
                m, s = compute_metrics(pred_frames[t], frames_gt[t])
                all_mse.append(m)
                all_ssim.append(s)
    
    return {
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'detector': detector.name,
    }


# ============================================================================
# EXPERIMENT 1: DETECTOR TYPE COMPARISON
# ============================================================================

def experiment_detector_comparison(model, speckle_patterns, patterns_flat, 
                                    device, num_samples=100):
    """Compare different detector types at the same photon flux."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: Detector Type Comparison")
    print("="*70)
    
    mean_photons = 100  # Same photon flux for all detectors
    
    detectors = [
        ClassicalDetector(noise_std=0.05),
        SNSPD(mean_photons=mean_photons),
        SPAD(mean_photons=mean_photons),
        SiPM(mean_photons=mean_photons),
    ]
    
    results = {}
    
    for detector in detectors:
        print(f"\nTesting {detector.name}...")
        
        # Create normalizer
        if isinstance(detector, QuantumDetector):
            normalizer = PhotonCountNormalizer(method='anscombe')
            # Fit on sample data
            sample_counts = []
            temp_dataset = GhostImagingWithDetector(
                speckle_patterns, detector, normalizer=None,
                seq_length=8, image_size=256, dataset_size=50
            )
            for i in range(50):
                sample_counts.append(temp_dataset[i]['buckets'].numpy())
            normalizer.fit(np.concatenate(sample_counts))
        else:
            normalizer = None
        
        results[detector.name] = evaluate_model_with_detector(
            model, speckle_patterns, patterns_flat, detector, normalizer,
            device, num_samples=num_samples
        )
        
        print(f"  MSE: {results[detector.name]['mse_mean']:.4f} ± {results[detector.name]['mse_std']:.4f}")
        print(f"  SSIM: {results[detector.name]['ssim_mean']:.4f} ± {results[detector.name]['ssim_std']:.4f}")
    
    return results


# ============================================================================
# EXPERIMENT 2: PHOTON FLUX ANALYSIS
# ============================================================================

def experiment_photon_flux(model, speckle_patterns, patterns_flat, 
                           device, num_samples=50):
    """Analyze performance vs number of photons per measurement."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: Photon Flux Analysis")
    print("="*70)
    
    photon_levels = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    results = {'SNSPD': {}, 'SPAD': {}, 'Classical': {}}
    
    for n_photons in photon_levels:
        print(f"\nTesting with {n_photons} mean photons/measurement...")
        
        # SNSPD
        detector = SNSPD(mean_photons=n_photons)
        normalizer = PhotonCountNormalizer(method='anscombe')
        
        temp_dataset = GhostImagingWithDetector(
            speckle_patterns, detector, normalizer=None,
            seq_length=8, image_size=256, dataset_size=20
        )
        sample_counts = [temp_dataset[i]['buckets'].numpy() for i in range(20)]
        normalizer.fit(np.concatenate(sample_counts))
        
        res = evaluate_model_with_detector(
            model, speckle_patterns, patterns_flat, detector, normalizer,
            device, num_samples=num_samples
        )
        results['SNSPD'][n_photons] = res
        
        # SPAD
        detector = SPAD(mean_photons=n_photons)
        normalizer = PhotonCountNormalizer(method='anscombe')
        temp_dataset = GhostImagingWithDetector(
            speckle_patterns, detector, normalizer=None,
            seq_length=8, image_size=256, dataset_size=20
        )
        sample_counts = [temp_dataset[i]['buckets'].numpy() for i in range(20)]
        normalizer.fit(np.concatenate(sample_counts))
        
        res = evaluate_model_with_detector(
            model, speckle_patterns, patterns_flat, detector, normalizer,
            device, num_samples=num_samples
        )
        results['SPAD'][n_photons] = res
        
        # Classical equivalent (noise proportional to 1/sqrt(photons))
        noise_std = 1.0 / np.sqrt(n_photons)
        detector = ClassicalDetector(noise_std=noise_std)
        
        res = evaluate_model_with_detector(
            model, speckle_patterns, patterns_flat, detector, None,
            device, num_samples=num_samples
        )
        results['Classical'][n_photons] = res
    
    return results


# ============================================================================
# EXPERIMENT 3: DARK COUNT ROBUSTNESS
# ============================================================================

def experiment_dark_counts(model, speckle_patterns, patterns_flat,
                           device, num_samples=50):
    """Analyze robustness to dark counts."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: Dark Count Robustness")
    print("="*70)
    
    dark_count_rates = [0, 10, 100, 1000, 10000, 100000]
    mean_photons = 100
    
    results = {}
    
    for dcr in dark_count_rates:
        print(f"\nTesting with dark count rate = {dcr} Hz...")
        
        detector = QuantumDetector(
            mean_photons=mean_photons,
            efficiency=0.9,
            dark_count_rate=dcr,
            integration_time=1e-3
        )
        detector.name = f"DCR={dcr}"
        
        normalizer = PhotonCountNormalizer(method='anscombe')
        temp_dataset = GhostImagingWithDetector(
            speckle_patterns, detector, normalizer=None,
            seq_length=8, image_size=256, dataset_size=20
        )
        sample_counts = [temp_dataset[i]['buckets'].numpy() for i in range(20)]
        normalizer.fit(np.concatenate(sample_counts))
        
        results[dcr] = evaluate_model_with_detector(
            model, speckle_patterns, patterns_flat, detector, normalizer,
            device, num_samples=num_samples
        )
        
        # Calculate signal-to-dark ratio
        dark_per_measurement = dcr * 1e-3
        signal = mean_photons * 0.9  # With efficiency
        sdr = signal / (dark_per_measurement + 1e-10)
        results[dcr]['signal_to_dark_ratio'] = sdr
        
        print(f"  Signal-to-Dark Ratio: {sdr:.1f}")
        print(f"  SSIM: {results[dcr]['ssim_mean']:.4f}")
    
    return results


# ============================================================================
# EXPERIMENT 4: DETECTION EFFICIENCY
# ============================================================================

def experiment_efficiency(model, speckle_patterns, patterns_flat,
                          device, num_samples=50):
    """Analyze impact of detection efficiency."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 4: Detection Efficiency Impact")
    print("="*70)
    
    efficiencies = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    mean_photons = 100
    
    results = {}
    
    for eff in efficiencies:
        print(f"\nTesting with efficiency = {eff*100:.0f}%...")
        
        detector = QuantumDetector(
            mean_photons=mean_photons,
            efficiency=eff,
            dark_count_rate=100,
            integration_time=1e-3
        )
        detector.name = f"Eff={eff}"
        
        normalizer = PhotonCountNormalizer(method='anscombe')
        temp_dataset = GhostImagingWithDetector(
            speckle_patterns, detector, normalizer=None,
            seq_length=8, image_size=256, dataset_size=20
        )
        sample_counts = [temp_dataset[i]['buckets'].numpy() for i in range(20)]
        normalizer.fit(np.concatenate(sample_counts))
        
        results[eff] = evaluate_model_with_detector(
            model, speckle_patterns, patterns_flat, detector, normalizer,
            device, num_samples=num_samples
        )
        
        print(f"  SSIM: {results[eff]['ssim_mean']:.4f}")
    
    return results


# ============================================================================
# EXPERIMENT 5: NORMALIZATION METHOD COMPARISON
# ============================================================================

def experiment_normalization(model, speckle_patterns, patterns_flat,
                             device, num_samples=50):
    """Compare different normalization methods for photon counts."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 5: Normalization Method Comparison")
    print("="*70)
    
    methods = ['none', 'sqrt', 'log', 'anscombe', 'freeman_tukey', 'minmax', 'zscore']
    
    detector = SNSPD(mean_photons=100)
    results = {}
    
    for method in methods:
        print(f"\nTesting normalization: {method}...")
        
        normalizer = PhotonCountNormalizer(method=method)
        
        # Fit normalizer if needed
        if method in ['minmax', 'zscore']:
            temp_dataset = GhostImagingWithDetector(
                speckle_patterns, detector, normalizer=None,
                seq_length=8, image_size=256, dataset_size=50
            )
            sample_counts = [temp_dataset[i]['buckets'].numpy() for i in range(50)]
            normalizer.fit(np.concatenate(sample_counts))
        
        results[method] = evaluate_model_with_detector(
            model, speckle_patterns, patterns_flat, detector, 
            normalizer if method != 'none' else None,
            device, num_samples=num_samples
        )
        
        print(f"  MSE: {results[method]['mse_mean']:.4f}")
        print(f"  SSIM: {results[method]['ssim_mean']:.4f}")
    
    return results


# ============================================================================
# EXPERIMENT 6: QUANTUM VS CLASSICAL AT DIFFERENT SNR
# ============================================================================

def experiment_snr_comparison(model, speckle_patterns, patterns_flat,
                              device, num_samples=50):
    """Compare quantum and classical detectors across SNR levels."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 6: Quantum vs Classical SNR Comparison")
    print("="*70)
    
    # For quantum: vary photon flux (lower = lower SNR)
    # For classical: vary noise level
    
    snr_levels = [5, 10, 15, 20, 25, 30, 40]  # dB
    
    results = {'SNSPD': {}, 'Classical': {}}
    
    for snr_db in snr_levels:
        print(f"\nTesting at SNR = {snr_db} dB...")
        
        # Convert SNR to photon count (approximate)
        # SNR = sqrt(N) for shot noise limited, so N = SNR^2
        snr_linear = 10 ** (snr_db / 20)
        n_photons = int(snr_linear ** 2)
        n_photons = max(1, min(10000, n_photons))
        
        # SNSPD
        detector = SNSPD(mean_photons=n_photons)
        normalizer = PhotonCountNormalizer(method='anscombe')
        
        temp_dataset = GhostImagingWithDetector(
            speckle_patterns, detector, normalizer=None,
            seq_length=8, image_size=256, dataset_size=20
        )
        sample_counts = [temp_dataset[i]['buckets'].numpy() for i in range(20)]
        normalizer.fit(np.concatenate(sample_counts))
        
        results['SNSPD'][snr_db] = evaluate_model_with_detector(
            model, speckle_patterns, patterns_flat, detector, normalizer,
            device, num_samples=num_samples
        )
        results['SNSPD'][snr_db]['n_photons'] = n_photons
        
        # Classical with equivalent Gaussian noise
        # SNR_dB = 20 * log10(1 / noise_std), so noise_std = 10^(-SNR_dB/20)
        noise_std = 10 ** (-snr_db / 20)
        detector = ClassicalDetector(noise_std=noise_std)
        
        results['Classical'][snr_db] = evaluate_model_with_detector(
            model, speckle_patterns, patterns_flat, detector, None,
            device, num_samples=num_samples
        )
        
        print(f"  SNSPD (N={n_photons}): SSIM = {results['SNSPD'][snr_db]['ssim_mean']:.4f}")
        print(f"  Classical (σ={noise_std:.4f}): SSIM = {results['Classical'][snr_db]['ssim_mean']:.4f}")
    
    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_detector_comparison(results, save_path='outputs/quantum_detector_comparison.png'):
    """Plot detector type comparison results."""
    
    detectors = list(results.keys())
    mse_vals = [results[d]['mse_mean'] for d in detectors]
    mse_stds = [results[d]['mse_std'] for d in detectors]
    ssim_vals = [results[d]['ssim_mean'] for d in detectors]
    ssim_stds = [results[d]['ssim_std'] for d in detectors]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    # MSE
    bars1 = axes[0].bar(detectors, mse_vals, yerr=mse_stds, capsize=5, 
                        color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('MSE by Detector Type (↓ lower is better)', fontsize=14)
    axes[0].set_yscale('log')
    
    # SSIM
    bars2 = axes[1].bar(detectors, ssim_vals, yerr=ssim_stds, capsize=5,
                        color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM by Detector Type (↑ higher is better)', fontsize=14)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_photon_flux(results, save_path='outputs/quantum_photon_flux.png'):
    """Plot photon flux analysis results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for detector_name, data in results.items():
        photons = sorted(data.keys())
        ssim_vals = [data[p]['ssim_mean'] for p in photons]
        mse_vals = [data[p]['mse_mean'] for p in photons]
        
        color = {'SNSPD': '#2ecc71', 'SPAD': '#f39c12', 'Classical': '#3498db'}[detector_name]
        
        axes[0].semilogx(photons, mse_vals, 'o-', label=detector_name, 
                         color=color, linewidth=2, markersize=8)
        axes[1].semilogx(photons, ssim_vals, 'o-', label=detector_name,
                         color=color, linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Mean Photons per Measurement', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('MSE vs Photon Flux', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Mean Photons per Measurement', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM vs Photon Flux', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Add photon-starved region
    axes[1].axvspan(1, 10, alpha=0.2, color='red', label='Photon-starved')
    axes[1].axvline(x=10, color='red', linestyle='--', alpha=0.5)
    axes[1].text(3, 0.1, 'Photon\nstarved', fontsize=10, ha='center', color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_dark_counts(results, save_path='outputs/quantum_dark_counts.png'):
    """Plot dark count robustness results."""
    
    dcrs = sorted(results.keys())
    ssim_vals = [results[d]['ssim_mean'] for d in dcrs]
    sdrs = [results[d]['signal_to_dark_ratio'] for d in dcrs]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.semilogx(dcrs, ssim_vals, 'go-', linewidth=2, markersize=10, label='SSIM')
    ax1.set_xlabel('Dark Count Rate (Hz)', fontsize=12)
    ax1.set_ylabel('SSIM', fontsize=12, color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.semilogx(dcrs, sdrs, 'b^--', linewidth=2, markersize=8, label='Signal/Dark Ratio')
    ax2.set_ylabel('Signal-to-Dark Ratio', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_yscale('log')
    
    # Add detector typical values
    ax1.axvline(x=10, color='green', linestyle=':', alpha=0.7)
    ax1.text(10, 0.95, 'SNSPD', fontsize=9, ha='center', color='green')
    ax1.axvline(x=1000, color='orange', linestyle=':', alpha=0.7)
    ax1.text(1000, 0.95, 'SPAD', fontsize=9, ha='center', color='orange')
    ax1.axvline(x=100000, color='red', linestyle=':', alpha=0.7)
    ax1.text(100000, 0.95, 'SiPM', fontsize=9, ha='center', color='red')
    
    plt.title('Impact of Dark Count Rate on Reconstruction Quality', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_snr_comparison(results, save_path='outputs/quantum_vs_classical_snr.png'):
    """Plot quantum vs classical SNR comparison."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    snr_levels = sorted(results['SNSPD'].keys())
    
    snspd_ssim = [results['SNSPD'][s]['ssim_mean'] for s in snr_levels]
    classical_ssim = [results['Classical'][s]['ssim_mean'] for s in snr_levels]
    
    snspd_mse = [results['SNSPD'][s]['mse_mean'] for s in snr_levels]
    classical_mse = [results['Classical'][s]['mse_mean'] for s in snr_levels]
    
    # MSE
    axes[0].semilogy(snr_levels, snspd_mse, 'go-', linewidth=2, markersize=10, 
                     label='SNSPD (Quantum)')
    axes[0].semilogy(snr_levels, classical_mse, 'b^--', linewidth=2, markersize=8,
                     label='Classical')
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('MSE (log scale)', fontsize=12)
    axes[0].set_title('MSE vs SNR', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM
    axes[1].plot(snr_levels, snspd_ssim, 'go-', linewidth=2, markersize=10,
                 label='SNSPD (Quantum)')
    axes[1].plot(snr_levels, classical_ssim, 'b^--', linewidth=2, markersize=8,
                 label='Classical')
    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM vs SNR', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Highlight quantum advantage region
    axes[1].fill_between([5, 15], [0, 0], [1, 1], alpha=0.1, color='green')
    axes[1].text(10, 0.05, 'Quantum\nAdvantage', fontsize=10, ha='center', color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_quantum_evaluation():
    """Run all quantum detection experiments."""
    
    # Setup
    SPECKLE_PATH = 'data/speckle_pattern.pt'
    MODEL_PATH = 'checkpoints/temporal_ghost_gpt_quantum_snspd.pt'
    OUTPUT_DIR = Path('outputs')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
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
    
    patterns_flat = torch.tensor(speckle_patterns).float()
    patterns_flat = patterns_flat.view(num_patterns, -1).to(DEVICE)
    
    # Load model
    print("Loading Temporal Ghost-GPT model...")
    model = TemporalGhostGPT(
        d_in=CONFIG['embedding_dim'],
        d_out=CONFIG['embedding_dim'],
        num_blocks=CONFIG['num_blocks'],
        number_of_heads=CONFIG['num_heads'],
        embedding_dim=CONFIG['embedding_dim'],
        flattened_image_size=CONFIG['image_size'] ** 2,
        context_size=num_patterns,
        final_image_size=CONFIG['image_size'] ** 2,
        seq_length=CONFIG['seq_length']
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    all_results = {}
    
    # Run experiments
    print("\n" + "="*70)
    print("QUANTUM GHOST IMAGING EVALUATION")
    print("="*70)
    
    # Experiment 1: Detector comparison
    results_detectors = experiment_detector_comparison(
        model, speckle_patterns, patterns_flat, DEVICE, num_samples=100
    )
    all_results['detector_comparison'] = results_detectors
    plot_detector_comparison(results_detectors)
    
    # Experiment 2: Photon flux
    results_flux = experiment_photon_flux(
        model, speckle_patterns, patterns_flat, DEVICE, num_samples=50
    )
    all_results['photon_flux'] = results_flux
    plot_photon_flux(results_flux)
    
    # Experiment 3: Dark counts
    results_dark = experiment_dark_counts(
        model, speckle_patterns, patterns_flat, DEVICE, num_samples=50
    )
    all_results['dark_counts'] = results_dark
    plot_dark_counts(results_dark)
    
    # Experiment 4: Detection efficiency
    results_eff = experiment_efficiency(
        model, speckle_patterns, patterns_flat, DEVICE, num_samples=50
    )
    all_results['efficiency'] = results_eff
    
    # Experiment 5: Normalization methods
    results_norm = experiment_normalization(
        model, speckle_patterns, patterns_flat, DEVICE, num_samples=50
    )
    all_results['normalization'] = results_norm
    
    # Experiment 6: SNR comparison
    results_snr = experiment_snr_comparison(
        model, speckle_patterns, patterns_flat, DEVICE, num_samples=50
    )
    all_results['snr_comparison'] = results_snr
    plot_snr_comparison(results_snr)
    
    # Print summary
    print("\n" + "="*70)
    print("QUANTUM EVALUATION SUMMARY")
    print("="*70)
    
    print("\n1. Detector Comparison (100 mean photons):")
    print(f"   {'Detector':<15} {'MSE':<15} {'SSIM':<15}")
    print("   " + "-"*45)
    for det, res in results_detectors.items():
        print(f"   {det:<15} {res['mse_mean']:.4f}±{res['mse_std']:.4f}   {res['ssim_mean']:.4f}±{res['ssim_std']:.4f}")
    
    print("\n2. Best Normalization Method:")
    best_norm = max(results_norm.items(), key=lambda x: x[1]['ssim_mean'])
    print(f"   {best_norm[0]}: SSIM = {best_norm[1]['ssim_mean']:.4f}")
    
    print("\n3. Minimum Photons for SSIM > 0.8:")
    for det, data in results_flux.items():
        for n_photons in sorted(data.keys()):
            if data[n_photons]['ssim_mean'] > 0.8:
                print(f"   {det}: {n_photons} photons")
                break
    
    # Save results
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj
    
    with open(OUTPUT_DIR / 'quantum_evaluation_results.json', 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR / 'quantum_evaluation_results.json'}")
    
    return all_results


if __name__ == "__main__":
    results = run_quantum_evaluation()