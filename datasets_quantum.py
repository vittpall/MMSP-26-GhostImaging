"""
Quantum Ghost Imaging Dataset
=============================
Simulates photon counting statistics for quantum detectors.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class QuantumMovingMNISTGhost(Dataset):
    """
    Moving MNIST dataset with quantum detection simulation.
    
    Models:
    - Poisson statistics for photon counting
    - Dark counts
    - Detection efficiency
    - Timing jitter (optional)
    """
    
    def __init__(self, speckle_patterns, seq_length=8, image_size=256,
                 dataset_size=5000, train=True,
                 # Quantum detector parameters
                 mean_photons_per_pattern=100,  # Average photons per measurement
                 detection_efficiency=0.9,       # Detector efficiency (0-1)
                 dark_count_rate=100,            # Dark counts per second
                 integration_time=1e-3,          # Integration time in seconds
                 timing_jitter=0,                # Timing jitter in seconds (0 = no timing)
                 detector_type='snspd'):         # 'snspd', 'spad', 'sipm'
        
        self.H = torch.tensor(speckle_patterns).float()
        self.M = self.H.shape[0]
        self.seq_length = seq_length
        self.image_size = image_size
        self.dataset_size = dataset_size
        
        # Quantum parameters
        self.mean_photons = mean_photons_per_pattern
        self.efficiency = detection_efficiency
        self.dark_count_rate = dark_count_rate
        self.integration_time = integration_time
        self.timing_jitter = timing_jitter
        self.detector_type = detector_type
        
        # Expected dark counts per measurement
        self.dark_counts_per_measurement = dark_count_rate * integration_time
        
        # Flatten speckle patterns for measurement computation
        self.H_flat = self.H.view(self.M, -1)
        
        # Load MNIST
        self.mnist = datasets.MNIST(
            root='./data', train=train, download=True,
            transform=transforms.ToTensor()
        )
        
        # Set detector-specific parameters
        self._set_detector_params()
    
    def _set_detector_params(self):
        """Set realistic parameters based on detector type."""
        if self.detector_type == 'snspd':
            # Superconducting Nanowire Single-Photon Detector
            self.efficiency = min(self.efficiency, 0.95)  # Max ~95%
            self.dark_count_rate = min(self.dark_count_rate, 100)  # Very low
            self.timing_jitter = max(self.timing_jitter, 50e-12)  # ~50ps
            self.dead_time = 40e-9  # 40ns dead time
            
        elif self.detector_type == 'spad':
            # Single-Photon Avalanche Diode
            self.efficiency = min(self.efficiency, 0.7)  # Max ~70%
            self.dark_count_rate = max(self.dark_count_rate, 1000)  # Higher noise
            self.timing_jitter = max(self.timing_jitter, 300e-12)  # ~300ps
            self.dead_time = 50e-9  # 50ns dead time
            self.afterpulse_prob = 0.01  # 1% afterpulsing
            
        elif self.detector_type == 'sipm':
            # Silicon Photomultiplier
            self.efficiency = min(self.efficiency, 0.5)  # PDE ~50%
            self.dark_count_rate = max(self.dark_count_rate, 100000)  # High noise
            self.timing_jitter = max(self.timing_jitter, 100e-12)  # ~100ps
            self.crosstalk_prob = 0.05  # 5% optical crosstalk
    
    def _simulate_photon_detection(self, true_intensity):
        """
        Simulate photon counting measurement.
        
        Args:
            true_intensity: Normalized intensity (0-1)
            
        Returns:
            photon_count: Integer number of detected photons
        """
        # Expected photons based on intensity
        expected_photons = true_intensity * self.mean_photons
        
        # Apply detection efficiency
        expected_detected = expected_photons * self.efficiency
        
        # Sample from Poisson distribution
        signal_counts = np.random.poisson(expected_detected)
        
        # Add dark counts
        dark_counts = np.random.poisson(self.dark_counts_per_measurement)
        
        # Detector-specific effects
        if self.detector_type == 'spad' and hasattr(self, 'afterpulse_prob'):
            # Afterpulsing adds extra counts proportional to signal
            afterpulses = np.random.binomial(signal_counts, self.afterpulse_prob)
            signal_counts += afterpulses
            
        elif self.detector_type == 'sipm' and hasattr(self, 'crosstalk_prob'):
            # Optical crosstalk
            crosstalk = np.random.binomial(signal_counts, self.crosstalk_prob)
            signal_counts += crosstalk
        
        total_counts = signal_counts + dark_counts
        
        return total_counts
    
    def _simulate_photon_arrivals(self, true_intensity, num_time_bins=100):
        """
        Simulate photon arrival times (for time-resolved detection).
        
        Returns:
            arrival_histogram: Binned photon arrival times
        """
        expected_photons = true_intensity * self.mean_photons * self.efficiency
        n_photons = np.random.poisson(expected_photons)
        
        if n_photons == 0:
            return np.zeros(num_time_bins)
        
        # Generate uniform arrival times within integration window
        arrival_times = np.random.uniform(0, self.integration_time, n_photons)
        
        # Add timing jitter
        if self.timing_jitter > 0:
            jitter = np.random.normal(0, self.timing_jitter, n_photons)
            arrival_times += jitter
            arrival_times = np.clip(arrival_times, 0, self.integration_time)
        
        # Bin into histogram
        histogram, _ = np.histogram(arrival_times, bins=num_time_bins, 
                                     range=(0, self.integration_time))
        
        return histogram
    
    def compute_quantum_buckets(self, image):
        """
        Compute bucket measurements with quantum detection.
        
        Args:
            image: [H, W] tensor
            
        Returns:
            buckets: [M] tensor of photon counts (integers)
        """
        img_flat = image.view(-1).numpy()
        
        # Compute true intensities (correlation with patterns)
        # Normalize to 0-1 range
        true_intensities = self.H_flat.numpy() @ img_flat
        true_intensities = (true_intensities - true_intensities.min()) / \
                          (true_intensities.max() - true_intensities.min() + 1e-8)
        
        # Simulate quantum detection for each pattern
        photon_counts = np.array([
            self._simulate_photon_detection(intensity) 
            for intensity in true_intensities
        ])
        
        return torch.tensor(photon_counts).float()
    
    def __getitem__(self, idx):
        # Get MNIST digit
        mnist_idx = idx % len(self.mnist)
        img, label = self.mnist[mnist_idx]
        
        # Resize to target size
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size),
                           mode='bilinear', align_corners=False).squeeze()
        
        # Generate motion trajectory
        trajectory = self._generate_trajectory()
        
        # Generate frames and quantum bucket measurements
        frames = []
        buckets = []
        
        for t in range(self.seq_length):
            frame_t = self._warp_image(img, trajectory[t])
            bucket_t = self.compute_quantum_buckets(frame_t)
            frames.append(frame_t)
            buckets.append(bucket_t)
        
        return {
            'buckets': torch.stack(buckets),      # [T, M] photon counts
            'frames': torch.stack(frames),         # [T, H, W]
            'label': label,
            'trajectory': torch.tensor(trajectory),
            'detector_type': self.detector_type,
            'mean_photons': self.mean_photons,
        }
    
    def _generate_trajectory(self):
        """Generate random motion trajectory."""
        T = self.seq_length
        motion_type = np.random.choice(['linear', 'oscillatory', 'random_walk'])
        speed = np.random.uniform(1, 10)
        
        if motion_type == 'linear':
            angle = np.random.uniform(0, 2 * np.pi)
            dx = speed * np.cos(angle) * np.arange(T)
            dy = speed * np.sin(angle) * np.arange(T)
        elif motion_type == 'oscillatory':
            freq = np.random.uniform(0.5, 1.5)
            t = np.linspace(0, 2 * np.pi * freq, T)
            dx = speed * 3 * np.sin(t)
            dy = speed * np.cos(t)
        else:  # random_walk
            steps = np.random.randn(T, 2) * speed * 0.5
            dx = np.cumsum(steps[:, 0])
            dy = np.cumsum(steps[:, 1])
        
        return np.stack([dx, dy], axis=1).astype(np.float32)
    
    def _warp_image(self, img, displacement):
        """Apply displacement to image."""
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
    
    def __len__(self):
        return self.dataset_size


class PhotonCountNormalizer:
    """
    Normalize photon counts for neural network input.
    
    Options:
    1. Log transform: log(1 + counts)
    2. Square root: sqrt(counts) - variance stabilizing
    3. Anscombe transform: 2 * sqrt(counts + 3/8) - makes Poisson approximately Gaussian
    4. Min-max normalization
    """
    
    def __init__(self, method='anscombe'):
        self.method = method
        self.stats = {}
    
    def fit(self, counts):
        """Compute normalization statistics."""
        if isinstance(counts, torch.Tensor):
            counts = counts.numpy()
        
        self.stats['min'] = counts.min()
        self.stats['max'] = counts.max()
        self.stats['mean'] = counts.mean()
        self.stats['std'] = counts.std()
    
    def transform(self, counts):
        """Apply normalization."""
        if isinstance(counts, np.ndarray):
            counts = torch.tensor(counts).float()
        
        if self.method == 'log':
            return torch.log1p(counts)
        
        elif self.method == 'sqrt':
            return torch.sqrt(counts)
        
        elif self.method == 'anscombe':
            # Anscombe transform: variance-stabilizing for Poisson
            return 2 * torch.sqrt(counts + 3/8)
        
        elif self.method == 'minmax':
            return (counts - self.stats['min']) / \
                   (self.stats['max'] - self.stats['min'] + 1e-8)
        
        elif self.method == 'zscore':
            return (counts - self.stats['mean']) / (self.stats['std'] + 1e-8)
        
        else:
            return counts
    
    def inverse_transform(self, normalized):
        """Reverse normalization (approximate for some methods)."""
        if self.method == 'log':
            return torch.expm1(normalized)
        
        elif self.method == 'sqrt':
            return normalized ** 2
        
        elif self.method == 'anscombe':
            # Inverse Anscombe
            return (normalized / 2) ** 2 - 3/8
        
        elif self.method == 'minmax':
            return normalized * (self.stats['max'] - self.stats['min']) + self.stats['min']
        
        else:
            return normalized