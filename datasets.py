import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os


# ============================================================================
# 1. MOVING MNIST GHOST IMAGING DATASET (unchanged)
# ============================================================================

class MovingMNISTGhost(torch.utils.data.Dataset):
    def __init__(self, speckle_patterns, seq_length=8, image_size=256,
                 dataset_size=10000, train=True):
        """
        speckle_patterns: numpy array [M, H, W] - your 188 patterns from Drive
        """
        self.H = torch.tensor(speckle_patterns).float()  # [M, 256, 256]
        self.M = self.H.shape[0]
        self.seq_length = seq_length
        self.image_size = image_size
        self.dataset_size = dataset_size

        # Flatten patterns for bucket computation
        self.H_flat = self.H.view(self.M, -1)  # [M, H*W]

        # Load MNIST
        self.mnist = datasets.MNIST(
            root='./data', train=train, download=True,
            transform=transforms.ToTensor()
        )

    def generate_trajectory(self, motion_type='linear'):
        T = self.seq_length
        if motion_type == 'linear':
            angle = np.random.uniform(0, 2*np.pi)
            speed = np.random.uniform(2, 8)
            dx = speed * np.cos(angle) * np.arange(T)
            dy = speed * np.sin(angle) * np.arange(T)
        elif motion_type == 'oscillatory':
            freq = np.random.uniform(0.5, 2)
            amplitude = np.random.uniform(10, 30)
            t = np.linspace(0, 2*np.pi*freq, T)
            dx = amplitude * np.sin(t)
            dy = amplitude * 0.3 * np.cos(t)
        elif motion_type == 'random_walk':
            steps = np.random.randn(T, 2) * 5
            dx = np.cumsum(steps[:, 0])
            dy = np.cumsum(steps[:, 1])
        return np.stack([dx, dy], axis=1).astype(np.float32)

    def warp_image(self, img, displacement):
        """Translate image by (dx, dy) pixels"""
        dx, dy = displacement
        theta = torch.tensor([
            [1, 0, -2*dx/self.image_size],
            [0, 1, -2*dy/self.image_size]
        ]).float().unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1, self.image_size, self.image_size),
                             align_corners=False)
        img_4d = img.unsqueeze(0).unsqueeze(0)
        warped = F.grid_sample(img_4d, grid, align_corners=False,
                               padding_mode='zeros')
        return warped.squeeze()

    def compute_buckets(self, image):
        """bucket_m = sum(H_m * image)"""
        img_flat = image.view(-1)       # [H*W]
        buckets = self.H_flat @ img_flat  # [M]
        return buckets

    def __getitem__(self, idx):
        # Get random MNIST digit
        mnist_idx = idx % len(self.mnist)
        img, label = self.mnist[mnist_idx]

        # Resize to target size
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size),
                            mode='bilinear', align_corners=False).squeeze()

        # Generate motion
        motion_type = np.random.choice(['linear', 'oscillatory', 'random_walk'])
        trajectory = self.generate_trajectory(motion_type)

        # Generate sequence
        frames = []
        buckets = []
        for t in range(self.seq_length):
            frame_t = self.warp_image(img, trajectory[t])
            bucket_t = self.compute_buckets(frame_t)
            frames.append(frame_t)
            buckets.append(bucket_t)

        return {
            'buckets': torch.stack(buckets),        # [T, M]
            'frames': torch.stack(frames),           # [T, H, W]
            'trajectory': torch.tensor(trajectory),
            'label': label
        }

    def __len__(self):
        return self.dataset_size


# ============================================================================
# 2. MOVING CIFAR-10 GHOST IMAGING DATASET
# ============================================================================

class MovingCIFAR10Ghost(torch.utils.data.Dataset):
    """
    Moving CIFAR-10 ghost imaging dataset.
    Drop-in replacement for MovingMNISTGhost — identical interface.
    CIFAR-10 images are converted to grayscale and resized to image_size.
    """

    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, speckle_patterns, seq_length=8, image_size=256,
                 dataset_size=10000, train=True):
        self.H = torch.tensor(speckle_patterns).float()
        self.M = self.H.shape[0]
        self.seq_length = seq_length
        self.image_size = image_size
        self.dataset_size = dataset_size
        self.H_flat = self.H.view(self.M, -1)

        self.cifar = datasets.CIFAR10(
            root='./data', train=train, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),                        # [1, 32, 32]
            ])
        )

    def generate_trajectory(self, motion_type='linear'):
        T = self.seq_length
        if motion_type == 'linear':
            angle = np.random.uniform(0, 2*np.pi)
            speed = np.random.uniform(2, 8)
            dx = speed * np.cos(angle) * np.arange(T)
            dy = speed * np.sin(angle) * np.arange(T)
        elif motion_type == 'oscillatory':
            freq = np.random.uniform(0.5, 2)
            amplitude = np.random.uniform(10, 30)
            t = np.linspace(0, 2*np.pi*freq, T)
            dx = amplitude * np.sin(t)
            dy = amplitude * 0.3 * np.cos(t)
        elif motion_type == 'random_walk':
            steps = np.random.randn(T, 2) * 5
            dx = np.cumsum(steps[:, 0])
            dy = np.cumsum(steps[:, 1])
        return np.stack([dx, dy], axis=1).astype(np.float32)

    def warp_image(self, img, displacement):
        dx, dy = displacement
        theta = torch.tensor([
            [1, 0, -2*dx/self.image_size],
            [0, 1, -2*dy/self.image_size]
        ]).float().unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1, self.image_size, self.image_size),
                             align_corners=False)
        warped = F.grid_sample(
            img.unsqueeze(0).unsqueeze(0), grid,
            align_corners=False, padding_mode='zeros'
        )
        return warped.squeeze()

    def compute_buckets(self, image):
        img_flat = image.view(-1)
        buckets = self.H_flat @ img_flat
        return buckets

    def __getitem__(self, idx):
        cifar_idx = idx % len(self.cifar)
        img, label = self.cifar[cifar_idx]              # [1, 32, 32]

        # Resize 32x32 -> image_size
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode='bilinear', align_corners=False
        ).squeeze()                                      # [image_size, image_size]

        motion_type = np.random.choice(['linear', 'oscillatory', 'random_walk'])
        trajectory = self.generate_trajectory(motion_type)

        frames = []
        buckets = []
        for t in range(self.seq_length):
            frame_t = self.warp_image(img, trajectory[t])
            bucket_t = self.compute_buckets(frame_t)
            frames.append(frame_t)
            buckets.append(bucket_t)

        return {
            'buckets': torch.stack(buckets),            # [T, M]
            'frames': torch.stack(frames),               # [T, H, W]
            'trajectory': torch.tensor(trajectory),
            'label': label,
            'class_name': self.CLASSES[label],
        }

    def __len__(self):
        return self.dataset_size


# ============================================================================
# 3. KVASIR ENDOSCOPY GHOST IMAGING DATASET
# ============================================================================
# Citation:
#   Pogorelov et al., "KVASIR: A Multi-Class Image Dataset for
#   Computer Aided Gastrointestinal Disease Detection", MMSys 2017.
# ============================================================================

class KvasirGhost(torch.utils.data.Dataset):
    """
    Kvasir endoscopy ghost imaging dataset.
    Simulates ghost imaging measurements from real colonoscopy frames,
    directly supporting the paper's endomicroscopy application claim.

    Motion scale is kept small (1-5 px default) to mimic physiological
    micro-motion from heartbeat and respiration during endoscopic acquisition.
    """

    CLASSES = [
        'dyed-lifted-polyps',
        'dyed-resection-margins',
        'esophagitis',
        'normal-cecum',
        'normal-pylorus',
        'normal-z-line',
        'polyps',
        'ulcerative-colitis',
    ]

    def __init__(self, speckle_patterns, seq_length=8, image_size=256,
                 dataset_size=2000, train=True,
                 kvasir_root='./data/kvasir-v2',
                 motion_scale=5.0):
        self.H = torch.tensor(speckle_patterns).float()
        self.M = self.H.shape[0]
        self.seq_length = seq_length
        self.image_size = image_size
        self.dataset_size = dataset_size
        self.motion_scale = motion_scale
        self.H_flat = self.H.view(self.M, -1)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                          # [1, H, W] in [0,1]
        ])

        # Collect all image paths
        self.image_paths = []
        self.image_labels = []
        for class_idx, class_name in enumerate(self.CLASSES):
            class_dir = os.path.join(kvasir_root, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Kvasir class folder not found: {class_dir}\n"
                    f"Download from https://datasets.simula.no/kvasir/ "
                    f"and unzip to {kvasir_root}"
                )
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.image_labels.append(class_idx)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {kvasir_root}. "
                "Check the folder structure."
            )

        # Deterministic 80/20 train/val split
        split_idx = int(0.8 * len(self.image_paths))
        if train:
            self.image_paths  = self.image_paths[:split_idx]
            self.image_labels = self.image_labels[:split_idx]
        else:
            self.image_paths  = self.image_paths[split_idx:]
            self.image_labels = self.image_labels[split_idx:]

        print(
            f"KvasirGhost: {len(self.image_paths)} images "
            f"({'train' if train else 'val'}), "
            f"dataset_size={dataset_size}"
        )

    def generate_trajectory(self, motion_type='linear'):
        T = self.seq_length
        s = self.motion_scale

        if motion_type == 'linear':
            angle = np.random.uniform(0, 2*np.pi)
            speed = np.random.uniform(0.5, s)
            dx = speed * np.cos(angle) * np.arange(T)
            dy = speed * np.sin(angle) * np.arange(T)
        elif motion_type == 'oscillatory':
            # Models breathing / heartbeat periodicity
            freq = np.random.uniform(0.5, 2.0)
            amplitude = np.random.uniform(1, s)
            t = np.linspace(0, 2*np.pi*freq, T)
            dx = amplitude * np.sin(t)
            dy = amplitude * 0.4 * np.cos(t)
        elif motion_type == 'random_walk':
            steps = np.random.randn(T, 2) * (s / 4)
            dx = np.cumsum(steps[:, 0])
            dy = np.cumsum(steps[:, 1])

        return np.stack([dx, dy], axis=1).astype(np.float32)

    def warp_image(self, img, displacement):
        dx, dy = displacement
        theta = torch.tensor([
            [1, 0, -2*dx/self.image_size],
            [0, 1, -2*dy/self.image_size]
        ]).float().unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1, self.image_size, self.image_size),
                             align_corners=False)
        warped = F.grid_sample(
            img.unsqueeze(0).unsqueeze(0), grid,
            align_corners=False,
            padding_mode='reflection'   # reflection > zeros for natural images
        )
        return warped.squeeze()

    def compute_buckets(self, image):
        img_flat = image.view(-1)
        buckets = self.H_flat @ img_flat
        return buckets

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_paths)
        img = Image.open(self.image_paths[img_idx]).convert('RGB')
        img = self.transform(img).squeeze(0)        # [image_size, image_size]
        label = self.image_labels[img_idx]

        motion_type = np.random.choice(['linear', 'oscillatory', 'random_walk'])
        trajectory = self.generate_trajectory(motion_type)

        frames = []
        buckets = []
        for t in range(self.seq_length):
            frame_t = self.warp_image(img, trajectory[t])
            bucket_t = self.compute_buckets(frame_t)
            frames.append(frame_t)
            buckets.append(bucket_t)

        return {
            'buckets': torch.stack(buckets),            # [T, M]
            'frames': torch.stack(frames),               # [T, H, W]
            'trajectory': torch.tensor(trajectory),
            'label': label,
            'class_name': self.CLASSES[label],
            'image_path': self.image_paths[img_idx],
        }

    def __len__(self):
        return self.dataset_size


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':

    speckle_patterns = torch.load('data/speckle_pattern.pt').numpy()

    # MNIST (unchanged)
    mnist_dataset = MovingMNISTGhost(
        speckle_patterns=speckle_patterns,
        seq_length=8, image_size=256, dataset_size=10000, train=True
    )
    s = mnist_dataset[0]
    print(f"MNIST   — buckets: {s['buckets'].shape}, frames: {s['frames'].shape}")

    # CIFAR-10
    cifar_dataset = MovingCIFAR10Ghost(
        speckle_patterns=speckle_patterns,
        seq_length=8, image_size=256, dataset_size=10000, train=True
    )
    s = cifar_dataset[0]
    print(f"CIFAR10 — buckets: {s['buckets'].shape}, frames: {s['frames'].shape}, "
          f"class: {s['class_name']}")

    # Kvasir
    kvasir_dataset = KvasirGhost(
        speckle_patterns=speckle_patterns,
        seq_length=8, image_size=256, dataset_size=2000, train=True,
        kvasir_root='./data/kvasir-v2', motion_scale=5.0
    )
    s = kvasir_dataset[0]
    print(f"Kvasir  — buckets: {s['buckets'].shape}, frames: {s['frames'].shape}, "
          f"class: {s['class_name']}")