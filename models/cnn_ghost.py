import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGhost(nn.Module):
    """
    Simple CNN baseline for ghost imaging reconstruction.
    Input:  bucket measurements [B, T, M]
    Output: reconstructed frames [B, T, H, W]
    Processes each frame independently — no temporal modelling.
    """

    def __init__(self, num_patterns, image_size=256, seq_length=8,
                 hidden_dim=512):
        super().__init__()
        self.image_size  = image_size
        self.seq_length  = seq_length
        self.num_patterns = num_patterns

        # Per-frame encoder: M -> hidden_dim -> spatial feature map
        self.encoder = nn.Sequential(
            nn.Linear(num_patterns, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16 * 16 * 16),   # 16 channels, 16x16 spatial
            nn.ReLU(),
        )

        # Decoder: upsample from 16x16 to image_size x image_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 128, 4, 2, 1),   # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 256x256
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),              # 256x256, 1 channel
            nn.Sigmoid(),
        )

    def forward(self, buckets):
        """
        buckets: [B, T, M]
        returns: [B, T, H, W]
        """
        B, T, M = buckets.shape
        # Process all frames in one batch pass
        x = buckets.view(B * T, M)           # [B*T, M]
        x = self.encoder(x)                  # [B*T, 16*16*16]
        x = x.view(B * T, 16, 16, 16)        # [B*T, 16, 16, 16]
        x = self.decoder(x)                  # [B*T, 1, H, W]
        x = x.view(B, T, self.image_size, self.image_size)
        return x