import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetGhost(nn.Module):
    """
    U-Net baseline for ghost imaging reconstruction.
    Input:  bucket measurements [B, T, M]
    Output: reconstructed frames [B, T, H, W]

    Fixes over the original:
    - Gradual projection 188 -> 256 -> 512 -> bottleneck (no 43x jump)
    - BatchNorm throughout to prevent activation collapse
    - Skip connections from projection stages to decoder stages
    - LeakyReLU instead of ReLU to prevent dead neurons
    """

    def __init__(self, num_patterns, image_size=256, seq_length=8,
                 base_ch=64):
        super().__init__()
        self.image_size   = image_size
        self.seq_length   = seq_length
        self.num_patterns = num_patterns
        self.base_ch      = base_ch

        # ----------------------------------------------------------------
        # Encoder: gradual expansion 188 -> base_ch*8 feature vector
        # Each stage saved for skip connections
        # ----------------------------------------------------------------
        self.enc1 = nn.Sequential(
            nn.Linear(num_patterns, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )                                           # -> [B*T, 256]

        self.enc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )                                           # -> [B*T, 512]

        self.enc3 = nn.Sequential(
            nn.Linear(512, base_ch * 8 * 4 * 4),   # -> spatial bottleneck
            nn.BatchNorm1d(base_ch * 8 * 4 * 4),
            nn.LeakyReLU(0.2),
        )                                           # -> [B*T, ch*8, 4, 4]

        # ----------------------------------------------------------------
        # Decoder: upsample 4->8->16->32->64->128->256
        # Skip connections add encoder features at matching resolutions
        # ----------------------------------------------------------------
        # up1: no skip at 4x4 bottleneck
        self.up1 = self._up_block(base_ch * 8,  base_ch * 8)   # 4  -> 8

        # up2-up4: skip connections from enc2/enc1 (broadcast spatially)
        self.up2 = self._up_block(base_ch * 8,  base_ch * 4)   # 8  -> 16
        self.up3 = self._up_block(base_ch * 4,  base_ch * 2)   # 16 -> 32
        self.up4 = self._up_block(base_ch * 2,  base_ch)       # 32 -> 64
        self.up5 = self._up_block(base_ch,       base_ch // 2) # 64 -> 128
        self.up6 = self._up_block(base_ch // 2, base_ch // 4)  # 128-> 256

        # Skip projection layers: flatten enc features -> channel injection
        # enc2 (512) -> injected at up2 output (base_ch*4 spatial)
        self.skip2 = nn.Linear(512, base_ch * 4)
        # enc1 (256) -> injected at up3 output (base_ch*2 spatial)
        self.skip3 = nn.Linear(256, base_ch * 2)

        self.final = nn.Sequential(
            nn.Conv2d(base_ch // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, buckets):
        """
        buckets: [B, T, M]
        returns: [B, T, H, W]
        """
        B, T, M = buckets.shape
        x = buckets.view(B * T, M)

        # Encoder — save intermediates for skip connections
        e1 = self.enc1(x)                          # [B*T, 256]
        e2 = self.enc2(e1)                         # [B*T, 512]
        e3 = self.enc3(e2)                         # [B*T, ch*8*4*4]

        x = e3.view(B * T, self.base_ch * 8, 4, 4) # [B*T, ch*8, 4, 4]

        # Decoder with skip connections
        x = self.up1(x)                            # [B*T, ch*8, 8, 8]
        x = self.up2(x)                            # [B*T, ch*4, 16, 16]

        # Inject enc2 skip: broadcast [B*T, ch*4] -> [B*T, ch*4, 16, 16]
        s2 = self.skip2(e2)                        # [B*T, ch*4]
        x  = x + s2.view(B * T, -1, 1, 1)

        x  = self.up3(x)                           # [B*T, ch*2, 32, 32]

        # Inject enc1 skip
        s3 = self.skip3(e1)                        # [B*T, ch*2]
        x  = x + s3.view(B * T, -1, 1, 1)

        x = self.up4(x)                            # [B*T, ch,   64, 64]
        x = self.up5(x)                            # [B*T, ch/2, 128,128]
        x = self.up6(x)                            # [B*T, ch/4, 256,256]
        x = self.final(x)                          # [B*T, 1,   256,256]

        return x.view(B, T, self.image_size, self.image_size)


# ============================================================================
# Quick sanity check
# ============================================================================
if __name__ == '__main__':
    model   = UNetGhost(num_patterns=188, image_size=256, seq_length=8)
    buckets = torch.randn(2, 8, 188)               # [B, T, M]
    out     = model(buckets)
    print(f"Input:  {buckets.shape}")
    print(f"Output: {out.shape}")                  # should be [2, 8, 256, 256]
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    # Should NOT be all zeros
    assert out.std() > 0.01, "Output collapsed to constant — check architecture"
    print("Sanity check passed")