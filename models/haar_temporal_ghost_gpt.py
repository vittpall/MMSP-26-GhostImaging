import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from models.temporal_ghost_gpt import TransformerBlock


def _haar_coefficient_sizes(num_patterns, num_levels):
    """
    Compute detail and approximation sizes produced by multi-level Haar DWT.

    Returns:
        detail_sizes: list of ints, detail coefficient counts per level,
                      ordered finest-first [d1_size, d2_size, ..., dL_size]
        c_low_size:   int, number of final approximation coefficients
    """
    M = num_patterns
    detail_sizes = []
    for _ in range(num_levels):
        M_padded = M if M % 2 == 0 else M + 1
        half = M_padded // 2
        detail_sizes.append(half)
        M = half
    return detail_sizes, M


class HaarWaveletTokenizer(nn.Module):
    """
    Multi-resolution tokenizer based on a 1D Haar wavelet decomposition.

    Replaces the [pattern_embed(Hi) || bucket_scalar(bi)] construction in
    TemporalGhostGPT.  Instead, the M-dimensional bucket vector is decomposed
    into approximation + detail coefficients at each scale.  Each resulting
    coefficient becomes a token whose context embedding is a *learned Haar
    basis embedding* (analogous to Hi in the original model).

    Token ordering (coarsest → finest):
        [c_low (size L), d_L (size L), d_{L-1}, ..., d_1 (size M/2)]
    """

    def __init__(self, num_patterns: int, embedding_dim: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        self.embedding_dim = embedding_dim

        detail_sizes, c_low_size = _haar_coefficient_sizes(num_patterns, num_levels)
        self.detail_sizes = detail_sizes          # [d1_size, ..., dL_size]
        self.c_low_size = c_low_size
        # Coarsest-first ordering for token sequence
        self.ordered_sizes = [c_low_size] + list(reversed(detail_sizes))
        self.total_tokens = c_low_size + sum(detail_sizes)

        # Learned basis embedding for each coefficient position — replaces Hi
        self.basis_embedding = nn.Embedding(self.total_tokens, embedding_dim - 1)
        # Standard positional embedding over all tokens
        self.pos_embedding = nn.Embedding(self.total_tokens, embedding_dim)

    def _decompose(self, buckets: torch.Tensor) -> torch.Tensor:
        """
        Apply L-level Haar DWT along the M dimension.

        Args:
            buckets: [B, T, M]
        Returns:
            [B, T, total_tokens] — ordered [c_low, d_L, ..., d_1]
        """
        details = []
        x = buckets
        for _ in range(self.num_levels):
            if x.shape[-1] % 2 != 0:
                x = F.pad(x, (0, 1))
            approx = (x[..., 0::2] + x[..., 1::2]) * (2 ** -0.5)
            detail  = (x[..., 0::2] - x[..., 1::2]) * (2 ** -0.5)
            details.append(detail)
            x = approx
        # x is c_low; details = [d_1 (finest), ..., d_L (coarsest)]
        return torch.cat([x] + list(reversed(details)), dim=-1)

    def forward(self, buckets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            buckets: [B, T, M]
        Returns:
            tokens: [B, T, total_tokens, embedding_dim]
        """
        B, T, _ = buckets.shape
        haar_coeffs = self._decompose(buckets)          # [B, T, total_tokens]

        positions = torch.arange(self.total_tokens, device=buckets.device)
        basis = self.basis_embedding(positions)          # [total_tokens, embed_dim-1]
        basis = basis.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        coeff_vals = haar_coeffs.unsqueeze(-1)           # [B, T, total_tokens, 1]
        tokens = torch.cat([basis, coeff_vals], dim=-1)  # [B, T, total_tokens, embed_dim]

        pos = self.pos_embedding(positions)              # [total_tokens, embed_dim]
        tokens = tokens + pos.unsqueeze(0).unsqueeze(0)

        return tokens


class HaarTemporalGhostGPT(LightningModule):
    """
    Temporal Ghost-GPT with Haar multi-scale tokenization.

    The speckle-pattern embedding is replaced by a HaarWaveletTokenizer:
    bucket measurements are decomposed into multi-resolution Haar coefficients
    which form the token sequence fed to the spatial+temporal transformer.

    The model does NOT require the speckle patterns at inference time.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_blocks: int,
        number_of_heads: int = 8,
        embedding_dim: int = 32,
        num_patterns: int = 188,
        final_image_size: int = 256 * 256,
        seq_length: int = 8,
        num_haar_levels: int = 3,
    ):
        super().__init__()
        self.seq_length = seq_length

        self.tokenizer = HaarWaveletTokenizer(num_patterns, embedding_dim, num_haar_levels)
        total_tokens = self.tokenizer.total_tokens

        # Spatial attention (within each frame)
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(d_in, d_out, number_of_heads)
            for _ in range(num_blocks // 2)
        ])

        # Temporal attention (across frames)
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(d_in, d_out, number_of_heads)
            for _ in range(num_blocks // 2)
        ])

        self.temporal_pos_embedding = nn.Embedding(seq_length, embedding_dim)

        # Output head — reuses call_transformer.batch_normalization pattern
        self.call_transformer = TransformerBlock(d_in, d_out, number_of_heads)
        self.final_projection_layer  = nn.Linear(d_out, 16)
        self.final_projection_layer2 = nn.Linear(total_tokens * 16, final_image_size)
        self.final_sigmoid_layer = nn.Sigmoid()

    def forward(self, buckets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            buckets: [B, T, M]  (speckle patterns NOT required)
        Returns:
            [B, T, H, W]
        """
        B, T, _ = buckets.shape
        total_tokens = self.tokenizer.total_tokens

        # ── Haar multi-scale tokenization ──────────────────────────────────
        tokens = self.tokenizer(buckets)          # [B, T, total_tokens, embed_dim]

        # ── Spatial attention (within each frame) ──────────────────────────
        tokens_s = tokens.view(B * T, total_tokens, -1)
        for block in self.spatial_blocks:
            tokens_s = block(tokens_s)
        tokens = tokens_s.view(B, T, total_tokens, -1)

        # ── Temporal attention (across frames) ────────────────────────────
        tokens_t = tokens.permute(0, 2, 1, 3).reshape(B * total_tokens, T, -1)
        temporal_pos = self.temporal_pos_embedding(
            torch.arange(T, device=buckets.device)
        )
        tokens_t = tokens_t + temporal_pos.unsqueeze(0)
        for block in self.temporal_blocks:
            tokens_t = block(tokens_t)
        tokens = tokens_t.view(B, total_tokens, T, -1).permute(0, 2, 1, 3)

        # ── Per-frame output projection ────────────────────────────────────
        outputs = []
        for t in range(T):
            ft = tokens[:, t, :, :]                          # [B, total_tokens, d_out]
            ft = self.call_transformer.batch_normalization(ft)
            ft = self.final_projection_layer(ft)             # [B, total_tokens, 16]
            ft = ft.view(B, -1)                              # [B, total_tokens*16]
            ft = self.final_projection_layer2(ft)            # [B, H*W]
            outputs.append(self.final_sigmoid_layer(ft))

        output = torch.stack(outputs, dim=1)                 # [B, T, H*W]
        H = W = int(np.sqrt(output.shape[-1]))
        return output.view(B, T, H, W)
