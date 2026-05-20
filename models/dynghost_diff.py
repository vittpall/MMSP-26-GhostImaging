import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from models.temporal_ghost_gpt import TemporalGhostGPT


# ============================================================================
# Cosine noise schedule
# ============================================================================

def cosine_beta_schedule(T, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps            = T + 1
    t                = torch.linspace(0, T, steps, dtype=torch.float64)
    alphas_cumprod   = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod   = alphas_cumprod / alphas_cumprod[0]
    betas            = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.9999).float()


# ============================================================================
# Sinusoidal timestep embedding
# ============================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] integer timesteps
        device   = t.device
        half_dim = self.dim // 2
        freq     = math.log(10000) / (half_dim - 1)
        freq     = torch.exp(torch.arange(half_dim, device=device) * -freq)
        emb      = t.float()[:, None] * freq[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, dim]


# ============================================================================
# FiLM conditioning
# ============================================================================

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale-and-shift a spatial feature map."""

    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.scale_shift = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        # x: [B, C, H, W]   cond: [B, cond_dim]
        params        = self.scale_shift(cond)         # [B, 2*C]
        gamma, beta   = params.chunk(2, dim=-1)        # each [B, C]
        gamma         = gamma[:, :, None, None]
        beta          = beta[:, :, None, None]
        return x * (1 + gamma) + beta


# ============================================================================
# Residual convolutional block with FiLM conditioning
# ============================================================================

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film  = FiLM(cond_dim, out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = nn.SiLU()

    def forward(self, x, cond):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, cond)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# ============================================================================
# Lightweight conditional U-Net denoiser  (~10–20 M params for base_ch=64)
# ============================================================================
class ConditionalUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, cond_dim=256,
                 time_dim=128, spatial_size=8):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]
        self.spatial_size = spatial_size

        # Timestep embedding — unchanged
        self.time_emb  = SinusoidalTimestepEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Spatial map projectors — one per resolution level
        # Resize spatial map [B, cond_dim, 8, 8] → [B, ch_i, H_i, W_i]
        # then add to feature map as a spatial bias
        self.sp0 = nn.Conv2d(cond_dim, ch[0], 1)  # full res
        self.sp1 = nn.Conv2d(cond_dim, ch[1], 1)  # /2
        self.sp2 = nn.Conv2d(cond_dim, ch[2], 1)  # /4
        self.sp3 = nn.Conv2d(cond_dim, ch[3], 1)  # /8

        # ResBlocks — unchanged, still use global time+cond via FiLM
        self.enc0 = ResBlock(in_ch,  ch[0], cond_dim)
        self.enc1 = ResBlock(ch[0],  ch[1], cond_dim)
        self.enc2 = ResBlock(ch[1],  ch[2], cond_dim)
        self.enc3 = ResBlock(ch[2],  ch[3], cond_dim)
        self.mid  = ResBlock(ch[3],  ch[3], cond_dim)
        self.dec3 = ResBlock(ch[3] + ch[3], ch[2], cond_dim)
        self.dec2 = ResBlock(ch[2] + ch[2], ch[1], cond_dim)
        self.dec1 = ResBlock(ch[1] + ch[1], ch[0], cond_dim)
        self.dec0 = ResBlock(ch[0] + ch[0], ch[0], cond_dim)
        self.out_conv = nn.Conv2d(ch[0], in_ch, 1)

        self.down = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False)

    def forward(self, x, tau, c, spatial_map=None):
        """
        x:           [B, 1, H, W]
        tau:         [B]
        c:           [B, cond_dim]       global conditioning (time + backbone)
        spatial_map: [B, cond_dim, 8, 8] spatial conditioning (optional)
        """
        t_emb = self.time_proj(self.time_emb(tau))
        cond  = t_emb + c                            # [B, cond_dim]

        # Helper: resize spatial map to target spatial size and project
        def spatial_bias(proj, target_h, target_w):
            if spatial_map is None:
                return 0
            sm = F.interpolate(spatial_map, size=(target_h, target_w),
                               mode='bilinear', align_corners=False)
            return proj(sm)   # [B, ch_i, target_h, target_w]

        # Encoder — ResBlock output + spatial bias at each level
        H, W = x.shape[2], x.shape[3]
        e0 = self.enc0(x,             cond) + spatial_bias(self.sp0, H,   W  )
        e1 = self.enc1(self.down(e0), cond) + spatial_bias(self.sp1, H//2, W//2)
        e2 = self.enc2(self.down(e1), cond) + spatial_bias(self.sp2, H//4, W//4)
        e3 = self.enc3(self.down(e2), cond) + spatial_bias(self.sp3, H//8, W//8)

        m  = self.mid(self.down(e3),  cond) + spatial_bias(self.sp3, H//16, W//16)

        # Decoder — skip connections unchanged
        d3 = self.dec3(torch.cat([self.up(m),  e3], dim=1), cond)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), cond)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), cond)
        d0 = self.dec0(torch.cat([self.up(d1), e0], dim=1), cond)

        return self.out_conv(d0)

        
# ============================================================================
# Backbone conditioning projection
# ============================================================================

class ConditioningProjection(nn.Module):
    """
    Projects per-frame backbone tokens [B, M, d_out] to a single conditioning
    vector [B, cond_dim] via LayerNorm → average pooling → linear projection.
    """

    def __init__(self, d_out, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_out)
        self.proj = nn.Linear(d_out, cond_dim)

    def forward(self, tokens):
        # tokens: [B, M, d_out]
        x = self.norm(tokens)    # [B, M, d_out]
        x = x.mean(dim=1)        # [B, d_out]  — average over measurement tokens
        return self.proj(x)      # [B, cond_dim]

class SpatialConditioningProjection(nn.Module):
    """
    Instead of collapsing all tokens to one vector,
    produce a spatial feature map [B, cond_dim, H', W']
    that the UNet can inject at each resolution level.
    """
    def __init__(self, d_out, cond_dim, spatial_size=8):
        super().__init__()
        self.spatial_size = spatial_size   # 8x8 = 64 spatial tokens
        self.norm   = nn.LayerNorm(d_out)
        self.proj   = nn.Linear(d_out, cond_dim)
        # Learnable spatial query: maps M tokens → spatial_size² tokens
        self.spatial_attn = nn.MultiheadAttention(d_out, num_heads=4,
                                                   batch_first=True)
        self.queries = nn.Parameter(
            torch.randn(1, spatial_size * spatial_size, d_out)
        )

    def forward(self, tokens):
        # tokens: [B, M, d_out]
        B = tokens.shape[0]
        x = self.norm(tokens)
        q = self.queries.expand(B, -1, -1)              # [B, S², d_out]
        out, _ = self.spatial_attn(q, x, x)             # [B, S², d_out]
        out = self.proj(out)                             # [B, S², cond_dim]
        H = W = self.spatial_size
        return out.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, cond_dim, H, W]

# ============================================================================
# DynGhostDiff
# ============================================================================

class DynGhostDiff(TemporalGhostGPT):
    """
    DynGhostDiff — DynGhost with an optional diffusion-based reconstruction head.

    diffusion=False : identical to TemporalGhostGPT; accepts its weights with
                      strict=True since no new parameters are introduced.
    diffusion=True  : replaces the final MLP output head with a conditional
                      U-Net denoiser trained with the DDPM objective and using
                      DDIM sampling at inference time.

    Constructor args (beyond TemporalGhostGPT):
        diffusion        bool    — enable diffusion head (default True)
        diffusion_steps  int     — T in DDPM (default 1000)
        base_ch          int     — base channel width of the U-Net (default 64)
        cond_dim         int     — conditioning vector dimension (default 256)
        time_dim         int     — sinusoidal timestep embedding dimension (default 128)
    """

    def __init__(self, d_in, d_out, num_blocks, number_of_heads=12,
                embedding_dim=5, flattened_image_size=106 * 106,
                context_size=154, final_image_size=256 * 256,
                seq_length=8, diffusion=True,
                diffusion_steps=1000, base_ch=64, cond_dim=256,
                time_dim=128, spatial_size=8):

        super().__init__(
            d_in=d_in, d_out=d_out, num_blocks=num_blocks,
            number_of_heads=number_of_heads, embedding_dim=embedding_dim,
            flattened_image_size=flattened_image_size,
            context_size=context_size, final_image_size=final_image_size,
            seq_length=seq_length,
        )

        self.diffusion       = diffusion
        self.diffusion_steps = diffusion_steps
        self.image_size      = int(math.sqrt(final_image_size))
        self.cond_dim        = cond_dim

        if diffusion:
            # Spatial conditioning: [B, M, d_out] → [B, cond_dim, 8, 8]
            self.cond_proj = SpatialConditioningProjection(
                d_out=d_out, cond_dim=cond_dim, spatial_size=spatial_size,
            )
            # Global conditioning: [B, d_out] → [B, cond_dim] for FiLM
            self.cond_proj_global = nn.Sequential(
                nn.LayerNorm(d_out),
                nn.Linear(d_out, cond_dim),
            )
            self.unet = ConditionalUNet(
                in_ch=1, base_ch=base_ch, cond_dim=cond_dim,
                time_dim=time_dim, spatial_size=spatial_size,
            )
            self._register_schedule(diffusion_steps)
    # ------------------------------------------------------------------
    # Noise schedule helpers
    # ------------------------------------------------------------------

    def _register_schedule(self, T):
        betas              = cosine_beta_schedule(T)
        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas',                          betas)
        self.register_buffer('alphas_cumprod',                 alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',            alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',
                             torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1.0 / alphas_cumprod - 1.0))

    # ------------------------------------------------------------------
    # Backbone (spatial + temporal attention, shared by both modes)
    # ------------------------------------------------------------------

    def _backbone(self, x, bucket_sum):
        """
        Reproduces the spatial + temporal transformer forward pass from
        TemporalGhostGPT.forward, stopping before the MLP output head.

        x:          [M, H*W]        speckle patterns
        bucket_sum: [B, T, M]       bucket measurements

        Returns: tokens [B, T, M, d_out]
        """
        B, T, M = bucket_sum.shape

        # --- pattern + bucket token construction ---
        pattern_embed = self.image_embedding_layer(x)              # [M, embed-1]
        pattern_embed = pattern_embed.unsqueeze(0).unsqueeze(0)    # [1, 1, M, embed-1]
        pattern_embed = pattern_embed.expand(B, T, -1, -1)         # [B, T, M, embed-1]

        bucket_expanded = bucket_sum.unsqueeze(-1)                 # [B, T, M, 1]
        tokens = torch.cat([pattern_embed, bucket_expanded], dim=-1)  # [B, T, M, embed]

        spatial_pos = self.pos_embedding_layer(torch.arange(M, device=x.device))
        tokens      = tokens + spatial_pos.unsqueeze(0).unsqueeze(0)

        # --- spatial attention (within each frame) ---
        tokens_spatial = tokens.view(B * T, M, -1)
        for module in self.main_body:
            tokens_spatial = module(tokens_spatial)
        tokens = tokens_spatial.view(B, T, M, -1)

        # --- temporal attention (across frames) ---
        tokens_temporal = tokens.permute(0, 2, 1, 3).reshape(B * M, T, -1)
        temporal_pos    = self.temporal_pos_embedding(torch.arange(T, device=x.device))
        tokens_temporal = tokens_temporal + temporal_pos.unsqueeze(0)
        for module in self.temporal_blocks:
            tokens_temporal = module(tokens_temporal)
        tokens = tokens_temporal.view(B, M, T, -1).permute(0, 2, 1, 3)  # [B, T, M, d_out]

        return tokens

    def _head_mlp(self, tokens):
        """
        Original MLP output head — identical to TemporalGhostGPT's output
        projection loop.  Returns [B, T, H, W].
        """
        B, T, M, _ = tokens.shape
        outputs = []
        for t in range(T):
            frame_tokens = tokens[:, t, :, :]
            frame_tokens = self.call_transformer.batch_normalization(frame_tokens)
            frame_tokens = self.final_projection_layer(frame_tokens)   # [B, M, 16]
            frame_tokens = frame_tokens.view(B, -1)                    # [B, M*16]
            frame_out    = self.final_projection_layer2(frame_tokens)  # [B, H*W]
            frame_out    = self.final_sigmoid_layer(frame_out)
            outputs.append(frame_out)
        output = torch.stack(outputs, dim=1)                           # [B, T, H*W]
        H = W  = self.image_size
        return output.view(B, T, H, W)

    # ------------------------------------------------------------------
    # Public forward (compatible with the existing forward() wrapper)
    # ------------------------------------------------------------------
    def forward(self, x, bucket_sum, num_steps=20, start_t=200):
        tokens     = self._backbone(x, bucket_sum)      # [B, T, M, d_out]
        if not self.diffusion:
            return self._head_mlp(tokens)

        det_frames = self._head_mlp(tokens)             # [B, T, H, W]
        B, T       = tokens.shape[:2]
        frames     = []

        for t in range(T):
            frame_tokens = tokens[:, t]                 # [B, M, d_out]

            # Global conditioning: mean pool → linear
            c_global  = self.cond_proj_global(
                frame_tokens.mean(dim=1)
            )                                           # [B, cond_dim]

            # Spatial conditioning: learned queries → 8×8 map
            c_spatial = self.cond_proj(frame_tokens)    # [B, cond_dim, 8, 8]

            det_frame = det_frames[:, t].unsqueeze(1)   # [B, 1, H, W]
            frame     = self.sample(c_global, c_spatial,
                                    num_steps=num_steps,
                                    det_frame=det_frame,
                                    start_t=start_t)
            frames.append(frame.squeeze(1))

        return torch.stack(frames, dim=1)
    # ------------------------------------------------------------------
    # DDPM forward process (noising)
    # ------------------------------------------------------------------

    def q_sample(self, x0, t, noise=None):
        """
        Add noise to x0 at timestep t using the DDPM forward kernel.

        x0:    [B, 1, H, W]   clean image in [-1, 1]
        t:     [B]            integer timestep indices

        Returns (x_tau, noise) both [B, 1, H, W].
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

    # ------------------------------------------------------------------
    # Predict x0 from noisy image (public API)
    # ------------------------------------------------------------------

    def predict_x0(self, x_tau, tau, c):
        """
        Predict the clean image x0 given a noisy observation x_tau, the
        diffusion timestep tau, and the per-frame conditioning vector c.

        x_tau: [B, 1, H, W]
        tau:   [B]            integer timestep indices
        c:     [B, cond_dim]

        Returns x0_pred [B, 1, H, W] in [-1, 1].
        """
        eps_pred     = self.unet(x_tau, tau, c)
        sqrt_recip   = self.sqrt_recip_alphas_cumprod[tau].view(-1, 1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[tau].view(-1, 1, 1, 1)
        return sqrt_recip * x_tau - sqrt_recipm1 * eps_pred

    # ------------------------------------------------------------------
    # DDIM sampling (deterministic, eta=0)
    # ------------------------------------------------------------------
    def sample(self, c_global, c_spatial, num_steps=20,
            det_frame=None, start_t=200):
        B      = c_global.shape[0]
        H = W  = self.image_size
        device = c_global.device

        if det_frame is not None:
            x0_init  = det_frame.to(device) * 2 - 1
            t_tensor = torch.tensor([start_t], device=device)
            x, _     = self.q_sample(x0_init, t_tensor.expand(B))
            step_seq = torch.linspace(start_t, 0, num_steps,
                                    dtype=torch.long, device=device)
        else:
            x        = torch.randn(B, 1, H, W, device=device)
            step_seq = torch.linspace(self.diffusion_steps - 1, 0, num_steps,
                                    dtype=torch.long, device=device)

        with torch.no_grad():
            for i, tau_idx in enumerate(step_seq):
                tau_batch      = tau_idx.expand(B)
                alpha_bar      = self.alphas_cumprod[tau_idx]
                alpha_bar_prev = (self.alphas_cumprod[step_seq[i + 1]]
                                if i + 1 < len(step_seq)
                                else self.alphas_cumprod_prev[0])

                # Pass both global and spatial conditioning to UNet
                eps_pred      = self.unet(x, tau_batch, c_global,
                                        spatial_map=c_spatial)
                x0_pred       = (x - (1 - alpha_bar).sqrt() * eps_pred) \
                                / alpha_bar.sqrt()
                x0_pred       = x0_pred.clamp(-1, 1)
                eps_direction = (x - alpha_bar.sqrt() * x0_pred) \
                                / (1 - alpha_bar).sqrt()
                x             = alpha_bar_prev.sqrt() * x0_pred + \
                                (1 - alpha_bar_prev).sqrt() * eps_direction

        return (x.clamp(-1, 1) + 1) / 2
    # ------------------------------------------------------------------
    # Training-time diffusion forward (Stage 2)
    # ------------------------------------------------------------------

    def forward_diffusion(self, x, bucket_sum, frames):
        """
        Stage 2 training forward pass.  Runs the transformer backbone, samples
        random noising timesteps, applies DDPM forward noise to each ground-truth
        frame, and predicts the noise with the U-Net denoiser.

        x:          [M, H*W]       speckle patterns
        bucket_sum: [B, T, M]      bucket measurements
        frames:     [B, T, H, W]   ground-truth frames in [0, 1]

        Returns
        -------
        eps_preds : [B, T, 1, H, W]   predicted noise
        noises    : [B, T, 1, H, W]   actual added noise
        x_taus    : [B, T, 1, H, W]   noisy frames
        timesteps : [B, T]            sampled integer timesteps
        conds     : [B, T, cond_dim]  per-frame conditioning vectors
        """
        tokens = self._backbone(x, bucket_sum)   # [B, T, M, d_out]
        B, T, H, W = frames.shape

        # Normalize [0, 1] → [-1, 1] for diffusion
        frames_norm = frames * 2 - 1

        timesteps = torch.randint(
            0, self.diffusion_steps, (B, T), device=frames.device
        )

        eps_preds_list, noises_list, x_taus_list, conds_list = [], [], [], []

        for t in range(T):
            frame_tokens = tokens[:, t]
            c_global     = self.cond_proj_global(frame_tokens.mean(dim=1))
            c_spatial    = self.cond_proj(frame_tokens)     # [B, cond_dim, 8, 8]

            frame_t      = frames_norm[:, t].unsqueeze(1)
            tau_t        = timesteps[:, t]
            x_tau, noise = self.q_sample(frame_t, tau_t)

            # Pass spatial map to UNet during training too
            eps_pred     = self.unet(x_tau, tau_t, c_global,
                                    spatial_map=c_spatial)
            eps_preds_list.append(eps_pred)
            noises_list.append(noise)
            x_taus_list.append(x_tau)
            conds_list.append(c_global)

        eps_preds = torch.stack(eps_preds_list, dim=1)    # [B, T, 1, H, W]
        noises    = torch.stack(noises_list,    dim=1)
        x_taus    = torch.stack(x_taus_list,   dim=1)
        conds     = torch.stack(conds_list,    dim=1)    # [B, T, cond_dim]

        return eps_preds, noises, x_taus, timesteps, conds
