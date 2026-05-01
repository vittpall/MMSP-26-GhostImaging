"""
Ablation Study for Temporal Ghost-GPT
=====================================
Tests the contribution of each component:
1. Full model (baseline)
2. No temporal attention
3. No temporal positional encoding
4. No SSIM loss (MSE only)
5. No temporal consistency loss
6. Different number of temporal blocks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
import time
import copy
import os

# Import your modules
from models.temporal_ghost_gpt import TemporalGhostGPT, TransformerBlock, MultiHeadedAttention
from datasets import MovingMNISTGhost
from pytorch_lightning import LightningModule


# ============================================================================
# ABLATION MODEL VARIANTS
# ============================================================================

class TemporalGhostGPT_NoTemporalAttention(LightningModule):
    """
    Ablation: Process each frame independently (no temporal attention).
    This is essentially running the original Ghost-GPT on each frame.
    """
    def __init__(self, d_in, d_out, num_blocks, number_of_heads=12,
                 embedding_dim=5, flattened_image_size=106*106,
                 context_size=154, final_image_size=256*256,
                 seq_length=8):
        super().__init__()
        
        self.seq_length = seq_length
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.flattened_image_size = flattened_image_size
        
        # Same as original - NO temporal components
        self.image_embedding_layer = nn.Linear(flattened_image_size, embedding_dim - 1)
        self.pos_embedding_layer = nn.Embedding(context_size, embedding_dim)
        
        # Only spatial blocks (full num_blocks, not split)
        self.main_body = nn.ModuleList([
            TransformerBlock(d_in, d_out, number_of_heads) 
            for _ in range(num_blocks)
        ])
        
        self.call_transformer = TransformerBlock(d_in, d_out, number_of_heads)
        self.final_projection_layer = nn.Linear(d_out, 16)
        self.final_projection_layer2 = nn.Linear(context_size * 16, final_image_size)
        self.final_sigmoid_layer = nn.Sigmoid()
    
    def forward(self, x, bucket_sum):
        """
        Process each frame independently - no temporal attention.
        x: [M, H*W] speckle patterns
        bucket_sum: [B, T, M] bucket measurements
        Returns: [B, T, H, W]
        """
        B, T, M = bucket_sum.shape
        outputs = []
        
        # Process each frame independently
        for t in range(T):
            bucket_t = bucket_sum[:, t, :]  # [B, M]
            
            # Embed patterns
            pattern_embed = self.image_embedding_layer(x)  # [M, embed-1]
            pattern_embed = pattern_embed.unsqueeze(0).expand(B, -1, -1)  # [B, M, embed-1]
            
            # Concatenate with bucket
            bucket_t_exp = bucket_t.unsqueeze(-1)  # [B, M, 1]
            tokens = torch.cat([pattern_embed, bucket_t_exp], dim=-1)  # [B, M, embed]
            
            # Add positional embedding
            pos = self.pos_embedding_layer(torch.arange(M, device=x.device))
            tokens = tokens + pos.unsqueeze(0)
            
            # Spatial attention only
            for module in self.main_body:
                tokens = module(tokens)
            
            # Output
            tokens = self.call_transformer.batch_normalization(tokens)
            tokens = self.final_projection_layer(tokens)
            tokens = tokens.view(B, -1)
            out = self.final_projection_layer2(tokens)
            out = self.final_sigmoid_layer(out)
            outputs.append(out)
        
        output = torch.stack(outputs, dim=1)  # [B, T, H*W]
        H = W = int(np.sqrt(output.shape[-1]))
        return output.view(B, T, H, W)


class TemporalGhostGPT_NoTemporalPosEncoding(LightningModule):
    """
    Ablation: Full temporal attention but NO temporal positional encoding.
    Tests if the model needs to know frame order.
    """
    def __init__(self, d_in, d_out, num_blocks, number_of_heads=12,
                 embedding_dim=5, flattened_image_size=106*106,
                 context_size=154, final_image_size=256*256,
                 seq_length=8):
        super().__init__()
        
        self.seq_length = seq_length
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.flattened_image_size = flattened_image_size
        
        self.image_embedding_layer = nn.Linear(flattened_image_size, embedding_dim - 1)
        self.pos_embedding_layer = nn.Embedding(context_size, embedding_dim)
        # NO temporal_pos_embedding!
        
        self.main_body = nn.ModuleList([
            TransformerBlock(d_in, d_out, number_of_heads) 
            for _ in range(num_blocks // 2)
        ])
        
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(d_in, d_out, number_of_heads) 
            for _ in range(num_blocks // 2)
        ])
        
        self.call_transformer = TransformerBlock(d_in, d_out, number_of_heads)
        self.final_projection_layer = nn.Linear(d_out, 16)
        self.final_projection_layer2 = nn.Linear(context_size * 16, final_image_size)
        self.final_sigmoid_layer = nn.Sigmoid()
    
    def forward(self, x, bucket_sum):
        B, T, M = bucket_sum.shape
        
        # Embed patterns
        pattern_embed = self.image_embedding_layer(x)
        pattern_embed = pattern_embed.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        
        bucket_expanded = bucket_sum.unsqueeze(-1)
        tokens = torch.cat([pattern_embed, bucket_expanded], dim=-1)
        
        # Spatial positional embedding
        spatial_pos = self.pos_embedding_layer(torch.arange(M, device=x.device))
        tokens = tokens + spatial_pos.unsqueeze(0).unsqueeze(0)
        
        # Spatial attention
        tokens_spatial = tokens.view(B * T, M, -1)
        for module in self.main_body:
            tokens_spatial = module(tokens_spatial)
        tokens = tokens_spatial.view(B, T, M, -1)
        
        # Temporal attention - NO positional encoding added!
        tokens_temporal = tokens.permute(0, 2, 1, 3).reshape(B * M, T, -1)
        # Skip: tokens_temporal = tokens_temporal + temporal_pos
        for module in self.temporal_blocks:
            tokens_temporal = module(tokens_temporal)
        tokens = tokens_temporal.view(B, M, T, -1).permute(0, 2, 1, 3)
        
        # Output
        outputs = []
        for t in range(T):
            frame_tokens = tokens[:, t, :, :]
            frame_tokens = self.call_transformer.batch_normalization(frame_tokens)
            frame_tokens = self.final_projection_layer(frame_tokens)
            frame_tokens = frame_tokens.view(B, -1)
            frame_out = self.final_projection_layer2(frame_tokens)
            frame_out = self.final_sigmoid_layer(frame_out)
            outputs.append(frame_out)
        
        output = torch.stack(outputs, dim=1)
        H = W = int(np.sqrt(output.shape[-1]))
        return output.view(B, T, H, W)


class TemporalGhostGPT_FewerTemporalBlocks(LightningModule):
    """
    Ablation: Fewer temporal attention blocks (1 or 2 instead of 4).
    Tests how many temporal blocks are needed.
    """
    def __init__(self, d_in, d_out, num_blocks, number_of_heads=12,
                 embedding_dim=5, flattened_image_size=106*106,
                 context_size=154, final_image_size=256*256,
                 seq_length=8, num_temporal_blocks=1):
        super().__init__()
        
        self.seq_length = seq_length
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.flattened_image_size = flattened_image_size
        
        self.image_embedding_layer = nn.Linear(flattened_image_size, embedding_dim - 1)
        self.pos_embedding_layer = nn.Embedding(context_size, embedding_dim)
        self.temporal_pos_embedding = nn.Embedding(seq_length, embedding_dim)
        
        # More spatial, fewer temporal
        num_spatial = num_blocks - num_temporal_blocks
        self.main_body = nn.ModuleList([
            TransformerBlock(d_in, d_out, number_of_heads) 
            for _ in range(num_spatial)
        ])
        
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(d_in, d_out, number_of_heads) 
            for _ in range(num_temporal_blocks)
        ])
        
        self.call_transformer = TransformerBlock(d_in, d_out, number_of_heads)
        self.final_projection_layer = nn.Linear(d_out, 16)
        self.final_projection_layer2 = nn.Linear(context_size * 16, final_image_size)
        self.final_sigmoid_layer = nn.Sigmoid()
    
    def forward(self, x, bucket_sum):
        B, T, M = bucket_sum.shape
        
        pattern_embed = self.image_embedding_layer(x)
        pattern_embed = pattern_embed.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        
        bucket_expanded = bucket_sum.unsqueeze(-1)
        tokens = torch.cat([pattern_embed, bucket_expanded], dim=-1)
        
        spatial_pos = self.pos_embedding_layer(torch.arange(M, device=x.device))
        tokens = tokens + spatial_pos.unsqueeze(0).unsqueeze(0)
        
        # Spatial attention
        tokens_spatial = tokens.view(B * T, M, -1)
        for module in self.main_body:
            tokens_spatial = module(tokens_spatial)
        tokens = tokens_spatial.view(B, T, M, -1)
        
        # Temporal attention (fewer blocks)
        tokens_temporal = tokens.permute(0, 2, 1, 3).reshape(B * M, T, -1)
        temporal_pos = self.temporal_pos_embedding(torch.arange(T, device=x.device))
        tokens_temporal = tokens_temporal + temporal_pos.unsqueeze(0)
        for module in self.temporal_blocks:
            tokens_temporal = module(tokens_temporal)
        tokens = tokens_temporal.view(B, M, T, -1).permute(0, 2, 1, 3)
        
        # Output
        outputs = []
        for t in range(T):
            frame_tokens = tokens[:, t, :, :]
            frame_tokens = self.call_transformer.batch_normalization(frame_tokens)
            frame_tokens = self.final_projection_layer(frame_tokens)
            frame_tokens = frame_tokens.view(B, -1)
            frame_out = self.final_projection_layer2(frame_tokens)
            frame_out = self.final_sigmoid_layer(frame_out)
            outputs.append(frame_out)
        
        output = torch.stack(outputs, dim=1)
        H = W = int(np.sqrt(output.shape[-1]))
        return output.view(B, T, H, W)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def compute_metrics(pred, target):
    """Compute MSE and SSIM"""
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    mse_val = mse(target_np, pred_np)
    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    
    return mse_val, ssim_val


def compute_temporal_consistency(pred_frames):
    """
    Measure temporal smoothness of predictions.
    Lower = smoother/more consistent
    """
    if isinstance(pred_frames, torch.Tensor):
        pred_frames = pred_frames.cpu().numpy()
    
    frame_diffs = []
    for t in range(1, pred_frames.shape[0]):
        diff = np.abs(pred_frames[t] - pred_frames[t-1]).mean()
        frame_diffs.append(diff)
    
    return np.mean(frame_diffs), np.std(frame_diffs)


def evaluate_model(model, dataset, patterns_flat, device, num_samples=100, model_name="Model"):
    """Evaluate a model and return metrics"""
    model.eval()
    
    all_mse = []
    all_ssim = []
    all_temporal_consistency = []
    times = []
    frame_ssim = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=f"Evaluating {model_name}"):
            sample = dataset[i]
            buckets = sample['buckets'].unsqueeze(0).to(device)
            frames_gt = sample['frames']
            
            start = time.time()
            pred_frames = model(patterns_flat, buckets).squeeze(0).cpu()
            times.append(time.time() - start)
            
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
            tc_mean, tc_std = compute_temporal_consistency(pred_frames)
            all_temporal_consistency.append(tc_mean)
    
    return {
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'temporal_consistency_mean': np.mean(all_temporal_consistency),
        'temporal_consistency_std': np.std(all_temporal_consistency),
        'time_mean_ms': np.mean(times) * 1000,
        'time_std_ms': np.std(times) * 1000,
        'frame_ssim': np.mean(frame_ssim, axis=0),
    }


# ============================================================================
# TRAINING FUNCTIONS FOR ABLATION VARIANTS
# ============================================================================

def train_ablation_model(model, train_dataset, val_dataset, patterns_flat, device,
                         num_epochs=10, batch_size=4, use_ssim_loss=True,
                         use_temporal_loss=True, model_name="model"):
    """Train an ablation model variant"""
    from pytorch_msssim import ssim as ssim_loss_fn
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            buckets = batch['buckets'].to(device)
            frames = batch['frames'].to(device)
            
            optimizer.zero_grad()
            pred_frames = model(patterns_flat, buckets)
            
            # MSE loss
            loss = criterion(pred_frames, frames)
            
            # SSIM loss (optional)
            if use_ssim_loss:
                ssim_val = 1 - ssim_loss_fn(pred_frames, frames, data_range=1.0, size_average=True)
                loss = loss + 0.5 * ssim_val
            
            # Temporal consistency loss (optional)
            if use_temporal_loss and pred_frames.shape[1] > 1:
                pred_diff = pred_frames[:, 1:] - pred_frames[:, :-1]
                true_diff = frames[:, 1:] - frames[:, :-1]
                temporal_loss = criterion(pred_diff, true_diff)
                loss = loss + 0.1 * temporal_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                buckets = batch['buckets'].to(device)
                frames = batch['frames'].to(device)
                pred_frames = model(patterns_flat, buckets)
                val_loss += criterion(pred_frames, frames).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"{model_name} Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'checkpoints/ablation_{model_name}.pt')
    
    return model, train_losses, val_losses


# ============================================================================
# MAIN ABLATION STUDY
# ============================================================================

def load_or_train(model, model_name, train_dataset, val_dataset, patterns_flat, device,
                  num_epochs, use_ssim_loss=True, use_temporal_loss=True, **train_kwargs):
    """Load model from checkpoint if it exists, otherwise train and save."""
    ckpt_path = f'checkpoints/ablation_{model_name}.pt'
    if os.path.exists(ckpt_path):
        print(f"  Found checkpoint {ckpt_path}, loading...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model, None, None
    print(f"  No checkpoint found at {ckpt_path}, training from scratch...")
    return train_ablation_model(
        model, train_dataset, val_dataset, patterns_flat, device,
        num_epochs=num_epochs, use_ssim_loss=use_ssim_loss,
        use_temporal_loss=use_temporal_loss, model_name=model_name, **train_kwargs
    )


def run_ablation_study():
    # Config
    SPECKLE_PATH = 'data/speckle_pattern.pt'
    FULL_MODEL_PATH = 'checkpoints/temporal_ghost_gpt_final.pt'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Hyperparameters (same as training)
    CONFIG = {
        'image_size': 256,
        'seq_length': 8,
        'batch_size': 4,
        'num_epochs': 10,  # Fewer epochs for ablation
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
    
    patterns_flat = torch.tensor(speckle_patterns).float()
    patterns_flat = patterns_flat.view(num_patterns, -1).to(DEVICE)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MovingMNISTGhost(
        speckle_patterns=speckle_patterns,
        seq_length=CONFIG['seq_length'],
        image_size=CONFIG['image_size'],
        dataset_size=2000,  # Smaller for faster ablation
        train=True
    )
    
    val_dataset = MovingMNISTGhost(
        speckle_patterns=speckle_patterns,
        seq_length=CONFIG['seq_length'],
        image_size=CONFIG['image_size'],
        dataset_size=200,
        train=False
    )
    
    test_dataset = MovingMNISTGhost(
        speckle_patterns=speckle_patterns,
        seq_length=CONFIG['seq_length'],
        image_size=CONFIG['image_size'],
        dataset_size=100,
        train=False
    )
    
    # ========================================================================
    # 1. LOAD FULL MODEL (BASELINE)
    # ========================================================================
    print("\n" + "="*60)
    print("1. Loading Full Temporal Ghost-GPT (Baseline)")
    print("="*60)
    
    full_model = TemporalGhostGPT(
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
    
    checkpoint = torch.load(FULL_MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        full_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        full_model.load_state_dict(checkpoint)
    
    results = {}
    results['Full Model'] = evaluate_model(full_model, test_dataset, patterns_flat, DEVICE, 
                                            num_samples=100, model_name="Full Model")
    
    # ========================================================================
    # 2. NO TEMPORAL ATTENTION
    # ========================================================================
    print("\n" + "="*60)
    print("2. Training: No Temporal Attention")
    print("="*60)
    
    no_temporal_model = TemporalGhostGPT_NoTemporalAttention(
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
    
    no_temporal_model, _, _ = load_or_train(
        no_temporal_model, "no_temporal", train_dataset, val_dataset, patterns_flat, DEVICE,
        num_epochs=CONFIG['num_epochs']
    )
    
    results['No Temporal Attention'] = evaluate_model(
        no_temporal_model, test_dataset, patterns_flat, DEVICE,
        num_samples=100, model_name="No Temporal Attention"
    )
    
    # ========================================================================
    # 3. NO TEMPORAL POSITIONAL ENCODING
    # ========================================================================
    print("\n" + "="*60)
    print("3. Training: No Temporal Positional Encoding")
    print("="*60)
    
    no_pos_model = TemporalGhostGPT_NoTemporalPosEncoding(
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
    
    no_pos_model, _, _ = load_or_train(
        no_pos_model, "no_temporal_pos", train_dataset, val_dataset, patterns_flat, DEVICE,
        num_epochs=CONFIG['num_epochs']
    )
    
    results['No Temporal Pos Encoding'] = evaluate_model(
        no_pos_model, test_dataset, patterns_flat, DEVICE,
        num_samples=100, model_name="No Temporal Pos Encoding"
    )
    
    # ========================================================================
    # 4. MSE ONLY (NO SSIM LOSS)
    # ========================================================================
    print("\n" + "="*60)
    print("4. Training: MSE Only (No SSIM Loss)")
    print("="*60)
    
    mse_only_model = TemporalGhostGPT(
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
    
    mse_only_model, _, _ = load_or_train(
        mse_only_model, "mse_only", train_dataset, val_dataset, patterns_flat, DEVICE,
        num_epochs=CONFIG['num_epochs'], use_ssim_loss=False
    )
    
    results['MSE Only (No SSIM)'] = evaluate_model(
        mse_only_model, test_dataset, patterns_flat, DEVICE,
        num_samples=100, model_name="MSE Only"
    )
    
    # ========================================================================
    # 5. NO TEMPORAL CONSISTENCY LOSS
    # ========================================================================
    print("\n" + "="*60)
    print("5. Training: No Temporal Consistency Loss")
    print("="*60)
    
    no_tc_model = TemporalGhostGPT(
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
    
    no_tc_model, _, _ = load_or_train(
        no_tc_model, "no_temporal_loss", train_dataset, val_dataset, patterns_flat, DEVICE,
        num_epochs=CONFIG['num_epochs'], use_temporal_loss=False
    )
    
    results['No Temporal Consistency Loss'] = evaluate_model(
        no_tc_model, test_dataset, patterns_flat, DEVICE,
        num_samples=100, model_name="No Temporal Loss"
    )
    
    # ========================================================================
    # 6. FEWER TEMPORAL BLOCKS (1 instead of 4)
    # ========================================================================
    print("\n" + "="*60)
    print("6. Training: 1 Temporal Block (instead of 4)")
    print("="*60)
    
    fewer_blocks_model = TemporalGhostGPT_FewerTemporalBlocks(
        d_in=CONFIG['embedding_dim'],
        d_out=CONFIG['embedding_dim'],
        num_blocks=CONFIG['num_blocks'],
        number_of_heads=CONFIG['num_heads'],
        embedding_dim=CONFIG['embedding_dim'],
        flattened_image_size=CONFIG['image_size'] * CONFIG['image_size'],
        context_size=num_patterns,
        final_image_size=CONFIG['image_size'] * CONFIG['image_size'],
        seq_length=CONFIG['seq_length'],
        num_temporal_blocks=1  # Only 1 temporal block
    ).to(DEVICE)
    
    fewer_blocks_model, _, _ = load_or_train(
        fewer_blocks_model, "1_temporal_block", train_dataset, val_dataset, patterns_flat, DEVICE,
        num_epochs=CONFIG['num_epochs']
    )
    
    results['1 Temporal Block'] = evaluate_model(
        fewer_blocks_model, test_dataset, patterns_flat, DEVICE,
        num_samples=100, model_name="1 Temporal Block"
    )
    
    # ========================================================================
    # PRINT RESULTS TABLE
    # ========================================================================
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS")
    print("="*100)
    print(f"{'Model Variant':<30} {'MSE':<18} {'SSIM':<18} {'Temp Consist':<18} {'Time (ms)':<15}")
    print("-"*100)
    
    for name, res in results.items():
        mse_str = f"{res['mse_mean']:.4f} (±{res['mse_std']:.4f})"
        ssim_str = f"{res['ssim_mean']:.4f} (±{res['ssim_std']:.4f})"
        tc_str = f"{res['temporal_consistency_mean']:.4f} (±{res['temporal_consistency_std']:.4f})"
        time_str = f"{res['time_mean_ms']:.1f}"
        print(f"{name:<30} {mse_str:<18} {ssim_str:<18} {tc_str:<18} {time_str:<15}")
    
    print("="*100)
    
    # ========================================================================
    # PLOT RESULTS
    # ========================================================================
    plot_ablation_results(results)
    plot_frame_ssim_comparison(results)
    
    # Save results
    import json
    with open('outputs/ablation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON
        results_json = {}
        for k, v in results.items():
            results_json[k] = {key: (val.tolist() if isinstance(val, np.ndarray) else val) 
                              for key, val in v.items()}
        json.dump(results_json, f, indent=2)
    print("\nSaved results to outputs/ablation_results.json")
    
    return results


def plot_ablation_results(results):
    """Create bar plots for ablation study"""
    models = list(results.keys())
    mse_vals = [results[m]['mse_mean'] for m in models]
    ssim_vals = [results[m]['ssim_mean'] for m in models]
    tc_vals = [results[m]['temporal_consistency_mean'] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['green'] + ['coral'] * (len(models) - 1)  # Green for full model
    
    # MSE
    bars1 = axes[0].bar(range(len(models)), mse_vals, color=colors)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Mean Squared Error (↓ lower is better)')
    axes[0].axhline(y=mse_vals[0], color='green', linestyle='--', alpha=0.5)
    
    # SSIM
    bars2 = axes[1].bar(range(len(models)), ssim_vals, color=colors)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('Structural Similarity (↑ higher is better)')
    axes[1].axhline(y=ssim_vals[0], color='green', linestyle='--', alpha=0.5)
    
    # Temporal Consistency
    bars3 = axes[2].bar(range(len(models)), tc_vals, color=colors)
    axes[2].set_xticks(range(len(models)))
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].set_ylabel('Frame-to-Frame Difference')
    axes[2].set_title('Temporal Consistency (↓ lower is smoother)')
    axes[2].axhline(y=tc_vals[0], color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('outputs/ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved ablation comparison to outputs/ablation_comparison.png")


def plot_frame_ssim_comparison(results):
    """Plot SSIM across frames for each model variant"""
    plt.figure(figsize=(12, 6))
    
    for name, res in results.items():
        if 'frame_ssim' in res:
            ssim_per_frame = res['frame_ssim']
            linestyle = '-' if name == 'Full Model' else '--'
            linewidth = 3 if name == 'Full Model' else 1.5
            plt.plot(range(len(ssim_per_frame)), ssim_per_frame, 
                    label=name, linestyle=linestyle, linewidth=linewidth)
    
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title('SSIM Across Frames - Ablation Comparison', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/ablation_frame_ssim.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved frame SSIM comparison to outputs/ablation_frame_ssim.png")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    results = run_ablation_study()