"""
Training Temporal Ghost-GPT with Quantum Detection Data
========================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models.temporal_ghost_gpt import TemporalGhostGPT
from datasets_quantum import QuantumMovingMNISTGhost, PhotonCountNormalizer
from pytorch_msssim import ssim as ssim_loss


def train_quantum_model():
    # Config
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    CONFIG = {
        'image_size': 256,
        'seq_length': 8,
        'batch_size': 4,
        'num_epochs': 30,
        'learning_rate': 3e-4,
        'weight_decay': 0.001,
        'num_blocks': 8,
        'num_heads': 8,
        'embedding_dim': 32,
        # Quantum detector settings
        'detector_type': 'snspd',
        'mean_photons': 100,
        'detection_efficiency': 0.9,
        'dark_count_rate': 100,
    }
    
    # Load speckle patterns
    print("Loading speckle patterns...")
    speckle_patterns = torch.load('data/speckle_pattern.pt')
    if isinstance(speckle_patterns, torch.Tensor):
        speckle_patterns = speckle_patterns.numpy()
    num_patterns = speckle_patterns.shape[0]
    
    patterns_flat = torch.tensor(speckle_patterns).float()
    patterns_flat = patterns_flat.view(num_patterns, -1).to(DEVICE)
    
    # Create quantum datasets
    print(f"Creating quantum datasets with {CONFIG['detector_type']} detector...")
    train_dataset = QuantumMovingMNISTGhost(
        speckle_patterns=speckle_patterns,
        seq_length=CONFIG['seq_length'],
        image_size=CONFIG['image_size'],
        dataset_size=5000,
        train=True,
        detector_type=CONFIG['detector_type'],
        mean_photons_per_pattern=CONFIG['mean_photons'],
        detection_efficiency=CONFIG['detection_efficiency'],
        dark_count_rate=CONFIG['dark_count_rate'],
    )
    
    val_dataset = QuantumMovingMNISTGhost(
        speckle_patterns=speckle_patterns,
        seq_length=CONFIG['seq_length'],
        image_size=CONFIG['image_size'],
        dataset_size=500,
        train=False,
        detector_type=CONFIG['detector_type'],
        mean_photons_per_pattern=CONFIG['mean_photons'],
        detection_efficiency=CONFIG['detection_efficiency'],
        dark_count_rate=CONFIG['dark_count_rate'],
    )
    
    # Initialize normalizer
    print("Fitting photon count normalizer...")
    normalizer = PhotonCountNormalizer(method='anscombe')
    
    # Collect sample counts for fitting
    sample_counts = []
    for i in range(100):
        sample = train_dataset[i]
        sample_counts.append(sample['buckets'].numpy())
    sample_counts = np.concatenate(sample_counts)
    normalizer.fit(sample_counts)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=4)
    
    # Create model
    print("Creating model...")
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
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                            weight_decay=CONFIG['weight_decay'])
    criterion = nn.MSELoss()

    # Resume from checkpoint if available
    start_epoch = 0
    best_val_loss = float('inf')
    ckpt_path = f'checkpoints/temporal_ghost_gpt_quantum_{CONFIG["detector_type"]}.pt'
    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if 'normalizer_stats' in checkpoint:
            normalizer.stats = checkpoint['normalizer_stats']
        print(f"  Resumed from epoch {start_epoch}, best val loss so far: {best_val_loss:.6f}")

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for batch in pbar:
            buckets = batch['buckets'].to(DEVICE)  # [B, T, M] photon counts
            frames = batch['frames'].to(DEVICE)     # [B, T, H, W]
            
            # Normalize photon counts
            buckets_normalized = normalizer.transform(buckets)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_frames = model(patterns_flat, buckets_normalized)
            
            # Losses
            mse_loss = criterion(pred_frames, frames)
            ssim_val = 1 - ssim_loss(pred_frames, frames, data_range=1.0, size_average=True)
            
            # Temporal consistency
            if pred_frames.shape[1] > 1:
                pred_diff = pred_frames[:, 1:] - pred_frames[:, :-1]
                true_diff = frames[:, 1:] - frames[:, :-1]
                temporal_loss = criterion(pred_diff, true_diff)
            else:
                temporal_loss = 0
            
            loss = mse_loss + 0.5 * ssim_val + 0.1 * temporal_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                buckets = batch['buckets'].to(DEVICE)
                frames = batch['frames'].to(DEVICE)
                buckets_normalized = normalizer.transform(buckets)
                pred_frames = model(patterns_flat, buckets_normalized)
                val_loss += criterion(pred_frames, frames).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  New best val loss!")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': CONFIG,
            'normalizer_stats': normalizer.stats,
            'best_val_loss': best_val_loss,
        }, ckpt_path)
        print(f"  Saved checkpoint.")
    
    print("Training complete!")


if __name__ == "__main__":
    train_quantum_model()