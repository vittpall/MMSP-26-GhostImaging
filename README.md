# Temporal Ghost Imaging with Transformer-Based Reconstruction

A deep learning framework for reconstructing dynamic scenes from ghost imaging measurements. The system uses speckle patterns to compute bucket detector readings and recovers temporal image sequences using transformer-based and CNN-based models.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Ablation Study](#ablation-study)
- [Additional Analyses](#additional-analyses)
- [Outputs](#outputs)

---

## Overview

Ghost imaging recovers an image by correlating a spatially resolved reference beam (speckle patterns) with a single-pixel bucket detector signal. This project extends ghost imaging to **dynamic scenes** (video sequences) by modeling temporal correlations with a GPT-style transformer.

**Models supported:**

| Key | Model | Description |
|-----|-------|-------------|
| `dynghost` | Temporal Ghost-GPT | Main model; uses speckle patterns + bucket readings across T frames |
| `ghostgpt` | Ghost-GPT | Single-frame baseline; uses speckle patterns per frame |
| `cnn` | CNN Baseline | Learns from bucket readings only (no explicit patterns at runtime) |
| `unet` | U-Net Baseline | Learns from bucket readings only (no explicit patterns at runtime) |

Classical algorithm baselines (DGI, Pseudo-Inverse, FISTA) are also included for comparison.

---

## Repository Structure

```
temporal_ghost_imaging/
├── training.py               # Main training script
├── evaluate.py               # Evaluation and comparison of all models
├── ablation_study.py         # Ablation experiments for DynGhost components
├── motions_study.py          # Analysis of different motion types
├── noise_robustness.py       # Noise robustness experiments
├── quantum_evaluation.py     # Quantum noise regime evaluation
├── training_quantum.py       # Training under quantum noise conditions
├── datasets.py               # Dataset classes (MNIST, CIFAR-10, Kvasir)
├── datasets_quantum.py       # Quantum-noise dataset variants
├── environment.yml           # Conda environment specification
├── training.pbs              # PBS job script for HPC clusters (ALCF)
│
├── models/
│   ├── temporal_ghost_gpt.py   # DynGhost — Temporal Ghost-GPT
│   ├── Ghost_GPT.py            # GhostGPT — single-frame transformer
│   ├── cnn_ghost.py            # CNN baseline
│   ├── unet_ghost.py           # U-Net baseline
│   └── classical_alg.py        # DGI, Pseudo-Inverse, FISTA algorithms
│
├── data/                       # Dataset and speckle pattern files (not tracked)
│   ├── speckle_pattern.pt      # Required: speckle patterns tensor [M, H, W]
│   ├── MNIST/                  # Auto-downloaded by torchvision
│   ├── cifar-10-batches-py/    # Auto-downloaded by torchvision
│   └── kvasir-v2/              # Manual download required (see below)
│
├── checkpoints/                # Saved model checkpoints (not tracked)
│   └── <model>_<dataset>_epoch<N>.pt
│
├── outputs/                    # Plots and evaluation JSON files
│   ├── evaluation_results.json
│   ├── reconstruction_*.png
│   ├── loss_*.png
│   └── ...
│
└── logs/                       # HPC job logs
```

---

## Environment Setup

### Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate PyTorch
```

### Manual pip install

The key dependencies are:

```bash
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning pytorch-msssim scikit-image matplotlib tqdm numpy scipy pylops
```

> **Python version:** 3.10  
> **CUDA version:** 11.8 (adjust torch/torchvision URLs for your CUDA version)

---

## Data Preparation

### 1. Speckle patterns (required for all models)

Place the speckle pattern file at `data/speckle_pattern.pt`. This should be a PyTorch tensor of shape `[M, H, W]` where `M` is the number of patterns and `H = W = 256`.

```bash
mkdir -p data
# Copy your speckle_pattern.pt into data/
```

### 2. Moving MNIST and Moving CIFAR-10

These datasets are downloaded **automatically** by torchvision the first time you run training or evaluation:

```bash
# MNIST is fetched to ./data/MNIST/
# CIFAR-10 is fetched to ./data/cifar-10-batches-py/
```

No manual steps are needed.

### 3. Kvasir Endoscopy Dataset

Kvasir must be downloaded manually:

1. Go to [https://datasets.simula.no/kvasir/](https://datasets.simula.no/kvasir/)
2. Download **Kvasir v2** (the multi-class version)
3. Unzip into `data/kvasir-v2/`

Expected folder structure after extraction:

```
data/kvasir-v2/
├── dyed-lifted-polyps/
├── dyed-resection-margins/
├── esophagitis/
├── normal-cecum/
├── normal-pylorus/
├── normal-z-line/
├── polyps/
└── ulcerative-colitis/
```

Each subfolder contains JPEG images. The dataset loader performs an automatic 80/20 train/val split.

---

## Training

Training is configured via the `CONFIG` dictionary at the top of [training.py](training.py).

### Configuration options

```python
CONFIG = {
    # Model: 'dynghost' | 'ghostgpt' | 'cnn' | 'unet'
    'model': 'dynghost',

    # Dataset: 'mnist' | 'cifar10' | 'kvasir'
    'dataset': 'mnist',

    # Shared hyperparameters
    'image_size':    256,
    'seq_length':    8,
    'batch_size':    4,
    'num_epochs':    30,
    'learning_rate': 3e-4,
    'weight_decay':  1e-3,

    # Transformer hyperparameters (dynghost / ghostgpt)
    'num_blocks':    8,
    'num_heads':     8,
    'embedding_dim': 32,

    # CNN-only
    'cnn_hidden_dim': 512,
}
```

### Run training

```bash
cd temporal_ghost_imaging
python training.py
```

Checkpoints are saved every 5 epochs and at the end of training:

```
checkpoints/<model>_<dataset>_epoch5.pt
checkpoints/<model>_<dataset>_epoch10.pt
...
checkpoints/<model>_<dataset>_final.pt
```

Training automatically resumes from the latest checkpoint if one exists for the selected model/dataset combination.

### Training on an HPC cluster (PBS)

A PBS job script is provided for ALCF systems (Sophia/Polaris):

```bash
qsub training.pbs
```

Edit `training.pbs` to adjust queue, walltime, and project allocation as needed.

---

## Evaluation

The evaluation script compares all available models and classical baselines on the test split. Checkpoint paths are configured at the top of [evaluate.py](evaluate.py).

### Configure checkpoints

Edit the checkpoint dictionaries in `evaluate.py` to point to your trained models:

```python
CHECKPOINTS = {
    'mnist': 'checkpoints/dynghost_mnist_final.pt',
    # 'cifar10': 'checkpoints/dynghost_cifar10_final.pt',
    # 'kvasir':  'checkpoints/dynghost_kvasir_final.pt',
}

CNN_CHECKPOINTS = {
    # 'mnist': 'checkpoints/cnn_mnist_final.pt',
}

UNET_CHECKPOINTS = {
    # 'mnist': 'checkpoints/unet_mnist_final.pt',
}

GHOSTGPT_CHECKPOINTS = {
    # 'mnist': 'checkpoints/ghostgpt_mnist_final.pt',
}
```

Any entry set to `None` or pointing to a missing file is silently skipped.

### Run evaluation

```bash
python evaluate.py
```

This produces:
- A printed table of MSE, SSIM, and inference time per method
- Reconstruction visualizations in `outputs/`
- Temporal consistency plots in `outputs/`
- `outputs/evaluation_results.json` with all numeric results

**Metrics reported:** MSE (lower is better), SSIM (higher is better), inference time in ms.

---

## Ablation Study

Tests the contribution of individual components of the DynGhost model (temporal attention, positional encoding, SSIM loss, temporal consistency loss, number of transformer blocks):

```bash
python ablation_study.py
```

Results are saved to `outputs/ablation_results.json` and `outputs/ablation_comparison.png`.

---

## Additional Analyses

### Motion type robustness

Evaluates performance across linear, oscillatory, and random-walk motion patterns:

```bash
python motions_study.py
```

### Noise robustness

Tests model performance under varying levels of measurement noise:

```bash
python noise_robustness.py
```

### Quantum noise regime

Evaluates reconstruction quality under photon-counting / quantum noise conditions:

```bash
python quantum_evaluation.py
```

To train under quantum noise:

```bash
python training_quantum.py
```

---

## Outputs

After running training and evaluation, the `outputs/` directory contains:

| File | Description |
|------|-------------|
| `reconstruction_<model>_<dataset>.png` | Side-by-side GT vs predicted frames |
| `loss_<model>_<dataset>.png` | Training and validation loss curves |
| `evaluation_results.json` | Full numeric results (MSE, SSIM, time) |
| `temporal_consistency_<dataset>.png` | SSIM per frame index over time |
| `reconstructions_<dataset>.png` | All-model comparison grid |
| `ablation_results.json` | Ablation study metrics |
| `noise_robustness_results.json` | Noise sweep results |
| `quantum_evaluation_results.json` | Quantum noise evaluation results |
