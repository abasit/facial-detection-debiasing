# Facial Detection Debiasing
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

PyTorch implementation of Debiasing Variational Autoencoders (DB-VAE) for mitigating algorithmic bias in facial detection systems.

## Overview

This repository implements and compares two approaches to facial detection:
- **Standard CNN**: Baseline convolutional neural network trained on CelebA + ImageNet
- **DB-VAE** ([Amini et al., 2019](http://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf)): Debiasing Variational Autoencoder that learns latent representations and adaptively resamples underrepresented features during training

The key innovation is using unsupervised learning to identify and correct demographic biases without manual annotation, addressing performance disparities across different demographic groups.

## Results

Performance across demographic groups (face detection accuracy):

| Model        | Light Female | Light Male | Dark Female | Dark Male | Bias Range |
|--------------|--------------|------------|-------------|-----------|------------|
| Standard CNN | 59.7%        | 55.7%      | 16.8%       | 5.4%      | 54.3%      |
| DB-VAE       | 60.7%        | 60.1%      | 15.5%       | 20.0%     | 45.2%      |

*DB-VAE achieves 270% improvement for the most underrepresented group (Dark Male: 5.4% → 20.0%) and reduces overall bias range by 9.1 percentage points*


## Installation

```bash
# Clone the repository
git clone https://github.com/abasit/facial-detection-debiasing.git
cd facial-detection-debiasing

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Train Standard CNN (Baseline)

```bash
python train_cnn.py --config config/config.yaml
```

### 2. Train DB-VAE (Debiased)

```bash
python train_dbvae.py --config config/config.yaml
```

### Configuration

Modify `config/config.yaml` to adjust:
- Training parameters (epochs, batch size, learning rate)
- Model architecture (filters, dimensions)
- VAE parameters (latent dimension, KL weight)

## Technical Approach

1. **Learn latent representations** of facial features using VAE encoder-decoder
2. **Compute feature distributions** across training data  
3. **Adaptively resample** rare features more frequently during training
4. **Train debiased classifier** with rebalanced data

## Acknowledgments

This implementation is based on research from MIT 6.S191 Introduction to Deep Learning:

> Amini, Alexander, et al. "Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure." *Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society*. 2019.

> Amini, Alexander, and Ava Soleimany. *MIT 6.S191: Introduction to Deep Learning*. Massachusetts Institute of Technology. http://introtodeeplearning.com

---