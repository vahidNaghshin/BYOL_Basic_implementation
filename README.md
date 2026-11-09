# Bootstrap Your Own Latent (BYOL) in PyTorch

A clean, step-by-step PyTorch implementation of **"Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"** (Grill et al., 2020) tailored for self-supervised representation learning on CIFAR-10.

## 🚀 Features

- **Dual-View Augmentation Pipeline**: Generates distinct augmented views ($v$ and $v'$) using Random Resized Crops, Horizontal Flips, Color Jitter, and Grayscale transformations adjusted for small $32\times32$ px resolutions.
- **Modified ResNet-18 Backbone**: Optimized for small-scale images by removing the initial max-pooling layer and using a smaller $3x3$ conv1 kernel to preserve spatial resolution.
- **Online & Target Networks**: Full implementation of projection MLPs, online predictor, and decoupled target-parameter updating.
- **Exponential Moving Average (EMA)**: Custom Parameter-EMA module to update the target network step-by-step without gradient calculations.
- **Symmetric BYOL Loss**: Fully vectorized implementation of the normalized Mean Squared Error (MSE) loss applied symmetrically to both view variations to stabilize representation learning.


## 📦 Requirements

- Python 3.x
- PyTorch
- torchvision
