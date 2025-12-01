# LatentGMMRegistration: Robust Point Cloud Alignment

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready Keras 3 implementation of **LatentGMMRegistration**, a robust semi-supervised framework for rigid point cloud registration. This model learns to align two point clouds by mapping them into a shared latent Gaussian Mixture Model (GMM) space, handling noise, outliers, and large transformations effectively without expensive iterative optimization during inference.

---

## Table of Contents

1. [Overview: The Registration Challenge](#1-overview-the-registration-challenge)
2. [Core Innovation: Deep Latent GMMs](#2-core-innovation-deep-latent-gmms)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [Component Reference](#4-component-reference)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Training Strategy](#6-training-strategy)
7. [Comprehensive Usage Examples](#7-comprehensive-usage-examples)
8. [Technical Details](#8-technical-details)
9. [Citation](#9-citation)

---

## 1. Overview: The Registration Challenge

Point cloud registration is the process of finding a spatial transformation (rotation and translation) that aligns a source point cloud to a target point cloud.

**Common Problems**:
*   **Iterative Closest Point (ICP)**: Requires good initialization, sensitive to noise/outliers, and slow due to iterative matching.
*   **Feature-based Methods**: Hand-crafted features often fail on smooth surfaces or with uniform density.

**LatentGMM Solution**:
Instead of matching points directly (which is N² complexity or requires k-d trees), LatentGMMRegistration learns to map both point clouds into a shared probabilistic space defined by a Gaussian Mixture Model. The alignment is then solved analytically between these GMM distributions, which is computationally efficient and naturally robust to noise.

---

## 2. Core Innovation: Deep Latent GMMs

Traditional GMM-based registration (like JRMPC) uses Expectation-Maximization (EM) to fit GMMs. This is slow.

**LatentGMMRegistration** replaces the slow E-step with a learned neural network:
1.  **Learned Features**: A DGCNN-style autoencoder extracts rich local and global geometric features.
2.  **Learned Correspondences**: A deep network predicts the probability ($\gamma$) that any point belongs to a specific Gaussian component.
3.  **Differentiable M-Step**: The GMM parameters (means and weights) are computed analytically from these probabilities.
4.  **Differentiable Solver**: The optimal rotation and translation are computed using a weighted Procrustes solution on the GMM components.

This entire pipeline is end-to-end differentiable, allowing the model to learn features specifically optimized for alignment.

---

## 3. Architecture Deep Dive

The model consists of a Siamese-style architecture processing source and target point clouds in parallel.

```
┌─────────────────────────────────────────────────────────────┐
│               LatentGMMRegistration Pipeline                │
│                                                             │
│  Input Source (N points)       Input Target (N points)      │
│         │                             │                     │
│         ▼                             ▼                     │
│  ┌─────────────┐               ┌─────────────┐              │
│  │ Autoencoder │               │ Autoencoder │ (Shared)     │
│  └──────┬──────┘               └──────┬──────┘              │
│         │ Features                    │ Features            │
│         ▼                             ▼                     │
│  ┌─────────────┐               ┌─────────────┐              │
│  │ Corresp Net │               │ Corresp Net │ (Shared)     │
│  └──────┬──────┘               └──────┬──────┘              │
│         │ Soft Assignments (γ)        │ Soft Assignments (γ)│
│         ▼                             ▼                     │
│  ┌─────────────┐               ┌─────────────┐              │
│  │ Compute GMM │               │ Compute GMM │ (Analytical) │
│  └──────┬──────┘               └──────┬──────┘              │
│         │ Source GMM (μ_x, π_x)       │ Target GMM (μ_y, π_y)
│         └──────────────┐      ┌───────┘                     │
│                        ▼      ▼                             │
│                 ┌────────────────────┐                      │
│                 │  Solver (SVD)      │ (Differentiable)     │
│                 └─────────┬──────────┘                      │
│                           │                                 │
│                           ▼                                 │
│                 Optimal Transform (R, t)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Component Reference

### 4.1 `LatentGMMRegistration` (Main Model)
The high-level orchestrator.
-   **Inputs**: Tuple `(source_pc, target_pc)`.
-   **Outputs**: Dictionary with `estimated_r`, `estimated_t`, and reconstructions.
-   **Losses**: Combines supervised transformation loss with unsupervised reconstruction loss.

### 4.2 `PointCloudAutoencoder`
Extracts geometric features using **EdgeConv** (Dynamic Graph CNN) layers.
-   **Encoder**: Stacks `EdgeConv` layers to learn local geometry, aggregating them into a global descriptor.
-   **Decoder**: Reconstructs the point cloud from the global descriptor, enforcing that the learned features preserve geometric information.

### 4.3 `CorrespondenceNetwork`
The "Brain" of the GMM assignment.
-   Takes local features (point-wise) and global features (context).
-   Outputs a $N \times K$ matrix of probabilities, assigning each of the $N$ points to $K$ latent Gaussian components.

---

## 5. Quick Start Guide

### Installation
```bash
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Basic Usage

```python
import keras
import numpy as np
from dl_techniques.models.latent_gmm import LatentGMMRegistration

# 1. Initialize the model
# num_gaussians=32 is a good starting point for moderate complexity
model = LatentGMMRegistration(
    num_gaussians=32,
    k_neighbors=20
)

# 2. Compile with an optimizer
# Loss logic is internal to the model (custom train_step), so loss=None
model.compile(optimizer="adam")

# 3. Create dummy data
# Batch of 8, 1024 points each, 3 dimensions (x, y, z)
batch_size = 8
num_points = 1024
source = np.random.rand(batch_size, num_points, 3).astype("float32")

# Create targets by rotating and translating sources
# (In real usage, you would load your dataset here)
angle = np.pi / 4
R = np.array([[np.cos(angle), -np.sin(angle), 0],
              [np.sin(angle),  np.cos(angle), 0],
              [0, 0, 1]], dtype="float32")
target = np.dot(source, R.T) + np.array([0.1, 0.2, 0.3], dtype="float32")
t_gt = np.tile([[0.1, 0.2, 0.3]], (batch_size, 1)).astype("float32")
R_gt = np.tile(R, (batch_size, 1, 1)).astype("float32")

# 4. Train
# Inputs: ((source, target), (R_gt, t_gt))
model.fit(
    x=(source, target),
    y=(R_gt, t_gt),
    batch_size=batch_size,
    epochs=5
)

# 5. Inference
results = model.predict((source, target))
print("Estimated Rotation:\n", results["estimated_r"][0])
```

---

## 6. Training Strategy

The model uses a **Semi-Supervised** loss function:

$$ \mathcal{L} = \lambda_{\text{chamfer}} \mathcal{L}_{\text{rec}} + \lambda_{\text{trans}} \mathcal{L}_{\text{pose}} $$

1.  **Reconstruction Loss ($\mathcal{L}_{\text{rec}}$)**: Chamfer Distance between input point clouds and the autoencoder's reconstruction. This ensures the learned features actually represent the shape geometry. This part is unsupervised.
2.  **Pose Loss ($\mathcal{L}_{\text{pose}}$)**: MSE between estimated $(R, t)$ and ground truth $(R_{gt}, t_{gt})$. This drives the GMM components to align in a way that solves the registration problem. This part is supervised.

---

## 7. Comprehensive Usage Examples

### Example 1: Inference Only (Unsupervised Alignment)
Once trained, the model can align point clouds without ground truth.

```python
# Assume model is trained
source_pc = load_point_cloud("chair_scan_1.ply") # (1, 2048, 3)
target_pc = load_point_cloud("chair_scan_2.ply") # (1, 2048, 3)

outputs = model.predict((source_pc, target_pc))

R_est = outputs["estimated_r"] # (1, 3, 3)
t_est = outputs["estimated_t"] # (1, 3)

# Apply transformation to align
aligned_source = np.dot(source_pc, R_est[0].T) + t_est[0]
```

### Example 2: Accessing Latent GMM Features
You might want to inspect the learned Gaussian components to see "where" the model thinks the key parts of the object are.

```python
# Forward pass with training=False
inputs = (source_pc, target_pc)
(x_rec, y_rec), (local_x, local_y), (global_x, global_y) = model.autoencoder(inputs)

# Get probabilities
gamma_x = model.correspondence_net((local_x, global_x))

# Compute the centers of the Gaussian components
from dl_techniques.models.latent_gmm import compute_gmm_params
weights, means = compute_gmm_params(source_pc, gamma_x)

# 'means' shape is (B, num_gaussians, 3)
# These act like keypoints describing the object
```

---

## 8. Technical Details

### Weighted Procrustes Solver
The core of the registration is the differentiable SVD solver. Given two sets of corresponding points (the GMM means $\mu_x$ and $\mu_y$) and their weights, it analytically finds $R, t$ to minimize:
$$ \min_{R, t} \sum_{k=1}^K w_k \| (R \mu_{x,k} + t) - \mu_{y,k} \|^2 $$
This closed-form solution is stable and provides exact gradients for backpropagation.

### EdgeConv Layers
The encoder uses EdgeConv, which dynamically computes graphs in feature space.
-   **Layer 1**: Neighbors based on spatial XYZ distance.
-   **Layer 2+**: Neighbors based on feature distance from previous layer.
This allows the network to group points that are semantically similar, not just spatially close.

---

## 9. Citation

If you use this implementation, please cite the concepts behind DeepGMR and DGCNN:

**DeepGMR (Registration Framework):**
```bibtex
@inproceedings{yuan2020deepgmr,
  title={DeepGMR: Learning Latent Gaussian Mixture Models for Registration},
  author={Yuan, Wentao and Eckart, Benjamin and Kim, Kihwan and Kautz, Jan and others},
  booktitle={ECCV},
  year={2020}
}
```

**DGCNN (Encoder Backbone):**
```bibtex
@article{wang2019dynamic,
  title={Dynamic Graph CNN for Learning on Point Clouds},
  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
  journal={ACM TOG},
  year={2019}
}
```