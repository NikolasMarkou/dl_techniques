# PRISM: Partitioned Representations for Iterative Sequence Modeling

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready implementation of **PRISM (Partitioned Representations for Iterative Sequence Modeling)** in **Keras 3**, based on the paper *"PRISM: A Hierarchical Multiscale Approach for Time Series Forecasting"* by Chen et al. (2025).

PRISM addresses the challenge of capturing both global trends and local fine-grained structures in time series data. It replaces standard attention mechanisms with a learnable **binary tree decomposition** combined with **Haar Wavelet** frequency analysis. This "Split-Transform-Weight-Merge" philosophy achieves State-of-the-Art performance on benchmarks while remaining computationally efficient and highly interpretable.

![image](https://via.placeholder.com/800x300.png?text=PRISM+Architecture+Diagram)
*(Placeholder: Conceptual visualization of the Time-Frequency Tree)*

---

## Table of Contents

1. [Overview: What is PRISM?](#1-overview-what-is-prism)
2. [The Problem PRISM Solves](#2-the-problem-prism-solves)
3. [How PRISM Works: Core Concepts](#3-how-prism-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Interpretability](#9-interpretability)
10. [Training and Best Practices](#10-training-and-best-practices)
11. [Serialization & Deployment](#11-serialization--deployment)
12. [Troubleshooting & FAQs](#12-troubleshooting--faqs)
13. [Citation](#13-citation)

---

## 1. Overview: What is PRISM?

### What is PRISM?

**PRISM** is a hierarchical forecasting model that organizes time series data through a binary temporal hierarchy combined with multiresolution frequency encoding. Unlike models that treat time and frequency separately, PRISM constructs a unified representation.

### Key Innovations

1.  **Hierarchical Time Decomposition**: It recursively splits the input signal into overlapping segments (binary tree), allowing the model to process long-term trends at the root and short-term fluctuations at the leaves.
2.  **Learnable Frequency Routing**: At every node of the tree, an "Importance Router" (a lightweight MLP) analyzes statistics of Haar Wavelet bands and dynamically assigns weights. This allows the model to "choose" which frequencies matter at which point in time.
3.  **Split-Transform-Weight-Merge**: A unique processing philosophy that decomposes signals, filters them based on relevance, and reconstructs a refined representation for forecasting.

---

## 2. The Problem PRISM Solves

### The Limitations of Existing Forecasters

Modern forecasting often falls into two traps:
1.  **Transformers (e.g., PatchTST)**: Excellent at global dependencies but computationally expensive ($O(L^2)$) and often struggle to separate signal from noise in high-frequency data.
2.  **Linear/Decomposition Models (e.g., DLinear)**: Very efficient but often lack the capacity to model complex, non-linear interactions between diverse time scales.

### How PRISM Changes the Game

PRISM bridges the gap by being both **hierarchical** and **adaptive**.

```
┌─────────────────────────────────────────────────────────────┐
│  The PRISM Solution                                         │
│                                                             │
│  1. Unified Hierarchy: Instead of flattening time, PRISM    │
│     builds a tree. This naturally handles the "Context"     │
│     problem: coarse scales set the context for fine scales. │
│                                                             │
│  2. Adaptive Filtering: Not all frequencies matter all the  │
│     time. PRISM's Router learns to suppress noise (e.g.,    │
│     high-freq jitter in a trend period) and highlight       │
│     signal (e.g., specific seasonality).                    │
│                                                             │
│  3. Efficiency: Uses Haar Wavelets (O(N)) and MLPs,         │
│     avoiding the quadratic cost of Attention.               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How PRISM Works: Core Concepts

### The Time-Frequency Tree

PRISM builds a binary tree over the time dimension.

1.  **Level 0 (Root)**: Sees the entire history. Captures global trends.
2.  **Level 1**: Splits history into 2 overlapping segments. Captures medium-term dynamics.
3.  **Level 2**: Splits into 4 segments. Captures local details.

### The Node Mechanism

Inside every node of this tree, the following process occurs:

```
┌──────────────────────────────────────────────────────────────────┐
│                        PRISM Node Logic                          │
│                                                                  │
│  Input Segment [Batch, T_seg, Channels]                          │
│           │                                                      │
│           ├─► Haar Wavelet Transform (DWT)                       │
│           │      (Decompose into K Frequency Bands)              │
│           │                                                      │
│           ├─► Statistics Extraction                              │
│           │      (Mean, Std, Derivatives of each band)           │
│           │                                                      │
│           ├─► Router MLP                                         │
│           │      (Input: Stats -> Output: Importance Weights)    │
│           │                                                      │
│           └─► Weighted Reconstruction                            │
│                  (Sum bands * weights)                           │
│                                                                  │
│  Output Segment [Batch, T_seg, Channels]                         │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

1.  **Input Projection**: Raw features are projected to a hidden dimension.
2.  **Time Tree**: The signal is passed through the hierarchical tree.
    *   Signal splits -> Processed by Nodes -> Stitched back together (Cross-fade).
3.  **Stacking**: Multiple PRISM Layers can be stacked (with residual connections).
4.  **Forecasting Head**: The flattened representation is mapped to the forecast horizon.

---

## 4. Architecture Deep Dive

### 4.1 `PRISMNode`

The atomic unit of the architecture. It contains:
*   **`HaarWaveletDecomposition`**: Decomposes input into Low-Pass (Approximation) and High-Pass (Detail) coefficients.
*   **`FrequencyBandRouter`**: A smart weighting mechanism. It calculates statistics (mean, std, min, max, derivatives) for each band and uses a softmax-normalized MLP to determine how much each frequency band contributes to the output.

### 4.2 `PRISMTimeTree`

This layer manages the recursion. It:
1.  Splits input tensors into $2^L$ overlapping segments.
2.  Routes each segment to a specific `PRISMNode`.
3.  Stitches the outputs back together using **linear cross-fading** to ensure smooth boundaries between time segments.

### 4.3 `PRISMLayer`

The user-facing layer that wraps the Time Tree with:
*   **Residual Connections**: `Output = Input + Tree(Input)`
*   **Layer Normalization**: Stabilizes training.
*   **Dropout**: Prevents overfitting.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure Keras 3 and backend (TensorFlow/Torch/JAX) are installed
pip install keras>=3.0 tensorflow  # or torch, jax
```

### Your First Forecast (30 seconds)

Let's forecast a synthetic sine wave.

```python
import keras
import numpy as np
from prism import PRISMModel

# 1. Generate dummy time-series data
# Shape: (Samples, Time, Features)
def generate_data(n=1000, seq_len=96, pred_len=24, channels=1):
    x = np.linspace(0, 100, n + seq_len + pred_len)
    data = np.sin(x)[:, None] # Add channel dim
    
    X, Y = [], []
    for i in range(n):
        X.append(data[i : i+seq_len])
        Y.append(data[i+seq_len : i+seq_len+pred_len])
    return np.array(X), np.array(Y)

X_train, y_train = generate_data()

# 2. Initialize the PRISM Model
model = PRISMModel.from_preset(
    "small",
    context_len=96,
    forecast_len=24,
    num_features=1
)

# 3. Compile
model.compile(optimizer="adam", loss="mse")

# 4. Train
model.fit(X_train, y_train, batch_size=32, epochs=5)

# 5. Forecast
forecast = model.predict(X_train[:1])
print(f"Input shape: {X_train[:1].shape}")       # (1, 96, 1)
print(f"Forecast shape: {forecast.shape}")       # (1, 24, 1)
```

---

## 6. Component Reference

### 6.1 `PRISMModel`

The high-level wrapper for end-to-end forecasting.

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `context_len` | `int` | Length of input history window. | Required |
| `forecast_len` | `int` | Length of prediction horizon. | Required |
| `num_features` | `int` | Number of input channels (multivariate). | Required |
| `num_layers` | `int` | Number of PRISM layers to stack. | `2` |
| `tree_depth` | `int` | Depth of the time binary tree (0 = no split). | `2` |
| `num_wavelet_levels` | `int` | Depth of Haar Wavelet decomposition. | `3` |
| `router_hidden_dim` | `int` | Size of the MLP in the importance router. | `64` |

### 6.2 `PRISMLayer`

The modular layer that can be inserted into custom Keras models.

| Parameter | Description |
| :--- | :--- |
| `overlap_ratio` | Controls how much adjacent time segments overlap (0.0 to 0.5). Helps smooth boundaries. Default `0.25`. |
| `router_temperature` | Scaling factor for Softmax. Lower (<1.0) makes selection sharper/sparse. Default `1.0`. |

---

## 7. Configuration & Model Variants

The implementation includes presets for common scales, derived from the paper's experiments. You can load these via `PRISMModel.from_preset("name", ...)`.

*   **`tiny`**: 1 Layer, Depth 1. Good for very short sequences or simple debugging.
*   **`small`**: 2 Layers, Depth 2. The standard baseline.
*   **`base`**: 3 Layers, Depth 2, Wider MLPs. For complex multivariate datasets (e.g., ETT, Weather).
*   **`large`**: 4 Layers, Depth 2, Hidden Dim 256. For large-scale pre-training or massive datasets.

---

## 8. Comprehensive Usage Examples

### Multivariate Forecasting with Custom Config

```python
from prism import PRISMModel

model = PRISMModel(
    context_len=336,
    forecast_len=96,
    num_features=7,       # e.g., ETTh1 dataset has 7 channels
    hidden_dim=128,       # Project inputs to 128-dim space
    num_layers=3,         # Stack 3 PRISM layers
    tree_depth=2,         # Split time into 4 segments (2^2)
    num_wavelet_levels=4, # Decompose into 5 frequency bands
    dropout_rate=0.2
)

model.compile(optimizer="adamw", loss="mae")
```

---

## 9. Interpretability

One of PRISM's strongest features is **interpretability**. Unlike black-box Transformers, PRISM explicitly calculates "Importance Scores" for frequency bands.

Although the high-level `PRISMModel` wraps these details, you can inspect the router weights in a custom training loop or by accessing layers directly:

```python
# Accessing the first PRISM layer's time tree
layer = model.prism_layers[0]
# Accessing the root node (Level 0)
root_node = layer.time_tree.prism_nodes[0][0]
# The router layer
router = root_node.router

# In a research setting, you can extract the output of 'router' 
# to see which frequency bands (Trends vs Details) the model 
# prioritized for a specific input sample.
```
*High weights on Low-Pass bands indicate the model is focusing on Trend. High weights on Detail bands indicate focus on rapid fluctuations.*

---

## 10. Training and Best Practices

1.  **Normalization**: Time-series data **must** be normalized (e.g., Z-Score or MinMax) before feeding into PRISM. The model does not include internal instance normalization (ReVIN) by default, though `PRISMLayer` includes `LayerNormalization`.
2.  **Overlap Ratio**: The default `0.25` is robust. If you see "jumpy" predictions at segment boundaries, try increasing this to `0.3` or `0.4`.
3.  **Context Length**: PRISM benefits from longer context windows due to its hierarchical nature. `336` or `512` are good starting points.
4.  **Tree Depth**: Deeper trees ($>3$) are rarely needed and increase computational cost. Depth `2` (4 segments) is usually the sweet spot.

---

## 11. Serialization & Deployment

PRISM is built with Keras 3 `register_keras_serializable`, ensuring full compatibility with the `.keras` format.

```python
# Save
model.save("weather_forecaster.keras")

# Load (Custom layers are handled automatically)
restored_model = keras.models.load_model("weather_forecaster.keras")

# Verify
preds = restored_model.predict(X_test)
```

---

## 12. Troubleshooting & FAQs

**Q: Why Haar Wavelets? Why not Fourier (FFT)?**
A: FFT assumes periodicity and global stationarity. Wavelets are localized in both time and frequency. Since PRISM splits time into segments, Haar Wavelets naturally align with this block-based processing and are computationally cheaper ($O(N)$).

**Q: Is this faster than Transformers?**
A: Yes, generally. The complexity is roughly linear with sequence length $O(L)$, whereas standard Attention is $O(L^2)$.

**Q: Can I use this for Classification?**
A: This implementation is specialized for Forecasting (`PRISMModel`). However, you can extract the `PRISMLayer` and attach a classification head (GlobalAveragePooling + Dense) instead of the forecasting head.

---

## 13. Citation

If you use PRISM in your research, please cite the original paper:

```bibtex
@article{chen2025prism,
  title={PRISM: A Hierarchical Multiscale Approach for Time Series Forecasting},
  author={Chen, Zihao and Andre, Alexandre and Ma, Wenrui and Knight, Ian and Shuvaev, Sergey and Dyer, Eva},
  journal={arXiv preprint arXiv:2512.24898},
  year={2025}
}
```