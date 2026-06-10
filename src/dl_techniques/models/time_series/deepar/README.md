# DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of **DeepAR** in **Keras 3**. DeepAR is a methodology for producing accurate probabilistic forecasts based on training an autoregressive recurrent network on multiple related time series.

This implementation provides a unified framework for handling real-valued data (Gaussian likelihood) and count data (Negative Binomial likelihood), solving the "cold start" and scaling problems often found in traditional forecasting.

---

## Table of Contents

1. [Overview: What is DeepAR and Why It Matters](#1-overview-what-is-deepar-and-why-it-matters)
2. [The Problem DeepAR Solves](#2-the-problem-deepar-solves)
3. [How DeepAR Works: Core Concepts](#3-how-deepar-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Likelihoods](#7-configuration--likelihoods)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Testing & Validation](#13-testing--validation)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Technical Details](#15-technical-details)
16. [Citation](#16-citation)

---

## 1. Overview: What is DeepAR and Why It Matters

### What is DeepAR?

**DeepAR** is a supervised learning algorithm for forecasting time series that learns a global model from historical data of all time series in the dataset. Unlike traditional methods (like ARIMA or ETS) that fit a separate model for each time series, DeepAR learns a single model across thousands or millions of related series.

It generates **probabilistic forecasts** by using Monte Carlo sampling to simulate multiple future paths, allowing you to compute any quantile (e.g., median, 90th percentile) to assess uncertainty and risk.

### Key Innovations

1.  **Global Learning**: Learns complex patterns (seasonality, trends) shared across items, allowing it to forecast new items with little history ("cold start").
2.  **Scale Handling**: Uses a specialized `ScaleLayer` to normalize inputs and denormalize outputs, enabling the model to learn from data spanning multiple orders of magnitude (e.g., a product selling 5 units vs 5,000 units).
3.  **Ancestral Sampling**: Produces valid probabilistic paths, not just marginal quantiles, preserving the correlation between time steps in the future.
4.  **Flexible Likelihoods**: Natively supports Gaussian for continuous data and Negative Binomial for count data (integers).

### Why DeepAR Matters

**Traditional Forecasting Models**:
```
Model: ARIMA (Statistical)
  1. Fits one model per time series.
  2. Does not share information between series.
  3. Hard to scale to millions of items.

Model: Standard LSTM (Point Forecast)
  1. Often predicts a single value (mean), ignoring uncertainty.
  2. Struggles with input scaling issues (convergence problems).
```

**DeepAR's Solution**:
```
DeepAR Approach:
  1. Combines RNNs with probabilistic likelihood heads.
  2. Scales inputs by their average historical value.
  3. Outputs distribution parameters (μ, σ) instead of values.
  4. Benefit: Scalable, calibrated uncertainty, and handles diverse magnitudes.
```

---

## 2. The Problem DeepAR Solves

### The Challenge of Diverse Scales

In real-world datasets (e.g., retail sales, server load), time series often differ drastically in magnitude.

```
┌─────────────────────────────────────────────────────────────┐
│  The Scale Problem                                          │
│                                                             │
│  Item A: Sales ~ 100,000 units/day                          │
│  Item B: Sales ~ 5 units/day                                │
│                                                             │
│  Neural networks struggle to learn weights that work for    │
│  both A and B simultaneously without normalization.         │
│  Standard normalization (z-score) is tricky when predicting │
│  future values in the original domain.                      │
└─────────────────────────────────────────────────────────────┘
```

### The Need for Uncertainty

Knowing that sales will be "50" is less useful than knowing "there is a 90% chance sales will be between 40 and 60". DeepAR provides this distribution, which is critical for:
-   **Inventory Optimization**: Balancing stock-out risk vs. overstock costs.
-   **Capacity Planning**: Preparing for peak loads (99th percentile) rather than average loads.

---

## 3. How DeepAR Works: Core Concepts

### The High-Level Architecture

DeepAR operates in two modes: **Training** (using Teacher Forcing) and **Prediction** (using Autoregressive Sampling).

```
┌──────────────────────────────────────────────────────────────────┐
│                       DeepAR Architecture                        │
│                                                                  │
│  Inputs: [Target, Covariates]                                    │
│             │                                                    │
│    ┌────────▼────────┐     ┌──────────────────────────────────┐  │
│    │ Scale Layer (ν) │◄────┤ Compute Scale (Mean of History)  │  │
│    └────────┬────────┘     └──────────────────────────────────┘  │
│             │                                                    │
│    ┌────────▼─────────┐                                          │
│    │ Stacked LSTMs    │  (Autoregressive Recurrent Network)      │
│    └────────┬─────────┘                                          │
│             │                                                    │
│    ┌────────▼─────────┐                                          │
│    │ Likelihood Head  │  (Gaussian or Negative Binomial)         │
│    └────────┬─────────┘                                          │
│             │                                                    │
│  ┌──────────▼──────────┐                                         │
│  │ Distribution Params │  (e.g., μ, σ)                           │
│  └─────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

```
STEP 1: CONDITIONING (History)
──────────────────────────────
Input: Previous values z_{t-1} & Covariates x_t
   ↓
Scale: z_scaled = z_{t-1} / ν
   ↓
LSTM: h_t = LSTM(z_scaled, x_t, h_{t-1})
   ↓
Head: μ, σ = Project(h_t)
   ↓
Output: Parameters for P(z_t | h_t)


STEP 2: PREDICTION (Future)
───────────────────────────
Input: Sampled value z_sample_{t-1} & Covariates x_t
   ↓
Scale: z_scaled = z_sample_{t-1} / ν
   ↓
LSTM: h_t = LSTM(z_scaled, x_t, h_{t-1})
   ↓
Head: μ, σ = Project(h_t)
   ↓
Sample: z_sample_t ~ Gaussian(μ * ν, σ * √ν)
   ↓
Loop: Feed z_sample_t back as input for next step
```

---

## 4. Architecture Deep Dive

### 4.1 `ScaleLayer`

The most critical component for stability.
-   **Forward**: Divides input by a scale factor $ \nu $ (usually $ 1 + \text{mean}(z_{history}) $).
-   **Inverse**: Multiplies model outputs by the scale factor to restore original magnitude.
-   Handles the logic differently for Mean (linear scaling) vs Standard Deviation (sqrt scaling) or Variance.

### 4.2 `GaussianLikelihoodHead`

Projects LSTM hidden states to distribution parameters for real-valued data.
-   **Mean ($\mu$)**: Affine transformation.
-   **Std ($\sigma$)**: Affine transformation followed by Softplus to ensure positivity.
-   Loss is computed as the negative log-likelihood of the observations given these parameters.

### 4.3 `NegativeBinomialLikelihoodHead`

Used for **count data** (non-negative integers).
-   Projects to Mean ($\mu$) and Shape ($\alpha$).
-   Models overdispersion (variance > mean), which is common in sales data.
-   Uses a Gamma-Poisson mixture formulation.

---

## 5. Quick Start Guide

### Installation

```bash
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First DeepAR Model

```python
import keras
import numpy as np
import matplotlib.pyplot as plt
from model import DeepAR

# 1. Generate synthetic data (Batch=32, Time=100, Dim=1)
# Sine wave with noise
t = np.linspace(0, 100, 100)
target = np.sin(t/10) + np.random.normal(0, 0.1, (32, 100, 1))
covariates = np.random.normal(0, 1, (32, 100, 5)) # Random covariates

# 2. Create Model
model = DeepAR(
    num_layers=2,
    hidden_dim=40,
    likelihood='gaussian',
    num_samples=100
)

# 3. Compile
model.compile(optimizer='adam', loss=model.gaussian_loss)

# 4. Train
# Inputs must be a dictionary
history = model.fit(
    x={'target': target, 'covariates': covariates},
    y=None, # y is unused; loss is calculated internally on 'target'
    epochs=10,
    batch_size=32
)

# 5. Predict
# Condition on first 80 steps, predict next 20
cond_len = 80
pred_len = 20

prediction_inputs = {
    'conditioning_target': target[:, :cond_len, :],
    'full_covariates': covariates # Uses all 100 steps
}

# Returns shape: (num_samples, batch, pred_len, target_dim)
samples = model.predict(prediction_inputs)

# 6. Visualize
# Calculate median and 90% interval
p50 = np.median(samples, axis=0)[0, :, 0]
p90 = np.percentile(samples, 90, axis=0)[0, :, 0]
p10 = np.percentile(samples, 10, axis=0)[0, :, 0]

plt.plot(np.arange(cond_len), target[0, :cond_len, 0], label='History')
plt.plot(np.arange(cond_len, 100), target[0, cond_len:, 0], label='True Future')
plt.plot(np.arange(cond_len, 100), p50, label='Median Pred')
plt.fill_between(np.arange(cond_len, 100), p10, p90, alpha=0.3, label='80% CI')
plt.legend()
plt.show()
```

---

## 6. Component Reference

### 6.1 `DeepAR` (Model Class)

**Location**: `model.DeepAR`

The main container class. Inherits from `keras.Model`.

```python
model = DeepAR(
    num_layers=3,
    hidden_dim=128,
    likelihood='gaussian', # or 'negative_binomial'
    dropout=0.1
)
```

### 6.2 Custom Blocks

**Location**: `deepar_blocks.py`

*   `ScaleLayer`: Handles normalization logic.
*   `GaussianLikelihoodHead`: Generates $\mu, \sigma$.
*   `NegativeBinomialLikelihoodHead`: Generates $\mu, \alpha$.
*   `DeepARCell`: Wraps LSTM cell logic (optional usage, the main model uses standard LSTM layers for efficiency).

---

## 7. Configuration & Likelihoods

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `num_layers` | int | 3 | Number of stacked LSTM layers. |
| `hidden_dim` | int | 40 | Number of units in each LSTM layer. |
| `likelihood` | str | 'gaussian' | 'gaussian' for real values, 'negative_binomial' for counts. |
| `dropout` | float | 0.0 | Dropout rate for LSTM inputs. |
| `num_samples`| int | 100 | Number of Monte Carlo samples during prediction. |
| `scale_epsilon`| float | 1.0 | Added to mean to prevent division by zero. |

### Choosing a Likelihood

*   **Gaussian**: Best for continuous data (temperature, voltage, large aggregate sales).
*   **Negative Binomial**: Best for count data, especially sparse data with many zeros or low integers (e.g., daily sales of a slow-moving product). It naturally outputs integers (approx) and handles zero-inflation better than Gaussian.

---

## 8. Comprehensive Usage Examples

### Example 1: Count Data Forecasting

When forecasting low-volume sales, use the Negative Binomial likelihood.

```python
model = DeepAR(
    num_layers=2,
    hidden_dim=64,
    likelihood='negative_binomial'
)

model.compile(optimizer='adam', loss=model.negative_binomial_loss)

# Target should be non-negative integers (floats are accepted but conceptually treated as counts)
model.fit({'target': count_data, 'covariates': feats}, epochs=5)
```

### Example 2: Prediction Mode Inputs

Unlike standard `model.predict(x)`, DeepAR requires specific keys during inference to handle the autoregressive loop.

*   `conditioning_target`: The history used to initialize the LSTM state and compute the scale.
*   `full_covariates`: Covariates for **both** the history and the future prediction horizon.

```python
# Total length = conditioning_len + prediction_len
# Covariates must cover the entire horizon
prediction = model.predict({
    'conditioning_target': history_data,  # Shape: (B, T_cond, D)
    'full_covariates': all_known_covariates # Shape: (B, T_cond + T_pred, F)
})
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Pre-computed Scales

By default, DeepAR computes the scale as the mean of the input target. If you have domain knowledge or want to fix scales (e.g., based on yearly average), you can pass them explicitly.

```python
# Manually compute scales
my_scales = np.mean(train_data, axis=1, keepdims=True) + 1.0

# Pass 'scale' in the input dict
model.fit({'target': train_data, 'covariates': cov, 'scale': my_scales})
```

### Pattern 2: Multi-variate Forecasting

DeepAR is primarily univariate (one target per series), but can model multiple *correlated* targets if `target_dim > 1`.

```python
# target_dim=3 means we predict 3 variables simultaneously
model = DeepAR(target_dim=3, hidden_dim=64)

# Data shape: (Batch, Time, 3)
model.fit(...)
```

---

## 10. Performance Optimization

### Batch Size
DeepAR benefits from large batch sizes (e.g., 128 or 256) because it learns global patterns. Small batches might lead to noisy gradient updates.

### Covariates
Including time-based covariates (e.g., "hour of day", "day of week", "is_holiday") is **crucial** for performance. Without them, the LSTM loses track of seasonality in long sequences.

### XLA Compilation
Using `jit_compile=True` usually works well with the LSTM layers in Keras 3.

```python
model.compile(optimizer='adam', loss=..., jit_compile=True)
```

---

## 11. Training and Best Practices

### Teacher Forcing
During training, the model receives the **true** previous target $z_{t-1}$ as input, not its own prediction. This stabilizes training.

### Loss Function Usage
You must use the specific loss function that matches your likelihood.

```python
# Correct
model = DeepAR(likelihood='gaussian')
model.compile(loss=model.gaussian_loss)

# Incorrect
model.compile(loss='mse') # Will not learn distribution parameters
```

### Scaling `epsilon`
If your data contains many zero-sequences, ensure `scale_epsilon` is large enough (e.g., 1.0) so the scale factor isn't dominated by noise. For very small value data (e.g., 0.001), lower `scale_epsilon`.

---

## 12. Serialization & Deployment

The model uses `keras.saving.register_keras_serializable`, making it compatible with the modern `.keras` format.

```python
# Save
model.save("deepar_model.keras")

# Load
# Note: You must pass the custom loss if loading for training
loaded_model = keras.models.load_model(
    "deepar_model.keras",
    custom_objects={
        'gaussian_loss': DeepAR.gaussian_loss,
        # or 'negative_binomial_loss': DeepAR.negative_binomial_loss
    }
)
```

---

## 13. Testing & Validation

### Unit Tests

```python
def test_deepar_shapes():
    batch, seq, dim = 4, 20, 1
    model = DeepAR(num_layers=1, hidden_dim=10, likelihood='gaussian')
    
    # Fake data
    x = {
        'target': np.random.normal(size=(batch, seq, dim)),
        'covariates': np.random.normal(size=(batch, seq, 5))
    }
    
    # Test Training Forward Pass
    out = model(x, training=True)
    assert 'mu' in out and 'sigma' in out
    assert out['mu'].shape == (batch, seq, dim)
    
    # Test Prediction Sampling
    pred_in = {
        'conditioning_target': x['target'][:, :10, :],
        'full_covariates': x['covariates']
    }
    # returns samples
    samples = model(pred_in, return_samples=True) 
    # Shape: (samples, batch, pred_len, dim)
    assert samples.shape == (100, batch, 10, dim)
    print("✅ Shapes Verified")
```

---

## 14. Troubleshooting & FAQs

**Issue: Loss is NaN.**
*   **Cause**: Usually exploding gradients or division by zero in the scale layer.
*   **Fix**: Clip gradients (`optimizer=keras.optimizers.Adam(clipnorm=1.0)`) or increase `scale_epsilon`.

**Issue: Predictions are flat lines.**
*   **Cause**: The LSTM hasn't learned the autoregressive connection.
*   **Fix**: Ensure `covariates` are normalized or embedded properly. Check that `target` isn't entirely noise. Increase `hidden_dim`.

**Q: Can I use this for classification?**
A: No, DeepAR is specifically for regression (continuous or count time series).

**Q: Why samples instead of just Mean/Variance?**
A: Non-linear transformations of future predictions (e.g., sum of sales over next week) require full trajectories to calculate correct quantiles. The sum of quantiles is not the quantile of the sum.

---

## 15. Technical Details

### Autoregressive Recurrent Networks
DeepAR assumes the value at time $t$, $z_t$, depends on the previous value $z_{t-1}$, the previous hidden state $h_{t-1}$, and current covariates $x_t$.
$$ h_t = \text{LSTM}(h_{t-1}, z_{t-1}, x_t) $$
$$ P(z_t | h_t) = \theta(h_t) $$

### Monte Carlo Sampling
During inference, we don't know $z_{t-1}$ for future steps. We approximate the distribution by drawing samples:
$$ \tilde{z}_t \sim P(z | \theta(\tilde{h}_t)) $$
This sampled value is fed back as input for step $t+1$. We repeat this $N$ times to get $N$ possible futures.

---

## 16. Citation

This implementation is based on the seminal paper by Amazon Research:

```bibtex
@article{salinas2017deepar,
  title={DeepAR: Probabilistic forecasting with autoregressive recurrent networks},
  author={Salinas, David and Flunkert, Valentin and Gasthaus, Jan and Januschowski, Tim},
  journal={International Journal of Forecasting},
  volume={36},
  number={3},
  pages={1181--1191},
  year={2020},
  publisher={Elsevier}
}
```