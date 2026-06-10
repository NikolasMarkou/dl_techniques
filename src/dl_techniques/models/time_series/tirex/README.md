# TiRex: Time Series Forecasting with Mixed Sequential Architectures

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of a **TiRex-inspired** time series forecasting model in **Keras 3**. This implementation is based on recent advancements in time series modeling that leverage hybrid architectures to capture both local and global dependencies.

The architecture's key feature is its use of **Mixed Sequential Blocks**, allowing for a flexible combination of LSTM and Transformer layers. It is designed for **probabilistic forecasting**, outputting a range of quantiles to provide a comprehensive view of prediction uncertainty.

---

## Table of Contents

1. [Overview: What is TiRex and Why It Matters](#1-overview-what-is-tirex-and-why-it-matters)
2. [The Problem TiRex Solves](#2-the-problem-tirex-solves)
3. [How TiRex Works: Core Concepts](#3-how-tirex-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
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

## 1. Overview: What is TiRex and Why It Matters

### What is TiRex?

**TiRex** is an advanced architecture for time series forecasting that synergizes the strengths of recurrent (LSTM) and attention-based (Transformer) models. Recent research on models like TiRex has shown that combining these approaches can lead to state-of-the-art performance in both short-term and long-term forecasting. This implementation is inspired by that philosophy.

Instead of relying on a single type of sequence processor, this model uses **Mixed Sequential Blocks**. Each block can be configured as an LSTM, a Transformer, or a hybrid of both, allowing the model to capture different types of temporal patterns simultaneously. The model is designed for **probabilistic forecasting**, predicting a range of quantiles rather than a single point estimate, which is crucial for decision-making under uncertainty.

### Key Innovations

1.  **Hybrid Sequential Blocks**: The core of the model is its ability to mix and match LSTM and Transformer layers. This allows it to leverage the stateful, ordered processing of LSTMs for local patterns and the global, long-range dependency modeling of Transformers.
2.  **Patch-based Tokenization**: The input time series is divided into patches (sub-sequences), which are then embedded. This is analogous to how Vision Transformers process images and allows the model to operate on higher-level temporal features.
3.  **Probabilistic Forecasting**: The model outputs predictions for multiple quantiles (e.g., 0.1, 0.5, 0.9), providing a full forecast distribution. This is essential for risk assessment and robust planning.
4.  **In-Context Learning Capabilities**: Architectures like TiRex are often designed for zero-shot or few-shot forecasting, where the model can make predictions on new time series with little to no specific training by using the recent history as context.

### Why TiRex Matters

**Traditional Forecasting Models**:
```
Model: ARIMA / ETS (Statistical)
  1. Relies on statistical properties like trend and seasonality.
  2. Works well for simple, regular patterns but struggles with complex,
     non-linear dynamics and external variables.

Model: Standard RNN/LSTM
  1. Good at capturing sequential dependencies.
  2. Struggles with very long-range dependencies due to the vanishing gradient problem
     and sequential processing bottleneck.

Model: Standard Transformer
  1. Excellent at capturing long-range, content-based relationships.
  2. Lacks the inherent sequential inductive bias of RNNs, which can make it less
     efficient for some time series tasks.
```

**TiRex's Solution**:
```
TiRex-inspired Approach:
  1. Combines LSTM and Transformer blocks in a unified architecture.
  2. The LSTM component models local, evolving states.
  3. The Transformer component models global relationships between distant time steps.
  4. The quantile head provides a rich, probabilistic output.
  5. Benefit: Achieves a "best-of-both-worlds" model that is robust across a wide
     variety of time series patterns and forecasting horizons. [4]
```

### Real-World Impact

This hybrid approach is critical for complex, real-world forecasting problems:

-   📈 **Finance**: Predicting stock prices, where both short-term momentum (LSTM) and long-term market regime changes (Transformer) are important.
-   ⚡ **Energy**: Forecasting electricity demand, which is influenced by both recent usage patterns and weekly/yearly seasonalities.
-   📦 **Supply Chain**: Predicting product demand, which depends on recent sales trends, promotions, and seasonal events.
-   ☁️ **Cloud Operations**: Forecasting resource usage (CPU, memory) which has both bursty, short-term behavior and long-term cyclical patterns.

---

## 2. The Problem TiRex Solves

### The Dichotomy of Time Series Patterns

Time series data is complex, often containing a mix of different patterns occurring at different scales.

```
┌─────────────────────────────────────────────────────────────┐
│  The Challenge of Multi-Scale Dependencies                  │
│                                                             │
│  1. Local Sequential Patterns:                              │
│     - How the value at time `t` depends on `t-1`, `t-2`,etc │
│     - Best captured by recurrent models like LSTMs that     │
│       process data in order and maintain a state.           │
│                                                             │
│  2. Global / Long-Range Patterns:                           │
│     - How an event today is related to a similar event that │
│       happened weeks or months ago (e.g., a holiday sale).  │
│     - Best captured by attention mechanisms that can        │
│       directly compare any two points in the sequence,      │
│       regardless of distance.                               │
└─────────────────────────────────────────────────────────────┘
```

Relying on only one type of model means you are compromising on the ability to effectively capture the other type of pattern.

### How The Mixed Architecture Changes the Game

This TiRex-inspired model provides a flexible framework to handle this dichotomy.

```
┌─────────────────────────────────────────────────────────────┐
│  The TiRex Hybrid Solution                                  │
│                                                             │
│  1. Configurable Blocks: You can design the model with a    │
│     stack of blocks, specifying the type for each one.      │
│     - Start with `lstm` blocks to summarize local trends.   │
│     - Follow with `transformer` blocks to find global       │
│       relationships between these summarized trends.        │
│     - Use `mixed` blocks for a tight integration of both.   │
│                                                             │
│  2. Probabilistic Output: Instead of a single, potentially  │
│     wrong prediction, the model outputs a range of possible │
│     outcomes, quantifying its own uncertainty. This is      │
│     vital for risk management.                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How TiRex Works: Core Concepts

### The High-Level Architecture

The model transforms a raw time series into probabilistic future predictions through a multi-stage pipeline.

```
┌──────────────────────────────────────────────────────────────────┐
│                     TiRex-inspired Architecture                  │
│                                                                  │
│  Input Time Series ───►┌────────────────┐                        │
│                        │ StandardScaler │                        │
│                        └────────┬───────┘                        │
│                                 │                                │
│                        ┌────────▼───────┐                        │
│                        │ PatchEmbedding │ (Tokenization)         │
│                        └────────┬───────┘                        │
│                                 │                                │
│                        ┌────────▼───────────┐                    │
│                        │ Mixed Sequential   │ (Repeated N times) │
│                        │ Blocks (LSTM/TF)   │                    │
│                        └────────┬───────────┘                    │
│                                 │                                │
│                        ┌────────▼───────────┐                    │
│                        │ GlobalAvgPooling   │                    │
│                        └────────┬───────────┘                    │
│                                 │                                │
│  ┌──────────────────┐  ┌────────▼───────┐                        │
│  │ Probabilistic    │◄─┤ Quantile Head  │                        │
│  │ Forecast         │  └────────────────┘                        │
│  └──────────────────┘                                            │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TiRex Complete Data Flow                            │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: PREPROCESSING & EMBEDDING
─────────────────────────────────
Input Time Series (B, L, F)
    │
    ├─► StandardScaler: Normalize the data (z-score).
    │
    ├─► NaN Handling: Create a mask for missing values.
    │
    ├─► Concatenate Data + Mask: (B, L, F * 2)
    │
    ├─► PatchEmbedding1D: Convert sequence into patches -> (B, num_patches, D)
    │
    └─► Input Projection (ResidualBlock) -> (B, num_patches, D)


STEP 2: SEQUENTIAL PROCESSING
─────────────────────────────
Patched Sequence (B, num_patches, D)
    │
    ├─► MixedSequentialBlock 1 (e.g., 'lstm')
    │   ├─► [Norm] -> LSTM -> [Add & Norm]
    │   └─► [Norm] -> FFN  -> [Add & Norm]
    │
    ├─► MixedSequentialBlock 2 (e.g., 'transformer')
    │   ├─► [Norm] -> Self-Attention -> [Add & Norm]
    │   └─► [Norm] -> FFN            -> [Add & Norm]
    │
    ├─► ... (repeat for N blocks) ...
    │
    └─► Final Hidden States (B, num_patches, D)


STEP 3: PROJECTION & FORECASTING
────────────────────────────────
Final Hidden States (B, num_patches, D)
    │
    ├─► Output Normalization
    │
    ├─► Global Average Pooling -> Pooled Features (B, D)
    │
    ├─► QuantileHead
    │   ├─► Dense Layer to predict (num_quantiles * pred_len) values
    │   └─► Reshape
    │
    └─► Quantile Forecast (B, num_quantiles, pred_len)
```

---

## 4. Architecture Deep Dive

### 4.1 `StandardScaler` Layer

A Keras-native, invertible z-score normalization layer.
-   Computes mean and standard deviation on-the-fly for each batch.
-   Handles `NaN` values gracefully.
-   Includes an `inverse_transform` method to convert model outputs back to the original data scale.

### 4.2 `PatchEmbedding1D` Layer

Converts the input time series into a sequence of tokens.
-   Implemented as a `Conv1D` layer where `kernel_size` is the `patch_size`.
-   `strides` controls the overlap between patches. `stride < patch_size` creates overlapping patches.
-   The input to this layer includes a mask concatenated to the features, allowing the model to learn from missing data.

### 4.3 `MixedSequentialBlock`

The core building block of the model. It can operate in three modes:
-   **`'lstm'`**: A Pre-Norm block with an LSTM layer followed by a Feed-Forward Network (FFN).
-   **`'transformer'`**: A standard Pre-Norm Transformer block (Self-Attention -> FFN).
-   **`'mixed'`**: A three-stage block: LSTM -> Self-Attention -> FFN, with residual connections and pre-normalization at each stage. This allows the model to first capture local context with the LSTM before modeling global relationships with attention.

### 4.4 `QuantileHead`

The final layer that produces the probabilistic forecast.
-   Takes the final pooled feature vector from the encoder.
-   Uses a single `Dense` layer to project this vector to the desired output shape: `num_quantiles * prediction_length`.
-   Reshapes the output to `(batch_size, num_quantiles, prediction_length)`.
-   Designed to be trained with a quantile loss (pinball loss).

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy matplotlib
```

### Your First Forecasting Model (30 seconds)

Let's build a simple model to forecast a sine wave.

```python
import keras
import numpy as np
import matplotlib.pyplot as plt

# Local imports from your project structure
from dl_techniques.models.time_series.tirex.model import create_tirex_by_variant

# 1. Generate synthetic time series data (sine wave)
def generate_data(num_samples, seq_len, pred_len):
    X = np.zeros((num_samples, seq_len, 1))
    y = np.zeros((num_samples, pred_len, 1))
    for i in range(num_samples):
        start = np.random.rand() * 10
        t = np.linspace(start, start + seq_len + pred_len, seq_len + pred_len)
        series = np.sin(t)
        X[i, :, 0] = series[:seq_len]
        y[i, :, 0] = series[seq_len:]
    return X, y

SEQ_LEN, PRED_LEN = 128, 32
X_train, y_train = generate_data(1000, SEQ_LEN, PRED_LEN)
X_val, y_val = generate_data(200, SEQ_LEN, PRED_LEN)

# 2. Create a tiny TiRex model
model = create_tirex_by_variant(
    variant="tiny",
    input_length=SEQ_LEN,
    prediction_length=PRED_LEN,
    quantile_levels=[0.1, 0.5, 0.9]
)

# 3. Define the quantile loss function (pinball loss)
def quantile_loss(y_true, y_pred):
    quantiles = model.quantile_levels
    y_true = ops.expand_dims(y_true, axis=1) # Shape: (B, 1, PRED_LEN)
    error = y_true - y_pred
    q = ops.reshape(ops.array(quantiles, dtype=error.dtype), (1, len(quantiles), 1))
    return ops.mean(ops.maximum(q * error, (q - 1) * error))

# 4. Compile and train the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=quantile_loss)
print("✅ TiRex model created and compiled successfully!")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)
print("✅ Training Complete!")

# 5. Generate and visualize a forecast
test_input = X_val[0:1]
forecasts = model.predict(test_input)[0] # Shape: (num_quantiles, PRED_LEN)

plt.figure(figsize=(12, 6))
plt.plot(np.arange(SEQ_LEN), test_input[0, :, 0], label="Input Context")
plt.plot(np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN), y_val[0, :, 0], label="Ground Truth", linestyle='--')
plt.plot(np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN), forecasts[1, :], label="Median Forecast (p50)")
plt.fill_between(np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN), forecasts[0, :], forecasts[2, :], color='orange', alpha=0.3, label="80% Prediction Interval (p10-p90)")
plt.legend()
plt.title("TiRex Probabilistic Forecast")
plt.show()
```

---

## 6. Component Reference

### 6.1 `TiRexCore` (Model Class)

**Purpose**: The main Keras `Model` subclass that assembles the entire TiRex-inspired architecture.

**Location**: `dl_techniques.models.time_series.tirex.model.TiRexCore`

```python
from dl_techniques.models.time_series.tirex.model import TiRexCore

model = TiRexCore.from_variant(
    "small",
    prediction_length=48,
    block_types=['lstm', 'lstm', 'mixed', 'transformer'] # Custom block stack
)
```

### 6.2 Factory Functions

**Location**: `dl_techniques.models.time_series.tirex.model`

#### `create_tirex_model(...)`
A general-purpose factory for creating a custom TiRex model from scratch.

#### `create_tirex_by_variant(...)`
The recommended way to create standard TiRex configurations (`tiny`, `small`, etc.).

---

## 7. Configuration & Model Variants

This implementation provides several standard configurations.

| Variant | Patch Size | Embed Dim | Blocks | Heads | Use Case |
|:---:|:---:|:---:|:---:|:---:|:---|
| **`tiny`** | 8 | 64 | 3 | 4 | Quick tests, simple series |
| **`small`**| 12 | 128 | 6 | 8 | General-purpose forecasting |
| **`medium`**| 16 | 256 | 8 | 8 | Complex datasets |
| **`large`** | 16 | 512 | 12| 16| High-performance, long sequences |

### Customizing `block_types`

The `block_types` parameter is a powerful way to customize the model's architecture.

-   **`['lstm'] * N`**: Creates a deep LSTM-based model (similar to a stacked LSTM with residual connections).
-   **`['transformer'] * N`**: Creates a deep Transformer-based model.
-   **`['lstm', 'lstm', 'transformer', 'transformer']`**: A common pattern where initial layers capture local patterns and later layers model global interactions.
-   **`['mixed'] * N`**: The default, where every block uses the full LSTM -> Attention -> FFN pipeline for maximum expressiveness.

---

## 8. Comprehensive Usage Examples

### Example 1: Multi-variate Forecasting

The model naturally handles multi-variate time series. Simply provide an input with multiple features.

```python
# Input shape: (batch_size, sequence_length, num_features)
# Let's say we have 5 features.
SEQ_LEN, PRED_LEN, NUM_FEATS = 96, 24, 5
X_train_multi = np.random.rand(1000, SEQ_LEN, NUM_FEATS)
# Target is the first feature
y_train_multi = X_train_multi[:, -PRED_LEN:, 0:1]

model = create_tirex_by_variant(
    "small",
    input_length=SEQ_LEN,
    prediction_length=PRED_LEN
)
# model.compile(...) and model.fit(X_train_multi, y_train_multi, ...)
```

### Example 2: Handling Missing Data

The model is designed to be robust to missing data (`NaN` values).

```python
# Introduce some missing values into the data
X_train_nan = X_train.copy()
X_train_nan[0, 10:20, 0] = np.nan
X_train_nan[1, 50:60, 0] = np.nan

# The model will automatically create a mask and handle the NaNs
# during the patch embedding step.
# model.fit(X_train_nan, y_train, ...)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Custom Quantiles for Risk Analysis

For financial forecasting, you might be interested in tail risk. You can configure the model to predict extreme quantiles.

```python
# Focus on 95% prediction interval and the 1% and 99% tails
risk_quantiles = [0.01, 0.025, 0.5, 0.975, 0.99]

risk_model = create_tirex_by_variant(
    "medium",
    input_length=256,
    prediction_length=30,
    quantile_levels=risk_quantiles
)
```

### Pattern 2: Using the Model as a Feature Extractor

You can create a version of the model that outputs the final hidden states, which can be used as rich features for other downstream tasks (e.g., classification, anomaly detection).

```python
# 1. Create the base TiRex model
base_model = create_tirex_by_variant("small", input_length=128, prediction_length=32)

# 2. Create a new model that outputs the pooled features before the quantile head
feature_extractor = keras.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("output_norm").output # Or the pooled output
)

# 3. Now you can extract features
features = feature_extractor.predict(X_val)
print(f"Extracted features shape: {features.shape}")
```

---

## 10. Performance Optimization

### Mixed Precision Training

For deep `large` models, mixed precision can significantly speed up training on compatible GPUs.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = create_tirex_by_variant("large", ...)
# ... compile and fit ...
```

### XLA Compilation

Use `jit_compile=True` on the `train_step` or `predict` function for graph compilation, which can provide a speed boost.

```python
model = create_tirex_by_variant("medium", ...)
model.compile(optimizer="adam", loss=quantile_loss, jit_compile=True)
# Keras will now attempt to compile the training step with XLA.```

---

## 11. Training and Best Practices

### The Quantile (Pinball) Loss

Probabilistic forecasting requires a special loss function. The **pinball loss** is the standard for quantile regression. The `Quick Start` guide shows a basic implementation. This loss penalizes over- and under-predictions asymmetrically based on the target quantile, encouraging the model to learn the entire conditional distribution.

### Handling Non-Stationarity

The built-in `StandardScaler` layer performs per-batch normalization, which makes the model adaptive to non-stationary data where the mean and variance can shift over time. This is a key advantage over using a global, fixed scaler.

### Tuning the Architecture

-   **`patch_size`**: A smaller `patch_size` allows the model to see more fine-grained patterns but creates longer sequences for the Transformer, increasing computational cost. A larger `patch_size` is more efficient but might smooth over important local details.
-   **`block_types`**: Experiment with the sequence of blocks. For data with strong seasonality, starting with `transformer` blocks to capture long-range dependencies might be effective. For noisy, trend-based data, starting with `lstm` blocks might be better.

---

## 12. Serialization & Deployment

The `TiRexCore` model and all its custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train model
model = create_tirex_by_variant("small", ...)
# model.compile(...) and model.fit(...)

# Save the entire model
model.save('my_tirex_model.keras')

# Load the model in a new session
loaded_model = keras.models.load_model('my_tirex_model.keras', custom_objects={'quantile_loss': quantile_loss})
```
*(Note: The custom loss function needs to be passed during loading if it's not a built-in Keras loss).*

---

## 13. Testing & Validation

### Unit Tests

```python
import keras
import numpy as np
from dl_techniques.models.time_series.tirex.model import create_tirex_by_variant

def test_model_creation_all_variants():
    """Test model creation for all variants."""
    for variant in TiRexCore.MODEL_VARIANTS.keys():
        model = create_tirex_by_variant(variant, input_length=64, prediction_length=16)
        assert model is not None
        print(f"✓ TiRex-{variant} created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = create_tirex_by_variant("tiny", input_length=96, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])
    dummy_input = np.random.rand(4, 96, 1)
    output = model.predict(dummy_input)
    assert output.shape == (4, 3, 24) # (batch, num_quantiles, pred_len)
    print("✓ Forward pass has correct shape")

# Run tests
if __name__ == '__main__':
    test_model_creation_all_variants()
    test_forward_pass_shape()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: The model outputs quantile crossings (e.g., the 90th percentile is lower than the 50th).**

-   **Cause**: The standard quantile loss doesn't explicitly penalize quantile crossing. While models tend to learn the correct ordering, it's not guaranteed, especially with limited data or during early training.
-   **Solution**: This is a known challenge in quantile regression. Advanced techniques involve adding a penalty term to the loss for crossings or using architectures that enforce monotonicity by design. For most applications, minor or infrequent crossings can be ignored or corrected with a post-processing step.

**Issue 2: The model is not learning and the loss is stagnant.**

-   **Cause 1**: The learning rate might be too high or too low.
-   **Cause 2**: The data may not be scaled properly. Although this model has a built-in scaler, ensure the input data doesn't have extreme outliers that could dominate the batch statistics.
-   **Cause 3**: The problem might be too complex for the chosen model variant. Try a larger model or a different configuration of `block_types`.

### Frequently Asked Questions

**Q: Why use patch embedding instead of feeding the time series directly?**

A: Patching serves two purposes: 1) It reduces the sequence length fed into the Transformer blocks, making self-attention computationally feasible for long time series. 2) It allows the model to learn local features within a patch before considering relationships between patches, which can be a more robust way to model time series.

**Q: Can this model be used for zero-shot forecasting?**

A: The architecture is suitable for it, but it would require pre-training on a massive and diverse dataset of time series, which is a significant undertaking. The original TiRex paper focuses on this pre-trained, zero-shot capability. This implementation provides the *architecture*, which you can then use for pre-training or standard train-from-scratch workflows.

---

## 15. Technical Details

### The Complementary Nature of LSTMs and Transformers

-   **LSTMs**: Their recurrent nature creates a strong inductive bias for ordered, sequential data. The hidden state acts as a compressed summary of the entire past, which is updated at each time step. This is computationally efficient (`O(L * D²)`) but can struggle to precisely recall very old information.
-   **Transformers**: Their self-attention mechanism is permutation-equivariant (ignoring positional encodings). It has no inherent bias for sequential order but can model content-based relationships between any two points in the sequence, regardless of distance. This is powerful but computationally more expensive (`O(L² * D)`).

The `mixed` block in this implementation hypothesizes that by feeding the output of an LSTM (which contains localized, stateful context) into a Transformer, the self-attention mechanism can operate on a more semantically meaningful sequence, leading to better overall performance.

---

## 16. Citation

This implementation is inspired by the principles of recent hybrid time series models. If using these concepts in research, consider citing relevant foundational work:

-   The original TiRex paper:
    ```bibtex
    @article{auer2025tirex,
      title={TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning},
      author={Auer, Andreas and Podest, Patrick and Klotz, Daniel and B{\"o}ck, Sebastian and Klambauer, G{\"u}nter and Hochreiter, Sepp},
      journal={arXiv preprint arXiv:2505.23719},
      year={2025}
    }
    ```