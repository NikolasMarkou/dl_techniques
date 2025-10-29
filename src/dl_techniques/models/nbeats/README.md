# N-BEATS: Neural Basis Expansion Analysis for Time Series

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of **N-BEATS (Neural Basis Expansion Analysis for Time Series)**, a deep learning architecture for time series forecasting that is often competitive with or superior to statistical and recurrent models. This implementation is designed for interpretability and performance, featuring specialized blocks for trend and seasonality decomposition.

This implementation follows the `dl_techniques` framework standards and modern Keras 3 best practices. It provides a modular, well-documented, and fully serializable model that works seamlessly across TensorFlow, PyTorch, and JAX backends. It also includes an integrated **Reversible Instance Normalization (RevIN)** layer, a critical component for handling distribution shifts and improving forecasting accuracy on real-world data.

---

## Table of Contents

1. [Overview: What is N-BEATS and Why It Matters](#1-overview-what-is-n-beats-and-why-it-matters)
2. [The Problem N-BEATS Solves](#2-the-problem-n-beats-solves)
3. [How N-BEATS Works: Core Concepts](#3-how-n-beats-works-core-concepts)
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

## 1. Overview: What is N-BEATS and Why It Matters

### What is N-BEATS?

**N-BEATS** is a deep neural architecture for time series forecasting based on the principle of **basis expansion**. Instead of using recurrent connections (like LSTMs) or self-attention (like Transformers), N-BEATS uses a deep stack of fully-connected layers to learn coefficients (`theta`) for a set of pre-defined or learnable basis functions. These functions then generate the forecast, allowing the model to decompose a time series into interpretable components like trend and seasonality.

### Key Innovations

1.  **Doubly Residual Stacking**: The model is composed of stacks, and each stack contains multiple blocks. Each block models a part of the input signal, produces a forecast, and a "backcast" (a reconstruction of its input). The backcast is subtracted from the input, and the residual is passed to the next block. This allows the model to progressively decompose the time series.
2.  **Interpretable by Design**: By using specific block types, the model can be forced to learn interpretable components. `TrendBlock`s use polynomial basis functions to model trends, and `SeasonalityBlock`s use Fourier series to model periodic patterns.
3.  **No Recurrence or Convolution**: The architecture is deliberately simple, using only fully-connected layers. This makes it fast and avoids the complex dependencies of recurrent or attention mechanisms.
4.  **Integrated RevIN Normalization**: This implementation includes Reversible Instance Normalization (RevIN), which normalizes each time series instance independently. This makes the model robust to distribution shifts—a common and difficult problem in time series—and typically improves accuracy by a significant margin.

### Why N-BEATS Matters

**The Time Series Forecasting Challenge**:
```
Problem: Forecast future values of a time series that may have complex trends,
         multiple seasonalities, and whose statistical properties change over time.
Traditional Approaches:
  - ARIMA/ETS: Strong statistical foundations, but struggle with non-linear patterns
    and multiple seasonalities.
  - LSTMs/Transformers: Powerful, but can be complex, slow to train, and often
    act as "black boxes," making their forecasts difficult to interpret or debug.
```

**N-BEATS's Solution**:
```
N-BEATS's Approach:
  1. Decompose the problem: Use residual stacking to break the signal down into
     simpler components that can be modeled independently.
  2. Be interpretable: Force stacks to model specific, understandable patterns
     like trend and seasonality using mathematical basis functions.
  3. Be robust: Use RevIN to handle shifts in the data's mean and variance,
     making the model more reliable in production environments.
  4. Benefit: Achieves state-of-the-art performance with a simpler, faster,
     and more interpretable architecture than many alternatives.
```

---

## 2. The Problem N-BEATS Solves

### The Challenge of Non-Stationarity and Distribution Shift

Real-world time series data is rarely well-behaved. Its statistical properties—mean, variance, and seasonal patterns—can change unexpectedly over time. This is known as **non-stationarity** or **distribution shift**.

```
┌─────────────────────────────────────────────────────────────┐
│  The Dilemma of Time Series Models                          │
│                                                             │
│  The Past is Not Always Like the Future:                    │
│    - A model trained on historical data may fail when the   │
│      underlying data distribution changes.                  │
│    - For example, a sales forecasting model trained before  │
│      a marketing campaign may not work well after it.       │
│                                                             │
│  The Black Box Problem:                                     │
│    - When a complex model like an LSTM makes a bad forecast,│
│      it can be very difficult to understand why. Was it the │
│      trend? A seasonal effect? A special event?             │
└─────────────────────────────────────────────────────────────┘
```

N-BEATS is explicitly designed to address both of these issues.

### How N-BEATS Changes the Game

It provides a principled framework for decomposition and robustness.

```
┌─────────────────────────────────────────────────────────────┐
│  The N-BEATS Decomposition & Normalization Strategy         │
│                                                             │
│  1. Handle Distribution Shift with RevIN:                   │
│     - By normalizing each input window, RevIN makes the main│
│       model's task stationary. The model can focus purely   │
│       on learning temporal patterns, not the series' scale. │
│     - The forecast is then denormalized back to the correct │
│       scale, making it accurate and robust.                 │
│                                                             │
│  2. Decompose the Signal with Residual Stacks:              │
│     - Stack 1 (Trend): Models the long-term trend, then     │
│       subtracts it from the signal.                         │
│     - Stack 2 (Seasonality): Models seasonal patterns in the│
│       remaining "de-trended" signal.                        │
│     - This hierarchical decomposition allows the model to   │
│       produce interpretable and debuggable forecasts.       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How N-BEATS Works: Core Concepts

### The Hierarchical Multi-Stack Architecture

The model processes an input window (`backcast_length`) to predict a future window (`forecast_length`) through a series of stacks.

```
┌──────────────────────────────────────────────────────────────────┐
│                      N-BEATS Architecture Stages                 │
│                                                                  │
│  Input ───► RevIN ───►┌──────────────────┐  (e.g., Trend Stack)  │
│  (H, D_in)             │      Stack 1     ├─► Forecast₁          │
│                        └────────┬─────────┘                      │
│                                 │ (Residual₁ = Input - Backcast₁)│
│                        ┌────────▼─────────┐  (e.g., Seasonality) │
│                        │      Stack 2     ├─► Forecast₂          │
│                        └────────┬─────────┘                      │
│                                 │ (Residual₂ = Res₁ - Backcast₂) │
│                                 ...                              │
│                                 │                                │
│ Final Forecast = Forecast₁ + Forecast₂ + ... ◄── RevIN Denorm◄───┘
│  (T, D_out)                                                      │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       N-BEATS Complete Data Flow                        │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: NORMALIZATION
─────────────────────
Input Time Series (B, H, D_in)
    │
    ├─► RevIN Layer (computes & stores instance-wise mean/std)
    │
    └─► Normalized Input (B, H, D_in)


STEP 2: STACK PROCESSING (Repeated for each stack)
──────────────────────────────────────────────────
Input to Stack `i` (initially the Normalized Input)
    │
    ├─► Pass through each Block in the Stack `i`
    │   - Each block `j` takes the residual from block `j-1`
    │   - It produces a backcast_j and a forecast_j
    │   - The residual is updated: residual_j = residual_{j-1} - backcast_j
    │
    ├─► Stack Forecast = Σ (forecast_j from all blocks in stack)
    │
    └─► Update Global Residual = Input to Stack `i` - Σ (backcast_j)


STEP 3: AGGREGATION & DENORMALIZATION
─────────────────────────────────────
Stack Forecasts (Forecast₁, Forecast₂, ...)
    │
    ├─► Final Forecast = Σ (All Stack Forecasts)
    │
    ├─► RevIN Denormalize (uses stored mean/std to restore scale)
    │
    └─► Denormalized Forecast (B, T, D_in)


STEP 4: OUTPUT PROJECTION
─────────────────────────
Denormalized Forecast
    │
    ├─► [Optional] Dense Layer if D_in != D_out
    │
    └─► Final Output (B, T, D_out)
```

---

## 4. Architecture Deep Dive

### 4.1 `RevIN` (Reversible Instance Normalization)

-   **Purpose**: To make the model robust to changes in the time series' mean and variance (distribution shift).
-   **Functionality**:
    1.  **Forward**: For each time series in a batch, it computes the mean and standard deviation across the time dimension, normalizes the series to have zero mean and unit variance, and *stores* these statistics.
    2.  **Denormalize**: After the main model makes a forecast in the normalized space, this layer uses the stored statistics to scale the forecast back to the original data distribution.

### 4.2 `NBeatsBlock` (Base Class)

-   **Purpose**: The core computational unit.
-   **Architecture**:
    1.  A stack of four fully-connected `Dense` layers with `silu` activation to extract features from the flattened input time series.
    2.  Two linear "theta" heads that project the features into coefficients for the backcast and forecast.
-   This base class is abstract; the actual signal generation happens in its subclasses.

### 4.3 `TrendBlock`

-   **Purpose**: To model the long-term trend of the time series.
-   **Basis Functions**: It uses **polynomials** (`t^0, t^1, t^2, ...`). The `thetas_dim` parameter determines the degree of the polynomial (e.g., `thetas_dim=3` fits a quadratic trend). The basis functions are defined over a continuous time index spanning both the backcast and forecast periods to ensure a smooth, continuous trend extrapolation.

### 4.4 `SeasonalityBlock`

-   **Purpose**: To model periodic patterns (e.g., daily, weekly, yearly cycles).
-   **Basis Functions**: It uses a **Fourier series** (`cos(2πkt), sin(2πkt)`). The `thetas_dim` parameter determines the number of harmonics `k` to use, allowing it to capture seasonalities of different frequencies.

### 4.5 `GenericBlock`

-   **Purpose**: To model any patterns not captured by the trend and seasonality blocks.
-   **Basis Functions**: It uses a fully **learnable linear basis**, implemented as two `Dense` layers. This gives it the flexibility to approximate any arbitrary signal component.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First N-BEATS Model (30 seconds)

Let's build an interpretable univariate forecasting model using the factory function.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.time_series.nbeats.model import create_nbeats_model

# 1. Define model parameters
BACKCAST_LENGTH = 96  # How many past steps to look at (e.g., 4 days of hourly data)
FORECAST_LENGTH = 24  # How many future steps to predict (e.g., 1 day)

# 2. Create and compile an interpretable N-BEATS model
# The factory handles optimal defaults and compilation.
model = create_nbeats_model(
    backcast_length=BACKCAST_LENGTH,
    forecast_length=FORECAST_LENGTH,
    stack_types=['trend', 'seasonality'] # Interpretable configuration
)
print("✅ N-BEATS model created and compiled successfully!")
model.summary()

# 3. Create dummy data for a forward pass
batch_size = 16
dummy_input = np.random.randn(batch_size, BACKCAST_LENGTH, 1).astype("float32")
dummy_target = np.random.randn(batch_size, FORECAST_LENGTH, 1).astype("float32")

# 4. Train for one step
# RevIN is active during training
loss = model.train_on_batch(dummy_input, dummy_target)
print(f"\n✅ Training step complete! Loss: {loss:.4f}")

# 5. Run inference
# RevIN's denormalization is active during inference
predictions = model.predict(dummy_input)
print(f"Predictions shape: {predictions.shape}") # (batch_size, forecast_length, 1)
```

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`NBeatsNet`** | `...nbeats.model.NBeatsNet` | The main Keras `Model` that assembles the N-BEATS stacks. |
| **`create_nbeats_model`** | `...nbeats.model.create_nbeats_model` | Recommended factory function to create and compile models. |
| **`NBeatsBlock`** | `...layers.time_series.nbeats_blocks.NBeatsBlock` | The abstract base class for all blocks. |
| **`GenericBlock`** | `...layers.time_series.nbeats_blocks.GenericBlock` | Block with a learnable basis for complex patterns. |
| **`TrendBlock`** | `...layers.time_series.nbeats_blocks.TrendBlock` | Block with a polynomial basis for trend modeling. |
| **`SeasonalityBlock`**| `...layers.time_series.nbeats_blocks.SeasonalityBlock`| Block with a Fourier basis for seasonality modeling. |
| **`RevIN`** | `...layers.time_series.revin.RevIN` | Reversible instance normalization layer for robustness. |

---

## 7. Configuration & Model Variants

N-BEATS is configured by the `stack_types` list, which determines its behavior.

| Configuration | `stack_types` | Description |
|:---:|:---|:---|
| **Interpretable** | `['trend', 'seasonality']` | The standard configuration for interpretable forecasting. It decomposes the signal into a trend and seasonal component. |
| **Generic**| `['generic', 'generic']` | A powerful "black-box" configuration that often achieves the highest accuracy but lacks interpretability. |
| **Hybrid** | `['trend', 'seasonality', 'generic']` | A configuration that first removes trend and seasonality, then models the complex remainder with a generic stack. |

You can create any combination of these stacks to suit your problem.

---

## 8. Comprehensive Usage Examples

### Example 1: Standard Interpretable Univariate Model

This is the most common use case, created easily with the factory.

```python
# The factory automatically calculates good theta_dims
model = create_nbeats_model(
    backcast_length=168,  # 7 days of hourly data
    forecast_length=24,   # Predict next day
    stack_types=['trend', 'seasonality'],
    hidden_layer_units=512,
    learning_rate=1e-4
)
```

### Example 2: High-Performance Generic Model

For tasks where accuracy is paramount and interpretability is not needed.

```python
# A generic model often benefits from more blocks and hidden units
model = create_nbeats_model(
    backcast_length=168,
    forecast_length=24,
    stack_types=['generic', 'generic', 'generic'], # Three generic stacks
    nb_blocks_per_stack=4,
    hidden_layer_units=1024
)
```

### Example 3: Multivariate Forecasting

This implementation fully supports multivariate forecasting (multiple input and output features).

```python
# Forecasting 3 target variables using 7 input variables
model = create_nbeats_model(
    backcast_length=96,
    forecast_length=24,
    stack_types=['trend', 'seasonality', 'generic'],
    input_dim=7,
    output_dim=3,
    hidden_layer_units=512
)
# Input shape: (batch, 96, 7)
# Output shape: (batch, 24, 3)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Customizing Theta Dimensions

You can manually control the complexity of the trend and seasonality blocks by setting `thetas_dim`.

```python
# Model a simple linear trend and a detailed seasonality with 10 harmonics
model = create_nbeats_model(
    backcast_length=96,
    forecast_length=24,
    stack_types=['trend', 'seasonality'],
    thetas_dim=[2, 20]  # Trend: degree 1 (t^0, t^1). Seasonality: 10 harmonics (10 cos + 10 sin)
)
```

---

## 10. Performance Optimization

### Mixed Precision Training

N-BEATS is composed of dense layers, which benefit significantly from mixed precision on modern GPUs.

```python
# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# The factory will create a model that uses mixed precision
model = create_nbeats_model(...)
```

### Backcast/Forecast Ratio

The performance of N-BEATS is sensitive to the ratio of `backcast_length` to `forecast_length`. A common rule of thumb is to use a ratio between **3 and 7**. The factory function will warn you if your chosen ratio is too low.

---

## 11. Training and Best Practices

### Optimizer and Loss Function

-   **Optimizer**: **Adam** or **AdamW** are excellent choices.
-   **Gradient Clipping**: N-BEATS training can sometimes be unstable. Using gradient clipping is **highly recommended**. The `create_nbeats_model` factory enables it by default (`clipnorm=1.0`).
-   **Loss Function**: **Mean Absolute Error (MAE)** is often preferred over Mean Squared Error (MSE) for N-BEATS, as it is less sensitive to outliers, which are common in time series.

### Learning Rate

-   A relatively small learning rate (e.g., `1e-4`) combined with a cosine decay schedule often yields the best and most stable results.

---

## 12. Serialization & Deployment

The `NBeatsNet` model and all its custom layers (`RevIN`, `TrendBlock`, etc.) are fully serializable using Keras 3's modern `.keras` format.

```python
# Create and train the model
model = create_nbeats_model(...)
# ... model.fit(...)

# Save the entire model to a single file
model.save('my_nbeats_model.keras')

# Load the model in a new session. All custom layers are automatically handled.
loaded_model = keras.models.load_model('my_nbeats_model.keras')
print("✅ N-BEATS model loaded successfully!")
```

---

## 13. Testing & Validation

### Unit Test Example

A simple test to verify the forward pass and shape correctness for a multivariate model.

```python
def test_multivariate_forward_pass():
    """Test the output shape for a multivariate model."""
    model = NBeatsNet(
        backcast_length=50,
        forecast_length=20,
        input_dim=5,
        output_dim=3
    )
    dummy_input = np.random.randn(4, 50, 5).astype("float32")
    output = model.predict(dummy_input)
    assert output.shape == (4, 20, 3)
    print("✓ Multivariate forward pass has correct shape")

test_multivariate_forward_pass()
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Training is unstable and the loss becomes `NaN`.**

-   **Cause**: This is common in N-BEATS. It's usually due to a learning rate that is too high or exploding gradients.
-   **Solution**:
    1.  Ensure you are using **gradient clipping**. The factory function sets `clipnorm=1.0` by default. You can try a smaller value like `0.5`.
    2.  Use a smaller learning rate (e.g., `1e-5` or `5e-5`).
    3.  Make sure `use_revin=True`, as it significantly improves stability.

### Frequently Asked Questions

**Q: Why is RevIN so important?**

A: Real-world time series often have their mean and variance change over time. RevIN makes the model's job easier by ensuring it always sees data with a consistent distribution (zero mean, unit variance). This decoupling of pattern learning from scale learning makes the model much more robust and accurate.

**Q: How do I choose the `backcast_length`?**

A: It should be long enough to capture at least one full cycle of the longest meaningful seasonality in your data. A common heuristic is to set it to be **3 to 7 times** your `forecast_length`. For example, to predict the next 24 hours, using the past 4-5 days (`96-120` hours) is a good starting point.

**Q: How does this multivariate implementation work?**

A: The input `(batch, backcast_length, input_dim)` is flattened into `(batch, backcast_length * input_dim)` before being processed by the blocks. This allows the model to learn cross-variable relationships. The forecast is first generated in the `input_dim` space to allow for correct RevIN denormalization, and then a final dense layer projects it to the desired `output_dim`.

---

## 15. Technical Details

### Doubly Residual Stacking

The core of N-BEATS is its hierarchical decomposition. Consider two stacks:

1.  **Input**: `x`
2.  **Stack 1 (Trend)**:
    -   Processes `x` and produces `backcast_1`, `forecast_1`.
    -   The residual passed to the next stack is `residual_1 = x - backcast_1`.
3.  **Stack 2 (Seasonality)**:
    -   Processes `residual_1` and produces `backcast_2`, `forecast_2`.
4.  **Final Forecast**:
    -   `forecast = forecast_1 + forecast_2`

This process allows the trend stack to model and remove the trend component, so the seasonality stack can focus on modeling periodicity in the de-trended signal. This is a powerful form of curriculum learning.

### Basis Functions

-   **Trend (Polynomial)**:
    `y = Σ_{i=0}^{p} θ_i * t^i`
    where `t` is a normalized time vector and `p` is the polynomial degree.
-   **Seasonality (Fourier)**:
    `y = Σ_{i=1}^{n} [ θ_{i,1}*cos(2πit) + θ_{i,2}*sin(2πit) ]`
    where `n` is the number of harmonics.

---

## 16. Citation

This implementation is based on the original N-BEATS and RevIN papers. If you use this model in your research, please cite the original works:

-   **N-BEATS**:
    ```bibtex
    @inproceedings{oreshkin2020n,
      title={N-BEATS: Neural basis expansion analysis for interpretable time series forecasting},
      author={Oreshkin, Boris N and Carpov, Dmitri and Chapados, Nicolas and Bengio, Yoshua},
      booktitle={International Conference on Learning Representations},
      year={2020}
    }
    ```
-   **RevIN**:
    ```bibtex
    @inproceedings{kim2022reversible,
      title={Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift},
      author={Kim, Taesung and Kim, Jinhee and Tae, Yungi and Park, Cheonbok and Choi, Jang-Ho and Choo, Jaegul},
      booktitle={International Conference on Learning Representations},
      year={2022}
    }
    ```