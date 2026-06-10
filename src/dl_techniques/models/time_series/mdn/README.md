# Mixture Density Network (MDN)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of a **Mixture Density Network (MDN)** in **Keras 3**, based on the foundational paper ["Mixture Density Networks"](http://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) by Christopher Bishop (1994).

The `MDNLayer` includes several practical improvements for training stability, such as independent processing paths, diversity regularization, and sigma constraints.

---

## Table of Contents

1. [Overview: What is MDN and Why It Matters](#1-overview-what-is-mdn-and-why-it-matters)
2. [The Problem MDN Solves](#2-the-problem-mdn-solves)
3. [How MDN Works: Core Concepts](#3-how-mdn-works-core-concepts)
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

## 1. Overview: What is MDN and Why It Matters

### What is an MDN?

A **Mixture Density Network (MDN)** is a neural network that learns to predict a full probability distribution for its output, rather than just a single point estimate. It achieves this by modeling the output as a **Mixture of Gaussian Distributions**.

For each input, an MDN predicts the parameters of this mixture:
- **Means (μ)**: The center of each Gaussian component.
- **Standard Deviations (σ)**: The spread or uncertainty of each component.
- **Mixing Coefficients (π)**: The weight or importance of each component.

### Key Innovations

1.  **Probabilistic Regression**: Goes beyond standard regression by providing a rich, probabilistic description of the target variable.
2.  **Uncertainty Quantification**: Naturally models and separates different types of uncertainty in the data and the model.
3.  **Multi-modality**: Can effectively model problems where a single input can map to multiple valid outputs (e.g., inverse problems).
4.  **Heteroscedasticity**: Can capture complex noise patterns where the level of uncertainty changes depending on the input.

### Why MDNs Matter

**Traditional Regression Problems**:
```
Problem: Predict house price based on square footage.
Traditional Solution:
  1. Train a model (e.g., Linear Regression, DNN).
  2. For a given size, predict a single price: 1500 sqft → $300,000.
  3. Limitation: The model provides no confidence estimate. Is it $300k ± $10k or ± $100k?
  4. Limitation: Cannot handle cases where similar houses sell for wildly different prices.
```

**MDN's Solution**:
```
MDN Approach:
  1. Train an MDN model.
  2. For a given size, predict a probability distribution.
  3. Output: 1500 sqft → A mixture of Gaussians, e.g.,
     - 70% chance of being around $290k (σ = $20k)
     - 30% chance of being around $350k (σ = $30k)
  4. Benefit: Captures uncertainty and potential multi-modality.
```

### Real-World Impact

MDNs address critical challenges where uncertainty is key:

- **📉 Finance**: Model asset prices, capturing market volatility and risk.
- **🤖 Robotics**: Predict possible future states for a robot's arm, enabling safer motion planning.
- **🔬 Scientific Modeling**: Model experimental outcomes that have inherent randomness.
- **🌦️ Time Series Forecasting**: Generate prediction intervals for future values, not just single forecasts.
- **🎮 Reinforcement Learning**: Model complex reward distributions in an environment.

---

## 2. The Problem MDN Solves

### The Limitations of Standard Regression

Traditional regression models, trained with Mean Squared Error (MSE), make a critical, often incorrect, assumption:

```
┌─────────────────────────────────────────────────────────────┐
│  Standard Regression Model (MSE Loss)                       │
│                                                             │
│  Implicit Assumption:                                       │
│    The data follows a Gaussian distribution with constant   │
│    variance (homoscedastic noise) around a single mean.     │
│                                                             │
│  This fails when:                                           │
│  1. Data is Multi-modal → One input has multiple plausible  │
│     outputs.                                                │
│  2. Noise is Heteroscedastic → Uncertainty changes with     │
│     the input.                                              │
│  3. We need to quantify confidence in predictions.          │
└─────────────────────────────────────────────────────────────┘
```

**Example: The Inverse Sine Wave Problem**
-   **Forward Problem**: `y = sin(x)`. For each `x`, there is one `y`. Easy.
-   **Inverse Problem**: `x = f(y)`. For a given `y`, there can be infinite `x` values. This is a multi-modal problem. A standard model trained on this would average all possible `x` values, producing a useless prediction.


*A standard model (left) fails on multi-modal data, while an MDN (right) correctly captures the multiple possible outputs.*

### How MDN Changes the Game

MDNs replace the restrictive assumption of a single Gaussian with a flexible, powerful alternative:

```
┌─────────────────────────────────────────────────────────────┐
│  MDN Workflow                                               │
│                                                             │
│  1. Predict parameters (μ, σ, π) for a mixture of Gaussians.│
│  2. Construct the full probability distribution P(y|x).     │
│  3. Maximize the likelihood of the training data under this │
│     distribution (Negative Log-Likelihood Loss).            │
│                                                             │
│  This allows the model to:                                  │
│  - Place Gaussian components at each mode of the data.      │
│  - Adjust the width (σ) of each component to match local    │
│    noise.                                                   │
│  - Adjust the weight (π) of each component based on its     │
│    prevalence.                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How MDN Works: Core Concepts

### The Three-Component Architecture

An MDN model is typically composed of two parts:

```
┌──────────────────────────────────────────────────────────────────┐
│                          MDN Architecture                        │
│                                                                  │
│  ┌─────────────────┐      ┌───────────────────────────────────┐  │
│  │ Feature Extrctr │      │           MDN Layer               │  │
│  │  (Standard NN)  │───►  │  (Splits into parallel heads)     │  │
│  │                 │      │                                   │  │
│  │ Learns a hidden │      │ ┌───────────────────────────────┐ │  │
│  │ representation  │      │ │      μ Head → Means (μ)       │ │  │
│  │    of input x   │      ├─┤      σ Head → Std Devs (σ)    │ │  │
│  │                 │      │ │      π Head → Weights (π)     │ │  │
│  │                 │      │ └───────────────────────────────┘ │  │
│  └─────────────────┘      └───────────────────────────────────┘  │
│                                       │                          │
│                                       ▼                          │
│                               ┌────────────────┐                 │
│                               │   Predicted    │                 │
│                               │  Distribution  │                 │
│                               │ P(y|x) = Σ π*N(μ,σ)              │
│                               └────────────────┘                 │
└──────────────────────────────────────────────────────────────────┘
```

### The Predicted Distribution

The final output is not a single value but a probability distribution defined by the equation:

**P(y | x) = Σᵢ πᵢ(x) * N(y | μᵢ(x), σᵢ(x))**

Where:
-   `πᵢ(x)`: The **mixing coefficient** for component `i`. This is the probability of selecting the i-th Gaussian. All `π`s sum to 1.
-   `N(y | μᵢ(x), σᵢ(x))`: A **Normal (Gaussian) distribution** with mean `μᵢ(x)` and standard deviation `σᵢ(x)`.

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MDN Complete Data Flow                              │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: Feature Extraction
───────────────────────────
Input (B, D_in)
    │
    ├─► Dense Layer 1
    ├─► [BatchNorm] -> Activation -> [Dropout]
    ├─► ...
    └─► Dense Layer N
            └─► Output: (B, D_hidden) ← HIDDEN FEATURES


STEP 2: MDN Parameter Prediction (within MDNLayer)
────────────────────────────────────────────────────
Hidden Features (B, D_hidden)
    │
    ├──► μ Path ──────────────► Dense → [BN] → Act → Dense(μ)
    │       └─► Means (μ): (B, num_mix * D_out)
    │
    ├──► σ Path ──────────────► Dense → [BN] → Act → Dense(σ) + Softplus
    │       └─► Std Devs (σ): (B, num_mix * D_out), guaranteed > 0
    │
    └──► π Path ──────────────► Dense → [BN] → Act → Dense(π)
            └─► Logits (π): (B, num_mix)


STEP 3: Output Concatenation
──────────────────────────────
    │
    ├─► Concatenate [μ, σ, π]
    │       └─► Final Output Vector (B, total_params)


STEP 4: Loss Calculation (During Training)
──────────────────────────────────────────
y_true & y_pred
    │
    ├─► Split y_pred back into μ, σ, π
    │
    ├─► Apply softmax to π logits to get mixture weights
    │
    ├─► For each data point y_true:
    │   Calculate probability density from each Gaussian: N(y_true | μᵢ, σᵢ)
    │
    ├─► Compute weighted sum of probabilities: Σᵢ πᵢ * N(y_true | μᵢ, σᵢ)
    │
    ├─► Calculate Negative Log-Likelihood: -log(Σᵢ πᵢ * N(...))
    │
    └─► Average over batch to get final loss


STEP 5: Sampling (During Inference)
───────────────────────────────────
y_pred
    │
    ├─► Split y_pred into μ, σ, π
    │
    ├─► Apply softmax to π logits
    │
    ├─► For each input in batch:
    │   1. Choose a component `i` by sampling from a Categorical(π) dist.
    │   2. Draw a sample from the chosen Gaussian: N(μᵢ, σᵢ)
    │
    └─► Return samples
```

---

## 4. Architecture Deep Dive

### 4.1 Feature Extraction Network (`MDNModel`)

This is a standard feed-forward network that learns a meaningful representation of the input data.

```
┌───────────────────────────────────────────────────┐
│              Feature Extractor                    │
└───────────────────────────────────────────────────┘
Input: (B, D_in)
  │
  ▼
┌──────────────────────────────────┐   ┐
│   Dense Layer (units, activation)│   │
│   + [Optional] Batch Norm        │   │
│   + [Optional] Dropout           │   │  Repeated for each
└──────────────────────────────────┘   │  hidden layer
  │                                    │
  ▼                                    │
┌──────────────────────────────────┐   │
│   ...                            │   │
└──────────────────────────────────┘   ┘
  │
  ▼
Output: (B, D_hidden) ← Feature Vector for MDN Layer
```

### 4.2 MDN Output Layer (`MDNLayer`)

This is the core of the MDN. This implementation uses a more advanced design with separate processing paths for each parameter type.

#### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                       MDNLayer (Internal)                           │
└─────────────────────────────────────────────────────────────────────┘
Input: (B, D_hidden)
  │
  ├───────────┬───────────┬───────────┐
  │           │           │           │
  ▼           ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ μ Path  │ │ σ Path  │ │ π Path  │
└─────────┘ └─────────┘ └─────────┘
  │           │           │
  ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Dense   │ │ Dense   │ │ Dense   │
│ (inter) │ │ (inter) │ │ (inter) │
└─────────┘ └─────────┘ └─────────┘
  │           │           │
  ▼           ▼           ▼
[BatchNorm] [BatchNorm] [BatchNorm]
  │           │           │
  ▼           ▼           ▼
Activation  Activation  Activation
  │           │           │
  ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│Dense(μ) │ │Dense(σ) │ │Dense(π) │
└─────────┘ └─────────┘ └─────────┘
  │           │           │
  │           │           │
  │           ▼           │
  │      Softplus + min   │
  │      (ensure σ > 0)   │
  │           │           │
  └───────────┼───────────┘
              │
              ▼
┌──────────────────────────────────┐
│ Concatenate [μ, σ, π]            │
└──────────────────────────────────┘
  │
  ▼
Output: (B, total_params)
```

#### Key Design Decisions Explained

**Q: Why separate processing paths for μ, σ, and π?**

A: The parameters represent different aspects of the distribution.
-   **μ (mean)** relates to the *location* of the prediction.
-   **σ (std dev)** relates to the *uncertainty* or spread.
-   **π (weight)** relates to the *importance* or probability of a mode.
Separate paths allow the network to learn specialized features for each task, leading to better stability and performance.

**Q: Why use `softplus` for σ?**

A: Standard deviations (σ) must be strictly positive. The `softplus` function, `log(1 + exp(x))`, maps any real number to a positive one. We also add a small minimum value (`min_sigma`) to prevent numerical instability from σ values approaching zero.

**Q: Why is there no activation on the final π head?**

A: The raw outputs of the π head are *logits*. These are converted into probabilities using the `softmax` function inside the loss function and sampling methods. This is numerically more stable than applying softmax directly in the layer.

**Q: What is Diversity Regularization?**

A: A common failure mode in MDNs is "component collapse," where multiple Gaussian components converge to the same mean. The diversity regularizer adds a small penalty to the loss if component means get too close to each other, encouraging them to spread out and capture different modes in the data.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy scipy
```

### Your First Probabilistic Model (30 seconds)

Let's solve a simple multi-modal problem: predicting `x` from `y = x * sin(x)`.

```python
import keras
import numpy as np
import matplotlib.pyplot as plt

# Local imports from your project structure
from dl_techniques.models.statistics.mdn import MDNModel

# 1. Generate multi-modal training data
X_train = np.random.uniform(-10, 10, 2000)
y_train = X_train * np.sin(X_train) + np.random.normal(0, 0.5, 2000)
# Swap X and y to create the inverse (multi-modal) problem
X_train, y_train = y_train.reshape(-1, 1), X_train.reshape(-1, 1)

# 2. Create an MDN model
model = MDNModel(
    hidden_layers=[32, 32],      # Feature extractor layers
    output_dimension=1,          # We are predicting a single value (x)
    num_mixtures=5,              # Use 5 Gaussians to model the distribution
)

# 3. Compile the model (uses the MDN loss automatically)
model.compile(optimizer='adam')
print("Model created and compiled successfully!")

# 4. Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
print("✅ Training Complete!")

# 5. Generate samples from the predicted distribution
X_test = np.linspace(-10, 10, 500).reshape(-1, 1)
samples = model.sample(X_test, num_samples=100) # (500, 100, 1)

# 6. Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_train, X_train, alpha=0.1, label='Training Data (x, y swapped)')
plt.scatter(X_test, samples[:, :, 0], color='r', alpha=0.05, label='MDN Samples')
plt.title('MDN learning a multi-modal distribution')
plt.xlabel('Input (y = x*sin(x))')
plt.ylabel('Output (x)')
plt.legend()
plt.show()
```

---

## 6. Component Reference

### 6.1 `MDNModel`

**Purpose**: An end-to-end Keras model combining a feature extractor with an `MDNLayer`.

**Location**: `dl_techniques.models.statistics.mdn.MDNModel`

```python
from dl_techniques.models.statistics.mdn import MDNModel

model = MDNModel(
    hidden_layers=[64, 32],
    output_dimension=2,
    num_mixtures=5,
    hidden_activation='relu',
    use_batch_norm=True,
    dropout_rate=0.1,
    kernel_regularizer=keras.regularizers.L2(1e-4)
)
```

**Key Parameters**:

| Parameter          | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `hidden_layers`    | List of integers defining the feature extractor size. |
| `output_dimension` | The dimensionality of the target variable `y`.        |
| `num_mixtures`     | The number of Gaussian components to use.             |
| `hidden_activation`| Activation for hidden layers (e.g., 'relu', 'tanh').  |
| `use_batch_norm`   | If True, adds Batch Normalization layers.             |
| `dropout_rate`     | Dropout rate for regularization.                      |

**Key Methods**:
-   `compile()`: Automatically uses the MDN negative log-likelihood loss.
-   `fit()`: Trains the model.
-   `sample(inputs, num_samples)`: Generates samples from the predicted distribution.
-   `predict_with_uncertainty(inputs)`: Provides point estimates and uncertainty decomposition.

### 6.2 `MDNLayer`

**Purpose**: The specialized output layer that predicts mixture parameters. Can be used in any Keras model.

**Location**: `dl_techniques.layers.statistics.mdn_layer.MDNLayer`

```python
from dl_techniques.layers.statistics.mdn_layer import MDNLayer

inputs = keras.Input(shape=(128,))
x = keras.layers.Dense(64, activation='relu')(inputs)
mdn_params = MDNLayer(
    output_dimension=1,
    num_mixtures=3,
    intermediate_units=32,
    diversity_regularizer_strength=0.01
)(x)

model = keras.Model(inputs, mdn_params)
```

**Key Parameters**:

| Parameter                          | Description                                                                   |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| `output_dimension`                 | The dimensionality of the target variable `y`.                                |
| `num_mixtures`                     | The number of Gaussian components.                                            |
| `intermediate_units`               | Units in the hidden layers within each parameter path (μ, σ, π).              |
| `diversity_regularizer_strength`   | Strength of the penalty for component collapse. Set > 0 to enable.            |
| `min_sigma`                        | A small positive value to clip standard deviations, preventing instability.   |
| `use_batch_norm`                   | Whether to use batch norm within the parameter paths.                         |

**Key Attributes**:
-   `loss_func`: The negative log-likelihood loss function. Can be passed to `model.compile()`.
-   `sample()`: The core sampling logic.
-   `split_mixture_params()`: A helper to split the concatenated output vector into μ, σ, and π tensors.

---

## 7. Configuration & Model Variants

### Choosing the Number of Mixtures

The `num_mixtures` is the most important hyperparameter.

| Num Mixtures | Use Case                                               | Risk                                        |
| :----------: | ------------------------------------------------------ | ------------------------------------------- |
|     **1**    | Standard regression with heteroscedastic uncertainty.  | Cannot model multi-modality.                |
|   **3 - 5**  | Good starting point for most problems.                 | Might be insufficient for very complex data. |
|  **5 - 10**  | Problems with complex, multi-modal distributions.      | Slower training, risk of overfitting.       |
|   **10+**    | Highly complex or high-dimensional output spaces.      | Prone to component collapse; needs regularization. |

**Guideline**: Start with a small number (e.g., 3 or 5) and increase if the model fails to capture all modes of the data. Use the `diversity_regularizer` for higher numbers of mixtures.

### Feature Extractor Depth and Width

-   **Shallow/Wide (`hidden_layers=[512]`)**: Good for problems where features don't require deep hierarchical processing.
-   **Deep/Narrow (`hidden_layers=[64, 64, 64]`)**: Better for problems with complex, structured input data where a hierarchy of features is beneficial.
-   **Regularization**: For deep or complex models, use `dropout_rate` and `kernel_regularizer` to prevent overfitting. `use_batch_norm=True` can improve training stability.

---

## 8. Comprehensive Usage Examples

### Example 1: Sampling from the Distribution

Sampling allows you to see the range of plausible predictions.

```python
# Assuming 'model' is trained and 'X_test' is available
num_samples = 200
samples = model.sample(X_test, num_samples=num_samples) # Shape: (B, 200, D_out)

# For a 1D output, we can visualize the samples
plt.figure(figsize=(10, 6))
# Plot each sample with low opacity
for i in range(num_samples):
    plt.scatter(X_test, samples[:, i, 0], color='red', alpha=0.02, s=10)
plt.scatter(X_train, y_train, alpha=0.1, label='Training Data') # Plot original data
plt.title(f'{num_samples} Samples from the Predicted Distribution')
plt.xlabel('Input')
plt.ylabel('Predicted Output Samples')
plt.show()
```

### Example 2: Calculating Point Estimates

While MDNs predict distributions, you can still get a single "best guess" prediction by taking the weighted average of the means.

```python
from dl_techniques.layers.statistics.mdn_layer import get_point_estimate

# Calculate the expected value E[y|x] = Σᵢ πᵢ(x) * μᵢ(x)
point_estimates = get_point_estimate(
    model=model,
    x_data=X_test,
    mdn_layer=model.mdn_layer
)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.1, label='Training Data')
plt.plot(X_test, point_estimates, color='black', lw=2, label='Point Estimate (Weighted Mean)')
plt.title('MDN Point Estimate')
plt.legend()
plt.show()
```

### Example 3: Decomposing and Visualizing Uncertainty

MDNs can separate uncertainty into two types:
-   **Aleatoric Uncertainty**: Inherent noise or randomness in the data. Cannot be reduced with more data.
-   **Epistemic Uncertainty**: Model uncertainty due to lack of training data in a certain region. Can be reduced with more data.

```python
from dl_techniques.layers.statistics.mdn_layer import get_uncertainty

# Get point estimates first
point_estimates = get_point_estimate(model, X_test, model.mdn_layer)

# Decompose uncertainty
total_variance, aleatoric_variance = get_uncertainty(
    model=model,
    x_data=X_test,
    mdn_layer=model.mdn_layer,
    point_estimates=point_estimates
)
epistemic_variance = total_variance - aleatoric_variance

# Convert to standard deviation for plotting
total_std = np.sqrt(total_variance)
aleatoric_std = np.sqrt(aleatoric_variance)
epistemic_std = np.sqrt(epistemic_variance)

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
ax1.set_title('Uncertainty Decomposition')
ax1.plot(X_test, point_estimates, color='black', label='Point Estimate')
ax1.fill_between(
    X_test.flatten(),
    (point_estimates - total_std).flatten(),
    (point_estimates + total_std).flatten(),
    alpha=0.3, color='gray', label='Total Uncertainty (±1σ)'
)
ax1.scatter(X_train, y_train, alpha=0.1)
ax1.legend()

ax2.plot(X_test, aleatoric_std, label='Aleatoric Std Dev (Data Noise)')
ax2.plot(X_test, epistemic_std, label='Epistemic Std Dev (Model Uncertainty)')
ax2.set_title('Uncertainty Components')
ax2.set_xlabel('Input')
ax2.legend()
plt.show()
```

### Example 4: Generating Prediction Intervals

Provide a rigorous confidence bound for your predictions.

```python
from dl_techniques.layers.statistics.mdn_layer import get_prediction_intervals

# We already have point_estimates and total_variance
lower_bound, upper_bound = get_prediction_intervals(
    point_estimates=point_estimates,
    total_variance=total_variance,
    confidence_level=0.95  # 95% confidence
)

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(X_test, point_estimates, color='black', label='Point Estimate')
plt.fill_between(
    X_test.flatten(),
    lower_bound.flatten(),
    upper_bound.flatten(),
    alpha=0.3, color='orange', label='95% Prediction Interval'
)
plt.scatter(X_train, y_train, alpha=0.1)
plt.title('MDN with 95% Prediction Intervals')
plt.legend()
plt.show()
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Diagnosing Component Collapse

"Component collapse" is when multiple mixture components model the same thing, wasting capacity. You can diagnose this with the `check_component_diversity` utility.

```python
from dl_techniques.layers.statistics.mdn_layer import check_component_diversity

# Train a model WITHOUT diversity regularization
model_no_reg = MDNModel(output_dimension=1, num_mixtures=10, hidden_layers=[32, 32])
# model_no_reg.mdn_layer.diversity_regularizer_strength = 0.0 # This is default
model_no_reg.compile(optimizer='adam')
model_no_reg.fit(X_train, y_train, epochs=50, verbose=0)

# Train a model WITH diversity regularization
model_with_reg = MDNModel(output_dimension=1, num_mixtures=10, hidden_layers=[32, 32])
model_with_reg.mdn_layer.diversity_regularizer_strength = 0.01 # Enable regularization
model_with_reg.compile(optimizer='adam')
model_with_reg.fit(X_train, y_train, epochs=50, verbose=0)

# Analyze diversity
diversity_no_reg = check_component_diversity(model_no_reg, X_test, model_no_reg.mdn_layer)
diversity_with_reg = check_component_diversity(model_with_reg, X_test, model_with_reg.mdn_layer)

print("--- Without Regularization ---")
print(f"Mean component separation: {diversity_no_reg['mean_component_separation']:.3f}")
print(f"Mean mixture weights: {np.round(diversity_no_reg['mean_mixture_weights'], 2)}")

print("\n--- With Regularization ---")
print(f"Mean component separation: {diversity_with_reg['mean_component_separation']:.3f}")
print(f"Mean mixture weights: {np.round(diversity_with_reg['mean_mixture_weights'], 2)}")

# Expected output: The regularized model will have a larger 'mean_component_separation'.
```

### Pattern 2: Conditional Mode Sampling

After getting the parameters, you can choose to sample only from the most likely component, or the one with the lowest variance, etc.

```python
# Predict parameters
params = model.predict(X_test)
mu, sigma, pi_logits = model.mdn_layer.split_mixture_params(params)
pi = keras.activations.softmax(pi_logits, axis=-1)

# Find the most likely component for each prediction
most_likely_component_idx = keras.ops.argmax(pi, axis=-1)

# Select the mean and sigma of only the most likely component
selected_mu = keras.ops.take_along_axis(mu, most_likely_component_idx[:, None, None], axis=1)
selected_sigma = keras.ops.take_along_axis(sigma, most_likely_component_idx[:, None, None], axis=1)

# Sample from this component
epsilon = keras.random.normal(ops.shape(selected_mu))
most_likely_samples = selected_mu + selected_sigma * epsilon

# This gives a less diverse but more "confident" set of samples.
```

---

## 10. Performance Optimization

### Mixed Precision Training

Use mixed precision for faster training and inference, especially on compatible GPUs.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = MDNModel(output_dimension=1, num_mixtures=5, hidden_layers=[64, 64])
model.compile(optimizer='adam')

# For training, this provides:
# - ~1.5-2x speedup on compatible GPUs
# - ~50% memory reduction
```

### TensorFlow XLA Compilation

Use XLA (Accelerated Linear Algebra) for additional speedup by compiling the computation graph.

```python
import tensorflow as tf

@tf.function(jit_compile=True)
def compiled_sample(model, inputs, num_samples):
    return model.sample(inputs, num_samples=num_samples)

# Usage
model = MDNModel(...) # create your model
# ... train it ...
samples = compiled_sample(model, X_test, num_samples=100)

# Expected speedup: 10-30% depending on hardware.
```

---

## 11. Training and Best Practices

### Monitoring the Loss

-   The MDN loss is a negative log-likelihood, so values are typically positive. A lower value is better.
-   The loss can be volatile initially, especially with multi-modal data, as components "decide" which mode to cover.
-   If the loss becomes `NaN`, it's almost always due to `σ` becoming too small. The `min_sigma` parameter in `MDNLayer` is designed to prevent this, but an overly aggressive learning rate can still cause issues.

### Choosing Hyperparameters

1.  **Start Simple**: Begin with a small number of mixtures (e.g., 3) and a simple feature extractor (e.g., `hidden_layers=[32]`).
2.  **Analyze Samples**: After initial training, generate samples and visualize them. Are there modes in your data that the model missed? If so, increase `num_mixtures`.
3.  **Check Diversity**: If you use a high `num_mixtures`, use `check_component_diversity` to see if components are collapsing. If they are, add or increase `diversity_regularizer_strength`.
4.  **Tune Learning Rate**: A learning rate scheduler (e.g., `ReduceLROnPlateau`) can be very helpful. Start with a standard rate like `1e-3` and decrease it if the loss plateaus.

---

## 12. Serialization & Deployment

The `MDNModel` and `MDNLayer` are fully serializable using Keras 3's modern saving mechanisms.

### Saving and Loading

```python
# Create and train model
model = MDNModel(output_dimension=1, num_mixtures=5, hidden_layers=[32])
model.compile(optimizer='adam')
model.fit(X_train, y_train, epochs=10)

# Save entire model
model.save('mdn_model.keras')
print("Model saved to mdn_model.keras")

# Load model in a new session
loaded_model = keras.models.load_model('mdn_model.keras')
print("Model loaded successfully")

# Verify
samples = loaded_model.sample(X_test, num_samples=1)
print(f"Generated samples with loaded model: {samples.shape}")
```

### Deployment

Because the model is a standard Keras `Model`, it can be deployed using any framework that supports Keras models, such as TensorFlow Serving, Flask/FastAPI, or converted to TFLite for on-device inference. The key is to remember that the model output is not a prediction, but parameters that you then use to sample or calculate statistics.

---

## 13. Testing & Validation

### Unit Tests

```python
import keras
import numpy as np

def test_model_creation():
    """Test that the model can be created."""
    model = MDNModel(output_dimension=1, num_mixtures=3, hidden_layers=[16])
    assert model is not None
    print("✓ Model created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = MDNModel(output_dimension=2, num_mixtures=5, hidden_layers=[16])
    dummy_input = keras.random.normal(shape=(10, 4))
    output = model(dummy_input)
    
    # Expected: (2 * D_out * N_mix) for μ,σ + N_mix for π
    # (2 * 2 * 5) + 5 = 25
    expected_shape = (10, 25)
    assert output.shape == expected_shape
    print("✓ Forward pass has correct shape")

def test_serialization():
    """Test model save/load."""
    model = MDNModel(output_dimension=1, num_mixtures=3, hidden_layers=[16])
    model.build(input_shape=(None, 10))
    
    # Save and load
    model.save('test_mdn_model.keras')
    loaded_model = keras.models.load_model('test_mdn_model.keras')
    
    # Check config
    assert model.get_config()['num_mixtures'] == loaded_model.get_config()['num_mixtures']
    print("✓ Serialization successful")

def test_sample_shape():
    """Test the output shape of the sample method."""
    model = MDNModel(output_dimension=2, num_mixtures=5, hidden_layers=[16])
    dummy_input = keras.random.normal(shape=(10, 4))
    samples = model.sample(dummy_input, num_samples=20)
    
    expected_shape = (10, 20, 2) # (batch, num_samples, output_dim)
    assert samples.shape == expected_shape
    print("✓ Sampling has correct shape")

# Run tests
if __name__ == '__main__':
    test_model_creation()
    test_forward_pass_shape()
    test_serialization()
    test_sample_shape()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Loss is NaN or Infinity**

-   **Cause**: Numerical instability, usually from a standard deviation (`σ`) becoming zero or negative.
-   **Solution 1**: Lower the learning rate.
-   **Solution 2**: Increase `min_sigma` in the `MDNLayer` (e.g., from `1e-3` to `1e-2`).
-   **Solution 3**: Use gradient clipping in your optimizer: `optimizer=keras.optimizers.Adam(clipnorm=1.0)`.

**Issue 2: The model only learns one mode of the data.**

-   **Cause 1**: Not enough training. MDNs can take longer to converge than standard models. Train for more epochs.
-   **Cause 2**: Insufficient model capacity. Try adding more units to `hidden_layers` or increasing `num_mixtures`.
-   **Cause 3**: Poor initialization. Sometimes, restarting the training can lead to a better solution.

**Issue 3: All mixture components are the same (component collapse).**

-   **Cause**: The model finds it easier to use multiple components for the same mode rather than exploring.
-   **Solution**: Enable the diversity regularizer by setting `diversity_regularizer_strength` to a small positive value (e.g., `0.01`) in the `MDNLayer`.

### Frequently Asked Questions

**Q: Can I use an MDN for classification?**

A: Not directly. MDNs are designed for continuous regression tasks. For probabilistic classification, you should use a standard network with a `softmax` output, which already models a probability distribution (a Categorical distribution).

**Q: How does this compare to Quantile Regression or Dropout-based uncertainty?**

A:
-   **Quantile Regression**: Predicts specific quantiles of the distribution. It's less flexible than an MDN, which models the entire distribution.
-   **MC Dropout**: Estimates *epistemic* uncertainty by running inference multiple times with dropout enabled. It's a great technique but doesn't explicitly model *aleatoric* (data) uncertainty or multi-modality in the same way an MDN does. MDNs model both.

**Q: Can I use distributions other than Gaussian?**

A: Yes, in theory. The framework can be extended to other distributions (e.g., Laplace for more robustness to outliers, or a mixture of Gammas for positive-only data). However, this implementation is specialized for Gaussian mixtures, as they are the most common and versatile.

---

## 15. Technical Details

### The MDN Loss Function

The model is trained by minimizing the **Negative Log-Likelihood (NLL)** of the data. For a single data point `(x, y)`, the likelihood is the probability density of observing `y` given `x`, which is `P(y|x)`. The loss is:

**Loss = -log(P(y | x)) = -log( Σᵢ πᵢ(x) * N(y | μᵢ(x), σᵢ(x)) )**

The Gaussian probability density function `N` is:

**N(y | μ, σ) = (1 / (σ * sqrt(2π))) * exp(- (y - μ)² / (2 * σ²) )**

Minimizing NLL is equivalent to maximizing the probability that the model would have generated the training data.

### Uncertainty Decomposition Formulas

The `get_uncertainty` function uses the **Law of Total Variance**:

**Var[Y] = E[Var[Y|X]] + Var[E[Y|X]]**

In our context:
-   `Total Variance` = `Var[y|x]`
-   `Aleatoric Variance` = `E[Var[y|x,θ]] = Σᵢ πᵢ * σᵢ²`
    -   The expected value of the variance of each component.
-   `Epistemic Variance` = `Var[E[y|x,θ]] = Σᵢ πᵢ * (μᵢ - E[y|x])²`
    -   The variance of the means of each component around the total expected value.
    -   Where `E[y|x] = Σᵢ πᵢ * μᵢ` is the point estimate.

---

## 16. Citation

If you use MDNs in your research, please cite the original paper:

```bibtex
@techreport{bishop1994mixture,
  title={Mixture density networks},
  author={Bishop, Christopher M},
  year={1994},
  institution={Aston University}
}
```


