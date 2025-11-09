# PowerMLP: An Efficient Alternative to Kolmogorov-Arnold Networks

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of **PowerMLP**, a highly efficient deep learning architecture designed as a practical and powerful alternative to Kolmogorov-Arnold Networks (KANs). PowerMLP achieves a superior balance of performance, speed, and resource usage by replacing the computationally expensive B-spline activations of KANs with a novel dual-branch design powered by efficient `ReLU-k` activations.

The architecture is built from `PowerMLPLayer` blocks, offering a significant speedup (~40x faster training) and resource reduction (~10x fewer FLOPs) compared to equivalent KANs, while delivering equal or better accuracy.

---

## Table of Contents

1. [Overview: What is PowerMLP and Why It Matters](#1-overview-what-is-powermlp-and-why-it-matters)
2. [The Problem PowerMLP Solves](#2-the-problem-powermlp-solves)
3. [How PowerMLP Works: Core Concepts](#3-how-powermlp-works-core-concepts)
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

## 1. Overview: What is PowerMLP and Why It Matters

### What is PowerMLP?

**PowerMLP** is a novel neural network architecture that provides the expressive power needed to solve complex tasks without the extreme computational overhead of recent models like Kolmogorov-Arnold Networks (KANs). It achieves this through an innovative **dual-branch layer** that combines two complementary non-linear pathways: one for learning sharp, piecewise patterns and another for modeling smooth, global functions.

### Key Innovations

1.  **Efficient ReLU-k Activation**: Instead of computationally intensive learnable B-splines used in KANs, PowerMLP uses a simple yet powerful powered ReLU activation, `ReLU-k(x) = max(0, x)^k`, where `k` is a fixed integer hyperparameter. This provides higher-order non-linearity with negligible overhead.
2.  **Dual-Branch Architecture**: Each `PowerMLPLayer` splits the computation into two parallel branches that are summed at the end:
    *   A **Main Branch** (`Dense -> ReLU-k`) captures sharp, local features.
    *   A **Basis Branch** (`BasisFunction -> Dense`) models smooth, global patterns.
3.  **Extreme Performance Gains**: This design choice leads to dramatic improvements in efficiency, making it a practical choice for real-world applications. Compared to KAN, PowerMLP is:
    *   **~40x faster** to train.
    *   **~10x more efficient** in terms of FLOPs.
    *   **~5x lower** in memory usage.

### Why PowerMLP Matters

**The Kolmogorov-Arnold Network (KAN) Problem**:
```
Problem: Create a highly expressive model that can learn complex mathematical functions.
KAN's Approach:
  1. Replace the simple linear transformations in an MLP with learnable B-spline
     basis functions on every connection.
  2. Limitation: Evaluating and training splines is extremely slow and memory-intensive,
     making KANs impractical for large datasets or real-time applications.
  3. Result: KANs are a powerful theoretical tool but a major computational bottleneck.
```

**PowerMLP's Solution**:
```
PowerMLP's Approach:
  1. Approximate the expressiveness of splines with a much cheaper combination of
     two complementary pathways.
  2. The main branch (with ReLU-k) acts as an efficient piecewise polynomial
     approximator.
  3. The basis branch learns a global approximation using smooth functions.
  4. Benefit: Achieves KAN-like accuracy with the speed and efficiency of a
     standard MLP, making advanced function approximation practical.
```

### Real-World Impact

PowerMLP is designed for applications where both high accuracy and efficiency are critical:

-   📈 **Tabular Data**: An excellent, high-performance replacement for Gradient Boosting Machines and standard MLPs.
-   💨 **Real-Time Systems**: Fast enough for applications requiring low-latency inference.
-   🔬 **Scientific Computing**: A powerful tool for function regression and modeling physical systems where KANs would be too slow.
-   ☁️ **Efficient Cloud Deployment**: Drastically reduces training and inference costs for large-scale services.

---

## 2. The Problem PowerMLP Solves

### The Expressiveness vs. Efficiency Trade-off

In deep learning, there is a constant tension between a model's ability to learn complex functions (expressiveness) and its computational cost.

```
┌─────────────────────────────────────────────────────────────┐
│  The Dilemma of Modern Architectures                        │
│                                                             │
│  Standard MLPs:                                             │
│    - Extremely fast and efficient.                          │
│    - May require significant depth or width to approximate  │
│      very complex functions.                                │
│                                                             │
│  Kolmogorov-Arnold Networks (KANs):                         │
│    - Highly expressive, capable of learning intricate       │
│      mathematical functions with fewer parameters.          │
│    - Suffer from massive computational overhead due to      │
│      their reliance on B-spline basis functions, making     │
│      them impractical for many real-world tasks.            │
└─────────────────────────────────────────────────────────────┘
```

The challenge is to design a model that captures the enhanced expressiveness of KANs while retaining the practical efficiency of MLPs.

### How PowerMLP Changes the Game

PowerMLP provides a principled architecture that delivers a state-of-the-art trade-off between these two competing goals.

```
┌─────────────────────────────────────────────────────────────┐
│  The PowerMLP Hybrid Strategy                               │
│                                                             │
│  1. Decompose the Function:                                 │
│     - Instead of using one complex, slow tool (splines),    │
│       PowerMLP combines two simple, fast tools.             │
│                                                             │
│  2. Main Branch (Local & Sharp):                            │
│     - The `ReLU-k` activation acts like a piecewise         │
│       polynomial, perfect for modeling sharp transitions    │
│       and local non-linearities.                            │
│                                                             │
│  3. Basis Branch (Global & Smooth):                         │
│     - The basis function branch is designed to capture      │
│       smooth, global trends or periodic patterns.           │
│                                                             │
│  Result: The sum of these two branches can approximate a    │
│  much wider class of functions than a standard MLP, but at  │
│  a fraction of the computational cost of a KAN.             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How PowerMLP Works: Core Concepts

### The Dual-Branch Layer Architecture

The core of the model is the `PowerMLPLayer`, which processes input through two parallel pathways.

```
┌──────────────────────────────────────────────────────────────────┐
│                         PowerMLP Layer                           │
│                                                                  │
│                      Input (..., input_dim)                      │
│                           ╱         ╲                            │
│                          ╱           ╲                           │
│                         ╱             ╲                          │
│  ┌────────────────────┐                ┌───────────────────────┐ │
│  │    Main Branch     │                │     Basis Branch      │ │
│  │ Dense → ReLU-k(x)  │                │ BasisFunc(x) → Dense  │ │
│  └────────┬───────────┘                └───────────┬───────────┘ │
│           │                                       │              │
│           ╲                                     ╱                │
│            ╲           ┌───────────┐          ╱                  │
│             └─────────►│ Element-  │◄────────┘                   │
│                        │ wise Add  │                             │
│                        └───────────┘                             │
│                              │                                   │
│                    Output (..., output_dim)                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

A full PowerMLP model is a stack of these layers, with optional regularization.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       PowerMLP Complete Data Flow                       │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: INPUT
─────────────
Input Tensor (B, D_in) -- (e.g., normalized tabular data or flattened image)


STEP 2: HIDDEN LAYERS (Repeated for each hidden layer)
─────────────────────────────────────────────────────
Input to Layer `i`
    │
    ├─► PowerMLPLayer(units=D_i, k)
    │   (Computes main and basis branches and adds them)
    │
    ├─► [Optional] Batch Normalization
    │
    ├─► [Optional] Dropout
    │
    └─► Output of Layer `i`


STEP 3: OUTPUT LAYER
────────────────────
Final Hidden Representation
    │
    ├─► Standard Dense Layer(units=D_out, activation)
    │
    └─► Final Output (B, D_out)
```

---

## 4. Architecture Deep Dive

### 4.1 `PowerMLPLayer`

-   **Purpose**: The main building block, designed to be a highly expressive and efficient dense layer.
-   **Implementation**: It contains two distinct computational paths:
    1.  **Main Branch**: A standard `Dense` layer followed by a `ReLUK` activation. This branch learns sharp, piecewise non-linear features. It includes a bias term.
    2.  **Basis Branch**: A `BasisFunction` layer followed by a `Dense` layer. This branch is designed to capture smooth, global patterns. By design, its dense layer has **no bias**.
-   The outputs of these two branches are summed element-wise.

### 4.2 `ReLUK` Activation

-   **Purpose**: A simple and efficient way to introduce higher-order non-linearity.
-   **Functionality**: It computes `max(0, x) ** k`, where `k` is an integer hyperparameter (typically 2, 3, or 4).
    *   When `k=1`, it is the standard ReLU.
    *   When `k > 1`, it behaves like a polynomial of degree `k` for positive inputs, creating smoother and more powerful activation functions than ReLU.

### 4.3 `BasisFunction` Layer

-   **Purpose**: To project the input into a space of smooth, continuous functions, making it easier for the basis branch to model global trends.
-   **Functionality**: This is a simple, non-trainable layer that expands the input features. For an input `x`, it might produce an output like `[x, sin(x), cos(x)]`. This gives the basis branch a richer set of features to work with, inspired by classical function approximation techniques like Fourier series.

---

## 5. Quick Start Guide

### Installation

```bash
# Install Keras 3 and choose a backend (e.g., tensorflow)
pip install keras tensorflow

# For PyTorch or JAX backends:
# pip install keras torch
# pip install keras "jax[gpu]"

# For development and testing
pip install pytest numpy
```

### Your First PowerMLP Model (30 seconds)

Let's build a small PowerMLP for MNIST classification.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.power_mlp.model import PowerMLP

# 1. Create a tiny PowerMLP model for MNIST (28x28 images, 10 classes)
# We use a pre-configured variant for convenience.
model = PowerMLP.from_variant(
    "tiny",
    num_classes=10,
    input_dim=28*28, # MNIST images are 28x28 = 784 pixels
    output_activation="softmax"
)

# 2. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy", # Use sparse for integer labels
    metrics=["accuracy"],
)
print("✅ PowerMLP model created and compiled successfully!")
model.summary()

# 3. Create dummy data for a forward pass
batch_size = 16
# For real data, ensure it's flattened and normalized!
dummy_images = np.random.randn(batch_size, 28 * 28).astype("float32")
dummy_labels = np.random.randint(0, 10, batch_size)

# 4. Train for one step
loss, acc = model.train_on_batch(dummy_images, dummy_labels)
print(f"\n✅ Training step complete! Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# 5. Run inference
predictions = model.predict(dummy_images)
print(f"Predictions shape: {predictions.shape}") # (batch_size, num_classes)
```

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`PowerMLP`** | `...power_mlp.model.PowerMLP` | The main Keras `Model` class that assembles the architecture. |
| **`create_power_mlp`** | `...power_mlp.model.create_power_mlp` | Factory function to create and compile a standard model. |
| **`..._regressor`** | `...power_mlp.model.create_power_mlp_regressor` | Factory for regression tasks (MSE loss, linear output). |
| **`..._binary_classifier`**|`...power_mlp.model.create_power_mlp_binary_classifier`| Factory for binary classification (BCE loss, sigmoid output). |
| **`PowerMLPLayer`** | `...layers.ffn.power_mlp_layer.PowerMLPLayer` | The core dual-branch layer. |
| **`ReLUK`** | `...layers.activations.relu_k.ReLUK` | The powered ReLU activation function. |
| **`BasisFunction`** | `...layers.activations.basis_function.BasisFunction` | The smooth basis function activation. |

---

## 7. Configuration & Model Variants

This implementation provides several pre-configured variants for common use cases.

| Variant | Hidden Units | `k` | Description |
|:---:|:---|:---|:---|
| **`micro`** | `[32, 16]` | 2 | A minimal model for simple tasks or testing. |
| **`tiny`** | `[64, 32]` | 3 | A small model for basic classification/regression. |
| **`small`**| `[128, 64, 32]` | 3 | A good default for standard datasets like MNIST/CIFAR. |
| **`base`** | `[256, 128, 64]` | 3 | A powerful model for most tabular or image tasks. |
| **`large`**| `[512, 256, 128]`| 4 | A deep model for complex, large-scale problems. |
| **`xlarge`**|`[1024, 512, 256, 128]`| 4 | An extra-large model for research or very demanding tasks. |

---

## 8. Comprehensive Usage Examples

### Example 1: Image Classification on CIFAR-10

PowerMLP operates on flattened inputs, making it suitable as a powerful MLP-style classifier.

```python
# 1. Create the model using a variant
# CIFAR-10 images are 32x32x3 = 3072 features
model = PowerMLP.from_variant(
    "small",
    num_classes=10,
    input_dim=3072,
    output_activation="softmax",
    dropout_rate=0.2,
    batch_normalization=True
)
# 2. Compile the model
# AdamW is often preferred for its better weight decay implementation.
model.compile(optimizer="adamw", loss="categorical_crossentropy", metrics=["accuracy"])

# 3. Train on flattened image data
# (x_train shape should be [num_samples, 3072])
# (y_train should be one-hot encoded for categorical_crossentropy)
# model.fit(x_train_flat, y_train_one_hot, ...)
```

### Example 2: Regression Task

Use the dedicated factory function for regression, which correctly configures the loss and output activation.

```python
# The hidden_units list must include the output dimension (1 for univariate regression)
model = create_power_mlp_regressor(
    hidden_units=[256, 128, 64, 1],
    k=4,
    learning_rate=1e-4,
    batch_normalization=True
)
# The model is pre-compiled with 'mse' loss and a linear output layer.
model.summary()
```

### Example 3: Binary Classification

The binary classifier factory ensures the model has a single output unit with a sigmoid activation.

```python
model = create_power_mlp_binary_classifier(
    hidden_units=[128, 64, 1],
    k=3,
    dropout_rate=0.3
)
# The model is pre-compiled with 'binary_crossentropy' loss and a sigmoid output.
model.summary()
```

---

## 9. Advanced Usage Patterns

### Choosing the right `k`

The power `k` in the `ReLU-k` activation is a key hyperparameter that controls the model's non-linearity.

-   **`k=1`**: The main branch uses a standard ReLU. The model is still a dual-branch network but with less aggressive non-linearity. A safe starting point.
-   **`k=2, k=3`**: These are excellent default values, providing a strong balance of expressive power and training stability. The `base` variant uses `k=3`.
-   **`k >= 4`**: Higher values create very sharp, high-degree polynomial-like activations. This can be powerful for complex functions but may lead to large activation values and training instability. When using `k >= 4`, it is **strongly recommended to enable `batch_normalization`**.

---

## 10. Performance Optimization

### Mixed Precision Training

PowerMLP is fully compatible with mixed precision training because it is composed of standard Keras layers (`Dense`) and simple element-wise operations. This can provide a significant speedup on modern GPUs with minimal code changes.

```python
# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# Create a model, which will now automatically use mixed precision
model = PowerMLP.from_variant("large", ...)
```

---

## 11. Training and Best Practices

-   **Start with a `base` Variant**: For most problems, starting with the `base` variant and tuning from there is an effective strategy.
-   **Normalize Your Inputs**: This is critical for any MLP, but especially for PowerMLP. The `x^k` operation can cause outputs to explode if inputs are not scaled to a small range (e.g., via `StandardScaler` to have zero mean and unit variance).
-   **Use `batch_normalization` for `k >= 3`**: Batch norm helps stabilize training by controlling the scale of activations, which is essential when using higher-power non-linearities.
-   **Optimizer**: **AdamW** is a strong default choice due to its effective weight decay implementation.
-   **Learning Rate Schedule**: A **cosine decay** schedule, optionally with a few epochs of linear warmup, generally works best for stable convergence.
-   **Gradient Clipping**: For very high `k` values (`k >= 5`) or deep networks, consider adding `clipnorm` or `clipvalue` to your optimizer as an extra safeguard against exploding gradients.
    ```python
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3, clipnorm=1.0)
    ```

---

## 12. Serialization & Deployment

The `PowerMLP` model and its custom `PowerMLPLayer` are fully serializable using Keras 3's modern `.keras` format. This ensures that the model's architecture, weights, and optimizer state are all saved correctly.

### Saving and Loading

The model class includes convenient `save_model` and `load_model` methods that handle the details for you.

```python
# Create and train model
model = PowerMLP.from_variant("tiny", num_classes=10, input_dim=784)
# ... model.fit(...)

# Save the entire model to a single file
model.save_model('my_powermlp_model.keras')

# Load the model in a new session, ready for inference or more training
loaded_model = PowerMLP.load_model('my_powermlp_model.keras')
print("✅ PowerMLP model loaded successfully!")
```

---

## 13. Testing & Validation

### Unit Test Example with Pytest

A robust test suite ensures reliability. Here is an example of a `pytest` test for checking the serialization cycle, a critical feature for production readiness.

```python
import pytest
import numpy as np
import keras
import tempfile
import os
from dl_techniques.models.power_mlp.model import PowerMLP

@pytest.fixture
def dummy_model_and_data():
    """Pytest fixture to provide a model and sample data."""
    model = PowerMLP.from_variant("micro", num_classes=5, input_dim=100)
    data = np.random.randn(4, 100).astype("float32")
    return model, data

def test_serialization_cycle(dummy_model_and_data):
    """CRITICAL TEST: Ensures a model can be saved and loaded correctly."""
    model, data = dummy_model_and_data
    original_prediction = model(data, training=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.keras")
        model.save_model(filepath)
        loaded_model = PowerMLP.load_model(filepath)

    loaded_prediction = loaded_model(data, training=False)
    np.testing.assert_allclose(
        original_prediction, loaded_prediction, rtol=1e-6, atol=1e-6
    )
    print("✓ Serialization cycle test passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Training is unstable or the loss explodes.**

-   **Cause**: This can happen if `k` is set too high (e.g., > 3) without proper regularization, as the powered activations can lead to very large numerical values.
-   **Solution**:
    1.  **Enable `batch_normalization`**. This is the most effective solution and is highly recommended for `k >= 3`.
    2.  Use a smaller value for `k` (2 or 3 are very stable).
    3.  Use a smaller learning rate and consider gradient clipping in your optimizer (e.g., `clipnorm=1.0`).

### Frequently Asked Questions

**Q: Is PowerMLP a type of KAN?**

A: No. It is an **efficient alternative** to KANs. It is inspired by the goal of creating highly expressive networks but achieves it using a different, much faster mechanism (dual branches with `ReLU-k`) instead of the slow B-splines used in KANs.

**Q: How does `PowerMLPLayer` compare to a standard `Dense` layer?**

A: It is significantly more expressive. A single `PowerMLPLayer` can often learn functions that would require multiple standard `Dense` layers, thanks to its ability to model both sharp and smooth patterns simultaneously.

**Q: When should I use PowerMLP?**

A: It is an excellent choice for tasks where a standard MLP might underfit but a KAN is computationally infeasible. It excels on **tabular data** and can also be used as a powerful classifier on **flattened image features**.

**Q: Can I use this in a Convolutional Neural Network (CNN)?**

A: Yes. PowerMLP works on vector inputs, so it is perfect for the classification head of a CNN. You would place it after the convolutional base and a `Flatten` layer.
```python
cnn_base = keras.applications.VGG16(include_top=False, input_shape=(...))
# ...
x = keras.layers.Flatten()(cnn_base.output)
# Use PowerMLP as the classifier
power_mlp_head = PowerMLP(hidden_units=[x.shape[-1], 256, 128, 10], ...)
outputs = power_mlp_head(x)
model = keras.Model(inputs=cnn_base.input, outputs=outputs)
```

---

## 15. Technical Details

### Mathematical Formulation

The core `PowerMLPLayer` computes the following function:
`f(x) = (max(0, W_main @ x + b_main))^k + W_basis @ φ(x)`

-   The term `(max(0, ...))^k` makes the main branch a **piecewise polynomial of degree `k`**. This allows it to approximate complex functions with sharp transitions very effectively.
-   The term `W_basis @ φ(x)` makes the basis branch a **linear combination of smooth basis functions**. For example, if `φ(x) = [x, sin(x), cos(x)]`, this branch learns a function similar to a truncated Fourier series, which is ideal for modeling global, smooth trends.

By combining these two function classes, PowerMLP can efficiently approximate a much broader set of functions than an MLP that relies on a single type of non-linearity.

---

## 16. Citation

This implementation is inspired by recent research into efficient and expressive neural network architectures. If using these concepts, please consider citing the relevant works:

-   On Kolmogorov-Arnold Networks (the motivation for efficient alternatives):
    ```bibtex
    @article{liu2024kan,
      title={KAN: Kolmogorov-Arnold Networks},
      author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Tegmark, Max},
      journal={arXiv preprint arXiv:2404.19756},
      year={2024}
    }
    ```