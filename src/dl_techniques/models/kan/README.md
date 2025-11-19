# KAN: Kolmogorov-Arnold Networks

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **Kolmogorov-Arnold Network (KAN)** architecture in **Keras 3**. This implementation is based on the recent paper by Liu et al. and provides a powerful alternative to traditional Multi-Layer Perceptrons (MLPs).

The architecture's key feature is its use of **learnable activation functions** on the edges of the network, parameterized by B-splines. This allows KANs to learn complex, non-linear relationships with potentially greater accuracy and parameter efficiency than MLPs.

---

## Table of Contents

1. [Overview: What is KAN and Why It Matters](#1-overview-what-is-kan-and-why-it-matters)
2. [The Problem KAN Solves](#2-the-problem-kan-solves)
3. [How KAN Works: Core Concepts](#3-how-kan-works-core-concepts)
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

## 1. Overview: What is KAN and Why It Matters

### What is KAN?

**KAN (Kolmogorov-Arnold Network)** is a novel neural network architecture inspired by the Kolmogorov-Arnold representation theorem. Unlike traditional MLPs which have fixed activation functions on nodes (e.g., ReLU, SiLU), KANs place **learnable activation functions on the edges** (connections) of the network.

Each of these learnable activations is parameterized as a B-spline, allowing the network to learn a wide variety of univariate function shapes for each connection. This fundamentally changes the network's learning paradigm: instead of learning linear transformations between fixed non-linearities, KANs learn the non-linear transformations themselves.

### Key Innovations of this Implementation

1.  **Faithful Architecture**: The core `KANLinear` layer accurately implements the `y_j = Σ_i φ_ij(x_i)` structure from the paper, where `φ_ij` is a learnable activation on the edge connecting input `i` to output `j`.
2.  **Keras 3 Native & Serializable**: The entire architecture is built using modern Keras 3 functional patterns, ensuring it is fully serializable to the `.keras` format and compatible with TensorFlow, PyTorch, and JAX.
3.  **Model Variants & Factories**: Includes predefined variants (`micro`, `small`, `medium`, etc.) and easy-to-use factory functions (`from_variant`, `from_layer_sizes`) for rapid model creation.
4.  **Adaptive Grids**: Includes a dedicated `update_kan_grids(x)` method on the model. This automatically performs a forward pass to capture hidden layer distributions and updates the B-spline grids for optimal expressivity.

### Why KAN Matters

**Traditional MLP**:
```
Model: Multi-Layer Perceptron (MLP)
  1. Has fixed, non-linear activation functions on nodes (e.g., ReLU, GELU, Swish).
  2. Learns linear transformations (weight matrices) to map between these fixed
     non-linearities.
  3. Can be prone to scaling issues and may require very deep/wide architectures
     to approximate complex functions.
```

**KAN's Solution**:
```
Model: Kolmogorov-Arnold Network (KAN)
  1. Places learnable activation functions (B-splines) on the edges. The nodes
     simply perform summation.
  2. Directly learns the optimal non-linear transformations for each connection
     in a data-driven way.
  3. Benefit: Often achieves higher accuracy with smaller, shallower networks.
     The learned splines can also be visualized, offering better interpretability.
```

### Real-World Impact

KANs are particularly promising for tasks where the underlying function is complex and not well-approximated by standard activation functions:
-   **Scientific Discovery**: Fitting symbolic formulas to data, solving PDEs.
-   **Finance**: Modeling complex, non-linear relationships in financial markets.
-   **Control Systems**: Learning intricate control policies in reinforcement learning.

---

## 2. The Problem KAN Solves

### The Rigidity of Fixed Activation Functions

MLPs have been incredibly successful, but their core design relies on a fundamental assumption: that complex functions can be approximated by composing many simple, *fixed* non-linearities (like ReLU).

This rigidity means MLPs might require a very large number of neurons and parameters to accurately model a function, especially if that function has a complex, non-standard shape.

### How KAN's Learnable Activations Change the Game

KANs remove this constraint by making the activation function itself a learnable part of the model.

```
┌─────────────────────────────────────────────────────────────┐
│  The KAN Flexible Solution                                  │
│                                                             │
│  1. Learnable Splines: Each connection in a KAN learns a    │
│     univariate function `φ(x)` represented as a B-spline.   │
│     By adjusting spline coefficients, the model can shape   │
│     this function to whatever form best fits the data.      │
│                                                             │
│  2. Expressiveness & Efficiency: A single KAN connection    │
│     can learn to be a sine wave, a Gaussian, a step         │
│     function, or anything in between. This allows KANs to   │
│     achieve better accuracy with fewer parameters.          │
│                                                             │
│  3. Interpretability: After training, you can plot the      │
│     learned spline function `φ_ij` for any connection to    │
│     understand how the model transforms a specific feature. │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How KAN Works: Core Concepts

### The High-Level Architecture

A KAN model is a stack of `KANLinear` layers. The final activation (e.g., `softmax`) is applied separately after the last layer.

```
┌──────────────────────────────────────────────────────────────────┐
│                     KAN Model Architecture                       │
│                                                                  │
│  Input Features ───►┌───────────┐                                │
│                     │ KANLinear │ (Layer 1, outputs logits)      │
│                     └─────┬─────┘                                │
│                           │                                      │
│                         .....                                    │
│                           │                                      │
│                     ┌─────▼─────┐                                │
│                     │ KANLinear │ (Layer N, outputs logits)      │
│                     └─────┬─────┘                                │
│                           │                                      │
│                     ┌─────▼─────┐                                │
│                     │ Activation│ (e.g., Softmax)                │
│                     └─────┬─────┘                                │
│                           │                                      │
│                     ┌─────▼─────┐                                │
│                     │  Output   │                                │
│                     └───────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow (Inside a `KANLinear` Layer)

The output `y_j` is the sum of learned activation functions `φ_ij` applied to each input `x_i`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     KANLinear Layer Data Flow                           │
└─────────────────────────────────────────────────────────────────────────┘

Input x (B, F_in)
    │
    ▼
For each connection from input i to output j:
    │
    ├─► Path 1: Base Activation
    │   └─► `base_activation(x_i)` -> `base_val`
    │
    └─► Path 2: Spline Activation
        ├─► `_compute_bspline_basis(x_i)` -> `basis` (num_basis,)
        └─► `einsum('k,k->', basis, C_ijk)` -> `spline_val`

    │
    ▼
STEP 1: COMBINE TO FORM φ_ij(x_i)
─────────────────────────────────
`φ_ij(x_i) = w_base_ij * base_val + w_spline_ij * spline_val`


STEP 2: AGGREGATE TO FORM OUTPUT y_j
─────────────────────────────────
`y_j = Σ_i φ_ij(x_i)`


STEP 3: FINAL OUTPUT
────────────────────────────────
Output y (B, F_out)
```

---

## 4. Architecture Deep Dive

### 4.1 `KANLinear` Layer

The fundamental building block of a KAN. It learns a unique activation `φ_ij` for each connection.
-   **`spline_weight`**: A tensor of coefficients for the B-spline basis functions. These are the learnable parameters that define the shape of the spline component.
-   **`spline_scaler` & `base_scaler`**: Learnable scalars that control the magnitude of the spline and base components respectively.
-   **`grid`**: A non-trainable weight that stores the B-spline knot vectors. This persists during saving/loading and is updated via `update_grid_from_samples`.
-   **`base_activation`**: A fixed activation function (e.g., `'swish'`) applied alongside the spline.
-   **No Bias Vector**: Unlike Dense layers, `KANLinear` does not typically use a separate bias vector, as the spline and base paths handle the mapping expressively.

### 4.2 `KAN` Model

A Keras `Model` subclass that stacks `KANLinear` layers.
-   **Functional Graph**: Constructed via the Keras Functional API for robustness.
-   **`update_kan_grids(x)`**: A critical utility that performs a forward pass to extract inputs for hidden layers and updates their grids to match the actual activation distribution.
-   **Factory Methods**: Provides `create_kan_model`, `from_variant`, and `from_layer_sizes` for rapid prototyping.

---

## 5. Quick Start Guide

### Your First KAN Model (30 seconds)

Let's build a KAN to learn a simple non-linear function: `y = sin(π*x1) + x2^2`.

```python
import keras
import numpy as np
from dl_techniques.models.kan.model import create_kan_model

# 1. Generate synthetic data
def generate_data(num_samples):
    X = np.random.rand(num_samples, 2) * 2 - 1  # Input in [-1, 1]
    y = np.sin(np.pi * X[:, 0]) + np.square(X[:, 1])
    return X, y

X_train, y_train = generate_data(2000)
X_val, y_val = generate_data(400)

# 2. Create KAN model for regression
# input_features=2, output_features=1 (for regression)
model = create_kan_model(
    variant="micro",
    input_features=2,
    output_features=1
)

# 3. IMPORTANT: Initialize grids with data distribution
# This adapts the B-splines to the input range [-1, 1]
model.update_kan_grids(X_train[:100])

# 4. Compile and Train
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

print("✅ KAN model created and compiled successfully!")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    verbose=1
)
```

---

## 6. Component Reference

### 6.1 `KAN` (Model Class)

**Purpose**: The main Keras `Model` subclass that assembles the KAN architecture.

```python
from dl_techniques.models.kan.model import KAN

# Create a custom KAN model
layer_configs = [
    {"features": 32, "grid_size": 8, "base_activation": "gelu"},
    {"features": 1, "activation": "linear"} 
]
model = KAN(layer_configs=layer_configs, input_features=10)
model.update_kan_grids(training_data) # Don't forget this!
```

### 6.2 Factory Functions

#### `create_kan_model(...)`
The recommended high-level factory for creating a standard KAN configuration.

#### `KAN.from_variant(...)`
Class method to create a KAN model from standard configurations (`micro`, `small`, etc.).

#### `KAN.from_layer_sizes(...)`
Class method to create a KAN with a uniform configuration for all layers, defined only by their node counts.

---

## 7. Configuration & Model Variants

This implementation provides several standard configurations.

| Variant | Hidden Features | Grid Size | Spline Order | Base Activation | Use Case |
|:---:|:---|:---:|:---:|:---:|:---|
| **`micro`** | `[16, 8]` | 3 | 3 | `swish`| Quick tests, simple functions |
| **`small`** | `[64, 32, 16]` | 5 | 3 | `swish`| MNIST, small datasets |
| **`medium`**| `[128, 64, 32]` | 7 | 3 | `gelu` | CIFAR-10, medium datasets |
| **`large`** | `[256, 128, 64, 32]`| 10| 3 | `gelu` | Complex tasks |
| **`xlarge`**| `[512, 256, 128, 64]`| 12| 3 | `gelu` | Large-scale datasets |

### Customizing a Variant

You can easily override the default settings of a variant.

```python
# Create a 'medium' KAN but use a smaller grid and a different base activation
model = create_kan_model(
    variant="medium",
    input_features=256,
    output_features=10,
    override_config={"grid_size": 5, "base_activation": "silu"}
)
```

---

## 8. Comprehensive Usage Examples

### Example 1: MNIST Image Classification

```python
import keras
from dl_techniques.models.kan.model import create_kan_model

# 1. Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# 2. Create model
model = create_kan_model(
    variant="small",
    input_features=784,
    output_features=10,
    output_activation="softmax"
)

# 3. Adapt grids to training data
model.update_kan_grids(x_train[:1000])

# 4. Compile and Train
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
```

### Example 2: Custom Architecture from Scratch

Define every layer's configuration manually for full control.

```python
from dl_techniques.models.kan.model import KAN

layer_configs = [
    {"features": 64, "grid_size": 8, "spline_order": 3, "base_activation": "gelu"},
    {"features": 32, "grid_size": 5, "spline_order": 2, "base_activation": "gelu"},
    # The last KANLinear layer produces logits; 'activation' specifies the
    # final activation to be applied *after* this layer.
    {"features": 10, "grid_size": 5, "activation": "softmax"}
]

custom_model = KAN(layer_configs=layer_configs, input_features=784)
custom_model.update_kan_grids(x_train[:500])
custom_model.compile(optimizer="adamw", loss="sparse_categorical_crossentropy")
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Visualizing the Learned Activation Functions

One of KAN's biggest advantages is interpretability. You can inspect the learned non-linear functions `φ_ij`.

```python
import matplotlib.pyplot as plt
import numpy as np
import keras

# Assume 'model' is a trained KAN model
first_kan_layer = model.get_layer("kan_layer_0")

# Choose an input and output feature to inspect
input_feature_idx = 0
output_feature_idx = 0

# Extract weights
spline_w = first_kan_layer.spline_weight[input_feature_idx, output_feature_idx, :]
spline_s = first_kan_layer.spline_scaler[input_feature_idx, output_feature_idx]
base_s = first_kan_layer.base_scaler[input_feature_idx, output_feature_idx]
grid = first_kan_layer.grid # Access the grid variable

# Generate a range of input values to plot
x_plot = np.linspace(first_kan_layer.grid_range[0], first_kan_layer.grid_range[1], 200)
x_tensor = keras.ops.convert_to_tensor(x_plot, dtype=model.dtype)

# 1. Compute spline component
basis_vals = first_kan_layer._compute_bspline_basis(x_tensor)
spline_val = keras.ops.einsum('bi,i->b', basis_vals, spline_w)

# 2. Compute base component
base_val = first_kan_layer.base_activation_fn(x_tensor)

# 3. Combine them to get the full activation function φ_ij(x)
phi_ij = (base_s * base_val) + (spline_s * spline_val)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(x_plot, phi_ij, label=f"φ(x) for connection ({input_feature_idx} -> {output_feature_idx})")
plt.xlabel("Input Feature Value (x_i)")
plt.ylabel("Learned Transformation (φ_ij)")
plt.title("KAN Learned Activation Function")
plt.legend()
plt.grid(True)
plt.show()
```

### Pattern 2: Hybrid Models (KAN + Standard Layers)

You can use `KANLinear` as a drop-in replacement for `Dense` in any Keras model.

```python
from dl_techniques.layers.kan_linear import KANLinear

inputs = keras.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(32, 3, activation="relu")(inputs)
x = keras.layers.Flatten()(x)
# Replace Dense layers with KANLinear for more expressive feature learning
x = KANLinear(features=128, grid_size=8, activation="gelu")(x)
x = KANLinear(features=64, grid_size=5, activation="gelu")(x)
outputs = keras.layers.Dense(10, activation="softmax")(x) # Standard output layer

hybrid_model = keras.Model(inputs, outputs)
```

---

## 10. Performance Optimization

### Mixed Precision Training

For larger variants, mixed precision can significantly speed up training on compatible GPUs.

```python
keras.mixed_precision.set_global_policy('mixed_float16')
model = create_kan_model(variant="large", ...)
```

### XLA Compilation

Use `jit_compile=True` for graph compilation, which can provide a speed boost.

```python
model.compile(optimizer="adam", loss="...", jit_compile=True)
```

---

## 11. Training and Best Practices

### The `grid_size` vs. Overfitting Trade-off

-   **Start small (`grid_size=3` to `5`)**: This encourages smoother, general functions and prevents overfitting.
-   **Increase if underfitting**: If loss plateaus too high, a larger `grid_size` (e.g., 7-10) provides more capacity. This increases parameter count significantly.

### Adaptive Grids

For best performance, the B-spline grids should adapt to the evolving distribution of activations in hidden layers. The `KAN` model class provides a dedicated method for this.

```python
class KANGridUpdateCallback(keras.callbacks.Callback):
    def __init__(self, data, update_freq=5):
        super().__init__()
        self.data = data
        self.update_freq = update_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.update_freq == 0:
            # Updates all layers in the model by running a forward pass
            self.model.update_kan_grids(self.data)
            print(f"\nUpdated KAN grids at epoch {epoch + 1}")

# Usage:
# grid_updater = KANGridUpdateCallback(X_train[:500])
# model.fit(..., callbacks=[grid_updater])
```

---

## 12. Serialization & Deployment

The `KAN` model and `KANLinear` layer are fully serializable using Keras 3's modern `.keras` format. The B-spline grids are saved as part of the model weights.

### Saving and Loading

```python
# Create and train model
model = create_kan_model(variant="small", ...)
# model.fit(...)

# Save the entire model
model.save('my_kan_model.keras')

# Load the model without needing custom_objects
loaded_model = keras.models.load_model('my_kan_model.keras')
```

---

## 13. Testing & Validation

```python
import keras
import numpy as np
from dl_techniques.models.kan.model import create_kan_model

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = create_kan_model("micro", input_features=32, output_features=5)
    dummy_input = np.random.rand(4, 32)
    output = model.predict(dummy_input)
    assert output.shape == (4, 5)
    print("✓ Forward pass has correct shape")

# Run test
test_forward_pass_shape()
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Training is slow compared to an MLP.**

-   **Cause**: `KANLinear` is more computationally intensive than `Dense` due to B-spline computation. Parameter count also scales with `grid_size`.
-   **Solution**: 1) Start with a smaller `grid_size`. 2) Use a shallower/narrower KAN. 3) Enable mixed precision and XLA compilation.

**Issue 2: The model is overfitting.**

-   **Cause**: `grid_size` is too large, allowing splines to fit noise.
-   **Solution**: 1) Reduce `grid_size`. 2) Add L2 regularization via the `kernel_regularizer` argument (supported in `kwargs`). 3) Add dropout layers between `KANLinear` layers.

### Frequently Asked Questions

**Q: When should I choose KAN over a standard MLP?**

A: Use a KAN when you suspect the underlying relationships in your data are highly non-linear and not well captured by standard activations. KANs are also a great choice when interpretability is important. For simple problems or when speed is the absolute priority, an MLP is still a strong baseline.

**Q: How do I choose the `base_activation`?**

A: The `base_activation` (e.g., `'swish'`) provides a well-behaved "scaffold" for the learnable spline. `swish` or `gelu` are good defaults as they are smooth and non-monotonic, giving the spline a good starting point.

---

## 15. Technical Details

### Kolmogorov-Arnold Representation Theorem

The theoretical foundation for KANs states that any multivariate continuous function `f(x1, ..., xn)` can be represented as a finite sum of compositions of univariate functions: `f(x) = Σ_q Φ_q( Σ_p ψ_{q,p}(x_p) )`. A two-layer KAN is a direct neural network realization of this theorem.

### B-Spline Parameterization

A B-spline is a piecewise polynomial function. By representing the learnable activation as a linear combination of B-spline basis functions, the model can approximate any continuous function on a given interval (`grid_range`). The learnable `spline_weight` parameters are the control points that shape the function.

---

## 16. Citation

This implementation is based on the original KAN paper. If you use this model or its concepts in your research, please cite the foundational work:

```bibtex
@article{liu2024kan,
  title={{KAN}: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```