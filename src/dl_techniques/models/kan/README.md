# KAN: Kolmogorov-Arnold Networks

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **Kolmogorov-Arnold Network (KAN)** architecture in **Keras 3**. This implementation is based on the recent paper by Liu et al. and provides a powerful alternative to traditional Multi-Layer Perceptrons (MLPs).

This implementation follows the `dl_techniques` framework standards and modern Keras 3 best practices. It provides a modular, well-documented, and fully serializable model that works seamlessly across TensorFlow, PyTorch, and JAX backends. The architecture's key feature is its use of **learnable activation functions** on the edges of the network, parameterized by B-splines. This allows KANs to learn complex, non-linear relationships with potentially greater accuracy and parameter efficiency than MLPs.

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

1.  **Dual-Pathway `KANLinear` Layer**: The core `KANLinear` layer combines a standard linear transformation with a learnable spline-based transformation. This provides both stability (from the linear path) and high expressiveness (from the spline path).
2.  **Keras 3 Native & Serializable**: The entire architecture is built using modern Keras 3 functional patterns, ensuring it is fully serializable to the `.keras` format and compatible with TensorFlow, PyTorch, and JAX.
3.  **Model Variants & Factories**: Includes predefined variants (`micro`, `small`, `medium`, etc.) and easy-to-use factory functions (`from_variant`, `from_layer_sizes`) for rapid model creation.
4.  **Numerical Stability**: The implementation incorporates several techniques like input normalization, gradient clipping, and careful weight initialization to ensure stable training.

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
-   **Image Processing**: Representing complex image transformations that go beyond simple convolutions.

---

## 2. The Problem KAN Solves

### The Rigidity of Fixed Activation Functions

MLPs have been incredibly successful, but their core design relies on a fundamental assumption: that complex functions can be approximated by composing many simple, *fixed* non-linearities (like ReLU).

```
┌─────────────────────────────────────────────────────────────┐
│  The Challenge of Fixed Non-Linearities                     │
│                                                             │
│  1. The ReLU function is a simple ramp. To approximate a    │
│     sine wave, an MLP needs many ReLU units, effectively    │
│     creating a piecewise linear approximation. This can be  │
│     inefficient.                                            │
│                                                             │
│  2. The "best" non-linearity is data-dependent. For some    │
│     problems, a periodic function might be ideal; for       │
│     others, a saturating one. An MLP cannot adapt its       │
│     activations.                                            │
└─────────────────────────────────────────────────────────────┘
```

This rigidity means MLPs might require a very large number of neurons and parameters to accurately model a function, especially if that function has a complex, non-standard shape.

### How KAN's Learnable Activations Change the Game

KANs remove this constraint by making the activation function itself a learnable part of the model.

```
┌─────────────────────────────────────────────────────────────┐
│  The KAN Flexible Solution                                  │
│                                                             │
│  1. Learnable Splines: Each connection in a KAN learns a    │
│     univariate function represented as a B-spline. By       │
│     adjusting the spline coefficients, the model can shape  │
│     this function to whatever form best fits the data.      │
│                                                             │
│  2. Expressiveness & Efficiency: A single KAN connection    │
│     can learn to be a sine wave, a Gaussian, a step         │
│     function, or anything in between. This allows KANs to   │
│     achieve better accuracy with fewer parameters.          │
│                                                             │
│  3. Interpretability: After training, you can plot the      │
│     learned spline function for any connection to understand│
│     how the model is transforming a specific feature.       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How KAN Works: Core Concepts

### The High-Level Architecture

A KAN model is a stack of `KANLinear` layers, which operate in sequence to transform the input features.

```
┌──────────────────────────────────────────────────────────────────┐
│                     KAN Model Architecture                       │
│                                                                  │
│  Input Features ───►┌───────────┐                                │
│                     │ KANLinear │ (Layer 1)                      │
│                     └─────┬─────┘                                │
│                           │                                      │
│                     ┌─────▼─────┐                                │
│                     │ KANLinear │ (Layer 2)                      │
│                     └─────┬─────┘                                │
│                           │                                      │
│                         .....                                    │
│                           │                                      │
│                     ┌─────▼─────┐                                │
│                     │ KANLinear │ (Layer N)                      │
│                     └─────┬─────┘                                │
│                           │                                      │
│                     ┌─────▼─────┐                                │
│                     │  Output   │                                │
│                     └───────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow (Inside a `KANLinear` Layer)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     KANLinear Layer Data Flow                           │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: DUAL-PATHWAY TRANSFORMATION
─────────────────────────────────
Input (B, F_in)
    │
    ├─► Path 1: Base Transform (Linear)
    │   └─► `x @ W_base` -> `base_output` (B, F_out)
    │
    └─► Path 2: Spline Transform (Non-Linear)
        ├─► `_normalize_inputs(x)`
        ├─► `_compute_spline_basis(x_norm)` -> `spline_basis` (B, F_in, num_basis)
        ├─► `einsum('...ik,iok->...o', spline_basis, W_spline)` -> `spline_output` (B, F_out)
        └─► `spline_output * spline_scaler` -> `scaled_spline_output` (B, F_out)


STEP 2: COMBINATION & ACTIVATION
────────────────────────────────
`base_output` (B, F_out)
`scaled_spline_output` (B, F_out)
    │
    ├─► `total_output = base_output + scaled_spline_output`
    │    (Residual connection is used if dimensions match)
    │
    └─► `activated_output = activation_fn(total_output)`


STEP 3: FINAL OUTPUT
────────────────────────────────
Output (B, F_out)
```

---

## 4. Architecture Deep Dive

### 4.1 `KANLinear` Layer

The fundamental building block of a KAN.
-   **`base_weight`**: A standard weight matrix for a linear transformation. This provides a stable, residual-like path that helps with optimization.
-   **`spline_weight`**: A tensor of coefficients for the B-spline basis functions. These are the learnable parameters that define the shape of the activation function for each `(input_feature, output_feature)` pair.
-   **`spline_scaler`**: A learnable scaling factor that controls the magnitude of the spline component, allowing the model to balance the linear and non-linear paths.
-   **`grid_size`**: Controls the number of pieces in the piecewise polynomial spline. A larger grid allows for more complex functions but increases parameters and the risk of overfitting.
-   **`spline_order`**: The degree of the polynomial pieces (e.g., 3 for cubic splines). This determines the smoothness of the learned functions.

### 4.2 `KAN` Model

A Keras `Model` subclass that stacks `KANLinear` layers.
-   It is constructed using the Keras functional API within the `__init__` method, a modern best practice that ensures robust serialization.
-   It provides convenient factory methods like `from_variant` and `from_layer_sizes` to simplify model creation for common use cases.

---

## 5. Quick Start Guide

### Your First KAN Model (30 seconds)

Let's build a KAN to learn a simple non-linear function: `y = sin(π*x1) + x2^2`.

```python
import keras
import numpy as np
import matplotlib.pyplot as plt

# Local imports from your project structure
from dl_techniques.models.kan.model import create_compiled_kan

# 1. Generate synthetic data
def generate_data(num_samples):
    X = np.random.rand(num_samples, 2) * 2 - 1  # Input in [-1, 1]
    y = np.sin(np.pi * X[:, 0]) + np.square(X[:, 1])
    return X, y

X_train, y_train = generate_data(2000)
X_val, y_val = generate_data(400)

# 2. Create a tiny, compiled KAN model for regression
# input_features=2, num_classes=1 (for regression)
model = create_compiled_kan(
    variant="micro",
    input_features=2,
    num_classes=1,
    loss="mean_squared_error",
    learning_rate=1e-3,
    metrics=["mean_absolute_error"]
)

# 3. Train the model
print("✅ KAN model created and compiled successfully!")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=1
)
print("✅ Training Complete!")

# 4. Evaluate and visualize a prediction
print("\nEvaluating on validation set:")
model.evaluate(X_val, y_val)

test_input = X_val[:1]
prediction = model.predict(test_input)[0]
ground_truth = y_val[0]

print(f"\nTest Input: {test_input[0]}")
print(f"Prediction: {prediction[0]:.4f}")
print(f"Ground Truth: {ground_truth:.4f}")
```

---

## 6. Component Reference

### 6.1 `KAN` (Model Class)

**Purpose**: The main Keras `Model` subclass that assembles the KAN architecture.

**Location**: `dl_techniques.models.kan.model.KAN`

```python
from dl_techniques.models.kan.model import KAN

# Create a custom KAN model
layer_configs = [
    {"features": 32, "grid_size": 8},
    {"features": 16, "grid_size": 5},
    {"features": 1, "activation": "linear"} # Regression output
]
model = KAN(layer_configs=layer_configs, input_features=10)
```

### 6.2 Factory Functions

**Location**: `dl_techniques.models.kan.model`

#### `create_compiled_kan(...)`
The recommended high-level factory for creating a compiled, ready-to-train KAN model from a variant.

#### `KAN.from_variant(...)`
Class method to create a KAN model from standard configurations (`tiny`, `small`, etc.).

#### `KAN.from_layer_sizes(...)`
Class method to create a KAN with a uniform configuration for all layers, defined only by their sizes.

---

## 7. Configuration & Model Variants

This implementation provides several standard configurations.

| Variant | Hidden Features | Grid Size | Spline Order | Default Activation | Use Case |
|:---:|:---|:---:|:---:|:---:|:---|
| **`micro`** | `[16, 8]` | 3 | 3 | `swish`| Quick tests, simple functions |
| **`small`** | `[64, 32, 16]` | 5 | 3 | `swish`| MNIST, small datasets |
| **`medium`**| `[128, 64, 32]` | 7 | 3 | `gelu` | CIFAR-10, medium datasets |
| **`large`** | `[256, 128, 64, 32]`| 10| 3 | `gelu` | Complex tasks |
| **`xlarge`**| `[512, 256, 128, 64]`| 12| 3 | `gelu` | Large-scale datasets |

### Customizing a Variant

You can easily override the default settings of a variant.

```python
# Create a 'medium' KAN but use a smaller grid and a different activation
model = KAN.from_variant(
    "medium",
    input_features=256,
    num_classes=10,
    override_config={"grid_size": 5, "activation": "silu"}
)
```

---

## 8. Comprehensive Usage Examples

### Example 1: MNIST Image Classification

```python
import keras
from dl_techniques.models.kan.model import create_compiled_kan

# 1. Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# 2. Create and compile a 'small' KAN for classification
model = create_compiled_kan(
    variant="small",
    input_features=784,
    num_classes=10,
    loss="sparse_categorical_crossentropy",
    learning_rate=1e-3,
    metrics=["accuracy"]
)
model.summary()

# 3. Train the model
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
```

### Example 2: Custom Architecture from Scratch

Define every layer's configuration manually for full control.

```python
from dl_techniques.models.kan.model import KAN

layer_configs = [
    {"features": 64, "grid_size": 8, "spline_order": 3, "activation": "gelu"},
    {"features": 32, "grid_size": 5, "spline_order": 2, "activation": "gelu"},
    {"features": 10, "grid_size": 5, "activation": "softmax"}
]

custom_model = KAN(layer_configs=layer_configs, input_features=784)
custom_model.compile(optimizer="adamw", loss="sparse_categorical_crossentropy")
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Visualizing the Learned Activation Functions

One of KAN's biggest advantages is interpretability. You can inspect the learned non-linear functions.

```python
# Assume 'model' is a trained KAN model
first_kan_layer = model.get_layer("kan_layer_0")

# Choose an input and output feature to inspect
input_feature_idx = 0
output_feature_idx = 0

# Extract the spline weights for this specific connection
spline_weights = first_kan_layer.spline_weight[input_feature_idx, output_feature_idx, :]

# Generate a range of input values to plot
x_plot = np.linspace(first_kan_layer.grid_range[0], first_kan_layer.grid_range[1], 100)
x_plot_tensor = keras.ops.convert_to_tensor(x_plot, dtype="float32")

# Compute the spline basis for these values
basis_functions = first_kan_layer._compute_spline_basis(x_plot_tensor[:, np.newaxis])
basis_values = basis_functions[:, 0, :] # Extract for our single feature

# Compute the learned function by combining basis with weights
learned_function = keras.ops.einsum('ib,b->i', basis_values, spline_weights)

# Plot the result
plt.figure()
plt.plot(x_plot, learned_function, label=f"Learned f(x) for ({input_feature_idx} -> {output_feature_idx})")
plt.xlabel("Input Feature Value")
plt.ylabel("Learned Transformation")
plt.title("KAN Learned Activation Function")
plt.legend()
plt.grid(True)
plt.show()
```

### Pattern 2: Hybrid Models (KAN + Standard Layers)

You can use `KANLinear` as a drop-in replacement for `Dense` in any Keras model.

```python
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

For larger variants (`medium`, `large`), mixed precision can significantly speed up training on compatible GPUs.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = create_compiled_kan(variant="large", ...)
# ... compile and fit ...
```

### XLA Compilation

Use `jit_compile=True` for graph compilation, which can provide a speed boost.

```python
model = create_compiled_kan(variant="medium", ...)
model.compile(optimizer="adam", loss="...", jit_compile=True)
```

---

## 11. Training and Best Practices

### The `grid_size` vs. Overfitting Trade-off

-   **Start small (`grid_size=3` to `5`)**: This encourages the model to learn smoother, more general functions and prevents overfitting.
-   **Increase if underfitting**: If the model loss plateaus too high, a larger `grid_size` (e.g., 7-10) can provide the necessary capacity to fit the data better. Be aware that this increases parameter count and memory usage significantly.

### Network Depth and Width

KANs can often achieve high performance with shallower and narrower architectures compared to MLPs. Start with a small variant and only scale up if necessary.

### Regularization

The `regularization_factor` applies L2 regularization to both the base and spline weights. This is crucial for preventing the learned splines from becoming too "wiggly" and overfitting to noise in the training data.

---

## 12. Serialization & Deployment

The `KAN` model and the `KANLinear` layer are fully serializable using Keras 3's modern `.keras` format, thanks to the `@keras.saving.register_keras_serializable()` decorator.

### Saving and Loading

```python
# Create and train model
model = create_compiled_kan(variant="small", ...)
# model.fit(...)

# Save the entire model
model.save('my_kan_model.keras')

# Load the model in a new session without needing custom_objects
loaded_model = keras.models.load_model('my_kan_model.keras')
```

---

## 13. Testing & Validation

```python
import keras
import numpy as np
from dl_techniques.models.kan.model import KAN

def test_model_creation_all_variants():
    """Test model creation for all variants."""
    for variant in KAN.VARIANT_CONFIGS.keys():
        model = KAN.from_variant(variant, input_features=64, num_classes=10)
        assert model is not None
        print(f"✓ KAN-{variant} created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = KAN.from_variant("tiny", input_features=32, num_classes=5)
    dummy_input = np.random.rand(4, 32)
    output = model.predict(dummy_input)
    assert output.shape == (4, 5) # (batch_size, num_classes)
    print("✓ Forward pass has correct shape")

# Run tests
if __name__ == '__main__':
    test_model_creation_all_variants()
    test_forward_pass_shape()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Training is slow compared to an MLP.**

-   **Cause**: The `KANLinear` layer is more computationally intensive than a standard `Dense` layer due to the spline basis computation. The number of parameters also scales with `grid_size`.
-   **Solution**: 1) Start with a smaller `grid_size`. 2) Use a shallower/narrower KAN architecture. 3) Enable mixed precision and XLA compilation.

**Issue 2: The model is overfitting.**

-   **Cause**: The `grid_size` is too large, allowing the splines to fit noise. The regularization might be too weak.
-   **Solution**: 1) Reduce `grid_size`. 2) Increase the `regularization_factor`. 3) Add dropout layers between `KANLinear` layers if needed.

### Frequently Asked Questions

**Q: When should I choose KAN over a standard MLP?**

A: Use a KAN when you suspect the underlying relationships in your data are highly non-linear and not well captured by standard activations like ReLU or GELU. KANs are also a great choice when interpretability is important, as you can visualize the learned functions. For simple problems or when speed is the absolute priority, an MLP is still a strong baseline.

**Q: How is this implementation's spline calculation different from the original paper?**

A: For enhanced numerical stability and simpler implementation, this version uses a Gaussian-like basis for the B-splines rather than the classic recursive definition. This approach is more robust to a wide range of inputs and easier to implement reliably across different backends, while still providing the necessary functional expressiveness.

---

## 15. Technical Details

### Kolmogorov-Arnold Representation Theorem

The theoretical foundation for KANs is the Kolmogorov-Arnold theorem, which states that any multivariate continuous function `f(x1, ..., xn)` can be represented as a finite sum of compositions of univariate functions: `f(x) = Σ_q Φ_q( Σ_p ψ_{q,p}(x_p) )`. A two-layer KAN is a direct neural network realization of this theorem, where the first `KANLinear` layer learns the inner functions `ψ` and the second learns the outer functions `Φ`.

### B-Spline Parameterization

A B-spline is a piecewise polynomial function defined by a set of control points. By representing the learnable activation as a linear combination of B-spline basis functions, the model can approximate any continuous function on a given interval (`grid_range`). The learnable `spline_weight` parameters are essentially the control points that shape the function.

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