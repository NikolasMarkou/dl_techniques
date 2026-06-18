# Initializers Module

The `dl_techniques.initializers` module provides a collection of advanced weight initializers for Keras, focusing on geometric and statistical properties to improve training stability, accelerate convergence, and encourage feature diversity in deep neural networks.

## Overview

This module offers eight specialized initializers that go beyond standard random distributions. They leverage principles from linear algebra and signal processing—such as orthogonality, wavelet theory, and polar/hyperspherical geometry—to construct weight matrices with desirable mathematical properties from the start of training. All initializers are implemented as standard Keras `Initializer` subclasses, supporting full serialization and seamless integration into any Keras model.

## Available Initializers

| Name | Class | Description | Use Case |
|------|-------|-------------|----------|
| `orthonormal` | `OrthonormalInitializer` | Generates a set of mutually orthogonal vectors with unit norm (orthonormal) via QR decomposition. | Stabilizing training and mitigating vanishing/exploding gradients in deep networks. |
| `he_orthonormal` | `HeOrthonormalInitializer`| Combines He normal seeding with QR decomposition to produce an orthonormal matrix. | Orthonormal initialization where the underlying random source is scaled for ReLU-based architectures. |
| `hypersphere_orthogonal` | `OrthogonalHypersphereInitializer` | Creates orthogonal vectors on a hypersphere of a specified radius. Falls back to a uniform distribution if orthogonality is impossible. | Maximizing initial feature diversity for embeddings, attention heads, or mixture-of-experts models. |
| `haar_wavelet` | `HaarWaveletInitializer` | Deterministically creates fixed 2x2 filters for 2D Haar wavelet decomposition. | Building non-trainable, engineered feature extractors for multi-resolution analysis in CNNs. |
| `polar` | `PolarInitializer` | Sets each weight vector (along a chosen axis) to an exact L2 norm with a uniform-on-sphere direction. | Equinorm / magnitude-controlled, well-conditioned initialization where chi-distributed Gaussian norms are undesirable. |
| `gabor_filters` | `GaborFiltersInitializer` | Deterministically fills a convolution kernel with a bank of Gabor filters whose orientation, frequency, scale, aspect, and phase are swept uniformly across the Ozbulak-Ekenel Table I intervals. | Pre-training-free transfer learning by initializing the first convolutional layer with edge/texture-selective low-level features. |

## Orthonormal Initializer

Generates a weight matrix where each row is a unit vector and is orthogonal to all other rows. This is achieved by applying QR decomposition to a random Gaussian matrix. Such matrices preserve signal norm (isometry), which helps prevent gradients from vanishing or exploding during backpropagation.

**Mathematical Constraint:** It is impossible to create more than `d` orthogonal vectors in a `d`-dimensional space. This initializer will raise a `ValueError` if the number of output vectors (e.g., `units` in a `Dense` layer) exceeds the feature dimensionality.

### Usage

```python
import keras
from dl_techniques.initializers import OrthonormalInitializer

# Create 64 orthonormal vectors in a 128-dimensional space
initializer = OrthonormalInitializer(seed=42)
weights = initializer(shape=(64, 128))

# Use in a Dense layer
layer = keras.layers.Dense(
    units=64,
    input_dim=128,
    kernel_initializer=OrthonormalInitializer(seed=123)
)

# This will raise a ValueError because 128 > 64
# invalid_layer = keras.layers.Dense(128, input_dim=64, kernel_initializer=initializer)
```

## He Orthonormal Initializer

This initializer combines the variance-scaling principle of He initialization with orthogonality. It first creates a random matrix from a He normal distribution (`stddev=sqrt(2/fan_in)`) and then applies QR decomposition to make it orthonormal.

**Key Insight:** The final weight matrix is strictly orthonormal and does **not** retain the He variance. The He normal distribution simply acts as a well-scaled random source *before* orthogonalization, potentially providing a better-conditioned starting point for the QR algorithm compared to a standard Gaussian.

### Usage

```python
import keras
from dl_techniques.initializers import HeOrthonormalInitializer

# Create an initializer for a Dense layer with 128 input features
initializer = HeOrthonormalInitializer(seed=42)

# This layer's kernel will be initialized as a 64x128 orthonormal matrix
layer = keras.layers.Dense(
    units=64,
    input_dim=128,
    kernel_initializer=initializer
)
```

## Orthogonal Hypersphere Initializer

This initializer creates weight vectors that are both mutually orthogonal and lie on the surface of a hypersphere with a specified `radius`. It intelligently handles cases where perfect orthogonality is mathematically impossible.

**Behavior Modes:**
1.  **Feasible (`num_vectors <= latent_dim`):** Generates a perfectly orthogonal set of vectors via QR decomposition and scales each to the desired `radius`.
2.  **Infeasible (`num_vectors > latent_dim`):** Issues a `UserWarning` and falls back to generating vectors that are uniformly distributed on the hypersphere's surface. This maximizes the *average* angular separation when perfect orthogonality cannot be achieved.

### Usage

```python
import keras
from dl_techniques.initializers import OrthogonalHypersphereInitializer

# Feasible case: 64 orthogonal vectors in 256D space on a hypersphere of radius 1.5
init_feasible = OrthogonalHypersphereInitializer(radius=1.5, seed=42)
layer_feasible = keras.layers.Embedding(
    input_dim=1000,
    output_dim=256,
    embeddings_initializer=init_feasible
) # The weights will have shape (1000, 256) - Infeasible! This will fallback.

# Corrected example for an embedding layer
# To get orthogonal embeddings, the output_dim must be >= input_dim
# This is unusual for embeddings but illustrates the principle for a weight matrix
# A better example is a Dense layer:
layer_dense_feasible = keras.layers.Dense(
    units=64,
    input_dim=256,
    kernel_initializer=init_feasible
) # Kernel shape (256, 64) -> Transposed (64, 256), so 64 vectors in 256D. Feasible.

# Infeasible case: Tries to create 512 orthogonal vectors in 128D space.
# Will issue a warning and fall back to uniform hypersphere distribution.
init_infeasible = OrthogonalHypersphereInitializer(radius=1.0, seed=42)
layer_infeasible = keras.layers.Dense(
    units=512,
    input_dim=128,
    kernel_initializer=init_infeasible # Kernel shape (128, 512) -> (512, 128). Infeasible.
)
```

## Haar Wavelet Initializer

This is a deterministic initializer that populates a 2x2 convolutional kernel with the four basis filters of the 2D Haar wavelet transform. It is designed to create a non-trainable layer that performs a single level of multi-resolution analysis, separating an input into approximation (LL), horizontal (LH), vertical (HL), and diagonal (HH) details.

### Usage

The `HaarWaveletInitializer` is typically used with a `Conv2D` or `DepthwiseConv2D` layer. A builder utility, `create_haar_depthwise_conv2d`, is provided for convenience.

```python
import keras
from dl_techniques.initializers import HaarWaveletInitializer, create_haar_depthwise_conv2d

# -- Method 1: Direct Initializer Usage --
haar_conv = keras.layers.Conv2D(
    filters=12, # 3 input channels * 4 (channel_multiplier)
    kernel_size=2,
    strides=2,
    padding='valid',
    kernel_initializer=HaarWaveletInitializer(),
    trainable=False, # Wavelet filters are typically fixed
    input_shape=(256, 256, 3)
)

# -- Method 2: Using the Builder Utility (Recommended) --
# This creates a DepthwiseConv2D layer pre-configured for wavelet decomposition.
haar_layer = create_haar_depthwise_conv2d(
    input_shape=(256, 256, 3),
    channel_multiplier=4, # Create all 4 detail coefficients per input channel
    trainable=False,
    name='haar_wavelet_decomposition'
)
# Input shape: (B, 256, 256, 3) -> Output shape: (B, 128, 128, 12)
```

## Polar Initializer

Samples weights "in polar coordinates": every vector along `axis` is given an
**exact** L2 norm with a direction drawn **uniformly on the unit sphere**. By
PolarQuant's Lemma 2, a Gaussian vector's direction is exactly uniform on the
sphere, so this is realized by normalizing a Gaussian and rescaling to the
target norm — for any shape, power-of-two or not.

Unlike He/Glorot/Gaussian sampling (whose per-vector norms are chi-distributed),
`PolarInitializer` gives every vector an identical, exact norm — useful for
"equinorm" initialization and precise magnitude control.

**Arguments:** `norm` (target L2 norm; `None` => `sqrt(2)`, the He-normal energy),
`axis` (vector axis; `0` = `fan_in` for a Dense kernel), `gain`, `seed`.

### Usage

```python
import keras
from dl_techniques.initializers import PolarInitializer

# Every output unit's weight vector starts with L2 norm exactly 1.0
layer = keras.layers.Dense(128, kernel_initializer=PolarInitializer(norm=1.0, axis=0))
```

It is the companion of `PolarWeightNorm` (see the module docstring of
`dl_techniques/layers/norms/polar_weight_norm.py`).

## Gabor Filters Initializer

This is a deterministic initializer that fills a convolutional kernel with a bank
of 2D Gabor filters, implementing the CNN initialization scheme of Ozbulak &
Ekenel, "Initialization of Convolutional Neural Networks by Gabor Filters". The
idea is to seed the **first convolutional layer** with biologically-motivated,
edge- and texture-selective features instead of random noise. Because a Gabor
bank already captures the kind of oriented, multi-scale low-level structure that
the early layers of a trained network learn anyway, this provides much of the
benefit of transfer learning *without* requiring a pretrained network — the
filters are a strong starting point that is then fine-tuned by ordinary training.

Each output channel `j` holds a distinct 2D Gabor filter evaluated on a grid
centered at the origin (paper Eq. 2):

```
x_theta =  x*cos(theta) + y*sin(theta)
y_theta = -x*sin(theta) + y*cos(theta)
g(x, y) = exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2))
          * cos(2*pi*x_theta/lambda + psi)
```

The number of distinct filters equals the number of output channels
(`n_filters = out_channels`). For each of the five parameters, `out_channels`
values are drawn with `np.linspace(min, max, out_channels)` across its Table I
interval, and channel `j` uses the `j`-th value of every parameter. The same 2D
Gabor filter is replicated identically across all input channels.

**Arguments:** all five arguments are `(min, max)` ranges swept uniformly across
the output channels:

- `sigma_range` — Gaussian envelope standard deviation (scale). Table I default
  `(2.0, 21.0)`. The minimum must be `> 0`.
- `theta_range` — filter orientation **in degrees**. Table I default
  `(0.0, 360.0)`.
- `lambda_range` — sinusoidal wavelength (frequency). Table I default
  `(8.0, 100.0)`.
- `gamma_range` — spatial aspect ratio (ellipticity of the envelope). Table I
  default `(0.0, 300.0)`.
- `psi_range` — phase offset **in degrees**. Table I default `(0.0, 360.0)`.

Note that `theta` and `psi` are specified in degrees and converted internally.
For the degenerate `out_channels == 1` case, every `np.linspace(min, max, 1)`
returns `[min]`, so the single filter uses the minimum endpoint of each range.

### Usage

A builder utility, `create_gabor_depthwise_conv2d`, is provided for convenience.
It applies the Gabor bank **per channel** (depthwise, no cross-channel mixing):
each of `filters` Gabor filters is applied independently to every input channel,
so the output has `in_channels * filters` channels. For a specific output width,
follow it with a `1x1` `Conv2D` projection.

```python
import keras
from dl_techniques.initializers import (
    GaborFiltersInitializer,
    create_gabor_depthwise_conv2d,
)

# -- Method 1: Direct Initializer Usage --
# The deterministic Gabor bank fills any Conv2D/DepthwiseConv2D kernel whose
# LAST axis is the filter count; it is replicated across the `in` axis.
gabor_conv = keras.layers.DepthwiseConv2D(
    kernel_size=7,
    depth_multiplier=96,           # 96 Gabor filters PER input channel
    padding='same',
    depthwise_initializer=GaborFiltersInitializer(),
    trainable=False,               # frozen orientation/frequency front-end
)
# Input (32, 32, 3) -> Output (32, 32, 3 * 96 = 288)

# -- Method 2: Using the Builder Utility --
# Frozen per-channel Gabor front-end (DepthwiseConv2D, output = in * filters).
gabor_layer = create_gabor_depthwise_conv2d(
    filters=96,
    kernel_size=7,
    name='gabor_front_end',
)
# For a specific output count, follow with a 1x1 projection:
proj = keras.layers.Conv2D(64, 1)   # 288 -> 64
```

## Integration with Keras Models

These initializers can be used with any Keras layer that accepts a `kernel_initializer` or similar argument.

```python
import keras
from dl_techniques.initializers import (
    OrthonormalInitializer,
    HeOrthonormalInitializer,
    OrthogonalHypersphereInitializer
)

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(
        256,
        # Use HeOrthonormal for a ReLU-based network
        kernel_initializer=HeOrthonormalInitializer(seed=1)
    ),
    keras.layers.ReLU(),
    keras.layers.Dense(
        128,
        # Use standard Orthonormal for subsequent layers
        kernel_initializer=OrthonormalInitializer(seed=2)
    ),
    keras.layers.ReLU(),
    keras.layers.Dense(
        10,
        # Use Hypersphere to encourage diverse features before softmax
        kernel_initializer=OrthogonalHypersphereInitializer(radius=1.2, seed=3)
    ),
    keras.layers.Softmax()
])

model.summary()
```