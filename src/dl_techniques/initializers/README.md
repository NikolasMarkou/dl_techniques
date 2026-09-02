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
| `gabor_filters` | `GaborFiltersInitializer` | Deterministically fills a convolution kernel with a bank of Gabor filters over a factorized orientation x scale x phase sweep (Ozbulak-Ekenel), DC-removed and energy-normalized to a He-like scale. | Pre-training-free transfer learning by initializing the first convolutional layer with edge/texture-selective low-level features. |

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
of 2D Gabor filters, following the CNN initialization scheme of Ozbulak &
Ekenel, "Initialization of Convolutional Neural Networks by Gabor Filters". The
idea is to seed the **first convolutional layer** with biologically-motivated,
edge- and texture-selective features instead of random noise. Because a Gabor
bank already captures the kind of oriented, multi-scale low-level structure that
the early layers of a trained network learn anyway, this provides much of the
benefit of transfer learning *without* requiring a pretrained network — the
filters are a strong starting point that is then fine-tuned by ordinary training.

Each output channel `j` holds a 2D Gabor filter evaluated on a grid centered at
`((kw - 1) / 2, (kh - 1) / 2)` (paper Eq. 2):

```
x_theta =  x*cos(theta) + y*sin(theta)
y_theta = -x*sin(theta) + y*cos(theta)
g(x, y) = exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2))
          * cos(2*pi*x_theta/lambda + psi)
```

The same 2D Gabor filter is replicated identically across all input channels, so
a `Conv2D` initialized this way responds to the unweighted **sum** of its input
channels (colour-blind at initialization, with a gain that grows with `in_ch`).
Use `create_gabor_depthwise_conv2d` when you want the bank applied per channel.

### Sweep strategy

`sweep="product"` (the default) builds a **factorized** bank: `n_theta`
orientations x `n_scale` scales x `n_psi` phases, with `sigma`, `lambda` and
`gamma` all riding the scale axis so the envelope width tracks the carrier
wavelength. `theta` and `psi` are periodic, so their upper endpoint is exclusive;
`theta` sweeps `[0, 180)` because `g(theta + 180, psi) == g(theta, -psi)`. From 4
filters the phase axis holds `{0, 180}` and from 16 it holds `{0, 90, 180, 270}`,
so every filter has a **phase-reversed sibling** — which is what makes a
rectifying activation on a frozen signed bank lossless — and, in larger banks, a
quadrature partner as well.

`sweep="diagonal"` is the original construction: all five parameters swept
jointly by one `np.linspace` with inclusive endpoints, channel `j` taking the
`j`-th sample of each. That is a 1D curve through a 5D space. Measured on a
96-filter bank it gives a maximum off-diagonal cosine similarity of **0.9999**,
29 pairs above 0.95, and an effective rank (99% spectral energy) of **52/96**. It
is retained only for reproducing the paper's exact construction.

### Normalization

With `normalize=True` (the default) each 2D filter has its DC component removed
and is scaled to a per-element RMS of `sqrt(2 / fan_in)`, `fan_in = kh*kw*in_ch`.
This is what makes the bank usable as an initializer. Measured on
`(11, 11, 3, 96)` with the old un-normalized defaults, per-filter L2 norms spanned
**0.12 to 4.60** (38x) and per-output-channel gain `sum |w|` spanned **0.54 to
100.3** — two orders of magnitude of activation scale at initialization, with the
DC component left in so many filters acted as biased blob detectors.

### Arguments

- `sigma_range` — Gaussian envelope standard deviation (scale). `None` (default)
  resolves at call time to `(0.30*k, 0.60*k)` with `k = min(kh, kw)`. If given,
  the minimum must be `> 0`.
- `theta_range` — filter orientation **in degrees**. Default `(0.0, 180.0)`,
  upper endpoint exclusive in `product` mode.
- `lambda_range` — sinusoidal wavelength (frequency). `None` (default) resolves to
  `(0.30*k, 1.00*k)`. If given, the minimum must be `> 0` — it divides
  `2*pi*x_theta`, and a `0` minimum used to yield a **silent all-NaN kernel**.
- `gamma_range` — spatial aspect ratio (ellipticity). Default `(0.5, 1.5)`; the
  minimum must be `>= 0`.
- `psi_range` — phase offset **in degrees**. Default `(0.0, 360.0)`, upper
  endpoint exclusive in `product` mode.
- `sweep` — `"product"` (default) or `"diagonal"`.
- `n_filters` — number of **distinct** filters; `None` (default) means `out_ch`.
  A smaller value tiles the bank cyclically across the output channels.
- `normalize` — DC-remove and energy-normalize each filter. Default `True`.

All bounds must be finite: a `nan` bound passes every naive comparison
(`nan > hi` and `nan <= 0` are both `False`) and is rejected up front. The
signature is closed — there is no `**kwargs`, because `keras.initializers.
Initializer` has no `__init__` to forward to. A one-filter bank takes the
**midpoint** of every range, not the minimum, since the endpoints are exactly the
degenerate extremes.

### Relationship to the reference implementation

The authors' reference code (`gabor_init.py` in
`github.com/gokhanozbulak/Gabor-Initialized-CNN`) calls `cv2.getGaborKernel` with,
for a kernel of size `k` and `n` filters:

| parameter | reference |
|---|---|
| `sigma` | `[5, k/2 + 1)` — kernel-size dependent |
| `lambda` | `== k` (constant) |
| `theta` | `[0, 360)` degrees |
| `gamma` | `[1.0, 3.0)` — a 0..300 slider **divided by 100** |
| `psi` | `[-90, 180)` degrees — a 90..360 slider minus 180 |

That is the arbiter for the parameter semantics, and it is why `gamma` defaults to
`(0.5, 1.5)` here and **not** to `(0.0, 300.0)`: an aspect ratio of 300 collapses
the `y_theta` envelope to sub-pixel width, leaving a single line of pixels through
the origin. Three divergences from the reference are deliberate and documented:
the scale parameters default to kernel-relative ranges, `lambda` is swept rather
than held constant, and the sweep is factorized rather than diagonal. Pass explicit
ranges with `sweep="diagonal"` to recover the reference behaviour.

### Usage

Two builders are provided. `create_gabor_conv2d` is the paper's own use case — a
**trainable** cross-channel warm start. `create_gabor_depthwise_conv2d` applies the
bank **per channel** (depthwise, no cross-channel mixing): each of
`filters_per_channel` Gabor filters is applied independently to every input
channel, so the output has `in_channels * filters_per_channel` channels; follow it
with a `1x1` `Conv2D` projection for a specific output width. It defaults to
`trainable=False` (a frozen front-end).

```python
import keras
from dl_techniques.initializers import (
    GaborFiltersInitializer,
    create_gabor_conv2d,
    create_gabor_depthwise_conv2d,
)

# -- Method 1: Direct Initializer Usage --
# The deterministic Gabor bank fills any Conv2D/DepthwiseConv2D kernel whose
# LAST axis is the filter count; it is replicated across the `in` axis.
gabor_conv = keras.layers.DepthwiseConv2D(
    kernel_size=11,
    depth_multiplier=96,           # 96 Gabor filters PER input channel
    padding='same',
    depthwise_initializer=GaborFiltersInitializer(),
    trainable=False,               # frozen orientation/frequency front-end
)
# Input (32, 32, 3) -> Output (32, 32, 3 * 96 = 288)

# -- Method 2: Trainable Conv2D warm start (the paper's use case) --
warm_start = create_gabor_conv2d(filters=96, kernel_size=11)
# Input (32, 32, 3) -> Output (32, 32, 96), trainable=True

# -- Method 3: Frozen per-channel front-end (output = in * filters) --
gabor_layer = create_gabor_depthwise_conv2d(
    filters_per_channel=96,        # `filters` is a deprecated alias
    kernel_size=11,
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