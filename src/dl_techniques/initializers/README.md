# Initializers Module

The `dl_techniques.initializers` module provides a collection of advanced weight initializers for Keras, focusing on geometric and statistical properties to improve training stability, accelerate convergence, and encourage feature diversity in deep neural networks.

## Overview

This module offers nine specialized initializers that go beyond standard random distributions (the table below covers the six with a dedicated section here; `LinearUpInitializer`, `IdentityPlusNoise` and `KANInitializer` are described in `CLAUDE.md`, which lists the complete public surface). They leverage principles from linear algebra and signal processing—such as orthogonality, wavelet theory, and polar/hyperspherical geometry—to construct weight matrices with desirable mathematical properties from the start of training. All initializers are implemented as standard Keras `Initializer` subclasses, supporting full serialization and seamless integration into any Keras model.

## Available Initializers

| Name | Class | Description | Use Case |
|------|-------|-------------|----------|
| `orthonormal` | `OrthonormalInitializer` | Generates a set of mutually orthogonal vectors with unit norm (orthonormal) via QR decomposition. | Stabilizing training and mitigating vanishing/exploding gradients in deep networks. |
| `he_orthonormal` | `HeOrthonormalInitializer`| Combines He normal seeding with QR decomposition to produce an orthonormal matrix. | Orthonormal initialization where the underlying random source is scaled for ReLU-based architectures. |
| `hypersphere_orthogonal` | `OrthogonalHypersphereInitializer` | Creates vectors on a hypersphere of a specified radius, mutually orthogonal where possible; beyond `latent_dim` vectors it stacks independent orthonormal bases (a tight frame) rather than degrading to uniform sampling. | Maximizing initial feature diversity for embeddings, attention heads, or mixture-of-experts models. |
| `haar_wavelet` | `HaarWaveletInitializer` | Deterministically creates the fixed, orthonormal 2x2 filter bank of the 2D Haar wavelet decomposition (every tap +/- 0.5); output slot `j` is sub-band `j % 4` for every input channel. | Building non-trainable, engineered feature extractors for multi-resolution analysis in CNNs. |
| `polar` | `PolarInitializer` | Sets each fan-in vector (every axis but the last by default, so He-correct for Dense AND Conv2D) to an exact L2 norm with a uniform-on-sphere direction. | Equinorm / magnitude-controlled, well-conditioned initialization where chi-distributed Gaussian norms are undesirable. |
| `gabor_filters` | `GaborFiltersInitializer` | Deterministically fills a convolution kernel with a bank of Gabor filters over a factorized orientation x scale x phase sweep (Ozbulak-Ekenel), DC-removed and energy-normalized to a He-like scale. | Pre-training-free transfer learning by initializing the first convolutional layer with edge/texture-selective low-level features. |

## Orthonormal Initializer

Generates a weight matrix where each row is a unit vector orthogonal to every other row, by applying
QR decomposition to a random Gaussian matrix. Such matrices preserve signal norm (isometry), which
helps prevent gradients from vanishing or exploding during backpropagation.

**Orientation — read this before using it on a `Dense` layer.** The shape is
`(n_clusters, feature_dims)`: the FIRST axis counts the vectors and the LAST is the space they live
in. A Keras `Dense` kernel is `(input_dim, units)`, so the constraint is **`input_dim <= units`** —
only a *widening* layer qualifies. This is a codebook / centroid initializer whose contract is
orthonormal **rows**; for a narrowing `Dense` or any `Conv2D`, use `keras.initializers.Orthogonal`
or `OrthogonalHypersphereInitializer` instead.

**Mathematical constraint:** it is impossible to create more than `d` orthogonal vectors in a
`d`-dimensional space, so `n_clusters > feature_dims` raises `ValueError`. So does a non-2D shape:
a 4-D convolution kernel raises rather than being silently reinterpreted, which
`tests/test_layers/test_convnext_v1_block.py` pins as a contract.

**Sign convention:** `Q *= sign(diag(R))`, the unique factorization with a positive `R` diagonal —
the same convention `HeOrthonormalInitializer` and `OrthogonalHypersphereInitializer` use, and the
one `keras.initializers.Orthogonal` uses. It canonicalizes across LAPACK / cuSOLVER / JAX builds
*and* preserves the Haar distribution. It replaced a convention keyed on `Q`'s first row, which was
equally deterministic but folded that row into the positive orthant on every draw: measured over
4000 seeds at `d=64`, `P(any entry of row 0 < 0)` was `0.000` and the mean cosine of row 0 to the
all-ones direction was **`0.801`**.

### Usage

```python
import keras
from dl_techniques.initializers import OrthonormalInitializer

# A codebook: 64 orthonormal vectors in a 128-dimensional space
initializer = OrthonormalInitializer(seed=42)
weights = initializer(shape=(64, 128))

# gain scales the rows; sqrt(2) is the conventional choice for a ReLU stack.
scaled = OrthonormalInitializer(gain=2.0 ** 0.5, seed=42)(shape=(64, 128))

# In a Dense layer the kernel is (input_dim, units), so only a WIDENING layer fits:
widening = keras.layers.Dense(units=128, kernel_initializer=initializer)
widening.build((None, 64))     # kernel (64, 128): 64 <= 128, fine

# narrowing = keras.layers.Dense(units=64, kernel_initializer=initializer)
# narrowing.build((None, 128)) # kernel (128, 64): raises ValueError
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

This initializer creates weight vectors that lie exactly on the surface of a hypersphere with a
specified `radius` and are mutually orthogonal wherever orthogonality is available.

The flattened matrix is `(num_vectors, latent_dim)` with `num_vectors = prod(shape[:-1])` and
`latent_dim = shape[-1]` — the same convention `keras.initializers.Orthogonal` uses. **For a Dense
kernel `(input_dim, units)` that means `num_vectors = input_dim` and `latent_dim = units`: the
initializer does NOT transpose.** A narrowing projection (`units < input_dim`) therefore lands in
the second regime below, as does effectively every `Conv2D` kernel.

**Behavior modes:**
1. **Orthogonal (`num_vectors <= latent_dim`):** a perfectly orthogonal set from a sign-corrected
   QR decomposition, each vector scaled to `radius`.
2. **Tight frame (`num_vectors > latent_dim`):** at most `latent_dim` rows can be mutually
   orthogonal, so `ceil(num_vectors / latent_dim)` **independent orthonormal bases** are stacked and
   truncated. Every vector still has norm exactly `radius`, and `1 / ceil(num_vectors / latent_dim)`
   of all pairs are still exactly orthogonal.

The second regime used to sample uniformly on the sphere and emit a `UserWarning` calling the
request "mathematically impossible". It is not impossible — only *all rows* being orthogonal is —
and the uniform construction was measurably worse. At `(512, 128)`:

| construction | max abs cos | mean abs cos | `cond(W)` | exactly-orthogonal pairs |
|---|---|---|---|---|
| uniform (previous) | 0.371 | 0.0706 | 2.92 | 0.2% |
| stacked bases (now) | 0.386 | 0.0530 | **1.0000** | **25.0%** |
| Welch lower bound | 0.0766 | n/a | n/a | n/a |

Neither is a good spherical code — both sit far above the Welch bound — but the uniform version's
singular values spanned 1.03 to 3.01, discarding the dynamical isometry that is the reason to reach
for orthogonal initialization at all. Pass `fallback='uniform'` to restore the old construction.

**Versus `keras.initializers.Orthogonal`:** for `num_vectors > latent_dim` Keras orthogonalizes the
other axis and returns orthonormal *columns*, giving `cond == 1` at any shape. That is the better
tool when a perfectly conditioned map is what you want; it leaves the row norms unequal, so the
vectors no longer share a hypersphere. This class keeps `||v_i|| == radius` as its invariant.

Note that for a `Conv2D` kernel the flattening makes a "vector" span `(kh, kw, in_ch)`, so the
separated set is not the per-filter weight vectors.

### Usage

```python
import keras
from dl_techniques.initializers import OrthogonalHypersphereInitializer

init = OrthogonalHypersphereInitializer(radius=1.5, seed=42)

# Orthogonal regime: a Dense kernel (128, 512) is 128 vectors in 512-D.
widening = keras.layers.Dense(units=512, kernel_initializer=init)   # input_dim 128

# Tight-frame regime: a Dense kernel (256, 64) is 256 vectors in 64-D, so the
# bank is 4 stacked orthonormal bases -- 25% of pairs exactly orthogonal, and
# every vector still on the hypersphere of radius 1.5.
narrowing = keras.layers.Dense(units=64, kernel_initializer=init)   # input_dim 256

# An embedding matrix (1000, 256) is likewise 1000 vectors in 256-D: 4 bases.
embedding = keras.layers.Embedding(
    input_dim=1000, output_dim=256, embeddings_initializer=init,
)
```

## Haar Wavelet Initializer

This is a deterministic initializer that populates a 2x2 convolutional kernel with the four basis
filters of the 2D Haar wavelet transform. It is designed to create a non-trainable layer that
performs a single level of multi-resolution analysis, separating an input into an approximation
band and three detail bands.

The 1D orthonormal Haar pair is `(a + b)/sqrt(2)` and `(a - b)/sqrt(2)`; applying it separably on
both axes gives four 2x2 filters in which **every tap has magnitude 0.5**:

| Slot | Name | Filter | Differences along |
|---|---|---|---|
| 0 | `LL` | `[[0.5, 0.5], [0.5, 0.5]]` | nothing (approximation) |
| 1 | `LH` | `[[0.5, -0.5], [0.5, -0.5]]` | width (responds to vertical edges) |
| 2 | `HL` | `[[0.5, 0.5], [-0.5, -0.5]]` | height (responds to horizontal edges) |
| 3 | `HH` | `[[0.5, -0.5], [-0.5, 0.5]]` | both (diagonal detail) |

With `scale=1.0` these are **orthonormal**: the Gram matrix is the identity, so the transform
preserves energy exactly, every sub-band has the same variance as the input, and it is inverted by
its own transpose. `scale != 1.0` keeps them orthogonal but multiplies energy by `scale**2`.

**Sub-band labels are library-dependent** — the "differences along" column is the contract, since
other wavelet libraries attach the words *horizontal* and *vertical* to the opposite band. And
because Keras convolution is cross-correlation with no kernel flip, the three detail bands carry
the opposite **sign** to a true convolution against the same filters; this matters only when
comparing against an external reference.

### Layout

Output slot `j` holds sub-band `j % 4` for **every** input channel, so the sub-band of a slot is a
property of `j` alone. `DepthwiseConv2D` orders its output channels input-channel-major, so output
channel `i * channel_multiplier + j` is sub-band `j % 4` of input channel `i`. A
`channel_multiplier` above 4 cycles the bank and therefore duplicates filters (warned about).

### Usage

A builder utility, `create_haar_depthwise_conv2d`, is provided for convenience. It rejects odd
spatial dimensions: a stride-2 `'valid'` convolution consumes whole 2x2 blocks, so an odd size
would silently drop the last row/column and break perfect reconstruction.

```python
import keras
from dl_techniques.initializers import HaarWaveletInitializer, create_haar_depthwise_conv2d

# -- Method 1: Using the Builder Utility (Recommended) --
# A DepthwiseConv2D pre-configured for per-channel wavelet decomposition.
haar_layer = create_haar_depthwise_conv2d(
    input_shape=(256, 256, 3),
    channel_multiplier=4,   # LL, LH, HL, HH per input channel
    trainable=False,        # wavelet filters are typically fixed
    name='haar_wavelet_decomposition',
)
# Input (B, 256, 256, 3) -> Output (B, 128, 128, 12), channel 4*i + j = sub-band j of input i

# -- Method 2: Direct Initializer Usage --
# On a DepthwiseConv2D the bank stays per channel. On a Conv2D, output j sums over the
# input channels, so a Conv2D initialized this way computes the Haar sub-band j % 4 of
# the unweighted SUM of its input channels -- rarely what you want.
haar_conv = keras.layers.DepthwiseConv2D(
    kernel_size=2,
    strides=2,
    padding='valid',
    depth_multiplier=4,
    depthwise_initializer=HaarWaveletInitializer(),
    trainable=False,
)
```

Note that a `kernel_regularizer` on a frozen layer is a **silent no-op**: Keras collects no
regularization loss from a non-trainable weight (measured: 0 loss terms frozen, 1 trainable). The
builder warns when both are set.

## Polar Initializer

Samples weights "in polar coordinates": every fan-in vector is given an
**exact** L2 norm with a direction drawn **uniformly on the unit sphere**. By
PolarQuant's Lemma 2, a Gaussian vector's direction is exactly uniform on the
sphere, so this is realized by normalizing a Gaussian and rescaling to the
target norm — for any shape, power-of-two or not.

Unlike He/Glorot/Gaussian sampling (whose per-vector norms are chi-distributed),
`PolarInitializer` gives every vector an identical norm — useful for "equinorm"
initialization and precise magnitude control.

### Which axes form a vector

`axis=None` (the default) means **every axis except the last**, i.e. the fan-in
block of each output unit: axis `0` for a `Dense` kernel `(fan_in, units)` and
axes `(0, 1, 2)` for a `Conv2D` kernel `(kh, kw, in_ch, out_ch)`. He variance is
defined over the whole fan-in, so normalizing a single axis of a conv kernel is
**not** He-equivalent — measured on `(3, 3, 64, 128)` with `axis=0`, each output
unit accumulated a fan-in energy of `384.0` instead of `2.0` (192x) and the
per-element std came out 13.9x above He's `sqrt(2/576)`, which compounds to
~2.6e11 over ten such layers. Pass an int or a tuple of ints only when you want
something other than the fan-in block.

### What it does and does not give you

It guarantees equal fan-in norms. It does **not** give dynamical isometry:
fixing the norms controls only the diagonal of `WᵀW`, leaving the singular-value
spectrum essentially Marchenko-Pastur, the same as Gaussian init — use
`keras.initializers.Orthogonal` for that.

The benefit also shrinks as `1/sqrt(2·fan_in)`. The relative spread of He-normal
column norms measures **17.9%** at `fan_in=16`, 8.9% at 64, 3.1% at 512 and 1.1%
at 4096; equinorm init removes exactly that spread, so it differs materially
from He only for narrow fan-ins.

"Exact" means exact to the compute dtype: max deviation from the target measures
`4.4e-16` in float64, `3.0e-07` in float32 and `2.6e-05` after a float16 cast, so
under a `mixed_float16` policy the guarantee is two orders of magnitude looser.

**Arguments:** `norm` (target L2 norm, must be positive; `None` => `sqrt(2)`, the
He-normal energy), `axis` (`None` = the fan-in block; or an int / tuple of ints),
`gain` (positive; it multiplies the target, so `gain=2.0, norm=None` targets
`2*sqrt(2)` — four times He's energy, not twice), `seed`.

Seeding follows the Keras contract: an initializer **instance** replays the same
tensor at every matching shape whether or not a seed was given, and a seedless
instance resolves its seed from the global RNG state so `keras.utils.set_random_seed`
controls it. Use `clone_initializer` when two weights must start differently.

### Usage

```python
import keras
from dl_techniques.initializers import PolarInitializer

# Every output unit's weight vector starts with L2 norm exactly 1.0
layer = keras.layers.Dense(128, kernel_initializer=PolarInitializer(norm=1.0))

# Each of the 64 filters starts with the He fan-in energy of 2.0
conv = keras.layers.Conv2D(64, 3, kernel_initializer=PolarInitializer())
```

It is the thematic companion of `PolarWeightNorm` (see the module docstring of
`dl_techniques/layers/norms/polar_weight_norm.py`) — the layer uses PolarQuant's
Definition 1 / Algorithm 1, this initializer uses Lemma 2. There is no code
dependency in either direction: `PolarWeightNorm` defaults to `"glorot_uniform"`.
The two agree on the Dense convention, since the layer's exact per-unit norm is
also a reduction over `fan_in`.

The closest prior art for the scheme itself is Salimans & Kingma's *Weight
Normalization* (2016), which separates a magnitude from a direction for the same
reason; PolarQuant is a KV-cache quantization paper and supports the
decomposition, not the initialization scheme.

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
    # NOTE both of these are WIDENING layers on purpose: a Dense kernel is
    # (input_dim, units), and both classes require input_dim <= units. A
    # narrowing layer raises ValueError -- an earlier version of this example
    # narrowed 784 -> 256 -> 128 and could not run.
    keras.layers.Dense(
        1024,
        # Use HeOrthonormal for a ReLU-based network
        kernel_initializer=HeOrthonormalInitializer(seed=1)
    ),
    keras.layers.ReLU(),
    keras.layers.Dense(
        2048,
        # Use standard Orthonormal for subsequent layers
        kernel_initializer=OrthonormalInitializer(seed=2)
    ),
    keras.layers.ReLU(),
    keras.layers.Dense(
        10,
        # Use Hypersphere to encourage diverse features before softmax. This
        # kernel is (128, 10) -- 128 vectors in 10-D -- so it is the tight-frame
        # regime: 13 stacked bases, every row still of norm 1.2.
        kernel_initializer=OrthogonalHypersphereInitializer(radius=1.2, seed=3)
    ),
    keras.layers.Softmax()
])

model.summary()
```