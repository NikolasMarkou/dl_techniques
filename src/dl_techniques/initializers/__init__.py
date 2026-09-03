"""
Advanced Keras Initializers.

This module provides a collection of advanced weight initializers for Keras,
designed to improve training stability and model performance in various
deep learning architectures. These initializers are based on established
mathematical principles from linear algebra, signal processing, and statistical
learning theory.

Available Initializers:
-   `OrthonormalInitializer`: Generates weight matrices with orthonormal rows,
    ideal for preserving signal norm and mitigating gradient issues.
-   `HeOrthonormalInitializer`: Combines He normal variance scaling with
    orthonormalization, providing a well-conditioned starting point for
    ReLU-based networks.
-   `OrthogonalHypersphereInitializer`: Creates mutually orthogonal weight
    vectors that lie on a hypersphere of a given radius, maximizing initial
    geometric separation.
-   `HaarWaveletInitializer`: Constructs the fixed, orthonormal 2D Haar wavelet
    filter bank (every tap +/- 0.5) for use in convolutional layers, enabling
    energy-preserving multi-resolution feature extraction; output slot `j` holds
    sub-band `j % 4` for every input channel.
-   `GaborFiltersInitializer`: Deterministic Gabor filter-bank initialization
    over a factorized orientation x scale x phase sweep (Ozbulak & Ekenel),
    DC-removed and energy-normalized to a He-like scale by default; paired with
    `create_gabor_conv2d` (cross-channel warm start, trainable) and
    `create_gabor_depthwise_conv2d`, which applies the bank PER CHANNEL
    (depthwise, no cross-channel mixing; output =
    `in_channels * filters_per_channel`).
-   `PolarInitializer`: Samples in polar coordinates -- exact per-vector L2 norm
    with a uniform-on-sphere direction (PolarQuant Lemma 2), enabling equinorm
    initialization. By default the norm is taken over the fan-in block (every
    axis but the last), so the He-equivalent target is correct for Conv2D
    kernels as well as Dense ones.
-   `LinearUpInitializer`: THERA heat-field frequency init -- 2D frequency vectors
    drawn uniformly over a disk of radius `pi*scale` (`r = pi*scale*sqrt(U)`),
    producing a `(2, N)` x/y-row matrix for SIREN-style neural heat fields.
-   `IdentityPlusNoise`: `eye(H) + RandomNormal(stddev, seed)` for SQUARE 2-D
    matrices only -- a near-identity start for a mixing/coupling matrix that is
    meant to begin as (close to) a no-op. `stddev=0` gives the exact identity.
    Used by `WaveFieldAttention`'s cross-head `field_coupling` matrix.
-   `clone_initializer`: returns an INDEPENDENT copy of an initializer. A single
    seedless initializer INSTANCE reused across several weights emits the SAME
    tensor at every matching shape (Keras 3 behaviour -- the instance
    self-assigns a seed), which is how `PowerMLPLayer`'s two branches came to
    start bit-identical. Clone per site where the two weights play DIFFERENT
    architectural roles.
-   `KANInitializer`: variance-controlled init for Kolmogorov-Arnold Network
    residual (`base_scaler`) and spline (`spline_weight`) roles, using the
    Rigas et al. per-role variance schemes (`power_law`, `glorot_inspired`,
    `baseline`) from arXiv:2509.03417, with exactly-integrated activation and
    B-spline expectation constants. None of the three schemes is unit-gain --
    only `glorot_inspired` is width-independent; call `expected_forward_gain`
    instead of assuming depth stability. Paired via `create_kan_initializers`.
"""

from .haar_wavelet_initializer import (
    HaarWaveletInitializer,
    create_haar_depthwise_conv2d,
)
from .gabor_filters_initializer import (
    GaborFiltersInitializer,
    create_gabor_conv2d,
    create_gabor_depthwise_conv2d,
)
from .he_orthonormal_initializer import HeOrthonormalInitializer
from .orthonormal_initializer import OrthonormalInitializer
from .hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer
from .polar_initializer import PolarInitializer
from .linear_up_initializer import LinearUpInitializer
from .identity_plus_noise import IdentityPlusNoise
from .clone import clone_initializer
from .kan_initializer import (
    KANInitializer,
    create_kan_initializers,
)

__all__ = [
    "clone_initializer",
    "HaarWaveletInitializer",
    "create_haar_depthwise_conv2d",
    "GaborFiltersInitializer",
    "create_gabor_conv2d",
    "create_gabor_depthwise_conv2d",
    "HeOrthonormalInitializer",
    "OrthonormalInitializer",
    "OrthogonalHypersphereInitializer",
    "PolarInitializer",
    "LinearUpInitializer",
    "IdentityPlusNoise",
    "KANInitializer",
    "create_kan_initializers",
]