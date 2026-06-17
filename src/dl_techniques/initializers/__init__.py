"""Advanced Keras Initializers.

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
-   `HaarWaveletInitializer`: Constructs fixed 2D Haar wavelet filters for
    use in convolutional layers, enabling multi-resolution feature extraction.
-   `GaborFiltersInitializer`: Deterministic Gabor filter-bank initialization for
    the first Conv2D layer, sweeping Table I parameters (Ozbulak & Ekenel).
-   `PolarInitializer`: Samples in polar coordinates -- exact per-vector L2 norm
    with a uniform-on-sphere direction (PolarQuant Lemma 2), enabling equinorm
    initialization.
-   `LinearUpInitializer`: THERA heat-field frequency init -- 2D frequency vectors
    drawn uniformly over a disk of radius `pi*scale` (`r = pi*scale*sqrt(U)`),
    producing a `(2, N)` x/y-row matrix for SIREN-style neural heat fields.
-   `KANInitializer`: variance-controlled init for Kolmogorov-Arnold Network
    residual (`base_scaler`) and spline (`spline_weight`) roles, using the
    Rigas et al. (2026) per-role variance schemes (`power_law`,
    `glorot_inspired`, `baseline`); paired via `create_kan_initializers`.
"""

from .haar_wavelet_initializer import (
    HaarWaveletInitializer,
    create_haar_depthwise_conv2d,
)
from .gabor_filters_initializer import GaborFiltersInitializer
from .he_orthonormal_initializer import HeOrthonormalInitializer
from .orthonormal_initializer import OrthonormalInitializer
from .hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer
from .polar_initializer import PolarInitializer
from .linear_up_initializer import LinearUpInitializer
from .kan_initializer import (
    KANInitializer,
    create_kan_initializers,
)

__all__ = [
    "HaarWaveletInitializer",
    "create_haar_depthwise_conv2d",
    "GaborFiltersInitializer",
    "HeOrthonormalInitializer",
    "OrthonormalInitializer",
    "OrthogonalHypersphereInitializer",
    "PolarInitializer",
    "LinearUpInitializer",
    "KANInitializer",
    "create_kan_initializers",
]