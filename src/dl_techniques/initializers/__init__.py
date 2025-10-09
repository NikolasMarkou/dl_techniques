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
"""

from .haar_wavelet_initializer import (
    HaarWaveletInitializer,
    create_haar_depthwise_conv2d,
)
from .he_orthonormal_initializer import HeOrthonormalInitializer
from .orthonormal_initializer import OrthonormalInitializer
from .hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer

__all__ = [
    "HaarWaveletInitializer",
    "create_haar_depthwise_conv2d",
    "HeOrthonormalInitializer",
    "OrthonormalInitializer",
    "OrthogonalHypersphereInitializer",
]