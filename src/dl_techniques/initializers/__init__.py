"""Custom Keras weight initializers.

Each initializer targets a specific weight geometry: orthonormal codebooks,
hypersphere-constrained vectors, fixed filter banks, exact per-vector norms, or
the per-role variances a KAN edge needs. Read the class docstring for the shape
contract before wiring one into a layer, since several accept only one rank.

Available initializers:

-   ``OrthonormalInitializer``: weight matrices with orthonormal rows,
    optionally scaled by ``gain``. Preserves signal norm and mitigates gradient
    issues. Shape is ``(n_clusters, feature_dims)``, so on a Dense kernel the
    constraint reads ``input_dim <= units``. It is a codebook/centroid
    initializer, not a general Dense or Conv2D one.
-   ``HeOrthonormalInitializer``: He-normal variance scaling followed by
    orthonormalization, giving a well-conditioned start for ReLU networks.
-   ``OrthogonalHypersphereInitializer``: weight vectors on a hypersphere of a
    given radius, mutually orthogonal where the shape allows it. Beyond
    ``latent_dim`` vectors it stacks independent orthonormal bases into a tight
    frame, which is the ordinary case for the narrowing projections of its three
    default consumers.
-   ``HaarWaveletInitializer``: the fixed orthonormal 2D Haar wavelet filter
    bank, every tap +/- 0.5, for energy-preserving multi-resolution feature
    extraction. Output slot ``j`` holds sub-band ``j % 4`` for every input
    channel. Paired with ``create_haar_depthwise_conv2d``.
-   ``GaborFiltersInitializer``: deterministic Gabor filter-bank initialization
    over a factorized orientation x scale x phase sweep (Ozbulak & Ekenel),
    DC-removed and energy-normalized to a He-like scale by default. Paired with
    ``create_gabor_conv2d`` for a trainable cross-channel warm start and
    ``create_gabor_depthwise_conv2d``, which applies the bank per channel with
    no cross-channel mixing, giving ``in_channels * filters_per_channel``
    outputs.
-   ``PolarInitializer``: samples in polar coordinates, giving an exact
    per-vector L2 norm with a uniform-on-sphere direction (PolarQuant Lemma 2),
    which enables equinorm initialization. The norm is taken over the fan-in
    block, every axis but the last, so the He-equivalent target is correct for
    Conv2D kernels as well as Dense ones.
-   ``LinearUpInitializer``: THERA heat-field frequency init. 2D frequency
    vectors drawn uniformly over a disk of radius ``pi*scale``
    (``r = pi*scale*sqrt(U)``), producing a ``(2, N)`` x/y-row matrix for
    SIREN-style neural heat fields.
-   ``IdentityPlusNoise``: ``eye(H) + RandomNormal(stddev, seed)`` for square
    2-D matrices only. A near-identity start for a mixing or coupling matrix
    that should begin as close to a no-op; ``stddev=0`` gives the exact
    identity. Used by ``WaveFieldAttention``'s cross-head ``field_coupling``
    matrix.
-   ``clone_initializer``: returns an independent copy of an initializer. A
    single seedless initializer instance reused across several weights emits the
    same tensor at every matching shape, which is Keras 3 behaviour: the
    instance self-assigns a seed. That is how ``PowerMLPLayer``'s two branches
    came to start bit-identical. Clone per site where the two weights play
    different architectural roles.
-   ``KANInitializer``: variance-controlled init for Kolmogorov-Arnold Network
    residual (``base_scaler``) and spline (``spline_weight``) roles, using the
    Rigas et al. per-role variance schemes (``power_law``, ``glorot_inspired``,
    ``baseline``) from arXiv:2509.03417, with exactly-integrated activation and
    B-spline expectation constants. None of the three schemes is unit-gain, and
    only ``glorot_inspired`` is width-independent; call
    ``expected_forward_gain`` rather than assuming depth stability. Paired via
    ``create_kan_initializers``.
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
