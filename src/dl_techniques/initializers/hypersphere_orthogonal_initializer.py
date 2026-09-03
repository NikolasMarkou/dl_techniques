"""
Initialize weights as orthogonal vectors on a hypersphere.

This initializer populates a weight tensor with vectors that lie exactly on the
surface of a hypersphere of a specified radius and are mutually orthogonal
wherever orthogonality is available. Its purpose is to establish maximal
geometric separation between initial weight vectors, encouraging feature
diversity from the start.

Architecture and Mathematical Foundations:
The flattened weight matrix is ``(num_vectors, latent_dim)`` with
``num_vectors = prod(shape[:-1])`` and ``latent_dim = shape[-1]`` -- the same
convention ``keras.initializers.Orthogonal`` uses. At most ``latent_dim``
vectors in ``latent_dim`` dimensions can be mutually orthogonal, which gives two
regimes.

1.  **Orthogonal regime** (``num_vectors <= latent_dim``): a random
    ``(latent_dim, num_vectors)`` matrix ``A`` is factorized ``A = QR``; ``Q``
    has orthonormal columns, so ``Q.T`` has orthonormal rows. The vectors
    satisfy ``v_i . v_j = 0`` and ``||v_i|| = radius`` to float32 precision
    (measured residuals ~6e-08 at ``(64, 128)``).

2.  **Tight-frame regime** (``num_vectors > latent_dim``): no ``num_vectors``
    rows can be mutually orthogonal, but the set does not have to be given up
    on. ``ceil(num_vectors / latent_dim)`` INDEPENDENT orthonormal bases are
    stacked and truncated to ``num_vectors`` rows. Every row still has norm
    exactly ``radius`` and a large fraction of all pairs are still EXACTLY
    orthogonal.

Why the second regime is not "fall back to random"
---------------------------------------------------
It previously sampled uniformly on the sphere, described as maximizing "average
angular separation". That property does not exist: the average pairwise cosine
is zero for any antipodally balanced configuration, so uniform sampling
maximizes nothing, and maximizing the MINIMUM separation is the Tammes /
spherical-code problem, where uniform random is a poor solution. Measured at
``(512, 128)``:

| construction        | max abs cos | mean abs cos | cond(W) | exactly-orthogonal pairs |
|---------------------|-------------|--------------|---------|--------------------------|
| uniform (previous)  | 0.371       | 0.0706       | 2.92    | 0.2%                     |
| stacked bases (now) | 0.386       | 0.0530       | 1.0000  | 25.0%                    |
| Welch lower bound   | 0.0766      | n/a          | n/a     | n/a                      |

Both sit far above the Welch bound, so neither is a good spherical code; the
difference that matters is conditioning. The uniform construction's singular
values spanned 1.03 to 3.01, discarding the dynamical isometry that is the whole
reason the Saxe reference is cited. Stacked bases are an exact tight frame when
``latent_dim`` divides ``num_vectors`` (``cond == 1``) and are bounded by
``sqrt(2)`` otherwise -- measured 1.0000 at ``(512, 128)``, 1.118 at
``(576, 128)``, 1.0086 at ``(30000, 512)``.

Set ``fallback='uniform'`` to restore the previous behaviour.

Relationship to ``keras.initializers.Orthogonal``
--------------------------------------------------
For ``num_vectors > latent_dim`` Keras orthogonalizes the OTHER axis, returning
orthonormal COLUMNS (``cond == 1`` exactly, at any shape). That is the right
tool when a perfectly conditioned map is what you want. It is not what this
class promises: orthonormal columns leave the row norms unequal, so the vectors
no longer lie on a common hypersphere. This class keeps ``||v_i|| == radius`` as
its invariant and buys back conditioning within that constraint.

Note on convolutional kernels: the flattening makes a "vector" indexed by
``(kh, kw, in_ch)`` and the vector space the output-channel axis, so the
separated set is not the per-filter weight vectors. That is Keras's convention
too, but the "expert specialization" framing below describes a Dense or
embedding kernel, not a conv one.

References:
    - For orthogonal initialization benefits in deep learning:
      Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). *Exact
      solutions to the nonlinear dynamics of learning in deep linear
      neural networks*.
    - For the method of generating uniform points on a sphere (the
      ``fallback='uniform'`` path):
      Marsaglia, G. (1972). *Choosing a Point from the Surface of a
      Sphere*. The Annals of Mathematical Statistics.
    - For the coherence bound quoted above:
      Welch, L. R. (1974). *Lower bounds on the maximum cross correlation of
      signals*. IEEE Transactions on Information Theory.

"""

import keras
import numpy as np
from typing import Optional, Sequence, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

#: Accepted values for the ``fallback`` argument (used when num_vectors > latent_dim).
FALLBACK_MODES = ("block_orthogonal", "uniform")

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.initializers.hypersphere_orthogonal_initializer")
class OrthogonalHypersphereInitializer(keras.initializers.Initializer):
    """
    Orthogonal hypersphere weight initializer with mathematical dimensionality constraints.

    Creates weight vectors that lie exactly on a hypersphere of the given radius
    and are mutually orthogonal wherever orthogonality is available. The
    flattened matrix is ``(num_vectors, latent_dim)`` with
    ``num_vectors = prod(shape[:-1])``.

    **Intent**: Provide geometrically well-separated initial weights for neural
    networks where weight diversity is crucial, such as in mixture models,
    attention mechanisms, or embedding layers.

    **Mathematical behavior**:

    1. ``num_vectors <= latent_dim``: perfectly orthogonal vectors from a
       sign-corrected QR decomposition. ``||v_i|| = radius`` and
       ``v_i . v_j = 0``, to float32 precision.
    2. ``num_vectors > latent_dim``: ``ceil(num_vectors / latent_dim)``
       independent orthonormal bases stacked and truncated. Every row still has
       norm exactly ``radius``, ``1 / ceil(num_vectors / latent_dim)`` of all
       pairs are still exactly orthogonal, and the result is a tight (or nearly
       tight) frame -- see the module docstring for the measured comparison
       against the uniform sampling this replaced.

    **Geometric properties**:
    - All vectors lie exactly on the hypersphere: ``||v|| = radius``
    - Maximum angular separation when orthogonal
    - Deterministic given the same seed (reproducible)

    Args:
        radius: Float, radius of the hypersphere. All initialized vectors will
            have this L2 norm. Must be positive and finite. Defaults to 1.0.
        fallback: Construction used when ``num_vectors > latent_dim``. One of
            :data:`FALLBACK_MODES`: ``'block_orthogonal'`` (default) stacks
            independent orthonormal bases; ``'uniform'`` samples uniformly on the
            sphere, the previous behaviour, which is worse conditioned.
        seed: Optional integer seed. Following the Keras contract an instance
            replays the same tensor at every matching shape whether or not a seed
            is given; a seedless instance resolves one from the global RNG state,
            so ``keras.utils.set_random_seed`` controls it. Use
            ``dl_techniques.initializers.clone_initializer`` when two weights must
            start differently.

    Raises:
        ValueError: If ``radius`` is not positive and finite, if ``fallback`` is
            not a member of :data:`FALLBACK_MODES`, if the shape has fewer than
            two dimensions, or if any dimension is not positive.

    Example:
        ```python
        # Orthogonal vectors on the unit hypersphere
        initializer = OrthogonalHypersphereInitializer()
        weights = initializer(shape=(10, 128))   # 10 orthogonal vectors in 128D

        # Custom radius and seed
        initializer = OrthogonalHypersphereInitializer(radius=2.0, seed=42)
        weights = initializer(shape=(64, 256))   # orthogonal: 64 <= 256

        # More vectors than dimensions: stacked orthonormal bases
        initializer = OrthogonalHypersphereInitializer(radius=1.5)
        weights = initializer(shape=(512, 128))  # 4 bases, 25% of pairs orthogonal

        layer = keras.layers.Dense(
            units=64,
            kernel_initializer=OrthogonalHypersphereInitializer(radius=1.5),
        )
        ```

    Note:
        This initializer is particularly useful for:
        - Embedding layers where diverse initial vectors improve learning
        - Attention mechanisms requiring well-separated query/key vectors
        - Mixture of experts where expert specialization benefits from orthogonality

        For a ``Conv2D`` kernel the flattening makes a "vector" span
        ``(kh, kw, in_ch)``, so the separated set is not the per-filter weight
        vectors; see the module docstring.

    Mathematical Background:
        The maximum number of mutually orthogonal vectors in n-dimensional space
        is n. Beyond that, stacking independent orthonormal bases keeps every
        vector on the hypersphere AND keeps a fixed fraction of pairs exactly
        orthogonal, which uniform sampling does not.
    """

    def __init__(
            self,
            radius: float = 1.0,
            fallback: str = "block_orthogonal",
            seed: Optional[int] = None,
    ) -> None:
        if not np.isfinite(radius):
            raise ValueError(f"radius must be finite, got {radius}")
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if fallback not in FALLBACK_MODES:
            raise ValueError(
                f"fallback must be one of {FALLBACK_MODES}, got {fallback!r}"
            )

        self.radius = float(radius)
        self.fallback = fallback

        # Mirrors keras.initializers.RandomNormal: the config keeps whatever the
        # caller passed, the resolved seed drives the draw. np.random is seeded
        # by keras.utils.set_random_seed, so a seedless instance is reproducible
        # under a global seed -- np.random.default_rng(None) was not.
        self.seed = seed
        self._draw_seed = seed if seed is not None else int(
            np.random.randint(0, 2 ** 30)
        )

    # -----------------------------------------------------------------
    # construction helpers
    # -----------------------------------------------------------------

    def _orthonormal_rows(
        self, num_vectors: int, latent_dim: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``num_vectors`` orthonormal rows in ``latent_dim`` dimensions.

        Args:
            num_vectors: Rows to produce; must be ``<= latent_dim``.
            latent_dim: Dimensionality of the space.
            rng: Generator supplying the Gaussian seed matrix.

        Returns:
            A ``(num_vectors, latent_dim)`` array with orthonormal rows.
        """
        random_matrix = rng.normal(size=(latent_dim, num_vectors))
        q_matrix, r_matrix = np.linalg.qr(random_matrix)

        # LAPACK's Householder QR fixes the sign of R's diagonal, which makes Q
        # NON-uniform: measured over 2000 seeds at d=8, Q[0, 0] was negative in
        # 2000 of 2000 draws and E[Q[0, 0]] = -0.2897 where Haar measure gives 0.
        # Multiplying by sign(diag(R)) restores it. Matches the convention in
        # he_orthonormal_initializer.py.
        q_matrix = q_matrix * np.sign(np.diag(r_matrix))

        return q_matrix.T

    def _generate_orthogonal_set(
        self, num_vectors: int, latent_dim: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Truly orthogonal vectors (the ``num_vectors <= latent_dim`` case).

        Args:
            num_vectors: Number of orthogonal vectors to generate.
            latent_dim: Dimensionality of the latent space.
            rng: Generator supplying the Gaussian seed matrix.

        Returns:
            Array of shape ``(num_vectors, latent_dim)``, rows scaled to radius.
        """
        return self._orthonormal_rows(num_vectors, latent_dim, rng) * self.radius

    def _generate_block_orthogonal(
        self, num_vectors: int, latent_dim: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Stacked independent orthonormal bases (the ``num_vectors > latent_dim`` case).

        Args:
            num_vectors: Number of vectors to generate.
            latent_dim: Dimensionality of the latent space.
            rng: Generator supplying each block's Gaussian seed matrix.

        Returns:
            Array of shape ``(num_vectors, latent_dim)``, every row of norm
            ``radius`` and ``1 / ceil(num_vectors / latent_dim)`` of all pairs
            exactly orthogonal.
        """
        n_blocks = -(-num_vectors // latent_dim)  # ceil
        blocks = [
            self._orthonormal_rows(latent_dim, latent_dim, rng)
            for _ in range(n_blocks)
        ]
        return np.concatenate(blocks, axis=0)[:num_vectors] * self.radius

    def _generate_uniform_hypersphere(
        self, num_vectors: int, latent_dim: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Vectors uniformly distributed on the hypersphere surface.

        Retained behind ``fallback='uniform'``; it is worse conditioned than the
        stacked-bases default (measured ``cond`` 2.92 versus 1.0000 at
        ``(512, 128)``).

        Args:
            num_vectors: Number of vectors to generate.
            latent_dim: Dimensionality of the latent space.
            rng: Generator supplying the Gaussian draws.

        Returns:
            Array of shape ``(num_vectors, latent_dim)``, rows scaled to radius.
        """
        random_vectors = rng.normal(size=(num_vectors, latent_dim))
        vector_norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
        vector_norms = np.maximum(vector_norms, 1e-12)
        return (random_vectors / vector_norms) * self.radius

    # -----------------------------------------------------------------

    def __call__(
            self,
            shape: Sequence[int],
            dtype: Optional[str] = None,
            **kwargs: Any,
    ) -> Any:
        """
        Generate the orthogonal-hypersphere initialization tensor.

        Args:
            shape: Tensor shape ``(..., latent_dim)``; at least 2 dimensions.
            dtype: Data type of the returned tensor. ``None`` falls back to
                ``keras.config.floatx()``.
            **kwargs: Additional arguments (unused).

        Returns:
            Tensor of the requested shape whose flattened rows lie on the
            hypersphere of radius ``radius``.

        Raises:
            ValueError: If the shape has fewer than two dimensions or any
                dimension is not positive.
        """
        if dtype is None:
            dtype = keras.config.floatx()
        dtype = getattr(dtype, "name", None) or str(dtype)

        shape = tuple(int(d) for d in shape)
        if len(shape) < 2:
            raise ValueError(
                f"expected a shape with at least 2 dimensions -- a rank-1 shape "
                f"has no vector axis and would make the whole tensor a single "
                f"direction of norm {self.radius}, which is meaningless for a "
                f"bias. Got {shape}"
            )
        if any(d <= 0 for d in shape):
            raise ValueError(f"all dimensions must be positive, got {shape}")

        latent_dim = shape[-1]
        num_vectors = int(np.prod(shape[:-1]))

        # Set the norm in at least float32: the QR and the row norms would lose
        # two orders of magnitude of accuracy in half precision.
        compute_dtype = dtype if dtype in ("float32", "float64") else "float32"
        rng = np.random.default_rng(self._draw_seed)

        if num_vectors <= latent_dim:
            vectors = self._generate_orthogonal_set(num_vectors, latent_dim, rng)
        elif self.fallback == "block_orthogonal":
            n_blocks = -(-num_vectors // latent_dim)
            logger.debug(
                f"{num_vectors} vectors in {latent_dim} dimensions: at most "
                f"{latent_dim} rows can be mutually orthogonal, so the bank is "
                f"{n_blocks} stacked orthonormal bases -- every vector still has "
                f"norm {self.radius} and 1 in {n_blocks} pairs is still exactly "
                f"orthogonal"
            )
            vectors = self._generate_block_orthogonal(num_vectors, latent_dim, rng)
        else:
            logger.warning(
                f"{num_vectors} vectors in {latent_dim} dimensions with "
                f"fallback='uniform': sampling uniformly on the sphere. This is "
                f"measurably worse conditioned than the 'block_orthogonal' "
                f"default (cond 2.92 vs 1.0000 at (512, 128)) and leaves almost "
                f"no pair exactly orthogonal."
            )
            vectors = self._generate_uniform_hypersphere(num_vectors, latent_dim, rng)

        return keras.ops.convert_to_tensor(
            vectors.astype(compute_dtype).reshape(shape), dtype=dtype
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get initializer configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters needed to
            reconstruct this initializer instance.
        """
        config = super().get_config()
        config.update({
            "radius": self.radius,
            "fallback": self.fallback,
            "seed": self.seed,
        })
        return config

    def __repr__(self) -> str:
        """String representation of the initializer."""
        return (
            f"{self.__class__.__name__}("
            f"radius={self.radius}, fallback={self.fallback!r}, "
            f"seed={self.seed})"
        )

# ---------------------------------------------------------------------
