"""Hypersphere initializer whose vectors are orthogonal where the shape allows.

Provides :class:`OrthogonalHypersphereInitializer`, which fills a weight tensor
with vectors that lie exactly on a hypersphere of a given radius and are
mutually orthogonal wherever orthogonality is available. Use it when initial
weight vectors should be maximally separated, such as in embeddings, attention
projections, or mixture components.

The flattened weight matrix is ``(num_vectors, latent_dim)`` with
``num_vectors = prod(shape[:-1])`` and ``latent_dim = shape[-1]``, the same
convention ``keras.initializers.Orthogonal`` uses. At most ``latent_dim``
vectors in ``latent_dim`` dimensions can be mutually orthogonal, which gives two
regimes.

1. **Orthogonal** (``num_vectors <= latent_dim``): a random ``(latent_dim,
   num_vectors)`` matrix ``A`` is factored ``A = QR``. ``Q`` has orthonormal
   columns, so ``Q.T`` has orthonormal rows. The vectors satisfy
   ``v_i . v_j = 0`` and ``||v_i|| = radius`` to float32 precision, with
   measured residuals around 6e-08 at ``(64, 128)``.
2. **Tight frame** (``num_vectors > latent_dim``): no ``num_vectors`` rows can
   be mutually orthogonal, so ``ceil(num_vectors / latent_dim)`` independent
   orthonormal bases are stacked and truncated to ``num_vectors`` rows. Every
   row still has norm exactly ``radius`` and a fixed fraction of all pairs are
   still exactly orthogonal.

Why the second regime stacks bases instead of sampling
------------------------------------------------------
Uniform sampling on the sphere is available as ``fallback='uniform'`` and is
worse conditioned. It also maximizes nothing: the average pairwise cosine is
zero for any antipodally balanced configuration, and maximizing the minimum
separation is the Tammes / spherical-code problem, where uniform random is a
poor solution. Measured at ``(512, 128)``:

| construction    | max abs cos | mean abs cos | cond(W) | exactly-orthogonal pairs |
|-----------------|-------------|--------------|---------|--------------------------|
| uniform         | 0.371       | 0.0706       | 2.92    | 0.2%                     |
| stacked bases   | 0.386       | 0.0530       | 1.0000  | 25.0%                    |
| Welch bound     | 0.0766      | n/a          | n/a     | n/a                      |

Both sit far above the Welch bound, so neither is a good spherical code. The
difference that matters is conditioning. The uniform construction's singular
values span 1.03 to 3.01, which discards the dynamical isometry the Saxe
reference is cited for. Stacked bases are an exact tight frame when
``latent_dim`` divides ``num_vectors`` (``cond == 1``) and are bounded by
``sqrt(2)`` otherwise: measured 1.0000 at ``(512, 128)``, 1.118 at
``(576, 128)``, 1.0086 at ``(30000, 512)``.

Relationship to ``keras.initializers.Orthogonal``
--------------------------------------------------
For ``num_vectors > latent_dim`` Keras orthogonalizes the other axis and
returns orthonormal columns, with ``cond == 1`` exactly at any shape. That is
the right tool when a perfectly conditioned map is what you want. Orthonormal
columns leave the row norms unequal, so the vectors no longer lie on a common
hypersphere. This class keeps ``||v_i|| == radius`` as its invariant and buys
back conditioning within that constraint.

Convolutional kernels: the flattening makes a "vector" indexed by ``(kh, kw,
in_ch)`` and the vector space the output-channel axis, so the separated set is
not the per-filter weight vectors. That is Keras's convention too, but the
"expert specialization" framing below describes a Dense or embedding kernel,
not a conv one.

References:
    - Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). *Exact solutions to
      the nonlinear dynamics of learning in deep linear neural networks*.
    - Marsaglia, G. (1972). *Choosing a Point from the Surface of a Sphere*.
      The Annals of Mathematical Statistics. For the ``fallback='uniform'``
      path.
    - Welch, L. R. (1974). *Lower bounds on the maximum cross correlation of
      signals*. IEEE Transactions on Information Theory. For the coherence
      bound quoted above.
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
    """Place weight vectors on a hypersphere, orthogonal where the shape allows.

    The flattened matrix is ``(num_vectors, latent_dim)`` with
    ``num_vectors = prod(shape[:-1])``. Every row has norm exactly ``radius``,
    and the construction depends on which regime the shape falls into.

    **Architecture overview:**

    .. code-block:: text

        requested shape (..., latent_dim)
                        │
                        ▼
        ┌──────────────────────────────────┐
        │ validate: rank >= 2, all dims > 0│  raises ValueError
        └────────────────┬─────────────────┘
                         ▼
            num_vectors = prod(shape[:-1])
            latent_dim  = shape[-1]
                         │
            ┌────────────┴─────────────┐
            │                          │
     num_vectors <= latent_dim   num_vectors > latent_dim
            │                          │
            ▼                    ┌─────┴──────┐
    ┌───────────────┐            │            │
    │ QR of one     │      fallback=      fallback=
    │ Gaussian      │   'block_orthogonal'  'uniform'
    │ (latent_dim,  │            │            │
    │  num_vectors) │            ▼            ▼
    │ sign-corrected│   ┌────────────────┐ ┌──────────────┐
    └───────┬───────┘   │ stack n_blocks │ │ Gaussian rows│
            │           │ orthonormal    │ │ divided by   │
            │           │ bases, truncate│ │ their norms  │
            │           └───────┬────────┘ └──────┬───────┘
            └───────────────────┴─────────────────┘
                                │ [num_vectors, latent_dim]
                                ▼
                            * radius
                                │
                                ▼
                    reshape to requested shape
                                │
                                ▼
                        (..., latent_dim)

    ``n_blocks = ceil(num_vectors / latent_dim)``, and ``1 / n_blocks`` of all
    pairs stay exactly orthogonal.

    **Regimes:**

    .. code-block:: text

        regime            condition                  guarantee
        ---------------   ------------------------   ----------------------
        orthogonal        num_vectors <= latent_dim  v_i . v_j == 0
        block_orthogonal  num_vectors >  latent_dim  cond(W) <= sqrt(2)
        uniform           num_vectors >  latent_dim  cond(W) ~ 2.92
                          and fallback='uniform'     at (512, 128)

    Every regime keeps ``||v_i|| == radius``.

    The QR runs in at least float32: the decomposition and the row norms lose
    two orders of magnitude of accuracy in half precision.

    :param radius: Radius of the hypersphere. Every vector gets this L2 norm.
        Must be positive and finite.
    :type radius: float
    :param fallback: Construction used when ``num_vectors > latent_dim``. One of
        :data:`FALLBACK_MODES`. ``'block_orthogonal'`` stacks independent
        orthonormal bases; ``'uniform'`` samples uniformly on the sphere, which
        is worse conditioned.
    :type fallback: str
    :param seed: Optional integer seed. Following the Keras contract an instance
        replays the same tensor at every matching shape whether or not a seed is
        given; a seedless instance resolves one from the global RNG state, so
        ``keras.utils.set_random_seed`` controls it. Use
        ``dl_techniques.initializers.clone_initializer`` when two weights must
        start differently.
    :type seed: int or None

    :ivar radius: The coerced hypersphere radius.
    :vartype radius: float
    :ivar fallback: The selected fallback mode.
    :vartype fallback: str
    :ivar seed: The seed as passed by the caller.
    :vartype seed: int or None

    :raises ValueError: If ``radius`` is not positive and finite, if
        ``fallback`` is not a member of :data:`FALLBACK_MODES`, if the shape has
        fewer than two dimensions, or if any dimension is not positive.

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
        Useful for embedding layers where diverse initial vectors improve
        learning, attention mechanisms needing well-separated query/key vectors,
        and mixture-of-experts layers where expert specialization benefits from
        orthogonality.

        For a ``Conv2D`` kernel the flattening makes a "vector" span
        ``(kh, kw, in_ch)``, so the separated set is not the per-filter weight
        vectors. See the module docstring.
    """

    def __init__(
            self,
            radius: float = 1.0,
            fallback: str = "block_orthogonal",
            seed: Optional[int] = None,
    ) -> None:
        """Validate the radius and fallback, and resolve the draw seed.

        :param radius: Positive, finite hypersphere radius.
        :type radius: float
        :param fallback: One of :data:`FALLBACK_MODES`.
        :type fallback: str
        :param seed: Optional integer seed.
        :type seed: int or None
        :raises ValueError: If ``radius`` is not positive and finite, or
            ``fallback`` is not a member of :data:`FALLBACK_MODES`.
        """
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
        # caller passed, the resolved seed drives the draw. keras.utils.
        # set_random_seed seeds np.random, so a seedless instance stays
        # reproducible under a global seed.
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
        """Build ``num_vectors`` orthonormal rows in ``latent_dim`` dimensions.

        :param num_vectors: Rows to produce; must be ``<= latent_dim``.
        :type num_vectors: int
        :param latent_dim: Dimensionality of the space.
        :type latent_dim: int
        :param rng: Generator supplying the Gaussian seed matrix.
        :type rng: numpy.random.Generator
        :return: A ``(num_vectors, latent_dim)`` array with orthonormal rows.
        :rtype: numpy.ndarray
        """
        random_matrix = rng.normal(size=(latent_dim, num_vectors))
        q_matrix, r_matrix = np.linalg.qr(random_matrix)

        # LAPACK's Householder QR fixes the sign of R's diagonal, which makes Q
        # non-uniform: measured over 2000 seeds at d=8, Q[0, 0] was negative in
        # 2000 of 2000 draws and E[Q[0, 0]] = -0.2897 where Haar measure gives 0.
        # Multiplying by sign(diag(R)) restores it.
        q_matrix = q_matrix * np.sign(np.diag(r_matrix))

        return q_matrix.T

    def _generate_orthogonal_set(
        self, num_vectors: int, latent_dim: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Build truly orthogonal vectors for the ``num_vectors <= latent_dim`` case.

        :param num_vectors: Number of orthogonal vectors to generate.
        :type num_vectors: int
        :param latent_dim: Dimensionality of the latent space.
        :type latent_dim: int
        :param rng: Generator supplying the Gaussian seed matrix.
        :type rng: numpy.random.Generator
        :return: A ``(num_vectors, latent_dim)`` array, rows scaled to ``radius``.
        :rtype: numpy.ndarray
        """
        return self._orthonormal_rows(num_vectors, latent_dim, rng) * self.radius

    def _generate_block_orthogonal(
        self, num_vectors: int, latent_dim: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Stack independent orthonormal bases for the ``num_vectors > latent_dim`` case.

        :param num_vectors: Number of vectors to generate.
        :type num_vectors: int
        :param latent_dim: Dimensionality of the latent space.
        :type latent_dim: int
        :param rng: Generator supplying each block's Gaussian seed matrix.
        :type rng: numpy.random.Generator
        :return: A ``(num_vectors, latent_dim)`` array, every row of norm
            ``radius`` and ``1 / ceil(num_vectors / latent_dim)`` of all pairs
            exactly orthogonal.
        :rtype: numpy.ndarray
        """
        # ceil division
        n_blocks = -(-num_vectors // latent_dim)
        blocks = [
            self._orthonormal_rows(latent_dim, latent_dim, rng)
            for _ in range(n_blocks)
        ]
        return np.concatenate(blocks, axis=0)[:num_vectors] * self.radius

    def _generate_uniform_hypersphere(
        self, num_vectors: int, latent_dim: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample vectors uniformly on the hypersphere surface.

        Reached only through ``fallback='uniform'``. It is worse conditioned
        than the stacked-bases default: measured ``cond`` 2.92 against 1.0000 at
        ``(512, 128)``.

        :param num_vectors: Number of vectors to generate.
        :type num_vectors: int
        :param latent_dim: Dimensionality of the latent space.
        :type latent_dim: int
        :param rng: Generator supplying the Gaussian draws.
        :type rng: numpy.random.Generator
        :return: A ``(num_vectors, latent_dim)`` array, rows scaled to ``radius``.
        :rtype: numpy.ndarray
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
        """Generate the orthogonal-hypersphere initialization tensor.

        :param shape: Tensor shape ``(..., latent_dim)``; at least 2 dimensions.
        :type shape: sequence of int
        :param dtype: Data type of the result. ``None`` falls back to
            ``keras.config.floatx()``.
        :type dtype: str or None
        :param kwargs: Additional arguments (unused).
        :return: A tensor of the requested shape whose flattened rows lie on the
            hypersphere of radius ``radius``.
        :rtype: tensor
        :raises ValueError: If the shape has fewer than two dimensions or any
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

        # Set the norm in at least float32: the QR and the row norms lose two
        # orders of magnitude of accuracy in half precision.
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
        """Return the constructor arguments for serialization.

        :return: A dict holding ``radius``, ``fallback`` and ``seed``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "radius": self.radius,
            "fallback": self.fallback,
            "seed": self.seed,
        })
        return config

    def __repr__(self) -> str:
        """Return the constructor-like representation.

        :return: A string naming the radius, fallback and seed.
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}("
            f"radius={self.radius}, fallback={self.fallback!r}, "
            f"seed={self.seed})"
        )

# ---------------------------------------------------------------------
