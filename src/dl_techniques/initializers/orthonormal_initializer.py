"""Orthonormal-row weight initializer built on a thin QR decomposition.

Provides :class:`OrthonormalInitializer`, which fills a ``(n_clusters,
feature_dims)`` matrix with rows that are unit norm and mutually orthogonal,
plus the three validation and construction helpers it shares with
:mod:`dl_techniques.initializers.he_orthonormal_initializer`.

Orthonormal rows preserve the norm of signals during forward and backward
propagation, which stabilizes training against vanishing and exploding
gradients.

How the matrix is built
-----------------------
1. Draw a random ``(feature_dims, n_clusters)`` Gaussian matrix ``A``. Only the
   ``n_clusters`` vectors actually requested are factorized, a thin QR costing
   ``O(feature_dims * n_clusters**2)``. Factorizing the full ``feature_dims x
   feature_dims`` square instead is ``O(d**3)``: 9.08 s and a 67.1 MB buffer
   for a 64-vector codebook in 4096 dimensions, against 0.16 s for the thin
   factorization.
2. Decompose ``A = QR``. ``Q`` has orthonormal columns, so ``q_i . q_j =
   delta_ij``.
3. Transpose ``Q``, giving the requested orthonormal rows.

Shape constraint
----------------
No more than ``d`` mutually orthogonal vectors exist in ``d`` dimensions, so
``n_clusters <= feature_dims`` is enforced. The initializer is 2-D only: a 4-D
convolution kernel raises ``ValueError`` rather than being reinterpreted, which
``tests/test_layers/test_convnext_v1_block.py::test_unsupported_repo_initializers_raise``
pins. On a ``Dense`` kernel ``(input_dim, units)`` the constraint reads
``input_dim <= units``, so only a widening layer qualifies. For a narrowing
``Dense`` or any ``Conv2D``, use ``keras.initializers.Orthogonal`` or
``OrthogonalHypersphereInitializer``. This class is a codebook / centroid
initializer whose contract is orthonormal rows.

The sign convention
-------------------
QR is unique only up to the signs of ``Q``'s columns, and LAPACK, cuSOLVER and
JAX builds may return different ones for the same ``A``. The canonical choice
``Q *= sign(diag(R))`` yields the unique factorization with a positive ``R``
diagonal. ``he_orthonormal_initializer.py`` and
``hypersphere_orthogonal_initializer.py`` use the same convention.

Do not key the convention on the first row of ``Q`` instead. That is equally
deterministic, but it folds row 0 into the positive orthant on every draw.
Measured over 4000 seeds at ``d=64``, ``P(any entry of row 0 < 0)`` was 0.000
and ``E[entry of row 0]`` was ``+0.10010``, matching the theoretical
``sqrt(2/(pi*d)) = 0.09974`` for a half-normal fold and giving row 0 a mean
cosine of 0.801 to the all-ones direction. The damage lands entirely on centroid
0 of every codebook. Under ``sign(diag(R))`` the same measurement reads
``P = 1.000``, ``E = -0.00002``, cosine ``-0.0002``, matching
``keras.initializers.Orthogonal``, because that convention preserves the Haar
distribution.

References:
    - Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). *Exact solutions to
      the nonlinear dynamics of learning in deep linear neural networks*.
    - Mishkin, D., & Matas, J. (2015). *All you need is a good init*.
"""

import keras
import numpy as np
from typing import Optional, Any, Tuple, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# shared validation
#
# One home for rules that OrthonormalInitializer and HeOrthonormalInitializer
# both apply, so a single edit reddens both suites.
# ---------------------------------------------------------------------


def validate_orthonormal_seed(seed: Optional[int], class_name: str) -> Optional[int]:
    """Validate a seed argument and coerce it to a Python ``int``.

    :param seed: The seed to validate. ``numpy`` integers are accepted and
        coerced. ``bool`` is rejected, since ``isinstance(True, int)`` is
        ``True`` and ``seed=True`` would silently behave as ``seed=1``.
    :type seed: int or None
    :param class_name: Owning class name. Accepted for a uniform signature; the
        messages are shared verbatim between the two classes.
    :type class_name: str
    :return: The coerced seed.
    :rtype: int or None
    :raises ValueError: If the seed is not ``None`` and not a non-negative
        integer.
    """
    del class_name
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"Seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")
    return int(seed)


def validate_orthonormal_shape(
    shape: Tuple[int, ...], class_name: str
) -> Tuple[int, int]:
    """Validate a ``(n_clusters, feature_dims)`` shape for a row-orthonormal set.

    :param shape: The requested tensor shape. Entries may be any integer type;
        they are coerced with ``int()`` rather than type-checked, because a
        shape that has been through ``TensorShape`` or a config round trip
        carries ``np.int64``, which is not an ``int`` subclass.
    :type shape: tuple of int
    :param class_name: Owning class name, used in the rank error message.
    :type class_name: str
    :return: The pair ``(n_clusters, feature_dims)``.
    :rtype: tuple of int
    :raises ValueError: If the shape is not 2-D, holds a non-positive or
        non-integral dimension, or requests more vectors than dimensions.
    """
    if len(shape) != 2:
        # Stays a ValueError and stays outside any try/except: a 4-D convolution
        # kernel must raise ValueError specifically, which
        # tests/test_layers/test_convnext_v1_block.py and its v2 twin pin.
        raise ValueError(
            f"{class_name} requires a 2D shape (n_clusters, feature_dims), "
            f"got shape with {len(shape)} dimensions: {shape}"
        )

    try:
        n_clusters, feature_dims = (int(d) for d in shape)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Shape dimensions must be integers, got {tuple(type(d).__name__ for d in shape)}"
        ) from error

    if n_clusters <= 0 or feature_dims <= 0:
        raise ValueError(
            f"Shape dimensions must be positive, got n_clusters={n_clusters}, "
            f"feature_dims={feature_dims}"
        )

    if n_clusters > feature_dims:
        raise ValueError(
            f"Cannot create {n_clusters} orthogonal vectors in "
            f"{feature_dims}-dimensional space. n_clusters ({n_clusters}) must be "
            f"<= feature_dims ({feature_dims})"
        )

    return n_clusters, feature_dims


def orthonormal_rows_from_qr(matrix: Any, dtype: str) -> Any:
    """Return sign-corrected orthonormal rows from a tall Gaussian matrix.

    :param matrix: The Gaussian seed matrix, tall: ``(feature_dims,
        n_clusters)``.
    :type matrix: tensor
    :param dtype: Compute dtype of ``matrix``.
    :type dtype: str
    :return: A ``(n_clusters, feature_dims)`` tensor with orthonormal rows.
    :rtype: tensor
    """
    q, r = keras.ops.qr(matrix, mode="reduced")

    # Canonicalize on diag(R), the unique factorization with a positive R
    # diagonal. Keying on Q's first row is equally deterministic but folds that
    # row into the positive orthant: measured E[row 0] = +0.10010 against 0 here.
    signs = keras.ops.sign(keras.ops.diagonal(r))
    # sign(0) is 0 and would zero a column. A Gaussian makes that measure-zero,
    # but the clamp costs nothing.
    signs = keras.ops.where(
        keras.ops.equal(signs, keras.ops.cast(0.0, dtype)),
        keras.ops.ones_like(signs),
        signs,
    )
    return keras.ops.transpose(q * keras.ops.expand_dims(signs, axis=0))


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.initializers.orthonormal_initializer")
class OrthonormalInitializer(keras.initializers.Initializer):
    """Build orthonormal row vectors with a thin, sign-canonicalized QR.

    **Architecture overview:**

    .. code-block:: text

        requested shape (n_clusters, feature_dims)
                        │
                        ▼
        ┌──────────────────────────────────┐
        │ validate: rank 2, all positive,  │
        │ n_clusters <= feature_dims       │  raises ValueError
        └────────────────┬─────────────────┘
                         ▼
        ┌──────────────────────────────────┐
        │ keras.random.normal              │
        │ (thin: only what was asked for)  │
        └────────────────┬─────────────────┘
                         │ [feature_dims, n_clusters]
                         ▼
        ┌──────────────────────────────────┐
        │ QR, mode='reduced'               │
        │ compute dtype float32 or float64 │
        └────────┬────────────────┬────────┘
             Q   │                │  R
      [f_dims, n_cl]              │ [n_cl, n_cl]
                 │                ▼
                 │        signs = sign(diag(R))
                 │           (0 clamped to 1)
                 ▼                │
        ┌──────────────────────────────────┐
        │ Q * signs, then transpose        │
        └────────────────┬─────────────────┘
                         │ [n_clusters, feature_dims]
                         ▼
                  * gain  ('gain' != 1.0 only)
                         │
                         ▼
                  cast to dtype
                         │
                         ▼
            [n_clusters, feature_dims]

    Half precision has no usable QR kernel, so the decomposition runs in
    float32 and the result is cast. bfloat16 raises outright in TensorFlow, and
    float16 runs while degrading orthonormality to ~1.1e-03 against 1.8e-07 in
    float32.

    :param gain: Positive multiplicative scale for the orthonormal rows.
        ``1.0`` leaves them unit norm; ``sqrt(2)`` is conventional for a ReLU
        stack, matching ``keras.initializers.Orthogonal(gain=...)``. Must be
        finite.
    :type gain: float
    :param seed: Random seed. Following the Keras contract an instance replays
        the same tensor at every matching shape whether or not a seed is given;
        a seedless instance resolves one from the global RNG state, so
        ``keras.utils.set_random_seed`` controls it.
    :type seed: int or None

    :ivar gain: The coerced scale factor.
    :vartype gain: float
    :ivar seed: The seed as passed by the caller.
    :vartype seed: int or None

    :raises ValueError: If ``gain`` is not positive and finite, if the seed is
        invalid, or if the requested shape cannot produce orthonormal vectors.

    Example:
        >>> # A codebook / centroid matrix: 10 orthonormal vectors in 50-D
        >>> initializer = OrthonormalInitializer(seed=123)
        >>> orthonormal_matrix = initializer((10, 50))

        >>> # A Dense kernel is (input_dim, units), so the constraint reads
        >>> # input_dim <= units and only a widening layer qualifies.
        >>> layer = keras.layers.Dense(64, kernel_initializer=OrthonormalInitializer())
        >>> _ = layer.build((None, 32))     # kernel (32, 64): 32 <= 64, fine
        >>> # layer.build((None, 128))      # kernel (128, 64): raises ValueError
    """

    def __init__(self, gain: float = 1.0, seed: Optional[int] = None) -> None:
        """Validate the gain and seed, and resolve the draw seed.

        :param gain: Positive, finite multiplicative scale for the orthonormal
            rows.
        :type gain: float
        :param seed: Random seed for reproducible initialization.
        :type seed: int or None
        :raises ValueError: If ``gain`` is not positive and finite, or the seed
            is not a non-negative integer.
        """
        super().__init__()

        if not np.isfinite(gain):
            raise ValueError(f"gain must be finite, got {gain}")
        if gain <= 0:
            raise ValueError(f"gain must be positive, got {gain}")

        self.gain = float(gain)
        self.seed = validate_orthonormal_seed(seed, type(self).__name__)

        # Mirrors keras.initializers.RandomNormal: the config keeps what the
        # caller passed, a resolved seed drives the draw. keras.utils.
        # set_random_seed seeds np.random, so a seedless instance stays
        # reproducible under a global seed.
        self._draw_seed = self.seed if self.seed is not None else int(
            np.random.randint(0, 2 ** 30)
        )

        logger.debug(
            f"Initialized OrthonormalInitializer(gain={self.gain}, seed={self.seed})"
        )

    def _validate_seed(self) -> None:
        """Validate the stored seed.

        The test suite asserts this method exists. The rule itself lives in
        :func:`validate_orthonormal_seed`.

        :return: Nothing.
        :rtype: None
        :raises ValueError: If the stored seed is invalid.
        """
        validate_orthonormal_seed(self.seed, type(self).__name__)

    def _validate_shape(self, shape: Tuple[int, ...]) -> Tuple[int, int]:
        """Validate the shape and extract its two dimensions.

        :param shape: The requested tensor shape.
        :type shape: tuple of int
        :return: The pair ``(n_clusters, feature_dims)``.
        :rtype: tuple of int
        :raises ValueError: If the shape is invalid or ``n_clusters >
            feature_dims``.
        """
        return validate_orthonormal_shape(shape, type(self).__name__)

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate orthonormal vectors using QR decomposition.

        Validation runs before any backend call and is not wrapped, so a bad
        shape raises ``ValueError`` and not a backend error type.

        :param shape: Desired shape ``(n_clusters, feature_dims)``.
        :type shape: tuple of int
        :param dtype: Data type of the result. ``None`` falls back to
            ``keras.config.floatx()``.
        :type dtype: str or None
        :param kwargs: Additional arguments (unused).
        :return: Orthonormal vectors of shape ``(n_clusters, feature_dims)``,
            each row a unit vector times ``gain``, all rows mutually orthogonal.
        :rtype: tensor
        :raises ValueError: If the shape is invalid or ``n_clusters >
            feature_dims``.
        """
        # Ahead of every backend call: see validate_orthonormal_shape.
        n_clusters, feature_dims = self._validate_shape(shape)

        if dtype is None:
            dtype = keras.config.floatx()
        dtype = getattr(dtype, "name", None) or str(dtype)

        # Half precision has no usable QR kernel, so decompose in float32 and
        # cast: float16 degrades orthonormality to ~1.1e-03 against 1.8e-07.
        compute_dtype = dtype if dtype in ("float32", "float64") else "float32"

        logger.debug(
            f"Generating {n_clusters} orthonormal vectors in {feature_dims}D space "
            f"with dtype {dtype}"
        )

        random_matrix = keras.random.normal(
            shape=(feature_dims, n_clusters),
            dtype=compute_dtype,
            seed=self._draw_seed,
        )
        vectors = orthonormal_rows_from_qr(random_matrix, compute_dtype)

        if self.gain != 1.0:
            vectors = vectors * keras.ops.cast(self.gain, compute_dtype)

        return keras.ops.cast(vectors, dtype)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: A dict holding ``gain`` and ``seed``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "gain": self.gain,
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OrthonormalInitializer":
        """Rebuild an initializer from a config dict.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: dict
        :return: A new initializer.
        :rtype: OrthonormalInitializer
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return the constructor-like representation.

        :return: A string naming the gain and seed.
        :rtype: str
        """
        return f"OrthonormalInitializer(gain={self.gain}, seed={self.seed})"

    def __str__(self) -> str:
        """Return a human-readable description.

        :return: A one-line description naming the gain and seed.
        :rtype: str
        """
        return f"OrthonormalInitializer with gain={self.gain}, seed={self.seed}"

# ---------------------------------------------------------------------
