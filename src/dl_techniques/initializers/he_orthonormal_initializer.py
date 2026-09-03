"""Orthonormal-row initializer seeded from a He-normal random matrix.

Provides :class:`HeOrthonormalInitializer`, which draws a He-normal matrix and
orthonormalizes it with a QR decomposition. The result is well scaled for deep
ReLU stacks at the draw stage and norm preserving after orthonormalization,
which helps against vanishing and exploding gradients.

How the matrix is built
-----------------------
1. **He-normal seeding.** Draw ``A`` from a truncated normal centered at 0 with
   ``sigma = sqrt(2 / fan_in)``. That scaling keeps activation variance roughly
   constant across layers of ReLU neurons.
2. **Orthonormalization.** Factor ``A = QR`` and take the sign-corrected ``Q``.

QR normalizes the columns of ``Q`` to unit L2 norm, so the final rows are
strictly orthonormal and do not carry the ``2 / fan_in`` variance of the He
draw. The He step contributes a well-conditioned random matrix to orthogonalize,
not the output variance.

The shape rules and the ``sign(diag(R))`` convention are shared with
:mod:`dl_techniques.initializers.orthonormal_initializer`, which owns them.

References:
    - He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into
      Rectifiers: Surpassing Human-Level Performance on ImageNet
      Classification*.
    - Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). *Exact solutions to
      the nonlinear dynamics of learning in deep linear neural networks*.
"""

import keras
from typing import Optional, Any, Tuple, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers.orthonormal_initializer import (
    validate_orthonormal_seed,
    validate_orthonormal_shape,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.initializers.he_orthonormal_initializer")
class HeOrthonormalInitializer(keras.initializers.Initializer):
    """Orthonormalize a He-normal matrix into orthonormal row vectors.

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
        │ keras.initializers.HeNormal      │
        │ sigma = sqrt(2 / fan_in)         │
        └────────────────┬─────────────────┘
                         │ [n_clusters, feature_dims]
                         ▼
                     transpose
                         │ [feature_dims, n_clusters]
                         ▼
        ┌──────────────────────────────────┐
        │ QR                               │
        └────────┬────────────────┬────────┘
             Q   │                │  R
      [f_dims, n_cl]              │ [n_cl, n_cl]
                 │                ▼
                 │      signs = +1 where diag(R) >= 0
                 │              else -1
                 ▼                │
        ┌──────────────────────────────────┐
        │ Q * signs, then transpose        │
        └────────────────┬─────────────────┘
                         │ [n_clusters, feature_dims]
                         ▼
                  cast to dtype
                         │
                         ▼
            [n_clusters, feature_dims]

    The QR step normalizes every row to unit norm, so the He variance does not
    survive into the result.

    :param seed: Random seed for reproducible initialization. ``None`` leaves
        the draw non-deterministic.
    :type seed: int or None

    :ivar seed: The validated seed.
    :vartype seed: int or None

    :raises ValueError: If the seed is invalid, or if the requested shape
        cannot produce orthonormal vectors (``n_clusters > feature_dims``).

    Example:
        >>> # Basic usage with a Dense layer
        >>> initializer = HeOrthonormalInitializer(seed=42)
        >>> layer = keras.layers.Dense(64, kernel_initializer=initializer)

        >>> # 10 orthonormal vectors in 50-D
        >>> initializer = HeOrthonormalInitializer(seed=123)
        >>> orthonormal_matrix = initializer((10, 50))
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Validate the seed and build the internal He-normal initializer.

        :param seed: Random seed for reproducible initialization.
        :type seed: int or None
        :raises ValueError: If the seed is not a non-negative integer.
        """
        super().__init__()
        self.seed = seed
        self._validate_seed()

        self._he_normal = keras.initializers.HeNormal(seed=seed)

        logger.info(f"Initialized HeOrthonormalInitializer with seed={seed}")

    def _validate_seed(self) -> None:
        """Validate the stored seed and coerce it in place.

        The rule lives in
        :func:`dl_techniques.initializers.orthonormal_initializer.validate_orthonormal_seed`,
        which this class and ``OrthonormalInitializer`` share.

        :return: Nothing.
        :rtype: None
        :raises ValueError: If the stored seed is invalid.
        """
        self.seed = validate_orthonormal_seed(self.seed, type(self).__name__)

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
        dtype: Optional[Union[str, Any]] = None
    ) -> Any:
        """Draw a He-normal matrix and orthonormalize it.

        Shape validation runs before the draw and outside the ``try``, so a bad
        shape raises ``ValueError``. Any backend failure inside the draw or the
        decomposition is re-raised as ``RuntimeError``.

        :param shape: Desired shape ``(n_clusters, feature_dims)``.
        :type shape: tuple of int
        :param dtype: Data type of the result. ``None`` falls back to
            ``keras.config.floatx()``.
        :type dtype: str or None
        :return: Orthonormal vectors of shape ``(n_clusters, feature_dims)``,
            each row a unit vector, all rows mutually orthogonal.
        :rtype: tensor
        :raises ValueError: If the shape is invalid or ``n_clusters >
            feature_dims``.
        :raises RuntimeError: If the draw or the QR decomposition fails.
        """
        n_clusters, feature_dims = self._validate_shape(shape)

        if dtype is None:
            dtype = keras.config.floatx()

        if hasattr(dtype, 'name'):
            dtype = dtype.name

        logger.debug(
            f"Generating {n_clusters} He orthonormal vectors in {feature_dims}D space "
            f"with dtype {dtype}"
        )

        try:
            he_matrix = self._he_normal(shape, dtype=dtype)

            # Transposing first puts the vectors on the column axis, so Q's
            # orthonormal columns become orthonormal rows after the transpose
            # below. Shape here: (feature_dims, n_clusters).
            he_matrix_t = keras.ops.transpose(he_matrix)
            # Q: (feature_dims, n_clusters).
            q, r = keras.ops.linalg.qr(he_matrix_t)

            # Sign convention: make the diagonal of R positive. Shape (n_clusters,).
            r_diag = keras.ops.diagonal(r)
            signs = keras.ops.where(
                keras.ops.greater_equal(r_diag, keras.ops.cast(0.0, dtype)),
                keras.ops.cast(1.0, dtype),
                keras.ops.cast(-1.0, dtype)
            )

            # Shape after this: (feature_dims, n_clusters).
            q_signed = q * keras.ops.expand_dims(signs, axis=0)

            # Shape after this: (n_clusters, feature_dims).
            orthonormal_vectors = keras.ops.transpose(q_signed)

            orthonormal_vectors = keras.ops.cast(orthonormal_vectors, dtype)

            logger.debug(f"Successfully generated He orthonormal vectors with shape {keras.ops.shape(orthonormal_vectors)}")
            return orthonormal_vectors

        except Exception as e:
            logger.error(f"Failed to generate He orthonormal vectors: {str(e)}")
            raise RuntimeError(
                f"Failed to generate He orthonormal vectors for shape {shape}: {str(e)}"
            ) from e

    def _gram_schmidt_orthogonalize(self, vectors: Any, dtype: str) -> Any:
        """Reject a Gram-Schmidt request: it needs dynamic loops.

        Kept as a documented reference point. Keras ops do not support the
        dynamic loops the algorithm requires, so :meth:`__call__` uses QR.

        :param vectors: Input vectors of shape ``(n_vectors, feature_dims)``.
        :type vectors: tensor
        :param dtype: Compute dtype.
        :type dtype: str
        :return: Never returns.
        :rtype: tensor
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Gram-Schmidt implementation requires dynamic loops not supported in Keras ops"
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: A dict holding ``seed``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HeOrthonormalInitializer":
        """Rebuild an initializer from a config dict.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: dict
        :return: A new initializer.
        :rtype: HeOrthonormalInitializer
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return the constructor-like representation.

        :return: A string naming the seed.
        :rtype: str
        """
        return f"HeOrthonormalInitializer(seed={self.seed})"

    def __str__(self) -> str:
        """Return a human-readable description.

        :return: A one-line description naming the seed.
        :rtype: str
        """
        return f"HeOrthonormalInitializer with seed={self.seed}"

# ---------------------------------------------------------------------
