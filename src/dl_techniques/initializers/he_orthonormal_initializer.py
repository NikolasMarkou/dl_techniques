"""
HeOrthonormalInitializer: A Keras initializer that applies He normal initialization followed by orthonormalization.

This module implements a custom Keras initializer that first applies He normal initialization
and then orthonormalizes the resulting vectors using QR decomposition. This combines the
variance scaling benefits of He initialization with the geometric properties of orthonormal vectors.

The implementation follows these key steps:
1. Generate a random matrix using He normal initialization
2. Apply QR decomposition to obtain orthogonal vectors (Q matrix)
3. Ensure numerical stability by fixing the signs based on the diagonal of R
4. Extract the orthonormal vectors as rows of the transposed Q matrix

Mathematical Background
-----------------------
He normal initialization provides proper variance scaling for ReLU networks, while orthonormal
vectors ensure that the weight vectors are mutually orthogonal and have unit length after
the QR decomposition process.

Examples
--------
>>> # Initialize a Dense layer with He orthonormal weights
>>> dense = keras.layers.Dense(
...     units=64,
...     kernel_initializer=HeOrthonormalInitializer(seed=42)
... )
>>>
>>> # Initialize clustering centroids with He orthonormal vectors
>>> initializer = HeOrthonormalInitializer(seed=123)
>>> centroids = initializer((10, 128))
"""

import keras
from keras import ops
from typing import Optional, Any, Tuple, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HeOrthonormalInitializer(keras.initializers.Initializer):
    """Custom initializer that applies He normal initialization followed by orthonormalization.

    This initializer first generates weights using He normal initialization, then applies
    QR decomposition to create orthonormal vectors. The approach preserves the initial
    variance scaling intent of He initialization while ensuring orthogonality.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible initialization. If None, the initialization
        will be non-deterministic.

    Raises
    ------
    ValueError
        If the requested shape cannot produce orthonormal vectors (i.e., when
        n_clusters > feature_dims).

    Examples
    --------
    >>> # Basic usage with a Dense layer
    >>> initializer = HeOrthonormalInitializer(seed=42)
    >>> layer = keras.layers.Dense(64, kernel_initializer=initializer)

    >>> # Direct tensor creation
    >>> initializer = HeOrthonormalInitializer(seed=123)
    >>> orthonormal_matrix = initializer((10, 50))  # 10 orthonormal vectors in 50D space
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the He orthonormal initializer.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible initialization.
        """
        super().__init__()
        self.seed = seed
        self._validate_seed()

        # Create internal He normal initializer
        self._he_normal = keras.initializers.HeNormal(seed=seed)

        logger.info(f"Initialized HeOrthonormalInitializer with seed={seed}")

    def _validate_seed(self) -> None:
        """Validate the seed parameter.

        Raises
        ------
        ValueError
            If seed is not None and not a non-negative integer.
        """
        if self.seed is not None:
            if not isinstance(self.seed, int):
                raise ValueError(f"Seed must be an integer, got {type(self.seed).__name__}")
            if self.seed < 0:
                raise ValueError(f"Seed must be non-negative, got {self.seed}")

    def _validate_shape(self, shape: Tuple[int, ...]) -> Tuple[int, int]:
        """Validate and extract dimensions from the shape.

        Parameters
        ----------
        shape : tuple of int
            The requested tensor shape.

        Returns
        -------
        tuple of int
            A tuple of (n_clusters, feature_dims).

        Raises
        ------
        ValueError
            If shape is invalid or if n_clusters > feature_dims.
        """
        if len(shape) != 2:
            raise ValueError(
                f"HeOrthonormalInitializer requires a 2D shape (n_clusters, feature_dims), "
                f"got shape with {len(shape)} dimensions: {shape}"
            )

        n_clusters, feature_dims = shape

        if not isinstance(n_clusters, int) or not isinstance(feature_dims, int):
            raise ValueError(
                f"Shape dimensions must be integers, got {type(n_clusters).__name__} "
                f"and {type(feature_dims).__name__}"
            )

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

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, Any]] = None
    ) -> Any:
        """Generate orthonormal vectors using He normal initialization followed by QR decomposition.

        Parameters
        ----------
        shape : tuple of int
            Desired shape of the output tensor (n_clusters, feature_dims).
        dtype : str or dtype, optional
            Desired data type of the output tensor. If None, uses the default
            Keras dtype.

        Returns
        -------
        tensor
            Orthonormal vectors of shape (n_clusters, feature_dims). Each row
            is a unit vector, and all rows are mutually orthogonal.

        Raises
        ------
        ValueError
            If the shape is invalid or if n_clusters > feature_dims.

        Notes
        -----
        The algorithm works as follows:
        1. Generate a He normal matrix and apply QR decomposition
        2. Extract orthonormal vectors from the Q matrix
        3. Apply sign convention for deterministic results

        This guarantees orthonormality up to numerical precision.
        """
        n_clusters, feature_dims = self._validate_shape(shape)

        if dtype is None:
            dtype = keras.config.floatx()

        # Ensure dtype is a string for consistency
        if hasattr(dtype, 'name'):
            dtype = dtype.name

        logger.debug(
            f"Generating {n_clusters} He orthonormal vectors in {feature_dims}D space "
            f"with dtype {dtype}"
        )

        try:
            # Generate He normal matrix of the desired shape directly
            he_matrix = self._he_normal(shape, dtype=dtype)

            # Apply QR decomposition to the transposed matrix
            # This gives us orthonormal columns that become orthonormal rows after transpose
            he_matrix_t = ops.transpose(he_matrix)  # Shape: (feature_dims, n_clusters)
            q, r = ops.linalg.qr(he_matrix_t)  # Q: (feature_dims, n_clusters)

            # Apply sign convention: make diagonal elements of R positive
            r_diag = ops.diagonal(r)  # Shape: (n_clusters,)
            signs = ops.where(
                ops.greater_equal(r_diag, ops.cast(0.0, dtype)),
                ops.cast(1.0, dtype),
                ops.cast(-1.0, dtype)
            )

            # Apply signs to Q
            q_signed = q * ops.expand_dims(signs, axis=0)  # Shape: (feature_dims, n_clusters)

            # Transpose to get orthonormal rows
            orthonormal_vectors = ops.transpose(q_signed)  # Shape: (n_clusters, feature_dims)

            # Ensure correct dtype
            orthonormal_vectors = ops.cast(orthonormal_vectors, dtype)

            logger.debug(f"Successfully generated He orthonormal vectors with shape {ops.shape(orthonormal_vectors)}")
            return orthonormal_vectors

        except Exception as e:
            logger.error(f"Failed to generate He orthonormal vectors: {str(e)}")
            raise RuntimeError(
                f"Failed to generate He orthonormal vectors for shape {shape}: {str(e)}"
            ) from e

    def _gram_schmidt_orthogonalize(self, vectors: Any, dtype: str) -> Any:
        """Apply Gram-Schmidt orthogonalization to the rows of a matrix.

        This method is kept for reference but not currently used due to
        dynamic loop limitations in Keras ops.

        Parameters
        ----------
        vectors : tensor
            Input vectors of shape (n_vectors, feature_dims).
        dtype : str
            Desired dtype for computations.

        Returns
        -------
        tensor
            Orthonormalized vectors of the same shape.
        """
        # This implementation would require dynamic loops which are
        # not efficiently supported in Keras ops
        raise NotImplementedError(
            "Gram-Schmidt implementation requires dynamic loops not supported in Keras ops"
        )

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the initializer.

        Returns
        -------
        dict
            Configuration dictionary containing all parameters needed to
            recreate this initializer.
        """
        config = super().get_config()
        config.update({
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HeOrthonormalInitializer":
        """Create an initializer from its configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary as returned by get_config().

        Returns
        -------
        HeOrthonormalInitializer
            A new initializer instance with the specified configuration.
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return a string representation of the initializer."""
        return f"HeOrthonormalInitializer(seed={self.seed})"

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"HeOrthonormalInitializer with seed={self.seed}"

# ---------------------------------------------------------------------