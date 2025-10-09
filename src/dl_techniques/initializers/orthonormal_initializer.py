"""Initialize weights as a set of orthonormal vectors.

This initializer constructs a weight matrix whose rows form an orthonormal set,
meaning each row vector has a unit norm and is orthogonal to every other row
vector. Such a geometric configuration is highly desirable in deep neural
networks as it helps preserve the norm of signals during forward and backward
propagation, a property known as isometry. This can significantly stabilize
training by mitigating the vanishing and exploding gradients problem.

Architecture and Mathematical Foundations:
The generation of the orthonormal matrix relies on QR decomposition, a
standard procedure in linear algebra. The conceptual process is as follows:

1.  A random square matrix `A` is sampled from a Gaussian distribution. The
    dimensionality of this matrix corresponds to the feature dimension of the
    weights (`feature_dims x feature_dims`).

2.  This random matrix `A` undergoes QR decomposition, factorizing it into
    `A = QR`, where `Q` is an orthogonal matrix and `R` is an upper
    triangular matrix. An orthogonal matrix `Q` has the defining property
    that its columns (and rows) form an orthonormal basis. That is, for
_   any two column vectors `q_i` and `q_j` from `Q`, their dot product is
    `q_i · q_j = δ_ij`, where `δ_ij` is the Kronecker delta.

3.  The final weight matrix is constructed by taking a subset of the rows
    from the resulting orthogonal matrix `Q`.

A key mathematical constraint is that the number of vectors being initialized
(e.g., `n_clusters` or `units`) cannot exceed the dimensionality of the
vector space (`feature_dims`). It is mathematically impossible to construct
more than `d` mutually orthogonal vectors in a `d`-dimensional space. This
initializer enforces this constraint. To ensure reproducibility for a given
seed, a sign-flipping convention is applied to `Q`, as QR decomposition is
only unique up to the signs of its columns.

References:
    - Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). *Exact
      solutions to the nonlinear dynamics of learning in deep linear
      neural networks*. This paper provides a foundational theoretical
      analysis showing how orthogonal initialization prevents gradient
      issues in deep linear networks.
    - Mishkin, D., & Matas, J. (2015). *All you need is a good init*. This
      work demonstrates the practical benefits of orthogonal initialization
      for deep convolutional networks.

"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Any, Tuple, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class OrthonormalInitializer(keras.initializers.Initializer):
    """Custom initializer for orthonormal vectors using QR decomposition.

    This initializer creates a set of orthonormal vectors by generating a random matrix
    and applying QR decomposition to obtain orthogonal vectors with unit length. The
    approach ensures numerical stability and mathematical correctness.

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
    >>> initializer = OrthonormalInitializer(seed=42)
    >>> layer = keras.layers.Dense(64, kernel_initializer=initializer)

    >>> # Direct tensor creation
    >>> initializer = OrthonormalInitializer(seed=123)
    >>> orthonormal_matrix = initializer((10, 50))  # 10 orthonormal vectors in 50D space
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the orthonormal initializer.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible initialization.
        """
        super().__init__()
        self.seed = seed
        self._validate_seed()
        logger.info(f"Initialized OrthonormalInitializer with seed={seed}")

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
                f"OrthonormalInitializer requires a 2D shape (n_clusters, feature_dims), "
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

    def _extract_diagonal(self, matrix: Any) -> Any:
        """Extract diagonal elements from a square matrix.

        This is a helper method to extract diagonal elements using available keras.ops.
        Since keras.ops doesn't have a direct diagonal extraction function, we use
        a loop-based approach to extract diagonal elements.

        Parameters
        ----------
        matrix : tensor
            A square matrix from which to extract diagonal elements.

        Returns
        -------
        tensor
            A 1D tensor containing the diagonal elements.
        """
        matrix_shape = ops.shape(matrix)
        min_dim = ops.numpy.minimum(matrix_shape[0], matrix_shape[1])

        # Extract diagonal elements one by one and stack them
        diagonal_elements = []
        for i in range(int(ops.convert_to_numpy(min_dim))):
            diagonal_elements.append(matrix[i, i])

        return ops.stack(diagonal_elements)

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, Any]] = None
    ) -> Any:
        """Generate orthonormal vectors using QR decomposition.

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
        1. Generate a random matrix of shape (feature_dims, feature_dims)
        2. Apply QR decomposition: A = QR
        3. Ensure deterministic results by using a simple sign convention
        4. Extract the first n_clusters rows from Q

        This guarantees orthonormality up to numerical precision.
        """
        n_clusters, feature_dims = self._validate_shape(shape)

        if dtype is None:
            dtype = keras.config.floatx()

        # Ensure dtype is a string for consistency
        if hasattr(dtype, 'name'):
            dtype = dtype.name

        logger.debug(
            f"Generating {n_clusters} orthonormal vectors in {feature_dims}D space "
            f"with dtype {dtype}"
        )

        try:
            # Create random matrix using numpy for consistent seeding
            rng = np.random.RandomState(self.seed)

            # Use the target dtype for the random matrix
            if dtype.startswith('float64') or dtype == 'double':
                np_dtype = np.float64
            else:
                np_dtype = np.float32

            random_matrix = rng.randn(feature_dims, feature_dims).astype(np_dtype)

            # Convert to tensor for keras.ops compatibility
            random_tensor = ops.convert_to_tensor(random_matrix, dtype=dtype)

            # Compute QR decomposition using keras.ops
            q, r = ops.linalg.qr(random_tensor, mode="reduced")

            # Ensure deterministic results by applying a simple sign convention
            # We'll make the first element of each column have a consistent sign
            # This is simpler and avoids complex indexing operations

            # Get the first row of Q matrix
            first_row = q[0, :]  # Shape: (feature_dims,)

            # Create sign corrections based on the first row
            signs = ops.where(
                ops.numpy.greater_equal(first_row, ops.cast(0.0, dtype)),
                ops.ones_like(first_row),
                ops.cast(-1.0, dtype) * ops.ones_like(first_row)
            )

            # Apply sign corrections to each column
            q_corrected = q * ops.expand_dims(signs, axis=0)

            # Extract the first n_clusters rows to get desired number of vectors
            orthonormal_vectors = q_corrected[:n_clusters, :]

            # Ensure the result has the correct dtype
            result = ops.cast(orthonormal_vectors, dtype)

            logger.debug(f"Successfully generated orthonormal vectors with shape {ops.shape(result)}")
            return result

        except Exception as e:
            logger.error(f"Failed to generate orthonormal vectors: {str(e)}")
            raise RuntimeError(
                f"Failed to generate orthonormal vectors for shape {shape}: {str(e)}"
            ) from e

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
    def from_config(cls, config: Dict[str, Any]) -> "OrthonormalInitializer":
        """Create an initializer from its configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary as returned by get_config().

        Returns
        -------
        OrthonormalInitializer
            A new initializer instance with the specified configuration.
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return a string representation of the initializer."""
        return f"OrthonormalInitializer(seed={self.seed})"

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"OrthonormalInitializer with seed={self.seed}"

# ---------------------------------------------------------------------

