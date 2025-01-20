import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Union, Literal, List, Any, Tuple, Dict

# ---------------------------------------------------------------------

class OrthonormalInitializer(keras.initializers.Initializer):
    """Custom initializer for orthonormal centroids.

    This initializer creates a set of orthonormal vectors using QR decomposition
    and ensures numerical stability.

    Args:
        seed: Optional[int]
            Random seed for initialization
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the orthonormal initializer."""
        self.seed = seed

    def __call__(self, shape: Tuple[int, ...], dtype: Any = None) -> tf.Tensor:
        """Generate orthonormal vectors.

        Args:
            shape: Tuple[int, ...]
                Desired shape of the output tensor (n_clusters, feature_dims)
            dtype: Any
                Desired dtype of the output tensor

        Returns:
            tf.Tensor: Orthonormal vectors of shape (n_clusters, feature_dims)

        Raises:
            ValueError: If n_clusters > feature_dims (cannot create orthogonal vectors)
        """
        n_clusters, feature_dims = shape

        if n_clusters > feature_dims:
            raise ValueError(
                f"Cannot create {n_clusters} orthogonal vectors in "
                f"{feature_dims}-dimensional space. n_clusters must be <= feature_dims"
            )

        # Create random matrix
        rng = np.random.RandomState(self.seed)
        random_matrix = rng.randn(feature_dims, feature_dims).astype('float32')

        # Compute QR decomposition
        q, r = tf.linalg.qr(random_matrix)

        # Ensure consistent signs (make diagonal of R positive)
        d = tf.linalg.diag_part(r)
        ph = tf.cast(tf.sign(d), dtype=q.dtype)
        q *= tf.expand_dims(ph, axis=0)

        # Take first n_clusters rows
        orthonormal_vectors = q[:n_clusters, :]

        return tf.cast(orthonormal_vectors, dtype=dtype)

    def get_config(self) -> Dict[str, Any]:
        """Get initializer configuration."""
        return {'seed': self.seed}

# ---------------------------------------------------------------------
