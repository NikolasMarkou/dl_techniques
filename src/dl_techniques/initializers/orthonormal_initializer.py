"""
OrthonormalInitializer: A Keras initializer for generating orthonormal vectors.

This module implements a custom Keras initializer that creates sets of orthonormal vectors
using QR decomposition. Orthonormal vectors are particularly useful in machine learning
applications such as:

1. Clustering algorithms (e.g., k-means) where orthogonal centroids can improve convergence
2. Self-organizing maps where orthogonal weight initialization can lead to better space coverage
3. Representation learning where orthogonal bases can improve feature disentanglement
4. Deep neural networks where orthogonal weight matrices help mitigate vanishing/exploding gradients

The implementation follows these key steps:
1. Generate a random matrix using NumPy's random number generator
2. Apply QR decomposition to obtain orthogonal vectors (Q matrix)
3. Ensure numerical stability by fixing the signs based on the diagonal of R
4. Extract the first n_clusters rows to get the desired number of orthogonal vectors

This approach guarantees that the resulting vectors are orthonormal (orthogonal and unit length)
and provides better numerical stability compared to simpler approaches like Gram-Schmidt
orthogonalization.

Mathematical background:
A set of vectors {v₁, v₂, ..., vₙ} is orthonormal if:
- ⟨vᵢ, vⱼ⟩ = 0 for all i ≠ j (orthogonality)
- ‖vᵢ‖ = 1 for all i (unit length)

Where ⟨·,·⟩ denotes the inner product and ‖·‖ the Euclidean norm.

References:
    [1] Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). Exact solutions to the
        nonlinear dynamics of learning in deep linear neural networks.
        arXiv preprint arXiv:1312.6120.

    [2] Hu, T., Pehlevan, C., & Chklovskii, D. B. (2014). A Hebbian/anti-Hebbian network
        for online sparse dictionary learning derived from symmetric matrix factorization.
        In 2014 48th Asilomar Conference on Signals, Systems and Computers (pp. 613-619). IEEE.

    [3] Mishkin, D., & Matas, J. (2015). All you need is a good init.
        arXiv preprint arXiv:1511.06422.

    [4] Bansal, N., Chen, X., & Wang, Z. (2018). Can we gain more from orthogonality
        regularizations in training deep networks?
        Advances in Neural Information Processing Systems, 31.

Note:
    The implementation enforces that the number of clusters (n_clusters) must be less than
    or equal to the feature dimensions. This is a mathematical constraint as you cannot
    have more than n orthogonal vectors in an n-dimensional space.

Example usage:
    ```python
    # Initialize a Dense layer with orthonormal weights
    dense = keras.layers.Dense(
        units=64,
        kernel_initializer=OrthonormalInitializer(seed=42)
    )

    # Initialize clustering centroids orthogonally
    centroids = OrthonormalInitializer(seed=123)((10, 128))
    ```
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Union, Literal, List, Any, Tuple, Dict

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
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
