import keras
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class OrthogonalHypersphereInitializer(keras.initializers.Initializer):
    """
    Orthogonal hypersphere weight initializer with mathematical dimensionality constraints.

    This initializer creates weight vectors that are mutually orthogonal and lie on a
    hypersphere of specified radius. It handles the fundamental mathematical constraint
    that the number of orthogonal vectors cannot exceed the dimensionality of the space
    they occupy.

    **Intent**: Provide geometrically well-separated initial weights for neural networks
    where weight diversity is crucial, such as in mixture models, attention mechanisms,
    or embedding layers where distinct weight vectors improve learning dynamics.

    **Mathematical Behavior**:

    1. **Feasible Case** (num_vectors ≤ latent_dim):
       - Generates perfectly orthogonal vectors using QR decomposition
       - Each vector has exact radius magnitude: ||v_i|| = radius
       - All vectors satisfy orthogonality: v_i · v_j = 0 for i ≠ j
       - Process: Random Matrix → QR Decomposition → Scale by radius

    2. **Infeasible Case** (num_vectors > latent_dim):
       - Falls back to uniform distribution on hypersphere surface
       - Vectors are maximally separated but not perfectly orthogonal
       - Issues warning about mathematical impossibility
       - Process: Random Gaussian → Normalize → Scale by radius

    **Geometric Properties**:
    - All vectors lie exactly on hypersphere: ||v|| = radius
    - Maximum angular separation when orthogonal
    - Deterministic given same seed (reproducible)

    Args:
        radius: Float, radius of the hypersphere. All initialized vectors will have
            this L2 norm magnitude. Must be positive. Defaults to 1.0.
        seed: Optional integer, random seed for reproducible initialization.
            Uses numpy's random number generator. Defaults to None.

    Returns:
        KerasTensor: Initialized weight tensor of the requested shape with vectors
        positioned on hypersphere surface.

    Raises:
        ValueError: If radius is not positive.

    Example:
        ```python
        # Basic usage - orthogonal vectors on unit hypersphere
        initializer = OrthogonalHypersphereInitializer()
        weights = initializer(shape=(10, 128))  # 10 vectors in 128D space

        # Custom radius and seed for reproducibility
        initializer = OrthogonalHypersphereInitializer(radius=2.0, seed=42)
        weights = initializer(shape=(64, 256))  # Feasible: 64 < 256

        # Infeasible case (will fall back with warning)
        initializer = OrthogonalHypersphereInitializer(radius=1.5)
        weights = initializer(shape=(512, 128))  # Infeasible: 512 > 128

        # In a layer
        layer = keras.layers.Dense(
            units=64,
            kernel_initializer=OrthogonalHypersphereInitializer(radius=1.5)
        )
        ```

    Note:
        This initializer is particularly useful for:
        - Embedding layers where diverse initial vectors improve learning
        - Attention mechanisms requiring well-separated query/key vectors
        - Mixture of experts where expert specialization benefits from orthogonality
        - Any scenario where weight vector diversity is more important than magnitude

    Mathematical Background:
        The maximum number of mutually orthogonal vectors in n-dimensional space is n.
        When more vectors are requested, uniform hypersphere distribution provides
        the next-best geometric property: maximum average angular separation.
    """

    def __init__(
            self,
            radius: float = 1.0,
            seed: Optional[int] = None
    ) -> None:
        # Validate inputs
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")

        # Store configuration
        self.radius = float(radius)
        self.seed = seed

    def __call__(
            self,
            shape: Tuple[int, ...],
            dtype: Optional[str] = None
    ) -> keras.KerasTensor:
        """
        Generate orthogonal hypersphere initialization tensor.

        Args:
            shape: Tuple of integers specifying tensor shape. Expected format is
                (..., latent_dim) where latent_dim is the vector dimensionality.
            dtype: Optional string specifying data type. Defaults to backend default.

        Returns:
            KerasTensor of specified shape with orthogonal vectors on hypersphere.
        """
        if not shape:
            raise ValueError("shape cannot be empty")
        if len(shape) < 1:
            raise ValueError("shape must have at least one dimension")

        # Extract grid shape and latent dimension
        grid_shape = shape[:-1]
        latent_dim = shape[-1]

        if latent_dim <= 0:
            raise ValueError(f"latent dimension must be positive, got {latent_dim}")

        # Calculate total number of vectors needed
        num_vectors = int(np.prod(grid_shape)) if grid_shape else 1

        # Handle dimensionality constraint
        if num_vectors > latent_dim:
            # Mathematically impossible case - fall back to uniform hypersphere
            warnings.warn(
                f"Orthogonality constraint violation: requesting {num_vectors} orthogonal vectors "
                f"in {latent_dim}-dimensional space (maximum possible: {latent_dim}). "
                f"Falling back to uniform hypersphere distribution for maximum separation.",
                UserWarning,
                stacklevel=2
            )

            # Generate uniform distribution on hypersphere
            initialized_vectors = self._generate_uniform_hypersphere(num_vectors, latent_dim)

        else:
            # Feasible case - generate true orthogonal set
            initialized_vectors = self._generate_orthogonal_set(num_vectors, latent_dim)

        # Reshape to requested shape and convert to KerasTensor
        final_weights = keras.ops.reshape(
            keras.ops.convert_to_tensor(initialized_vectors, dtype=dtype),
            shape
        )

        return final_weights

    def _generate_orthogonal_set(self, num_vectors: int, latent_dim: int) -> np.ndarray:
        """
        Generate truly orthogonal vectors using QR decomposition.

        Args:
            num_vectors: Number of orthogonal vectors to generate.
            latent_dim: Dimensionality of the latent space.

        Returns:
            Array of shape (num_vectors, latent_dim) with orthogonal vectors.
        """
        # Set up random number generator
        rng = np.random.default_rng(self.seed)

        # Generate random matrix for QR decomposition
        # Shape: (latent_dim, num_vectors) for proper QR factorization
        random_matrix = rng.normal(size=(latent_dim, num_vectors)).astype(np.float32)

        # QR decomposition gives orthonormal columns in Q
        q_matrix, _ = np.linalg.qr(random_matrix)

        # Extract first num_vectors columns and transpose to get row vectors
        # Shape: (num_vectors, latent_dim)
        orthonormal_vectors = q_matrix[:, :num_vectors].T

        # Scale to desired radius
        orthogonal_vectors = orthonormal_vectors * self.radius

        return orthogonal_vectors

    def _generate_uniform_hypersphere(self, num_vectors: int, latent_dim: int) -> np.ndarray:
        """
        Generate vectors uniformly distributed on hypersphere surface.

        Args:
            num_vectors: Number of vectors to generate.
            latent_dim: Dimensionality of the latent space.

        Returns:
            Array of shape (num_vectors, latent_dim) with vectors on hypersphere.
        """
        # Set up random number generator
        rng = np.random.default_rng(self.seed)

        # Generate random vectors from multivariate Gaussian
        random_vectors = rng.normal(size=(num_vectors, latent_dim)).astype(np.float32)

        # Normalize each vector to unit length (avoid division by zero)
        vector_norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
        vector_norms = np.maximum(vector_norms, 1e-8)  # Numerical stability
        unit_vectors = random_vectors / vector_norms

        # Scale to desired radius
        scaled_vectors = unit_vectors * self.radius

        return scaled_vectors

    def get_config(self) -> Dict[str, Any]:
        """
        Get initializer configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters needed to
            reconstruct this initializer instance.
        """
        return {
            "radius": self.radius,
            "seed": self.seed
        }

    def __repr__(self) -> str:
        """String representation of the initializer."""
        return (
            f"{self.__class__.__name__}("
            f"radius={self.radius}, seed={self.seed})"
        )

# ---------------------------------------------------------------------
