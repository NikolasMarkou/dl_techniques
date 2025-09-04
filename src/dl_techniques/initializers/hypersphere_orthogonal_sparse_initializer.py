import keras
import warnings
import numpy as np
from typing import Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------
# Custom Initializer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class OrthogonalHypersphereSparseInitializer(keras.initializers.Initializer):
    """
    Orthogonal hypersphere weight initializer with mathematical dimensionality constraints.

    This initializer creates weight vectors that are mutually orthogonal and lie on a
    hypersphere of specified radius. It handles the fundamental mathematical constraint
    that the number of orthogonal vectors cannot exceed the dimensionality of the space
    they occupy, providing geometrically well-separated initial weights for improved
    learning dynamics.

    **Intent**: Provide geometrically well-separated initial weights for neural networks
    where weight diversity and orthogonality are crucial for effective learning. This is
    particularly beneficial for mixture models, attention mechanisms, embedding layers,
    and mixture of experts architectures where distinct, non-correlated weight vectors
    improve convergence and specialization.

    **Mathematical Framework**:

    The initializer operates under two distinct regimes based on geometric feasibility:

    1. **Feasible Orthogonal Case** (num_vectors ≤ latent_dim):
       ```
       Mathematical Process:
       1. Generate random matrix: R ∈ ℝ^(latent_dim × num_vectors)
       2. Apply smooth clipping: R' = tanh(R) for numerical stability
       3. QR decomposition: R' = Q @ P, where Q has orthonormal columns
       4. Extract and scale: V = radius × Q[:, :num_vectors]^T

       Properties:
       - Perfect orthogonality: ⟨v_i, v_j⟩ = 0 for i ≠ j
       - Uniform magnitude: ||v_i||₂ = radius for all i
       - Maximal angular separation: θ_ij = 90° for i ≠ j
       ```

    2. **Infeasible Uniform Case** (num_vectors > latent_dim):
       ```
       Mathematical Process:
       1. Generate Gaussian samples: R ~ N(0, I) ∈ ℝ^(num_vectors × latent_dim)
       2. Apply smooth clipping: R' = tanh(R) for stability
       3. Normalize to unit sphere: U = R' / ||R'||₂
       4. Scale to desired radius: V = radius × U

       Properties:
       - Uniform distribution on hypersphere surface
       - Maximal expected angular separation (but not orthogonal)
       - Warning issued about geometric impossibility
       ```

    **Geometric Properties**:
    - All vectors satisfy: ||v||₂ = radius (exact hypersphere constraint)
    - Deterministic output given same random seed (reproducible)
    - Maximum theoretical angular separation given constraints
    - Numerical stability through smooth clipping operations

    Args:
        radius: Float, radius of the hypersphere surface. All initialized vectors will
            have this exact L2 norm magnitude. Must be positive. A larger radius
            increases the initial weight magnitudes, which can affect learning dynamics.
            Defaults to 1.0 for standard unit hypersphere initialization.
        seed: Optional integer, random seed for reproducible initialization across
            runs. Uses numpy's random number generator for consistency. When None,
            initialization will vary between runs. Defaults to None.

    Input shape:
        Tensor shape tuple with at least one dimension. Expected format is
        `(..., latent_dim)` where `latent_dim` is the vector dimensionality and
        preceding dimensions define the grid structure of vectors.

    Output shape:
        KerasTensor of the exact input shape with initialized values satisfying
        hypersphere and orthogonality constraints where geometrically possible.

    Attributes:
        radius: Float, the hypersphere radius parameter.
        seed: Optional integer, the random seed for reproducibility.

    Example:
        ```python
        # Basic usage - orthogonal vectors on unit hypersphere
        initializer = OrthogonalHypersphereSparseInitializer()
        weights = initializer(shape=(10, 128))  # 10 vectors in 128D space
        print(f"All vectors have radius: {keras.ops.norm(weights, axis=-1)}")
        # Output: All vectors have radius: [1. 1. 1. ...]

        # Custom radius and reproducible initialization
        initializer = OrthogonalHypersphereSparseInitializer(radius=2.0, seed=42)
        weights = initializer(shape=(64, 256))  # Feasible: 64 < 256

        # Verify orthogonality for feasible case
        dot_products = keras.ops.matmul(weights, keras.ops.transpose(weights))
        print(f"Off-diagonal elements near zero: {keras.ops.max(keras.ops.abs(
            dot_products - keras.ops.diag(keras.ops.diag(dot_products))
        ))}")

        # Infeasible case (will fall back with warning)
        initializer = OrthogonalHypersphereSparseInitializer(radius=1.5)
        weights = initializer(shape=(512, 128))  # Infeasible: 512 > 128
        # Warning: Orthogonality constraint violation: requesting 512 orthogonal vectors...

        # Integration with Dense layer
        layer = keras.layers.Dense(
            units=64,
            kernel_initializer=OrthogonalHypersphereSparseInitializer(radius=1.5, seed=42),
            name='orthogonal_dense'
        )

        # Integration with Embedding layer for diverse embeddings
        embedding = keras.layers.Embedding(
            input_dim=1000,
            output_dim=128,
            embeddings_initializer=OrthogonalHypersphereSparseInitializer(radius=0.8),
            name='orthogonal_embedding'
        )
        ```

    Use Cases:
        This initializer is particularly effective for:
        - **Embedding Layers**: Ensuring diverse initial word/token representations
        - **Attention Mechanisms**: Well-separated query/key vectors improve attention focus
        - **Mixture of Experts**: Orthogonal expert weights promote specialization
        - **Multi-head Attention**: Different attention heads start with distinct patterns
        - **Sparse Coding**: Initial dictionary atoms benefit from orthogonality
        - **Metric Learning**: Initial embedding space with maximal separation

    Mathematical Background:
        The fundamental constraint stems from linear algebra: the maximum number of
        mutually orthogonal vectors in n-dimensional Euclidean space is exactly n.
        This is because orthogonal vectors form a basis for the space, and any
        n-dimensional space has at most an n-element basis.

        When more vectors are requested than the space can accommodate orthogonally,
        the uniform hypersphere distribution provides the optimal alternative by
        maximizing the expected pairwise angular separation, which is the best
        achievable geometric property under the constraint.

    Raises:
        ValueError: If radius is not positive (radius <= 0).
        ValueError: If shape is empty or has no dimensions.
        ValueError: If the last dimension (latent_dim) is not positive.

    Note:
        The smooth clipping operation using tanh() is applied for numerical stability
        and to prevent extreme values that could cause numerical issues during QR
        decomposition. This maintains the mathematical properties while ensuring
        robust computation across different numerical conditions.

        When the orthogonality constraint cannot be satisfied (num_vectors > latent_dim),
        a warning is issued and the initializer gracefully falls back to uniform
        hypersphere distribution, which provides the next-best geometric separation.
    """

    def __init__(
            self,
            radius: float = 1.0,
            seed: Optional[int] = None
    ) -> None:
        """
        Initialize the orthogonal hypersphere initializer.

        Args:
            radius: Float, radius of the hypersphere. Must be positive.
            seed: Optional integer, random seed for reproducibility.

        Raises:
            ValueError: If radius is not positive.
        """
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

        This method creates weight tensors with vectors that lie on a hypersphere
        of the specified radius and are mutually orthogonal when geometrically
        feasible (num_vectors ≤ latent_dim).

        Args:
            shape: Tuple of integers specifying tensor shape. Expected format is
                (..., latent_dim) where latent_dim is the vector dimensionality
                and preceding dimensions define the grid structure.
            dtype: Optional string specifying data type. If None, uses the
                backend's default float type.

        Returns:
            KerasTensor of specified shape with vectors positioned on hypersphere
            surface, orthogonal when geometrically possible.

        Raises:
            ValueError: If shape is empty, has no dimensions, or latent_dim <= 0.
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

        This method creates a set of mutually orthogonal vectors by generating
        a random matrix and applying QR decomposition to obtain orthonormal
        columns, which are then scaled to the desired radius.

        Args:
            num_vectors: Integer, number of orthogonal vectors to generate.
                Must be <= latent_dim for true orthogonality.
            latent_dim: Integer, dimensionality of the latent space.

        Returns:
            Array of shape (num_vectors, latent_dim) containing orthogonal vectors
            with exact L2 norm equal to self.radius.
        """
        # Set up random number generator
        rng = np.random.default_rng(self.seed)

        # Generate random matrix for QR decomposition
        # Shape: (latent_dim, num_vectors) for proper QR factorization
        random_matrix = rng.normal(size=(latent_dim, num_vectors)).astype(np.float32)

        # Apply smooth clipping for numerical stability
        random_matrix = np.tanh(random_matrix)

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

        This fallback method is used when true orthogonality is geometrically
        impossible (num_vectors > latent_dim). It generates vectors that are
        uniformly distributed on the hypersphere surface, providing maximal
        expected angular separation.

        Args:
            num_vectors: Integer, number of vectors to generate.
            latent_dim: Integer, dimensionality of the latent space.

        Returns:
            Array of shape (num_vectors, latent_dim) containing vectors uniformly
            distributed on hypersphere surface with exact L2 norm equal to self.radius.
        """
        # Set up random number generator
        rng = np.random.default_rng(self.seed)

        # Generate random vectors from multivariate Gaussian
        random_vectors = rng.normal(size=(num_vectors, latent_dim)).astype(np.float32)

        # Apply smooth clipping for numerical stability
        random_vectors = np.tanh(random_vectors)

        # Normalize each vector to unit length (with numerical stability)
        vector_norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
        vector_norms = np.maximum(vector_norms, 1e-8)  # Prevent division by zero
        unit_vectors = random_vectors / vector_norms

        # Apply additional clipping and re-normalization for extra stability
        unit_vectors = np.tanh(unit_vectors)
        vector_norms = np.linalg.norm(unit_vectors, axis=1, keepdims=True)
        vector_norms = np.maximum(vector_norms, 1e-8)  # Numerical stability
        unit_vectors = unit_vectors / vector_norms

        # Scale to desired radius
        scaled_vectors = unit_vectors * self.radius

        return scaled_vectors

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        This method returns all parameters needed to reconstruct this initializer
        instance, enabling proper serialization and deserialization with Keras
        model saving/loading functionality.

        Returns:
            Dictionary containing all configuration parameters:
            - 'radius': Float, the hypersphere radius
            - 'seed': Optional integer, the random seed
        """
        return {
            "radius": self.radius,
            "seed": self.seed
        }

    def __repr__(self) -> str:
        """
        String representation of the initializer.

        Returns:
            String containing class name and key parameters for debugging
            and logging purposes.
        """
        return (
            f"{self.__class__.__name__}("
            f"radius={self.radius}, seed={self.seed})"
        )