"""
Rigid Simplex Layer with learnable rotation and bounded scaling.

This module implements a constrained projection layer that maintains fixed
geometric structure while allowing training for rotation alignment and scaling.

Mathematical Background: Equiangular Tight Frames (ETF)
========================================================

In Linear Algebra, a Regular Simplex centered at the origin is a specific type
of matrix called an **Equiangular Tight Frame (ETF)**.

1. The Vectors: "Maximum Repulsion"
-----------------------------------

Geometrically, a simplex consists of points equidistant from each other. In
Linear Algebra, we treat these points as vectors originating from the origin.

If you have N dimensions, you can fit N orthogonal (perpendicular) vectors
with dot product exactly 0. However, if you want to squeeze **N+1** vectors
into that same space while keeping them perfectly symmetric, they cannot be
orthogonal (90 degrees). They must "push away" from each other as much as
mathematically possible.

**The Linear Algebra Rule:**

For a regular simplex centered at the origin with unit vectors v_i:

- **Self-alignment:** v_i . v_i = 1 (Length is 1)
- **Cross-alignment:** v_i . v_j = -1/N (for i != j)

This negative dot product means every vector is slightly pointing *away*
from every other vector.

2. The Gram Matrix
------------------

If you stack these vectors into a matrix V (where each row is a point), and
calculate the correlation matrix (Gram Matrix G = V @ V.T), you get a very
specific, beautiful structure.

For a 2D Simplex (Triangle, 3 points), N=2::

    G = [[ 1.0, -0.5, -0.5],
         [-0.5,  1.0, -0.5],
         [-0.5, -0.5,  1.0]]

For an N-dimensional Simplex (N+1 points)::

    G[i,i] = 1      (diagonal)
    G[i,j] = -1/N   (off-diagonal, i != j)

**Why this matters for Neural Networks:**

Minimizing the off-diagonal elements of this matrix is called "minimizing
coherence." The Simplex is the theoretical limit of how small you can make
the off-diagonal elements for N+1 vectors. This maximizes the **diversity**
of the neurons.

3. Linear Dependence (Rank)
---------------------------

This is the most counter-intuitive part:

- You have N+1 vectors (rows)
- You are in N dimensional space (columns)

Therefore, the matrix is **Rank Deficient**. The Rank is N, not N+1.
This means the vectors are **Linearly Dependent**::

    sum(v_i for i in 1..N+1) = 0

**In English:** If you sum up all the vectors in a simplex, they perfectly
cancel each other out to zero.

**In Physics:** It is a state of perfect equilibrium. If you put identical
springs between all the vertices, the net force on the center is zero.

4. Eigenvalues (The "Isometry")
-------------------------------

If V is our simplex matrix (shape (N+1, N)):

1. **V @ V.T (Gram Matrix):** Describes the angles between points.
   Has 1s on diagonal and -1/N elsewhere.

2. **V.T @ V (Covariance Matrix):** Describes how the space is stretched.

For a regular simplex::

    V.T @ V = ((N+1) / N) * I

This is the "Holy Grail" property for Deep Learning initialization:

- **Isometry:** The matrix preserves the magnitude of gradients. It doesn't
  stretch or shrink the signal as it passes through.
- **Whiteness:** The features defined by these vectors are perfectly
  uncorrelated in the embedding space.

Summary
-------

This module generates a matrix W such that:

1. **Rows are normalized:** ||w_i|| = 1
2. **Rows are maximally separated:** w_i . w_j = -1/N
3. **Columns are orthogonal:** W.T @ W is proportional to I

It creates a "rigid crystal" of vectors that are perfectly balanced in space,
ensuring that no direction is favored over another and every neuron is as
different from its neighbors as mathematically possible.
"""

import keras
import numpy as np
from keras import ops
from keras import initializers
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RigidSimplexLayer(keras.layers.Layer):
    """
    Projects inputs onto a fixed Simplex structure with learnable rotation and scaling.

    **Intent**: Implement a layer that maintains rigid geometric structure (Simplex)
    while allowing training only for rotation alignment and bounded scaling. This
    forces the network to perform template matching rather than learning arbitrary
    filter shapes.

    **Architecture**:
    ```
    Input(shape=[batch, ..., input_dim])
           ↓
    MatMul with rotation_kernel: [input_dim, input_dim]
           ↓
    Rotated Input (aligned to Simplex orientation)
           ↓
    MatMul with static_simplex: [input_dim, units] (NON-TRAINABLE)
           ↓
    Multiply by global_scale: scalar (BOUNDED to [scale_min, scale_max])
           ↓
    Output(shape=[batch, ..., units])
    ```

    **Training Constraints**:
    - `static_simplex`: Fixed geometry, trainable=False
    - `rotation_kernel`: Learns optimal input rotation (soft orthogonality via loss)
    - `global_scale`: Bounded scalar between [scale_min, scale_max]

    **Theoretical Implications**:
    By using this layer, the network is forced to perform **Template Matching**:
    - Standard Dense: "Learn any shape of filters that separate the data."
    - This Layer: "Rotate the input data until it aligns with a fixed crystal structure."

    Args:
        units: Dimensionality of the output space (number of Simplex projections).
        scale_min: Minimum allowed scaling factor. Defaults to 0.5.
        scale_max: Maximum allowed scaling factor. Defaults to 2.0.
        orthogonality_penalty: Weight for orthogonality regularization loss
            (encourages R^T R ≈ I). Defaults to 1e-4.
        rotation_initializer: Initializer for rotation matrix. Defaults to 'identity'.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        N-D tensor: `(batch_size, ..., input_dim)`.

    Output shape:
        N-D tensor: `(batch_size, ..., units)`.

    Note:
        The layer adds an orthogonality regularization loss during training to
        encourage the rotation kernel to remain a valid rotation matrix.

    Example:
        >>> layer = RigidSimplexLayer(units=64, scale_min=0.1, scale_max=5.0)
        >>> inputs = keras.random.normal((32, 128))
        >>> outputs = layer(inputs)  # Shape: (32, 64)
    """

    def __init__(
            self,
            units: int,
            scale_min: float = 0.5,
            scale_max: float = 2.0,
            orthogonality_penalty: float = 1e-4,
            rotation_initializer: Union[str, initializers.Initializer] = 'identity',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if scale_min >= scale_max:
            raise ValueError(
                f"scale_min ({scale_min}) must be less than scale_max ({scale_max})"
            )
        if orthogonality_penalty < 0:
            raise ValueError(
                f"orthogonality_penalty must be non-negative, got {orthogonality_penalty}"
            )

        # Store configuration
        self.units = units
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.orthogonality_penalty = orthogonality_penalty
        self.rotation_initializer = initializers.get(rotation_initializer)

        # Weight attributes - created in build()
        self.static_simplex = None
        self.rotation_kernel = None
        self.global_scale = None
        self._input_dim = None

    def _create_simplex_matrix(
            self,
            input_dim: int,
            output_dim: int
    ) -> np.ndarray:
        """
        Generate Simplex weight matrix.

        Creates a normalized Simplex structure in input_dim space with output_dim
        vertices. The Simplex is centered at origin with unit-normalized vertices.

        A natural Simplex has N+1 vertices in N dimensions. This method generates
        the Simplex and tiles/slices to match the requested output dimensionality.

        Args:
            input_dim: Input dimensionality.
            output_dim: Output dimensionality (number of Simplex projections).

        Returns:
            Weight matrix of shape (input_dim, output_dim) as float32.
        """
        dimensions = input_dim
        matrix = np.identity(dimensions, dtype=np.float32)

        # Calculate the last point to be equidistant from all others
        # This creates a regular simplex in N dimensions
        last_point = np.ones((1, dimensions), dtype=np.float32) * \
                     ((1.0 + np.sqrt(dimensions + 1.0)) / dimensions)

        matrix = np.vstack([matrix, last_point])

        # Center points at origin
        mean_m = np.mean(matrix, axis=0)
        matrix = matrix - mean_m

        # Normalize to unit vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        matrix = matrix / norms

        # Transpose to get (input_dim, N+1) shape
        W = matrix.T

        # Tile or slice to match requested output_dim
        current_cols = W.shape[1]

        if output_dim > current_cols:
            tile_factor = int(np.ceil(output_dim / current_cols))
            W = np.tile(W, (1, tile_factor))

        W = W[:, :output_dim]

        return W.astype(np.float32)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's weights.

        Creates:
        - static_simplex: Non-trainable Simplex geometry (input_dim, units)
        - rotation_kernel: Trainable rotation matrix (input_dim, input_dim)
        - global_scale: Trainable bounded scalar (1,)

        Args:
            input_shape: Shape tuple of input tensor.

        Raises:
            ValueError: If last dimension of input is not defined.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        self._input_dim = input_dim

        # 1. Static Simplex (frozen weights - geometry remains rigid)
        simplex_weights = self._create_simplex_matrix(input_dim, self.units)
        self.static_simplex = self.add_weight(
            name='static_simplex',
            shape=(input_dim, self.units),
            initializer=initializers.Constant(simplex_weights),
            trainable=False,
            dtype=self.dtype,
        )

        # 2. Trainable rotation matrix (learns optimal input alignment)
        self.rotation_kernel = self.add_weight(
            name='rotation_kernel',
            shape=(input_dim, input_dim),
            initializer=self.rotation_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        # 3. Bounded scaling factor
        self.global_scale = self.add_weight(
            name='global_scale',
            shape=(1,),
            initializer=initializers.Constant(1.0),
            constraint=ValueRangeConstraint(min_value=self.scale_min, max_value=self.scale_max),
            trainable=True,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass computation.

        Applies rotation to inputs, projects onto static Simplex, and scales output.
        Adds orthogonality regularization loss for the rotation kernel.

        Args:
            inputs: Input tensor of shape (batch, ..., input_dim).
            training: Boolean flag for training mode (unused but kept for API).

        Returns:
            Output tensor of shape (batch, ..., units).
        """
        # 1. Add orthogonality regularization loss (soft constraint for rotation)
        # R^T * R should approximate Identity for valid rotation
        r_t_r = ops.matmul(
            ops.transpose(self.rotation_kernel),
            self.rotation_kernel
        )
        identity = ops.eye(self._input_dim, dtype=self.dtype)
        ortho_loss = ops.mean(ops.square(r_t_r - identity))
        self.add_loss(self.orthogonality_penalty * ortho_loss)

        # 2. Rotate inputs to align with Simplex
        rotated_inputs = ops.matmul(inputs, self.rotation_kernel)

        # 3. Project onto static Simplex
        outputs = ops.matmul(rotated_inputs, self.static_simplex)

        # 4. Apply bounded scaling
        outputs = outputs * self.global_scale

        return outputs

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape from input shape.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple with last dimension replaced by units.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Configuration dictionary containing all __init__ parameters.
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'scale_min': self.scale_min,
            'scale_max': self.scale_max,
            'orthogonality_penalty': self.orthogonality_penalty,
            'rotation_initializer': initializers.serialize(self.rotation_initializer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RigidSimplexLayer':
        """
        Create layer from configuration.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            New RigidSimplexLayer instance.
        """
        if 'rotation_initializer' in config:
            config['rotation_initializer'] = initializers.deserialize(
                config['rotation_initializer']
            )
        return cls(**config)

# ---------------------------------------------------------------------
