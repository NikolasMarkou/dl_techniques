import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..embedding.continuous_sin_cos_embedding import ContinuousSinCosEmbed

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SupernodePooling(keras.layers.Layer):
    """Supernode pooling layer with message passing for point clouds.

    This layer implements a graph-based pooling mechanism that:
    1. Selects supernode positions from input points
    2. Finds neighbors of supernodes within a radius or k-NN
    3. Aggregates information from neighbors using message passing
    4. Outputs aggregated features for each supernode

    This is particularly useful for point cloud processing where you need
    to pool information from local neighborhoods while preserving spatial
    relationships through learned embeddings.

    Args:
        hidden_dim: Integer, output dimension of the pooled features. Must be positive.
        ndim: Integer, number of coordinate dimensions (2 for 2D, 3 for 3D).
            Must be positive.
        radius: Optional float, radius for neighbor selection. If provided,
            all points within this radius of each supernode are considered.
            Cannot be used together with k_neighbors. Must be positive if specified.
        k_neighbors: Optional integer, number of nearest neighbors to consider
            for each supernode. Cannot be used together with radius.
            Must be positive if specified.
        max_neighbors: Integer, maximum number of neighbors per supernode
            to prevent memory issues. Must be positive. Defaults to 32.
        mode: String, positional encoding mode. Either "abspos" (absolute
            positions) or "relpos" (relative positions). Defaults to "relpos".
        activation: String or callable, activation function for message MLP.
            Defaults to "gelu".
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: String or Initializer, initializer for kernel weights.
            Defaults to "glorot_uniform".
        bias_initializer: String or Initializer, initializer for bias vectors.
            Defaults to "zeros".
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Dictionary with keys:
        - "positions": 2D tensor with shape `(num_points, ndim)` - point coordinates
        - "supernode_indices": 1D tensor with shape `(num_supernodes,)` - indices
          of points that should serve as supernodes

    Output shape:
        3D tensor with shape: `(batch_size, num_supernodes, hidden_dim)`

    Call arguments:
        inputs: Dictionary containing:
            - positions: Point coordinates tensor of shape (num_points, ndim)
            - supernode_indices: Supernode indices tensor of shape (num_supernodes,)
        training: Boolean indicating training mode.

    Returns:
        Pooled supernode features tensor.

    Example:
        ```python
        # 3D point cloud with 1000 points
        positions = keras.random.uniform((1000, 3)) * 10
        supernode_indices = keras.random.uniform((100,), 0, 1000, seed=42)
        supernode_indices = ops.cast(supernode_indices, "int32")

        pooling = SupernodePooling(
            hidden_dim=256,
            ndim=3,
            radius=2.0,
            mode="relpos"
        )

        features = pooling({
            "positions": positions,
            "supernode_indices": supernode_indices
        })
        print(features.shape)  # (1, 100, 256)
        ```

    Raises:
        ValueError: If exactly one of radius or k_neighbors is not specified.
        ValueError: If hidden_dim, ndim, or max_neighbors are not positive.
        ValueError: If radius or k_neighbors are specified but not positive.
        ValueError: If mode is not "abspos" or "relpos".

    Notes:
        This implementation approximates graph operations using dense computations
        for Keras compatibility. For very large point clouds, consider using
        specialized graph neural network libraries.
    """

    def __init__(
            self,
            hidden_dim: int,
            ndim: int,
            radius: Optional[float] = None,
            k_neighbors: Optional[int] = None,
            max_neighbors: int = 32,
            mode: str = "relpos",
            activation: Union[str, callable] = "gelu",
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if ndim <= 0:
            raise ValueError(f"ndim must be positive, got {ndim}")
        if max_neighbors <= 0:
            raise ValueError(f"max_neighbors must be positive, got {max_neighbors}")

        # Validate neighbor selection parameters
        if (radius is None) == (k_neighbors is None):
            raise ValueError("Exactly one of radius or k_neighbors must be specified")

        if radius is not None and radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if k_neighbors is not None and k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")

        if mode not in ["abspos", "relpos"]:
            raise ValueError(f"mode must be 'abspos' or 'relpos', got {mode}")

        # Store ALL configuration
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.radius = radius
        self.k_neighbors = k_neighbors
        self.max_neighbors = max_neighbors
        self.mode = mode
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)

        # Position embedding for absolute positions
        self.pos_embed = ContinuousSinCosEmbed(
            dim=self.hidden_dim,
            ndim=self.ndim,
            name="pos_embed"
        )

        # Conditional relative position embedding
        if self.mode == "relpos":
            # Relative position embedding (includes distance magnitude)
            self.rel_pos_embed = ContinuousSinCosEmbed(
                dim=self.hidden_dim,
                ndim=self.ndim + 1,  # relative coords + distance magnitude
                assert_positive=False,
                name="rel_pos_embed"
            )
        else:
            self.rel_pos_embed = None

        # Message passing MLP
        if self.mode == "abspos":
            message_input_dim = self.hidden_dim * 2  # source + target pos embeddings
        else:
            message_input_dim = self.hidden_dim

        self.message_mlp = keras.Sequential([
            keras.layers.Dense(
                self.hidden_dim,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="message_dense1"
            ),
            keras.layers.Dense(
                self.hidden_dim,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="message_dense2"
            )
        ], name="message_mlp")

        # Projection layer to combine message and position embeddings
        self.proj_layer = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="proj"
        )

    def build(self, input_shape: Union[Dict[str, Tuple[Optional[int], ...]], Any]) -> None:
        """Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        if not isinstance(input_shape, dict):
            raise ValueError("Input shape must be a dictionary with 'positions' and 'supernode_indices' keys")

        if "positions" not in input_shape:
            raise ValueError("Input must contain 'positions' key")
        if "supernode_indices" not in input_shape:
            raise ValueError("Input must contain 'supernode_indices' key")

        pos_shape = input_shape["positions"]
        if len(pos_shape) != 2 or pos_shape[-1] != self.ndim:
            raise ValueError(f"positions must have shape (num_points, {self.ndim}), got {pos_shape}")

        # Build sub-layers in computational order

        # Position embeddings need expanded shapes for building
        pos_embed_input_shape = (1,) + pos_shape  # Add batch dimension
        self.pos_embed.build(pos_embed_input_shape)

        if self.rel_pos_embed is not None:
            # Relative position embedding input shape includes distance
            rel_pos_shape = pos_shape[:-1] + (self.ndim + 1,)  # Replace last dim
            rel_pos_embed_input_shape = pos_shape + (self.ndim + 1,)  # (num_points, num_supernodes, ndim+1)
            self.rel_pos_embed.build(rel_pos_embed_input_shape)

        # Message MLP input dimension
        if self.mode == "abspos":
            message_input_dim = self.hidden_dim * 2  # source + target pos embeddings
        else:
            message_input_dim = self.hidden_dim

        # Build message MLP with appropriate input shape
        # Message MLP processes (num_points, num_supernodes, message_input_dim)
        message_input_shape = pos_shape + (message_input_dim,)  # (num_points, num_supernodes, message_input_dim)
        self.message_mlp.build(message_input_shape)

        # Projection layer combines aggregated messages with position embeddings
        proj_input_dim = self.hidden_dim * 2  # aggregated + position embedding
        proj_input_shape = (pos_shape[0], proj_input_dim)  # (num_supernodes, hidden_dim * 2)
        self.proj_layer.build(proj_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply supernode pooling with message passing.

        Args:
            inputs: Dictionary containing positions and supernode_indices.
            training: Boolean indicating training mode.

        Returns:
            Pooled supernode features.
        """
        positions = inputs["positions"]  # (num_points, ndim)
        supernode_indices = inputs["supernode_indices"]  # (num_supernodes,)

        num_points = ops.shape(positions)[0]
        num_supernodes = ops.shape(supernode_indices)[0]

        # Get supernode positions
        supernode_positions = ops.take(positions, supernode_indices, axis=0)  # (num_supernodes, ndim)

        # Find neighbors for each supernode
        if self.radius is not None:
            neighbor_mask = self._radius_neighbors(positions, supernode_positions)
        else:
            neighbor_mask = self._knn_neighbors(positions, supernode_positions)

        # Create messages
        messages = self._create_messages(positions, supernode_positions, neighbor_mask, training=training)

        # Aggregate messages for each supernode
        aggregated = self._aggregate_messages(messages, neighbor_mask)

        # Get supernode position embeddings
        supernode_pos_embed = self.pos_embed(ops.expand_dims(supernode_positions, 0), training=training)  # (1, num_supernodes, hidden_dim)
        supernode_pos_embed = ops.squeeze(supernode_pos_embed, 0)  # (num_supernodes, hidden_dim)

        # Combine aggregated messages with position embeddings
        combined = ops.concatenate([aggregated, supernode_pos_embed], axis=-1)  # (num_supernodes, hidden_dim * 2)
        output = self.proj_layer(combined, training=training)  # (num_supernodes, hidden_dim)

        # Add batch dimension for consistency
        output = ops.expand_dims(output, 0)  # (1, num_supernodes, hidden_dim)

        return output

    def _radius_neighbors(self, positions: keras.KerasTensor, supernode_positions: keras.KerasTensor) -> keras.KerasTensor:
        """Find neighbors within radius for each supernode."""
        # Compute pairwise distances
        # positions: (num_points, ndim) -> (num_points, 1, ndim)
        # supernode_positions: (num_supernodes, ndim) -> (1, num_supernodes, ndim)
        pos_expanded = ops.expand_dims(positions, 1)
        super_expanded = ops.expand_dims(supernode_positions, 0)

        # Compute squared distances
        diff = pos_expanded - super_expanded  # (num_points, num_supernodes, ndim)
        sq_distances = ops.sum(ops.square(diff), axis=-1)  # (num_points, num_supernodes)

        # Create mask for points within radius
        mask = sq_distances <= (self.radius ** 2)  # (num_points, num_supernodes)

        return mask

    def _knn_neighbors(self, positions: keras.KerasTensor, supernode_positions: keras.KerasTensor) -> keras.KerasTensor:
        """Find k nearest neighbors for each supernode."""
        num_points = ops.shape(positions)[0]

        # Compute pairwise distances (same as radius method)
        pos_expanded = ops.expand_dims(positions, 1)
        super_expanded = ops.expand_dims(supernode_positions, 0)

        diff = pos_expanded - super_expanded
        sq_distances = ops.sum(ops.square(diff), axis=-1)  # (num_points, num_supernodes)

        # Find k nearest neighbors for each supernode
        k = ops.minimum(self.k_neighbors, num_points)

        # Get indices of k smallest distances for each supernode
        _, top_k_indices = ops.top_k(-sq_distances, k=k)  # (num_points, k) - using negative for smallest

        # Create mask from top-k indices
        mask = ops.zeros_like(sq_distances, dtype="bool")

        # This is a simplified approach - in practice, you'd use scatter operations
        # For now, we'll use top-k based selection
        threshold_distances = ops.take_along_axis(-ops.sort(-sq_distances, axis=0)[:k, :],
                                                  ops.array([k - 1]), axis=0)  # (1, num_supernodes)
        mask = sq_distances <= threshold_distances

        return mask

    def _create_messages(
            self,
            positions: keras.KerasTensor,
            supernode_positions: keras.KerasTensor,
            neighbor_mask: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Create messages from neighbors to supernodes."""
        num_points = ops.shape(positions)[0]
        num_supernodes = ops.shape(supernode_positions)[0]

        if self.mode == "abspos":
            # Absolute position mode: embed all positions
            pos_embed_all = self.pos_embed(ops.expand_dims(positions, 0), training=training)  # (1, num_points, hidden_dim)
            pos_embed_all = ops.squeeze(pos_embed_all, 0)  # (num_points, hidden_dim)

            super_embed_all = self.pos_embed(
                ops.expand_dims(supernode_positions, 0), training=training)  # (1, num_supernodes, hidden_dim)
            super_embed_all = ops.squeeze(super_embed_all, 0)  # (num_supernodes, hidden_dim)

            # Combine source and target embeddings
            pos_expanded = ops.expand_dims(pos_embed_all, 1)  # (num_points, 1, hidden_dim)
            super_expanded = ops.expand_dims(super_embed_all, 0)  # (1, num_supernodes, hidden_dim)

            combined = ops.concatenate([
                ops.tile(pos_expanded, [1, num_supernodes, 1]),
                ops.tile(super_expanded, [num_points, 1, 1])
            ], axis=-1)  # (num_points, num_supernodes, hidden_dim * 2)

        else:
            # Relative position mode: embed relative positions + distance
            pos_expanded = ops.expand_dims(positions, 1)  # (num_points, 1, ndim)
            super_expanded = ops.expand_dims(supernode_positions, 0)  # (1, num_supernodes, ndim)

            relative_pos = super_expanded - pos_expanded  # (num_points, num_supernodes, ndim)
            distances = ops.sqrt(
                ops.sum(ops.square(relative_pos), axis=-1, keepdims=True))  # (num_points, num_supernodes, 1)

            # Combine relative position with distance magnitude
            rel_pos_with_dist = ops.concatenate([relative_pos, distances],
                                                axis=-1)  # (num_points, num_supernodes, ndim + 1)

            # Embed relative positions
            combined = self.rel_pos_embed(rel_pos_with_dist, training=training)  # (num_points, num_supernodes, hidden_dim)

        # Apply message MLP
        messages = self.message_mlp(combined, training=training)  # (num_points, num_supernodes, hidden_dim)

        # Mask out non-neighbors
        mask_expanded = ops.expand_dims(ops.cast(neighbor_mask, messages.dtype), -1)
        messages = messages * mask_expanded

        return messages

    def _aggregate_messages(self, messages: keras.KerasTensor, neighbor_mask: keras.KerasTensor) -> keras.KerasTensor:
        """Aggregate messages for each supernode."""
        # Sum messages for each supernode
        aggregated = ops.sum(messages, axis=0)  # (num_supernodes, hidden_dim)

        # Normalize by number of neighbors to get mean
        neighbor_counts = ops.sum(ops.cast(neighbor_mask, "float32"), axis=0, keepdims=True)  # (1, num_supernodes)
        neighbor_counts = ops.maximum(neighbor_counts, 1.0)  # Avoid division by zero

        aggregated = aggregated / ops.transpose(neighbor_counts)  # (num_supernodes, hidden_dim)

        return aggregated

    def compute_output_shape(self, input_shape: Union[Dict[str, Tuple[Optional[int], ...]], Any]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        if not isinstance(input_shape, dict) or "supernode_indices" not in input_shape:
            raise ValueError("Cannot determine output shape without supernode_indices")

        # Output shape is (batch_size=1, num_supernodes, hidden_dim)
        # Note: num_supernodes is dynamic, so we use None
        return (1, None, self.hidden_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "ndim": self.ndim,
            "radius": self.radius,
            "k_neighbors": self.k_neighbors,
            "max_neighbors": self.max_neighbors,
            "mode": self.mode,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
