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

    Selects supernode positions from input points, finds their neighbours
    (within a radius or via k-NN), creates positional messages using
    continuous sin/cos embeddings, and aggregates them through a two-layer
    MLP. The aggregated messages are concatenated with supernode position
    embeddings and projected to the output dimension.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────┐
        │  Inputs (dict)                    │
        │  positions     [P, ndim]          │
        │  supernode_idx [S]                │
        └───────────────┬───────────────────┘
                        ▼
        ┌───────────────────────────────────┐
        │  Neighbour Selection              │
        │  (radius or k-NN) → mask [P, S]   │
        └───────────────┬───────────────────┘
                        ▼
        ┌───────────────────────────────────┐
        │  Message Creation                 │
        │  Pos Embed (abs or rel)           │
        │  → Message MLP [P, S, H]          │
        │  × mask                           │
        └───────────────┬───────────────────┘
                        ▼
        ┌───────────────────────────────────┐
        │  Mean Aggregation → [S, H]        │
        └───────────────┬───────────────────┘
                        ▼
        ┌───────────────────────────────────┐
        │  Concat(agg, pos_embed) → [S, 2H] │
        │  Dense → [S, H]                   │
        │  Expand → [1, S, H]               │
        └───────────────────────────────────┘

    :param hidden_dim: Output dimension of pooled features. Must be positive.
    :type hidden_dim: int
    :param ndim: Number of coordinate dimensions (2 or 3). Must be positive.
    :type ndim: int
    :param radius: Radius for neighbour selection. Mutually exclusive with
        ``k_neighbors``.
    :type radius: Optional[float]
    :param k_neighbors: Number of nearest neighbours. Mutually exclusive with
        ``radius``.
    :type k_neighbors: Optional[int]
    :param max_neighbors: Maximum neighbours per supernode. Defaults to 32.
    :type max_neighbors: int
    :param mode: Positional encoding mode (``"abspos"`` or ``"relpos"``).
        Defaults to ``"relpos"``.
    :type mode: str
    :param activation: Activation for message MLP. Defaults to ``"gelu"``.
    :type activation: Union[str, callable]
    :param use_bias: Whether to use bias. Defaults to ``True``.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to ``"glorot_uniform"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors.
        Defaults to ``"zeros"``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the ``Layer`` base class.
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

        :param inputs: Dictionary with ``positions`` and ``supernode_indices``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Pooled supernode features ``(1, num_supernodes, hidden_dim)``.
        :rtype: keras.KerasTensor
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
