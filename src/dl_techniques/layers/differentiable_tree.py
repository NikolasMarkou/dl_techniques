import keras
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional
from keras import ops, initializers, activations, layers

@keras.saving.register_keras_serializable()
class DifferentiableTreeDense(keras.layers.Layer):
    """
    Differentiable Tree Dense Layer.

    This layer acts as a drop-in replacement for a standard `keras.layers.Dense`
    layer, transforming an input feature vector into an output vector. It replaces
    a single, large matrix multiplication with a hierarchical series of smaller,
    more efficient multiplications, which can be advantageous for very large
    output dimensions.

    **Intent**: To provide a scalable alternative to `keras.layers.Dense` by
    decomposing a large linear transformation into a soft decision tree structure.
    This can potentially reduce computational cost and parameter count for certain
    architectures.

    **Architecture**:
    The layer organizes the output space into a balanced binary tree.
    - **Internal Nodes**: Learnable routers that, given an input, calculate a
      "soft" probability of traversing left or right down the tree.
    - **Leaf Nodes**: Small, independent `Dense` layers that each transform the
      input features into a smaller chunk of the final output vector.
    - **Combination Strategy**: A configurable final step to combine the outputs
      of all the leaves into the final tensor.

    ```
    Input Features (h)
           ↓
    Internal Node n₁ (Router) → P(Path to Leaf₁|h), P(Path to Leaf₂|h), ...
          /      \
         /        \
    Leaf Node L₁   Leaf Node L₂
       ↓              ↓
    Output₁ = L₁(h)  Output₂ = L₂(h)
    ```

    **Mathematical Operations**:
    1.  **Routing Probabilities**: At each internal node `n`, given features `h`:
        `P(d|n, h) = softmax(h @ W_n)`, where `d` is the direction {Left, Right}.
    2.  **Path Probabilities**: For each leaf `L`, a path probability `P(L|h)` is
        calculated by multiplying the routing probabilities along its unique path.
        This is computed efficiently for all leaves simultaneously.
    3.  **Leaf Outputs**: Each leaf `L` computes its own dense transformation:
        `Output_L = h @ W_L + b_L`.
    4.  **Combination**: The weighted leaf outputs are combined using one of two
        strategies defined by `leaf_combination`:
        - **`'concat'` (default)**: The final output is a simple concatenation of
          the weighted leaf outputs.
          `Final = Concat([P(L₁|h) * Output_L₁, P(L₂|h) * Output_L₂, ...])`
        - **`'dense'`**: The concatenated outputs are passed through a final
          learnable `Dense` layer to produce the output. This allows the model
          to learn an optimal linear combination of all leaf features.
          `Final = Dense(Concat([...]))`

    Args:
        output_dim: Integer, the final dimensionality of the output space.
            Must be positive.
        leaf_output_dim: Integer, the output dimension of each individual leaf
            node's dense transformation. The number of leaves will be
            `ceil(output_dim / leaf_output_dim)`. Defaults to 128.
        leaf_combination: String, the strategy to combine leaf outputs.
            Must be one of `'concat'` or `'dense'`. `'concat'` is faster and has
            fewer parameters, while `'dense'` is more expressive.
            Defaults to `'concat'`.
        activation: Activation function to apply to the final output. Can be a
            string name (e.g., 'relu') or a Keras activation function.
            Defaults to `None` (linear activation).
        kernel_initializer: Initializer for all weights (routing, leaf, and
            combination). Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for all biases (leaf and combination).
            Defaults to 'zeros'.
        **kwargs: Additional arguments for the `Layer` base class (e.g., name,
            dtype, trainable).

    Input shape:
        N-D tensor with shape `(batch_size, ..., feature_dim)`. The layer operates
        on the last dimension.

    Output shape:
        - If `leaf_combination='concat'`: `(..., effective_output_dim)`, where
          `effective_output_dim` is `num_leaves * leaf_output_dim`. This may be
          larger than `output_dim` if padding is required.
        - If `leaf_combination='dense'`: `(..., output_dim)`. The output is
          projected to the exact requested dimension.

    Attributes:
        routing_weights: Trainable weights for the internal routing nodes.
            Shape: `(num_internal_nodes, feature_dim, 2)`. Created in `build()`.
        leaf_weights: Trainable weights for the leaf dense transformations.
            Shape: `(num_leaves, feature_dim, leaf_output_dim)`. Created in `build()`.
        leaf_biases: Trainable biases for the leaf dense transformations.
            Shape: `(num_leaves, leaf_output_dim)`. Created in `build()`.
        combination_layer: The final `Dense` layer used if `leaf_combination='dense'`.
            A standard `keras.layers.Dense` instance.

    Example:
        ```python
        feature_dim = 256
        output_dim = 4096

        inputs = keras.Input(shape=(feature_dim,))
        # Use 'dense' combination to learn a final projection
        tree_output = DifferentiableTreeDense(
            output_dim=output_dim,
            leaf_combination='dense',
            activation='relu'
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=tree_output)

        # The output shape will be exactly (None, 4096)
        model.summary()
        ```

    Raises:
        ValueError: If `output_dim` or `leaf_output_dim` are not positive, or
            if `leaf_combination` is not a valid option.
    """

    def __init__(
            self,
            output_dim: int,
            leaf_output_dim: int = 128,
            leaf_combination: str = 'concat',
            activation: Union[str, None, keras.layers.Activation] = None,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ):
        super().__init__(**kwargs)

        # --- 1. Configuration and Validation ---
        # Store all constructor arguments as instance attributes. This is crucial
        # for `get_config()` to work correctly for serialization.
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if leaf_output_dim <= 0:
            raise ValueError(f"leaf_output_dim must be positive, got {leaf_output_dim}")
        if leaf_combination not in ['concat', 'dense']:
            raise ValueError(
                "leaf_combination must be one of ['concat', 'dense'], "
                f"got '{leaf_combination}'"
            )

        self.output_dim = output_dim
        self.leaf_output_dim = leaf_output_dim
        self.leaf_combination = leaf_combination
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # --- 2. Tree Structure Calculation ---
        # These are derived properties, not direct configuration. They don't need
        # to be in `get_config()` as they can be re-computed from the config.
        self.num_leaves = int(np.ceil(self.output_dim / self.leaf_output_dim))
        if self.num_leaves <= 1:
            self.num_internal_nodes = 0
            self.tree_depth = 0
        else:
            self.num_internal_nodes = self.num_leaves - 1
            self.tree_depth = int(np.ceil(np.log2(self.num_leaves)))

        # The actual output dimension of the concatenated leaves, which might be padded.
        self.effective_output_dim = self.num_leaves * self.leaf_output_dim

        # --- 3. CREATE Sub-layers (if any) ---
        # Following the Golden Rule: CREATE sub-layers in `__init__`.
        # They remain unbuilt until the parent layer's `build` method is called.
        if self.leaf_combination == 'dense':
            self.combination_layer = layers.Dense(
                units=self.output_dim,
                activation=None,  # Final activation is applied by this layer
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                name="combination_dense"
            )
        else:
            self.combination_layer = None

        # --- 4. Pre-computation and Non-trainable State ---
        # Pre-compute paths from root to each leaf. This is constant and can be
        # done at initialization. It's stored as a non-trainable Keras Variable.
        if self.num_internal_nodes > 0:
            np_path_nodes = np.full((self.num_leaves, self.tree_depth), -1, dtype=np.int32)
            np_path_directions = np.full((self.num_leaves, self.tree_depth), -1, dtype=np.int32)

            for leaf_idx in range(self.num_leaves):
                tree_node_idx = self.num_internal_nodes + leaf_idx
                path, dirs = [], []
                curr = tree_node_idx
                while curr > 0:
                    parent = (curr - 1) // 2
                    direction = (curr - 1) % 2
                    path.append(parent)
                    dirs.append(direction)
                    curr = parent
                path_len = len(path)
                np_path_nodes[leaf_idx, :path_len] = path[::-1]
                np_path_directions[leaf_idx, :path_len] = dirs[::-1]

            self.path_nodes_map = keras.Variable(
                np_path_nodes, trainable=False, name="path_nodes_map", dtype="int32"
            )
            self.path_directions_map = keras.Variable(
                np_path_directions, trainable=False, name="path_directions_map", dtype="int32"
            )

        # --- 5. Initialize Weight Attributes ---
        # Declare weight attributes, but do not create them. They will be
        # created in `build()` once the input shape is known.
        self.routing_weights = None
        self.leaf_weights = None
        self.leaf_biases = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Creates the layer's learnable weights and builds its sub-layers.

        This method is called automatically by Keras the first time the layer is
        used. It adheres to the "Golden Rule" by creating weights and building
        sub-layers here, not in `__init__`.
        """
        input_shape = tuple(input_shape)
        feature_dim = input_shape[-1]
        if feature_dim is None:
            raise ValueError(
                "The last dimension of the input shape cannot be None. "
                "Please provide a defined input shape."
            )

        # 1. CREATE the layer's own weights using `add_weight`.
        if self.num_internal_nodes > 0:
            self.routing_weights = self.add_weight(
                name="routing_weights",
                shape=(self.num_internal_nodes, feature_dim, 2),
                initializer=self.kernel_initializer,
                trainable=True,
            )
        self.leaf_weights = self.add_weight(
            name="leaf_weights",
            shape=(self.num_leaves, feature_dim, self.leaf_output_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.leaf_biases = self.add_weight(
            name="leaf_biases",
            shape=(self.num_leaves, self.leaf_output_dim),
            initializer=self.bias_initializer,
            trainable=True,
        )

        # 2. CRITICAL: BUILD the sub-layers. This ensures their weights are
        # created, which is essential for correct weight restoration on model load.
        if self.combination_layer is not None:
            # The input shape to the combination layer is the concatenated output.
            combination_input_shape = input_shape[:-1] + (self.effective_output_dim,)
            self.combination_layer.build(combination_input_shape)

        # 3. Always call the parent's `build` method at the end.
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Performs the forward pass of the layer."""
        features = inputs
        # Get the shape of the batch dimensions (e.g., (batch_size,) or (batch_size, seq_len))
        batch_shape = ops.shape(features)[:-1]

        # --- Step 1: Calculate Path Probabilities for All Leaves ---
        if self.num_internal_nodes > 0:
            routing_logits = ops.einsum("...f,nfd->...nd", features, self.routing_weights)
            routing_probs = ops.softmax(routing_logits, axis=-1)

            path_step_probs = ops.take(routing_probs, self.path_nodes_map, axis=-2)
            path_dir_one_hot = ops.one_hot(self.path_directions_map, 2, dtype=self.compute_dtype)
            selected_probs = ops.sum(path_step_probs * path_dir_one_hot, axis=-1)

            mask = ops.cast(ops.not_equal(self.path_nodes_map, -1), self.compute_dtype)
            masked_selected_probs = ops.where(mask, selected_probs, 1.0)
            leaf_path_probs = ops.prod(masked_selected_probs, axis=-1)
        else:
            # If there is only one leaf, its path probability is always 1.
            ones_shape = batch_shape + (self.num_leaves,)
            leaf_path_probs = ops.ones(ones_shape, dtype=self.compute_dtype)

        # --- Step 2: Calculate Leaf Outputs ---
        leaf_outputs = ops.einsum("...f,lfd->...ld", features, self.leaf_weights)
        leaf_outputs += self.leaf_biases

        # --- Step 3: Weight Leaf Outputs by Path Probabilities ---
        weighted_leaf_outputs = leaf_outputs * ops.expand_dims(leaf_path_probs, axis=-1)

        # --- Step 4: Combine into Final Output Vector ---
        reshape_shape = batch_shape + (self.effective_output_dim,)
        concatenated_outputs = ops.reshape(weighted_leaf_outputs, reshape_shape)

        if self.leaf_combination == 'dense':
            combined_output = self.combination_layer(concatenated_outputs)
        else:  # 'concat'
            combined_output = concatenated_outputs

        # --- Step 5: Apply Final Activation ---
        # self.activation is a callable function (e.g., relu or linear).
        final_output = self.activation(combined_output)

        return final_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Computes the output shape of the layer."""
        if self.leaf_combination == 'dense':
            # Returns the exact user-specified output dimension.
            return input_shape[:-1] + (self.output_dim,)
        else:  # 'concat'
            # Returns the padded dimension from concatenated leaves.
            return input_shape[:-1] + (self.effective_output_dim,)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        # Start with the parent class's configuration.
        config = super().get_config()
        # Add all constructor arguments to the config.
        config.update({
            'output_dim': self.output_dim,
            'leaf_output_dim': self.leaf_output_dim,
            'leaf_combination': self.leaf_combination,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config