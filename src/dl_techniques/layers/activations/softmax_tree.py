import keras
import numpy as np
from keras import ops, initializers
from typing import Optional, Union, Tuple, List, Dict, Any


@keras.saving.register_keras_serializable()
class DifferentiableSoftmaxTree(keras.layers.Layer):
    """
    Differentiable Softmax Tree (Learnable Hierarchical Classifier) layer.

    This layer implements a hierarchical classification mechanism where routing
    decisions within the tree are parameterized and learned. It serves as a
    highly efficient alternative to the standard softmax function for tasks
    with a very large number of output classes (e.g., >10,000).

    **Intent**: Provide a scalable, production-ready output layer for large-scale
    classification that replaces the standard `Dense + Softmax` head. It is
    designed to be a drop-in component that handles both prediction and loss
    computation internally for maximum efficiency.

    **Architecture**:
    Classes are arranged as leaves in a balanced binary tree. Each internal node
    acts as a learnable binary router.
    ```
    Input Features (h)
           ↓
    Internal Node n₁ (Router) → P(Left|n₁,h), P(Right|n₁,h)
          /      \
         /        \
    Node n₂       Node n₃ → P(Left|n₃,h), P(Right|n₃,h)
      ...         /     \
                 ...   Class c (Leaf Node)
    ```

    **Mathematical Operations**:
    1.  **Routing Probability**: At node $n$, given features $h$:
        $P(d|n, h) = \text{softmax}(h \cdot W_n)$, where $d \in \{L, R\}$.
    2.  **Class Probability**: The probability of a class $c$ is the product of
        routing probabilities along the unique path from the root to leaf $c$.
        $P(c|h) = \prod_{(n, d) \in \text{Path}(c)} P(d|n, h)$
    3.  **Training Loss (NLL)**: The layer computes the Negative Log-Likelihood
        loss in the numerically stable log domain:
        $L(c|h) = - \sum_{(n, d) \in \text{Path}(c)} \log(P(d|n, h))$

    Args:
        num_classes: Integer, the total number of output classes. Must be > 1.
        feature_dim: Integer, the dimension of the input feature vector.
            Must be positive.
        initializer: Initializer for the internal routing node weights.
            Accepts string names or Initializer instances.
            Defaults to 'glorot_uniform'.
        **kwargs: Additional arguments for Layer base class (name, dtype, etc.).

    Input shape:
        A list or tuple containing two tensors:
        1. `features`: A tensor of shape `(batch_size, ..., feature_dim)`.
        2. `targets`: An integer tensor of shape `(batch_size, ..., 1)`
           containing the ground-truth class indices.

    Output shape:
        A tensor of shape `(batch_size, ...)` containing the NLL loss per sample.

    Attributes:
        num_internal_nodes: The number of learnable routing nodes in the tree.
        tree_depth: The maximum depth of the binary tree.
        node_weights: The layer's trainable weights, with shape
            `(num_internal_nodes, feature_dim, 2)`.

    Example:
        ```python
        # 1. Setup for a large-scale classification task
        num_classes = 50000
        feature_dim = 512

        # 2. Define inputs for features and ground-truth targets
        input_features = keras.Input(shape=(feature_dim,), name="features")
        target_indices = keras.Input(shape=(1,), dtype="int32", name="target_class")

        # 3. Instantiate the layer as the model's head
        soft_tree = DifferentiableSoftmaxTree(num_classes, feature_dim)

        # 4. The layer's output *is* the loss
        nll_loss = soft_tree([input_features, target_indices])

        # 5. Create a model where the output is the computed loss
        model = keras.Model(inputs=[input_features, target_indices], outputs=nll_loss)

        # 6. Compile with a dummy loss, as the layer handles the real loss
        model.compile(optimizer="adam", loss=lambda y_true, y_pred: y_pred)
        ```

    Raises:
        ValueError: If `num_classes` or `feature_dim` are invalid.
        ValueError: If `input_shape` during build is not a list of two shapes
            or if the feature dimension mismatches.
    """

    def __init__(
            self,
            num_classes: int,
            feature_dim: int,
            initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # --- Configuration and Validation ---
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.initializer = initializers.get(initializer)

        # --- CREATE Tree Structure (Balanced Binary Tree) ---
        # A balanced binary tree with N leaves has N-1 internal (routing) nodes.
        self.num_internal_nodes = self.num_classes - 1
        self.tree_depth = int(np.ceil(np.log2(self.num_classes)))

        # Pre-compute paths for all classes to enable efficient batch gathering.
        # path_nodes: indices of internal nodes on the path to a class.
        # path_directions: 0 (left) or 1 (right) decision at each node.
        np_path_nodes = np.full((self.num_classes, self.tree_depth), -1, dtype=np.int32)
        np_path_directions = np.full((self.num_classes, self.tree_depth), -1, dtype=np.int32)

        # Build paths iteratively from each leaf up to the root.
        for class_idx in range(self.num_classes):
            # Map class index to a leaf node index in the full tree structure.
            tree_node_idx = self.num_internal_nodes + class_idx
            current_path, current_dirs = [], []

            curr = tree_node_idx
            while curr > 0:  # Stop when we reach the root (index 0)
                parent = (curr - 1) // 2
                # An odd index is a left child (direction 0), even is right (direction 1).
                direction = (curr - 1) % 2
                current_path.append(parent)
                current_dirs.append(direction)
                curr = parent

            # Reverse to get the correct root-to-leaf order and store.
            path_len = len(current_path)
            np_path_nodes[class_idx, :path_len] = current_path[::-1]
            np_path_directions[class_idx, :path_len] = current_dirs[::-1]

        # Convert maps to non-trainable Keras variables for graph integration.
        self.path_nodes_map = keras.Variable(
            np_path_nodes, trainable=False, name="path_nodes_map", dtype="int32"
        )
        self.path_directions_map = keras.Variable(
            np_path_directions, trainable=False, name="path_directions_map", dtype="int32"
        )

        # Initialize weight attribute, which will be created in build().
        self.node_weights = None

    def build(self, input_shape: Union[Tuple, List]) -> None:
        """
        Creates the layer's learnable routing weights. This method follows the
        "Golden Rule" by creating weights only after the input shape is known.
        """
        # Robustly validate the expected input structure.
        if (not isinstance(input_shape, (list, tuple)) or
                len(input_shape) != 2 or
                not all(isinstance(shape, (list, tuple)) for shape in input_shape)):
            raise ValueError(
                "Layer expects a list/tuple of two input shapes (features, targets). "
                f"Got: {input_shape}"
            )

        features_shape, _ = input_shape
        actual_feature_dim = features_shape[-1]

        if actual_feature_dim != self.feature_dim:
            raise ValueError(
                f"Input feature dimension ({actual_feature_dim}) does not match "
                f"configured feature_dim ({self.feature_dim})."
            )

        # CREATE the layer's own weights.
        # Shape: (Num Internal Nodes, Feature Dim, 2 branches (Left/Right))
        self.node_weights = self.add_weight(
            name="node_routing_weights",
            shape=(self.num_internal_nodes, self.feature_dim, 2),
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """
        Performs the forward pass, computing the NLL loss for the given targets.
        """
        features, targets = inputs

        # Ensure targets are integers for gathering operations.
        targets = ops.cast(targets, "int32")
        targets_flat = ops.squeeze(targets, axis=-1)

        # 1. Gather Paths: Retrieve the pre-computed paths for the target classes.
        batch_path_nodes = ops.take(self.path_nodes_map, targets_flat, axis=0)
        batch_path_dirs = ops.take(self.path_directions_map, targets_flat, axis=0)

        # 2. Prepare Masking: Create a mask to ignore padded steps in paths.
        mask = ops.cast(ops.not_equal(batch_path_nodes, -1), features.dtype)
        safe_path_nodes = ops.where(batch_path_nodes == -1, 0, batch_path_nodes)

        # 3. Gather Weights: Select the routing weights for nodes in the batch paths.
        path_weights = ops.take(self.node_weights, safe_path_nodes, axis=0)

        # 4. Compute Logits: Perform a batched dot product.
        # `features` shape: (batch, feature_dim)
        # We expand it to (batch, 1, feature_dim, 1) to correctly broadcast with
        # `path_weights` of shape (batch, tree_depth, feature_dim, 2).
        features_expanded = ops.expand_dims(ops.expand_dims(features, axis=1), axis=3)

        # The result is the [score_left, score_right] logits at each node.
        # Shape: (batch, tree_depth, 2)
        node_logits = ops.sum(features_expanded * path_weights, axis=2)

        # 5. Compute Log-Probabilities: Apply stable log_softmax.
        node_log_probs = ops.log_softmax(node_logits, axis=-1)

        # 6. Select Correct Path Probabilities: Use one-hot masking to pick the
        #    log-probability corresponding to the correct direction at each step.
        safe_path_dirs = ops.where(batch_path_dirs == -1, 0, batch_path_dirs)
        decisions_one_hot = ops.one_hot(safe_path_dirs, 2, dtype=node_log_probs.dtype)
        selected_log_probs = ops.sum(node_log_probs * decisions_one_hot, axis=-1)

        # 7. Calculate Total Log-Probability: Sum log-probs along the path.
        # This is equivalent to multiplying probabilities in normal space.
        masked_log_probs = selected_log_probs * mask
        total_log_prob = ops.sum(masked_log_probs, axis=-1)

        # 8. Return NLL Loss: The final loss is the negative of the total log-prob.
        return -total_log_prob

    def compute_output_shape(self, input_shape: List[Tuple]) -> Tuple:
        """
        Computes the output shape of the layer. The output is a scalar loss
        for each sample, so we drop the feature dimension.
        """
        features_shape, _ = input_shape
        return features_shape[:-1]

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'initializer': initializers.serialize(self.initializer),
        })
        return config