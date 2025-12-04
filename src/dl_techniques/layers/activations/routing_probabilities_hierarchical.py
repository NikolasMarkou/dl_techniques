"""
Trainable hierarchical routing tree for large-scale classification.

This module provides a learnable alternative to the standard Dense -> Softmax
architecture. Unlike RoutingProbabilitiesLayer which uses fixed deterministic
projections, this layer learns the optimal routing decisions via backpropagation.

Complete Architecture Flow::

    INPUT: Features [batch, D]
      │
      │  Learnable Projection Phase
      ├──────────────────────────────────────────────────┐
      │  Weights W: [D, log₂(padded_N)]                  │
      │  Bias b:    [log₂(padded_N)]                     │
      │                                                  │
      │  logits = <input, W> + b                         │
      │  p_k = σ(logits)           (sigmoid)             │
      └──────────────────────────────────────────────────┘
      │
      ↓  [p₀, p₁, ..., p_{d-1}]  where d = log₂(padded_N)
      │
      │  Hierarchical Routing Phase
      ├──────────────────────────────────────────────────┐
      │  Initialize: root_prob = 1.0                     │
      │                                                  │
      │  For level k = 0 to d-1:                         │
      │    For each node at level k:                     │
      │      left_child_prob  = node_prob × (1 - p_k)    │
      │      right_child_prob = node_prob × p_k          │
      │                                                  │
      │  Result: 2^d leaf probabilities                  │
      └──────────────────────────────────────────────────┘
      │
      ↓  [leaf₀, leaf₁, ..., leaf_{2^d-1}]  (sum = 1.0)
      │
      │  Slicing & Renormalization Phase
      ├──────────────────────────────────────────────────┐
      │  If N != 2^d (not power of 2):                   │
      │    selected = [leaf₀, ..., leaf_{N-1}]           │
      │    normalized = selected / sum(selected)         │
      │  Else:                                           │
      │    normalized = leaves (already sum to 1.0)      │
      └──────────────────────────────────────────────────┘
      │
      ↓
    OUTPUT: Class Probabilities [batch, N]  (sum = 1.0)

Binary Tree Structure::

    The concept is identical to the deterministic routing layer, but the
    decisions are learned.

    1. **Padding**: The output dimension N is padded to the next power of
       two (`padded_dim`). Depth d = log₂(padded_dim).

    2. **Learnable Projections**: A trainable Dense projection maps the
       input features to d logits.

    3. **Probabilistic Decisions**: Sigmoid activation converts logits to
       branching probabilities.

    4. **Hierarchical Routing**: Probability mass flows from the root to
       leaves based on these decisions.

Foundational Mathematics
------------------------
Unlike the deterministic version which uses Cosine Similarity, this layer
uses a standard Affine Transformation for decision making.

    z = xW + b

    Where:
    - x is the input vector [1, D]
    - W is the learnable weight matrix [D, d]
    - b is the learnable bias vector [d]

    The probability of taking the 'right' branch at depth k is:
    P(right_k) = σ(z_k)

References
----------
- Morin, F., & Bengio, Y. (2005). "Hierarchical Probabilistic Neural
  Network Language Model". AISTATS.
"""

import math
import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HierarchicalRoutingLayer(keras.layers.Layer):
    """
    Trainable hierarchical routing layer for probabilistic classification.

    This layer acts as a drop-in replacement for a standard Dense+Softmax
    output head. It reduces the computational complexity of the output
    projection from O(N) to O(log N) by learning a sequence of binary
    decisions.

    **Architecture Overview**::

        Input Features [batch, ..., D]
               ↓
        ┌──────────────────────────────────────┐
        │  Trainable Dense Projection          │  (Learnable Weights)
        │  d = log₂(padded_dim) outputs        │
        └──────────────────────────────────────┘
               ↓
        Decision Logits [batch, d] → Sigmoid → Decision Probs [batch, d]
               ↓
        ┌──────────────────────────────────────┐
        │    Hierarchical Probability Tree     │
        │  (binary splits at each level k)     │
        └──────────────────────────────────────┘
               ↓
        Leaf Probabilities [batch, padded_dim] → Slice & Renormalize
               ↓
        Output Probabilities [batch, ..., output_dim]  (sums to 1.0)

    **Processing Pipeline**:

    1. **Output Dimension Logic**: Calculates `padded_dim` (next power of 2)
       to ensure a complete binary tree.

    2. **Learnable Decision Making**: Projects inputs using a trainable kernel
       `W` of shape `(input_dim, log2(padded_dim))` and bias `b`.

    3. **Probabilistic Routing**: Traverses the binary tree, splitting
       probability mass at each level.

    4. **Slicing & Renormalization**: Selects probabilities for original
       classes and renormalizes.

    :param output_dim: Dimensionality of the output space (number of classes).
    :type output_dim: int
    :param axis: Axis along which the routing is applied. Defaults to -1.
    :type axis: int
    :param epsilon: Small float added to prevent numerical issues.
    :type epsilon: float
    :param kernel_initializer: Initializer for the weight matrix.
    :param bias_initializer: Initializer for the bias vector.
    :param kernel_regularizer: Regularizer function for the weight matrix.
    :param use_bias: Whether to use a bias vector. Defaults to True.
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            output_dim: int,
            axis: int = -1,
            epsilon: float = 1e-7,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            use_bias: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Initialize the HierarchicalRoutingLayer.
        """
        super().__init__(**kwargs)

        if not isinstance(output_dim, int) or output_dim <= 1:
            raise ValueError(
                f"The 'output_dim' must be an integer greater than 1, "
                f"but received: {output_dim}"
            )

        if not isinstance(axis, int):
            raise ValueError(
                f"The 'axis' must be an integer, but received: {axis}"
            )

        self.output_dim = output_dim
        self.axis = axis
        self.epsilon = epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        # Calculated in build
        self.padded_output_dim = None
        self.num_decisions = None
        self._normalized_axis = None

        # Trainable weights
        self.kernel = None
        self.bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer by creating trainable weights and tree structure.

        :param input_shape: Shape tuple of the input tensor.
        """
        # Normalize axis to handle negative indices
        input_rank = len(input_shape)
        if self.axis < 0:
            self._normalized_axis = input_rank + self.axis
        else:
            self._normalized_axis = self.axis

        if self._normalized_axis < 0 or self._normalized_axis >= input_rank:
            raise ValueError(
                f"axis {self.axis} is out of bounds for input shape "
                f"{input_shape}"
            )

        # Calculate padded dimensions for tree structure
        # Next power of 2: 2^ceil(log2(output_dim))
        self.padded_output_dim = 1 << (self.output_dim - 1).bit_length()
        self.num_decisions = int(math.log2(self.padded_output_dim))

        input_dim = input_shape[self._normalized_axis]
        if input_dim is None:
            raise ValueError(
                f"The dimension at axis {self.axis} of input_shape must be "
                f"defined to create trainable weights, but got None."
            )

        logger.info(
            f"[{self.name}] Built for {self.output_dim} classes along axis "
            f"{self.axis}. Padded to {self.padded_output_dim} for tree "
            f"construction, requiring {self.num_decisions} learnable decisions."
        )

        # Create Trainable Weights
        # Shape: (input_dim, num_decisions)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.num_decisions),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.compute_dtype
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.num_decisions,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.compute_dtype
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Define the forward pass logic of the layer.

        **Processing Steps**::

            0. Axis manipulation & reshape to 2D
               ↓
            1. Compute learnable decision probabilities (MatMul)
               ↓
            2. Initialize root probability (1.0 for all samples)
               ↓
            3. Iteratively split probabilities through binary tree
               ↓
            4. Slice to output_dim and renormalize
               ↓
            5. Reshape back to original structure
               ↓
            6. Transpose to restore original axis order

        :param inputs: Input tensor of arbitrary rank.
        :param training: Whether the layer is in training mode.
        """
        # Step 0: Handle axis manipulation for arbitrary rank tensors
        # Move the target axis to the last position and flatten others
        # to ensure graph safety and correct dense projection.
        input_rank = len(inputs.shape)

        # Create permutation to move target axis to last position
        perm = list(range(input_rank))
        perm[self._normalized_axis] = input_rank - 1
        perm[input_rank - 1] = self._normalized_axis

        # Transpose if axis is not already last
        if self._normalized_axis != input_rank - 1:
            inputs_transposed = ops.transpose(inputs, perm)
        else:
            inputs_transposed = inputs

        # Reshape to 2D: (batch_size, input_dim)
        input_dim = inputs_transposed.shape[-1]
        inputs_2d = ops.reshape(inputs_transposed, (-1, input_dim))

        # Step 1: Compute learnable routing decisions
        # Linear projection: inputs_2d @ kernel + bias
        # Shape: (batch_size, num_decisions)
        decision_logits = ops.matmul(inputs_2d, self.kernel)

        if self.use_bias:
            decision_logits = decision_logits + self.bias

        # Apply sigmoid to convert logits to probabilities (0 to 1)
        decision_probs = ops.sigmoid(decision_logits)

        # Clip decision probabilities to prevent exactly 0 or 1
        # Crucial for numerical stability in downstream loss
        decision_probs = ops.clip(
            decision_probs, self.epsilon, 1.0 - self.epsilon
        )

        # Step 2: Initialize root probability using ones_like for XLA safety
        # Start with probability mass of 1.0 for each batch item
        # Shape: (batch_size, 1)
        # We derive the shape from inputs_2d to handle dynamic batch sizes.
        ones_template = inputs_2d[:, 0:1]
        padded_probs = ops.ones_like(ones_template)

        # Step 3: Iteratively build the tree by splitting probabilities
        # At each level, split each existing leaf into two children
        #
        # Visual representation:
        #               Root              Level 0          Level 1
        #              [1.0]      →      [L  R]    →    [LL LR RL RR]
        #
        for i in range(self.num_decisions):
            # Get routing probability for current decision level
            # Shape: (batch_size, 1)
            p_go_right = decision_probs[:, i:i + 1]
            p_go_left = 1.0 - p_go_right

            # Split probability mass for each leaf
            probs_for_left_branches = padded_probs * p_go_left
            probs_for_right_branches = padded_probs * p_go_right

            # Interleave left and right branches
            # Stack creates shape: (batch_size, 2**i, 2)
            combined = ops.stack(
                [probs_for_left_branches, probs_for_right_branches],
                axis=2
            )

            # Reshape to flatten, creating interleaved structure
            # Final shape: (batch_size, 2**(i+1))
            padded_probs = ops.reshape(combined, (-1, 2 ** (i + 1)))

        # Step 4: Handle non-power-of-2 output dimensions
        # If output_dim != padded_dim, slice and renormalize
        if self.output_dim == self.padded_output_dim:
            final_probs = padded_probs
        else:
            # Slice to get only the true class probabilities
            unnormalized_probs = padded_probs[:, :self.output_dim]

            # Compute sum of sliced probabilities (will be < 1.0)
            prob_sum = ops.sum(unnormalized_probs, axis=-1, keepdims=True)

            # Renormalize to ensure output sums to 1.0
            final_probs = unnormalized_probs / (prob_sum + self.epsilon)

        # Step 5: Reshape back to original shape (with axis still at last)
        # We construct the target shape symbolically to be graph-safe.

        # Get symbolic shape of transposed input
        input_transposed_shape = ops.shape(inputs_transposed)
        input_transposed_shape_tensor = ops.convert_to_tensor(
            input_transposed_shape, dtype="int32"
        )

        # Slice to get all dimensions except the last one (B, H, W...)
        batch_shape_tensor = input_transposed_shape_tensor[:-1]

        # Create a tensor for the new output dimension
        target_dim_tensor = ops.convert_to_tensor(
            [self.output_dim], dtype="int32"
        )

        # Concatenate to form the full target shape: (B, H, W, output_dim)
        target_shape_tensor = ops.concatenate(
            [batch_shape_tensor, target_dim_tensor],
            axis=0
        )

        # Reshape the 2D final_probs back to the rank of the input
        outputs_transposed = ops.reshape(final_probs, target_shape_tensor)

        # Step 6: Transpose back to original axis order if needed
        if self._normalized_axis != input_rank - 1:
            outputs = ops.transpose(outputs_transposed, perm)
        else:
            outputs = outputs_transposed

        return outputs

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.
        """
        output_shape = list(input_shape)

        # Determine which axis to modify
        if self._normalized_axis is not None:
            axis_to_modify = self._normalized_axis
        else:
            input_rank = len(input_shape)
            axis_to_modify = (input_rank + self.axis if self.axis < 0
                              else self.axis)

        output_shape[axis_to_modify] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.
        """
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
