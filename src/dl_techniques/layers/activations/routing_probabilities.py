"""
Deterministic, parameter-free routing tree for classification.

This module provides a non-trainable alternative to the standard softmax
activation function for multi-class classification. Instead of learning a
dense transformation, it computes a probability distribution by routing an
initial probability mass through a fixed binary decision tree. This approach
is computationally efficient and introduces a structured, hierarchical bias
without adding any trainable parameters to the model.

The routing process works as follows:

1. **Padding**: The number of classes ``output_dim`` is padded to the
   next highest power of two, ``padded_dim``, to ensure a complete
   binary tree structure can be formed. The number of routing
   decisions (tree depth) is ``d = log2(padded_dim)``.

2. **Deterministic Projections**: For each of the ``d`` decisions, a
   fixed, non-trainable weight vector is pre-computed. The input
   feature vector is projected onto each of these ``d`` vectors to
   produce ``d`` scalar logits.

3. **Probabilistic Decisions**: Each logit is passed through a sigmoid
   activation function to yield ``d`` probabilities. Each probability
   ``p_k`` represents the likelihood of taking the "right" branch at
   level ``k`` of the tree.

4. **Hierarchical Routing**: The layer simulates the flow of
   probability mass, starting with 1.0 at the root. At each level ``k``,
   the probability mass at every node is split between its left and
   right children according to ``1 - p_k`` and ``p_k``, respectively.

5. **Renormalization**: After ``d`` splits, the accumulated mass at each
   of the ``padded_dim`` leaves forms a valid probability distribution.
   This distribution is then truncated to the original ``output_dim`` and
   renormalized to sum to 1.0.

The mechanism relies on two key mathematical ideas: deterministic feature
extraction using basis functions and hierarchical probability decomposition.

The weight vectors used for the projections are generated from a cosine basis:
``w_{k,i} = cos(2*pi * (k+1) * i / D)``

where ``D`` is the input feature dimension. The decision logit is the dot product:
``z_k = <x, w_k> = Sigma_i x_i * w_{k,i}``

The probability of reaching a specific leaf (class) is the product of the
probabilities of the choices along its unique path from the root:
``P(leaf) = product_{k=0}^{d-1} P(b_k)``

where branch probabilities are determined by:
``P(right_k) = sigma(z_k) = 1 / (1 + e^{-z_k})``
``P(left_k)  = 1 - sigma(z_k)``

References:
    - Zhang, Z., et al. (2024). "Softmax-free Large-scale Language Modeling".
      arXiv preprint arXiv:2402.01258.
    - Morin, F., & Bengio, Y. (2005). "Hierarchical Probabilistic Neural
      Network Language Model". AISTATS.
"""

import math
import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RoutingProbabilitiesLayer(keras.layers.Layer):
    """
    Non-trainable hierarchical routing layer for probabilistic classification.

    This layer provides a deterministic, parameter-free alternative to softmax
    by building a probabilistic routing tree. It computes routing decisions
    directly from input features using deterministic patterns (Fourier-like
    cosine basis functions), then routes probability mass through a binary
    tree to produce class probabilities.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────┐
        │    Input Features [batch, ..., D]       │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Deterministic Weight Projections       │
        │  w_{k,i} = cos(2pi*(k+1)*i / D)        │
        │  d = log2(padded_dim) projections       │
        │  z_k = <x, w_k>  (dot product)         │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Sigmoid Activation                     │
        │  p_k = sigma(z_k) -> Decision Probs    │
        │  [batch, d]                             │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Hierarchical Probability Tree          │
        │                                         │
        │            Root (p=1.0)                  │
        │           ┌───┴───┐                     │
        │       (1-p0)    (p0)                    │
        │       ┌──┴──┐  ┌──┴──┐                  │
        │      ...   ... ...   ...                │
        │       │     │   │     │                  │
        │      L0    L1  L2    L3  ...            │
        │                                         │
        │  Binary splits at each level k          │
        │  left = parent * (1 - p_k)              │
        │  right = parent * p_k                   │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Slice & Renormalize                    │
        │  Keep first output_dim leaves           │
        │  Renormalize to sum = 1.0               │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Output Probabilities                   │
        │  [batch, ..., output_dim] (sum = 1.0)   │
        └─────────────────────────────────────────┘

    :param output_dim: Dimensionality of the output space. If None, will be
        inferred from the dimension at the specified axis of the input shape
        during build(). Must be an integer greater than 1.
    :type output_dim: Optional[int]
    :param axis: Axis along which the routing is applied. Defaults to -1
        (the last axis), following the same convention as softmax. Can be
        negative to index from the end.
    :type axis: int
    :param epsilon: Small float added to prevent numerical issues during
        probability clipping and renormalization.
    :type epsilon: float
    :param kwargs: Additional arguments for the Layer base class (e.g., name).
    :type kwargs: Any
    """

    def __init__(
            self,
            output_dim: Optional[int] = None,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        """
        Initialize the RoutingProbabilitiesLayer.

        :param output_dim: Optional integer for output dimensionality.
        :type output_dim: Optional[int]
        :param axis: Integer specifying the axis along which to apply routing.
        :type axis: int
        :param epsilon: Small float for numerical stability.
        :type epsilon: float
        :param kwargs: Additional layer arguments.
        :type kwargs: Any
        """
        super().__init__(**kwargs)

        if output_dim is not None:
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
        self.padded_output_dim = None
        self.num_decisions = None
        self.decision_weights = None
        self._normalized_axis = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer by computing output dimensions and weight patterns.

        Computes the padded output dimension to form a complete binary tree,
        determines the number of routing decisions (tree depth), and generates
        deterministic weight patterns using cosine basis functions.

        Weight pattern generation (cosine basis):
        For each decision ``k`` (row) and feature ``i`` (column):
        ``w[k,i] = cos(2*pi * (k+1) * i / D)``

        These patterns are near-orthogonal and capture different
        "frequencies" in the input features, enabling diverse routing
        decisions without training.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If axis is out of bounds for input shape.
        :raises ValueError: If output_dim cannot be inferred and is None.
        :raises ValueError: If output_dim is not greater than 1.
        """
        # Normalize axis to handle negative indices
        input_rank = len(input_shape)
        if self.axis < 0:
            self._normalized_axis = input_rank + self.axis
        else:
            self._normalized_axis = self.axis

        # Validate normalized axis
        if self._normalized_axis < 0 or self._normalized_axis >= input_rank:
            raise ValueError(
                f"axis {self.axis} is out of bounds for input shape "
                f"{input_shape}"
            )

        # Infer output_dim from input shape at specified axis if not provided
        if self.output_dim is None:
            if input_shape[self._normalized_axis] is None:
                raise ValueError(
                    f"Cannot infer output_dim when the dimension at axis "
                    f"{self.axis} of input_shape is None. Please provide "
                    f"output_dim explicitly."
                )
            self.output_dim = int(input_shape[self._normalized_axis])
            logger.info(
                f"[{self.name}] Inferred output_dim={self.output_dim} "
                f"from input shape: {input_shape} at axis {self.axis}"
            )

        # Validate output_dim
        if self.output_dim <= 1:
            raise ValueError(
                f"output_dim must be greater than 1, got {self.output_dim}"
            )

        # Calculate padded dimensions for tree structure
        # Next power of 2: 2^ceil(log2(output_dim))
        self.padded_output_dim = 1 << (self.output_dim - 1).bit_length()
        self.num_decisions = int(math.log2(self.padded_output_dim))

        logger.info(
            f"[{self.name}] Built for {self.output_dim} classes along axis "
            f"{self.axis}. Padded to {self.padded_output_dim} for tree "
            f"construction, requiring {self.num_decisions} routing decisions."
        )

        # Precompute deterministic weight patterns for each decision
        # Uses Fourier-like cosine basis to create diverse, orthogonal
        # patterns
        input_dim = input_shape[self._normalized_axis]
        decision_weights_list = []

        for decision_idx in range(self.num_decisions):
            # Create a unique pattern for each decision using cosine basis
            # This ensures different decisions respond to different feature
            # patterns
            weights = []
            for feature_idx in range(input_dim):
                # Cosine basis with varying frequency based on decision index
                weight = math.cos(
                    2.0 * math.pi * (decision_idx + 1) * feature_idx /
                    input_dim
                )
                weights.append(weight)

            # Convert to tensor and normalize to have unit L2 norm
            weight_tensor = ops.convert_to_tensor(
                weights, dtype=self.compute_dtype
            )
            weight_norm = ops.sqrt(ops.sum(ops.square(weight_tensor)))
            normalized_weights = weight_tensor / (weight_norm + self.epsilon)

            decision_weights_list.append(normalized_weights)

        # Stack all weight patterns into a single tensor for efficient
        # computation
        # Shape: (num_decisions, input_dim)
        # This is a regular attribute, not a Keras weight, as it's
        # recomputed deterministically on build, so no state needs saving.
        self.decision_weights = ops.stack(decision_weights_list, axis=0)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply hierarchical routing to transform inputs into class probabilities.

        Applies deterministic projections using cosine basis weight patterns,
        then routes probability mass through a binary tree via iterative
        splitting and interleaving.

        :param inputs: Input tensor of arbitrary rank. The routing is applied
            along the specified axis. All other dimensions are treated as
            batch dimensions.
        :type inputs: keras.KerasTensor
        :param training: Boolean or None, whether the layer is in training
            mode. Not used in this layer as it has no trainable parameters.
        :type training: Optional[bool]
        :return: Output tensor of the same shape as inputs, except the
            dimension at the specified axis may be different if
            output_dim != input_dim. Probabilities sum to 1.0 across the
            specified axis.
        :rtype: keras.KerasTensor
        """
        # Step 0: Handle axis manipulation for arbitrary rank tensors
        # Move the target axis to the last position for easier computation
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
        # Using -1 for the batch dimension handles dynamic shapes safely
        # in graph mode, unlike passing a symbolic batch_size calculation.
        input_dim = inputs_transposed.shape[-1]
        inputs_2d = ops.reshape(inputs_transposed, (-1, input_dim))

        # Step 1: Compute deterministic routing decisions from inputs
        # Shape: inputs_2d = (batch_size, input_dim)
        # Shape: decision_weights = (num_decisions, input_dim)
        # Result shape: (batch_size, num_decisions)
        decision_logits = ops.matmul(
            inputs_2d, ops.transpose(self.decision_weights)
        )

        # Apply sigmoid to convert logits to probabilities (0 to 1)
        # Each value represents the probability of taking the "right" branch
        decision_probs = ops.sigmoid(decision_logits)

        # Clip decision probabilities to prevent exactly 0 or 1
        # This avoids zero probabilities in the final output which would
        # cause NaN loss from log(0) in cross-entropy
        decision_probs = ops.clip(
            decision_probs, self.epsilon, 1.0 - self.epsilon
        )

        # Step 2: Initialize root probability using ones_like for XLA safety
        # Start with probability mass of 1.0 for each batch item
        # Shape: (batch_size, 1)
        ones_template = inputs_2d[:, 0:1]
        padded_probs = ops.ones_like(ones_template)

        # Step 3: Iteratively build the tree by splitting probabilities
        # At each level, split each existing leaf into two children
        for i in range(self.num_decisions):
            # Get routing probability for current decision level
            # Shape: (batch_size, 1)
            p_go_right = decision_probs[:, i:i + 1]
            p_go_left = 1.0 - p_go_right

            # Split probability mass for each leaf
            # Left children get: parent_prob * p_go_left
            # Right children get: parent_prob * p_go_right
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
            # Note: Using -1 for batch dimension is graph-safe
            padded_probs = ops.reshape(combined, (-1, 2 ** (i + 1)))

        # After the loop, padded_probs contains a valid probability
        # distribution over padded_output_dim leaves, guaranteed to sum
        # to 1.0

        # Step 4: Handle non-power-of-2 output dimensions
        # If output_dim != padded_dim, we need to slice and renormalize
        if self.output_dim == self.padded_output_dim:
            # Output dimension is already a power of 2, no adjustment needed
            final_probs = padded_probs
        else:
            # Slice to get only the true class probabilities
            unnormalized_probs = padded_probs[:, :self.output_dim]

            # Compute sum of sliced probabilities (will be < 1.0)
            prob_sum = ops.sum(unnormalized_probs, axis=-1, keepdims=True)

            # Renormalize to ensure output sums to 1.0
            final_probs = unnormalized_probs / prob_sum

        # Step 5: Reshape back to original shape (with axis still at last
        # position)
        # Reverse the 2D flattening from Step 0

        # Get the symbolic shape tensor of the transposed input.
        input_transposed_shape = ops.shape(inputs_transposed)
        input_transposed_shape_tensor = ops.convert_to_tensor(
            input_transposed_shape, dtype="int32"
        )

        # Slice to get all dimensions except the last one (B, H, W)
        batch_shape_tensor = input_transposed_shape_tensor[:-1]

        # Create a tensor for the new output dimension
        target_dim_tensor = ops.convert_to_tensor(
            [self.output_dim], dtype="int32"
        )

        # Concatenate to form the full target shape: (B, H, W, output_dim)
        # Both inputs to concatenate are now strictly tensors.
        target_shape_tensor = ops.concatenate(
            [batch_shape_tensor, target_dim_tensor],
            axis=0
        )

        # Reshape the 2D final_probs back to the rank of the input
        outputs_transposed = ops.reshape(final_probs, target_shape_tensor)

        # Step 6: Transpose back to original axis order if needed
        # Reverse the transpose from Step 0 to restore original axis layout
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

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple. Same as input shape except at the
            specified axis, which will be output_dim if specified.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)

        # Determine which axis to modify
        if self._normalized_axis is not None:
            axis_to_modify = self._normalized_axis
        else:
            # During shape inference before build, normalize the axis
            input_rank = len(input_shape)
            axis_to_modify = (input_rank + self.axis if self.axis < 0
                              else self.axis)

        if self.output_dim is not None:
            output_shape[axis_to_modify] = self.output_dim
        # If output_dim is None, it will be inferred in build()
        # and the output shape will match the input shape

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'axis': self.axis,
            'epsilon': self.epsilon,
        })
        return config


# ---------------------------------------------------------------------
