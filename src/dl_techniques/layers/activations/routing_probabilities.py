"""
A deterministic, parameter-free routing tree for classification.

This layer provides a non-trainable alternative to the standard softmax
activation function for multi-class classification. Instead of learning a
dense transformation, it computes a probability distribution by routing an
initial probability mass through a fixed binary decision tree. This approach
is computationally efficient and introduces a structured, hierarchical bias
without adding any trainable parameters to the model.

Architecture:
    The core concept is to model the choice among `N` classes as a sequence
    of `d = log₂(N)` binary (left/right) decisions. This is represented
    conceptually as a complete binary tree of depth `d` with `N` leaf nodes,
    where each leaf corresponds to a class.

    The routing process works as follows:
    1.  **Padding**: The number of classes `output_dim` is padded to the
        next highest power of two, `padded_dim`, to ensure a complete
        binary tree structure can be formed. The number of routing
        decisions (tree depth) is `d = log₂(padded_dim)`.
    2.  **Deterministic Projections**: For each of the `d` decisions, a
        fixed, non-trainable weight vector is pre-computed. The input
        feature vector is projected onto each of these `d` vectors to
        produce `d` scalar logits.
    3.  **Probabilistic Decisions**: Each logit is passed through a sigmoid
        activation function to yield `d` probabilities. Each probability
        `p_k` represents the likelihood of taking the "right" branch at
        level `k` of the tree.
    4.  **Hierarchical Routing**: The layer simulates the flow of probability
        mass, starting with 1.0 at the root. At each level `k`, the
        probability mass at every node is split between its left and right
        children according to `1 - p_k` and `p_k`, respectively.
    5.  **Renormalization**: After `d` splits, the accumulated mass at each
        of the `padded_dim` leaves forms a valid probability
        distribution. This distribution is then truncated to the original
        `output_dim` and renormalized to sum to 1.0.

Foundational Mathematics:
    The mechanism relies on two key mathematical ideas: deterministic feature
    extraction using basis functions and hierarchical probability decomposition.

    1.  **Deterministic Weight Patterns**: The weight vectors used for the
        projections are not learned but are generated from a cosine basis,
        similar to a Fourier series. The weight for the `i`-th input feature
        in the `k`-th decision vector is given by:
        `w_{k,i} = cos(2π * (k+1) * i / D)`
        where `D` is the input feature dimension. This creates a set of
        structurally diverse, near-orthogonal vectors that are sensitive to
        different patterns (or "frequencies") in the input features without
        requiring any training. The decision logit is the dot product
        `z_k = <x, w_k>`.

    2.  **Probabilistic Tree Traversal**: The probability of reaching a
        specific leaf (class) is the product of the probabilities of the
        choices made along its unique path from the root. If a path is
        defined by a sequence of choices `(b_0, b_1, ..., b_{d-1})`, where
        `b_k ∈ {left, right}`, the leaf probability is:
        `P(leaf) = ∏_{k=0}^{d-1} P(b_k)`
        The branch probabilities at level `k` are determined by the sigmoid
        of the corresponding logit: `P(right_k) = σ(z_k)` and
        `P(left_k) = 1 - σ(z_k)`.

References:
    This architecture is a direct implementation of the "Deterministic
    Routing" mechanism proposed for softmax-free language models. For a
    detailed conceptual background, refer to:
    - Zhang, Z., et al. (2024). "Softmax-free Large-scale Language
      Modeling". arXiv preprint arXiv:2402.01258.

    The underlying concept of structuring output probabilities hierarchically
    is related to the work on Hierarchical Softmax:
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
    A non-trainable hierarchical routing layer for probabilistic classification.

    This layer provides a deterministic, parameter-free alternative to softmax by
    building a probabilistic routing tree. It computes routing decisions directly
    from input features using deterministic patterns (Fourier-like basis functions).

    **Intent**:
    To provide a parameter-free alternative to softmax that produces valid
    probability distributions through hierarchical routing. Acts as a drop-in
    replacement where you want tree-based structure without additional trainable
    parameters.

    **Architecture**:
    1. **Output Dimension Inference**: If `output_dim` is None, it is inferred
       from the input shape along the specified axis during build().
    2. **Padding**: Given `output_dim = N`, calculate `padded_dim`, the smallest
       power of two such that `padded_dim >= N`.
    3. **Deterministic Decision Making**: For each of `k = log₂(padded_dim)`
       routing decisions, compute a scalar from the inputs using precomputed
       deterministic weight patterns (based on cosine basis functions) and
       apply sigmoid activation.
    4. **Probabilistic Routing**: Traverse a binary tree, splitting probability
       mass at each of the `k` levels to produce a distribution over
       `padded_dim` virtual classes.
    5. **Slicing & Renormalization**: Select the probabilities for the original
       `N` classes and renormalize to ensure a valid probability distribution.

    Args:
        output_dim: Optional integer, the dimensionality of the output space.
            If None, will be inferred from the dimension at the specified axis
            of the input shape during build(). Must be an integer greater than 1.
        axis: Integer, the axis along which the routing is applied. Defaults to -1
            (the last axis), following the same convention as softmax. Can be
            negative to index from the end.
        epsilon: A small float added to prevent numerical issues during
            probability clipping and renormalization. Defaults to 1e-7.
        **kwargs: Additional arguments for the `Layer` base class (e.g., `name`).

    Example:
        >>> # As a drop-in replacement for softmax on the last axis
        >>> inputs = keras.layers.Input(shape=(128,))
        >>> logits = keras.layers.Dense(10)(inputs)
        >>> probs = RoutingProbabilitiesLayer(output_dim=10)(logits)
        >>>
        >>> # With axis parameter for arbitrary shapes
        >>> inputs = keras.layers.Input(shape=(32, 64, 10))
        >>> # Apply routing along axis 1
        >>> probs = RoutingProbabilitiesLayer(axis=1)(inputs)  # shape: (32, 64, 10)
        >>>
        >>> # Infer output_dim from input
        >>> probs = RoutingProbabilitiesLayer()(logits)
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

        Args:
            output_dim: Optional integer for output dimensionality.
            axis: Integer specifying the axis along which to apply routing.
            epsilon: Small float for numerical stability.
            **kwargs: Additional layer arguments.
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
        Builds the layer by computing output dimensions and decision weight patterns.

        Args:
            input_shape: Shape tuple of the input tensor.
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
                f"axis {self.axis} is out of bounds for input shape {input_shape}"
            )

        # Infer output_dim from input shape at the specified axis if not provided
        if self.output_dim is None:
            if input_shape[self._normalized_axis] is None:
                raise ValueError(
                    f"Cannot infer output_dim when the dimension at axis {self.axis} "
                    f"of input_shape is None. Please provide output_dim explicitly."
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
        self.padded_output_dim = 1 << (self.output_dim - 1).bit_length()
        self.num_decisions = int(math.log2(self.padded_output_dim))

        logger.info(
            f"[{self.name}] Built for {self.output_dim} classes along axis {self.axis}. "
            f"Padded to {self.padded_output_dim} for tree construction, "
            f"requiring {self.num_decisions} routing decisions."
        )

        # Precompute deterministic weight patterns for each decision
        # Uses Fourier-like cosine basis to create diverse, orthogonal patterns
        input_dim = input_shape[self._normalized_axis]
        decision_weights_list = []

        for decision_idx in range(self.num_decisions):
            # Create a unique pattern for each decision using cosine basis
            # This ensures different decisions respond to different feature patterns
            weights = []
            for feature_idx in range(input_dim):
                # Cosine basis with varying frequency based on decision index
                weight = math.cos(
                    2.0 * math.pi * (decision_idx + 1) * feature_idx / input_dim
                )
                weights.append(weight)

            # Convert to tensor and normalize to have unit L2 norm
            weight_tensor = ops.convert_to_tensor(weights, dtype=self.compute_dtype)
            weight_norm = ops.sqrt(ops.sum(ops.square(weight_tensor)))
            normalized_weights = weight_tensor / (weight_norm + self.epsilon)

            decision_weights_list.append(normalized_weights)

        # Stack all weight patterns into a single tensor for efficient computation
        # Shape: (num_decisions, input_dim)
        # This is a regular attribute, not a Keras weight, as it's recomputed
        # deterministically on build, so no state needs to be saved.
        self.decision_weights = ops.stack(decision_weights_list, axis=0)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Defines the forward pass logic of the layer.

        Args:
            inputs: Input tensor of arbitrary rank. The routing is applied along
                the specified axis. All other dimensions are treated as batch dimensions.
            training: Boolean or None, whether the layer is in training mode.

        Returns:
            Output tensor of the same shape as inputs, except the dimension at
            the specified axis may be different if output_dim != input_dim.
            Probabilities sum to 1.0 across the specified axis.
        """
        # Step 0: Handle axis manipulation for arbitrary rank tensors
        # Move the target axis to the last position for easier computation
        input_shape = ops.shape(inputs)
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
        # where batch_size is the product of all dimensions except the last
        transposed_shape = ops.shape(inputs_transposed)
        batch_size = ops.prod(transposed_shape[:-1])
        input_dim = transposed_shape[-1]

        inputs_2d = ops.reshape(inputs_transposed, (batch_size, input_dim))

        # Step 1: Compute deterministic routing decisions from inputs
        # For each decision, compute weighted sum of inputs using precomputed patterns
        # Shape: inputs_2d = (batch_size, input_dim)
        # Shape: decision_weights = (num_decisions, input_dim)
        # Result shape: (batch_size, num_decisions)
        decision_logits = ops.matmul(inputs_2d, ops.transpose(self.decision_weights))

        # Apply sigmoid to convert logits to probabilities (0 to 1)
        # Each value represents the probability of taking the "right" branch
        decision_probs = ops.sigmoid(decision_logits)

        # Clip decision probabilities to prevent exactly 0 or 1
        # This avoids zero probabilities in the final output which would
        # cause NaN loss from log(0) in cross-entropy
        decision_probs = ops.clip(
            decision_probs, self.epsilon, 1.0 - self.epsilon
        )

        # Step 2: Initialize root probability
        # Start with probability mass of 1.0 for each batch item
        # Shape: (batch_size, 1)
        padded_probs = ops.ones((batch_size, 1), dtype=self.compute_dtype)

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
            padded_probs = ops.reshape(combined, (batch_size, 2 ** (i + 1)))

        # After the loop, padded_probs contains a valid probability distribution
        # over padded_output_dim leaves, guaranteed to sum to 1.0

        # Step 4: Handle non-power-of-2 output dimensions
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

        # Step 5: Reshape back to original shape (with axis still at last position)
        output_shape_transposed = list(transposed_shape)
        output_shape_transposed[-1] = self.output_dim

        # Convert to concrete values where possible for reshape
        output_shape_concrete = []
        for i, dim in enumerate(output_shape_transposed[:-1]):
            if i < len(inputs_transposed.shape) - 1 and inputs_transposed.shape[i] is not None:
                output_shape_concrete.append(inputs_transposed.shape[i])
            else:
                output_shape_concrete.append(dim)
        output_shape_concrete.append(self.output_dim)

        outputs_transposed = ops.reshape(final_probs, output_shape_concrete)

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
        Computes the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple. Same as input shape except at the specified axis,
            which will be output_dim if specified.
        """
        output_shape = list(input_shape)

        # Determine which axis to modify
        if self._normalized_axis is not None:
            axis_to_modify = self._normalized_axis
        else:
            # During shape inference before build, normalize the axis
            input_rank = len(input_shape)
            axis_to_modify = input_rank + self.axis if self.axis < 0 else self.axis

        if self.output_dim is not None:
            output_shape[axis_to_modify] = self.output_dim
        # If output_dim is None, it will be inferred in build()
        # and the output shape will match the input shape

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'axis': self.axis,
            'epsilon': self.epsilon,
        })
        return config

# ---------------------------------------------------------------------