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
    """
    def __init__(
            self,
            output_dim: Optional[int] = None,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if output_dim is not None:
            if not isinstance(output_dim, int) or output_dim <= 1:
                raise ValueError(f"The 'output_dim' must be an integer greater than 1, but received: {output_dim}")
        if not isinstance(axis, int):
            raise ValueError(f"The 'axis' must be an integer, but received: {axis}")
        self.output_dim = output_dim
        self.axis = axis
        self.epsilon = epsilon
        self.padded_output_dim = None
        self.num_decisions = None
        self.decision_weights = None
        self._normalized_axis = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        input_rank = len(input_shape)
        if self.axis < 0:
            self._normalized_axis = input_rank + self.axis
        else:
            self._normalized_axis = self.axis
        if self._normalized_axis < 0 or self._normalized_axis >= input_rank:
            raise ValueError(f"axis {self.axis} is out of bounds for input shape {input_shape}")
        if self.output_dim is None:
            if input_shape[self._normalized_axis] is None:
                raise ValueError(f"Cannot infer output_dim when the dimension at axis {self.axis} of input_shape is None. Please provide output_dim explicitly.")
            self.output_dim = int(input_shape[self._normalized_axis])
            logger.info(f"[{self.name}] Inferred output_dim={self.output_dim} from input shape: {input_shape} at axis {self.axis}")
        if self.output_dim <= 1:
            raise ValueError(f"output_dim must be greater than 1, got {self.output_dim}")
        self.padded_output_dim = 1 << (self.output_dim - 1).bit_length()
        self.num_decisions = int(math.log2(self.padded_output_dim))
        logger.info(f"[{self.name}] Built for {self.output_dim} classes along axis {self.axis}. Padded to {self.padded_output_dim} for tree construction, requiring {self.num_decisions} routing decisions.")
        input_dim = input_shape[self._normalized_axis]
        decision_weights_list = []
        for decision_idx in range(self.num_decisions):
            weights = []
            for feature_idx in range(input_dim):
                weight = math.cos(2.0 * math.pi * (decision_idx + 1) * feature_idx / input_dim)
                weights.append(weight)
            weight_tensor = ops.convert_to_tensor(weights, dtype=self.compute_dtype)
            weight_norm = ops.sqrt(ops.sum(ops.square(weight_tensor)))
            normalized_weights = weight_tensor / (weight_norm + self.epsilon)
            decision_weights_list.append(normalized_weights)
        self.decision_weights = ops.stack(decision_weights_list, axis=0)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        input_rank = len(inputs.shape)
        perm = list(range(input_rank))
        perm[self._normalized_axis] = input_rank - 1
        perm[input_rank - 1] = self._normalized_axis
        if self._normalized_axis != input_rank - 1:
            inputs_transposed = ops.transpose(inputs, perm)
        else:
            inputs_transposed = inputs
        transposed_shape = ops.shape(inputs_transposed)
        batch_size = ops.prod(transposed_shape[:-1])
        input_dim = transposed_shape[-1]
        inputs_2d = ops.reshape(inputs_transposed, (batch_size, input_dim))

        decision_logits = ops.matmul(inputs_2d, ops.transpose(self.decision_weights))
        decision_probs = ops.sigmoid(decision_logits)
        decision_probs = ops.clip(decision_probs, self.epsilon, 1.0 - self.epsilon)

        # Step 2: Initialize root probability using ones_like for XLA safety
        # Create a slice of the input with shape (batch_size, 1)
        ones_template = inputs_2d[:, 0:1]
        # Create a tensor of ones with the same dynamic shape
        padded_probs = ops.ones_like(ones_template)

        for i in range(self.num_decisions):
            p_go_right = decision_probs[:, i:i + 1]
            p_go_left = 1.0 - p_go_right
            probs_for_left_branches = padded_probs * p_go_left
            probs_for_right_branches = padded_probs * p_go_right
            combined = ops.stack([probs_for_left_branches, probs_for_right_branches], axis=2)
            padded_probs = ops.reshape(combined, (batch_size, 2 ** (i + 1)))

        if self.output_dim == self.padded_output_dim:
            final_probs = padded_probs
        else:
            unnormalized_probs = padded_probs[:, :self.output_dim]
            prob_sum = ops.sum(unnormalized_probs, axis=-1, keepdims=True)
            final_probs = unnormalized_probs / prob_sum

        output_shape_transposed = list(transposed_shape)
        output_shape_transposed[-1] = self.output_dim
        output_shape_concrete = []
        for i, dim in enumerate(output_shape_transposed[:-1]):
            if i < len(inputs_transposed.shape) - 1 and inputs_transposed.shape[i] is not None:
                output_shape_concrete.append(inputs_transposed.shape[i])
            else:
                output_shape_concrete.append(dim)
        output_shape_concrete.append(self.output_dim)
        outputs_transposed = ops.reshape(final_probs, output_shape_concrete)

        if self._normalized_axis != input_rank - 1:
            outputs = ops.transpose(outputs_transposed, perm)
        else:
            outputs = outputs_transposed
        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        output_shape = list(input_shape)
        if self._normalized_axis is not None:
            axis_to_modify = self._normalized_axis
        else:
            input_rank = len(input_shape)
            axis_to_modify = input_rank + self.axis if self.axis < 0 else self.axis
        if self.output_dim is not None:
            output_shape[axis_to_modify] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'output_dim': self.output_dim, 'axis': self.axis, 'epsilon': self.epsilon})
        return config
# ---------------------------------------------------------------------