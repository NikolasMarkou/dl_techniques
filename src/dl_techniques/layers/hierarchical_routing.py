import math
import keras
from keras import ops
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HierarchicalRoutingLayer(keras.layers.Layer):
    """
    A generalized probabilistic binary tree layer for large-scale classification.

    This layer provides an efficient alternative to a standard `Dense` -> `Softmax`
    for tasks with any number of output classes (> 1). It works by building a
    probabilistic routing tree for a "padded" output space (the next highest
    power of two) and then renormalizing the probabilities for the true classes.

    **Intent**:
    To reduce the computational cost of the output layer from O(N) to O(log₂N),
    where N is the number of output classes. This is highly beneficial in fields
    like natural language processing where vocabulary sizes can be very large.

    **Architecture**:
    1.  **Padding**: Given `output_dim = N`, calculate `padded_dim`, the smallest
        power of two such that `padded_dim >= N`.
    2.  **Decision Making**: A `Dense` layer learns `k = log₂(padded_dim)` routing
        decisions, each with a sigmoid activation.
    3.  **Probabilistic Routing**: A binary tree is traversed, splitting
        probability mass at each of the `k` levels, producing a full
        probability distribution over the `padded_dim` virtual classes.
    4.  **Slicing**: The probabilities corresponding to the original `N` classes
        are selected from the padded distribution.
    5.  **Renormalization**: These `N` probabilities are divided by their sum to
        create a new, valid probability distribution.

    Args:
        output_dim: Integer, the dimensionality of the output space. Can be any
            integer greater than 1.
        epsilon: A small float added to the denominator during renormalization
            to prevent division-by-zero errors. Defaults to 1e-7.
        kernel_initializer: Initializer for the kernel of the internal
            `Dense` layer. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the bias of the internal
            `Dense` layer. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer function applied to the kernel
            of the internal `Dense` layer.
        **kwargs: Additional arguments for the `Layer` base class (e.g., `name`).
    """
    def __init__(
            self,
            output_dim: int,
            epsilon: float = 1e-7,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            use_bias: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(output_dim, int) or output_dim <= 1:
            raise ValueError(
                f"The 'output_dim' must be an integer greater than 1, "
                f"but received: {output_dim}"
            )
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.padded_output_dim = 1 << (output_dim - 1).bit_length()
        self.num_decisions = int(math.log2(self.padded_output_dim))
        logger.info(
            f"[{self.name}] Initialized for {self.output_dim} classes. "
            f"Padded to {self.padded_output_dim} for tree construction, "
            f"requiring {self.num_decisions} routing decisions."
        )
        self.decision_dense = keras.layers.Dense(
            units=self.num_decisions,
            use_bias=self.use_bias,
            activation='sigmoid',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='decision_dense'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        logger.info(
            f"[{self.name}] Building with input shape: {input_shape}."
        )
        self.decision_dense.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Defines the forward pass logic of the layer.
        """
        # Step 1: Get the probabilities for each of the binary decisions.
        # The sigmoid activation ensures each output is a probability (0 to 1)
        # representing the choice to go down the "right" path at each level.
        # Shape: (batch_size, num_decisions)
        decision_probs = self.decision_dense(inputs, training=training)

        # ------------------- STABILITY FIX -------------------
        # Clip decision probabilities to prevent them from becoming exactly 0 or 1.
        # This avoids multiplying by zero during tree traversal, which could lead
        # to zero probabilities in the final output. A zero probability for the
        # true class causes a NaN loss from log(0) in cross-entropy.
        decision_probs = ops.clip(
            decision_probs, self.epsilon, 1.0 - self.epsilon
        )
        # -----------------------------------------------------

        # Get the batch size dynamically for backend-agnostic compatibility.
        batch_size = ops.shape(inputs)[0]

        # Initialize the output probabilities at the root of the tree.
        # We start with a single probability mass of 1.0 for each item in the batch.
        # Shape: (batch_size, 1)
        padded_probs = ops.ones((batch_size, 1), dtype=self.compute_dtype)

        # Step 2: Iteratively traverse the tree, splitting the probability mass
        # at each level according to the learned decisions.
        for i in range(self.num_decisions):
            # Get the decision probability for the current level.
            # This is the probability of taking the "right" branch.
            # Shape: (batch_size, 1)
            p_go_right = decision_probs[:, i:i + 1]
            p_go_left = 1.0 - p_go_right

            # At each level, every existing probability leaf is split into two.
            # The probability mass of a leaf P is split into P * p_go_left
            # for its new left child and P * p_go_right for its new right child.

            # Calculate the probabilities for all the new left children.
            # Broadcasting: (batch_size, 2**i) * (batch_size, 1) -> (batch_size, 2**i)
            probs_for_left_branches = padded_probs * p_go_left

            # Calculate the probabilities for all the new right children.
            probs_for_right_branches = padded_probs * p_go_right

            # We now need to interleave these two sets of probabilities.
            # For example, if we have left_probs=[L0, L1] and right_probs=[R0, R1],
            # we want the final arrangement to be [L0, R0, L1, R1].
            # Stacking on a new axis and then reshaping achieves this efficiently.

            # 1. Stack to create a new dimension:
            # Shape: (batch_size, 2**i, 2)
            combined = ops.stack(
                [probs_for_left_branches, probs_for_right_branches], axis=2
            )

            # 2. Reshape to flatten the last two dimensions, interleaving the values:
            # Shape: (batch_size, 2**(i+1))
            padded_probs = ops.reshape(combined, (batch_size, 2 ** (i + 1)))

        # By the end of the loop, `padded_probs` is a full probability distribution
        # over the 2**num_decisions leaves of the tree.
        # The sum of probabilities across axis=1 is guaranteed to be 1.0.

        # Step 3: Slice and re-normalize if the target dimension is not a power of 2.
        if self.output_dim == self.padded_output_dim:
            # If output_dim is a power of 2, the distribution is already correct.
            return padded_probs

        # If not, we discard the "padded" probabilities and re-normalize.
        # 3a. Slice to get the probabilities for only the true classes.
        unnormalized_probs = padded_probs[:, :self.output_dim]

        # 3b. Compute the sum of these probabilities. The sum will be < 1.0
        # because we discarded the probability mass routed to padded classes.
        # `keepdims=True` is crucial for correct broadcasting during division.
        prob_sum = ops.sum(unnormalized_probs, axis=-1, keepdims=True)

        # 3c. Renormalize to ensure the final output is a valid probability
        # distribution that sums to 1. Epsilon prevents division by zero.
        final_probs = unnormalized_probs / (prob_sum + self.epsilon)

        return final_probs


    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'epsilon': self.epsilon,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
