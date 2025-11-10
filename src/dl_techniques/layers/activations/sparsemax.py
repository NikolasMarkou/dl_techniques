"""
Projects a vector of logits onto the probability simplex for sparse outputs.

This layer implements the Sparsemax activation function, a sparse alternative
to the conventional softmax function. While softmax maps logits to a dense
probability distribution where all elements are positive, Sparsemax projects
the logits onto the probability simplex using a Euclidean (L2) projection.
This process yields a sparse probability distribution where many of the output
values are exactly zero.

The conceptual foundation of Sparsemax lies in its geometric interpretation.
Given an input vector of logits `z`, the function finds the point `p` on the
probability simplex (i.e., `p_i >= 0` and `sum(p_i) = 1`) that is closest to
`z` in terms of Euclidean distance. This contrasts with softmax, which can be
viewed as a projection based on KL divergence. The L2 projection naturally
results in sparse solutions, making it highly suitable for applications like
attention mechanisms where the model should focus on a small, interpretable
subset of inputs.

The algorithm to compute this projection involves three main steps:
1.  **Sorting**: The input logits `z` are sorted in descending order.
2.  **Support Identification**: The algorithm identifies the size of the
    support of the resulting probability distribution, denoted as `k(z)`.
    This is the number of non-zero probabilities in the output. It is found
    by locating the largest `k` for which `1 + k * z_k > sum_{j=1 to k}(z_j)`,
    where `z_j` are the sorted logits.
3.  **Thresholding and Projection**: A threshold value, `τ(z)`, is calculated
    based on the cumulative sum of the top `k(z)` sorted logits. This
    threshold is then subtracted from the original logits, and a `max(0, ...)`
    operation is applied. The final projection is given by the equation:
    `sparsemax(z)_i = max(0, z_i - τ(z))`.

This procedure guarantees that the output is a valid, sparse probability
distribution, differentiable almost everywhere, allowing it to be used as a
drop-in replacement for softmax in neural network architectures.

References:
    - Martins & Astudillo, 2016. "From Softmax to Sparsemax: A Sparse
      Model of Attention and Multi-Label Classification".
      (https://arxiv.org/abs/1602.02068)
"""

import keras
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Sparsemax(keras.layers.Layer):
    """
    Sparsemax activation function layer for sparse probability distributions.

    Sparsemax projects input logits onto the probability simplex using Euclidean
    distance (L2 projection) instead of KL divergence (like softmax), resulting in
    sparse outputs where many probabilities are exactly zero. This is particularly
    useful for attention mechanisms and multi-label classification where sparse
    selections are desired.

    **Intent**: Provide a differentiable sparse alternative to softmax that produces
    probability distributions with many exact zeros, improving interpretability and
    potentially reducing overfitting in attention-based models.

    **Architecture**:
    ```
    Input: logits [..., K]
           ↓
    Sort: z_sorted = sort(logits, descending)
           ↓
    Cumsum: cumsum_k = Σ(z_1...z_k)
           ↓
    Support: find largest k where 1 + k·z_k > cumsum_k
           ↓
    Threshold: τ(z) = (cumsum_k_z - 1) / k_z
           ↓
    Project: p_i = max(0, z_i - τ(z))
           ↓
    Output: sparse probabilities [..., K] where Σp_i = 1
    ```

    **Mathematical Operations**:
    1. **Support Calculation**:
       k(z) = max{k ∈ [K] : 1 + k·z_k > Σ_{j=1}^k z_j}
       where z_1 ≥ z_2 ≥ ... ≥ z_K are sorted logits

    2. **Threshold Computation**:
       τ(z) = (Σ_{j=1}^{k(z)} z_j - 1) / k(z)

    3. **Projection**:
       sparsemax(z)_i = max(0, z_i - τ(z))

    **Properties**:
    - Output is a valid probability distribution: p_i ≥ 0, Σp_i = 1
    - Many outputs are exactly zero (sparse)
    - Differentiable everywhere (though not strictly differentiable at boundaries)
    - Reduces to softmax behavior when all inputs should be non-zero

    References:
        Martins & Astudillo (2016). "From Softmax to Sparsemax: A Sparse
        Model of Attention and Multi-Label Classification". ICML 2016.
        https://arxiv.org/abs/1602.02068

    Args:
        axis: Integer, axis along which to compute sparsemax normalization.
            Typically -1 for the last axis (class/attention dimension).
            Must be a valid axis index for input tensors. Defaults to -1.
        **kwargs: Additional keyword arguments passed to the Layer base class,
            such as `name`, `dtype`, `trainable`.

    Input shape:
        Arbitrary tensor with shape `(..., K)` where K is the number of classes
        or attention positions. Common shapes:
        - 2D: `(batch_size, num_classes)` for classification
        - 3D: `(batch_size, sequence_length, vocab_size)` for attention

    Output shape:
        Same shape as input. Values form a valid probability distribution along
        the specified axis (sum to 1.0, all non-negative, many exactly zero).

    Attributes:
        axis: The normalization axis.

    Example:
        ```python
        # Classification scenario
        layer = Sparsemax()
        logits = keras.ops.convert_to_tensor([[-2.0, 0.0, 0.5, 3.0]])
        probabilities = layer(logits)
        # Output: [[0.0, 0.0, 0.167, 0.833]] (approximately)
        # Note: First two values are exactly zero (sparse!)

        # Attention mechanism
        attention_layer = Sparsemax(axis=-1)
        attention_scores = keras.random.normal((2, 10, 512))
        attention_weights = attention_layer(attention_scores)
        # Many attention weights will be exactly zero

        # Compare with softmax (no exact zeros)
        softmax_probs = keras.activations.softmax(logits)
        # Output: [[0.001, 0.067, 0.110, 0.822]] (no exact zeros)
        ```

    Note:
        - Sparsemax is differentiable but gradient computation is more complex
          than softmax. The gradient is piece-wise linear.
        - For numerical stability, input logits should be reasonably scaled
          (similar considerations as softmax).
        - The sparsity pattern (which outputs are zero) depends on input values
          and can change during training.
        - Consider using SparsemaxLoss instead of categorical cross-entropy
          for optimal training with sparsemax outputs.
    """

    def __init__(
            self,
            axis: int = -1,
            **kwargs: Any
    ) -> None:
        """
        Initialize Sparsemax activation layer.

        Args:
            axis: Axis along which to compute sparsemax normalization.
            **kwargs: Additional layer arguments (name, dtype, etc.).

        Raises:
            ValueError: If axis is not an integer.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(axis, int):
            raise ValueError(
                f"axis must be an integer, got {type(axis).__name__}"
            )

        self.axis = axis

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply sparsemax activation to input logits.

        The computation follows these steps:
        1. Sort logits in descending order
        2. Compute cumulative sums of sorted logits
        3. Find support size k(z) where 1 + k·z_k > cumsum_k
        4. Calculate threshold τ(z) = (cumsum_k(z) - 1) / k(z)
        5. Project: output = max(0, input - τ)

        Args:
            inputs: Input tensor of logits, arbitrary shape.
            training: Optional boolean for training mode (unused, but standard).

        Returns:
            Sparse probability distribution with same shape as inputs.
            Values sum to 1.0 along specified axis, many are exactly zero.
        """
        # Sort logits in descending order along specified axis
        sorted_logits = keras.ops.sort(inputs, axis=self.axis)
        sorted_logits = keras.ops.flip(sorted_logits, axis=self.axis)

        # Get shape information for the normalization axis
        shape = keras.ops.shape(inputs)
        k = shape[self.axis]

        # Compute cumulative sums: cumsum[k] = sum(z_1 to z_k)
        z_cumsum = keras.ops.cumsum(sorted_logits, axis=self.axis)

        # Create k values [1, 2, ..., K] for support calculation
        # Need to broadcast to match input dimensions
        k_values = keras.ops.arange(1, k + 1, dtype=inputs.dtype)

        # Reshape k_values to broadcast correctly
        # Create shape: [1, 1, ..., 1, K] where K is at self.axis position
        ndim = len(inputs.shape)
        target_shape = [1] * ndim
        # Handle negative axis
        actual_axis = self.axis if self.axis >= 0 else ndim + self.axis
        target_shape[actual_axis] = k
        k_values = keras.ops.reshape(k_values, target_shape)

        # Find support: elements where 1 + k * z_k - cumsum_k > 0
        # This identifies which k is the cutoff point
        support = 1.0 + k_values * sorted_logits - z_cumsum
        support_mask = keras.ops.cast(support > 0, inputs.dtype)

        # k_z: largest k where support > 0 (support size)
        k_z = keras.ops.sum(support_mask, axis=self.axis, keepdims=True)

        # Get cumulative sum at position k_z - 1 (0-indexed)
        indices = keras.ops.cast(k_z - 1, "int32")
        z_cumsum_at_k = keras.ops.take_along_axis(
            z_cumsum,
            indices,
            axis=self.axis
        )

        # Compute threshold: τ(z) = (cumsum_k(z) - 1) / k(z)
        tau = (z_cumsum_at_k - 1.0) / k_z

        # Project onto simplex: sparsemax(z) = max(0, z - τ)
        output = keras.ops.maximum(inputs - tau, 0.0)

        return output

    def compute_output_shape(
            self,
            input_shape: tuple
    ) -> tuple:
        """
        Compute output shape (same as input shape).

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration with all constructor
            parameters needed to recreate the layer.
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
        })
        return config

# ---------------------------------------------------------------------
