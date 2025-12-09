"""
Projects a vector of logits onto the probability simplex for sparse outputs.

This layer implements the Sparsemax activation function, a sparse alternative
to the conventional softmax function. While softmax maps logits to a dense
probability distribution where all elements are positive, Sparsemax projects
the logits onto the probability simplex using a Euclidean (L2) projection.

--------------------------------------------------------------------------
ARCHITECTURAL DECISIONS & XLA COMPATIBILITY NOTES
--------------------------------------------------------------------------
This implementation is heavily "opinionated" to ensure stability with XLA
(Accelerated Linear Algebra) compilation, which is used by TensorFlow and JAX.
Standard Python/Numpy idioms often fail during graph compilation due to
dynamic shape inference issues.

1.  **Flattening vs. N-D Broadcasting**:
    *   *Attempt*: Operating directly on N-D tensors (e.g., [Batch, Seq, Heads, K]).
    *   *Failure*: XLA often fails to infer broadcast shapes dynamically when
        mixing Rank-1 support vectors with Rank-N inputs inside `where` or
        boolean masking ops.
    *   *Decision*: We flatten all inputs to 2D `(N, K)` before processing.
        This reduces the problem to a canonical Rank-2 vs Rank-1 operation,
        which compilers can optimize reliably without shape ambiguity.

2.  **Masking vs. Gathering (`take_along_axis`)**:
    *   *Attempt*: Using `ops.take_along_axis` to select the cumulative sum
        value at the threshold index `k(z)`.
    *   *Failure*: `take_along_axis` with dynamic indices (indices determined
        by data values during the forward pass) forces the compiler to generate
        dynamic slice operations. If the compiler cannot prove the bounds are
        valid at compile-time, it often throws `BroadcastArgs` or `Invalid Argument`
        errors.
    *   *Decision*: We use `one_hot` encoding + multiplication (`sum(vals * mask)`).
        While computationally slightly heavier (O(K) vs O(1) fetch), it relies
        purely on matrix arithmetic (Multiplication/Add), which is shape-static
        and universally supported by all hardware accelerators.

3.  **Explicit Reshaping**:
    *   *Decision*: We explicitly reshape support vectors to `(1, K)` rather
        than relying on implicit NumPy-style broadcasting. This removes any
        ambiguity in the computation graph regarding which dimension is being
        broadcasted.

References:
    - Martins & Astudillo, 2016. "From Softmax to Sparsemax: A Sparse
      Model of Attention and Multi-Label Classification".
      (https://arxiv.org/abs/1602.02068)
"""

import keras
from keras import ops
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Sparsemax(keras.layers.Layer):
    """
    Sparsemax activation function layer for sparse probability distributions.

    **Intent**: Provide a differentiable sparse alternative to softmax that produces
    probability distributions with many exact zeros.

    **XLA-Safe Implementation**:
    This class employs a "Flatten-Mask-Restore" strategy to avoid dynamic
    tensor slicing and ambiguous broadcasting, making it robust for
    `@tf.function` and `jit_compile=True` environments.

    Args:
        axis: Integer, axis along which to compute sparsemax normalization.
            Typically -1. Defaults to -1.
        **kwargs: Additional keyword arguments passed to the Layer base class.
    """

    def __init__(
            self,
            axis: int = -1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(axis, int):
            raise ValueError(f"axis must be an integer, got {type(axis).__name__}")
        self.axis = axis

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply sparsemax activation to input logits.

        Args:
            inputs: Input tensor of logits, arbitrary shape.
            training: Unused.

        Returns:
            Sparse probability distribution with same shape as inputs.
        """
        # Store original shape for restoration
        input_shape = inputs.shape
        ndim = len(input_shape)

        # Normalize axis to positive index (e.g., -1 -> 2 for rank 3)
        axis = self.axis if self.axis >= 0 else ndim + self.axis

        # =====================================================================
        # DECISION 1: Standardize Memory Layout (Permutation)
        # =====================================================================
        # Operations like `sort` and `cumsum` are most efficient on the last
        # contiguous dimension in memory. If the user wants to normalize a
        # middle dimension (e.g., axis=1), we transpose it to the end.
        if axis != ndim - 1:
            # Create permutation: [0, ..., axis-1, axis+1, ..., axis]
            perm_order = list(range(ndim))
            perm_order.pop(axis)
            perm_order.append(axis)

            inputs_permuted = ops.transpose(inputs, perm_order)

            # Prepare inverse permutation to restore later
            inv_perm_order = list(range(ndim - 1))
            inv_perm_order.insert(axis, ndim - 1)
        else:
            inputs_permuted = inputs
            inv_perm_order = None

        # =====================================================================
        # DECISION 2: Flatten to 2D (The "Anti-Broadcast" Strategy)
        # =====================================================================
        # XLA struggles to broadcast a computed 1D support vector against a
        # dynamic ND tensor (e.g., 5D tensor in video transformers).
        # By collapsing all batch dimensions into one 'N', we guarantee the
        # operation is always (N, K) vs (1, K) or (N, 1).
        # This makes the graph topology static and predictable.

        # Use symbolic shape to handle dynamic batch sizes (None)
        permuted_shape = ops.shape(inputs_permuted)

        # Determine K (the feature dimension size)
        # We prefer the static shape if available for compile-time optimization
        if input_shape[axis] is not None:
            k = int(input_shape[axis])
        else:
            k = permuted_shape[-1]

        # Reshape to (-1, K)
        # -1 infers the total batch size dynamically
        inputs_2d = ops.reshape(inputs_permuted, (-1, k))

        # =====================================================================
        # CORE ALGORITHM: Sparsemax Logic
        # =====================================================================

        # 1. Sort logits (descending)
        # Necessary to find the "elbow" where probabilities drop to zero.
        sorted_logits = ops.sort(inputs_2d, axis=-1)
        sorted_logits = ops.flip(sorted_logits, axis=-1)

        # 2. Cumulative Sum
        # Used to check the condition: 1 + k * z_k > sum(z_1..z_k)
        z_cumsum = ops.cumsum(sorted_logits, axis=-1)

        # 3. Create range vector [1, 2, ..., K]
        k_values = ops.arange(1, k + 1, dtype=inputs.dtype)

        # Explicit Reshape: (K,) -> (1, K)
        # "Attempt": Just use k_values.
        # "Fix": Reshape to (1, K) so XLA sees explicit Rank-2 broadcasting.
        k_values = ops.reshape(k_values, (1, k))

        # 4. Support Identification
        # Calculate the condition for every element.
        # support > 0 means that element is part of the active set.
        support = 1.0 + k_values * sorted_logits - z_cumsum
        support_mask = ops.cast(support > 0, inputs.dtype)

        # k_z is the count of active elements (the "support size")
        # Shape: (N, 1) - One value per sample in the flattened batch
        k_z = ops.sum(support_mask, axis=-1, keepdims=True)

        # =====================================================================
        # DECISION 3: Arithmetic Masking vs. Index Gathering
        # =====================================================================
        # "Attempt": `ops.take_along_axis(z_cumsum, k_z - 1, axis=-1)`
        # "Problem": XLA treats `k_z - 1` as a dynamic index. Slicing with
        #            dynamic indices requires the compiler to support dynamic
        #            memory access patterns, which often fails graph fusion.
        # "Fix": Use One-Hot Encoding + Multiplication.
        #        This converts an Index lookup (Gather) into Math (Mult/Sum).
        #        Math is always XLA-safe.

        # Cast k_z to int32 for indexing/one-hot operations
        support_indices = ops.cast(k_z - 1, "int32")

        # Reshape to 1D to satisfy one_hot requirements
        support_indices = ops.reshape(support_indices, (-1,))

        # Create One-Hot Mask: (N, K)
        # Only the position corresponding to k(z) will be 1.0, others 0.0
        gather_mask = ops.one_hot(support_indices, k, dtype=inputs.dtype)

        # Select the cumulative sum at the threshold boundary
        # z_cumsum * mask zeroes out everything except the target value.
        # Summing collapses the row to the single target value.
        z_cumsum_at_k = ops.sum(z_cumsum * gather_mask, axis=-1, keepdims=True)

        # =====================================================================
        # Final Projection
        # =====================================================================

        # Calculate Tau (Threshold)
        # τ = (sum(z_support) - 1) / |support|
        tau = (z_cumsum_at_k - 1.0) / k_z

        # Projection: P = max(0, z - τ)
        # This naturally sets elements outside the support to exactly zero.
        output_2d = ops.maximum(inputs_2d - tau, 0.0)

        # =====================================================================
        # DECISION 4: Restore Structure
        # =====================================================================
        # Un-flatten and un-permute to return a tensor indistinguishable
        # from the input structure.

        # Reshape back to permuted shape (e.g., [Batch, Seq, Heads, K])
        output_permuted = ops.reshape(output_2d, permuted_shape)

        # Transpose back if we changed the axis order
        if inv_perm_order is not None:
            output = ops.transpose(output_permuted, inv_perm_order)
        else:
            output = output_permuted

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
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
        })
        return config

# ---------------------------------------------------------------------