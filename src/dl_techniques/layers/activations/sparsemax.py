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
        mixing Rank-1 support vectors with Rank-N inputs inside ``where`` or
        boolean masking keras.ops.
    *   *Decision*: We flatten all inputs to 2D ``(N, K)`` before processing.
        This reduces the problem to a canonical Rank-2 vs Rank-1 operation,
        which compilers can optimize reliably without shape ambiguity.

2.  **Masking vs. Gathering (take_along_axis)**:
    *   *Attempt*: Using ``ops.take_along_axis`` to select the cumulative sum
        value at the threshold index ``k(z)``.
    *   *Failure*: ``take_along_axis`` with dynamic indices forces the compiler
        to generate dynamic slice operations. If the compiler cannot prove the
        bounds are valid at compile-time, it often throws errors.
    *   *Decision*: We use ``one_hot`` encoding + multiplication (``sum(vals * mask)``).
        While computationally slightly heavier (O(K) vs O(1) fetch), it relies
        purely on matrix arithmetic, which is shape-static and universally
        supported by all hardware accelerators.

3.  **Explicit Reshaping**:
    *   *Decision*: We explicitly reshape support vectors to ``(1, K)`` rather
        than relying on implicit NumPy-style broadcasting. This removes any
        ambiguity in the computation graph regarding which dimension is being
        broadcasted.

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
    """Sparsemax activation function layer for sparse probability distributions.

    Sparsemax projects logits onto the probability simplex using an L2
    projection, producing distributions with many exact zeros. Unlike softmax,
    which always assigns non-zero probabilities to all classes, Sparsemax
    encourages sparsity. This XLA-safe implementation employs a
    "Flatten-Mask-Restore" strategy to avoid dynamic tensor slicing and
    ambiguous broadcasting.

    **Architecture Overview:**

    .. code-block:: text

        Input: logits (batch, ..., K)
                │
                ▼
        ┌───────────────────────────┐
        │  Permute axis to last dim │
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Flatten to 2D: (N, K)    │
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Sort descending          │
        │  Cumulative sum           │
        │  Find support set k(z)    │
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Compute threshold tau    │
        │  tau = (cumsum - 1) / k   │
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Project: max(z - tau, 0) │
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Restore original shape   │
        └───────────┬───────────────┘
                    │
                    ▼
        Output: (batch, ..., K) sparse probabilities

    :param axis: Axis along which to compute sparsemax normalization.
        Typically -1. Defaults to -1. Must be in ``[-ndim, ndim - 1]`` for the
        rank of the tensor the layer is called on; that range can only be
        checked at call time, so an out-of-range value is rejected there.
    :type axis: int
    :param kwargs: Additional keyword arguments passed to the Layer base class.
    """

    def __init__(
            self,
            axis: int = -1,
            **kwargs: Any
    ) -> None:
        """Initialize the Sparsemax layer.

        :param axis: Axis along which to compute sparsemax normalization.
        :type axis: int
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If axis is not an integer, or is a bool. The RANGE
            of ``axis`` depends on the input rank and is therefore validated in
            :meth:`call`, not here.
        """
        super().__init__(**kwargs)
        # `bool` is a subclass of `int`, so `isinstance(True, int)` is True and
        # `Sparsemax(axis=True)` used to be accepted and silently behave as
        # `axis=1`. Reject it: a bool is never a meaningful axis.
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise ValueError(f"axis must be an integer, got {type(axis).__name__}")
        self.axis = axis

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply sparsemax activation to input logits.

        :param inputs: Input tensor of logits, arbitrary shape.
        :type inputs: keras.KerasTensor
        :param training: Unused.
        :type training: Optional[bool]
        :return: Sparse probability distribution with same shape as inputs.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``axis`` is out of range for the rank of
            ``inputs``, i.e. outside ``[-ndim, ndim - 1]``.
        """
        # Store original shape for restoration
        input_shape = inputs.shape
        ndim = len(input_shape)

        # DECISION plan-2026-07-29T110112-09832856/D-014
        # ---------------------------------------------------------------
        # RANGE-CHECK `axis` HERE, BEFORE ANY `list.pop` / `ops.transpose`.
        # DO NOT move this to `__init__` (the rank is not known there) and DO
        # NOT delete it as "defensive programming" — it is the only thing
        # standing between a public constructor argument and an UNCATCHABLE
        # PROCESS ABORT. Measured on the pre-fix bytes, one case per process
        # (a shared process cannot survive the second band), with
        # `norm = ndim + axis`:
        #   * `norm == -1` (i.e. `axis == -(ndim + 1)`): SILENT WRONG ANSWER.
        #     `perm_order.pop(-1)` + `append(-1)` makes the forward transpose a
        #     no-op, so the projection itself is numerically correct, but
        #     `inv_perm_order.insert(-1, ndim - 1)` builds a NON-INVERSE
        #     permutation and the right answer is returned in the wrong layout.
        #     Input `(2, 4, 5)` -> output `(2, 5, 4)`, last-axis sums
        #     `[0.1687, 1.3085, 1.0049, 0.7024]` — not a distribution — while
        #     `compute_output_shape` still declares `(2, 4, 5)`, so the
        #     functional shape contract is violated too.
        #   * `norm in [-ndim, -2]`: TF C++ `LOG(FATAL)` in
        #     `tensor_shape.cc:356 Check failed: d >= 0`, SIGABRT, exit 134.
        #     NO PYTHON `except` CAN CATCH IT. It kills the interpreter.
        #   * `norm < -ndim` or `axis >= ndim`: `IndexError` from `list.pop`,
        #     loud but naming neither the axis nor the rank.
        # Python's `list.pop` / `list.insert` accept negative indices silently,
        # which is precisely what converts a user error into a wrong answer
        # instead of a raise. `keras.layers.Softmax` is rejected at the op
        # level; this layer's transpose shim intercepts before the op is ever
        # reached, so the shim has to do the rejecting itself.
        # Guarded by `TestSparsemax::test_out_of_range_axis_raises_value_error`
        # (both bands) and `::test_every_in_range_axis_is_accepted_at_ranks_1_to_4`
        # (the over-rejection control). See decisions.md D-014 / D-004.
        # ---------------------------------------------------------------
        if not -ndim <= self.axis < ndim:
            raise ValueError(
                f"axis={self.axis} is out of range for an input of rank "
                f"{ndim} (shape {tuple(input_shape)}); axis must be in "
                f"[{-ndim}, {ndim - 1}]"
            )

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

            inputs_permuted = keras.ops.transpose(inputs, perm_order)

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
        permuted_shape = keras.ops.shape(inputs_permuted)

        # Determine K (the feature dimension size)
        # We prefer the static shape if available for compile-time optimization
        if input_shape[axis] is not None:
            k = int(input_shape[axis])
        else:
            k = permuted_shape[-1]

        # Reshape to (-1, K)
        # -1 infers the total batch size dynamically
        inputs_2d = keras.ops.reshape(inputs_permuted, (-1, k))

        # =====================================================================
        # CORE ALGORITHM: Sparsemax Logic
        # =====================================================================

        # 0. Shift by the row max, then WIDEN the reduction.
        # Sparsemax is shift-invariant, so subtracting the row max is EXACT and
        # moves the final cancellation from scale |z| down to the row spread.
        # The reduction below then runs in `reduction_dtype`, which is what the
        # ramp / cumsum / support test / k_z count actually need.
        row_max = keras.ops.max(inputs_2d, axis=-1, keepdims=True)
        shifted = inputs_2d - row_max

        # DECISION plan-2026-07-29T070705-9bfc04c5/D-007
        # ---------------------------------------------------------------
        # The reduction dtype WIDENS; it must never NARROW. Only float16 and
        # bfloat16 lack the range and integer precision the reduction needs,
        # so only they are promoted to float32. float32 is unchanged (the
        # expression is a no-op there) and float64 KEEPS float64.
        # DO NOT hard-code "float32" here. That was tried: it silently
        # narrowed the float64 policy, moving corpus worst-case error from
        # 1.31e-15 to 1.99e-08, with every test still green. Degrading the
        # most precise policy without an alarm is the same species of defect
        # this reduction was widened to remove.
        # Spelled off `inputs.dtype` (the bits actually received), NOT
        # `self.compute_dtype`, and normalised through
        # `standardize_dtype` so the membership test cannot silently miss a
        # backend dtype object and re-narrow float64.
        # ---------------------------------------------------------------
        input_dtype = keras.backend.standardize_dtype(inputs.dtype)
        reduction_dtype = (
            "float32" if input_dtype in ("float16", "bfloat16") else input_dtype
        )
        shifted_f32 = keras.ops.cast(shifted, reduction_dtype)

        # 1. Sort logits (descending)
        # Necessary to find the "elbow" where probabilities drop to zero.
        sorted_logits = keras.ops.sort(shifted_f32, axis=-1)
        sorted_logits = keras.ops.flip(sorted_logits, axis=-1)

        # 2. Cumulative Sum
        # Used to check the condition: 1 + k * z_k > sum(z_1..z_k)
        z_cumsum = keras.ops.cumsum(sorted_logits, axis=-1)

        # 3. Create range vector [1, 2, ..., K]
        k_values = keras.ops.arange(1, k + 1, dtype=reduction_dtype)

        # Explicit Reshape: (K,) -> (1, K)
        # "Attempt": Just use k_values.
        # "Fix": Reshape to (1, K) so XLA sees explicit Rank-2 broadcasting.
        k_values = keras.ops.reshape(k_values, (1, k))

        # 4. Support Identification
        # Calculate the condition for every element.
        # support > 0 means that element is part of the active set.
        support = 1.0 + k_values * sorted_logits - z_cumsum
        support_mask = keras.ops.cast(support > 0, reduction_dtype)

        # k_z is the count of active elements (the "support size")
        # Shape: (N, 1) - One value per sample in the flattened batch
        k_z = keras.ops.sum(support_mask, axis=-1, keepdims=True)

        # =====================================================================
        # DECISION 3: One-Hot Selection vs. Index Gathering
        # =====================================================================
        # This block has TWO properties. Both are load-bearing; neither may be
        # dropped in favour of the other.
        #
        # (1) XLA-SAFE (the original reason, unchanged).
        #     "Attempt": `ops.take_along_axis(z_cumsum, k_z - 1, axis=-1)`
        #     "Problem": XLA treats `k_z - 1` as a dynamic index. Slicing with
        #                dynamic indices requires the compiler to support
        #                dynamic memory access patterns, which often fails
        #                graph fusion.
        #     "Fix": Use One-Hot Encoding + selection over the last axis. This
        #            converts an Index lookup (Gather) into shape-static math,
        #            which is universally supported by all accelerators.
        #
        # (2) NON-FINITE-SAFE (new; see the D-017 anchor below).
        #     The selection is spelled with `ops.where`, NOT with the
        #     `z_cumsum * gather_mask` product it replaced. A product forms
        #     `-inf * 0.0 = NaN` at every masked position, and the reduction
        #     then propagates that NaN into `tau` and hence into EVERY element
        #     of the row. `ops.where` never evaluates the non-selected operand
        #     arithmetically, so no such product is ever formed.

        # Cast k_z to int32 for indexing/one-hot operations
        support_indices = keras.ops.cast(k_z - 1, "int32")

        # Reshape to 1D to satisfy one_hot requirements
        support_indices = keras.ops.reshape(support_indices, (-1,))

        # Create One-Hot Mask: (N, K)
        # Only the position corresponding to k(z) will be 1.0, others 0.0
        gather_mask = keras.ops.one_hot(support_indices, k, dtype=reduction_dtype)

        # Select the cumulative sum at the threshold boundary: keep `z_cumsum`
        # where the one-hot is set, substitute an exact zero everywhere else,
        # then collapse the row to that single value.
        z_cumsum_at_k = keras.ops.sum(
            keras.ops.where(gather_mask > 0, z_cumsum, keras.ops.zeros_like(z_cumsum)),
            axis=-1, keepdims=True,
        )

        # =====================================================================
        # Final Projection
        # =====================================================================

        # Calculate Tau (Threshold)
        # tau = (sum(z_support) - 1) / |support|
        # `tau` is the ONLY value cast back: it is (N, 1), and casting it here
        # keeps the observable output dtype equal to the compute dtype, so
        # `compute_output_shape` stays truthful and no `compute_output_spec`
        # override is needed.
        tau = keras.ops.cast((z_cumsum_at_k - 1.0) / k_z, inputs.dtype)

        # Projection: P = max(0, z - tau)
        # This naturally sets elements outside the support to exactly zero.
        # MUST read `shifted`, NOT `inputs_2d`: `tau` was derived from the
        # SHIFTED row, and it is this line's cancellation being at scale
        # `spread` rather than `max|z|` that closes the D-017(d) plateau route.
        output_2d = keras.ops.maximum(shifted - tau, 0.0)

        # =====================================================================
        # DECISION 4: Restore Structure
        # =====================================================================
        # Un-flatten and un-permute to return a tensor indistinguishable
        # from the input structure.

        # Reshape back to permuted shape (e.g., [Batch, Seq, Heads, K])
        output_permuted = keras.ops.reshape(output_2d, permuted_shape)

        # Transpose back if we changed the axis order
        if inv_perm_order is not None:
            output = keras.ops.transpose(output_permuted, inv_perm_order)
        else:
            output = output_permuted

        return output

    def compute_output_shape(
            self,
            input_shape: tuple
    ) -> tuple:
        """Compute output shape (same as input shape).

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        :return: Output shape tuple, identical to input.
        :rtype: tuple
        :raises ValueError: If ``axis`` is out of range for ``input_shape``'s
            rank. The range is the same one :meth:`call` enforces, and the two
            MUST agree — see the DECISION anchor below.
        """
        # ---------------------------------------------------------------
        # DECISION plan-2026-07-29T110112-09832856/D-022
        # THIS CHECK IS NOT DEFENSIVE PROGRAMMING; DO NOT DELETE IT AS SUCH.
        # `call()` rejects an out-of-range `axis` (D-014). Before this check
        # existed, `compute_output_shape` did NOT, so the two disagreed: a
        # functional/symbolic build reached `compute_output_shape`, was told
        # the output shape was the input shape, and wired a graph that could
        # only fail later at call time. Worse, for `axis == -(ndim+1)` the
        # declared shape was a LIE about what the pre-D-014 layer actually
        # emitted (measured: `(2,4,5)` declared, `(2,5,4)` produced).
        # A shape function that accepts a configuration `call()` rejects
        # moves the error from build time to run time, which is the wrong
        # direction. The two predicates must stay identical; if the one in
        # `call()` changes, change this one in the same edit.
        # Guarded by `TestSparsemax::test_compute_output_shape_rejects_the_
        # same_axes_as_call` (it asserts the two predicates AGREE on a swept
        # band, not merely that each raises somewhere).
        # ---------------------------------------------------------------
        ndim = len(input_shape)
        if not -ndim <= self.axis < ndim:
            raise ValueError(
                f"axis={self.axis} is out of range for an input of rank "
                f"{ndim} (shape {tuple(input_shape)}); axis must be in "
                f"[{-ndim}, {ndim - 1}]"
            )
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
        })
        return config

# ---------------------------------------------------------------------
