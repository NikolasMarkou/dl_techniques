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
        boolean masking ops.
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
from keras import ops
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
        Typically -1. Defaults to -1.
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
        :raises ValueError: If axis is not an integer.
        """
        super().__init__(**kwargs)
        if not isinstance(axis, int):
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
        support_indices = ops.cast(k_z - 1, "int32")

        # Reshape to 1D to satisfy one_hot requirements
        support_indices = ops.reshape(support_indices, (-1,))

        # Create One-Hot Mask: (N, K)
        # Only the position corresponding to k(z) will be 1.0, others 0.0
        gather_mask = ops.one_hot(support_indices, k, dtype=inputs.dtype)

        # DECISION plan-2026-07-28T134123-420f6ccb/D-017
        # ---------------------------------------------------------------
        # (a) WHAT THIS LINE FIXES — "Defect A", the reported bug.
        #     DO NOT rewrite this back to `ops.sum(z_cumsum * gather_mask, ...)`.
        #     An input `-inf` (an attention mask) puts `-inf` into `z_cumsum`
        #     from the first masked entry onward; `gather_mask` is 0 there;
        #     `-inf * 0.0 = NaN`; the reduction returns NaN; `tau` is NaN; and
        #     the final projection makes EVERY element of the row NaN,
        #     including the unmasked ones. Measured: 128/128 NaN through
        #     `MultiHeadCrossAttention(probability_type='sparsemax')`. This is
        #     NOT fp16-specific — it reproduces under the plain float32 policy
        #     at K=256/2048/4096. Reverting this line turns RED:
        #     `tests/test_layers/test_activations/test_sparsemax.py::TestSparsemax`
        #     `::test_partial_mask_neg_inf_no_nan` (all 6 parametrizations),
        #     `::test_sparsemax_property_random_masks` (all 3 policies) and
        #     `::test_attention_integration_sparsemax_fp16_no_nan`.
        #     This is the same `ops.where`-over-arithmetic-masking rule that
        #     `layers/attention/common.py:159-163` (D-002) already prescribes.
        #
        # (b) WHAT IS STILL BROKEN — OPEN, UNGUARDED, DEFERRED to a follow-up
        #     plan. This layer is NOT numerically safe. Four further defects
        #     were MEASURED and are NOT fixed here; a green test file does not
        #     mean otherwise. All four share one root cause: the reduction
        #     (ramp, cumsum, support test, `k_z` count) runs in the COMPUTE
        #     dtype, which under fp16/bf16 has neither the range nor the
        #     integer precision the algorithm needs.
        #       - Defect B (line ~220): an overflow-born non-finite `z_cumsum`
        #         is ADMITTED to the support, because
        #         `support = 1 + finite - (-inf) = +inf > 0`. Measured at
        #         spread 16.95, fp16, K=4096, with NO `-inf` in the input:
        #         `k_z = 1863` where the exact answer is 1; `sum(out) = 16.95`.
        #       - Defect C (line ~210): `ops.arange(1, k+1, dtype=inputs.dtype)`
        #         cannot represent `k+1`, so the reshape raises. The break is
        #         NON-MONOTONE: it raises at fp16 K=2048 and bf16 K=256/257,
        #         but is fine at fp16 K=4096 and bf16 K=512. Do not assume a
        #         threshold.
        #       - Defect D (line ~220): round-off absorbs the literal `1.0`, so
        #         `support == 0` everywhere, `k_z == 0`, `one_hot(-1)` is
        #         all-zero and `tau = -inf`. On an ALL-FINITE row the output is
        #         then all `+inf`; on a row that ALSO carries an `-inf` mask the
        #         masked slots compute `-inf - (-inf)` and the row returns NaN.
        #         Measured onsets (K=4, 2 of 4 masked for the NaN route):
        #         float32 |z| >= 1.68e7 (7.77e7 at K=512), fp16 |z| >= 2048,
        #         bf16 |z| >= 300.
        #       - Defect E (line ~225): `k_z = ops.sum(support_mask)` accumulates
        #         in the compute dtype and hits the same integer wall. Measured
        #         under fp16 on the TF/GPU tree reduction: 2049 -> 2048,
        #         2051 -> 2052, 4095 -> 4096 (2050 / 3000 / 4094 are exact).
        #         The 2051 -> 2052 overshoot selects a MASKED position whose
        #         `z_cumsum` is `-inf`, so `tau = -inf` and the row dies
        #         (measured: nan=2045, inf=2051 at K=4096).
        #         An earlier revision of this anchor also claimed that the
        #         4095 -> 4096 overshoot indexes OUT OF RANGE for depth 4096 and
        #         yields a silently wrong finite answer. That claim was FALSE and
        #         has been DELETED: 4095 is a valid index into depth 4096, and no
        #         end-to-end input reaching an out-of-range one-hot could be
        #         constructed (every K where the index would truly overrun raises
        #         Defect C first). Do not reinstate it without a repro.
        #     Every onset quoted above is reduction-order dependent: they were
        #     measured on the TensorFlow GPU tree reduction, and a sequential
        #     accumulation moves them. Treat a non-reproduction on another
        #     backend as a different measurement, not as absence of the defect.
        #
        # (c) THIS LINE CANNOT HELP DEFECT E OR DEFECT D. In both, the `-inf`
        #     reaches `tau` ITSELF — for E it sits at the SELECTED index, for D
        #     it is manufactured by `k_z == 0` — rather than at a masked-out
        #     operand of this selection, so `ops.where` cannot exclude it and
        #     `tau` is `-inf` regardless of how the selection is spelled. Fixing
        #     E and D requires widening the reduction dtype (and, for D,
        #     subtracting the row max), not re-spelling this expression.
        #
        # (d) OUTPUT PRECISION under fp16 is ~1 ulp of `max|z|` (measured worst
        #     case 1.685 ulp over the committed property grid), because
        #     `out = max(z - tau, 0)` is a cancellation at scale `|z|`. The
        #     tests' oracle tolerance is derived from that resolution rather
        #     than fixed. Subtracting the row max before the cumsum (exact:
        #     sparsemax is shift-invariant) takes the same grid from 86/128 to
        #     0/128 violations — deferred to the follow-up plan, not done here.
        #
        # (e) THE LOUDNESS GUARD (the three lines just before the projection,
        #     below). Replacing the arithmetic gather with `ops.where` above
        #     removed a `-inf * 0.0` product that, on Defect-B inputs, had been
        #     manufacturing a NaN which INCIDENTALLY MASKED B's wrong answer.
        #     Measured on `full((1, 4096), -16.95)` with `z[0, 0] = 0.0` (all
        #     finite, `mixed_float16`): before, `nan=4096`; after, `nan=0,
        #     sum=16.94` where the correct sum is 1.0 — a loud failure turned
        #     silent. The guard restores loudness:
        #       WHAT IT DOES: if a row's INPUT is entirely finite yet its
        #       `z_cumsum` contains a non-finite value, that row has OVERFLOWED
        #       the compute dtype, its `k_z` is meaningless, and its `tau` is
        #       forced to NaN so the whole row fails visibly. The predicate is
        #       the load-bearing part: a LEGITIMATELY masked row HAS `-inf` in
        #       its input, so it is never poisoned (measured: `[2, 1, -inf,
        #       -inf]` still gives `[1, 0, 0, 0]`, err 0.0, float32 and fp16).
        #       WHAT IT DOES NOT DO: it does not FIX any of B/C/D/E — the answer
        #       is still wrong, it is merely no longer quiet. It does not fire
        #       for Defect D (measured float32 K=4 at 1.68e7: the cumsum stays
        #       finite, so the output is `+inf` exactly as before the guard),
        #       nor for Defect C (which raises), nor for Defect E (whose input
        #       carries `-inf`). It is bit-identical on every well-behaved row
        #       (measured `max|diff| == 0.0` over 41 cases / 34,513 values in
        #       float32 and float64). Rows it DOES restore to loud: all-finite
        #       float32 `1e37` (K=512) and `3e38` (K=4), and float64 `1e307`
        #       (K=512) — all NaN again, matching pre-`ops.where` behaviour.
        # ---------------------------------------------------------------
        # Select the cumulative sum at the threshold boundary: keep `z_cumsum`
        # where the one-hot is set, substitute an exact zero everywhere else,
        # then collapse the row to that single value.
        z_cumsum_at_k = ops.sum(
            ops.where(gather_mask > 0, z_cumsum, ops.zeros_like(z_cumsum)),
            axis=-1, keepdims=True,
        )

        # =====================================================================
        # Final Projection
        # =====================================================================

        # Calculate Tau (Threshold)
        # tau = (sum(z_support) - 1) / |support|
        tau = (z_cumsum_at_k - 1.0) / k_z

        # LOUDNESS GUARD — see clause (e) of the D-017 anchor above.
        # A row whose INPUT is entirely finite but whose `z_cumsum` went
        # non-finite has OVERFLOWED the compute dtype: its `k_z` is wrong and
        # its answer is garbage, so it must fail LOUDLY (NaN) rather than
        # return a plausible finite number. A legitimately masked row HAS
        # `-inf` in its input, so it is never poisoned by this predicate.
        finite_in = ops.all(ops.isfinite(inputs_2d), axis=-1, keepdims=True)
        ovf = ops.any(ops.logical_not(ops.isfinite(z_cumsum)), axis=-1, keepdims=True)
        tau = ops.where(ops.logical_and(finite_in, ovf), ops.full_like(tau, float("nan")), tau)

        # Projection: P = max(0, z - tau)
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
        """Compute output shape (same as input shape).

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        :return: Output shape tuple, identical to input.
        :rtype: tuple
        """
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
