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

        # 0. Shift by the row max, then WIDEN the reduction.
        # Sparsemax is shift-invariant, so subtracting the row max is EXACT and
        # moves the final cancellation from scale |z| down to the row spread.
        # The reduction below then runs in `reduction_dtype`, which is what the
        # ramp / cumsum / support test / k_z count actually need.
        row_max = ops.max(inputs_2d, axis=-1, keepdims=True)
        shifted = inputs_2d - row_max

        # DECISION plan-2026-07-29-9bfc04c5/D-007
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
        shifted_f32 = ops.cast(shifted, reduction_dtype)

        # 1. Sort logits (descending)
        # Necessary to find the "elbow" where probabilities drop to zero.
        sorted_logits = ops.sort(shifted_f32, axis=-1)
        sorted_logits = ops.flip(sorted_logits, axis=-1)

        # 2. Cumulative Sum
        # Used to check the condition: 1 + k * z_k > sum(z_1..z_k)
        z_cumsum = ops.cumsum(sorted_logits, axis=-1)

        # 3. Create range vector [1, 2, ..., K]
        k_values = ops.arange(1, k + 1, dtype=reduction_dtype)

        # Explicit Reshape: (K,) -> (1, K)
        # "Attempt": Just use k_values.
        # "Fix": Reshape to (1, K) so XLA sees explicit Rank-2 broadcasting.
        k_values = ops.reshape(k_values, (1, k))

        # 4. Support Identification
        # Calculate the condition for every element.
        # support > 0 means that element is part of the active set.
        support = 1.0 + k_values * sorted_logits - z_cumsum
        support_mask = ops.cast(support > 0, reduction_dtype)

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
        gather_mask = ops.one_hot(support_indices, k, dtype=reduction_dtype)

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
        #     A SECOND FAMILY, found only by adversarial review, is fixed by
        #     the same line: rows whose fp16 CUMSUM overflows while their
        #     answer is unaffected, because `k_z` is selected long before the
        #     overflow. Pre-plan these were ALL-NaN (the same `-inf * 0.0`
        #     product, with the `-inf` born from overflow rather than from the
        #     input); they are now exactly correct. Measured under
        #     `mixed_float16`, K=256, `err = 0.0` vs a float64 oracle:
        #       - `z = [400, 300 x255]`   — no mask anywhere in the input
        #                                   (`300 * 256 = 76800 > 65504`);
        #       - `z = [2, 1, -1e4 x254]` — the LARGE-FINITE-NEGATIVE mask
        #                                   convention, which is what an
        #                                   attention mask bias degrades to
        #                                   once it is cast to fp16.
        #     Both were `nan=256/256, sum=0` before this line and are
        #     `[1, 0, ...], sum = 1.0` after. Guarded by
        #     `::test_finite_cumsum_overflow_rows_are_correct_not_nan`.
        #
        # (b) FOUR FURTHER DEFECTS — B, C, D and E — ALL **CLOSED** by
        #     plan-2026-07-29-9bfc04c5. They stay on the record with the
        #     magnitudes at which they were first measured, because those
        #     magnitudes are what made them findable and are what any
        #     regression would have to reproduce. Do NOT read them as open.
        #     All four shared ONE root cause: the reduction (ramp, cumsum,
        #     support test, `k_z` count) ran in the COMPUTE dtype, which under
        #     fp16/bf16 has neither the range nor the integer precision the
        #     algorithm needs, and which under float32 gives out near 1.7e7.
        #     Three mechanisms, all visible above in `call()`, closed them:
        #       M1 — subtract the ROW MAX first (`shifted`). Exact: sparsemax
        #            is shift-invariant. It moves every later cancellation from
        #            scale `max|z|` down to the row SPREAD.
        #       M2 — build the `arange` ramp in the reduction dtype rather than
        #            in the compute dtype.
        #       M3 — run sort / cumsum / support / `k_z` / one-hot in a
        #            reduction dtype that WIDENS but never NARROWS. That rule,
        #            and the measured cost of getting it wrong, live in the
        #            `plan-2026-07-29-9bfc04c5/D-007` anchor above; they are
        #            deliberately not restated here, and nothing in this clause
        #            overrides them.
        #       - Defect B (line ~220 pre-fix): an overflow-born non-finite
        #         `z_cumsum` was ADMITTED to the support, because
        #         `support = 1 + finite - (-inf) = +inf > 0`. Measured at
        #         spread 16.95, fp16, K=4096, with NO `-inf` in the input:
        #         `k_z = 1863` where the exact answer is 1; `sum(out) = 16.95`.
        #         CLOSED by M1 + M3: after the shift the cumsum is bounded by
        #         `K * spread` and is accumulated in float32, so it no longer
        #         overflows on this row. Post-fix on that exact input:
        #         `maxerr = 0` against the exact-rational oracle.
        #       - Defect C (line ~210 pre-fix):
        #         `ops.arange(1, k+1, dtype=inputs.dtype)` could not represent
        #         `k+1`, so the reshape RAISED. The break was NON-MONOTONE in
        #         K: it raised at fp16 K=2048 and bf16 K=256/257, but not at
        #         fp16 K=4096 or bf16 K=512. Do not assume a threshold.
        #         CLOSED by M2: the bf16 K=256 row now builds and runs.
        #       - Defect D (line ~220 pre-fix): round-off absorbed the literal
        #         `1.0`, so `support == 0` everywhere, `k_z == 0`,
        #         `one_hot(-1)` was all-zero and `tau = -inf`. On an ALL-FINITE
        #         row the output was then all `+inf`; on a row that ALSO
        #         carried an `-inf` mask the masked slots computed
        #         `-inf - (-inf)` and the row returned NaN. Measured onsets
        #         (K=4, 2 of 4 masked for the NaN route): float32 |z| >= 1.68e7
        #         (7.77e7 at K=512), fp16 |z| >= 2048, bf16 |z| >= 300.
        #         CLOSED by M1 (with M3): once the row sits at scale `spread`
        #         the literal `1.0` is no longer absorbed. Post-fix at all
        #         three onsets, in both the all-finite and the masked form:
        #         `maxerr = 0`.
        #       - Defect E (line ~225 pre-fix): `k_z = ops.sum(support_mask)`
        #         accumulated in the compute dtype and hit the same integer
        #         wall. Measured under fp16 on the TF/GPU tree reduction:
        #         2049 -> 2048, 2051 -> 2052, 4095 -> 4096 (2050 / 3000 / 4094
        #         were exact). The 2051 -> 2052 overshoot selected a MASKED
        #         position whose `z_cumsum` was `-inf`, so `tau = -inf` and the
        #         row died (measured: nan=2045, inf=2051 at K=4096).
        #         CLOSED by M3: the count is exact in float32 far beyond
        #         K=4096. Post-fix on that row: `maxerr = 1.05e-09`.
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
        #     XLA, STATED PRECISELY — an earlier blanket claim that fp16 and
        #     bf16 "do not compile at all" was too strong in exactly this
        #     direction. Re-measured on scratch-reverted bytes over
        #     K in {8, 256, 257, 512, 2048, 4096}: **fp16 failed at EVERY K**,
        #     and failed at XLA LOWERING (`ValueError: Unsupported dtype19 ...
        #     Range[Tidx=DT_HALF]`); **bf16 failed only at K=256/257**, and
        #     failed EAGERLY on `ops.reshape` ("Input to reshape is a tensor
        #     with 255 values, but the requested shape has 256") — i.e. bf16's
        #     compile failure merely TRACKED its eager Defect-C raise rather
        #     than being an independent XLA limitation. All 24 (K x dtype) grid
        #     points now compile under `tf.function(jit_compile=True)` and
        #     match eager, guarded by `::TestSparsemaxXLACapability`.
        #
        # (c) WHY THE `ops.where` SPELLING ALONE COULD NOT CLOSE D OR E
        #     (historical, and still true — do not expect this line to carry
        #     that load). In both, the `-inf` reached `tau` ITSELF — for E it
        #     sat at the SELECTED index, for D it was manufactured by
        #     `k_z == 0` — rather than at a masked-out operand of this
        #     selection, so `ops.where` could not exclude it and `tau` was
        #     `-inf` regardless of how the selection was spelled. What closed
        #     them was M1 and M3 above, not a re-spelling of this expression.
        #     This line remains load-bearing for Defect A; see (a).
        #
        # (d) OUTPUT PRECISION under fp16 WAS ~1 ulp of `max|z|` (measured
        #     worst case 1.685 ulp over the committed property grid), because
        #     `out = max(z - tau, 0)` is a cancellation — at scale `max|z|`
        #     before M1, at scale `spread` after it. The tests' oracle
        #     tolerance is derived from that resolution rather than fixed.
        #     AT LARGE K THAT PER-ELEMENT LIMIT BECAME A ROW-SUM DEFECT.
        #     Measured under `mixed_float16` on plateau rows, with NO overflow,
        #     NO non-finite value and no alarm of any kind:
        #       K=512,  256 entries at |z| = 20 -> `sum(out) = 4.000`;
        #       K=512,  257 entries            -> 4.016;
        #       K=1024, 1023 entries           -> 15.984;
        #       K=1024, 259 entries            -> 4.047   (correct sum: 1.0).
        #     Mechanism: the correct per-entry mass `1/256 = 0.0039` is NOT
        #     REPRESENTABLE at scale 20 (`ulp_fp16(20) = 0.0156`), so the
        #     nearest non-zero result was one whole ulp; the per-element error
        #     was 0.0117 = 0.75 ulp, i.e. SUB-ULP and as good as fp16 can do,
        #     but K of them added up. It never got a letter — it was this
        #     clause's cancellation limit, not a separate defect, and it
        #     measured IDENTICAL in the pre-plan bytes and in the M2-only
        #     bytes, so no spelling of the selection above affected it.
        #     **CLOSED by M1**, which is the only thing that could close it:
        #     the shifted row's ulp is ~2e-6, not 0.0156. This is why the
        #     projection below MUST read `shifted` and not `inputs_2d` —
        #     reading `inputs_2d` there silently reopens this clause, with no
        #     non-finite value and no raise to warn you.
        #
        # (e) THE ACCEPTED LOUD -> SILENT CONVERSION NO LONGER EXISTS, BUT ITS
        #     PROHIBITION IS STILL IN FORCE — it is the most important thing in
        #     this anchor. Historically, on an all-finite Defect-B row whose
        #     cumsum OVERFLOWED, the pre-Defect-A code returned NaN by accident
        #     (the `-inf * 0.0` product at the NON-SELECTED positions
        #     manufactured one); removing that product left the row FINITE and
        #     UNNORMALISED (`mixed_float16`, K=4096, `full(-16.95)` with
        #     `z[0] = 0.0`: `nan=4096` before, `nan=0, sum(out) = 16.938`
        #     after, correct sum 1.0), i.e. wrong both times but silently so.
        #     M1 + M3 removed the premise entirely: that row is now simply
        #     CORRECT. It is pinned by `TestSparsemaxClosedDefects`
        #     `::test_defect_b_closed_max_entry_takes_all_mass_on_the_spread_row`.
        #
        #     YOU MUST STILL NOT RE-INVENT THE "LOUDNESS GUARD". A guard that
        #     forces `tau` to NaN when the INPUT is all finite yet `z_cumsum` is
        #     not was written, shipped, measured and then REVERTED, because it
        #     DESTROYED exactly-correct answers on two ORDINARY input families —
        #     the very ones clause (a) records as its second win:
        #       - `z = [400, 300 x255]`   — no mask anywhere in the input;
        #       - `z = [2, 1, -1e4 x254]` — the large-finite-negative mask
        #                                   convention.
        #     Both were ALL-NaN under the guard; both re-measure at `maxerr = 0`
        #     against the exact oracle with the current code. The guard's
        #     premise — "a legitimately masked row HAS `-inf` in its input, so
        #     it is never poisoned" — is simply false; legitimacy does not
        #     require `-inf`.
        #     A CUMSUM-FINITENESS PREDICATE CANNOT DISTINGUISH THESE CASES.
        #     Overflowing the cumsum is neither necessary nor sufficient for a
        #     wrong answer: in both families `k_z` is selected long before the
        #     overflow happens, so the overflow is irrelevant to the result. Any
        #     predicate on `z_cumsum` therefore trades a wrong-but-loud row for
        #     a correct-but-destroyed one, on commoner shapes.
        #
        # (f) DELIBERATE SCOPE-OUTS AND MEASURED LIMITS (plan-2026-07-29-9bfc04c5).
        #     Recorded so a future reader sees choices, not omissions.
        #
        #     THE OUTPUT-SIDE VALIDITY PREDICATE (`|sum(out) - 1| > tol`) WAS
        #     DROPPED FROM SCOPE, not forgotten. An earlier revision of clause
        #     (e) promised it to this follow-up as "the ACCURATE predicate".
        #     The evidence for calling it necessary is gone: every family that
        #     appeared to require it — Defect B, both Defect-D forms, Defect E,
        #     and clause (d)'s plateau rows, under all four dtype policies — is
        #     correct without it, and the one apparent survivor turned out to be
        #     an oracle-input artifact (the oracle had been fed the intended
        #     Python literals instead of the bits the layer actually received).
        #     Do not add it speculatively; if you ever construct a row that is
        #     finite, plausible and un-normalised, that is new evidence and the
        #     decision can be revisited on it.
        #
        #     THE CAST-BACK POINT. `tau` — an `(N, 1)` tensor — is the ONLY
        #     value cast back to the compute dtype, at the `ops.cast` below.
        #     The layer's observable output dtype therefore still equals its
        #     compute dtype, `compute_output_shape` stays truthful, and no
        #     `compute_output_spec` override is needed. Do NOT "improve" this by
        #     returning float32 under an fp16 policy: that changes the dtype
        #     contract for every consumer, and M1 already removed the precision
        #     argument for doing so (clause (d)).
        #     CONSUMER CONSTRAINT: `losses/sparsemax_loss.py` does not currently
        #     accommodate that compute-dtype output contract under a non-default
        #     policy. That is a defect in the LOSS and its fix belongs in the
        #     loss's own file — it is NOT a reason to make this layer return
        #     float32.
        #
        #     FULLY-MASKED ROWS (`all(z[i]) == -inf`) RETURN ALL-NaN, under all
        #     four dtype policies, IDENTICALLY before and after this plan:
        #     `row_max = -inf`, `shifted = -inf - (-inf) = NaN`, and that NaN
        #     reaches `tau` and hence every element. Measured, not assumed —
        #     neither fixed nor worsened here, and out of scope: upstream
        #     `apply_attention_mask(rescue_axis=...)` rescues such rows before
        #     they reach this layer. A row with a SINGLE finite entry is exact
        #     (`out` is that one-hot, `sum = 1.0`), so the degeneracy is
        #     strictly the all-masked case.
        #
        #     COST. The widened reduction costs ~12% eager wall-clock and 2x
        #     memory in the reduction intermediates; the XLA capability grid
        #     adds ~5-11s to the activations test gate.
        #
        #     TOLERANCE / DUPLICATION. The test file's `_oracle_atol` derivation
        #     re-states D-007's reduction-dtype rule on purpose (an oracle must
        #     not share code with the thing it checks). A future change to
        #     `reduction_dtype` must move that copy in LOCKSTEP. Note also that
        #     the guard against re-narrowing float64 is THIS ANCHOR plus D-007's
        #     recorded measurement — NOT the tolerance test: the narrowed
        #     float64 error (1.99e-08) sits far under the 1e-3 TF32 floor those
        #     assertions carry, so they would stay green through the regression.
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
        # `tau` is the ONLY value cast back: it is (N, 1), and casting it here
        # keeps the observable output dtype equal to the compute dtype, so
        # `compute_output_shape` stays truthful and no `compute_output_spec`
        # override is needed.
        tau = ops.cast((z_cumsum_at_k - 1.0) / k_z, inputs.dtype)

        # Projection: P = max(0, z - tau)
        # This naturally sets elements outside the support to exactly zero.
        # MUST read `shifted`, NOT `inputs_2d`: `tau` was derived from the
        # SHIFTED row, and it is this line's cancellation being at scale
        # `spread` rather than `max|z|` that closes the D-017(d) plateau route.
        output_2d = ops.maximum(shifted - tau, 0.0)

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
