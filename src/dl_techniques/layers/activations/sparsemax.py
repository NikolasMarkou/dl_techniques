"""
Sparsemax: the Euclidean projection of logits onto the probability simplex.

Softmax gives every class a non-zero share. Sparsemax solves
``argmin_p ||p - z||^2`` over the simplex instead, which sets everything
outside a support set to exactly zero. The result still sums to 1, so it is a
drop-in replacement for softmax wherever you want an attention or class
distribution that can genuinely ignore options.

The support size is not a knob; it follows from how spread out the row is.
Measured over 5000 rows of 20 logits: drawn from N(0, 1) the mean support is
2.77 of 20 classes, so 17.23 outputs are exactly 0.0; scaled down to N(0,
0.01) the support is 20.0 of 20 and nothing is zero at all. On the single row
``[3, 1, 2, 0.5, -1]`` sparsemax returns ``[1, 0, 0, 0, 0]`` where softmax
returns ``[0.624, 0.084, 0.229, 0.051, 0.011]``. A near-uniform row gets a
near-uniform answer.

The implementation is shaped by XLA, which TensorFlow and JAX both compile
through. Three habits carry the weight, and each is marked in ``call``:
flatten to rank 2 before doing anything, because XLA fails to infer broadcast
shapes when a rank-1 support vector meets a rank-5 tensor; select with a
one-hot product rather than ``take_along_axis``, because a dynamic index
blocks graph fusion; and reshape support vectors explicitly to ``(1, K)``
rather than leaning on NumPy-style broadcasting.

References:
    - Martins & Astudillo, 2016. "From Softmax to Sparsemax: A Sparse
      Model of Attention and Multi-Label Classification".
      (https://arxiv.org/abs/1602.02068)
"""

import keras
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .common import axis_is_in_range, normalize_axis

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Sparsemax(keras.layers.Layer):
    """Projects logits onto the probability simplex, producing exact zeros.

    Each row along ``axis`` is mapped to the closest point on the simplex in
    Euclidean distance. Outputs are non-negative and sum to 1, and classes
    outside the support come out as exactly 0.0. Output shape and dtype equal
    the input's. The layer owns no weights.

    Internally it is a flatten-mask-restore pipeline: move ``axis`` to the
    end, collapse everything else into one batch dimension, do the projection
    on a rank-2 tensor, then undo both moves. That shape discipline is what
    keeps the layer XLA-compilable.

    **Architecture Overview:**

    .. code-block:: text

        z  logits  [..., K]   `axis` selects the K dimension
                      ▼
        ┌───────────────────────────┐
        │ transpose axis -> last    │  (optional)
        └─────────────┬─────────────┘
                      ▼  [..., K]
        ┌───────────────────────────┐
        │ reshape to (N, K)         │
        └─────────────┬─────────────┘
                      ▼  [N, K]
        ┌───────────────────────────┐
        │ s = z - rowmax(z)         │
        │ cast to reduction dtype   │
        └─────────────┬─────────────┘
                      ▼
        ┌───────────────────────────┐
        │ sort desc, cumsum         │
        │ k_z = support size        │
        │ tau = (cum[k_z] - 1)/k_z  │
        └─────────────┬─────────────┘
                      ▼  s [N, K], tau [N, 1]
        ┌───────────────────────────┐
        │ p = max(s - tau, 0)       │
        └─────────────┬─────────────┘
                      ▼  [N, K]
        ┌───────────────────────────┐
        │ reshape back              │
        │ inverse transpose (opt.)  │
        └─────────────┬─────────────┘
                      ▼
        p  [..., K]   sums to 1, zeros outside the support

    ``K`` is the size of ``axis`` and ``N`` is the product of every other
    dimension. The transpose pair only runs when ``axis`` is not already the
    last dimension. Subtracting the row max is exact, because sparsemax is
    shift-invariant, and it moves the cancellation in ``s - tau`` down from
    the scale of ``max|z|`` to the scale of the row's spread.

    :param axis: Axis to project along. Defaults to -1. The valid range,
        ``[-ndim, ndim - 1]``, depends on the rank of the tensor the layer is
        called on, so it can only be checked at call time. ``__init__``
        checks the type; ``call`` and ``compute_output_shape`` check the
        range, with identical predicates.
    :type axis: int
    :param kwargs: Additional keyword arguments passed to the Layer base
        class.

    :raises ValueError: If ``axis`` is not an ``int``, or is a ``bool``.
        Raised from ``__init__``.

    Note:
        The output dtype follows the layer's compute dtype, not the dtype of
        the array you pass in, because Keras autocasts inputs first. Measured
        on the same float32 array: ``Sparsemax(dtype="float64")`` returns
        float64, ``Sparsemax(dtype="mixed_float16")`` returns float16. The
        reduction that finds the support runs wider than float16 either way;
        see the D-007 anchor in ``call``.
    """

    def __init__(
            self,
            axis: int = -1,
            **kwargs: Any
    ) -> None:
        """Validate the type of ``axis`` and store it.

        :param axis: Axis to project along.
        :type axis: int
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If ``axis`` is not an integer, or is a bool. The
            RANGE of ``axis`` depends on the input rank and is therefore
            validated in :meth:`call`, not here.
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
        """Project each row along ``axis`` onto the probability simplex.

        :param inputs: Logits, any shape and rank.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Non-negative tensor of the same shape as ``inputs``, summing
            to 1 along ``axis``, with exact zeros outside the support.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``axis`` is out of range for the rank of
            ``inputs``, i.e. outside ``[-ndim, ndim - 1]``.
        """
        # Store original shape for restoration
        input_shape = inputs.shape
        ndim = len(input_shape)

        # DECISION plan-2026-07-29T110112-09832856/D-014
        # Range-check `axis` before any list.pop / ops.transpose. Do NOT move it to
        # __init__ (rank unknown there) or delete it as defensive. Measured on the
        # pre-fix bytes, writing `norm = ndim + axis`: list.pop accepts negatives, so
        # `norm == -1` returned right numbers in a wrong layout ((2,4,5) in, (2,5,4)
        # out) and `norm in [-ndim, -2]` aborted the process (SIGABRT, exit 134,
        # uncatchable). Predicate = `common.axis_is_in_range`, SHARED with D-022.
        # Guard: TestSparsemax::test_out_of_range_axis_raises_value_error. See D-014.
        if not axis_is_in_range(self.axis, ndim):
            raise ValueError(
                f"axis={self.axis} is out of range for an input of rank "
                f"{ndim} (shape {tuple(input_shape)}); axis must be in "
                f"[{-ndim}, {ndim - 1}]"
            )

        # Normalize axis to positive index (e.g., -1 -> 2 for rank 3)
        axis = normalize_axis(self.axis, ndim)

        # Step 1: move the target axis to the end.
        # `sort` and `cumsum` want the last, contiguous axis. If `axis` is not
        # already last, transpose it there and keep the inverse permutation so
        # step 4 can undo it.
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

        # Step 2: flatten to 2D.
        # XLA cannot reliably broadcast a computed 1-D support vector against a
        # rank-5 tensor. Collapsing every leading dimension into one N makes
        # each op an (N, K) against (1, K) or (N, 1) pairing, so the graph
        # topology is static.

        # Symbolic shape, so a dynamic (None) batch size still works.
        permuted_shape = keras.ops.shape(inputs_permuted)

        # Prefer the static K when it is known; the compiler can use it.
        if input_shape[axis] is not None:
            k = int(input_shape[axis])
        else:
            k = permuted_shape[-1]

        # -1 infers the collapsed batch size at run time.
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
        # The reduction dtype only WIDENS: float16/bfloat16 -> float32, all else
        # unchanged, read off `inputs.dtype` (not `self.compute_dtype`). Do NOT
        # hard-code "float32": that silently narrowed float64 and moved worst-case
        # error 1.31e-15 -> 1.99e-08 with every test green. See decisions.md D-007.
        input_dtype = keras.backend.standardize_dtype(inputs.dtype)
        reduction_dtype = (
            "float32" if input_dtype in ("float16", "bfloat16") else input_dtype
        )
        shifted_f32 = keras.ops.cast(shifted, reduction_dtype)

        # 1. Sort descending, to find the elbow where the tail drops to zero.
        sorted_logits = keras.ops.sort(shifted_f32, axis=-1)
        sorted_logits = keras.ops.flip(sorted_logits, axis=-1)

        # 2. Running total, for the support test 1 + k*z_k > sum(z_1..z_k).
        z_cumsum = keras.ops.cumsum(sorted_logits, axis=-1)

        # 3. Range vector [1, 2, ..., K], reshaped to (1, K) so XLA sees an
        # explicit rank-2 broadcast rather than an implicit NumPy one.
        k_values = keras.ops.arange(1, k + 1, dtype=reduction_dtype)

        k_values = keras.ops.reshape(k_values, (1, k))

        # 4. Support test, elementwise. A positive value means the element is
        # in the active set.
        support = 1.0 + k_values * sorted_logits - z_cumsum
        support_mask = keras.ops.cast(support > 0, reduction_dtype)

        # k_z is the support size, one value per flattened row: (N, 1).
        k_z = keras.ops.sum(support_mask, axis=-1, keepdims=True)

        # Step 3: pick cumsum[k_z] with a one-hot, not a gather.
        # Do NOT use ops.take_along_axis: `k_z - 1` is a dynamic index, and
        # slicing with one defeats XLA graph fusion. Do NOT use the
        # `z_cumsum * gather_mask` product either: `-inf * 0.0` is NaN at every
        # masked position and the reduction spreads it across the whole row.
        # `ops.where` never evaluates the operand it does not select.

        # Cast k_z to int32 for indexing/one-hot operations
        support_indices = keras.ops.cast(k_z - 1, "int32")

        # Reshape to 1D to satisfy one_hot requirements
        support_indices = keras.ops.reshape(support_indices, (-1,))

        # One-hot mask (N, K): 1.0 only at the threshold index k(z).
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

        # The threshold tau = (sum over the support - 1) / |support|.
        # `tau` is the ONLY value cast back: it is (N, 1), and casting it here
        # keeps the observable output dtype equal to the compute dtype, so
        # `compute_output_shape` stays truthful and no `compute_output_spec`
        # override is needed.
        tau = keras.ops.cast((z_cumsum_at_k - 1.0) / k_z, inputs.dtype)

        # The projection itself. Everything below tau lands on exactly zero.
        # MUST read `shifted`, NOT `inputs_2d`: `tau` was derived from the
        # SHIFTED row, and it is this line's cancellation being at scale
        # `spread` rather than `max|z|` that closes the D-017(d) plateau route.
        output_2d = keras.ops.maximum(shifted - tau, 0.0)

        # Step 4: restore the original layout.

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
        """Return the input shape unchanged, after the same axis check.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: tuple
        :return: Output shape tuple, identical to input.
        :rtype: tuple
        :raises ValueError: If ``axis`` is out of range for ``input_shape``'s
            rank. The range is the same one :meth:`call` enforces, because both
            call ``common.axis_is_in_range`` -- see the DECISION anchor below.
        """
        # DECISION plan-2026-07-29T110112-09832856/D-022
        # Predicate = `common.axis_is_in_range`, the SAME function `call` uses
        # (D-014); do NOT re-type the comparison here, and do NOT delete the check
        # as defensive: without it a symbolic build was told the output shape was
        # the input shape, a lie for axis == -(ndim+1) ((2,4,5) declared, (2,5,4)
        # produced). D-022; TestSparsemax::test_compute_output_shape_rejects_*.
        ndim = len(input_shape)
        if not axis_is_in_range(self.axis, ndim):
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
