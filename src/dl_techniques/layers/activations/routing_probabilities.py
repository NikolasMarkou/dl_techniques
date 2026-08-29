"""
Hierarchical routing tree for classification, deterministic or trainable.

``RoutingProbabilitiesLayer`` turns ``D`` input features into ``N`` class
probabilities using ``d = log2(next_pow2(N))`` binary decisions instead of
``N`` output logits. ``mode`` picks what makes those decisions:

- ``mode="deterministic"`` (default): a fixed cosine-basis projection. No
  trainable parameters. A drop-in alternative to softmax that adds a
  hierarchical structural bias.
- ``mode="trainable"``: an affine projection ``x W + b``. A drop-in
  replacement for ``Dense -> Softmax`` that costs ``d`` decisions instead of
  ``N`` logits.

Both modes share one tree:

1. **Padding.** ``output_dim`` is rounded up to the next power of two,
   ``padded_dim``. The tree has ``d = log2(padded_dim)`` levels.

2. **Decision logits.** One logit ``z_k`` per level. Deterministic mode:
   ``z_k = <x, w_k>`` with ``w_{k,i} = cos(2*pi * (k+1) * i / D)``,
   L2-normalized per column. Trainable mode: ``z = x W + b`` with a learnable
   ``W`` of shape ``[D, d]``.

3. **Decisions.** ``p_k = sigmoid(z_k)``, clipped into
   ``[epsilon, 1 - epsilon]``, is the probability of going right at level
   ``k``.

4. **Routing.** Mass starts at 1.0 at the root and splits at every level:
   ``left = parent * (1 - p_k)``, ``right = parent * p_k``.

5. **Slice and renormalize.** The ``padded_dim`` leaf masses are cut to the
   first ``output_dim`` and divided by their sum.

Caveats:

- **Class index order matters.** Class ``j`` is the ``j``-th leaf, left to
  right. Neighbouring indices share a long path prefix; distant indices share
  none. In trainable mode ``W`` can permute classes to fit that topology. In
  deterministic mode it cannot, so the layer only helps when the class index
  space already carries structure. Language-model token IDs usually do not.

- **A non-power-of-two ``output_dim`` gives an unbalanced tree.** Structural
  masks force every discarded leaf to exactly zero mass before the slice, so
  the slice throws nothing away. Measured at ``output_dim=10``,
  ``padded_dim=16``: the six padded leaves are 0.0 exactly, and the row sum
  before renormalizing is 1.0 to within 2e-07 over 50 random inputs. What
  remains is a shape cost. Six of the fifteen internal nodes are forced left
  and ignore their logit. Every one of the ``d`` decisions still receives a
  non-zero gradient in both modes, which is the invariant that reproduces. In
  ``deterministic`` mode the kernel is non-trainable, so a tape must
  ``watch`` it explicitly to see those gradients at all. The magnitudes
  depend entirely on the loss, the batch and the input, so no range for them
  is quoted here. Prefer an ``output_dim`` at or near a power of two.

- **Input scale matters in deterministic mode.** Every basis column has
  ``||w_k|| = 1`` (measured 1.0) but nothing constrains ``||x||``, so logits
  scale linearly with the input norm. Unnormalized inputs can push the
  sigmoid into the clip and starve the gradient. Set
  ``input_normalization="rms"`` or ``"l2"`` when the upstream layer does not
  normalize.

- **Under fp16 the output stays float32.** The tree multiplies up to ``d``
  clipped sigmoids together, so leaf masses get small fast. At
  ``output_dim=50000`` the mean leaf mass is 2.0e-05 (that is 1/50000, so it
  is exact, not sampled) and well over 90% of the leaves fall below the
  smallest fp16 normal, 6.104e-05. The exact count depends on the input;
  measured 92.4% to 95.6% across six N(0,1) inputs of shape (2, 64). A final
  cast to fp16 would flush most of the distribution to zero and break the
  sum-to-one invariant. The layer widens the decision logits to at least
  float32 before the sigmoid -- a never-narrow floor, so a float64 policy
  keeps float64 -- runs the whole tree at that dtype, and skips the cast back
  under fp16. Measured output dtype on the same float32 input: ``float32`` gives
  float32, ``mixed_float16`` gives float32, ``mixed_bfloat16`` gives
  bfloat16, ``float64`` gives float64. bfloat16 has fp32's exponent range so
  it needs no override. Both ``RoutingProbabilitiesLayer.call`` (at the final
  cast) and ``RoutingProbabilitiesLayer.compute_output_spec`` carry the
  reasoning; the record is the D-005 entry in the owning plan's
  ``decisions.md``.

- **The cosine basis is orthonormal in the supported regime.** ``build``
  rejects ``input_dim < num_decisions`` outright and warns below
  ``2 * num_decisions``. At ``input_dim >= 2 * num_decisions`` the columns
  are exactly orthonormal, because two columns ``k1 != k2`` can only overlap
  when ``k1 + k2 == input_dim``, which needs ``input_dim <= 2*d - 1``.
  Measured max off-diagonal Gram entry 0.0 to 2.7e-07 over ``(D, d)`` pairs
  (6,3) (7,3) (10,3) (10,5) (11,5) (12,4) (16,3) (16,4) (16,8) (17,3) (17,8)
  (20,10) (21,10) (64,6) (65,6) (100,7) (768,10). Below the bound two columns
  alias onto each other and the off-diagonal is exactly 1.0000, which is what
  the Nyquist warning in ``build`` announces.

References:
    - Zhang, Z., et al. (2024). "Softmax-free Large-scale Language Modeling".
      arXiv preprint arXiv:2402.01258.
    - Morin, F., & Bengio, Y. (2005). "Hierarchical Probabilistic Neural
      Network Language Model". AISTATS.
"""

import keras
import functools
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .common import axis_is_in_range, normalize_axis
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------
# Cosine basis (module-level, cached)
# ---------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def _cached_cosine_basis(
        input_dim: int,
        num_decisions: int,
        norm_eps: float = 1e-12,
) -> np.ndarray:
    """Build the L2-normalized cosine basis used by deterministic mode.

    Column ``k`` (0-based) holds ``cos(2*pi * (k+1) * i / input_dim)`` for
    ``i`` in ``0 .. input_dim-1``, divided by its own L2 norm. Built in
    float64 and returned in float64. It used to be narrowed to float32 on
    the way out, which pinned the initializer to float32 precision even
    under a float64 policy: the realised kernel then differed from the
    float64 basis by 1.371915e-08. Keeping float64 drops that to exactly
    0.0 against a hand-recomputed float64 basis, and cannot move a float32
    or mixed_float16 layer, because rounding float64 to float32 once inside
    ``add_weight`` is bit-identical to rounding it twice: the realised
    float32 kernel is byte-for-byte the same before and after this change
    (measured, both policies, delta exactly 0.0).

    Columns are orthonormal whenever ``input_dim >= 2 * num_decisions``: two
    columns ``k1 != k2`` can only overlap when ``k1 + k2 == input_dim``, and
    ``k1 + k2`` never exceeds ``2 * num_decisions - 1``. Below that bound a
    pair aliases and the two columns become identical. ``build`` rejects
    ``input_dim < num_decisions`` and warns below ``2 * num_decisions``.

    Cached on all three arguments, so rebuilding a layer of the same shape
    re-uses the array. Callers must treat the result as read-only. It is
    handed to ``keras.initializers.Constant``, which copies it on use.

    :param input_dim: Feature dimension ``D``. Number of rows.
    :type input_dim: int
    :param num_decisions: Tree depth ``d``. Number of columns.
    :type num_decisions: int
    :param norm_eps: Added to each column norm before dividing, so a
        zero-norm column cannot divide by zero.
    :type norm_eps: float
    :return: Basis of shape ``(input_dim, num_decisions)``, dtype float64.
    :rtype: np.ndarray
    """
    i = np.arange(input_dim, dtype=np.float64)
    k = np.arange(1, num_decisions + 1, dtype=np.float64)
    basis = np.cos(2.0 * np.pi * np.outer(i, k) / input_dim)
    col_norms = np.sqrt(np.sum(np.square(basis), axis=0, keepdims=True))
    basis = basis / (col_norms + norm_eps)
    return basis


# ---------------------------------------------------------------------
# Structural validity masks for non-pow2 output_dim
# ---------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def _compute_validity_masks(
        output_dim: int,
        padded_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the per-parent overrides that zero out the padded leaves.

    Returns two flat float32 arrays, each of length ``padded_dim - 1`` --
    one entry per internal node, levels concatenated. Level ``k`` has
    ``2**k`` parents and occupies the slice
    ``[2**k - 1 : 2**(k+1) - 1]``. ``call()`` uses them as:

    .. code-block:: text

        p_eff[k, j] = sigmoid(z_k) * mask_mul[k, j] + mask_add[k, j]

    Three cases, by which of a parent's two subtrees still contain a leaf
    below ``output_dim``:

    - both valid -> ``mul=1, add=0``, the sigmoid is used unchanged
    - left only -> ``mul=0, add=0``, the decision is forced left
    - right only -> ``mul=0, add=1``, the decision is forced right

    The third case never happens here. Valid leaves are always the first
    ``output_dim``, a prefix, so a right subtree can only hold one if its
    left sibling does too. Measured over ``output_dim`` 2 to 2048: 0 of 2047
    produced a nonzero ``mask_add``. The case is kept in the formula because
    it costs nothing and makes the formula total.

    When ``output_dim == padded_dim`` every parent has both subtrees valid,
    so ``mask_mul`` is all ones and ``mask_add`` all zeros and the masks are
    a no-op.

    :param output_dim: Number of real classes ``N``.
    :type output_dim: int
    :param padded_dim: Tree width. Must be a positive power of two and at
        least ``output_dim``.
    :type padded_dim: int
    :return: ``(mask_mul, mask_add)``, both shape ``(padded_dim - 1,)``,
        dtype float32.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: If ``padded_dim`` is not a positive power of two.
    """
    if padded_dim & (padded_dim - 1) != 0 or padded_dim < 1:
        raise ValueError(f"padded_dim must be a positive power of two, got {padded_dim}")
    # Number of decision levels.
    d = padded_dim.bit_length() - 1
    # Leaf validity: True for the first ``output_dim`` leaves.
    valid = np.zeros(padded_dim, dtype=bool)
    valid[:output_dim] = True
    # Bottom-up: subtree_valid[k] is a length-2**k boolean array,
    # subtree_valid[k][m] = True iff node m at level k has any valid leaf.
    subtree_valid = [None] * (d + 1)
    subtree_valid[d] = valid
    for k in range(d - 1, -1, -1):
        prev = subtree_valid[k + 1]
        subtree_valid[k] = prev[0::2] | prev[1::2]
    # Per-decision-level masks.
    mul_chunks = []
    add_chunks = []
    for k in range(d):
        # Children of level k, length 2**(k+1), interleaved left/right.
        children = subtree_valid[k + 1]
        valid_left = children[0::2]
        valid_right = children[1::2]
        mul_chunks.append((valid_left & valid_right).astype(np.float32))
        add_chunks.append((valid_right & ~valid_left).astype(np.float32))
    mask_mul = np.concatenate(mul_chunks) if mul_chunks else np.zeros((0,), np.float32)
    mask_add = np.concatenate(add_chunks) if add_chunks else np.zeros((0,), np.float32)
    return mask_mul, mask_add


# ---------------------------------------------------------------------


# One class with a `mode` flag, not two classes and not a base plus two
# subclasses. The axis handling, the tree build and the slice/renormalize are
# identical in both modes, so this keeps them in one place.
@register_dl_technique("dl_techniques.layers.activations.routing_probabilities")
class RoutingProbabilitiesLayer(keras.layers.Layer):
    """
    Hierarchical routing layer: N class probabilities from log2(N) decisions.

    Instead of producing one logit per class, this layer produces
    ``d = log2(next_pow2(output_dim))`` binary decisions and walks them down a
    probability tree. Class ``j`` is the ``j``-th leaf, left to right. Output
    shape equals the input shape with ``axis`` replaced by ``output_dim``, and
    the output sums to 1 along ``axis``.

    ``mode`` picks what makes the decisions:

    - ``"deterministic"`` (default): a fixed cosine-basis projection, stored
      as one non-trainable weight. ``output_dim`` may be ``None``, in which
      case ``build`` infers it from the input dimension at ``axis``.
    - ``"trainable"``: a learnable projection with an optional bias.
      ``output_dim`` is required and must be greater than 1.

    **Architecture Overview:**

    .. code-block:: text

        x  [..., D]   `axis` selects the D dimension
                        │
                        ▼
        ┌────────────────────────────────┐
        │ transpose axis <-> last        │  (optional)
        │ reshape to (batch, D)          │
        └───────────────┬────────────────┘
                        ▼  [batch, D]
        ┌────────────────────────────────┐
        │ l2 / rms normalize             │  (optional)
        └───────────────┬────────────────┘
                        ▼
        ┌────────────────────────────────┐
        │ z = x @ kernel (+ bias)        │  weights
        │ widen to >= float32 (floor)    │
        └───────────────┬────────────────┘
                        ▼  z [batch, d]
        ┌────────────────────────────────┐
        │ p = clip(sigmoid(z), eps,      │
        │          1 - eps)              │
        └───────────────┬────────────────┘
                        ▼  p [batch, d]
        ┌────────────────────────────────┐
        │ tree loop over d levels        │
        │ p_eff = p * mul + add          │
        │ left  = parent * (1 - p_eff)   │
        │ right = parent * p_eff         │
        └───────────────┬────────────────┘
                        ▼  [batch, padded_dim]
        ┌────────────────────────────────┐
        │ slice to output_dim            │
        │ divide by row sum              │  (optional)
        └───────────────┬────────────────┘
                        ▼  [batch, N]
        ┌────────────────────────────────┐
        │ inputs.dtype == float16?       │
        └───────┬────────────────┬───────┘
            yes │                │ no
                ▼                ▼
          keep float32  cast to input dtype
                └───────┬────────┘
                        ▼
        ┌────────────────────────────────┐
        │ reshape to batch shape + N     │
        │ inverse transpose              │  (optional)
        └───────────────┬────────────────┘
                        ▼
        p  [..., N]   sums to 1 over the N classes

    The two boxes marked ``(optional)`` on the transpose and the inverse
    transpose are skipped when ``axis`` is already the last dimension. The
    normalization box runs only when ``input_normalization`` is set. The
    ``divide by row sum`` step runs only when ``normalize=True``; it corrects
    floating-point drift and nothing else, since the structural masks already
    force the padded leaves to exactly zero. Measured over 50 random inputs at
    ``output_dim=10``: turning it on changes the output by less than 1e-07
    (largest single difference seen 5.960e-08, one float32 ulp at 1.0).

    Exact zeros appear in the padded leaf array, not in the output. The
    padded leaves at index ``>= output_dim`` are 0.0 exactly, and the slice
    then discards them. Over the same 50 inputs no output entry was ever 0.0;
    the smallest one seen was 4.7e-04, but that figure is a property of those
    inputs, not of the layer.

    :param output_dim: Number of classes. Required and greater than 1 in
        ``"trainable"`` mode. May be ``None`` in ``"deterministic"`` mode, in
        which case ``build`` sets it from the input dimension at ``axis``.
        ``bool`` is rejected even though it is an ``int`` in Python.
    :type output_dim: Optional[int]
    :param axis: Axis the routing runs along. Defaults to -1.
    :type axis: int
    :param epsilon: Decision probabilities are clipped into
        ``[epsilon, 1 - epsilon]``. Must be in ``[0, 0.5)``; at 0.5 or above
        the clip bounds cross and the backends disagree. ``epsilon=0``
        disables the clip. In ``"trainable"`` mode that can zero the gradient
        on saturated decisions, so the constructor logs it at info level.
    :type epsilon: float
    :param mode: ``"deterministic"`` (default) or ``"trainable"``.
    :type mode: str
    :param kernel_initializer: Initializer for the kernel. Trainable mode
        only. Ignored in deterministic mode, where the kernel is the cosine
        basis.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the bias. Trainable mode only.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernel. Trainable mode
        only.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the bias. Trainable mode only.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kernel_constraint: Constraint on the kernel. Trainable mode only.
    :type kernel_constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param bias_constraint: Constraint on the bias. Trainable mode only.
    :type bias_constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param use_bias: Add a bias to the projection. Trainable mode only. No
        bias variable is created in deterministic mode whatever this is set
        to.
    :type use_bias: bool
    :param normalize: Divide the sliced leaf masses by their sum. Defaults to
        ``True``. This only cleans up floating-point drift; the structural
        masks already force the padded leaves to exactly zero. Set ``False``
        to consume raw masses.
    :type normalize: bool
    :param input_normalization: ``None`` (default), ``"l2"`` or ``"rms"``.
        Applied to the flattened input before the projection. ``"l2"``
        divides by the per-sample L2 norm, ``"rms"`` by the per-sample RMS.
        Worth setting in deterministic mode when the upstream layer does not
        normalize, because cosine-basis logits scale with ``||x||``.
    :type input_normalization: Optional[str]
    :param kwargs: Passed to ``keras.layers.Layer``.

    :raises ValueError: From ``__init__`` if ``mode`` or
        ``input_normalization`` is not one of the accepted values, if
        ``epsilon`` is not a float in ``[0, 0.5)``, if ``axis`` is not an
        integer, or if ``output_dim`` is invalid for the chosen mode. From
        ``build`` if ``axis`` is out of bounds for the input rank, if the
        dimension at ``axis`` is ``None``, or if deterministic mode gets an
        ``input_dim`` below ``num_decisions``.

    :ivar kernel: The projection, shape ``(input_dim, num_decisions)``. A
        non-trainable ``cosine_basis`` weight in deterministic mode, a
        trainable ``kernel`` in trainable mode. ``None`` until ``build``.
    :vartype kernel: Optional[keras.Variable]
    :ivar bias: Shape ``(num_decisions,)``. Created only when
        ``mode="trainable"`` and ``use_bias=True``; ``None`` otherwise.
    :vartype bias: Optional[keras.Variable]
    :ivar padded_output_dim: ``output_dim`` rounded up to a power of two.
        ``None`` until ``build``.
    :vartype padded_output_dim: Optional[int]
    :ivar num_decisions: Tree depth, ``log2(padded_output_dim)``. ``None``
        until ``build``.
    :vartype num_decisions: Optional[int]

    Note:
        Deterministic mode still stores every trainable-only argument so a
        config round-trip is symmetric, and warns once at construction
        listing what it is ignoring. ``use_bias`` is always in that list,
        both values of it, because no bias variable is ever created.
    """

    _VALID_MODES = ("deterministic", "trainable")
    _VALID_INPUT_NORMS = (None, "l2", "rms")
    # Floor on the renormalization denominator, so a zero row sum cannot
    # divide by zero. The divide always runs at AT LEAST float32 -- the
    # decision logits are widened to a float32 floor before the sigmoid and
    # the tree accumulates at that dtype -- so 1e-7 is safely above float32's
    # smallest normal (1.2e-38), and above float64's. Do not tie this constant
    # to the smallest float16 normal; the divide never runs in float16.
    _RENORM_TINY = 1e-7
    # Floor used when L2-normalizing the cosine basis columns, and when
    # normalizing the input in call(). Kept separate from ``self.epsilon``,
    # which is the sigmoid clip, so the two can be tuned independently.
    _BASIS_NORM_EPS = 1e-12

    def __init__(
            self,
            output_dim: Optional[int] = None,
            axis: int = -1,
            epsilon: float = 1e-7,
            mode: str = "deterministic",
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            kernel_constraint: Optional[Union[str, keras.constraints.Constraint]] = None,
            bias_constraint: Optional[Union[str, keras.constraints.Constraint]] = None,
            use_bias: bool = True,
            input_normalization: Optional[str] = None,
            normalize: bool = True,
            **kwargs: Any
    ) -> None:
        """Validate every argument and store it. No weights are made here.

        Shapes are unknown at this point, so ``axis`` is only type-checked;
        ``build`` range-checks it. See the class docstring for the arguments
        and what each raises.
        """
        super().__init__(**kwargs)

        if mode not in self._VALID_MODES:
            raise ValueError(
                f"'mode' must be one of {self._VALID_MODES}, got: {mode!r}"
            )

        if input_normalization not in self._VALID_INPUT_NORMS:
            raise ValueError(
                f"'input_normalization' must be one of "
                f"{self._VALID_INPUT_NORMS}, got: {input_normalization!r}"
            )

        # ``ops.clip(p, eps, 1-eps)`` is undefined once eps >= 0.5, because
        # the lower bound passes the upper one, and the backends degrade
        # differently. eps == 0 disables the clip, which exact-math tests want.
        if (not isinstance(epsilon, (int, float))
                or isinstance(epsilon, bool)
                or not (0.0 <= float(epsilon) < 0.5)):
            raise ValueError(
                f"'epsilon' must be a float in [0, 0.5), got: {epsilon!r}"
            )
        epsilon = float(epsilon)

        # Accept both Python int and numpy integer types for axis.
        if isinstance(axis, (int, np.integer)) and not isinstance(axis, bool):
            axis = int(axis)
        else:
            raise ValueError(
                f"The 'axis' must be an integer, but received: {axis}"
            )

        if mode == "trainable":
            # Reject bool. ``isinstance(True, int)`` is True in Python, so
            # the int check alone would let ``output_dim=True`` through.
            if (not isinstance(output_dim, (int, np.integer))
                    or isinstance(output_dim, bool)
                    or output_dim <= 1):
                raise ValueError(
                    f"In 'trainable' mode, 'output_dim' must be an integer "
                    f"greater than 1, but received: {output_dim!r}"
                )
            output_dim = int(output_dim)
        else:
            # Deterministic mode. output_dim may be None here; build() fills
            # it in from the input shape.
            if output_dim is not None:
                # Reject bool, same reason as the trainable branch above.
                if (not isinstance(output_dim, (int, np.integer))
                        or isinstance(output_dim, bool)
                        or output_dim <= 1):
                    raise ValueError(
                        f"The 'output_dim' must be an integer greater than 1, "
                        f"but received: {output_dim!r}"
                    )
                output_dim = int(output_dim)

        # Track the user-provided value separately so get_config() preserves
        # the original ``None`` semantics in deterministic mode (H1).
        self._user_output_dim = output_dim
        self.output_dim = output_dim
        self.axis = axis
        self.epsilon = epsilon
        self.mode = mode
        self.use_bias = use_bias
        self.normalize = bool(normalize)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.input_normalization = input_normalization

        # Warn once when a deterministic-mode layer is given a trainable-only
        # kwarg that differs from its default. Those kwargs are stored so a
        # config round-trip is symmetric, but they change nothing.
        #
        # Compare through ``serialize`` so a customized GlorotUniform(seed=42)
        # or a non-default Zeros() is caught. Read only ``class_name`` and
        # ``config``, dropping ``shared_object_id``: Keras assigns a fresh id
        # on every deserialization, so including it would fire on every layer
        # reloaded from disk.
        def _init_id(init: keras.initializers.Initializer) -> Tuple[str, Any]:
            cfg = keras.initializers.serialize(init)
            return (cfg.get("class_name"), cfg.get("config"))

        if mode == "deterministic":
            default_kernel_id = _init_id(keras.initializers.GlorotUniform())
            default_bias_id = _init_id(keras.initializers.Zeros())
            ignored = []
            if _init_id(self.kernel_initializer) != default_kernel_id:
                ignored.append("kernel_initializer")
            if _init_id(self.bias_initializer) != default_bias_id:
                ignored.append("bias_initializer")
            if self.kernel_regularizer is not None:
                ignored.append("kernel_regularizer")
            if self.bias_regularizer is not None:
                ignored.append("bias_regularizer")
            if self.kernel_constraint is not None:
                ignored.append("kernel_constraint")
            if self.bias_constraint is not None:
                ignored.append("bias_constraint")
            # ``use_bias`` has no effect in deterministic mode at either
            # value, because no bias is ever created. Report it unconditionally.
            # Do NOT narrow this back to ``use_bias=False`` only: that left
            # ``use_bias=True`` callers believing they had a learnable bias.
            ignored.append(f"use_bias={use_bias}")
            if ignored:
                logger.warning(
                    f"[{self.name}] mode='deterministic' ignores trainable-only "
                    f"kwargs: {ignored}. They are stored for round-trip "
                    f"serialization but have no effect on layer behavior "
                    f"(no bias variable is created in either case)."
                )

        # epsilon=0 turns off the sigmoid clip. In deterministic mode that is
        # a documented escape hatch for exact-math tests: the cosine basis
        # cannot drive a logit to +/-inf. In trainable mode a learned weight
        # can, and a saturated decision then contributes zero gradient. Log it
        # at info level so it shows up in a training log.
        if self.epsilon == 0.0 and self.mode == "trainable":
            logger.info(
                f"[{self.name}] mode='trainable' with epsilon=0.0 disables "
                f"the sigmoid clip; saturated decisions will produce zero "
                f"gradient through this layer. Set epsilon>0 (default 1e-7) "
                f"unless this is a deliberate exact-math configuration."
            )

        self.supports_masking = True

        # Computed in build()
        self.padded_output_dim: Optional[int] = None
        self.num_decisions: Optional[int] = None
        self._normalized_axis: Optional[int] = None
        self._build_input_shape: Optional[Tuple[Optional[int], ...]] = None

        # Projection weight, shape [input_dim, num_decisions]. Non-trainable
        # cosine basis in deterministic mode, learnable in trainable mode.
        self.kernel = None
        # Trainable mode only.
        self.bias = None
        # Per-level structural masks ensuring zero mass on invalid leaves
        # when ``output_dim < padded_output_dim`` (recomputed in build()).
        # Held as numpy arrays, NOT as add_weight. They are a deterministic
        # function of (output_dim, padded_output_dim), so putting them in the
        # checkpoint costs ~8 * (padded_output_dim - 1) bytes per mask per
        # layer -- about 512KB at vocab 65536 -- to store something build()
        # recomputes for free. The conversion to a backend tensor happens
        # inside call(), which is what keeps it out of build's transient
        # FuncGraph. See the longer note at the assignment in build().
        self._mask_mul_np: Optional[np.ndarray] = None
        self._mask_add_np: Optional[np.ndarray] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Resolve the tree dimensions and create the projection weight.

        Normalizes ``axis`` against the input rank, fills in ``output_dim``
        if deterministic mode left it ``None``, derives
        ``padded_output_dim`` and ``num_decisions``, creates the kernel and
        optional bias, and computes the structural masks.

        Returns early if the layer is already built.

        :param input_shape: Shape of the input, including the batch axis.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``axis`` is out of bounds for the rank, if
            ``output_dim`` is ``None`` in trainable mode, if the dimension at
            ``axis`` is ``None``, if the resolved ``output_dim`` is 1 or less,
            or if deterministic mode gets ``input_dim < num_decisions``.
        """
        if self.built:
            return

        # Stash shape so get_build_config() can return it for save/load.
        self._build_input_shape = tuple(input_shape)

        # Normalize axis. The range predicate is `common.axis_is_in_range`,
        # the SAME function compute_output_shape uses, so the two range checks
        # cannot drift apart. Only the message differs, and deliberately: this
        # one's wording is asserted by
        # TestRoutingProbabilitiesLayer::test_invalid_axis_out_of_bounds.
        input_rank = len(input_shape)
        if not axis_is_in_range(self.axis, input_rank):
            raise ValueError(
                f"axis {self.axis} is out of bounds for input shape "
                f"{input_shape}"
            )

        self._normalized_axis = normalize_axis(self.axis, input_rank)

        input_dim = input_shape[self._normalized_axis]

        # Infer output_dim if needed (deterministic mode only)
        if self.output_dim is None:
            if self.mode != "deterministic":
                # __init__ rejects this, so reaching here means the layer
                # was mutated after construction.
                raise ValueError(
                    "output_dim cannot be None in 'trainable' mode."
                )
            if input_dim is None:
                raise ValueError(
                    f"Cannot infer output_dim when the dimension at axis "
                    f"{self.axis} of input_shape is None. Please provide "
                    f"output_dim explicitly."
                )
            self.output_dim = int(input_dim)
            logger.debug(
                f"[{self.name}] Inferred output_dim={self.output_dim} "
                f"from input shape: {input_shape} at axis {self.axis}"
            )

        if self.output_dim <= 1:
            raise ValueError(
                f"output_dim must be greater than 1, got {self.output_dim}"
            )

        # Next power of two at or above output_dim. bit_length is exact;
        # a float log2 is not at large vocabularies.
        self.padded_output_dim = 1 << (self.output_dim - 1).bit_length()
        self.num_decisions = self.padded_output_dim.bit_length() - 1

        if input_dim is None:
            raise ValueError(
                f"The dimension at axis {self.axis} of input_shape must "
                f"be defined to build the projection kernel, got None."
            )

        # The cosine basis needs enough rows to keep its columns distinct.
        # Below num_decisions it is rank-deficient. Below 2 * num_decisions
        # a pair of columns aliases onto each other and becomes identical
        # (measured: max off-diagonal Gram entry exactly 1.0000), so warn.
        if self.mode == "deterministic":
            if input_dim < self.num_decisions:
                raise ValueError(
                    f"In deterministic mode the input dimension at axis "
                    f"{self.axis} ({input_dim}) must be at least "
                    f"num_decisions={self.num_decisions} (= log2 of next "
                    f"power-of-two of output_dim={self.output_dim}); "
                    f"otherwise the cosine basis is rank-deficient."
                )
            if input_dim < 2 * self.num_decisions:
                logger.warning(
                    f"[{self.name}] input_dim={input_dim} is below 2 * "
                    f"num_decisions={2 * self.num_decisions}; cosine basis "
                    f"columns may be near-degenerate (Nyquist regime)."
                )

        logger.debug(
            f"[{self.name}] ({self.mode}) Built for {self.output_dim} "
            f"classes along axis {self.axis}. Padded to "
            f"{self.padded_output_dim}, requiring {self.num_decisions} "
            f"routing decisions."
        )

        # Both modes put a [input_dim, num_decisions] projection on
        # `self.kernel`. The only difference is whether it is trainable.
        #
        # In deterministic mode the cosine basis goes through add_weight with
        # trainable=False, NOT into a plain tensor attribute. A plain tensor
        # created in build() is captured by the FuncGraph that Keras uses for
        # symbolic tracing and reads as "out of scope" when the layer is
        # reused. A non-trainable weight is tracked by the layer and lives
        # outside any transient graph.
        #
        # Do NOT pass dtype=self.compute_dtype here. Variables belong in the
        # layer's variable_dtype, normally float32, even under a
        # mixed-precision policy; Keras casts to compute_dtype inside call().
        if self.mode == "deterministic":
            cosine_np = _cached_cosine_basis(
                input_dim, self.num_decisions, self._BASIS_NORM_EPS
            )
            self.kernel = self.add_weight(
                name="cosine_basis",
                shape=(input_dim, self.num_decisions),
                initializer=keras.initializers.Constant(cosine_np),
                trainable=False,
            )
        else:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(input_dim, self.num_decisions),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
        if self.mode == "trainable" and self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.num_decisions,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )

        # Structural masks: override the decision at every internal node
        # whose subtree holds no leaf below output_dim, so those leaves end
        # up with exactly zero mass. For a power-of-two output_dim every
        # entry is (mul=1, add=0) and the masks do nothing.
        #
        # Held as numpy arrays, NOT add_weight -- see the note where they are
        # declared in __init__ for the size argument. build() recomputes them
        # on every load.
        #
        # This does not contradict the "no plain tensors in build()" rule two
        # blocks up. What is stored here is a numpy array, not a backend
        # tensor. The conversion happens inside call(), at the
        # convert_to_tensor lines just after the matmul, so it lives in
        # call's trace and not in build's transient FuncGraph. If a save/load
        # test ever fails on this path, that distinction is the first thing
        # to check.
        self._mask_mul_np, self._mask_add_np = _compute_validity_masks(
            self.output_dim, self.padded_output_dim
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Route the inputs down the tree and return class probabilities.

        Follow the Architecture Overview in the class docstring: transpose,
        flatten, project, sigmoid, walk the tree, slice, renormalize, restore
        the shape.

        :param inputs: Input tensor. The dimension at ``axis`` must equal the
            ``input_dim`` the layer was built for.
        :type inputs: keras.KerasTensor
        :param training: Accepted for API compatibility and unused. Both
            modes are deterministic: no dropout, no stochastic routing.
        :type training: Optional[bool]
        :return: Probabilities with the same shape as ``inputs`` except at
            ``axis``, which becomes ``output_dim``, summing to 1 along that
            axis. float32 under a float16 compute dtype, ``inputs.dtype``
            otherwise.
        :rtype: keras.KerasTensor
        """
        # --- Step 0: Move target axis to last, flatten to 2D ---
        # ``perm`` swaps two positions, so it is its own inverse. Step 6
        # transposes back with the SAME ``perm``. Do NOT change this to a
        # move-to-end permutation without also computing its inverse there.
        input_rank = len(inputs.shape)
        perm = list(range(input_rank))
        perm[self._normalized_axis] = input_rank - 1
        perm[input_rank - 1] = self._normalized_axis

        if self._normalized_axis != input_rank - 1:
            inputs_transposed = keras.ops.transpose(inputs, perm)
        else:
            inputs_transposed = inputs

        # Take the feature dim from the kernel, not from
        # ``inputs.shape[-1]``, which can be ``None`` under symbolic tracing.
        feature_dim = self.kernel.shape[0]
        inputs_2d = keras.ops.reshape(inputs_transposed, (-1, feature_dim))

        # --- Step 0b: Optional input normalization ---
        # Cosine-basis logits scale linearly with ``||x||``; without
        # normalization the sigmoid saturates and gradient is starved.
        if self.input_normalization == "l2":
            inv_norm = keras.ops.rsqrt(
                keras.ops.sum(keras.ops.square(inputs_2d), axis=-1, keepdims=True)
                + self._BASIS_NORM_EPS
            )
            inputs_2d = inputs_2d * inv_norm
        elif self.input_normalization == "rms":
            inv_norm = keras.ops.rsqrt(
                keras.ops.mean(keras.ops.square(inputs_2d), axis=-1, keepdims=True)
                + self._BASIS_NORM_EPS
            )
            inputs_2d = inputs_2d * inv_norm

        # --- Step 1: Decision logits ---
        decision_logits = keras.ops.matmul(inputs_2d, self.kernel)
        if self.bias is not None:
            decision_logits = decision_logits + self.bias

        # Resolve the dtype the tree runs at: a never-narrow FLOOR, the wider
        # of the incoming dtype and float32. Never an absolute target.
        #
        # Widening is mandatory under fp16, where the clip stops working at
        # the top end: np.float16(1 - 1e-7) is exactly 1.0, so the upper clip
        # is a no-op and a saturated sigmoid leaves p_go_left = 1 - 1.0 = 0.0,
        # zeroing a whole subtree. (The lower end is fine: np.float16(1e-7) is
        # a subnormal 1e-07, not zero.) bfloat16 has even less mantissa, so it
        # is widened too.
        #
        # Do NOT write this as ``cast(decision_logits, "float32")``. That form
        # fixes fp16 but NARROWS a float64 policy back to float32 precision,
        # so a caller who selects float64 to tighten the sum-to-one invariant
        # gets float32's ~1e-07 floor and no warning. Measured before the fix:
        # float32 policy 8.0e-08, float64 policy 8.4e-08 -- identical.
        #
        # The sigmoid, the clip, the masks and the whole tree accumulation all
        # run at ``tree_dtype``; they must agree or the widening is undone by
        # an implicit promotion. Recorded as L-30 in the owning plan.
        incoming_dtype = keras.backend.standardize_dtype(decision_logits.dtype)
        tree_dtype = (
            "float32" if incoming_dtype in ("float16", "bfloat16")
            else incoming_dtype
        )
        decision_logits = keras.ops.cast(decision_logits, tree_dtype)

        decision_probs = keras.ops.sigmoid(decision_logits)
        decision_probs = keras.ops.clip(
            decision_probs, self.epsilon, 1.0 - self.epsilon
        )

        # --- Step 2: Initialize root probability mass = 1.0 ---
        # decision_probs is already at ``tree_dtype`` from the cast above.
        # These three operands take the SAME dtype: if any of them stayed at a
        # hardcoded float32 the accumulation would be pulled back down to
        # float32 under a float64 policy and the widening above would buy
        # nothing.
        mask_mul = keras.ops.convert_to_tensor(self._mask_mul_np, dtype=tree_dtype)
        mask_add = keras.ops.convert_to_tensor(self._mask_add_np, dtype=tree_dtype)
        batch_size = keras.ops.shape(inputs_2d)[0]
        padded_probs = keras.ops.ones((batch_size, 1), dtype=tree_dtype)

        # --- Step 3: Iteratively split tree (with per-parent overrides) ---
        # At each level k, p_eff[k, j] = p_decision * mask_mul[k, j]
        # + mask_add[k, j]. This forces decisions toward subtrees that
        # contain valid leaves and produces EXACTLY zero mass on every leaf
        # at index >= output_dim, regardless of the decision logits.
        offset = 0
        for i in range(self.num_decisions):
            # 2**i parents at this level; mul_i and add_i are (2**i,).
            level_size = 1 << i
            mul_i = mask_mul[offset:offset + level_size]
            add_i = mask_add[offset:offset + level_size]
            offset += level_size

            # This level's decision, shape (batch, 1).
            p_dec = decision_probs[:, i:i + 1]
            # Broadcast (batch, 1) * (2**i,) + (2**i,) -> (batch, 2**i)
            p_go_right = p_dec * mul_i + add_i
            p_go_left = 1.0 - p_go_right

            probs_for_left = padded_probs * p_go_left
            probs_for_right = padded_probs * p_go_right

            combined = keras.ops.stack(
                [probs_for_left, probs_for_right], axis=2
            )
            padded_probs = keras.ops.reshape(combined, (-1, 2 ** (i + 1)))

        # --- Step 4: Slice and renormalize (fp drift cleanup) ---
        # The structural masks already put exactly 0 on every leaf at index
        # >= output_dim, so the slice discards no mass and the row already
        # sums to 1 up to roundoff. The divide only cleans up that drift:
        # measured max difference with and without it at output_dim=10 is
        # below 1e-07 over 50 random inputs.
        if self.output_dim == self.padded_output_dim:
            unnormalized_probs = padded_probs
        else:
            unnormalized_probs = padded_probs[:, :self.output_dim]
        if self.normalize:
            prob_sum = keras.ops.sum(unnormalized_probs, axis=-1, keepdims=True)
            safe_denom = keras.ops.maximum(prob_sum, self._RENORM_TINY)
            final_probs = unnormalized_probs / safe_denom
        else:
            final_probs = unnormalized_probs

        # Under fp16 ONLY, return float32 instead of the compute dtype. This
        # is a scoped override of the mixed-precision contract, kept so the
        # output still sums to 1. Measured at output_dim=50000: mean leaf mass
        # 2.0e-05, and over 90% of leaves below the smallest fp16 normal
        # (6.104e-05) at every input tried, so casting back would flush most
        # of them to zero.
        # bfloat16 has fp32's exponent range, so bf16 and fp32 callers keep
        # the usual behaviour of output dtype == input dtype.
        # ``compute_output_spec`` declares the same fork; the two must agree.
        # Recorded as the D-005 entry in the owning plan's decisions.md.
        if inputs.dtype == "float16":
            # Keep final_probs as fp32.
            pass
        else:
            final_probs = keras.ops.cast(final_probs, inputs.dtype)

        # --- Step 5: Reshape back to original rank ---
        input_transposed_shape = keras.ops.shape(inputs_transposed)
        input_transposed_shape_tensor = keras.ops.convert_to_tensor(
            input_transposed_shape, dtype="int32"
        )
        batch_shape_tensor = input_transposed_shape_tensor[:-1]
        target_dim_tensor = keras.ops.convert_to_tensor(
            [self.output_dim], dtype="int32"
        )
        target_shape_tensor = keras.ops.concatenate(
            [batch_shape_tensor, target_dim_tensor], axis=0
        )
        outputs_transposed = keras.ops.reshape(final_probs, target_shape_tensor)

        # --- Step 6: Restore original axis order ---
        if self._normalized_axis != input_rank - 1:
            outputs = keras.ops.transpose(outputs_transposed, perm)
        else:
            outputs = outputs_transposed

        return outputs

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the input shape with ``axis`` replaced by ``output_dim``.

        Works before ``build``. If ``output_dim`` is still ``None`` it is
        inferred from the dimension at ``axis``, matching what ``build``
        will do.

        :param input_shape: Shape of the input, including the batch axis.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The output shape.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``axis`` is out of bounds for the rank of
            ``input_shape``, or if ``output_dim`` is ``None`` and the
            dimension at ``axis`` is also ``None``.
        """
        output_shape = list(input_shape)
        input_rank = len(input_shape)
        # Recompute the normalized axis from THIS argument's rank. Do NOT
        # use the cached ``self._normalized_axis``: it holds the rank seen at
        # build() time, so a later call with a different-rank shape -- from a
        # wrapper layer or an outer model -- would resolve to the wrong axis.
        # ``self.axis`` is the configured value and the source of truth.
        # ``common.normalize_axis`` / ``axis_is_in_range`` are pure functions of
        # ``(axis, rank)`` and read no layer state, so sharing them with
        # ``build`` preserves that property instead of undoing it.
        if not axis_is_in_range(self.axis, input_rank):
            raise ValueError(
                f"axis {self.axis} is out of bounds for input shape "
                f"{input_shape}"
            )

        normalized_axis = normalize_axis(self.axis, input_rank)

        if self.output_dim is not None:
            output_shape[normalized_axis] = self.output_dim
        else:
            # Pre-build deterministic mode with output_dim=None: try to
            # infer from the input shape, matching what build() will do.
            inferred = input_shape[normalized_axis]
            if inferred is None:
                raise ValueError(
                    "Cannot compute output shape: output_dim is None and "
                    f"input shape at axis {self.axis} is also None. Pass "
                    "output_dim explicitly or call build() first."
                )
            output_shape[normalized_axis] = inferred

        return tuple(output_shape)

    def compute_output_spec(self, inputs):
        """Declare the symbolic output spec, overriding the dtype under fp16.

        Keras' default implementation declares the symbolic output dtype as
        ``self.compute_dtype``. Under ``mixed_float16`` that is float16, and
        the surrounding Functional graph would then coerce the float32 tensor
        ``call()`` actually returns back down to fp16 -- the exact cast the
        final block of ``call()`` avoids. So this declares float32 instead.
        Change the two together or they disagree.

        Only the dtype is overridden. The shape still comes from
        :meth:`compute_output_shape`.

        :param inputs: Symbolic input, or a shape tuple.
        :return: Spec with the routed shape and the declared dtype.
        :rtype: keras.KerasTensor
        """
        input_shape = inputs.shape if hasattr(inputs, "shape") else inputs
        output_shape = self.compute_output_shape(input_shape)
        # Under a float16 compute dtype the output stays float32.
        if self.compute_dtype == "float16":
            out_dtype = "float32"
        else:
            # Match the input dtype. This is what bf16 and fp32 callers get,
            # and it mirrors the cast ``call()`` applies at runtime.
            out_dtype = (
                inputs.dtype if hasattr(inputs, "dtype") else self.compute_dtype
            )
        return keras.KerasTensor(output_shape, dtype=out_dtype)

    def get_build_config(self) -> Dict[str, Any]:
        """Return the build-time input shape so the layer can rebuild on load.

        Without this, a layer whose parent's ``build()`` never calls it --
        an attention module that gates routing on a flag, for instance --
        would come back from a ``.keras`` file with no kernel and no bias.

        :return: ``{"input_shape": <shape>}`` if built, otherwise ``{}``.
        :rtype: Dict[str, Any]
        """
        if self.built and self._build_input_shape is not None:
            return {"input_shape": self._build_input_shape}
        return {}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Rebuild from the config :meth:`get_build_config` produced.

        Does nothing if the config is empty, which is what an unbuilt layer
        saved.

        :param config: Value returned by :meth:`get_build_config`.
        :type config: Dict[str, Any]
        """
        if config and "input_shape" in config:
            self.build(config["input_shape"])

    def get_config(self) -> Dict[str, Any]:
        """Serialize every constructor argument, in both modes.

        ``output_dim`` is written back as the user supplied it, which may be
        ``None``, not as the value ``build`` resolved. A deterministic-mode
        layer that inferred its ``output_dim`` therefore infers it again
        after a reload, instead of being pinned to whatever shape it first
        saw.

        Trainable-only arguments are serialized in deterministic mode too, so
        the round trip is symmetric.

        :return: Config dict, including the base Layer config.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "output_dim": self._user_output_dim,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "mode": self.mode,
            "use_bias": self.use_bias,
            "input_normalization": self.input_normalization,
            "normalize": self.normalize,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": keras.initializers.serialize(
                self.bias_initializer
            ),
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": keras.regularizers.serialize(
                self.bias_regularizer
            ),
            "kernel_constraint": keras.constraints.serialize(
                self.kernel_constraint
            ),
            "bias_constraint": keras.constraints.serialize(
                self.bias_constraint
            ),
        })
        return config


# ---------------------------------------------------------------------
