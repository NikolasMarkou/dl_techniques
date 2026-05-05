"""
Hierarchical routing tree for classification (deterministic or trainable).

This module unifies two routing variants behind a single layer,
``RoutingProbabilitiesLayer``, selectable via the ``mode`` parameter:

1. ``mode="deterministic"`` (default): Non-trainable, parameter-free routing
   using a fixed cosine basis projection. A drop-in alternative to softmax that
   introduces a structured, hierarchical bias without adding any trainable
   parameters.

2. ``mode="trainable"``: Learnable routing using a standard affine projection
   (``W x + b``). A drop-in replacement for ``Dense -> Softmax`` whose output
   projection cost is reduced from ``O(N)`` to ``O(log N)`` decisions.

Both modes share the same hierarchical probability tree:

1. **Padding**: ``output_dim`` is padded to the next power of two,
   ``padded_dim``. The number of routing decisions is
   ``d = log2(padded_dim)``.

2. **Decision Logits**: For each of the ``d`` decisions, a logit ``z_k`` is
   produced. In deterministic mode, ``z_k = <x, w_k>`` with
   ``w_{k,i} = cos(2*pi * (k+1) * i / D)``. In trainable mode,
   ``z = x W + b`` for a learnable ``W`` of shape ``[D, d]``.

3. **Probabilistic Decisions**: ``p_k = sigmoid(z_k)`` is the probability of
   taking the right branch at level ``k``.

4. **Hierarchical Routing**: Probability mass starts at 1.0 and is split at
   each level: ``left = parent * (1 - p_k)``, ``right = parent * p_k``.

5. **Renormalization**: The accumulated mass at each of the ``padded_dim``
   leaves is sliced to ``output_dim`` and renormalized to sum to 1.0.

Caveats:

- **Class index ordering is load-bearing.** Class ``j`` is the ``j``-th leaf
  in left-to-right tree traversal. In trainable mode the projection ``W`` can
  permute classes to match an arbitrary semantic structure. In deterministic
  mode the topology is fixed: classes whose indices are numerically adjacent
  share long path prefixes; classes with distant indices share none.
  Performance therefore depends on whether the class index space encodes
  meaningful structure (e.g., language-model token IDs typically do not).

- **Slice-then-renormalize is structurally biased for non-pow2 ``N``.** When
  ``output_dim < padded_dim`` the discarded leaves form a contiguous tail of
  the leaf array, so the first decision's "right" branch is always the one
  partially or fully truncated. Renormalization corrects the sum but the
  asymmetry couples decisions in the gradient and creates a per-decision
  bias whose magnitude grows as ``output_dim`` moves away from a power of
  two. Prefer ``output_dim`` close to (or equal to) a power of two.

- **Input scale matters in deterministic mode.** ``z_k = <x, w_k>`` with
  ``||w_k|| = 1`` but no constraint on ``||x||``. Logits scale linearly with
  input norm, so unnormalized inputs can saturate the sigmoid into the
  clipping range and starve gradients. If your upstream layer does not
  normalize, set ``input_normalization="rms"`` (or ``"l2"``) on this layer.

- **Mixed precision.** The hierarchical product accumulates up to
  ``log2(padded_dim)`` clipped sigmoid factors. Under ``float16`` the
  sigmoid clip floor (``epsilon=1e-7``) lies below the smallest normal
  (~6.1e-5), so deep trees can underflow. The layer therefore casts the
  decision logits to ``float32`` BEFORE the sigmoid+clip, and runs the
  tree accumulation in ``float32`` regardless of the compute dtype.
  Output dtype: under ``float16`` compute_dtype the output stays
  ``float32`` to preserve the sum-to-one invariant (individual leaves at
  large vocab are below the smallest fp16 normal); under ``bfloat16`` /
  ``float32`` the output matches the input dtype as usual. This scoped
  override of the mixed-precision compute_dtype contract is anchored as
  ``# DECISION D-005`` at the cast site.

- **Cosine basis is not orthogonal.** Columns are L2-normalized, but the
  set ``{cos(2*pi*k*i/D) : k=1..d}`` is mutually orthogonal only for
  specific commensurate ``(D, k)`` pairs. The basis is a fixed projection,
  not an orthonormal frame.

References:
    - Zhang, Z., et al. (2024). "Softmax-free Large-scale Language Modeling".
      arXiv preprint arXiv:2402.01258.
    - Morin, F., & Bengio, Y. (2005). "Hierarchical Probabilistic Neural
      Network Language Model". AISTATS.
"""

import math
import functools
import keras
import numpy as np
from keras import ops
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Cosine basis (module-level, cached)
# ---------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def _cached_cosine_basis(
        input_dim: int,
        num_decisions: int,
        norm_eps: float = 1e-12,
) -> np.ndarray:
    """L2-normalized cosine basis as numpy, shape ``(input_dim, num_decisions)``.

    Cached on ``(input_dim, num_decisions)``: rebuilding a layer with the
    same shape re-uses the previously computed basis. The returned array is
    treated as read-only by callers (it is passed to
    ``keras.initializers.Constant`` which copies it on use).
    """
    i = np.arange(input_dim, dtype=np.float64)
    k = np.arange(1, num_decisions + 1, dtype=np.float64)
    basis = np.cos(2.0 * np.pi * np.outer(i, k) / input_dim)
    col_norms = np.sqrt(np.sum(np.square(basis), axis=0, keepdims=True))
    basis = basis / (col_norms + norm_eps)
    return basis.astype(np.float32)


# ---------------------------------------------------------------------
# Structural validity masks for non-pow2 output_dim
# ---------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def _compute_validity_masks(
        output_dim: int,
        padded_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-level decision overrides forcing invalid-leaf mass to be exactly 0.

    Returns two flat numpy arrays of length ``padded_dim - 1`` (the total
    number of internal-node positions across all decision levels). At
    decision level ``k`` (with ``2**k`` parent positions), the slice
    ``[2**k - 1 : 2**(k+1) - 1]`` of each array gives per-parent overrides
    consumed by ``call()``:

    .. code-block:: text

        p_eff[k, j] = sigmoid(z_k) * mask_mul[k, j] + mask_add[k, j]

    With:

    - both children valid -> ``mask_mul=1, mask_add=0`` (use sigmoid as-is)
    - only right child valid -> ``mask_mul=0, mask_add=1`` (force right)
    - only left child valid -> ``mask_mul=0, mask_add=0`` (force left)

    For ``output_dim == padded_dim`` (pow2 case), every parent has both
    subtrees valid, ``mask_mul=1`` and ``mask_add=0`` everywhere — exactly
    the unmasked tree.
    """
    if padded_dim & (padded_dim - 1) != 0 or padded_dim < 1:
        raise ValueError(f"padded_dim must be a positive power of two, got {padded_dim}")
    d = padded_dim.bit_length() - 1  # number of decisions
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
        children = subtree_valid[k + 1]  # length 2**(k+1)
        valid_left = children[0::2]
        valid_right = children[1::2]
        mul_chunks.append((valid_left & valid_right).astype(np.float32))
        add_chunks.append((valid_right & ~valid_left).astype(np.float32))
    mask_mul = np.concatenate(mul_chunks) if mul_chunks else np.zeros((0,), np.float32)
    mask_add = np.concatenate(add_chunks) if add_chunks else np.zeros((0,), np.float32)
    return mask_mul, mask_add


# ---------------------------------------------------------------------


# DECISION D-001: A single class with a `mode` flag is preferred over two
# distinct classes or an inheritance hierarchy. This keeps the shared
# axis-manipulation, tree-build, and slice/renormalize logic in one place.
# Trainable-only kwargs are accepted in deterministic mode for config
# round-trip symmetry, but a warning is logged if any non-default
# trainable-only kwarg is supplied to a deterministic-mode layer.
@keras.saving.register_keras_serializable()
class RoutingProbabilitiesLayer(keras.layers.Layer):
    """
    Hierarchical routing layer for probabilistic classification.

    Supports two modes selected via ``mode``:

    - ``"deterministic"``: parameter-free routing using a fixed cosine basis
      projection. ``output_dim`` may be ``None`` (inferred at build time
      from the input shape at ``axis``).
    - ``"trainable"``: learnable routing via a Dense projection. ``output_dim``
      is required.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────┐
        │    Input Features [batch, ..., D]       │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Decision Projection                    │
        │  deterministic: z = <x, cos_basis>      │
        │  trainable:     z = x W + b             │
        │  -> [batch, d]                          │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Sigmoid + Clip                         │
        │  p_k = sigma(z_k) in [eps, 1-eps]       │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Hierarchical Probability Tree          │
        │                                         │
        │            Root (p=1.0)                 │
        │           ┌───┴───┐                     │
        │       (1-p0)    (p0)                    │
        │       ┌──┴──┐  ┌──┴──┐                  │
        │      ...   ... ...   ...                │
        │       │     │   │     │                 │
        │      L0    L1  L2    L3  ...            │
        │                                         │
        │  Binary splits at each level k          │
        │  left = parent * (1 - p_k)              │
        │  right = parent * p_k                   │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Structural Validity Mask (non-pow2 N)  │
        │  At each level, force decisions toward  │
        │  subtrees that contain valid leaves.    │
        │  Invalid leaves -> exactly 0 mass.      │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Slice & Renormalize to output_dim      │
        │  Keep first output_dim leaves           │
        │  Renormalize to sum = 1.0 (fp drift)    │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │  Output Probabilities [batch, ..., N]   │
        └─────────────────────────────────────────┘

    :param output_dim: Dimensionality of the output space (number of classes).
        In ``"deterministic"`` mode this may be ``None`` and is inferred from
        the dimension at ``axis`` of the input shape during build. In
        ``"trainable"`` mode it must be an integer greater than 1.
    :type output_dim: Optional[int]
    :param axis: Axis along which the routing is applied. Defaults to -1.
    :type axis: int
    :param epsilon: Small float for sigmoid clipping (probabilities are
        clipped into ``[epsilon, 1 - epsilon]``).
    :type epsilon: float
    :param mode: Routing mode. ``"deterministic"`` (default) for the
        parameter-free cosine-basis projection, or ``"trainable"`` for a
        learnable Dense projection.
    :type mode: str
    :param kernel_initializer: Initializer for the trainable kernel
        (``"trainable"`` mode only).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the trainable bias.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the trainable kernel.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the trainable bias.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kernel_constraint: Constraint for the trainable kernel.
    :type kernel_constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param bias_constraint: Constraint for the trainable bias.
    :type bias_constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param use_bias: Whether to use a bias vector in trainable mode.
    :type use_bias: bool
    :param normalize: If ``True`` (default), the sliced leaf masses are
        divided by their sum (defensive cleanup of fp roundoff from sigmoid
        clipping). If ``False``, the divide is skipped and the raw masses
        are returned. With structural masking, invalid leaves are exactly
        zero either way; the sum over valid leaves equals 1 up to fp drift.
        Set to ``False`` if you want to consume raw masses downstream.
    :type normalize: bool
    :param input_normalization: Optional input normalization applied before
        the projection. ``None`` (default) leaves inputs unchanged.
        ``"l2"`` divides by per-sample L2 norm; ``"rms"`` divides by per-sample
        RMS. Recommended in deterministic mode when the upstream layer does
        not normalize: cosine-basis logits scale linearly with ``||x||``, so
        unnormalized inputs can saturate the sigmoid into the clipping range
        and starve gradients.
    :type input_normalization: Optional[str]
    :param kwargs: Additional arguments for the Layer base class.
    """

    _VALID_MODES = ("deterministic", "trainable")
    _VALID_INPUT_NORMS = (None, "l2", "rms")
    # Smallest representable value used as a clamp on the renormalization
    # denominator. The renormalization divide runs in float32 regardless of
    # the compute dtype (decision logits are cast to float32 before the
    # sigmoid+clip and the tree accumulates in float32), so the 1e-7 floor
    # is float32-safe. The earlier rationale tying this constant to the
    # smallest float16 normal no longer applies and was misleading.
    # See C2/M3 in review and DECISION D-005.
    _RENORM_TINY = 1e-7
    # Smallest divisor used inside cosine-basis L2 normalization. Decoupled
    # from ``self.epsilon`` (which controls sigmoid clipping) so the two can
    # be tuned independently. See M4 in review.
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

        # H-1: validate epsilon. ``ops.clip(p, eps, 1-eps)`` is undefined
        # when eps >= 0.5 (min > max) and silently degrades by backend.
        # eps == 0 disables clipping (a valid choice for exact-math tests).
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
            if not isinstance(output_dim, (int, np.integer)) or output_dim <= 1:
                raise ValueError(
                    f"In 'trainable' mode, 'output_dim' must be an integer "
                    f"greater than 1, but received: {output_dim}"
                )
            output_dim = int(output_dim)
        else:  # deterministic
            if output_dim is not None:
                if (not isinstance(output_dim, (int, np.integer))
                        or output_dim <= 1):
                    raise ValueError(
                        f"The 'output_dim' must be an integer greater than 1, "
                        f"but received: {output_dim}"
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

        # H-4: warn (once) when trainable-only kwargs differ from their
        # canonical defaults in deterministic mode (they are stored for
        # round-trip serialization but have no effect on layer behavior).
        # Compare via ``serialize`` so a customized GlorotUniform(seed=42)
        # or Zeros() with a non-default config is correctly detected.
        # We strip ``shared_object_id`` because Keras assigns a fresh id
        # per deserialization, which would otherwise produce false positives
        # after save/load.
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
            if not use_bias:
                ignored.append("use_bias")
            if ignored:
                logger.warning(
                    f"[{self.name}] mode='deterministic' ignores trainable-only "
                    f"kwargs: {ignored}. They are stored for round-trip "
                    f"serialization but have no effect on layer behavior."
                )

        self.supports_masking = True

        # Computed in build()
        self.padded_output_dim: Optional[int] = None
        self.num_decisions: Optional[int] = None
        self._normalized_axis: Optional[int] = None
        self._build_input_shape: Optional[Tuple[Optional[int], ...]] = None

        # Projection weight (shape: [input_dim, num_decisions]).
        # Non-trainable cosine basis in deterministic mode, learnable in trainable mode.
        self.kernel = None
        self.bias = None  # trainable mode only
        # Per-level structural masks ensuring zero mass on invalid leaves
        # when ``output_dim < padded_output_dim`` (created in build()).
        # Both have shape ``(padded_output_dim - 1,)`` and are concatenated
        # by level: level k spans indices [2**k - 1, 2**(k+1) - 1).
        self._mask_mul = None
        self._mask_add = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer: compute tree dims and create projection state."""
        # Stash shape so get_build_config() can return it for save/load.
        self._build_input_shape = tuple(input_shape)
        # M7: mark layer built before creating weights so error tracebacks
        # surface from add_weight, not from "layer not built" downstream.
        super().build(input_shape)

        # Normalize axis
        input_rank = len(input_shape)
        if self.axis < 0:
            self._normalized_axis = input_rank + self.axis
        else:
            self._normalized_axis = self.axis

        if self._normalized_axis < 0 or self._normalized_axis >= input_rank:
            raise ValueError(
                f"axis {self.axis} is out of bounds for input shape "
                f"{input_shape}"
            )

        input_dim = input_shape[self._normalized_axis]

        # Infer output_dim if needed (deterministic mode only)
        if self.output_dim is None:
            if self.mode != "deterministic":
                # Defensive: __init__ should have rejected this, but check.
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

        # Padded power-of-two tree size. H2: integer-exact via bit_length.
        self.padded_output_dim = 1 << (self.output_dim - 1).bit_length()
        self.num_decisions = self.padded_output_dim.bit_length() - 1

        if input_dim is None:
            raise ValueError(
                f"The dimension at axis {self.axis} of input_shape must "
                f"be defined to build the projection kernel, got None."
            )

        # H3: validate that the projection has enough columns to be
        # independent. ``input_dim < num_decisions`` makes the cosine basis
        # rank-deficient by construction; warn near the Nyquist boundary.
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

        # DECISION D-003: both modes share the same projection
        # [input_dim, num_decisions] with attribute name `self.kernel`. The
        # only difference is whether it is a trainable Keras weight.
        # DECISION D-004: in deterministic mode the cosine basis is stored
        # as a non-trainable Keras weight (add_weight(trainable=False))
        # rather than a plain tensor. Plain tensors created inside build()
        # get captured in the FuncGraph used by Keras' compute_output_spec
        # symbolic tracing, which then becomes "out of scope" when the
        # layer is reused. Non-trainable weights are tracked by the layer
        # and live outside any transient graph.
        # C1: Do NOT pass dtype=self.compute_dtype here. Variables should
        # live in the layer's variable_dtype (typically float32) even under
        # mixed-precision policies; Keras automatically casts to
        # compute_dtype inside call().
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

        # Structural validity masks: ensure that any leaf at index >=
        # output_dim receives EXACTLY zero probability mass, by overriding
        # decisions at internal nodes whose subtree contains no valid leaf.
        # For pow2 output_dim every entry is (mul=1, add=0) and the masks
        # are a no-op. Stored as non-trainable weights so they survive
        # save/load and are tracked by Keras (matches D-004 reasoning).
        mask_mul_np, mask_add_np = _compute_validity_masks(
            self.output_dim, self.padded_output_dim
        )
        flat_len = self.padded_output_dim - 1
        self._mask_mul = self.add_weight(
            name="leaf_mask_mul",
            shape=(flat_len,),
            initializer=keras.initializers.Constant(mask_mul_np),
            trainable=False,
        )
        self._mask_add = self.add_weight(
            name="leaf_mask_add",
            shape=(flat_len,),
            initializer=keras.initializers.Constant(mask_add_np),
            trainable=False,
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply hierarchical routing to produce class probabilities.

        ``training`` is accepted for API compatibility but unused: the layer
        is deterministic in both modes (no dropout, no stochastic routing).
        """
        # --- Step 0: Move target axis to last, flatten to 2D ---
        # M2: ``perm`` is intentionally a self-inverse swap (not a
        # move-to-end). The output is transposed back with the SAME ``perm``
        # at the end of call(); changing this to a move-to-end would
        # require computing a separate inverse permutation.
        input_rank = len(inputs.shape)
        perm = list(range(input_rank))
        perm[self._normalized_axis] = input_rank - 1
        perm[input_rank - 1] = self._normalized_axis

        if self._normalized_axis != input_rank - 1:
            inputs_transposed = ops.transpose(inputs, perm)
        else:
            inputs_transposed = inputs

        # C3: pull the static feature dim from the kernel rather than from
        # ``inputs.shape[-1]`` which can be ``None`` under symbolic tracing.
        feature_dim = self.kernel.shape[0]
        inputs_2d = ops.reshape(inputs_transposed, (-1, feature_dim))

        # --- Step 0b: Optional input normalization (H-2) ---
        # Cosine-basis logits scale linearly with ``||x||``; without
        # normalization the sigmoid saturates and gradient is starved.
        if self.input_normalization == "l2":
            inv_norm = ops.rsqrt(
                ops.sum(ops.square(inputs_2d), axis=-1, keepdims=True)
                + self._BASIS_NORM_EPS
            )
            inputs_2d = inputs_2d * inv_norm
        elif self.input_normalization == "rms":
            inv_norm = ops.rsqrt(
                ops.mean(ops.square(inputs_2d), axis=-1, keepdims=True)
                + self._BASIS_NORM_EPS
            )
            inputs_2d = inputs_2d * inv_norm

        # --- Step 1: Decision logits ---
        decision_logits = ops.matmul(inputs_2d, self.kernel)
        if self.bias is not None:
            decision_logits = decision_logits + self.bias

        # B-1 fix: cast logits to float32 BEFORE the sigmoid+clip. Under fp16
        # mixed precision the sigmoid clip floor (epsilon=1e-7) is below the
        # smallest fp16 normal (~6.1e-5), so clipping in fp16 would round the
        # floor to zero. Doing the sigmoid+clip and the entire tree
        # accumulation in float32 is the supported path under mixed precision.
        decision_logits = ops.cast(decision_logits, "float32")

        decision_probs = ops.sigmoid(decision_logits)
        decision_probs = ops.clip(
            decision_probs, self.epsilon, 1.0 - self.epsilon
        )

        # --- Step 2: Initialize root probability mass = 1.0 ---
        # decision_probs is already float32 from the cast above; the tree
        # accumulation continues in float32 regardless of compute dtype.
        mask_mul = ops.cast(self._mask_mul, "float32")
        mask_add = ops.cast(self._mask_add, "float32")
        batch_size = ops.shape(inputs_2d)[0]
        padded_probs = ops.ones((batch_size, 1), dtype="float32")

        # --- Step 3: Iteratively split tree (with per-parent overrides) ---
        # At each level k, p_eff[k, j] = p_decision * mask_mul[k, j]
        # + mask_add[k, j]. This forces decisions toward subtrees that
        # contain valid leaves and produces EXACTLY zero mass on every leaf
        # at index >= output_dim, regardless of the decision logits.
        offset = 0
        for i in range(self.num_decisions):
            level_size = 1 << i  # 2**i parents at this level
            mul_i = mask_mul[offset:offset + level_size]  # shape (2**i,)
            add_i = mask_add[offset:offset + level_size]
            offset += level_size

            p_dec = decision_probs[:, i:i + 1]  # shape (batch, 1)
            # Broadcast (batch, 1) * (2**i,) + (2**i,) -> (batch, 2**i)
            p_go_right = p_dec * mul_i + add_i
            p_go_left = 1.0 - p_go_right

            probs_for_left = padded_probs * p_go_left
            probs_for_right = padded_probs * p_go_right

            combined = ops.stack(
                [probs_for_left, probs_for_right], axis=2
            )
            padded_probs = ops.reshape(combined, (-1, 2 ** (i + 1)))

        # --- Step 4: Slice and renormalize (fp drift cleanup) ---
        # With structural masking, leaves at index >= output_dim are
        # exactly 0 by construction, so the slice is a no-op on mass and
        # the sum is 1.0 up to fp roundoff from the sigmoid clip. The
        # renormalize remains as a defensive cleanup of that drift.
        if self.output_dim == self.padded_output_dim:
            unnormalized_probs = padded_probs
        else:
            unnormalized_probs = padded_probs[:, :self.output_dim]
        if self.normalize:
            prob_sum = ops.sum(unnormalized_probs, axis=-1, keepdims=True)
            safe_denom = ops.maximum(prob_sum, self._RENORM_TINY)
            final_probs = unnormalized_probs / safe_denom
        else:
            final_probs = unnormalized_probs

        # DECISION D-005: scoped override of the mixed-precision compute_dtype
        # contract — under fp16 ONLY, keep the output in float32 to preserve
        # the sum-to-one invariant. At large vocab (e.g. 50K+) individual leaf
        # masses (~2e-5) are below the smallest fp16 normal (~6.1e-5), so a
        # final cast to fp16 would clobber every leaf to zero. bfloat16 has
        # fp32-like dynamic range so this concern doesn't apply; bf16/fp32
        # callers see the historical behavior (output dtype == input dtype).
        # Anchor: see plans/.../decisions.md (D-005 entry) and the module-
        # level "Mixed precision" docstring.
        if inputs.dtype == "float16":
            pass  # keep final_probs as fp32
        else:
            final_probs = ops.cast(final_probs, inputs.dtype)

        # --- Step 5: Reshape back to original rank ---
        input_transposed_shape = ops.shape(inputs_transposed)
        input_transposed_shape_tensor = ops.convert_to_tensor(
            input_transposed_shape, dtype="int32"
        )
        batch_shape_tensor = input_transposed_shape_tensor[:-1]
        target_dim_tensor = ops.convert_to_tensor(
            [self.output_dim], dtype="int32"
        )
        target_shape_tensor = ops.concatenate(
            [batch_shape_tensor, target_dim_tensor], axis=0
        )
        outputs_transposed = ops.reshape(final_probs, target_shape_tensor)

        # --- Step 6: Restore original axis order ---
        if self._normalized_axis != input_rank - 1:
            outputs = ops.transpose(outputs_transposed, perm)
        else:
            outputs = outputs_transposed

        return outputs

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape: input shape with `axis` replaced by `output_dim`.

        M6: when ``output_dim`` is ``None`` and the layer has not been built,
        attempt to infer it from the dimension at ``axis`` of ``input_shape``.
        Raise if the dimension is also ``None`` (cannot infer).
        """
        output_shape = list(input_shape)
        input_rank = len(input_shape)
        # B-3: always recompute the normalized axis from the argument's rank.
        # The cached ``self._normalized_axis`` reflects the rank seen at
        # build() time; if compute_output_shape is later called with a
        # different-rank shape (e.g. by a wrapper layer or an outer model),
        # using the cached value yields the wrong axis. ``self.axis`` is the
        # source of truth for the configured axis; resolve it against the
        # actual input shape.
        normalized_axis = (
            input_rank + self.axis if self.axis < 0 else self.axis
        )

        if normalized_axis < 0 or normalized_axis >= input_rank:
            raise ValueError(
                f"axis {self.axis} is out of bounds for input shape "
                f"{input_shape}"
            )

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
        """Override the symbolic output dtype for D-005.

        Keras' default ``compute_output_spec`` (when ``compute_output_shape``
        is implemented) declares the symbolic output dtype as
        ``self.compute_dtype``. Under ``mixed_float16`` that is ``float16``,
        which would force the runtime tensor returned by ``call()`` to be
        coerced to fp16 by the surrounding Functional graph. We must keep
        the output in fp32 under fp16 compute_dtype (D-005 — see the cast
        site in ``call()`` and the module-level "Mixed precision" docstring).

        This override only changes the declared dtype; the shape logic still
        delegates to ``compute_output_shape``.
        """
        from keras import KerasTensor  # local import to avoid module cycles
        input_shape = inputs.shape if hasattr(inputs, "shape") else inputs
        output_shape = self.compute_output_shape(input_shape)
        # D-005: under fp16 compute_dtype, output stays fp32.
        if self.compute_dtype == "float16":
            out_dtype = "float32"
        else:
            # Match the input dtype (bf16/fp32 callers see historical
            # behavior). The runtime ``call()`` casts to ``inputs.dtype``.
            out_dtype = (
                inputs.dtype if hasattr(inputs, "dtype") else self.compute_dtype
            )
        return KerasTensor(output_shape, dtype=out_dtype)

    def get_build_config(self) -> Dict[str, Any]:
        """Return the input shape so the layer can rebuild on load.

        Ensures that when the layer is a child of a parent whose ``build()``
        does not eagerly invoke routing (e.g. attention modules that gate
        routing on a flag), Keras can still reconstruct the kernel/bias
        variables at load time.
        """
        if self.built and self._build_input_shape is not None:
            return {"input_shape": self._build_input_shape}
        return {}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Rebuild from a saved build config produced by ``get_build_config``."""
        if config and "input_shape" in config:
            self.build(config["input_shape"])

    def get_config(self) -> Dict[str, Any]:
        """Serialize all parameters (both modes) for round-trip stability.

        H1: the original user-supplied ``output_dim`` (which may be ``None``)
        is preserved across save/load, rather than the value resolved during
        build. This keeps the inference behavior of deterministic-mode
        layers symmetric across reconstruction.
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
