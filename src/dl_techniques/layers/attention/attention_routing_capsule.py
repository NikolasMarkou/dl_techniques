"""Capsule layers V2 — single-step attention routing with decoupled length & probability.

This module provides a modernised capsule routing primitive that addresses
several documented limitations of the iterative dynamic-routing scheme used in
`dl_techniques.layers.capsules.RoutingCapsule`:

* **No iterative inner loop.** Routing is computed in a single forward pass via
  attention-style scoring. This makes the layer fully parallel, XLA-fusable, and
  removes the sequential ``b_{i+1} = b_i + agreement(v_i)`` data dependency that
  blocks GPU/TPU utilization in the legacy implementation.
* **Decoupled length and probability.** The capsule output is constructed as
  ``v = sigmoid(prob_head(s)) * (s / ||s||)`` — magnitude is a learned scalar
  per capsule (the detection probability under margin loss), and direction is a
  unit vector encoding the pose. This eliminates the squash function's
  saturation at zero and the conflation of "vector magnitude" with "detection
  probability".
* **Optional Top-K sparsity.** Each routing softmax can be restricted to the
  top-k input capsules, keeping per-output cost sub-quadratic when scaling to
  many input capsules (a la mixture-of-experts gating).
* **Optional auxiliary load-balancing loss.** When enabled, an importance loss
  on the routing assignments is added via ``self.add_loss`` during training to
  discourage coupling collapse (a single output capsule monopolizing the
  routing).

The companion :class:`CapsuleBlockV2` wraps :class:`AttentionRoutingCapsule`
with optional dropout and a length-preserving direction-only normalizer.

Architecture:
    Two layers, one of which composes the other::

        AttentionRoutingCapsule : (B, N_in, D_in) -> (B, N_out, D_out)
            u_hat -> score -> [top-k] -> softmax -> aggregate -> decoupled output

        CapsuleBlockV2 : (B, N_in, D_in) -> (B, N_out, D_out)
            AttentionRoutingCapsule -> [Dropout] -> [direction-only LayerNorm]

    ``CapsuleBlockV2`` forwards every routing argument through to the capsule it
    owns, so the two classes share one parameter vocabulary. It is consumed in
    production by ``models/capsnet/model_v2.py``.

Foundational Mathematics:
    Routing is one softmax over agreement scores rather than an iterative
    agreement loop::

        u_hat[b,i,o,:] = W[i,o,:,:] @ u[b,i,:]
        score[b,i,o]   = <u_hat[b,i,o,:], q[o,:]> / sqrt(D_out)
        a              = softmax(score, axis = output | input)
        s[b,o,:]       = sum_i a[b,i,o] * u_hat[b,i,o,:]  (+ bias)
        v[b,o,:]       = sigmoid(prob_head(s)) * s / (||s|| + eps)

    The final line is the decoupling: ``||v||`` is a learned scalar in ``(0, 1)``
    read as a detection probability, while ``v / ||v||`` carries the pose. The
    classic ``squash`` conflates the two and saturates to zero magnitude for
    small ``||s||``; this form cannot.

References:
    - Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between
      capsules. Advances in NeurIPS 30.
    - Hahn, T., Pyeon, M., & Kim, G. (2019). Self-Routing Capsule Networks.
    - Tsai, Y.-H. H., et al. (2020). Capsules with Inverted Dot-Product Attention
      Routing.
    - Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The
      Sparsely-Gated Mixture-of-Experts Layer (load-balancing loss).
"""

# ---------------------------------------------------------------------

import keras
from typing import Optional, Tuple, Union, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations.squash import SquashLayer  # noqa: F401  (kept for API parity)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AttentionRoutingCapsule(keras.layers.Layer):
    """Single-step attention-routing capsule layer.

    A drop-in replacement for the iterative :class:`RoutingCapsule` that uses
    a learned per-output query to score the prediction vectors ``u_hat`` and
    a single softmax to compute the routing weights, then aggregates with a
    decoupled magnitude/direction output.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────────────┐
        │     AttentionRoutingCapsule — single-step attention routing     │
        │                                                                 │
        │  Input u [B, N_in, D_in] — N_in must be STATIC: W is indexed    │
        │  per input capsule, and build() raises if it is None.           │
        │                             ▼                                   │
        │  u_hat = einsum('iode,bie->biod', W, u)   a pose transform,     │
        │          W     [N_in, N_out, D_out, D_in]  NOT a Q/K/V          │
        │          u_hat [B, N_in, N_out, D_out]     projection           │
        │                             ▼                                   │
        │  score = sum_d(u_hat · q) / sqrt(D_out)   q [1,1,N_out,D_out]   │
        │          [B, N_in, N_out]  (a DIVIDE, not a reciprocal mult)    │
        │                             ▼                                   │
        │  optional top_k: where(score >= kth, score, -1e9), taken on     │
        │  the SAME axis the softmax reduces — so no row is all-masked    │
        │                             ▼                                   │
        │  a = softmax(score, axis=2 'output' | axis=1 'input')           │
        │                             ▼                                   │
        │  s = sum_i (a · u_hat)  (+ bias)           [B, N_out, D_out]    │
        │                             ▼                                   │
        │  decoupled output — length and pose are learned separately:     │
        │    direction = s / (||s|| + eps)           unit pose vector     │
        │    mag       = sigmoid(prob_head(s))       Dense(1) over D_out  │
        │    v         = mag * direction             ||v|| ∈ (0, 1)       │
        │                             ▼                                   │
        │  Output v [B, N_out, D_out]                                     │
        │                                                                 │
        │  No heads, no residual, no mask argument.  When training and    │
        │  use_load_balancing: add_loss(weight · cv²(usage)) as well.     │
        └─────────────────────────────────────────────────────────────────┘

    :param num_capsules: Number of output capsules ``N_out``. Must be positive.
    :type num_capsules: int
    :param dim_capsules: Dimension ``D_out`` of each output capsule vector. Must be
        positive. Also sets the softmax temperature ``1 / sqrt(D_out)``.
    :type dim_capsules: int
    :param softmax_axis: Either ``"output"`` (each input capsule competes for
        output assignment — matches dynamic-routing semantics; softmax over axis 2)
        or ``"input"`` (each output capsule receives a normalised mixture of
        inputs — like classic attention; softmax over axis 1). Defaults to
        ``"output"``.
    :type softmax_axis: Literal["output", "input"]
    :param top_k: If set, restrict each softmax row to the ``top_k`` largest scores
        along the soft-maxed axis; the remaining entries are replaced with a large
        negative constant before the softmax. Clamped to the axis size at call
        time. ``None`` (default) disables masking entirely.
    :type top_k: Optional[int]
    :param use_bias: Whether to add a learned bias of shape
        ``(1, N_out, D_out)`` to the routing aggregate ``s`` before the output is
        computed. Defaults to ``True``.
    :type use_bias: bool
    :param use_load_balancing: Whether to attach an auxiliary importance loss on
        the routing assignments via ``self.add_loss``. The loss is added **only
        when ``training`` is truthy** — it is absent from inference graphs.
        Defaults to ``False``.
    :type use_load_balancing: bool
    :param load_balancing_weight: Scalar multiplier on the auxiliary loss when
        ``use_load_balancing=True``. Must be non-negative. Defaults to ``0.01``.
    :type load_balancing_weight: float
    :param eps: Numerical-stability constant. Added under the square root of the
        direction normalization and to both denominators of the load-balancing
        coefficient of variation. Defaults to ``1e-7``.
    :type eps: float
    :param kernel_initializer: Initializer for the transformation tensor ``W``,
        the per-output query ``q``, AND the internal ``prob_head`` Dense kernel.
        Accepts a string spec or an ``Initializer``; normalized through
        ``keras.initializers.get``. Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer applied to ``W`` and ``q``.
        Accepts a string spec (e.g. ``"l2"``), a serialized dict, or a
        ``Regularizer``; normalized via ``get``. Defaults to ``None``.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to the ``Layer`` base
        class (``name``, ``dtype``, ...).
    :type kwargs: Any

    :raises ValueError: If ``num_capsules`` is not positive.
    :raises ValueError: If ``dim_capsules`` is not positive.
    :raises ValueError: If ``softmax_axis`` is not ``"output"`` or ``"input"``.
    :raises ValueError: If ``top_k`` is given and not positive.
    :raises ValueError: If ``load_balancing_weight`` is negative.
    :raises ValueError: From ``build()``, if the input is not 3D, or if the
        number of input capsules is not statically known.
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        softmax_axis: Literal["output", "input"] = "output",
        top_k: Optional[int] = None,
        use_bias: bool = True,
        use_load_balancing: bool = False,
        load_balancing_weight: float = 0.01,
        eps: float = 1e-7,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_capsules <= 0:
            raise ValueError(f"num_capsules must be positive, got {num_capsules}")
        if dim_capsules <= 0:
            raise ValueError(f"dim_capsules must be positive, got {dim_capsules}")
        if softmax_axis not in ("output", "input"):
            raise ValueError(
                f"softmax_axis must be 'output' or 'input', got {softmax_axis!r}"
            )
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive or None, got {top_k}")
        if not (0.0 <= load_balancing_weight):
            raise ValueError(
                f"load_balancing_weight must be non-negative, got {load_balancing_weight}"
            )

        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.softmax_axis = softmax_axis
        self.top_k = top_k
        self.use_bias = use_bias
        self.use_load_balancing = use_load_balancing
        self.load_balancing_weight = float(load_balancing_weight)
        self.eps = float(eps)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Per-capsule scalar probability head:
        # maps a D_out-dim capsule vector to a scalar probability via Dense(1).
        self.prob_head = keras.layers.Dense(
            units=1,
            activation=None,  # raw logit; sigmoid applied in call()
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            name="prob_head",
        )

        # Set in build()
        self.W: Optional[keras.Variable] = None
        self.q: Optional[keras.Variable] = None
        self.bias: Optional[keras.Variable] = None
        self.num_input_capsules: Optional[int] = None
        self.input_dim_capsules: Optional[int] = None

    # ------------------------------------------------------------------
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create ``W``, ``q``, the optional bias, and build the probability head.

        :param input_shape: Shape of the input capsule tensor, expected as
            ``(batch, num_input_capsules, input_dim_capsules)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If ``input_shape`` is not rank 3.
        :raises ValueError: If ``num_input_capsules`` (axis 1) is ``None``. It is
            needed statically because ``W``'s first axis is per-input-capsule.

        :return: ``None``.
        :rtype: None
        """
        # DECISION plan-2026-07-27T130643-38c5646a/D-013
        # This guard, and its twin in CapsuleBlockV2.build below, are the ONE
        # executable change plan-2026-07-27T130643-38c5646a made to a forward-adjacent
        # path. That plan's invariant 1 is "identical numerics, no executable change";
        # these two `if self.built: return` lines are its single, deliberate carve-out.
        #
        # WHY IT IS SAFE: on the ONLY path any test or production caller exercises —
        # one build() per layer instance — the guard is unreachable, because
        # `self.built` is False on entry and `super().build()` sets it on the way out.
        # It can only fire on a SECOND build(), which today does the wrong thing: it
        # re-enters the three add_weight() calls below (W, q, bias) and orphans the
        # variables Keras has already restored into. So the guard changes behavior
        # only in the case that was already broken.
        #
        # WHY IT WAS ADDED AT ALL: this file predates the package-wide sweep in commit
        # 1cdd4767, which added exactly this guard to 29 other build() methods. It was
        # the only attention module still missing it (the plan predicted the gap was in
        # `ideogram4_attention.py` — that prediction was wrong; ideogram4 already had
        # it).
        #
        # WHAT NOT TO DO: do not remove it to "restore byte-identity" with the
        # pre-plan file. The correct record is that the exception is charged in
        # decisions.md D-013, not that the guard is reverted.
        #
        # NOT COVERED BY ANY TEST, in either direction: the capsnet suite only ever
        # builds once. See decisions.md D-013.
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(
                "Expected 3D input shape [batch, num_input_capsules, "
                f"input_dim_capsules], got {input_shape}"
            )

        self.num_input_capsules = input_shape[1]
        self.input_dim_capsules = input_shape[2]

        if self.num_input_capsules is None:
            raise ValueError(
                "AttentionRoutingCapsule requires a known num_input_capsules "
                "dimension at build time."
            )

        # Transformation weights W: shape (N_in, N_out, D_out, D_in).
        # Used via einsum for robust graph-mode behavior (the legacy
        # matmul+squeeze pattern can lose static shape info under
        # tf.function tracing).
        self.W = self.add_weight(
            shape=(
                self.num_input_capsules,
                self.num_capsules,
                self.dim_capsules,
                self.input_dim_capsules,
            ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="capsule_transformation_weights",
        )

        # Learned query per output capsule.
        # Shape chosen so the dot product u_hat · q is broadcastable.
        self.q = self.add_weight(
            shape=(1, 1, self.num_capsules, self.dim_capsules),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="capsule_routing_query",
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(1, self.num_capsules, self.dim_capsules),
                initializer="zeros",
                trainable=True,
                name="capsule_bias",
            )

        self.prob_head.build((None, self.num_capsules, self.dim_capsules))

        logger.info(
            f"Built AttentionRoutingCapsule: {self.num_input_capsules} -> "
            f"{self.num_capsules} capsules, softmax_axis={self.softmax_axis}, "
            f"top_k={self.top_k}, use_load_balancing={self.use_load_balancing}"
        )

        super().build(input_shape)

    # ------------------------------------------------------------------
    def _apply_top_k_mask(
        self,
        score: keras.KerasTensor,
        axis: int,
    ) -> keras.KerasTensor:
        """Mask all but the top-k entries of ``score`` along ``axis``.

        ``keras.ops.top_k`` only operates on the last axis, so scoring along axis 1
        is done by transposing to last, thresholding, and transposing the boolean
        keep-mask back — the score tensor itself is never permuted in place.

        :param score: Routing scores of shape ``(B, N_in, N_out)``.
        :type score: keras.KerasTensor
        :param axis: Axis to select the top-k along; ``1`` (input capsules) or
            ``2`` (output capsules). Matches the softmax axis.
        :type axis: int

        :return: ``score`` with every non-top-k entry replaced by a large negative
            constant, same shape and dtype as the input.
        :rtype: keras.KerasTensor
        """
        k = int(self.top_k)
        # Clamp to the axis size — the static shape is known after build.
        if axis == 1:
            axis_size = self.num_input_capsules
        else:
            axis_size = self.num_capsules
        if axis_size is not None:
            k = min(k, axis_size)

        if axis == 1:
            # Move axis 1 to last for keras.ops.top_k.
            score_t = keras.ops.transpose(score, (0, 2, 1))  # (B, N_out, N_in)
        else:
            score_t = score  # (B, N_in, N_out)
        # keras.ops.top_k operates on the last axis.
        topk_values, _ = keras.ops.top_k(score_t, k=k)
        # The k-th value (smallest of the top-k) is the threshold per row.
        threshold = topk_values[..., -1:]
        keep = score_t >= threshold
        if axis == 1:
            keep = keras.ops.transpose(keep, (0, 2, 1))
        # Use a large negative number so post-softmax weight ≈ 0.
        #
        # fp16 note (rubric R13; measured, not argued): under
        # `mixed_precision.set_global_policy('mixed_float16')` this cast produces
        # -inf, because np.float16(-1e9) == -inf. That is SAFE HERE, and only
        # because of the `ops.where` form on the next line combined with the
        # top-k construction: `keep` is true for at least k >= 1 entries of every
        # row along `axis`, and the softmax that consumes this tensor runs along
        # THAT SAME axis — so no softmax row is all -inf, and no `0 * -inf`
        # product exists anywhere. Probed under mixed_float16 at
        # (top_k=2, axis=output), (top_k=2, axis=input) and the worst case
        # (top_k=1, axis=output): 0 NaN out of 96 outputs each time.
        #
        # WHAT NOT TO DO: do not rewrite this as the arithmetic form
        # `score + (1 - keep) * -1e9`. That form is the one that DOES produce
        # `0 * -inf = NaN` in fp16. It WAS a live defect at ten sites in this
        # package; all ten now call `common.apply_attention_mask` instead.
        neg_inf = keras.ops.cast(-1e9, score.dtype)
        masked = keras.ops.where(keep, score, neg_inf)
        return masked

    # ------------------------------------------------------------------
    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Route input capsules to output capsules in a single forward pass.

        :param inputs: Input capsule tensor of shape ``(B, N_in, D_in)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag. Truthy enables the optional
            load-balancing auxiliary loss; it has no other effect (this layer has
            no dropout and no normalization statistics).
        :type training: Optional[bool]

        :return: Output capsules of shape ``(B, N_out, D_out)`` whose per-capsule
            norm lies in ``(0, 1)`` by construction.
        :rtype: keras.KerasTensor
        """
        # inputs: (B, N_in, D_in)
        # u_hat[b, i, o, d] = sum_e W[i, o, d, e] * inputs[b, i, e]
        # einsum keeps static shapes intact under tf.function tracing.
        u_hat = keras.ops.einsum("iode,bie->biod", self.W, inputs)  # (B, N_in, N_out, D_out)

        # Score: (u_hat · q) / sqrt(D_out)
        # u_hat: (B, N_in, N_out, D_out); q: (1, 1, N_out, D_out)
        score = keras.ops.sum(u_hat * self.q, axis=-1)  # (B, N_in, N_out)
        # R13: deliberately NOT `common.compute_attention_scale`. This site
        # DIVIDES by sqrt(D_out); the helper returns the reciprocal 1/sqrt(D_out)
        # for call sites that MULTIPLY. Swapping in the helper would change a
        # divide into a multiply — a real (if tiny) numerics change, forbidden by
        # this pass. Measured across 27 dims: `float(x) ** 0.5` differs from the
        # helper's value for 26 of them, as it must (it is the reciprocal).
        score = score / float(self.dim_capsules) ** 0.5

        # Optional Top-K masking before softmax.
        if self.top_k is not None:
            if self.softmax_axis == "output":
                score = self._apply_top_k_mask(score, axis=2)
            else:
                score = self._apply_top_k_mask(score, axis=1)

        # Softmax along the chosen axis.
        if self.softmax_axis == "output":
            a = keras.activations.softmax(score, axis=2)  # competition over outputs
        else:
            a = keras.activations.softmax(score, axis=1)  # mixture over inputs

        # Aggregate: s_j = sum_i (a_ij * u_hat_ij)
        # a: (B, N_in, N_out) -> expand to (B, N_in, N_out, 1) for broadcast.
        a_exp = keras.ops.expand_dims(a, axis=-1)
        s = keras.ops.sum(a_exp * u_hat, axis=1)  # (B, N_out, D_out)

        if self.use_bias and self.bias is not None:
            s = s + self.bias

        # Decoupled output: magnitude (sigmoid head) * unit direction.
        # Direction.
        s_norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(s), axis=-1, keepdims=True) + self.eps)
        direction = s / s_norm  # (B, N_out, D_out)

        # Magnitude: sigmoid(prob_head(s)) — scalar per capsule.
        mag_logit = self.prob_head(s)  # (B, N_out, 1)
        mag = keras.activations.sigmoid(mag_logit)

        v = mag * direction  # (B, N_out, D_out), ||v|| ∈ (0, 1) by construction

        # Optional load-balancing auxiliary loss (training only).
        if self.use_load_balancing and training:
            # Importance: mean assignment per output capsule, averaged over
            # batch and input capsules. Penalize variance to encourage
            # uniform usage. Mirrors Shazeer et al. (2017) "importance loss".
            if self.softmax_axis == "output":
                # a normalised over output axis -> usage = mean over (B, N_in)
                usage = keras.ops.mean(a, axis=(0, 1))  # (N_out,)
            else:
                # a normalised over input axis -> usage = mean over (B, N_out)
                usage = keras.ops.mean(a, axis=(0, 2))  # (N_in,)
            # Coefficient of variation squared: cv^2 = var / mean^2.
            mean_u = keras.ops.mean(usage) + self.eps
            var_u = keras.ops.mean(keras.ops.square(usage - mean_u))
            aux = var_u / (keras.ops.square(mean_u) + self.eps)
            self.add_loss(self.load_balancing_weight * aux)

        return v

    # ------------------------------------------------------------------
    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config only.

        :param input_shape: Input shape ``(batch, N_in, D_in)``. Only the batch
            entry is used; the capsule axes come from the constructor arguments.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``(batch, num_capsules, dim_capsules)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], self.num_capsules, self.dim_capsules)

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary containing every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "num_capsules": self.num_capsules,
                "dim_capsules": self.dim_capsules,
                "softmax_axis": self.softmax_axis,
                "top_k": self.top_k,
                "use_bias": self.use_bias,
                "use_load_balancing": self.use_load_balancing,
                "load_balancing_weight": self.load_balancing_weight,
                "eps": self.eps,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CapsuleBlockV2(keras.layers.Layer):
    """Capsule block built on :class:`AttentionRoutingCapsule`.

    Wraps :class:`AttentionRoutingCapsule` with optional dropout and an
    optional length-preserving direction-only LayerNormalization. The
    direction-only normalizer matches the bug-fixed behavior of the legacy
    :class:`CapsuleBlock` — it normalizes the unit-direction subspace
    without rescaling capsule magnitudes (which encode detection
    probability).

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────────────┐
        │    CapsuleBlockV2 — routing + dropout + direction-only norm     │
        │                                                                 │
        │  Input u [B, N_in, D_in]                                        │
        │                             ▼                                   │
        │  AttentionRoutingCapsule ('block_routing')  [B, N_out, D_out]   │
        │  — the box above; every routing argument is forwarded to it     │
        │                             ▼                                   │
        │  Dropout        only if dropout_rate > 0 (else no sub-layer)    │
        │                             ▼                                   │
        │  direction-only LayerNormalization, only if the flag is set.    │
        │  This split/recombine is what keeps the block LENGTH-           │
        │  PRESERVING:                                                    │
        │    mag = ||x||                     magnitude held aside         │
        │    dir = x / mag                                                │
        │    dir = LayerNorm(dir)            pose normalized ...          │
        │    dir = dir / ||dir||             ... then re-unit-ized        │
        │    x   = mag * dir                 magnitude restored           │
        │                             ▼                                   │
        │  Output v [B, N_out, D_out]                                     │
        │                                                                 │
        │  A plain LayerNorm on x would rescale ||x|| and destroy the     │
        │  detection probability the routing capsule just encoded in it.  │
        └─────────────────────────────────────────────────────────────────┘

    :param num_capsules: Number of output capsules. Forwarded to
        :class:`AttentionRoutingCapsule`, which validates it.
    :type num_capsules: int
    :param dim_capsules: Dimension of each output capsule vector. Forwarded to
        :class:`AttentionRoutingCapsule`, which validates it.
    :type dim_capsules: int
    :param dropout_rate: Dropout applied to the routed capsules. Must be in
        ``[0.0, 1.0)``. A value of ``0.0`` creates no Dropout sub-layer at all.
        Defaults to ``0.0``.
    :type dropout_rate: float
    :param direction_only_norm: Whether to apply the length-preserving
        direction-only ``LayerNormalization`` shown above. ``False`` creates no
        normalization sub-layer. Must be a ``bool``. Defaults to ``False``.
    :type direction_only_norm: bool
    :param softmax_axis: Routing softmax axis, forwarded verbatim to
        :class:`AttentionRoutingCapsule`. Defaults to ``"output"``.
    :type softmax_axis: Literal["output", "input"]
    :param top_k: Top-k routing sparsity, forwarded verbatim to
        :class:`AttentionRoutingCapsule`. Defaults to ``None``.
    :type top_k: Optional[int]
    :param use_bias: Whether the routing capsule adds a learned bias to its
        aggregate. Forwarded verbatim. Defaults to ``True``.
    :type use_bias: bool
    :param use_load_balancing: Whether the routing capsule attaches its auxiliary
        importance loss during training. Forwarded verbatim. Defaults to
        ``False``.
    :type use_load_balancing: bool
    :param load_balancing_weight: Weight on that auxiliary loss. Forwarded
        verbatim. Defaults to ``0.01``.
    :type load_balancing_weight: float
    :param eps: Numerical-stability constant. Forwarded to the routing capsule
        AND used locally under both square roots of the direction-only
        normalizer. Defaults to ``1e-7``.
    :type eps: float
    :param kernel_initializer: Initializer for the routing capsule's ``W`` and
        ``q``. Resolved here via ``keras.initializers.get`` and the resolved
        object is what is forwarded. Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer forwarded to the routing
        capsule. Resolved here via ``keras.regularizers.get`` and the resolved
        object is what is forwarded. Defaults to ``None``.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to the ``Layer`` base
        class (``name``, ``dtype``, ...).
    :type kwargs: Any

    :raises ValueError: If ``dropout_rate`` is not in ``[0.0, 1.0)``.
    :raises TypeError: If ``direction_only_norm`` is not a ``bool``.
    :raises ValueError: From the wrapped :class:`AttentionRoutingCapsule`, for any
        invalid routing argument (see its ``:raises:`` list).
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        dropout_rate: float = 0.0,
        direction_only_norm: bool = False,
        softmax_axis: Literal["output", "input"] = "output",
        top_k: Optional[int] = None,
        use_bias: bool = True,
        use_load_balancing: bool = False,
        load_balancing_weight: float = 0.01,
        eps: float = 1e-7,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")
        if not isinstance(direction_only_norm, bool):
            raise TypeError(
                f"direction_only_norm must be boolean, got {type(direction_only_norm)}"
            )

        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.dropout_rate = float(dropout_rate)
        self.direction_only_norm = direction_only_norm
        self.softmax_axis = softmax_axis
        self.top_k = top_k
        self.use_bias = use_bias
        self.use_load_balancing = use_load_balancing
        self.load_balancing_weight = float(load_balancing_weight)
        self.eps = float(eps)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        # Resolved here too, matching AttentionRoutingCapsule above: this class
        # has its OWN get_config() entry (`keras.regularizers.serialize`), so
        # storing the raw argument would serialize a string spec as a bare string
        # and a reloaded dict as a dict, neither of which is a Regularizer.
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.routing = AttentionRoutingCapsule(
            num_capsules=self.num_capsules,
            dim_capsules=self.dim_capsules,
            softmax_axis=self.softmax_axis,
            top_k=self.top_k,
            use_bias=self.use_bias,
            use_load_balancing=self.use_load_balancing,
            load_balancing_weight=self.load_balancing_weight,
            eps=self.eps,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="block_routing",
        )

        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(self.dropout_rate, name="block_dropout")
        else:
            self.dropout = None

        if self.direction_only_norm:
            self.layer_norm = keras.layers.LayerNormalization(axis=-1, name="block_dir_norm")
        else:
            self.layer_norm = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the routing capsule and, if configured, dropout and the norm.

        Sub-layers are built explicitly and in computational order so that every
        weight variable exists before Keras restores a checkpoint into them.

        :param input_shape: Shape of the input capsule tensor,
            ``(batch, N_in, D_in)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: Propagated from the wrapped
            :class:`AttentionRoutingCapsule` if the input is not rank 3 or the
            input-capsule count is not static.

        :return: ``None``.
        :rtype: None
        """
        # DECISION plan-2026-07-27T130643-38c5646a/D-013 (second of two)
        # Idempotency guard (rubric R7). See the full rationale at the identical guard
        # in AttentionRoutingCapsule.build above — a second build() here would re-enter
        # the sub-layer builds after weight restoration. Same invariant-1 carve-out,
        # same lack of test coverage in either direction.
        if self.built:
            return

        self.routing.build(input_shape)
        out_shape = self.routing.compute_output_shape(input_shape)
        if self.dropout is not None:
            self.dropout.build(out_shape)
        if self.layer_norm is not None:
            self.layer_norm.build(out_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Route, optionally drop out, optionally normalize the pose direction.

        :param inputs: Input capsule tensor of shape ``(B, N_in, D_in)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to the routing capsule,
            the dropout layer and the normalizer.
        :type training: Optional[bool]

        :return: Output capsules of shape ``(B, N_out, D_out)``.
        :rtype: keras.KerasTensor
        """
        x = self.routing(inputs, training=training)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if self.layer_norm is not None:
            # Length-preserving direction-only LN — same recipe as the
            # bug-fixed legacy CapsuleBlock.
            mag = keras.ops.sqrt(keras.ops.sum(keras.ops.square(x), axis=-1, keepdims=True) + self.eps)
            direction = x / mag
            direction_normed = self.layer_norm(direction, training=training)
            dir_mag = keras.ops.sqrt(
                keras.ops.sum(keras.ops.square(direction_normed), axis=-1, keepdims=True) + self.eps
            )
            direction_unit = direction_normed / dir_mag
            x = mag * direction_unit

        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Delegate to the wrapped routing capsule.

        Neither dropout nor the direction-only normalizer changes the shape, so
        the block's output shape is exactly the routing capsule's.

        :param input_shape: Input shape ``(batch, N_in, D_in)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``(batch, num_capsules, dim_capsules)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return self.routing.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary containing every ``__init__`` argument, including the
            routing arguments forwarded to :class:`AttentionRoutingCapsule`.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "num_capsules": self.num_capsules,
                "dim_capsules": self.dim_capsules,
                "dropout_rate": self.dropout_rate,
                "direction_only_norm": self.direction_only_norm,
                "softmax_axis": self.softmax_axis,
                "top_k": self.top_k,
                "use_bias": self.use_bias,
                "use_load_balancing": self.use_load_balancing,
                "load_balancing_weight": self.load_balancing_weight,
                "eps": self.eps,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config


# ---------------------------------------------------------------------
