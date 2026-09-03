"""
Attention-based capsule routing, in :class:`AttentionRoutingCapsule` and the
:class:`CapsuleBlockV2` wrapper that adds optional dropout and normalization.

Classic dynamic routing runs an iterative agreement loop
(``b_{i+1} = b_i + agreement(v_i)``), which is sequential and cannot be fused.
This module replaces the loop with one learned query per output capsule and a
single softmax over agreement scores: ``score = <u_hat, q> / sqrt(D_out)``,
softmax, aggregate — one parallel pass. It also decouples what a capsule's
magnitude means: instead of squashing a vector's length into a probability
(which ties pose and confidence together and saturates near zero), the output
is ``v = sigmoid(prob_head(s)) * s / ||s||`` — direction carries pose,
magnitude is a learned scalar read as detection probability, and ``||v||``
still lies in ``(0, 1)`` for margin loss. Two optional mechanisms, top-k
routing sparsity and a load-balancing auxiliary loss, are ported from
sparsely-gated mixture-of-experts routing rather than from capsule literature.

``CapsuleBlockV2``'s optional normalizer is length-preserving: a plain
``LayerNormalization`` would rescale the output magnitude and destroy the
detection probability just encoded in it, so the block splits magnitude and
direction, normalizes only the direction, and restores the magnitude. It is
consumed by ``models/vision/capsnet/model_v2.py``.

Core arithmetic::

    u_hat[b,i,o,:] = W[i,o,:,:] @ u[b,i,:]
    score[b,i,o]   = <u_hat[b,i,o,:], q[o,:]> / sqrt(D_out)
    a              = softmax(score, axis = output | input)
    s[b,o,:]       = sum_i a[b,i,o] * u_hat[b,i,o,:]  (+ bias)
    v[b,o,:]       = sigmoid(prob_head(s)) * s / sqrt(||s||^2 + eps)

References:
    - Sabour et al., 2017. Dynamic Routing Between Capsules. NeurIPS 30.
      (https://arxiv.org/abs/1710.09829)
    - Hinton et al., 2018. Matrix Capsules with EM Routing. ICLR.
    - Hahn et al., 2019. Self-Routing Capsule Networks. NeurIPS 32.
    - Tsai et al., 2020. Capsules with Inverted Dot-Product Attention Routing.
      ICLR. (https://arxiv.org/abs/2002.04764)
    - Shazeer et al., 2017. Outrageously Large Neural Networks: The
      Sparsely-Gated Mixture-of-Experts Layer. (the importance / load-balancing
      loss) (https://arxiv.org/abs/1701.06538)
    - Vaswani et al., 2017. Attention Is All You Need. (the scaled dot-product
      form the routing score borrows) (https://arxiv.org/abs/1706.03762)
"""

import keras
from typing import Optional, Tuple, Union, Dict, Any, Literal

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.attention.attention_routing_capsule")
class AttentionRoutingCapsule(keras.layers.Layer):
    """
    Single-step attention-routing capsule layer.

    A drop-in replacement for the iterative :class:`RoutingCapsule`. A learned
    per-output query scores the prediction vectors ``u_hat``, one softmax turns
    those scores into routing weights, and the aggregate is emitted with its
    magnitude and direction produced separately: direction is the unit pose
    vector, magnitude is ``sigmoid`` of a learned scalar head, so ``||v||`` lies
    in ``(0, 1)`` without ``squash``'s saturation at zero.

    Everything is one forward pass. There is no inner loop and no sequential
    dependency, so the layer is fully parallel and XLA-fusable.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────────────────────────────┐
        │  Input u [B, N_in, D_in]                                     │
        │    N_in must be static: W is indexed per input capsule, and  │
        │    build() raises if it is None.                             │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  u_hat = einsum('iode,bie->biod', W, u)                      │
        │    W     [N_in, N_out, D_out, D_in]   a pose transform,      │
        │    u_hat [B, N_in, N_out, D_out]      not a Q/K/V projection │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  score = sum_d(u_hat · q) / sqrt(D_out)                      │
        │    q [1, 1, N_out, D_out];  [B, N_in, N_out]                 │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  optional top_k: where(score >= kth, score, -1e9)            │
        │    taken on the same axis the softmax reduces, so no row     │
        │    ends up all-masked                                        │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  a = softmax(score, axis=2 'output' | axis=1 'input')        │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  s = sum_i (a · u_hat)  (+ bias)          [B, N_out, D_out]  │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  decoupled output — length and pose learned separately:      │
        │    direction = s / sqrt(||s||^2 + eps)  unit pose vector     │
        │    mag       = sigmoid(prob_head(s))  Dense(1) over D_out    │
        │    v         = mag * direction        ||v|| in (0, 1)        │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Output v [B, N_out, D_out]                                  │
        └──────────────────────────────────────────────────────────────┘

        No heads, no residual, no mask argument. When training and
        use_load_balancing: add_loss(weight * cv^2(usage)) as well.

    Softmax axis:

    .. code-block:: text

        'output'  (the default)
            softmax over axis 2. Each input capsule competes for where
            to send itself. These are dynamic-routing semantics.

        'input'
            softmax over axis 1. Each output capsule receives a
            normalised mixture of inputs. This is classic attention.

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
        the routing assignments via ``self.add_loss``. The loss is added only
        when ``training`` is truthy; it is absent from inference graphs.
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

    Input shape:
        3D tensor with shape ``(batch_size, num_input_capsules,
        input_dim_capsules)``. The capsule axis must be statically known.

    Output shape:
        3D tensor with shape ``(batch_size, num_capsules, dim_capsules)``. One
        output mode only; every capsule's norm lies in ``(0, 1)``.

    Example:
        >>> # Classifier-style routing to 10 output capsules
        >>> caps = AttentionRoutingCapsule(num_capsules=10, dim_capsules=16)
        >>> u = keras.random.normal((4, 1152, 8))
        >>> v = caps(u)                       # (4, 10, 16)
        >>>
        >>> # Sparse routing with the load-balancing loss during training
        >>> caps = AttentionRoutingCapsule(
        ...     num_capsules=32, dim_capsules=16, top_k=8,
        ...     use_load_balancing=True, load_balancing_weight=0.01,
        ... )
        >>>
        >>> # Attention semantics instead of dynamic-routing semantics
        >>> caps = AttentionRoutingCapsule(10, 16, softmax_axis="input")

    Note:
        ``||v||`` is a detection probability and ``v / ||v||`` is a pose. That
        split comes out of the arithmetic here, not out of a convention.
        Anything downstream that rescales the capsule vector destroys the first
        while preserving the second. A plain ``LayerNormalization`` does exactly
        that. Use :class:`CapsuleBlockV2` for the length-preserving alternative.

    Attributes:
        W: Pose transformation tensor, ``(N_in, N_out, D_out, D_in)``.
        q: Learned per-output-capsule query, ``(1, 1, N_out, D_out)``.
        bias: Aggregate bias ``(1, N_out, D_out)``, or ``None``.
        prob_head: ``Dense(1)`` producing the magnitude logit per capsule.
        num_input_capsules: ``N_in``, resolved in ``build``.
        input_dim_capsules: ``D_in``, resolved in ``build``.
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
        """Validate the routing configuration and create the probability head.

        ``W``, ``q`` and the optional bias depend on the input capsule count and
        are therefore created in :meth:`build`, not here. See the class docstring
        for the parameter reference.
        """
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
        # activation=None emits a raw logit; call() applies the sigmoid.
        self.prob_head = keras.layers.Dense(
            units=1,
            activation=None,
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
            # Move axis 1 to last for keras.ops.top_k. Result: (B, N_out, N_in).
            score_t = keras.ops.transpose(score, (0, 2, 1))
        else:
            # Already (B, N_in, N_out), which top_k can consume directly.
            score_t = score
        # keras.ops.top_k operates on the last axis.
        topk_values, _ = keras.ops.top_k(score_t, k=k)
        # The k-th value (smallest of the top-k) is the threshold per row.
        threshold = topk_values[..., -1:]
        keep = score_t >= threshold
        if axis == 1:
            keep = keras.ops.transpose(keep, (0, 2, 1))
        # Safe under fp16 because top-k guarantees k >= 1 kept entries per row.
        # Never rewrite as `score + (1 - keep) * -1e9` (produces NaN in fp16).
        neg_inf = keras.ops.cast(-1e9, score.dtype)
        masked = keras.ops.where(keep, score, neg_inf)
        return masked


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

        :return: Output capsules of shape ``(B, N_out, D_out)``. The sigmoid
            magnitude head puts every per-capsule norm in ``(0, 1)``.
        :rtype: keras.KerasTensor
        """
        # inputs: (B, N_in, D_in)
        # u_hat[b, i, o, d] = sum_e W[i, o, d, e] * inputs[b, i, e]
        # einsum keeps static shapes intact under tf.function tracing.
        # Result: (B, N_in, N_out, D_out).
        u_hat = keras.ops.einsum("iode,bie->biod", self.W, inputs)

        # Score: (u_hat · q) / sqrt(D_out)
        # u_hat: (B, N_in, N_out, D_out); q: (1, 1, N_out, D_out).
        # Result: (B, N_in, N_out).
        score = keras.ops.sum(u_hat * self.q, axis=-1)
        # Divides by sqrt(D_out); do not swap in common.compute_attention_scale,
        # which returns the reciprocal for call sites that multiply.
        score = score / float(self.dim_capsules) ** 0.5

        # Optional top-k masking before softmax.
        if self.top_k is not None:
            if self.softmax_axis == "output":
                score = self._apply_top_k_mask(score, axis=2)
            else:
                score = self._apply_top_k_mask(score, axis=1)

        # Softmax along the chosen axis.
        if self.softmax_axis == "output":
            # Axis 2: the input capsules compete over output capsules.
            a = keras.activations.softmax(score, axis=2)
        else:
            # Axis 1: each output capsule gets a mixture over input capsules.
            a = keras.activations.softmax(score, axis=1)

        # Aggregate: s_j = sum_i (a_ij * u_hat_ij)
        # a: (B, N_in, N_out) -> expand to (B, N_in, N_out, 1) for broadcast.
        # s: (B, N_out, D_out).
        a_exp = keras.ops.expand_dims(a, axis=-1)
        s = keras.ops.sum(a_exp * u_hat, axis=1)

        if self.use_bias and self.bias is not None:
            s = s + self.bias

        # Decoupled output: magnitude (sigmoid head) * unit direction.
        # Direction, shape (B, N_out, D_out).
        s_norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(s), axis=-1, keepdims=True) + self.eps)
        direction = s / s_norm

        # Magnitude: sigmoid(prob_head(s)), one scalar per capsule.
        # mag_logit is (B, N_out, 1).
        mag_logit = self.prob_head(s)
        mag = keras.activations.sigmoid(mag_logit)

        # v is (B, N_out, D_out). The sigmoid puts ||v|| in (0, 1).
        v = mag * direction

        # Optional load-balancing auxiliary loss (training only).
        if self.use_load_balancing and training:
            # Importance: mean assignment per output capsule, averaged over
            # batch and input capsules. Penalize variance to encourage
            # uniform usage. Mirrors Shazeer et al. (2017) "importance loss".
            if self.softmax_axis == "output":
                # a is normalised over the output axis, so usage is the mean
                # over (B, N_in) and has shape (N_out,).
                usage = keras.ops.mean(a, axis=(0, 1))
            else:
                # a is normalised over the input axis, so usage is the mean
                # over (B, N_out) and has shape (N_in,).
                usage = keras.ops.mean(a, axis=(0, 2))
            # Coefficient of variation squared: cv^2 = var / mean^2.
            mean_u = keras.ops.mean(usage) + self.eps
            var_u = keras.ops.mean(keras.ops.square(usage - mean_u))
            aux = var_u / (keras.ops.square(mean_u) + self.eps)
            self.add_loss(self.load_balancing_weight * aux)

        return v


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




@register_dl_technique("dl_techniques.layers.attention.attention_routing_capsule")
class CapsuleBlockV2(keras.layers.Layer):
    """
    Capsule block: :class:`AttentionRoutingCapsule` plus optional dropout and norm.

    Wraps the routing capsule with optional dropout and an optional
    length-preserving direction-only ``LayerNormalization``, and forwards every
    routing argument through to the capsule it owns, so the two classes share one
    parameter vocabulary.

    The normalizer matches the bug-fixed behaviour of the legacy
    :class:`CapsuleBlock`: it normalizes the unit-direction subspace without
    rescaling capsule magnitudes, since those magnitudes are the detection
    probabilities the routing capsule just encoded.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input u [B, N_in, D_in]             │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  block_routing: AttentionRoutingCapsule   [B, N_out, D_out]  │
        │    the box above; every routing argument is forwarded to it  │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  block_dropout (only when dropout_rate > 0)                  │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  block_dir_norm (only when direction_only_norm) — the split/ │
        │  recombine keeps the block length-preserving:                │
        │    mag = sqrt(||x||^2 + eps)   magnitude held aside          │
        │    dir = x / mag                                             │
        │    dir = LayerNorm(dir)        pose normalized               │
        │    dir = dir / sqrt(||dir||^2 + eps)   re-unit-ized          │
        │    x   = mag * dir             magnitude restored            │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output v [B, N_out, D_out]          │
        └──────────────────────────────────────┘

        A plain LayerNorm on x would rescale ||x||, destroying the
        detection probability the routing capsule just encoded in it.

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
        and used locally under both square roots of the direction-only
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

    Input shape:
        3D tensor with shape ``(batch_size, num_input_capsules,
        input_dim_capsules)``. The capsule axis must be statically known.

    Output shape:
        3D tensor with shape ``(batch_size, num_capsules, dim_capsules)``.
        Neither dropout nor the direction-only normalizer changes the shape, and
        the normalizer does not change the norm either.

    Example:
        >>> block = CapsuleBlockV2(num_capsules=10, dim_capsules=16,
        ...                        dropout_rate=0.1, direction_only_norm=True)
        >>> u = keras.random.normal((4, 1152, 8))
        >>> v = block(u, training=False)      # (4, 10, 16)
        >>>
        >>> # Bare block: no dropout, no norm — neither sub-layer is created
        >>> block = CapsuleBlockV2(10, 16)

    Note:
        With ``direction_only_norm=True`` the block is length-preserving:
        ``||output|| == ||routing output||`` up to ``eps``, so the detection
        probabilities survive normalization. Don't substitute a plain
        ``LayerNormalization`` here — it discards them silently.

    Attributes:
        routing: The wrapped ``AttentionRoutingCapsule``, named ``block_routing``.
        dropout: The ``Dropout`` sub-layer, or ``None``.
        layer_norm: The ``LayerNormalization`` sub-layer, or ``None``.
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
        """Validate the block's own arguments and create the three sub-layers.

        Routing arguments are not validated here — they are forwarded verbatim and
        validated by :class:`AttentionRoutingCapsule`'s constructor. Dropout and
        the normalizer are created only when enabled, so a disabled feature
        contributes no sub-layer at all. See the class docstring for the parameter
        reference.
        """
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
        # Resolved here, not stored raw, so this class's own get_config() always
        # serializes a Regularizer object rather than a bare string or dict.
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
        # DECISION plan-2026-07-27T130643-38c5646a/D-013: keep this guard, a
        # second build() would re-enter sub-layer builds after checkpoint restore.
        # Twin guard in AttentionRoutingCapsule.build. See decisions.md.
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


