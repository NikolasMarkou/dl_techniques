r"""
Grouped differential attention, in :class:`GroupedDifferentialAttention`:
differential attention with an asymmetric ``G : 1`` split between a large
signal group and a small, shared noise-control group.

Differential attention (Ye et al., 2024) subtracts a second softmax map from
the first so that attention mass both maps place on irrelevant tokens cancels.
It spends exactly half of its heads on the noise map. Grouped differential
attention (Lim et al., 2025) observes that this over-allocates compute to the
noise side and starves the heads that carry semantic dependencies. It splits
``H`` heads into ``G + 1`` groups of ``h = H / (G + 1)`` heads. ``G`` groups are
signal heads, each with its own query and key. The one remaining group holds
the noise-control heads, and it is *shared* by all ``G`` signal groups, the
same way grouped-query attention shares one key/value head across several
query heads. Signal head ``i = g * h + j`` (group ``g``, offset ``j``) reads
noise head ``j`` and value head ``j``::

    A1_i = P(Q1_i K1_i^T s)          i = 0 .. G*h - 1
    A2_j = P(Q2_j K2_j^T s)          j = 0 .. h - 1
    head_i = (A1_i - lambda * A2_j) V_j        j = i mod h

Because both maps multiply the same value head, ``(A1 - lambda A2) V =
A1 V - lambda (A2 V)``, so the noise context ``A2_j V_j`` is computed once for
the ``h`` noise heads and reused by the ``G`` groups. That is the saving the
paper claims: the noise group costs ``h`` attention maps rather than ``G * h``,
and ``V`` has ``h`` heads rather than ``G * h``.

``lambda`` follows Ye et al.: ``exp(lambda_q1 . lambda_k1) - exp(lambda_q2 .
lambda_k2) + lambda_init`` with four learned vectors of width ``head_dim`` and
``lambda_init = 0.8 - 0.6 * exp(-0.3 * layer_idx)`` for a 0-based
``layer_idx``. Each head's output then goes through a shared RMSNorm over
``head_dim`` and is scaled by ``1 - lambda_init``. This differs from
:class:`DifferentialMultiHeadAttention`, which uses one clipped scalar and
omits the per-head norm, so the two layers are not numerically comparable.

References:
    - Lim et al., 2025. Grouped Differential Attention.
      (https://arxiv.org/abs/2510.06949)
    - Ye et al., 2024. Differential Transformer: Amplifying attention to the
      relevant context while canceling noise.
      (https://arxiv.org/abs/2410.05258)
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer
      Models from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Henry et al., 2020. Query-Key Normalization for Transformers. (the optional
      QK-norm hook) (https://arxiv.org/abs/2010.04245)
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers.clone import initializer_spec
from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.layers.norms.factory import create_normalization_layer

from .common import (
    apply_attention_mask,
    compute_attention_scale,
    create_qk_norms,
    dtype_name,
    merge_heads,
    probability_axis,
    split_heads,
    validate_logit_probability_type,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.attention.grouped_differential_attention")
class GroupedDifferentialAttention(keras.layers.Layer):
    """
    Grouped differential attention with ``G`` signal groups sharing one noise group.

    ``num_heads`` heads are split into ``group_ratio + 1`` groups of
    ``h = num_heads / (group_ratio + 1)``. The ``G * h`` signal heads each have
    their own query and key; the ``h`` noise heads and the ``h`` value heads are
    shared across the ``G`` signal groups. Every signal head outputs
    ``(A1 - lambda * A2) V`` and the layer returns the output projection of the
    ``G * h`` signal heads.

    ``call()`` is ``call(inputs, attention_mask=None, layer_idx=0, training=None)``,
    the same signature as :class:`DifferentialMultiHeadAttention`. ``layer_idx``
    is 0-based, selects ``lambda_init`` and is not part of ``get_config()``; a
    stack must pass its own depth on every call.

    Architecture:

    .. code-block:: text

                         inputs  [B, L, D]
                                  │
                                  ▼
            ┌───────────────────────────────────────────────────┐
            │ q1, k1 : Dense → [B, G·h, L, d]   (signal heads)  │
            │ q2, k2 : Dense → [B,   h, L, d]   (noise heads)   │
            │ v      : Dense → [B,   h, L, d]   (shared values) │
            └──────┬──────────────────────────────┬─────────────┘
                   │ A1 = P(q1 k1ᵀ s)             │ A2 = P(q2 k2ᵀ s)
                   │ [B, G·h, L, L]               │ [B, h, L, L]
                   ▼                              ▼
          out1 = A1 @ tile(v, G)          out2 = tile(A2 @ v, G)
                   │                              │
                   ▼                              ▼
            ┌───────────────────────────────────────────────────┐
            │ diff = out1 − lambda · out2        [B, G·h, L, d] │
            │ diff = RMSNorm(diff) · (1 − lambda_init)          │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
            merge heads → proj: Dense(dim) → dropout → [B, L, D]

        tile repeats the whole h-head block G times along the head axis,
        so signal head g·h + j meets noise head j and value head j.

    :param dim: Input and output dimension. Must be positive.
    :type dim: int
    :param num_heads: Total head count ``H`` (signal plus noise). Must be
        divisible by ``group_ratio + 1``.
    :type num_heads: int
    :param head_dim: Width ``d`` of every head. Independent of ``dim``.
    :type head_dim: int
    :param group_ratio: ``G``, the signal-to-noise head ratio ``G : 1``. At least
        1. ``1`` reproduces the balanced 1:1 split. The paper finds 3 best when
        training from scratch and 4 best for progressive continual training, with
        11 clearly worse. Defaults to 3.
    :type group_ratio: int
    :param dropout_rate: Output dropout after the projection. Defaults to 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout on both attention maps. Defaults to 0.0.
    :type attention_dropout_rate: float
    :param probability_type: Normalization applied to the scores of both maps,
        forwarded to :class:`ProbabilityOutput`. Defaults to ``"softmax"``.
    :type probability_type: str
    :param probability_config: Optional strategy arguments for
        :class:`ProbabilityOutput`.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization applied to the per-head queries and
        keys of each map. ``None`` disables QK-norm.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional arguments for the QK-norm layers.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kernel_initializer: Initializer for kernel weights. Defaults to
        ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_initializer: Initializer for bias weights. Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param activity_regularizer: Optional regularizer on the layer output.
    :type activity_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    :type kwargs: Any

    :ivar num_noise_heads: ``h = num_heads / (group_ratio + 1)``.
    :vartype num_noise_heads: int
    :ivar num_signal_heads: ``G * h``, the head count entering the output projection.
    :vartype num_signal_heads: int
    :ivar attn_dropout_1: Dropout on the signal map; created always, run only when
        ``attention_dropout_rate > 0``.
    :vartype attn_dropout_1: keras.layers.Dropout
    :ivar lambda_q1: Learned vector, width ``head_dim``, ``None`` until ``build()``.
    :vartype lambda_q1: Optional[keras.Variable]
    :ivar lambda_k1: Learned vector, width ``head_dim``, ``None`` until ``build()``.
    :vartype lambda_k1: Optional[keras.Variable]
    :ivar lambda_q2: Learned vector, width ``head_dim``, ``None`` until ``build()``.
    :vartype lambda_q2: Optional[keras.Variable]
    :ivar lambda_k2: Learned vector, width ``head_dim``, ``None`` until ``build()``.
    :vartype lambda_k2: Optional[keras.Variable]

    :raises ValueError: If ``dim``, ``num_heads``, ``head_dim`` or ``group_ratio`` is
        not positive, if ``num_heads`` is not divisible by ``group_ratio + 1``, if a
        dropout rate is outside ``[0, 1]``, or if ``probability_type`` is a routing
        or hierarchical variant.
    :raises ValueError: If sub-layer construction fails; the original error is logged
        and re-raised as a ``ValueError``.
    :raises ValueError: From ``build()``, if the input is not 3D or its last
        dimension is not ``dim``.

    Input shape:
        3D tensor ``(batch_size, sequence_length, dim)``. The optional
        ``attention_mask`` may be ``(batch, kv_seq)``, ``(batch, q_seq, kv_seq)`` or
        ``(batch, heads, q_seq, kv_seq)`` in keep-predicate semantics, where
        ``heads`` is 1 or broadcastable to both head counts. One mask serves both maps.

    Output shape:
        3D tensor ``(batch_size, sequence_length, dim)``.

    Example:
        >>> attn = GroupedDifferentialAttention(
        ...     dim=512, num_heads=8, head_dim=64, group_ratio=3)
        >>> x = keras.random.normal((2, 128, 512))
        >>> y = attn(x, training=False)                 # (2, 128, 512)
        >>> y = attn(x, layer_idx=5, training=False)    # depth-dependent lambda_init

    Note:
        Group-Differentiated Growth from the paper (replicating only the signal
        heads when scaling a trained model) is a training procedure and is not part
        of this layer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        group_ratio: int = 3,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sub-layer.

        The four lambda vectors are the only weights the layer owns directly; they
        are created in :meth:`build`. The parameters are documented once, in the
        class docstring.

        :raises ValueError: On an invalid size, rate or ratio, a ``num_heads`` not
            divisible by ``group_ratio + 1``, a routing ``probability_type``, or a
            sub-layer construction failure.
        """
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if group_ratio < 1:
            raise ValueError(f"group_ratio must be at least 1, got {group_ratio}")
        if num_heads % (group_ratio + 1) != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by group_ratio + 1 "
                f"({group_ratio + 1}): the heads split into group_ratio + 1 equal groups"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}"
            )

        validate_logit_probability_type(probability_type, "GroupedDifferentialAttention")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.group_ratio = group_ratio
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # A string or dict spec, never the caller's instance: each Dense resolves its own
        # initializer from it, so no two projections replay one random draw.
        self.kernel_initializer = initializer_spec(kernel_initializer)
        self.bias_initializer = initializer_spec(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # h noise/value heads, G * h signal heads.
        self.num_noise_heads = num_heads // (group_ratio + 1)
        self.num_signal_heads = group_ratio * self.num_noise_heads
        self._noise_dim = self.num_noise_heads * head_dim
        self._signal_dim = self.num_signal_heads * head_dim

        self.scale = compute_attention_scale(self.head_dim)

        try:
            dense_kwargs = {
                "bias_initializer": self.bias_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
            }

            self.q1_dense = keras.layers.Dense(
                self._signal_dim, kernel_initializer=self.kernel_initializer,
                name="q1", **dense_kwargs)
            self.k1_dense = keras.layers.Dense(
                self._signal_dim, kernel_initializer=self.kernel_initializer,
                name="k1", **dense_kwargs)
            self.q2_dense = keras.layers.Dense(
                self._noise_dim, kernel_initializer=self.kernel_initializer,
                name="q2", **dense_kwargs)
            self.k2_dense = keras.layers.Dense(
                self._noise_dim, kernel_initializer=self.kernel_initializer,
                name="k2", **dense_kwargs)
            self.v_dense = keras.layers.Dense(
                self._noise_dim, kernel_initializer=self.kernel_initializer,
                name="v", **dense_kwargs)
            self.proj = keras.layers.Dense(
                self.dim, kernel_initializer=self.kernel_initializer,
                name="proj", **dense_kwargs)

            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name="dropout")

            # Created always, built and run only when the rate is positive.
            self.attn_dropout_1 = keras.layers.Dropout(
                self.attention_dropout_rate, name="attn_dropout_1")
            self.attn_dropout_2 = keras.layers.Dropout(
                self.attention_dropout_rate, name="attn_dropout_2")

            self.attn_prob_1 = ProbabilityOutput(
                probability_type=self.probability_type,
                type_config=self.probability_config,
                name="attn_prob_1",
            )
            self.attn_prob_2 = ProbabilityOutput(
                probability_type=self.probability_type,
                type_config=self.probability_config,
                name="attn_prob_2",
            )

            self.q_norm_1, self.k_norm_1 = create_qk_norms(
                self.qk_norm_type, self.qk_norm_kwargs, "_1")
            self.q_norm_2, self.k_norm_2 = create_qk_norms(
                self.qk_norm_type, self.qk_norm_kwargs, "_2")

            # One RMSNorm over head_dim shared by every signal head, as in the
            # Differential Transformer's per-head normalization.
            self.head_norm = create_normalization_layer("rms_norm", name="head_norm")

        except Exception as e:
            logger.error(f"Failed to create GroupedDifferentialAttention sub-layers: {e}")
            raise ValueError(
                f"Failed to create GroupedDifferentialAttention sub-layers. "
                f"This might be due to invalid configuration parameters. "
                f"Original error: {e}"
            )

        self.lambda_q1 = None
        self.lambda_k1 = None
        self.lambda_q2 = None
        self.lambda_k2 = None

        logger.debug(
            f"GroupedDifferentialAttention: {self.num_signal_heads} signal heads, "
            f"{self.num_noise_heads} shared noise heads (G={self.group_ratio})"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the lambda vectors and explicitly build every sub-layer.

        :param input_shape: ``(batch_size, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank 3 or its last dimension
            is not ``dim``.
        """
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, seq_len, dim), got shape: {input_shape}"
            )
        if input_shape[-1] != self.dim:
            raise ValueError(
                f"Input dimension {input_shape[-1]} doesn't match expected dimension {self.dim}"
            )

        # Ye et al. initialise the four vectors from N(0, 0.1); one fresh instance per
        # vector, so the four draws are independent.
        for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
            setattr(self, name, self.add_weight(
                name=name,
                shape=(self.head_dim,),
                initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                trainable=True,
            ))

        self.q1_dense.build(input_shape)
        self.k1_dense.build(input_shape)
        self.q2_dense.build(input_shape)
        self.k2_dense.build(input_shape)
        self.v_dense.build(input_shape)

        batch, length = input_shape[0], input_shape[1]
        self.proj.build((batch, length, self._signal_dim))
        self.dropout_layer.build(input_shape)

        signal_scores = (batch, self.num_signal_heads, length, length)
        noise_scores = (batch, self.num_noise_heads, length, length)
        self.attn_prob_1.build(signal_scores)
        self.attn_prob_2.build(noise_scores)
        if self.attention_dropout_rate > 0.0:
            self.attn_dropout_1.build(signal_scores)
            self.attn_dropout_2.build(noise_scores)

        if self.q_norm_1 is not None:
            signal_qk = (batch, self.num_signal_heads, length, self.head_dim)
            noise_qk = (batch, self.num_noise_heads, length, self.head_dim)
            self.q_norm_1.build(signal_qk)
            self.k_norm_1.build(signal_qk)
            self.q_norm_2.build(noise_qk)
            self.k_norm_2.build(noise_qk)

        self.head_norm.build((batch, self.num_signal_heads, length, self.head_dim))

        super().build(input_shape)

    def get_lambda_init(self, layer_idx: int = 0) -> keras.KerasTensor:
        """Depth-dependent ``lambda_init = 0.8 - 0.6 * exp(-0.3 * layer_idx)``.

        Ye et al. write the exponent as ``-0.3 * (l - 1)`` for a 1-based layer
        ``l``; with the 0-based ``layer_idx`` used here that is ``-0.3 * layer_idx``.
        Result is in ``variable_dtype``.

        :param layer_idx: 0-based depth of this layer in the stack.
        :type layer_idx: int
        :return: Scalar tensor in ``[0.2, 0.8)``.
        :rtype: keras.KerasTensor
        """
        idx = keras.ops.cast(layer_idx, self.variable_dtype)
        return 0.8 - 0.6 * keras.ops.exp(-0.3 * idx)

    def get_lambda(self, layer_idx: int = 0) -> keras.KerasTensor:
        """Compute ``exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init(layer_idx)``.

        :param layer_idx: 0-based depth of this layer in the stack.
        :type layer_idx: int
        :return: Scalar lambda in ``variable_dtype``. :meth:`call` casts it to the
            attention dtype before combining the two contexts.
        :rtype: keras.KerasTensor
        """
        # Inside call() Keras autocasts variable reads to the compute dtype; the
        # exponentials are taken in variable_dtype so half precision cannot overflow them.
        dtype = self.variable_dtype
        q1, k1, q2, k2 = [
            keras.ops.cast(w, dtype)
            for w in (self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2)
        ]
        first = keras.ops.exp(keras.ops.sum(q1 * k1))
        second = keras.ops.exp(keras.ops.sum(q2 * k2))
        return first - second + self.get_lambda_init(layer_idx)

    def _apply_attention_mask(
        self,
        scores: keras.KerasTensor,
        attention_mask: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Broadcast the keep mask to rank 4 and apply it to ``scores``.

        Rank 2 is a key padding mask, rank 3 a per-query mask; both gain size-1
        batch-compatible head axes, so one mask serves the ``G * h`` signal scores and
        the ``h`` noise scores. Masking itself is ``apply_attention_mask``.

        :param scores: ``(batch, heads, q_seq, kv_seq)`` scores.
        :type scores: keras.KerasTensor
        :param attention_mask: Keep mask of rank 2, 3 or 4.
        :type attention_mask: keras.KerasTensor
        :return: Masked scores, same shape as ``scores``.
        :rtype: keras.KerasTensor
        """
        attention_mask = keras.ops.cast(attention_mask, scores.dtype)
        if len(attention_mask.shape) == 2:
            attention_mask = keras.ops.expand_dims(keras.ops.expand_dims(attention_mask, 1), 1)
        elif len(attention_mask.shape) == 3:
            attention_mask = keras.ops.expand_dims(attention_mask, 1)

        return apply_attention_mask(
            scores,
            attention_mask,
            out_dtype=dtype_name(scores),
            rescue_axis=probability_axis(self.probability_config),
        )

    def _attention_map(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        q_norm: Optional[keras.layers.Layer],
        k_norm: Optional[keras.layers.Layer],
        attn_prob: ProbabilityOutput,
        attn_dropout_layer: keras.layers.Dropout,
        attention_mask: Optional[keras.KerasTensor],
        training: Optional[bool],
    ) -> keras.KerasTensor:
        """Return one attention map ``P(q k^T s)`` of shape ``(B, heads, L, L)``.

        :param q: Per-head queries ``(B, heads, L, d)``.
        :type q: keras.KerasTensor
        :param k: Per-head keys ``(B, heads, L, d)``.
        :type k: keras.KerasTensor
        :param q_norm: Optional query norm.
        :type q_norm: Optional[keras.layers.Layer]
        :param k_norm: Optional key norm.
        :type k_norm: Optional[keras.layers.Layer]
        :param attn_prob: Probability normalization for this map.
        :type attn_prob: ProbabilityOutput
        :param attn_dropout_layer: Dropout on the map, run only when
            ``attention_dropout_rate`` is positive.
        :type attn_dropout_layer: keras.layers.Dropout
        :param attention_mask: Optional keep mask in the caller's raw rank.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: The attention probabilities.
        :rtype: keras.KerasTensor
        """
        if q_norm is not None:
            q = q_norm(q, training=training)
        if k_norm is not None:
            k = k_norm(k, training=training)

        scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))
        scores = scores * keras.ops.cast(self.scale, q.dtype)

        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)

        attn = attn_prob(scores, training=training)
        if self.attention_dropout_rate > 0.0:
            attn = attn_dropout_layer(attn, training=training)
        return attn

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        layer_idx: int = 0,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Apply grouped differential attention.

        :param inputs: ``(batch_size, sequence_length, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional keep mask of rank 2, 3 or 4, applied to both
            attention maps.
        :type attention_mask: Optional[keras.KerasTensor]
        :param layer_idx: 0-based depth selecting ``lambda_init``. Not stored in the
            config, so a stack must pass it on every call.
        :type layer_idx: int
        :param training: Whether the layer runs in training mode.
        :type training: Optional[bool]
        :return: ``(batch_size, sequence_length, dim)``.
        :rtype: keras.KerasTensor
        """
        # Static contract, re-asserted: build() sees only the first shape to arrive.
        if len(inputs.shape) != 3 or inputs.shape[-1] not in (None, self.dim):
            raise ValueError(
                f"Expected input (batch_size, seq_len, {self.dim}), got shape {tuple(inputs.shape)}"
            )

        q1 = split_heads(self.q1_dense(inputs), self.num_signal_heads, self.head_dim)
        k1 = split_heads(self.k1_dense(inputs), self.num_signal_heads, self.head_dim)
        q2 = split_heads(self.q2_dense(inputs), self.num_noise_heads, self.head_dim)
        k2 = split_heads(self.k2_dense(inputs), self.num_noise_heads, self.head_dim)
        v = split_heads(self.v_dense(inputs), self.num_noise_heads, self.head_dim)

        a1 = self._attention_map(
            q1, k1, self.q_norm_1, self.k_norm_1,
            self.attn_prob_1, self.attn_dropout_1, attention_mask, training)
        a2 = self._attention_map(
            q2, k2, self.q_norm_2, self.k_norm_2,
            self.attn_prob_2, self.attn_dropout_2, attention_mask, training)

        # Head i = g * h + j meets noise/value head j, so the h-head block is
        # repeated whole, G times (tile, not an element-wise repeat).
        repeats = (1, self.group_ratio, 1, 1)
        out1 = keras.ops.matmul(a1, keras.ops.tile(v, repeats))
        # (A1 - lambda A2) V = A1 V - lambda (A2 V): the noise context is
        # computed for h heads once and shared by the G signal groups.
        out2 = keras.ops.tile(keras.ops.matmul(a2, v), repeats)

        lambda_val = keras.ops.cast(self.get_lambda(layer_idx), out1.dtype)
        diff = out1 - lambda_val * out2

        diff = self.head_norm(diff, training=training)
        diff = diff * keras.ops.cast(1.0 - self.get_lambda_init(layer_idx), diff.dtype)

        output = self.proj(merge_heads(diff), training=training)
        return self.dropout_layer(output, training=training)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, ``(batch_size, sequence_length, dim)``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``input_shape``, since the projection maps back to ``dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration, covering every constructor parameter.

        ``layer_idx`` is a ``call`` argument and is absent.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'group_ratio': self.group_ratio,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
