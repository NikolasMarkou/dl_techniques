r"""
Differential multi-head attention, in :class:`DifferentialMultiHeadAttention`:
two parallel attention streams whose weighted difference cancels the
common-mode component of the attention distribution.

Ordinary softmax attention always allocates its full probability mass, even
when nothing in the context is worth attending to; that mass spreads thinly
over irrelevant tokens and enters the value aggregate as noise. This layer
borrows the differential-amplifier idea from analog electronics: two streams
read the same shared value matrix ``V`` through separate query/key
projections, and their weighted difference, ``out = (A1 - lambda * A2) V``,
keeps what only one stream selects and cancels what both select equally. The
mixing coefficient ``lambda`` is scheduled by depth (shallow layers near 0.2,
deep layers saturating near 0.8) and scaled by a learned parameter, clipped to
``[0.1, 0.9]`` for training stability.

Two steps from Ye et al.'s original paper are not implemented here: a
per-head GroupNorm on the differential output, and a rescale by
``(1 - lambda_init)`` to match a standard attention block's magnitude. A
caller comparing against published numbers should expect a different output
scale. `layer_idx` is a call argument, not constructor state (absent from
`get_config()`, a stack must pass its own depth each call), and there is no
`dim % num_heads` check since `head_dim` is an explicit required argument.

With ``s = 1 / sqrt(head_dim)`` and shared ``V``::

    A1  = P(Q1 K1^T s) ,   A2 = P(Q2 K2^T s)
    out = (A1 - lambda * A2) V

References:
    - Ye et al., 2024. Differential Transformer: Amplifying attention to the
      relevant context while canceling noise.
      (https://arxiv.org/abs/2410.05258)
    - Vaswani et al., 2017. Attention Is All You Need. NeurIPS.
      (https://arxiv.org/abs/1706.03762)
    - Martins and Astudillo, 2016. From Softmax to Sparsemax: A Sparse Model of
      Attention and Multi-Label Classification. (the alternative normalizations
      the pluggable ``ProbabilityOutput`` makes reachable)
      (https://arxiv.org/abs/1602.02068)
    - Henry et al., 2020. Query-Key Normalization for Transformers. (the optional
      QK-norm hook) (https://arxiv.org/abs/2010.04245)
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.layers.norms import create_normalization_layer

from .common import apply_attention_mask, compute_attention_scale
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.attention.differential_attention")
class DifferentialMultiHeadAttention(keras.layers.Layer):
    """
    Differential multi-head attention with pluggable normalization and optional QK-norm.

    Runs two parallel scaled-dot-product-attention streams and returns their
    weighted difference: ``Attention_diff = SDPA1(x) - lambda * SDPA2(x)``. The
    first stream captures the primary patterns. The second identifies the
    diffuse common-mode allocation. ``lambda`` controls how much of it is
    subtracted; it is scheduled by depth and scaled by a learned parameter.

    Each stream's normalization is its own :class:`ProbabilityOutput` instance
    (``self.attn_prob_1`` and ``self.attn_prob_2``), so any probability type
    works: softmax, sparsemax, threshmax or adaptive. Two separate instances
    are constructed so per-site debugging and weight inspection stay simple.

    ``call()`` takes an extra positional ``layer_idx: int = 0`` between
    ``attention_mask`` and ``training``: ``call(inputs, attention_mask=None,
    layer_idx=0, training=None)``. It selects the depth-dependent lambda
    schedule. Because it is a call argument rather than constructor state, it
    is absent from ``get_config()``, and a reloaded layer defaults to
    ``layer_idx=0`` unless the caller passes it again on every call. A stack
    of these layers must pass its own depth; this signature is part of the
    public contract, not a deviation to fix.

    The ``1 / sqrt(head_dim)`` temperature comes from
    :mod:`~dl_techniques.layers.attention.common`. Score normalization is the
    shared :class:`~dl_techniques.layers.activations.ProbabilityOutput`, and
    the optional QK-norms come from
    :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.
    There is no ``dim % num_heads`` check: ``head_dim`` is an explicit
    required constructor argument, so ``dim`` and ``num_heads * head_dim``
    are independent and the output projection reconciles them.

    Architecture:

    .. code-block:: text

                         inputs  [B, L, D]
                                  │
                                  ▼
                ┌──────────────────────────────────────┐
                │ 5 separate Dense → q1, k1, q2, k2, v │
                │ each [B, L, H·D_h] → [B, H, L, D_h]  │
                │ no fused QKV matmul                  │
                └───────┬───────────────────┬──────────┘
                        │ q1, k1, v         │ q2, k2, v
                        ▼                   ▼
                ┌────────────────┐  ┌────────────────┐
                │ stream 1       │  │ stream 2       │
                │ attn_prob_1    │  │ attn_prob_2    │
                │ attn_dropout_1 │  │ attn_dropout_2 │
                └───────┬────────┘  └───────┬────────┘
                        │ out1              │ out2
                        │  both [B, H, L, D_h]
                        ▼                   ▼
                ┌──────────────────────────────────────┐
                │ diff = out1 − lambda · out2          │
                │ lambda = get_lambda(layer_idx)       │
                └──────────────────┬───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ merge heads → [B, L, H · D_h]        │
                └──────────────────┬───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ proj: Dense(dim) → dropout_layer     │
                └──────────────────┬───────────────────┘
                                   ▼
                         output  [B, L, D]

        The same v tensor enters both streams. That is what makes the
        difference a cancellation. The rows of (A1 − lambda·A2) sum to
        1 − lambda, not to 1.

    One stream (``_stream``, run twice):

    .. code-block:: text

              q, k  [B, H, L, D_h]              v  [B, H, L, D_h]
                    │                                     │
                    ▼                                     │
        ┌───────────────────────────────┐                 │
        │ q_norm / k_norm  (optional,   │                 │
        │ only if qk_norm_type is set)  │                 │
        └───────────────┬───────────────┘                 │
                        ▼                                 │
        ┌───────────────────────────────┐                 │
        │ scores = q @ kᵀ * scale       │                 │
        │ scale = 1 / sqrt(head_dim)    │                 │
        └───────────────┬───────────────┘                 │
                        ▼                                 │
        ┌───────────────────────────────┐                 │
        │ attention_mask (optional)     │                 │
        │ keep-predicate semantics; one │                 │
        │ mask serves both streams      │                 │
        └───────────────┬───────────────┘                 │
                        ▼                                 │
        ┌───────────────────────────────┐                 │
        │ attn_prob: softmax, sparsemax │                 │
        │ threshmax or adaptive         │                 │
        └───────────────┬───────────────┘                 │
                        ▼                                 │
        ┌───────────────────────────────┐                 │
        │ attn_dropout (optional, only  │                 │
        │ if attention_dropout_rate > 0)│                 │
        └───────────────┬───────────────┘                 │
                        ▼                                 │
                       (@)◄───────────────────────────────┘
                        │
                        ▼
              context  [B, H, L, D_h]

    Lambda schedule by depth:

    .. code-block:: text

        layer_idx    0      1      2      4      8     16     ∞
        lambda    0.200  0.200  0.356  0.556  0.727  0.793  0.800

        Each entry is the schedule value alone. The learned scalar
        multiplies it, then the product is clipped to [0.1, 0.9].

        Re-derived 2026-08-27 from
        0.8 - 0.6 * exp(-0.3 * max(l - 1, 0)). Three entries printed
        here were wrong before that: layer_idx 4 read 0.530 (true
        0.556), 8 read 0.720 (true 0.727) and 16 read 0.797 (true
        0.793). The last was above the asymptote the same table
        states as 0.800.

    :param dim: Integer, input and output dimension. Must be positive and should be
        divisible by num_heads for optimal performance.
    :type dim: int
    :param num_heads: Integer, number of attention heads for both attention streams.
        Must be positive.
    :type num_heads: int
    :param head_dim: Integer, dimension of each attention head. Must be positive.
    :type head_dim: int
    :param dropout_rate: Float, output dropout rate applied after projection.
        Must be between 0 and 1. Defaults to 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Float, dropout rate applied to attention weights in
        both streams. Must be between 0 and 1. Defaults to 0.0.
    :type attention_dropout_rate: float
    :param lambda_init: Float, initial value for the lambda parameter controlling the
        balance between attention streams. Should be between 0 and 1.
        Defaults to 0.8.
    :type lambda_init: float
    :param probability_type: String identifier for the per-stream probability
        normalization strategy. Forwarded to :class:`ProbabilityOutput`. Both streams
        share the same type. Defaults to ``"softmax"``.
    :type probability_type: str
    :param probability_config: Optional dict of strategy-specific arguments forwarded
        to :class:`ProbabilityOutput`. Both streams share the same config.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied to each stream's
        per-head Q and K projections before computing attention scores (QK-norm).
        Forwarded to :func:`create_normalization_layer`. ``None`` disables QK-norm.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` when constructing per-stream Q/K norms.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kernel_initializer: String or Initializer, initializer for kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional Regularizer, regularizer applied to kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_initializer: String or Initializer, initializer for bias weights.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Optional Regularizer, regularizer applied to bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param activity_regularizer: Optional Regularizer, regularizer applied to layer output.
    :type activity_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments passed to Layer base class.
    :type kwargs: Any

    :ivar q1_dense: Stream 1 query projection.
    :vartype q1_dense: keras.layers.Dense
    :ivar k1_dense: Stream 1 key projection.
    :vartype k1_dense: keras.layers.Dense
    :ivar q2_dense: Stream 2 query projection.
    :vartype q2_dense: keras.layers.Dense
    :ivar k2_dense: Stream 2 key projection.
    :vartype k2_dense: keras.layers.Dense
    :ivar v_dense: The single SHARED value projection.
    :vartype v_dense: keras.layers.Dense
    :ivar proj: Output projection back to ``dim``.
    :vartype proj: keras.layers.Dense
    :ivar dropout_layer: Output dropout, applied after ``proj``.
    :vartype dropout_layer: keras.layers.Dropout
    :ivar attn_prob_1: Stream 1 probability normalization.
    :vartype attn_prob_1: ProbabilityOutput
    :ivar attn_prob_2: Stream 2 probability normalization.
    :vartype attn_prob_2: ProbabilityOutput
    :ivar attn_dropout_1: Stream 1 attention-weight dropout, or ``None``.
    :vartype attn_dropout_1: Optional[keras.layers.Dropout]
    :ivar attn_dropout_2: Stream 2 attention-weight dropout, or ``None``.
    :vartype attn_dropout_2: Optional[keras.layers.Dropout]
    :ivar q_norm_1: Stream 1 optional query norm, or ``None``.
    :vartype q_norm_1: Optional[keras.layers.Layer]
    :ivar k_norm_1: Stream 1 optional key norm, or ``None``.
    :vartype k_norm_1: Optional[keras.layers.Layer]
    :ivar q_norm_2: Stream 2 optional query norm, or ``None``.
    :vartype q_norm_2: Optional[keras.layers.Layer]
    :ivar k_norm_2: Stream 2 optional key norm, or ``None``.
    :vartype k_norm_2: Optional[keras.layers.Layer]
    :ivar lambda_param: The learned scalar multiplying the depth schedule.
        ``None`` until ``build()`` runs.
    :vartype lambda_param: Optional[keras.Variable]
    :ivar scale: The ``1 / sqrt(head_dim)`` temperature.
    :vartype scale: float

    :raises ValueError: If dim is not positive.
    :raises ValueError: If num_heads is not positive.
    :raises ValueError: If head_dim is not positive.
    :raises ValueError: If dropout rates are not between 0 and 1.
    :raises ValueError: If lambda_init is not between 0 and 1.
    :raises ValueError: If ``probability_type`` is a routing or hierarchical
        variant. Those consume features and require a fixed ``output_dim``,
        which is incompatible with score logits whose last axis is the kv
        sequence length.
    :raises ValueError: If sub-layer construction fails for any reason. The
        underlying exception is logged and re-raised as a ``ValueError``.
    :raises ValueError: From ``build()``, if the input is not 3D or its last
        dimension does not match ``dim``.

    Input shape:
        3D tensor with shape ``(batch_size, sequence_length, dim)``. The optional
        ``attention_mask`` may be ``(batch, kv_seq)``, ``(batch, q_seq, kv_seq)`` or
        ``(batch, num_heads, q_seq, kv_seq)``, in keep-predicate semantics, and is
        applied to both streams.

    Output shape:
        3D tensor with shape ``(batch_size, sequence_length, dim)``, unchanged
        from the input, since the output projection maps
        ``num_heads * head_dim`` back to ``dim``.

    Example:
        >>> attn = DifferentialMultiHeadAttention(dim=512, num_heads=8, head_dim=64)
        >>> x = keras.random.normal((2, 128, 512))
        >>> y = attn(x, training=False)                 # (2, 128, 512)
        >>>
        >>> # A layer deep in a stack passes its own depth
        >>> y = attn(x, layer_idx=11, training=False)
        >>>
        >>> # Sparse per-stream normalization plus QK-norm
        >>> attn = DifferentialMultiHeadAttention(
        ...     dim=512, num_heads=8, head_dim=64,
        ...     probability_type="sparsemax", qk_norm_type="rms_norm",
        ... )

    Note:
        ``head_dim`` is required and independent of ``dim``. This layer performs
        no ``dim % num_heads`` check, and ``num_heads * head_dim`` need not equal
        ``dim``. The output projection reconciles the two.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        lambda_init: float = 0.8,
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

        ``lambda_param`` is the only weight this layer owns directly, and it is
        created in :meth:`build`. See the class docstring for the full parameter
        reference; the parameters are listed there once rather than twice.

        :param dim: Input and output dimension. Must be positive.
        :type dim: int
        :param num_heads: Number of attention heads, used by both streams.
        :type num_heads: int
        :param head_dim: Dimension of each attention head. Must be positive.
        :type head_dim: int
        :param dropout_rate: Output dropout rate applied after projection.
        :type dropout_rate: float
        :param attention_dropout_rate: Dropout rate on attention weights.
        :type attention_dropout_rate: float
        :param lambda_init: Initial value of the learned lambda scalar.
        :type lambda_init: float
        :param probability_type: Per-stream normalization strategy name.
        :type probability_type: str
        :param probability_config: Optional strategy-specific arguments.
        :type probability_config: Optional[Dict[str, Any]]
        :param qk_norm_type: Optional QK-norm type. ``None`` disables QK-norm.
        :type qk_norm_type: Optional[str]
        :param qk_norm_kwargs: Optional arguments for the QK-norm layers.
        :type qk_norm_kwargs: Optional[Dict[str, Any]]
        :param kernel_initializer: Initializer for kernel weights.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Optional regularizer for kernel weights.
        :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
        :param bias_initializer: Initializer for bias weights.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param bias_regularizer: Optional regularizer for bias weights.
        :type bias_regularizer: Optional[keras.regularizers.Regularizer]
        :param activity_regularizer: Optional regularizer on the layer output.
        :type activity_regularizer: Optional[keras.regularizers.Regularizer]
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any

        :raises ValueError: If any size or rate is out of range, if
            ``probability_type`` is a routing or hierarchical variant, or if
            sub-layer construction fails.
        """
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}"
            )
        if not (0.0 <= lambda_init <= 1.0):
            raise ValueError(f"lambda_init must be between 0 and 1, got {lambda_init}")

        # Reject routing and hierarchical probability types. They need an
        # output_dim and consume features rather than score logits, which does
        # not fit attention scores whose last axis is the dynamic kv sequence
        # length.
        _ptype_lower = probability_type.lower()
        if _ptype_lower in (
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        ):
            raise ValueError(
                f"probability_type='{probability_type}' is not supported in "
                "DifferentialMultiHeadAttention: routing/hierarchical strategies "
                "require a fixed output_dim and consume features rather than "
                "score logits. Use one of: 'softmax', 'sparsemax', 'threshmax', "
                "'adaptive'."
            )

        # Store every __init__ parameter for get_config.
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.lambda_init = lambda_init
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # Store serialized initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Per-head projection width. Each stream has num_heads * head_dim
        # features for Q and K; V is shared between the streams.
        self._proj_dim = self.num_heads * self.head_dim

        # Scale factor for scaled dot-product attention.
        self.scale = compute_attention_scale(self.head_dim)

        # Sub-layers are created here, per the Keras 3 pattern.
        try:
            dense_kwargs = {
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
            }

            # Five separate projection Dense layers. One fused layer per
            # stream's Q/K would also work, but five separate layers keep
            # debugging trivial and match the per-site pattern this module
            # documents.
            self.q1_dense = keras.layers.Dense(self._proj_dim, name="q1", **dense_kwargs)
            self.k1_dense = keras.layers.Dense(self._proj_dim, name="k1", **dense_kwargs)
            self.q2_dense = keras.layers.Dense(self._proj_dim, name="q2", **dense_kwargs)
            self.k2_dense = keras.layers.Dense(self._proj_dim, name="k2", **dense_kwargs)
            self.v_dense = keras.layers.Dense(self._proj_dim, name="v", **dense_kwargs)

            # Output projection layer
            self.proj = keras.layers.Dense(
                self.dim,
                name='proj',
                **dense_kwargs,
            )

            # Output dropout layer
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name='dropout')

            # Per-stream attention-weight dropout. Matches the
            # `attention_dropout_rate` of the original MHA-based version.
            if self.attention_dropout_rate > 0.0:
                self.attn_dropout_1 = keras.layers.Dropout(
                    self.attention_dropout_rate, name="attn_dropout_1"
                )
                self.attn_dropout_2 = keras.layers.Dropout(
                    self.attention_dropout_rate, name="attn_dropout_2"
                )
            else:
                self.attn_dropout_1 = None
                self.attn_dropout_2 = None

            # Per-stream probability normalization. Two instances sharing the
            # same probability_type and probability_config.
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

            # Optional per-stream QK-norm. Each stream gets its own pair of
            # Q/K normalization layers so the streams stay independent.
            if self.qk_norm_type is not None:
                _qk_kwargs = self.qk_norm_kwargs or {}
                self.q_norm_1 = create_normalization_layer(
                    self.qk_norm_type, name="q_norm_1", **_qk_kwargs
                )
                self.k_norm_1 = create_normalization_layer(
                    self.qk_norm_type, name="k_norm_1", **_qk_kwargs
                )
                self.q_norm_2 = create_normalization_layer(
                    self.qk_norm_type, name="q_norm_2", **_qk_kwargs
                )
                self.k_norm_2 = create_normalization_layer(
                    self.qk_norm_type, name="k_norm_2", **_qk_kwargs
                )
            else:
                self.q_norm_1 = None
                self.k_norm_1 = None
                self.q_norm_2 = None
                self.k_norm_2 = None

        except Exception as e:
            logger.error(f"Failed to create DifferentialMultiHeadAttention sub-layers: {e}")
            raise ValueError(
                f"Failed to create DifferentialMultiHeadAttention sub-layers. "
                f"This might be due to invalid configuration parameters. "
                f"Original error: {e}"
            )

        # Weight attributes - created in build()
        self.lambda_param = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the lambda parameter and explicitly build every sub-layer.

        Each sub-layer is built by hand, at the shape it will actually see. The
        score shape for the probability layers and attention dropout, the
        per-head shape for the QK-norms, and ``(B, L, num_heads · head_dim)``
        for the output projection. Weight restoration requires those variables
        to already exist.

        :param input_shape: Shape tuple of the input tensor, expected as
            ``(batch_size, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank 3, or if its last
            dimension does not equal ``dim``.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, seq_len, dim), got shape: {input_shape}"
            )

        input_dim = input_shape[-1]
        if input_dim != self.dim:
            raise ValueError(
                f"Input dimension {input_dim} doesn't match expected dimension {self.dim}"
            )

        # Create the layer's own weight, the lambda parameter. The schedule
        # that consumes it, kept exactly as the previous implementation had it,
        # is lambda = clip(layer_dep_init * lambda_param, 0.1, 0.9) with
        # layer_dep_init = 0.8 - 0.6 * exp(-0.3 * max(layer_idx - 1, 0)).
        self.lambda_param = self.add_weight(
            name="lambda_param",
            shape=(1,),
            initializer=keras.initializers.Constant(self.lambda_init),
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Build projection layers explicitly for serialization.
        self.q1_dense.build(input_shape)
        self.k1_dense.build(input_shape)
        self.q2_dense.build(input_shape)
        self.k2_dense.build(input_shape)
        self.v_dense.build(input_shape)

        # Output projection consumes (B, L, num_heads*head_dim) and produces (B, L, dim).
        proj_input_shape = (input_shape[0], input_shape[1], self._proj_dim)
        self.proj.build(proj_input_shape)
        self.dropout_layer.build(input_shape)

        # Build per-stream probability layers with the attention-score shape.
        attn_shape = (input_shape[0], self.num_heads, input_shape[1], input_shape[1])
        self.attn_prob_1.build(attn_shape)
        self.attn_prob_2.build(attn_shape)

        if self.attn_dropout_1 is not None:
            self.attn_dropout_1.build(attn_shape)
            self.attn_dropout_2.build(attn_shape)

        # Build per-stream QK-norm layers with the per-head Q/K shape.
        if self.q_norm_1 is not None:
            qk_shape = (input_shape[0], self.num_heads, input_shape[1], self.head_dim)
            self.q_norm_1.build(qk_shape)
            self.k_norm_1.build(qk_shape)
            self.q_norm_2.build(qk_shape)
            self.k_norm_2.build(qk_shape)

        super().build(input_shape)

    def get_lambda(self, layer_idx: int = 0) -> keras.KerasTensor:
        """Compute the mixing coefficient for a given depth.

        The depth schedule ``0.8 - 0.6 * exp(-0.3 * max(layer_idx - 1, 0))``
        follows the paper's initialization strategy. The learned
        ``lambda_param`` multiplies it, and the product is clipped to
        ``[0.1, 0.9]`` for stability. At ``lambda -> 1`` the differential
        operator's rows sum to zero and the residual stream loses its DC
        component.

        :param layer_idx: Index of the layer in the network stack, 0-based.
            Selects the depth-dependent schedule value.
        :type layer_idx: int
        :return: The lambda value, bounded to ``[0.1, 0.9]``, in this layer's
            ``variable_dtype`` (the dtype of ``lambda_param`` itself).
            :meth:`call` casts it to the dtype of the attention streams before
            combining them.
        :rtype: keras.KerasTensor
        """
        dtype = self.variable_dtype

        # Depth-dependent value from the paper:
        # layer_dependent_init = 0.8 - 0.6 * exp(-0.3 * max(layer_idx - 1, 0))
        layer_factor = keras.ops.cast(layer_idx, dtype=dtype)
        exp_term = keras.ops.exp(-0.3 * keras.ops.maximum(layer_factor - 1.0, 0.0))
        layer_dependent_init = 0.8 - 0.6 * exp_term

        # The learned scalar multiplies the schedule. Clip for training
        # stability.
        lambda_val = keras.ops.clip(
            layer_dependent_init * keras.ops.cast(self.lambda_param[0], dtype),
            0.1,
            0.9,
        )

        return lambda_val

    def _apply_attention_mask(
        self,
        scores: keras.KerasTensor,
        attention_mask: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Broadcast the mask up to rank 4 and apply it to the scores.

        A rank-2 mask is treated as a padding mask over keys. A rank-3 mask is
        treated as a full per-query mask. Both are expanded on the head axis.
        Masking itself is delegated to the shared helper in ``common.py``,
        which owns the fp16-safe form.

        :param scores: Attention scores of shape ``(batch, num_heads, q_seq, kv_seq)``.
        :type scores: keras.KerasTensor
        :param attention_mask: Attention mask. Supported shapes: ``(batch, kv_seq)``
            (padding mask), ``(batch, q_seq, kv_seq)`` (full mask), or
            ``(batch, num_heads, q_seq, kv_seq)``.
        :type attention_mask: keras.KerasTensor
        :return: Masked scores tensor with same shape as input scores.
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
            # `getattr(d, "name", None) or str(d)`, not `keras.backend.standardize_dtype`:
            # a Keras-2 residue banned across `src/`, and `str` alone mis-renders a
            # `tf.DType`. Full note and the measured equivalence at `common.py`; D-007.
            out_dtype=(getattr(scores.dtype, "name", None) or str(scores.dtype)),
            rescue_axis=(self.probability_config or {}).get("axis", -1),
        )

    def _project_to_heads(
        self,
        x: keras.KerasTensor,
        batch_size: keras.KerasTensor,
        seq_len: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Reshape a projected tensor ``(B, L, H*D_h)`` to ``(B, H, L, D_h)``.

        :param x: A projection output of shape ``(B, L, num_heads * head_dim)``.
        :type x: keras.KerasTensor
        :param batch_size: Dynamic batch size, from ``keras.ops.shape``.
        :type batch_size: keras.KerasTensor
        :param seq_len: Dynamic sequence length, from ``keras.ops.shape``.
        :type seq_len: keras.KerasTensor
        :return: The same values as ``(B, num_heads, L, head_dim)``.
        :rtype: keras.KerasTensor
        """
        x = keras.ops.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return keras.ops.transpose(x, (0, 2, 1, 3))

    def _stream(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor,
        q_norm: Optional[keras.layers.Layer],
        k_norm: Optional[keras.layers.Layer],
        attn_prob: ProbabilityOutput,
        attn_dropout_layer: Optional[keras.layers.Dropout],
        attention_mask: Optional[keras.KerasTensor],
        training: Optional[bool],
    ) -> keras.KerasTensor:
        """Run a single SDPA stream and return its ``(B, H, L, D_h)`` context.

        Applies optional QK-norm, computes scaled dot-product scores, applies
        the optional mask, normalizes through the supplied
        ``ProbabilityOutput``, applies optional attention-weight dropout, and
        returns ``attn @ v``. Both streams call this with the same ``v``, which
        is what makes their difference a cancellation.

        :param q: Per-head queries, ``(B, H, L, D_h)``.
        :type q: keras.KerasTensor
        :param k: Per-head keys, ``(B, H, L, D_h)``.
        :type k: keras.KerasTensor
        :param v: Per-head values, ``(B, H, L, D_h)``. Shared between streams.
        :type v: keras.KerasTensor
        :param q_norm: This stream's query norm, or ``None``.
        :type q_norm: Optional[keras.layers.Layer]
        :param k_norm: This stream's key norm, or ``None``.
        :type k_norm: Optional[keras.layers.Layer]
        :param attn_prob: This stream's probability normalization layer.
        :type attn_prob: ProbabilityOutput
        :param attn_dropout_layer: This stream's attention dropout, or ``None``.
        :type attn_dropout_layer: Optional[keras.layers.Dropout]
        :param attention_mask: Optional mask, already in the caller's raw rank.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: The stream's context, ``(B, H, L, D_h)``.
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

        if attn_dropout_layer is not None:
            attn = attn_dropout_layer(attn, training=training)

        return keras.ops.matmul(attn, v)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        layer_idx: int = 0,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Apply the differential attention mechanism.

        Computes ``Attention_diff = SDPA1(x) - lambda * SDPA2(x)``. The first
        stream captures the primary attention patterns. The second identifies
        the common-mode allocation. ``lambda``, from the depth schedule,
        controls how much of it is cancelled.

        :param inputs: Input tensor of shape ``(batch_size, sequence_length, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask tensor. Can be 2D, 3D, or 4D
            for different masking strategies. One mask serves both streams.
        :type attention_mask: Optional[keras.KerasTensor]
        :param layer_idx: Index of the layer in the network stack, 0-based.
            Selects the depth-dependent lambda. Defaults to 0. Not stored in
            the config, so a stack must pass its own depth on every call.
        :type layer_idx: int
        :param training: Optional boolean indicating whether in training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, sequence_length, dim)`` after
            applying differential attention and output projection.
        :rtype: keras.KerasTensor
        """
        batch_size = keras.ops.shape(inputs)[0]
        seq_len = keras.ops.shape(inputs)[1]

        # Project to Q1, K1, Q2, K2, V and reshape to per-head format.
        q1 = self._project_to_heads(self.q1_dense(inputs), batch_size, seq_len)
        k1 = self._project_to_heads(self.k1_dense(inputs), batch_size, seq_len)
        q2 = self._project_to_heads(self.q2_dense(inputs), batch_size, seq_len)
        k2 = self._project_to_heads(self.k2_dense(inputs), batch_size, seq_len)
        v = self._project_to_heads(self.v_dense(inputs), batch_size, seq_len)

        # Two parallel SDPA streams. V is shared; the depth schedule combines
        # their outputs afterwards.
        out1 = self._stream(
            q1, k1, v,
            self.q_norm_1, self.k_norm_1,
            self.attn_prob_1, self.attn_dropout_1,
            attention_mask, training,
        )
        out2 = self._stream(
            q2, k2, v,
            self.q_norm_2, self.k_norm_2,
            self.attn_prob_2, self.attn_dropout_2,
            attention_mask, training,
        )

        # Depth-dependent lambda, same schedule as the original.
        lambda_val = keras.ops.cast(self.get_lambda(layer_idx), out2.dtype)

        # Differential attention: SDPA1 - lambda*SDPA2
        diff = out1 - lambda_val * out2

        # Merge heads: (B, H, L, D_h) -> (B, L, H*D_h)
        diff = keras.ops.transpose(diff, (0, 2, 1, 3))
        diff = keras.ops.reshape(diff, (batch_size, seq_len, self._proj_dim))

        # Apply output projection and dropout
        output = self.proj(diff, training=training)
        output = self.dropout_layer(output, training=training)

        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        The output projection maps ``num_heads * head_dim`` back to ``dim``, so
        the layer is shape-preserving even when those two differ.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization, includes all constructor parameters.

        ``layer_idx`` is absent: it is a ``call`` argument, so a reloaded
        layer defaults to ``layer_idx=0`` unless the caller passes it.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'lambda_init': self.lambda_init,
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

