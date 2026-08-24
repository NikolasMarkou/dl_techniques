"""
Self-attention block of the FastViT / MobileCLIP2 MCi backbone.

This module transcribes timm's ``AttentionBlock`` (together with the ``Attention``
token mixer it wraps) as used by the FastViT MCi image tower. It is the block that
replaces :class:`~dl_techniques.layers.fastvit.rep_mixer.FastVitRepMixerBlock` in
the deepest stage(s) of every MCi variant, where the feature map is small enough
for quadratic global attention to be affordable.

Structurally it is the classical pre-norm transformer block, with two additions
that FastViT shares with the rest of the ConvNeXt/DeiT lineage: a per-channel
LayerScale on each residual branch, and stochastic depth on each residual branch.

.. code-block:: text

    x = x + drop_path_1(layer_scale_1(Attention(Norm(x))))
    x = x + drop_path_2(layer_scale_2(ConvMlp(x)))

**The load-bearing detail is the rank-4 <-> rank-3 conversion.**

The whole MCi tower is channels-last rank-4 ``(B, H, W, C)``, but the repo's
shared attention layer accepts ONLY rank-3 ``(B, N, C)`` and raises ``ValueError``
on a 4-D hand-off (MEASURED). So this block owns the flatten and the reshape back.
Two things follow, and both are the kind of defect that shape assertions do not
catch:

1. The reshape back MUST be the exact inverse of the flatten. Flattening
   ``(B, H, W, C)`` row-major yields an H-major token order, and the inverse is
   ``reshape(..., (B, H, W, C))``. Producing ``(B, W, H, C)`` — or flattening a
   transposed tensor and reshaping back untransposed — silently transposes every
   feature map while every downstream shape still matches whenever ``H == W``.
   This repo has already shipped that defect class twice (a transposed relative
   position bias passed 219 tests; a flipped CLS slice passed 91/91), which is why
   the accompanying pin uses a NON-SQUARE input and an independently written
   oracle rather than a shape assertion.
2. Spatial dimensions may be dynamic (they are, under ``fit()``'s
   ``@tf.function`` tracing). The target shape is therefore derived with
   ``keras.ops.shape`` / ``keras.ops.stack``; a Python ``int()`` or ``tuple()``
   coercion of a traced dimension raises at trace time. A static fast path is
   used when the spatial dims are known, purely so the eager graph stays simple.

Two further reference details are explicit-or-silently-wrong:

* ``create_normalization_layer`` ``setdefault``s ``epsilon=1e-6``. The reference
  norm (``BatchNorm2d`` for mci0/1/2, ``LayerNormChannel`` for mci3/mci4) uses
  ``1e-5``, so the epsilon is passed EXPLICITLY.
* ``LearnableMultiplier`` defaults to ``constraint='non_neg'``, which would clamp
  a legitimately negative LayerScale gamma at zero. ``constraint=None`` is
  required; the shared helper in :mod:`.rep_mixer` states it in one place.

.. warning::
   **Recorded deviation from the reference (X-2).** timm's ``Attention`` is
   ``qkv_bias=False`` but its output projection is a plain ``nn.Linear(dim, dim)``,
   i.e. BIASED. The repo's shared ``MultiHeadAttention`` exposes a single
   ``use_bias`` flag governing BOTH projections, so the two cannot be set
   independently without editing a shared layer (out of scope). ``use_bias=False``
   is the closer of the two available settings: it costs exactly one missing bias
   vector of length ``dim`` per block, whereas ``use_bias=True`` would ADD a
   spurious ``3 * dim`` qkv bias the reference does not have. This is pinned by
   ``test_attention_has_no_bias_weights`` so it cannot drift silently.

References:
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
    - Vasu et al., 2024. MobileCLIP: Fast Image-Text Models through Multi-Modal
      Reinforced Training. (https://arxiv.org/abs/2311.17049)
    - Touvron et al., 2021. Going Deeper with Image Transformers (LayerScale).
      (https://arxiv.org/abs/2103.17239)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from keras import ops, initializers, regularizers, activations
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .conv_mlp import FastVitConvMlp
from .rep_mixer import _create_layer_scale, _REFERENCE_LAYER_SCALE_INIT
from ..stochastic_depth import StochasticDepth
from ..attention.factory import create_attention_layer
from ..norms.factory import create_normalization_layer
from .reference import REFERENCE_NORM_EPSILON

# ---------------------------------------------------------------------

#: Single definition of the reference epsilon lives in :mod:`.reference`.
_REFERENCE_NORM_EPSILON = REFERENCE_NORM_EPSILON

#: Reference per-head width. ``num_heads`` is derived as ``dim // head_dim``,
#: exactly as timm's ``Attention`` does.
_REFERENCE_HEAD_DIM = 32


@keras.saving.register_keras_serializable()
class FastVitAttentionBlock(keras.layers.Layer):
    """FastViT attention block: pre-norm global self-attention + a convolutional MLP.

    Channels-last transcription of timm's ``AttentionBlock``. Both halves are
    residual, each gated by its own per-channel LayerScale and guarded by its own
    stochastic-depth branch. The block preserves its input shape exactly.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │            Input [B, H, W, dim]              │
        └───────┬──────────────────────────┬───────────┘
                │                          ▼
                │            ┌──────────────────────────────┐
                │            │ norm (batch_norm | layer_norm│
                │            │       eps = 1e-5)            │
                │            └──────────────┬───────────────┘
                │                           ▼
                │            ┌──────────────────────────────┐
                │            │ reshape (B,H,W,C)→(B,H*W,C)  │
                │            │ MultiHeadAttention, no bias  │
                │            │ reshape (B,H*W,C)→(B,H,W,C)  │
                │            └──────────────┬───────────────┘
                │                           ▼
                │            ┌──────────────────────────────┐
                │            │ layer_scale_1 (per-channel)  │
                │            │ drop_path_1                  │
                │            └──────────────┬───────────────┘
                │                           │
                └────────────── + ──────────┘
                                │
                ┌───────────────┴──────────┬───────────────┐
                │                          ▼               │
                │            ┌──────────────────────────────┐
                │            │ mlp: FastVitConvMlp          │
                │            │  dw 7×7 + BN → 1×1 → act →   │
                │            │  drop → 1×1 → drop           │
                │            └──────────────┬───────────────┘
                │                           ▼
                │            ┌──────────────────────────────┐
                │            │ layer_scale_2 (per-channel)  │
                │            │ drop_path_2                  │
                │            └──────────────┬───────────────┘
                │                           │
                └────────────── + ──────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │            Output [B, H, W, dim]             │
        └──────────────────────────────────────────────┘

    .. note::
       ``StochasticDepth`` short-circuits to the identity only when ``training is
       False`` (or the rate is exactly 0.0); ``training=None`` runs the stochastic
       path. Deterministic tests must pass ``training=False`` EXPLICITLY.

    :param dim: Number of channels. The block preserves it. Must be positive and
        divisible by ``head_dim``.
    :type dim: int
    :param mlp_ratio: Expansion ratio of the ConvMlp bottleneck; the hidden width
        is ``int(dim * mlp_ratio)``. Must be positive and yield a positive hidden
        width. Defaults to 4.0.
    :type mlp_ratio: float
    :param head_dim: Per-head width. ``num_heads`` is ``dim // head_dim``. Must be
        positive and divide ``dim`` exactly. Defaults to 32 (the reference value).
    :type head_dim: int
    :param normalization_type: Key passed to
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.
        ``'batch_norm'`` reproduces the reference for mci0/mci1/mci2;
        ``'layer_norm'`` reproduces ``LayerNormChannel`` for mci3/mci4 (which, in
        channels-last, is exactly ``LayerNormalization(axis=-1, epsilon=1e-5)``).
        Defaults to ``'batch_norm'``.
    :type normalization_type: str
    :param dropout_rate: Dropout rate inside the ConvMlp. Must be in ``[0, 1)``.
        Defaults to 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout rate on the attention probabilities.
        Must be in ``[0, 1)``. Defaults to 0.0.
    :type attention_dropout_rate: float
    :param drop_path_rate: Per-sample stochastic-depth rate applied to BOTH
        residual branches. Must be in ``[0, 1)``. Defaults to 0.0.
    :type drop_path_rate: float
    :param layer_scale_init_value: Constant initialization for BOTH LayerScale
        gammas, or ``None`` to omit LayerScale on both branches. Defaults to
        ``1e-5``.
    :type layer_scale_init_value: Optional[float]
    :param activation: Activation used inside the ConvMlp. Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for the attention projections. Defaults
        to ``'he_normal'``. The ConvMlp keeps its own reference default
        (``TruncatedNormal(stddev=0.02)``, per timm's ``_init_weights``).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer applied to the attention
        projections and to every ConvMlp kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim`` or ``head_dim`` are not positive, if ``dim`` is
        not divisible by ``head_dim``, if ``mlp_ratio`` is not positive or yields a
        zero-width bottleneck, if ``dropout_rate`` / ``attention_dropout_rate`` /
        ``drop_path_rate`` fall outside ``[0, 1)``, if ``normalization_type`` is not
        a string, or if ``layer_scale_init_value`` is neither a real number nor
        ``None``.

    Example:
        >>> import numpy as np
        >>> block = FastVitAttentionBlock(dim=64, head_dim=32)
        >>> y = block(np.zeros((2, 8, 4, 64), dtype='float32'), training=False)
        >>> y.shape
        (2, 8, 4, 64)
    """

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.0,
            head_dim: int = _REFERENCE_HEAD_DIM,
            normalization_type: str = 'batch_norm',
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            layer_scale_init_value: Optional[float] = _REFERENCE_LAYER_SCALE_INIT,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if dim % head_dim != 0:
            raise ValueError(
                f"dim must be divisible by head_dim (num_heads is derived as "
                f"dim // head_dim), got dim={dim} and head_dim={head_dim} "
                f"(remainder {dim % head_dim})"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError(
                f"mlp_ratio={mlp_ratio} with dim={dim} yields a zero-width "
                f"bottleneck (int(dim * mlp_ratio) == {hidden_dim})"
            )
        if not isinstance(normalization_type, str):
            raise ValueError(
                f"normalization_type must be a string key accepted by "
                f"create_normalization_layer, got {normalization_type!r}"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if not 0.0 <= attention_dropout_rate < 1.0:
            raise ValueError(
                f"attention_dropout_rate must be in [0, 1), "
                f"got {attention_dropout_rate}"
            )
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}")
        if layer_scale_init_value is not None:
            if isinstance(layer_scale_init_value, bool) or not isinstance(
                    layer_scale_init_value, (int, float)):
                raise ValueError(
                    f"layer_scale_init_value must be a real number or None, "
                    f"got {layer_scale_init_value!r}"
                )

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.mlp_ratio = float(mlp_ratio)
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.hidden_dim = hidden_dim
        self.normalization_type = normalization_type
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = (
            None if layer_scale_init_value is None else float(layer_scale_init_value)
        )
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        # epsilon is EXPLICIT: the factory would otherwise setdefault 1e-6, while
        # the reference (BatchNorm2d / LayerNormChannel) uses 1e-5 (MEASURED).
        self.norm = create_normalization_layer(
            self.normalization_type,
            name='norm',
            epsilon=_REFERENCE_NORM_EPSILON,
        )
        # The shared attention layer accepts ONLY rank-3 input, so this block owns
        # the flatten/reshape (see the module docstring). `use_bias=False` is the
        # recorded deviation X-2 (qkv unbiased matches, proj unbiased does not).
        self.attn = create_attention_layer(
            'multi_head',
            name='attn',
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.layer_scale_1 = _create_layer_scale(
            self.layer_scale_init_value, name='layer_scale_1')
        self.layer_scale_2 = _create_layer_scale(
            self.layer_scale_init_value, name='layer_scale_2')
        # Created UNCONDITIONALLY: StochasticDepth accepts a rate of exactly 0.0
        # and short-circuits to the identity for it.
        self.drop_path_1 = StochasticDepth(
            drop_path_rate=self.drop_path_rate, name='drop_path_1')
        self.drop_path_2 = StochasticDepth(
            drop_path_rate=self.drop_path_rate, name='drop_path_2')
        self.mlp = FastVitConvMlp(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_regularizer=self.kernel_regularizer,
            name='mlp',
        )

    # ------------------------------------------------------------------
    # rank-4 <-> rank-3 conversion
    # ------------------------------------------------------------------

    def _to_sequence(self, x):
        """Flatten ``(B, H, W, C)`` to ``(B, H*W, C)`` in H-major token order.

        :param x: Rank-4 channels-last tensor.
        :return: A ``(sequence, spatial_shape)`` pair, where ``spatial_shape`` is
            the argument to feed back to :meth:`_to_spatial` so the reshape is the
            EXACT inverse of this flatten.
        :rtype: Tuple[Any, Any]

        .. warning::
           The returned token order is H-major (row-major over ``(H, W)``). Any
           other order — e.g. flattening a ``(B, W, H, C)`` transpose — must be
           inverted by the matching inverse, or the feature map is silently
           transposed. See ``test_non_square_input_roundtrips_orientation``.
        """
        static = tuple(x.shape)
        height, width = static[1], static[2]
        if height is not None and width is not None:
            # Static fast path: both spatial dims are known at trace time.
            return (
                ops.reshape(x, (-1, height * width, self.dim)),
                (-1, height, width, self.dim),
            )
        # Dynamic path: derive the dims as tensors. NEVER coerce a traced
        # dimension with int()/tuple() — that raises under @tf.function, which is
        # the regime fit() runs in.
        dynamic = ops.shape(x)
        sequence_shape = ops.stack(
            [dynamic[0], dynamic[1] * dynamic[2], dynamic[3]])
        spatial_shape = ops.stack(
            [dynamic[0], dynamic[1], dynamic[2], dynamic[3]])
        return ops.reshape(x, sequence_shape), spatial_shape

    def _to_spatial(self, sequence, spatial_shape):
        """Reshape ``(B, H*W, C)`` back to ``(B, H, W, C)``.

        :param sequence: Rank-3 tensor produced by :meth:`_to_sequence` (or by a
            shape-preserving op applied to it).
        :param spatial_shape: The second element returned by :meth:`_to_sequence`.
        :return: Rank-4 channels-last tensor with the original spatial layout.
        """
        return ops.reshape(sequence, spatial_shape)

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer, then the layer itself.

        :param input_shape: Shape of the input tensor, ``(B, H, W, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel count is not
            ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitAttentionBlock expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channel count must equal dim={self.dim}, "
                f"got {input_shape[-1]}"
            )

        input_shape = tuple(input_shape)
        batch, height, width, channels = input_shape
        tokens = None if (height is None or width is None) else height * width
        # The attention sub-layer validates rank and raises on a 4-D shape, so it
        # must be built with the FLATTENED rank-3 shape (MEASURED, F-6 P-1).
        sequence_shape = (batch, tokens, channels)

        self.norm.build(input_shape)
        self.attn.build(sequence_shape)
        if self.layer_scale_1 is not None:
            self.layer_scale_1.build(input_shape)
        self.drop_path_1.build(input_shape)

        self.mlp.build(input_shape)
        mlp_shape = self.mlp.compute_output_shape(input_shape)
        if self.layer_scale_2 is not None:
            self.layer_scale_2.build(mlp_shape)
        self.drop_path_2.build(mlp_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the FastViT attention block.

        :param inputs: Input tensor of shape ``(B, H, W, dim)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour — ``StochasticDepth`` treats ``None`` as
            training, and a BatchNormalization norm updates its moving statistics.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H, W, dim)``.
        """
        normed = self.norm(inputs, training=training)
        sequence, spatial_shape = self._to_sequence(normed)
        attended = self.attn(sequence, training=training)
        residual = self._to_spatial(attended, spatial_shape)

        if self.layer_scale_1 is not None:
            residual = self.layer_scale_1(residual, training=training)
        residual = self.drop_path_1(residual, training=training)
        x = ops.add(inputs, residual)

        residual = self.mlp(x, training=training)
        if self.layer_scale_2 is not None:
            residual = self.layer_scale_2(residual, training=training)
        residual = self.drop_path_2(residual, training=training)
        return ops.add(x, residual)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to the input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        return input_shape[:-1] + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'mlp_ratio': self.mlp_ratio,
            'head_dim': self.head_dim,
            'normalization_type': self.normalization_type,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'drop_path_rate': self.drop_path_rate,
            'layer_scale_init_value': self.layer_scale_init_value,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitAttentionBlock":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitAttentionBlock` instance.
        :rtype: FastVitAttentionBlock
        """
        config = dict(config)
        config['activation'] = activations.deserialize(config['activation'])
        config['kernel_initializer'] = initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)

# ---------------------------------------------------------------------
