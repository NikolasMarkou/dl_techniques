"""
One stage of the FastViT / MobileCLIP2 MCi backbone.

This module transcribes timm's ``FastVitStage``: the repeating unit the MCi image
tower is built from. A stage is three things in sequence, any of which may be
absent:

.. code-block:: text

    (optional) FastVitPatchEmbed downsample   # /stride, C_in -> dim
    (optional) RepConditionalPosEnc           # depthwise 7x7 + skip
    depth x (FastVitRepMixerBlock | FastVitAttentionBlock)

The token mixer is chosen once per stage: the shallow, high-resolution stages use
the convolutional RepMixer, and the deepest stage(s) — where the feature map is
small enough for quadratic attention — use global self-attention. The reference
never mixes the two kinds inside one stage, and neither does this class.

**Why the stage does NOT compute its own drop-path schedule.**

The reference computes the stochastic-depth schedule ONCE, globally, across every
block of every stage (``calculate_drop_path_rates(drop_path_rate, layers,
stagewise=True)``) and then hands each stage its slice. A per-stage computation
cannot reproduce that: stage 2 of a ``[2, 12, 24, 4]`` model must start where
stage 1's rates ended, not at zero. So ``drop_path_rates`` is an EXPLICIT list of
exactly ``depth`` floats supplied by the caller (the encoder), and the stage's
only job is to hand element ``i`` to block ``i``. That handoff is the one thing in
this class that a shape assertion can never check — a reversed or off-by-one list
produces an identically-shaped, identically-parameterized, subtly-wrong model —
so it carries its own behavioural pin
(``test_per_block_drop_path_wiring``), proven RED against BOTH a reversed and an
off-by-one injection.

**Why the blocks live in a FLAT Python list.**

A nested ``List[List[Layer]]`` attribute silently loses weights on a ``.keras``
round trip on this stack, while the layer count, the variable paths AND the
parameter total all still match — only an ELEMENTWISE weight comparison sees it
(MEASURED, repo-wide). ``self.blocks`` is therefore a single flat list, and the
round trip is pinned elementwise per block by
``test_roundtrip_preserves_block_weights_elementwise``.

.. note::
   ``FastVitRepMixerBlock`` is NOT the pre-existing standalone
   :class:`~dl_techniques.layers.repmixer_block.RepMixerBlock`. See the package
   ``README.md`` for the full disambiguation.

References:
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
    - Vasu et al., 2024. MobileCLIP: Fast Image-Text Models through Multi-Modal
      Reinforced Training. (https://arxiv.org/abs/2311.17049)
"""

import keras
from keras import initializers, regularizers, activations
from typing import Optional, Union, Tuple, List, Sequence, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .attention_block import FastVitAttentionBlock
from .patch_embed import FastVitPatchEmbed
from .rep_conditional_pos_enc import RepConditionalPosEnc
from .rep_mixer import FastVitRepMixerBlock, _REFERENCE_LAYER_SCALE_INIT

# ---------------------------------------------------------------------

#: The two token mixers a stage may be built from. The reference chooses one per
#: stage and never mixes them within a stage.
_TOKEN_MIXERS = ('repmixer', 'attention')


@keras.saving.register_keras_serializable()
class FastVitStage(keras.layers.Layer):
    """One FastViT stage: optional downsample, optional RepCPE, then ``depth`` blocks.

    Channels-last transcription of timm's ``FastVitStage``.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────────┐
        │            Input [B, H, W, C_in]                 │
        └────────────────────┬─────────────────────────────┘
                             ▼
        ┌──────────────────────────────────────────────────┐
        │ downsample: FastVitPatchEmbed                    │  (if downsample)
        │   LKC(k=down_patch_size, s=down_stride, dw)      │
        │   + MobileOneBlock(k=1)                          │
        │   → [B, ceil(H/s), ceil(W/s), dim]               │
        └────────────────────┬─────────────────────────────┘
                             ▼
        ┌──────────────────────────────────────────────────┐
        │ pos_emb: RepConditionalPosEnc                    │  (if use_pos_emb)
        │   dw conv(spatial_shape) + skip                  │
        └────────────────────┬─────────────────────────────┘
                             ▼
        ┌──────────────────────────────────────────────────┐
        │ block_0   drop_path_rates[0]                     │
        │ block_1   drop_path_rates[1]                     │
        │   ...     (FastVitRepMixerBlock | FastVitAttn)   │
        │ block_D-1 drop_path_rates[D-1]                   │
        └────────────────────┬─────────────────────────────┘
                             ▼
        ┌──────────────────────────────────────────────────┐
        │        Output [B, H', W', dim]                   │
        └──────────────────────────────────────────────────┘

    .. note::
       ``StochasticDepth`` short-circuits to the identity only when ``training is
       False`` (or the rate is exactly 0.0); ``training=None`` runs the stochastic
       path. Deterministic tests must pass ``training=False`` EXPLICITLY.

    :param dim: Output channel count of the stage. Every block preserves it. Must
        be positive.
    :type dim: int
    :param depth: Number of token-mixer blocks. Must be positive.
    :type depth: int
    :param token_mixer: Which block type to repeat, ``'repmixer'`` or
        ``'attention'``. Defaults to ``'repmixer'``.
    :type token_mixer: str
    :param downsample: Whether the stage begins with a
        :class:`~dl_techniques.layers.fastvit.patch_embed.FastVitPatchEmbed`.
        ``False`` for the first stage of every MCi variant (the stem has already
        done the /4). Defaults to ``True``.
    :type downsample: bool
    :param se_downsample: Whether the downsample's large-kernel convolution uses a
        squeeze-and-excitation block. Defaults to ``False``.
    :type se_downsample: bool
    :param use_pos_emb: Whether to insert a
        :class:`~dl_techniques.layers.fastvit.rep_conditional_pos_enc.RepConditionalPosEnc`
        after the downsample. Defaults to ``False``.
    :type use_pos_emb: bool
    :param pos_emb_spatial_shape: Depthwise kernel size of the positional encoding.
        Defaults to ``(7, 7)``, the reference value. Ignored when ``use_pos_emb``
        is ``False``.
    :type pos_emb_spatial_shape: Union[int, Tuple[int, int]]
    :param mlp_ratio: ConvMlp expansion ratio inside every block. Defaults to 4.0.
    :type mlp_ratio: float
    :param repmixer_kernel_size: Depthwise kernel size of the RepMixer token mixer.
        Used only when ``token_mixer == 'repmixer'``. Defaults to 3.
    :type repmixer_kernel_size: int
    :param head_dim: Per-head width of the attention token mixer; ``num_heads`` is
        ``dim // head_dim``. Used only when ``token_mixer == 'attention'``.
        Defaults to 32.
    :type head_dim: int
    :param normalization_type: Norm key for the attention block's pre-norm. Used
        only when ``token_mixer == 'attention'``. Defaults to ``'batch_norm'``.
    :type normalization_type: str
    :param down_patch_size: Kernel size of the downsample's large-kernel
        convolution. Defaults to 7.
    :type down_patch_size: int
    :param down_stride: Stride of the downsample. Defaults to 2.
    :type down_stride: int
    :param lkc_use_act: Whether the downsample's large-kernel convolution applies
        its activation. Defaults to ``True``.
    :type lkc_use_act: bool
    :param dropout_rate: Dropout rate inside every block's ConvMlp. Must be in
        ``[0, 1)``. Defaults to 0.0.
    :type dropout_rate: float
    :param drop_path_rates: Per-block stochastic-depth rates — a sequence of
        EXACTLY ``depth`` floats, block ``i`` receiving element ``i``. ``None``
        means all zeros. The stage never computes this schedule itself; see the
        module docstring. Defaults to ``None``.
    :type drop_path_rates: Optional[Sequence[float]]
    :param layer_scale_init_value: Constant initialization for every LayerScale
        gamma, or ``None`` to omit LayerScale. Defaults to ``1e-5``.
    :type layer_scale_init_value: Optional[float]
    :param activation: Activation used inside the blocks and the downsample.
        Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for the convolution / projection
        kernels. Defaults to ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer applied to every kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim``, ``depth``, ``down_patch_size``, ``down_stride``
        or ``repmixer_kernel_size`` are not positive, if ``token_mixer`` is not one
        of ``'repmixer'`` / ``'attention'``, if ``dropout_rate`` is outside
        ``[0, 1)``, or if ``drop_path_rates`` is given with a length other than
        ``depth``.

    Example:
        >>> import numpy as np
        >>> stage = FastVitStage(dim=64, depth=2, token_mixer='repmixer',
        ...                      downsample=True, drop_path_rates=[0.0, 0.1])
        >>> y = stage(np.zeros((2, 16, 16, 32), dtype='float32'), training=False)
        >>> y.shape
        (2, 8, 8, 64)
    """

    def __init__(
            self,
            dim: int,
            depth: int,
            token_mixer: str = 'repmixer',
            downsample: bool = True,
            se_downsample: bool = False,
            use_pos_emb: bool = False,
            pos_emb_spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
            mlp_ratio: float = 4.0,
            repmixer_kernel_size: int = 3,
            head_dim: int = 32,
            normalization_type: str = 'batch_norm',
            down_patch_size: int = 7,
            down_stride: int = 2,
            lkc_use_act: bool = True,
            dropout_rate: float = 0.0,
            drop_path_rates: Optional[Sequence[float]] = None,
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
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if token_mixer not in _TOKEN_MIXERS:
            raise ValueError(
                f"token_mixer must be one of "
                f"{', '.join(repr(name) for name in _TOKEN_MIXERS)}, "
                f"got {token_mixer!r}"
            )
        if repmixer_kernel_size <= 0:
            raise ValueError(
                f"repmixer_kernel_size must be positive, "
                f"got {repmixer_kernel_size}"
            )
        if down_patch_size <= 0:
            raise ValueError(
                f"down_patch_size must be positive, got {down_patch_size}")
        if down_stride <= 0:
            raise ValueError(f"down_stride must be positive, got {down_stride}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        resolved_rates = self._resolve_drop_path_rates(drop_path_rates, depth)

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.depth = depth
        self.token_mixer = token_mixer
        self.downsample_enabled = bool(downsample)
        self.se_downsample = bool(se_downsample)
        self.use_pos_emb = bool(use_pos_emb)
        self.pos_emb_spatial_shape = pos_emb_spatial_shape
        self.mlp_ratio = float(mlp_ratio)
        self.repmixer_kernel_size = repmixer_kernel_size
        self.head_dim = head_dim
        self.normalization_type = normalization_type
        self.down_patch_size = down_patch_size
        self.down_stride = down_stride
        self.lkc_use_act = bool(lkc_use_act)
        self.dropout_rate = dropout_rate
        self.drop_path_rates = resolved_rates
        self.layer_scale_init_value = (
            None if layer_scale_init_value is None else float(layer_scale_init_value)
        )
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        # `downsample` and `pos_emb` are genuinely absent (None) when disabled
        # rather than being an Identity: the reference substitutes nn.Identity,
        # and a None attribute makes "is this stage a downsampling stage?"
        # answerable by inspection instead of by counting weights.
        self.downsample = (
            FastVitPatchEmbed(
                embed_dim=self.dim,
                patch_size=self.down_patch_size,
                stride=self.down_stride,
                use_se=self.se_downsample,
                lkc_use_act=self.lkc_use_act,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='downsample',
            )
            if self.downsample_enabled else None
        )
        self.pos_emb = (
            RepConditionalPosEnc(
                dim=self.dim,
                spatial_shape=self.pos_emb_spatial_shape,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='pos_emb',
            )
            if self.use_pos_emb else None
        )

        # FLAT list — a nested List[List[Layer]] silently drops weights on a
        # .keras round trip while counts, paths and parameter totals all match.
        # Block i receives drop_path_rates[i]; nothing else in this class is as
        # easy to get subtly and invisibly wrong.
        self.blocks: List[keras.layers.Layer] = [
            self._create_block(index) for index in range(self.depth)
        ]

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_drop_path_rates(
            drop_path_rates: Optional[Sequence[float]],
            depth: int
    ) -> List[float]:
        """Validate ``drop_path_rates`` and normalize it to a list of ``depth`` floats.

        :param drop_path_rates: ``None`` (meaning all zeros) or a sequence of
            exactly ``depth`` numbers.
        :type drop_path_rates: Optional[Sequence[float]]
        :param depth: Number of blocks in the stage.
        :type depth: int
        :return: A list of ``depth`` floats.
        :rtype: List[float]
        :raises ValueError: If the sequence length differs from ``depth``, or if
            any entry is not a real number.
        """
        if drop_path_rates is None:
            return [0.0] * depth
        try:
            rates = list(drop_path_rates)
        except TypeError:
            raise ValueError(
                f"drop_path_rates must be None or a sequence of {depth} floats, "
                f"got {drop_path_rates!r}"
            )
        if len(rates) != depth:
            raise ValueError(
                f"drop_path_rates must have exactly one entry per block: "
                f"depth={depth} but len(drop_path_rates)={len(rates)}. The stage "
                f"does NOT compute the schedule itself — the encoder slices the "
                f"global stagewise schedule and passes this stage its slice."
            )
        for rate in rates:
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise ValueError(
                    f"drop_path_rates entries must be real numbers, got {rate!r}"
                )
        return [float(rate) for rate in rates]

    def _create_block(self, index: int) -> keras.layers.Layer:
        """Create block ``index`` of the requested token-mixer type.

        :param index: Position of the block inside the stage; also the index into
            ``drop_path_rates``.
        :type index: int
        :return: An unbuilt block layer.
        :rtype: keras.layers.Layer
        """
        # Names are `block_{i}` and NOT derived from the token mixer, so the
        # variable paths are stable across a repmixer/attention configuration
        # change and a checkpoint's layout does not depend on it.
        name = f'block_{index}'
        drop_path_rate = self.drop_path_rates[index]

        if self.token_mixer == 'repmixer':
            return FastVitRepMixerBlock(
                dim=self.dim,
                kernel_size=self.repmixer_kernel_size,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=self.layer_scale_init_value,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=name,
            )
        return FastVitAttentionBlock(
            dim=self.dim,
            mlp_ratio=self.mlp_ratio,
            head_dim=self.head_dim,
            normalization_type=self.normalization_type,
            dropout_rate=self.dropout_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=self.layer_scale_init_value,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=name,
        )

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer in FORWARD order, then the layer itself.

        The downsample changes both spatial dimensions and the channel count, so
        the positional encoding and every block must be built on the DOWNSAMPLED
        shape — not on ``input_shape``.

        :param input_shape: Shape of the input tensor, ``(B, H, W, C_in)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4, or if the stage has no
            downsample and the input channel count is not ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitStage expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        input_shape = tuple(input_shape)

        if self.downsample is not None:
            self.downsample.build(input_shape)
            current_shape = self.downsample.compute_output_shape(input_shape)
        else:
            if input_shape[-1] is not None and input_shape[-1] != self.dim:
                raise ValueError(
                    f"A stage without a downsample cannot change the channel "
                    f"count: input channels must equal dim={self.dim}, got "
                    f"{input_shape[-1]}. Set downsample=True to project them."
                )
            current_shape = input_shape

        if self.pos_emb is not None:
            self.pos_emb.build(current_shape)
            current_shape = self.pos_emb.compute_output_shape(current_shape)

        for block in self.blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the stage.

        :param inputs: Input tensor of shape ``(B, H, W, C_in)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour — every block contains BatchNormalization and
            a stochastic-depth branch that treats ``None`` as training.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H', W', dim)``.
        """
        x = inputs
        if self.downsample is not None:
            x = self.downsample(x, training=training)
        if self.pos_emb is not None:
            x = self.pos_emb(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        Only the downsample changes the shape; the positional encoding and every
        block preserve it.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(B, ceil(H/down_stride), ceil(W/down_stride), dim)`` when the
            stage downsamples, else ``(B, H, W, dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        if not self.downsample_enabled:
            return input_shape[:-1] + (self.dim,)
        stride = self.down_stride
        height = (
            None if input_shape[1] is None
            else (input_shape[1] + stride - 1) // stride
        )
        width = (
            None if input_shape[2] is None
            else (input_shape[2] + stride - 1) // stride
        )
        return (input_shape[0], height, width, self.dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'depth': self.depth,
            'token_mixer': self.token_mixer,
            'downsample': self.downsample_enabled,
            'se_downsample': self.se_downsample,
            'use_pos_emb': self.use_pos_emb,
            'pos_emb_spatial_shape': self.pos_emb_spatial_shape,
            'mlp_ratio': self.mlp_ratio,
            'repmixer_kernel_size': self.repmixer_kernel_size,
            'head_dim': self.head_dim,
            'normalization_type': self.normalization_type,
            'down_patch_size': self.down_patch_size,
            'down_stride': self.down_stride,
            'lkc_use_act': self.lkc_use_act,
            'dropout_rate': self.dropout_rate,
            'drop_path_rates': list(self.drop_path_rates),
            'layer_scale_init_value': self.layer_scale_init_value,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitStage":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitStage` instance.
        :rtype: FastVitStage
        """
        config = dict(config)
        spatial_shape = config.get('pos_emb_spatial_shape')
        if isinstance(spatial_shape, list):
            config['pos_emb_spatial_shape'] = tuple(spatial_shape)
        config['activation'] = activations.deserialize(config['activation'])
        config['kernel_initializer'] = initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)

# ---------------------------------------------------------------------
