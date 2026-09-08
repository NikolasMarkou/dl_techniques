"""Restormer's three composite building blocks, as used by DocRes.

DocRes's network is an unmodified Restormer, and Restormer is built from
exactly three composites above the layer level:

* :class:`RestormerTransformerBlock` -- pre-norm MDTA, pre-norm GDFN, one
  residual add after each;
* :class:`RestormerDownsample` -- a bias-free ``3x3`` convolution that HALVES
  the channel count, followed by a ``PixelUnshuffle2D(2)`` that multiplies it
  by four; net ``n -> 2n`` channels at half resolution;
* :class:`RestormerUpsample` -- its mirror; net ``n -> n/2`` channels at double
  resolution.

The two attention/FFN primitives the block composes,
:class:`~dl_techniques.layers.attention.multi_dconv_head_transposed_attention.MultiDconvHeadTransposedAttention`
and
:class:`~dl_techniques.layers.ffn.gated_dconv_ffn.GatedDConvFeedForward`,
live in the shared ``layers/`` packages and are imported DIRECTLY here rather
than through ``create_attention_layer`` / ``create_ffn_layer``. That is a
deliberate exception to the factory-first rule in ``layers/CLAUDE.md``: the
factories exist so a caller can make a *configurable slot* -- "put whatever
attention the config names here" -- and these are not slots. Restormer's block
is MDTA and GDFN by definition; substituting either produces a different
published network, not a variant of this one. The block therefore names them,
and the only knobs it forwards (``num_heads``, ``ffn_expansion_factor``,
``use_bias``) are the ones the reference itself exposes.

References:
    - Zamir et al., 2022. Restormer: Efficient Transformer for High-Resolution
      Image Restoration. CVPR 2022. (https://arxiv.org/abs/2111.09881)
      ``basicsr/models/archs/restormer_arch.py:171-189`` is the Downsample /
      Upsample pair transcribed below; ``:126-137`` is the transformer block.
    - Zhang et al., 2024. DocRes: A Generalist Model Toward Unifying Document
      Image Restoration Tasks. CVPR 2024. (https://arxiv.org/abs/2405.04408)
"""

import keras
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.attention.multi_dconv_head_transposed_attention import (
    MultiDconvHeadTransposedAttention,
)
from dl_techniques.layers.ffn.gated_dconv_ffn import GatedDConvFeedForward
from dl_techniques.layers.pooling.pixel_unshuffle import (
    PixelShuffle2D,
    PixelUnshuffle2D,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# The upstream LayerNorm epsilon.
#
# `WithBias_LayerNorm` in `restormer_arch.py:66-77` divides by
# `sqrt(sigma + 1e-5)`. Keras' `LayerNormalization` DEFAULTS to `epsilon=1e-3`
# -- 100x larger, with no shape symptom and no warning -- so every construction
# site below passes `epsilon=` explicitly. `layers/CLAUDE.md` § "The factory
# contract", rule 5, requires exactly this: a directly-constructed
# normalization layer states its epsilon with a cited reference.
# ---------------------------------------------------------------------

RESTORMER_LAYERNORM_EPSILON: float = 1e-5

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.doc_res.components")
class RestormerTransformerBlock(keras.layers.Layer):
    """One Restormer transformer block: pre-norm MDTA then pre-norm GDFN.

    Both sub-blocks are wrapped in a residual add, and both normalizations are
    applied BEFORE their sub-block (the pre-norm arrangement), so the residual
    path from input to output carries no normalization at all:

    .. code-block:: text

        inputs [B, H, W, dim]
           │
           ├───────────────────────────────┐
           │                               │
           ▼                               │
        LayerNorm(axis=-1, eps=1e-5)       │
           ▼                               │
        MultiDconvHeadTransposedAttention  │
           ▼                               │
          (+)  ◄───────────────────────────┘
           │
           ├───────────────────────────────┐
           │                               │
           ▼                               │
        LayerNorm(axis=-1, eps=1e-5)       │
           ▼                               │
        GatedDConvFeedForward              │
           ▼                               │
          (+)  ◄───────────────────────────┘
           │
           ▼
        output [B, H, W, dim]

    Shape is preserved end to end; the block is a residual refinement, never a
    resampler.

    :param dim: Channel width of the block's input and output. Must be positive
        and divisible by ``num_heads``.
    :type dim: int
    :param num_heads: Number of MDTA attention heads. In DocRes this is
        ``[1, 2, 4, 8]`` across the four encoder levels, which keeps the
        per-head channel width at 48 everywhere.
    :type num_heads: int
    :param ffn_expansion_factor: GDFN hidden-width multiplier. The hidden width
        is ``int(dim * ffn_expansion_factor)`` with Python truncation, so 2.66
        gives 127 at ``dim=48``, not 128. Defaults to 2.66, the DocRes value.
    :type ffn_expansion_factor: float
    :param use_bias: Whether the convolutions inside MDTA and GDFN carry a
        bias. Defaults to ``False``, which is what every Restormer and DocRes
        call site uses.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``dim`` or ``num_heads`` is not positive, or if
        ``dim`` is not divisible by ``num_heads`` (both raised by
        ``MultiDconvHeadTransposedAttention``), or if
        ``ffn_expansion_factor`` is not positive.
    :raises ValueError: From ``build()``, if the input is not rank 4 or its
        trailing dimension does not equal ``dim``.

    Input shape:
        4D tensor ``(batch, height, width, dim)``. ``height`` and ``width`` may
        be ``None``.

    Output shape:
        4D tensor ``(batch, height, width, dim)`` -- identical to the input.

    :ivar norm1: Pre-attention normalization.
    :vartype norm1: keras.layers.LayerNormalization
    :ivar attn: The MDTA sub-block.
    :vartype attn: MultiDconvHeadTransposedAttention
    :ivar norm2: Pre-FFN normalization.
    :vartype norm2: keras.layers.LayerNormalization
    :ivar ffn: The GDFN sub-block.
    :vartype ffn: GatedDConvFeedForward

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.models.vision.image_restoration.doc_res.components \
            import RestormerTransformerBlock

        x = keras.random.normal((2, 32, 32, 96))
        y = RestormerTransformerBlock(dim=96, num_heads=2)(x)
        # y.shape == (2, 32, 32, 96)

    References:
        - Zamir et al., 2022. Restormer. CVPR 2022,
          ``restormer_arch.py:126-137``.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            ffn_expansion_factor: float = 2.66,
            use_bias: bool = False,
            **kwargs: Any
    ) -> None:
        """Create the two normalizations and the two sub-blocks.

        :param dim: Channel width of input and output.
        :type dim: int
        :param num_heads: Number of MDTA heads.
        :type num_heads: int
        :param ffn_expansion_factor: GDFN hidden-width multiplier.
        :type ffn_expansion_factor: float
        :param use_bias: Whether MDTA's and GDFN's convolutions carry a bias.
        :type use_bias: bool
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: On any invalid ``dim`` / ``num_heads`` /
            ``ffn_expansion_factor`` combination; the two sub-blocks raise.
        """
        super().__init__(**kwargs)

        self.dim = dim
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.use_bias = use_bias

        # `epsilon` is passed explicitly at BOTH sites: Keras defaults to
        # 1e-3, upstream `WithBias_LayerNorm` uses 1e-5
        # (`restormer_arch.py:77`). See RESTORMER_LAYERNORM_EPSILON above.
        self.norm1 = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=RESTORMER_LAYERNORM_EPSILON,
            name="norm1",
        )
        self.attn = MultiDconvHeadTransposedAttention(
            dim=dim,
            num_heads=num_heads,
            use_bias=use_bias,
            name="attn",
        )
        self.norm2 = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=RESTORMER_LAYERNORM_EPSILON,
            name="norm2",
        )
        self.ffn = GatedDConvFeedForward(
            dim=dim,
            ffn_expansion_factor=ffn_expansion_factor,
            use_bias=use_bias,
            name="ffn",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all four sub-layers explicitly.

        Every sub-layer ``call()`` runs is built here, so all weight variables
        exist before weight restoration during model loading.

        :param input_shape: Shape tuple, expected ``(batch, height, width, dim)``.
        :type input_shape: tuple

        :raises ValueError: If ``input_shape`` is not rank 4, or its trailing
            dimension does not equal ``dim``.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Expected input channels ({input_shape[-1]}) to match "
                f"block dim ({self.dim})"
            )

        # Shape is preserved through the whole block, so one shape builds all
        # four sub-layers.
        self.norm1.build(input_shape)
        self.attn.build(input_shape)
        self.norm2.build(input_shape)
        self.ffn.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the two pre-norm residual sub-blocks in order.

        :param inputs: Tensor of shape ``(batch, height, width, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag, forwarded to the two
            sub-blocks.
        :type training: Optional[bool]
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        x = inputs + self.attn(self.norm1(inputs), training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``input_shape`` unchanged: the block is shape-preserving.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: The same shape.
        :rtype: tuple
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to recreate this block.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "use_bias": self.use_bias,
        })
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.doc_res.components")
class RestormerDownsample(keras.layers.Layer):
    """Halve the resolution and DOUBLE the channel width, ``n -> 2n``.

    The composition is a bias-free ``3x3`` convolution that halves the channel
    count, followed by a pixel-unshuffle at scale 2 that multiplies it by four:

    .. code-block:: text

        inputs [B, H, W, n]
            ▼
        Conv2D(n // 2, 3x3, padding='same', use_bias=False)
            │  [B, H, W, n/2]
            ▼
        PixelUnshuffle2D(scale=2)
            │  [B, H/2, W/2, (n/2) * 4]
            ▼
        output [B, H/2, W/2, 2n]

    The NET effect -- ``n -> 2n`` channels, ``H, W -> H/2, W/2`` -- is what
    produces the ``int(dim * 2**k)`` width progression in Restormer's encoder.
    Reading only the convolution ("it halves the channels") gets the sign of
    the change backwards.

    :param n_feat: Input channel count. Must be positive and even, since the
        convolution emits ``n_feat // 2`` channels.
    :type n_feat: int
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``n_feat`` is not a positive even integer.
    :raises ValueError: From ``build()``, if the input is not rank 4 or its
        trailing dimension does not equal ``n_feat``.

    Input shape:
        4D tensor ``(batch, height, width, n_feat)``. ``height`` and ``width``
        may be ``None`` at build time but must be even at call time.

    Output shape:
        4D tensor ``(batch, height // 2, width // 2, n_feat * 2)``.

    :ivar conv: The bias-free channel-halving convolution.
    :vartype conv: keras.layers.Conv2D
    :ivar shuffle: The pixel-unshuffle.
    :vartype shuffle: PixelUnshuffle2D

    Example:

    .. code-block:: python

        x = keras.random.normal((2, 32, 32, 48))
        y = RestormerDownsample(n_feat=48)(x)
        # y.shape == (2, 16, 16, 96)

    References:
        - Zamir et al., 2022. Restormer. CVPR 2022,
          ``restormer_arch.py:171-179``.
    """

    def __init__(self, n_feat: int, **kwargs: Any) -> None:
        """Create the convolution and the pixel-unshuffle.

        :param n_feat: Input channel count; must be positive and even.
        :type n_feat: int
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any
        :raises ValueError: If ``n_feat`` is not a positive even integer.
        """
        super().__init__(**kwargs)

        if n_feat <= 0:
            raise ValueError(f"n_feat must be positive, got {n_feat}")
        if n_feat % 2 != 0:
            raise ValueError(
                f"n_feat must be even so the convolution can emit n_feat // 2 "
                f"channels, got {n_feat}"
            )

        self.n_feat = n_feat

        # `use_bias=False` is HARD-CODED, not forwarded from an outer flag.
        # Upstream's `Downsample.__init__` takes only `n_feat` and writes
        # `bias=False` on this conv (`restormer_arch.py:175`) even when the
        # surrounding `Restormer(bias=...)` argument is True. This layer
        # therefore exposes no `use_bias` parameter at all: adding one and
        # threading it here would make the port configurable in a way the
        # reference is not, and would silently change the parameter count of
        # every level. Do not "fix" this into consistency with the blocks'
        # `use_bias`. Guarded by
        # `test_components.py::test_every_convolution_is_bias_free`.
        self.conv = keras.layers.Conv2D(
            filters=n_feat // 2,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="conv",
        )

        # DECISION plan-2026-09-08T111844-de235227/D-007: this repo's
        # `PixelUnshuffle2D` uses a DIFFERENT channel-block ordering from
        # `torch.nn.PixelUnshuffle`. MEASURED (D-007, r=2): the repo maps
        # spatial offset (i, j) of source channel c to output channel
        # `(i*r + j) * C_in + c`, while torch maps it to `c*r*r + i*r + j`;
        # 10 of 12 positions differ at C_in=3. The two agree only in the
        # degenerate C_in == 1 case, so any guard written at one channel
        # proves nothing.
        #
        # Harmless HERE, and only here, because this site is BRACKETED by
        # learnable per-channel-parameterised ops: the `conv` above owns a
        # separate filter per output channel, and every consumer of this
        # layer's output in `doc_res/model.py` is either another Conv2D or a
        # `LayerNormalization` with learnable per-channel gamma/beta. A fixed
        # permutation of the channel axis is exactly absorbable by permuting
        # the following op's per-input-channel weights, so the two orderings
        # span the identical function family WHEN TRAINING FROM SCRATCH.
        # `test_components.py::test_a_channel_permutation_is_absorbed_by_*`
        # measures that absorption bit-identically rather than asserting it.
        #
        # LOAD-BEARING CONSEQUENCE: the absorption argument holds only for
        # from-scratch training. It makes this port INCOMPATIBLE WITH LOADING
        # UPSTREAM DOCRES PYTORCH WEIGHTS -- a transferred checkpoint would
        # feed every downstream conv a permuted channel axis it was never
        # trained on. That is why `pretrained=True` raises
        # `NotImplementedError`. If weight transfer is ever wanted, the fix is
        # an explicit permutation layer at each of these sites, NOT a tweak
        # here. See decisions.md D-007.
        self.shuffle = PixelUnshuffle2D(scale=2, name="pixel_unshuffle")

    @property
    def op_sequence(self) -> List[keras.layers.Layer]:
        """The sub-layers ``call()`` applies, in application order.

        ``call()`` iterates THIS list, so a test that reads it is reading the
        real forward path rather than a restatement of it. That is what lets
        the D-007 bracketing guard assert an adjacency ("the op immediately
        before the pixel-unshuffle is per-channel parameterised") instead of
        grepping the source text.

        :return: ``[conv, shuffle]``.
        :rtype: List[keras.layers.Layer]
        """
        return [self.conv, self.shuffle]

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the convolution and the pixel-unshuffle.

        :param input_shape: Shape tuple, expected ``(batch, height, width, n_feat)``.
        :type input_shape: tuple
        :raises ValueError: If ``input_shape`` is not rank 4 or its trailing
            dimension does not equal ``n_feat``.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.n_feat:
            raise ValueError(
                f"Expected input channels ({input_shape[-1]}) to match "
                f"n_feat ({self.n_feat})"
            )

        self.conv.build(input_shape)
        self.shuffle.build(
            (input_shape[0], input_shape[1], input_shape[2], self.n_feat // 2)
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Halve the resolution and double the channel width.

        :param inputs: Tensor of shape ``(batch, height, width, n_feat)``.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag, forwarded to each op.
        :type training: Optional[bool]
        :return: Tensor of shape
            ``(batch, height // 2, width // 2, n_feat * 2)``.
        :rtype: keras.KerasTensor
        """
        x = inputs
        for op in self.op_sequence:
            x = op(x, training=training)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the halved-resolution, doubled-width output shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: ``(batch, height // 2, width // 2, n_feat * 2)``.
        :rtype: tuple
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        return (
            batch,
            None if height is None else height // 2,
            None if width is None else width // 2,
            self.n_feat * 2,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to recreate this layer.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({"n_feat": self.n_feat})
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.doc_res.components")
class RestormerUpsample(keras.layers.Layer):
    """Double the resolution and HALVE the channel width, ``n -> n/2``.

    The mirror of :class:`RestormerDownsample`: a bias-free ``3x3``
    convolution that doubles the channel count, followed by a pixel-shuffle at
    scale 2 that divides it by four.

    .. code-block:: text

        inputs [B, H, W, n]
            ▼
        Conv2D(n * 2, 3x3, padding='same', use_bias=False)
            │  [B, H, W, 2n]
            ▼
        PixelShuffle2D(block_size=2)
            │  [B, 2H, 2W, 2n / 4]
            ▼
        output [B, 2H, 2W, n/2]

    :param n_feat: Input channel count. Must be positive and even, since the
        pixel-shuffle divides ``2 * n_feat`` by four.
    :type n_feat: int
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``n_feat`` is not a positive even integer.
    :raises ValueError: From ``build()``, if the input is not rank 4 or its
        trailing dimension does not equal ``n_feat``.

    Input shape:
        4D tensor ``(batch, height, width, n_feat)``.

    Output shape:
        4D tensor ``(batch, height * 2, width * 2, n_feat // 2)``.

    :ivar conv: The bias-free channel-doubling convolution.
    :vartype conv: keras.layers.Conv2D
    :ivar shuffle: The pixel-shuffle.
    :vartype shuffle: PixelShuffle2D

    Example:

    .. code-block:: python

        x = keras.random.normal((2, 16, 16, 96))
        y = RestormerUpsample(n_feat=96)(x)
        # y.shape == (2, 32, 32, 48)

    References:
        - Zamir et al., 2022. Restormer. CVPR 2022,
          ``restormer_arch.py:181-189``.
    """

    def __init__(self, n_feat: int, **kwargs: Any) -> None:
        """Create the convolution and the pixel-shuffle.

        :param n_feat: Input channel count; must be positive and even.
        :type n_feat: int
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any
        :raises ValueError: If ``n_feat`` is not a positive even integer.
        """
        super().__init__(**kwargs)

        if n_feat <= 0:
            raise ValueError(f"n_feat must be positive, got {n_feat}")
        if n_feat % 2 != 0:
            raise ValueError(
                f"n_feat must be even so the pixel-shuffle can divide "
                f"2 * n_feat by four, got {n_feat}"
            )

        self.n_feat = n_feat

        # `use_bias=False` is HARD-CODED here for the same reason as in
        # `RestormerDownsample`: upstream's `Upsample.__init__` takes only
        # `n_feat` and writes `bias=False` (`restormer_arch.py:185`)
        # regardless of the outer `Restormer(bias=...)` argument. Do not add a
        # `use_bias` parameter and thread it here.
        self.conv = keras.layers.Conv2D(
            filters=n_feat * 2,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="conv",
        )

        # DECISION plan-2026-09-08T111844-de235227/D-007: this repo's
        # `PixelShuffle2D` uses a DIFFERENT channel-block ordering from
        # `torch.nn.PixelShuffle`. MEASURED (D-007, r=2, C'=3): the repo reads
        # output channel c' at spatial offset (i, j) from input channel
        # `(i*r + j) * C' + c'`, while torch reads it from `c'*r*r + i*r + j`;
        # 10 of 12 positions differ. The two agree only when C' == 1, so a
        # single-channel guard would be blind to the difference. The repo's
        # `PixelShuffle2D` IS an exact round-trip inverse of its own
        # `PixelUnshuffle2D` (measured max|diff| = 0.0 both ways) -- it is
        # internally consistent, just not torch-ordered.
        #
        # Harmless HERE, and only here, for the bracketing reason argued at
        # `RestormerDownsample.shuffle`: the `conv` above is learnable
        # per-output-channel and every consumer downstream is a Conv2D or a
        # `LayerNormalization` with learnable per-channel gamma/beta, either of
        # which absorbs a fixed channel permutation exactly by permuting its
        # own per-input-channel weights. Measured, not asserted, by
        # `test_components.py::test_a_channel_permutation_is_absorbed_by_*`.
        #
        # LOAD-BEARING CONSEQUENCE: this makes the port INCOMPATIBLE WITH
        # TRANSFERRING UPSTREAM DOCRES PYTORCH WEIGHTS. The absorption holds
        # only because the surrounding weights are LEARNED under this
        # ordering; an imported checkpoint was learned under torch's. That is
        # why `pretrained=True` raises `NotImplementedError`. The fix, should
        # weight transfer ever be wanted, is an explicit permutation layer at
        # each of these sites. See decisions.md D-007.
        self.shuffle = PixelShuffle2D(block_size=2, name="pixel_shuffle")

    @property
    def op_sequence(self) -> List[keras.layers.Layer]:
        """The sub-layers ``call()`` applies, in application order.

        See :meth:`RestormerDownsample.op_sequence` -- ``call()`` iterates this
        list, so the D-007 bracketing guard reads the real forward path.

        :return: ``[conv, shuffle]``.
        :rtype: List[keras.layers.Layer]
        """
        return [self.conv, self.shuffle]

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the convolution and the pixel-shuffle.

        :param input_shape: Shape tuple, expected ``(batch, height, width, n_feat)``.
        :type input_shape: tuple
        :raises ValueError: If ``input_shape`` is not rank 4 or its trailing
            dimension does not equal ``n_feat``.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.n_feat:
            raise ValueError(
                f"Expected input channels ({input_shape[-1]}) to match "
                f"n_feat ({self.n_feat})"
            )

        self.conv.build(input_shape)
        self.shuffle.build(
            (input_shape[0], input_shape[1], input_shape[2], self.n_feat * 2)
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Double the resolution and halve the channel width.

        :param inputs: Tensor of shape ``(batch, height, width, n_feat)``.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag, forwarded to each op.
        :type training: Optional[bool]
        :return: Tensor of shape
            ``(batch, height * 2, width * 2, n_feat // 2)``.
        :rtype: keras.KerasTensor
        """
        x = inputs
        for op in self.op_sequence:
            x = op(x, training=training)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the doubled-resolution, halved-width output shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: ``(batch, height * 2, width * 2, n_feat // 2)``.
        :rtype: tuple
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        return (
            batch,
            None if height is None else height * 2,
            None if width is None else width * 2,
            self.n_feat // 2,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to recreate this layer.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({"n_feat": self.n_feat})
        return config

# ---------------------------------------------------------------------
