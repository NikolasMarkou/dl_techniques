"""
PW-FNet: image restoration with a Fourier token mixer.

Defines :class:`PW_FNet`, a U-Net for tasks like deraining, deblurring and
dehazing, built from :class:`PW_FNet_Block`, :class:`PWFNetDownsample` and
:class:`PWFNetUpsample`. Each block mixes tokens in the frequency domain
instead of with self-attention: a pointwise convolution, an FFT, a pointwise
convolution over the concatenated real and imaginary parts, then an inverse
FFT, which reaches global context at ``O(N log N)`` cost instead of the
``O(N^2)`` of attention. The encoder and decoder have two levels, and the
model returns three tensors, one per scale, for hierarchical supervision;
each one is the input at that scale plus a predicted residual. Normalization
and the block feed-forward network come from the layer factories. The number
of levels is fixed and cannot be changed through the block-count lists.

References:
    - Jiang et al., 2025. Global Modeling Matters: A Fast, Lightweight and
      Effective Baseline for Efficient Image Restoration (PW-FNet).
      (https://arxiv.org/abs/2507.13663)
    - Lee-Thorp et al., 2021. FNet: Mixing Tokens with Fourier Transforms.
      (https://arxiv.org/abs/2105.03824)
    - Chen et al., 2022. Simple Baselines for Image Restoration (NAFNet).
      ECCV 2022. (https://arxiv.org/abs/2204.04676)
    - Ronneberger et al., 2015. U-Net.
      (https://arxiv.org/abs/1505.04597)
"""

import keras
from keras import ops
from typing import Optional, Tuple, List, Dict, Any


# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.signal_processing.fft_layers import FFTLayer, IFFTLayer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------
# Core PW-FNet Block
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.pw_fnet.model")
class PW_FNet_Block(keras.layers.Layer):
    """Mix tokens in the frequency domain, then apply a feed-forward network.

    The block has two residual stages: normalization plus a Fourier token
    mixer, then normalization plus a feed-forward network. The token mixer
    stands in for self-attention and reaches global context at ``O(N log N)``
    cost instead of ``O(N^2)``. Both residuals are taken from the tensor
    entering the stage, before its normalization. The normalization type comes
    from the normalization factory. The feed-forward network is either the
    spatial one with a depthwise convolution or one built by the FFN factory.

    Block:

    .. code-block:: text

        input x [B, H, W, C]
              │
              ├───────────────────────────────────────┐ residual
              ▼                                       │
        ┌────────────────┐                            │
        │ norm1          │                            │
        └────────────────┘                            │
              │                                       │
              ▼                                       │
        ┌────────────────┐                            │
        │ token mixer    │                            │
        └────────────────┘                            │
              │                                       │
              ▼                                       │
              + ◄─────────────────────────────────────┘
              │
              ├───────────────────────────────────────┐ residual
              ▼                                       │
        ┌────────────────┐                            │
        │ norm2          │                            │
        └────────────────┘                            │
              │                                       │
              ▼                                       │
        ┌────────────────┐                            │
        │ ffn            │  (spatial or factory)      │
        └────────────────┘                            │
              │                                       │
              ▼                                       │
              + ◄─────────────────────────────────────┘
              │
              ▼
        output [B, H, W, C]

    Token mixer:

    .. code-block:: text

        norm1 output [B, H, W, C]
              │
              ▼
        ┌────────────────────┐
        │ conv 1x1  expand   │
        └────────────────────┘
              │ [B, H, W, hid]
              ▼
        ┌────────────────────┐
        │ fft                │
        └────────────────────┘
              │ [B, H, W, 2*hid]  real and imag concatenated
              ▼
        ┌────────────────────┐
        │ conv 1x1  freq     │
        └────────────────────┘
              │ [B, H, W, 2*hid]
              ▼
           gelu
              │
              ▼
        ┌────────────────────┐
        │ ifft               │
        └────────────────────┘
              │ [B, H, W, hid]
              ▼
        ┌────────────────────┐
        │ conv 1x1  project  │
        └────────────────────┘
              │
              ▼
           [B, H, W, C]

    hid is int(dim * ffn_expansion_factor).

    Feed-forward network:

    .. code-block:: text

        norm2 output [B, H, W, C]
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
        ┌──────────────┐  ┌──────────────────┐
        │ conv 1x1 hid │  │ ffn(ffn_type)    │
        │ dwconv 3x3   │  └──────────────────┘
        │ gelu         │
        │ conv 1x1 -> C│
        └──────────────┘
          (default)        (use_spatial_ffn=False)

    Both paths return C channels.

    :param dim: Number of input and output channels. Must be positive.
    :param ffn_expansion_factor: Expansion factor for the hidden dimension in
        the FFN and the token mixer. Sets capacity and cost. Must be positive.
        Defaults to 2.0.
    :param normalization_type: Type of normalization to use. Accepts any type
        from the normalization factory: 'layer_norm', 'rms_norm',
        'zero_centered_rms_norm', 'band_rms', etc. Defaults to 'layer_norm'.
    :param norm1_kwargs: Optional arguments for the first normalization layer,
        the one before the token mixer. Passed to the normalization factory.
    :param norm2_kwargs: Optional arguments for the second normalization layer,
        the one before the FFN. Passed to the normalization factory.
    :param use_spatial_ffn: If True, use the spatial FFN with a depthwise
        convolution, which is the original architecture and the recommended
        setting for image restoration. If False, use a factory FFN.
        Defaults to True.
    :param ffn_type: Type of FFN to use when use_spatial_ffn=False. Options
        include 'mlp', 'swiglu', 'geglu', 'glu', 'differential', 'residual',
        etc. Required when use_spatial_ffn=False, ignored otherwise.
    :param ffn_kwargs: Optional arguments for the factory FFN, used only when
        use_spatial_ffn=False. Passed to the FFN factory.
    :param **kwargs: Additional arguments for the Layer base class.

    :raises ValueError: If dim or ffn_expansion_factor is not positive, or if
        use_spatial_ffn=False and ffn_type is not given.

    Example:
        >>> # Default configuration (original architecture)
        >>> block = PW_FNet_Block(dim=64)
        >>>
        >>> # With RMS normalization
        >>> block = PW_FNet_Block(
        ...     dim=64,
        ...     normalization_type='rms_norm',
        ...     norm1_kwargs={'epsilon': 1e-6, 'use_scale': True}
        ... )
        >>>
        >>> # With factory FFN
        >>> block = PW_FNet_Block(
        ...     dim=64,
        ...     use_spatial_ffn=False,
        ...     ffn_type='swiglu',
        ...     ffn_kwargs={'dropout_rate': 0.1}
        ... )
    """

    def __init__(
            self,
            dim: int,
            ffn_expansion_factor: float = 2.0,
            normalization_type: str = 'layer_norm',
            norm1_kwargs: Optional[Dict[str, Any]] = None,
            norm2_kwargs: Optional[Dict[str, Any]] = None,
            use_spatial_ffn: bool = True,
            ffn_type: Optional[str] = None,
            ffn_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Create the two normalization layers, the token mixer and the FFN.

        Arguments are documented on the class.

        :raises ValueError: If dim or ffn_expansion_factor is not positive, or
            if use_spatial_ffn=False and ffn_type is not given.
        """
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if ffn_expansion_factor <= 0:
            raise ValueError(
                f"ffn_expansion_factor must be positive, got {ffn_expansion_factor}"
            )
        if not use_spatial_ffn and ffn_type is None:
            raise ValueError(
                "ffn_type must be specified when use_spatial_ffn=False"
            )

        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.normalization_type = normalization_type
        self.norm1_kwargs = norm1_kwargs or {}
        self.norm2_kwargs = norm2_kwargs or {}
        self.use_spatial_ffn = use_spatial_ffn
        self.ffn_type = ffn_type
        self.ffn_kwargs = ffn_kwargs or {}

        hidden_dim = int(dim * ffn_expansion_factor)
        # Kept for introspection; call() uses the sub-layer shapes.
        self._hidden_dim = hidden_dim

        self.norm1 = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm1",
            **self.norm1_kwargs
        )
        self.norm2 = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm2",
            **self.norm2_kwargs
        )

        self.token_mixer_expand = keras.layers.Conv2D(
            hidden_dim,
            kernel_size=1,
            use_bias=True,
            name="token_mixer_expand"
        )
        self.fft = FFTLayer(name="fft")
        # The FFT output holds the real and imaginary parts concatenated.
        self.freq_conv = keras.layers.Conv2D(
            hidden_dim * 2,
            kernel_size=1,
            use_bias=True,
            name="freq_conv"
        )
        self.ifft = IFFTLayer(name="ifft")
        self.token_mixer_project = keras.layers.Conv2D(
            dim,
            kernel_size=1,
            use_bias=True,
            name="token_mixer_project"
        )

        if self.use_spatial_ffn:
            self._setup_spatial_ffn(hidden_dim)
        else:
            self._setup_factory_ffn(hidden_dim)

    def _setup_spatial_ffn(self, hidden_dim: int) -> None:
        """Create the spatial FFN layers, the default configuration.

        Its depthwise convolution keeps spatial structure, which suits image
        restoration.

        :param hidden_dim: Hidden dimension for FFN expansion.
        """
        self.ffn_expand = keras.layers.Conv2D(
            hidden_dim,
            kernel_size=1,
            use_bias=True,
            name="ffn_expand"
        )
        self.ffn_depthwise = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            use_bias=True,
            name="ffn_depthwise"
        )
        self.ffn_project = keras.layers.Conv2D(
            self.dim,
            kernel_size=1,
            use_bias=True,
            name="ffn_project"
        )

    def _setup_factory_ffn(self, hidden_dim: int) -> None:
        """Create a factory FFN from ``ffn_type``.

        Factory FFNs are Dense-based, so they keep spatial structure less well
        than the spatial FFN on image tasks.

        :param hidden_dim: Hidden dimension for FFN expansion.
        """
        self.ffn = create_ffn_layer(
            ffn_type=self.ffn_type,
            hidden_dim=hidden_dim,
            output_dim=self.dim,
            name="ffn",
            **self.ffn_kwargs
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer in forward order.

        Building them here rather than on the first call means all weight
        variables exist before weights are loaded.

        :param input_shape: Shape tuple of the input tensor.
        """
        self.norm1.build(input_shape)
        self.token_mixer_expand.build(input_shape)
        expanded_shape = self.token_mixer_expand.compute_output_shape(input_shape)

        self.fft.build(expanded_shape)
        fft_output_shape = self.fft.compute_output_shape(expanded_shape)

        self.freq_conv.build(fft_output_shape)
        freq_conv_output_shape = self.freq_conv.compute_output_shape(
            fft_output_shape
        )

        self.ifft.build(freq_conv_output_shape)
        ifft_output_shape = self.ifft.compute_output_shape(
            freq_conv_output_shape
        )

        self.token_mixer_project.build(ifft_output_shape)

        self.norm2.build(input_shape)

        if self.use_spatial_ffn:
            self.ffn_expand.build(input_shape)
            ffn_expanded_shape = self.ffn_expand.compute_output_shape(input_shape)
            self.ffn_depthwise.build(ffn_expanded_shape)
            self.ffn_project.build(ffn_expanded_shape)
        else:
            self.ffn.build(input_shape)

        super().build(input_shape)

    def _token_mixer_forward(
            self,
            x: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Run the frequency-domain token mixer.

        :param x: Input tensor of shape (batch, height, width, dim).

        :return: Token-mixed features of shape (batch, height, width, dim).
        """
        x_expanded = self.token_mixer_expand(x)

        x_fft = self.fft(x_expanded)
        x_freq = self.freq_conv(x_fft)
        x_freq = keras.activations.gelu(x_freq, approximate=False)
        x_ifft = self.ifft(x_freq)

        x_token_mixed = self.token_mixer_project(x_ifft)

        return x_token_mixed

    def _spatial_ffn_forward(
            self,
            x: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Run the spatial FFN.

        :param x: Input tensor of shape (batch, height, width, dim).

        :return: FFN output of shape (batch, height, width, dim).
        """
        x_ffn_expanded = self.ffn_expand(x)
        x_ffn_depthwise = self.ffn_depthwise(x_ffn_expanded)
        x_ffn_depthwise = keras.activations.gelu(x_ffn_depthwise, approximate=False)
        x_ffn_projected = self.ffn_project(x_ffn_depthwise)

        return x_ffn_projected

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run both residual stages and return the block output.

        :param inputs: Input tensor of shape (batch, height, width, dim).
        :param training: Training-mode flag. Forwarded to the factory FFN only;
            the normalization layers and the spatial FFN do not take it.

        :return: Output tensor of shape (batch, height, width, dim).
        """
        x_norm1 = self.norm1(inputs)

        x_token_mixed = self._token_mixer_forward(x_norm1)

        x = inputs + x_token_mixed

        x_norm2 = self.norm2(x)

        if self.use_spatial_ffn:
            x_ffn = self._spatial_ffn_forward(x_norm2)
        else:
            x_ffn = self.ffn(x_norm2, training=training)

        return x + x_ffn

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape tuple of input tensor.

        :return: Output shape tuple, identical to the input.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this layer.

        :return: Dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "normalization_type": self.normalization_type,
            "norm1_kwargs": self.norm1_kwargs,
            "norm2_kwargs": self.norm2_kwargs,
            "use_spatial_ffn": self.use_spatial_ffn,
            "ffn_type": self.ffn_type,
            "ffn_kwargs": self.ffn_kwargs,
        })
        return config


# ---------------------------------------------------------------------
# Scaling Layers
# ---------------------------------------------------------------------

# DECISION plan-2026-09-01T110541-dcc1574a/D-001: keep the PWFNet prefix; the bare
# name Downsample is taken in the legacy alias namespace. See decisions.md.
@register_dl_technique("dl_techniques.models.pw_fnet.model")
class PWFNetDownsample(keras.layers.Layer):
    """Halve the spatial resolution with a strided convolution.

    A 4x4 convolution with stride 2 learns the downsampling filter and sets the
    output channel count to ``dim``.

    .. code-block:: text

        input [B, H, W, C]
              │
              ▼
        ┌──────────────────────┐
        │ conv 4x4  stride 2   │
        └──────────────────────┘
              │
              ▼
        [B, H/2, W/2, dim]

    :param dim: Number of output channels. Must be positive.
    :param **kwargs: Additional arguments for the Layer base class.

    :raises ValueError: If dim is not positive.
    """

    def __init__(self, dim: int, **kwargs: Any) -> None:
        """Create the strided convolution.

        :param dim: Number of output channels.
        :param **kwargs: Additional Layer arguments.

        :raises ValueError: If dim is not positive.
        """
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim

        self.conv = keras.layers.Conv2D(
            dim,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=True,
            name="down_conv"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the internal convolution layer."""
        if not self.conv.built:
            self.conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Downsample the input tensor.

        :param inputs: Input tensor of shape (batch, height, width, channels).

        :return: Downsampled tensor of shape (batch, height//2, width//2, dim).
        """
        return self.conv(inputs)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape after downsampling.

        :param input_shape: Shape tuple of input tensor.

        :return: Output shape tuple with halved spatial dimensions.
        """
        return self.conv.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this layer.

        :return: Dictionary containing the dim parameter.
        """
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


# DECISION plan-2026-09-01T110541-dcc1574a/D-001: keep the PWFNet prefix; the bare
# name Upsample is taken in the legacy alias namespace. See decisions.md.
@register_dl_technique("dl_techniques.models.pw_fnet.model")
class PWFNetUpsample(keras.layers.Layer):
    """Double the spatial resolution with a transposed convolution.

    A 2x2 transposed convolution with stride 2 learns the upsampling filter and
    sets the output channel count to ``dim``.

    .. code-block:: text

        input [B, H, W, C]
              │
              ▼
        ┌──────────────────────┐
        │ convT 2x2  stride 2  │
        └──────────────────────┘
              │
              ▼
        [B, H*2, W*2, dim]

    :param dim: Number of output channels. Must be positive.
    :param **kwargs: Additional arguments for the Layer base class.

    :raises ValueError: If dim is not positive.
    """

    def __init__(self, dim: int, **kwargs: Any) -> None:
        """Create the transposed convolution.

        :param dim: Number of output channels.
        :param **kwargs: Additional Layer arguments.

        :raises ValueError: If dim is not positive.
        """
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim

        self.conv_transpose = keras.layers.Conv2DTranspose(
            dim,
            kernel_size=2,
            strides=2,
            padding="same",
            use_bias=True,
            name="up_conv_transpose"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the internal transposed convolution layer."""
        if not self.conv_transpose.built:
            self.conv_transpose.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Upsample the input tensor.

        :param inputs: Input tensor of shape (batch, height, width, channels).

        :return: Upsampled tensor of shape (batch, height*2, width*2, dim).
        """
        return self.conv_transpose(inputs)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape after upsampling.

        :param input_shape: Shape tuple of input tensor.

        :return: Output shape tuple with doubled spatial dimensions.
        """
        # Conv2DTranspose can return a list; callers rely on a tuple.
        return tuple(self.conv_transpose.compute_output_shape(input_shape))

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this layer.

        :return: Dictionary containing the dim parameter.
        """
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


# ---------------------------------------------------------------------
# PW-FNet Main Model
# ---------------------------------------------------------------------

# Number of encoder/decoder levels; the topology is written out by name.
_NUM_SCALES = 2


@register_dl_technique("dl_techniques.models.pw_fnet.model")
class PW_FNet(keras.Model):
    """Restore an image and return one prediction per scale.

    A U-Net with two encoder levels, two decoder levels and a bottleneck, so
    three resolutions in all: full, half and quarter. Each of the three output
    heads predicts a residual that is added to the input pooled to that scale,
    which gives hierarchical supervision during training. Every block is a
    :class:`PW_FNet_Block`, so the normalization type and the feed-forward
    network are set once here and used throughout.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
                 │
                 ▼
        ┌───────────────────┐
        │ intro conv 3x3    │
        └───────────────────┘
                 │  [B, H, W, w]
                 ▼
        ┌───────────────────┐
        │ encoder_level1    │──┐ skip1
        └───────────────────┘  │
                 │             │
                 ▼             │
        ┌───────────────────┐  │
        │ down1 conv 4x4 /2 │  │
        └───────────────────┘  │
                 │             │   [B, H/2, W/2, 2w]
                 ▼             │
        ┌───────────────────┐  │
        │ encoder_level2    │──┼──┐ skip2
        └───────────────────┘  │  │
                 │             │  │
                 ▼             │  │
        ┌───────────────────┐  │  │
        │ down2 conv 4x4 /2 │  │  │
        └───────────────────┘  │  │
                 │             │  │   [B, H/4, W/4, 4w]
                 ▼             │  │
        ┌───────────────────┐  │  │
        │ bottleneck        │  │  │
        └───────────────────┘  │  │
                 ├─────────────┼──┼──► output_l2 + input_l2 ─► out_l2
                 ▼             │  │
        ┌───────────────────┐  │  │
        │ up2 convT 2x2     │  │  │
        └───────────────────┘  │  │
                 │             │  │
                 ▼             │  │
              concat ◄─────────┼──┘
                 │             │
                 ▼             │
        ┌───────────────────┐  │
        │ reduce2 conv 1x1  │  │
        └───────────────────┘  │
                 │             │
                 ▼             │
        ┌───────────────────┐  │
        │ decoder_level2    │  │
        └───────────────────┘  │
                 ├─────────────┼──► output_l1 + input_l1 ─► out_l1
                 ▼             │
        ┌───────────────────┐  │
        │ up1 convT 2x2     │  │
        └───────────────────┘  │
                 │             │
                 ▼             │
              concat ◄─────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ reduce1 conv 1x1  │
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ decoder_level1    │
        └───────────────────┘
                 │
                 ▼
           output_l0 + inputs ─► out_l0

    input_l1 and input_l2 are the input average-pooled by 2 and by 4; the
    return order is [out_l0, out_l1, out_l2].

    :param img_channels: Number of channels in input and output images, for
        example 3 for RGB. Must be positive.
    :param width: Base channel width. Sets capacity and cost. Typical values
        are 32 to 64. Must be positive.
    :param middle_blk_num: Number of PW-FNet blocks in the bottleneck. Typical
        values are 4 to 12. Must be non-negative.
    :param enc_blk_nums: Block counts for the two encoder levels,
        ``[level1, level2]``. Must have exactly 2 entries. Defaults to [2, 2].
    :param dec_blk_nums: Block counts for the two decoder levels,
        ``[decoder_level2, decoder_level1]``. Must have exactly 2 entries.
        Defaults to [2, 2].
    :param normalization_type: Type of normalization used throughout the model.
        Accepts any type from the normalization factory: 'layer_norm',
        'rms_norm', 'zero_centered_rms_norm', 'band_rms', 'dynamic_tanh', etc.
        Defaults to 'layer_norm'.
    :param norm_kwargs: Optional arguments passed to every normalization layer.
    :param use_spatial_ffn: If True, every block uses the spatial FFN with a
        depthwise convolution, which is the original architecture. If False,
        every block uses a factory FFN. Defaults to True.
    :param ffn_type: Type of factory FFN used when ``use_spatial_ffn=False``.
        Options include 'mlp', 'swiglu', 'geglu', 'glu', 'differential', etc.
        Ignored when ``use_spatial_ffn=True``.
    :param ffn_kwargs: Optional arguments for the factory FFN, used only when
        ``use_spatial_ffn=False``.
    :param **kwargs: Additional arguments for the Model base class.

    :raises ValueError: If any argument is out of range, if the two block-count
        lists differ in length or do not have exactly 2 entries, or if
        ``use_spatial_ffn=False`` and ``ffn_type`` is not given.

    Note:
        The level count is fixed at two and the block-count lists do not
        change it. The encoder, decoder and output heads are each written out
        by name (``down1``/``down2``, ``up2``/``up1``,
        ``output_l2``/``output_l1``/``output_l0``) and ``call`` returns three
        tensors. ``enc_blk_nums`` and ``dec_blk_nums`` set only how many blocks
        run at each of the two levels; any other length raises ``ValueError``.

    Example:
        >>> # Default configuration (original architecture)
        >>> model = PW_FNet(img_channels=3, width=32)
        >>>
        >>> # With RMS normalization
        >>> model = PW_FNet(
        ...     img_channels=3,
        ...     width=32,
        ...     normalization_type='rms_norm',
        ...     norm_kwargs={'epsilon': 1e-6, 'use_scale': True}
        ... )
        >>>
        >>> # With factory FFN
        >>> model = PW_FNet(
        ...     img_channels=3,
        ...     width=32,
        ...     use_spatial_ffn=False,
        ...     ffn_type='swiglu',
        ...     ffn_kwargs={'dropout_rate': 0.1}
        ... )
    """

    def __init__(
            self,
            img_channels: int = 3,
            width: int = 32,
            middle_blk_num: int = 4,
            enc_blk_nums: Optional[List[int]] = None,
            dec_blk_nums: Optional[List[int]] = None,
            normalization_type: str = 'layer_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
            use_spatial_ffn: bool = True,
            ffn_type: Optional[str] = None,
            ffn_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Create the intro convolution, both paths and the three output heads.

        Arguments are documented on the class.

        :raises ValueError: If any argument is out of range, if the two
            block-count lists differ in length or do not have exactly 2
            entries, or if use_spatial_ffn=False and ffn_type is not given.
        """
        super().__init__(**kwargs)

        if enc_blk_nums is None:
            enc_blk_nums = [2, 2]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2]
        if norm_kwargs is None:
            norm_kwargs = {}
        if ffn_kwargs is None:
            ffn_kwargs = {}

        if img_channels <= 0:
            raise ValueError(f"img_channels must be positive, got {img_channels}")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if middle_blk_num < 0:
            raise ValueError(
                f"middle_blk_num must be non-negative, got {middle_blk_num}"
            )
        if not enc_blk_nums:
            raise ValueError("enc_blk_nums cannot be empty")
        if not dec_blk_nums:
            raise ValueError("dec_blk_nums cannot be empty")
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError(
                f"enc_blk_nums and dec_blk_nums must have same length, "
                f"got {len(enc_blk_nums)} and {len(dec_blk_nums)}"
            )
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-052: the 2-level depth stays fixed;
        # encoder, decoder and output heads are each written out by name. See decisions.md.
        if len(enc_blk_nums) != _NUM_SCALES:
            raise ValueError(
                f"PW_FNet has a FIXED {_NUM_SCALES}-level encoder/decoder and "
                f"returns exactly 3 output scales, so enc_blk_nums and "
                f"dec_blk_nums must each have exactly {_NUM_SCALES} entries "
                f"(blocks per level), got {len(enc_blk_nums)}: "
                f"{list(enc_blk_nums)}. These lists set the block COUNT at each "
                f"level, not the number of levels."
            )
        if any(n < 0 for n in enc_blk_nums):
            raise ValueError("All values in enc_blk_nums must be non-negative")
        if any(n < 0 for n in dec_blk_nums):
            raise ValueError("All values in dec_blk_nums must be non-negative")
        if not use_spatial_ffn and ffn_type is None:
            raise ValueError(
                "ffn_type must be specified when use_spatial_ffn=False"
            )

        self.img_channels = img_channels
        self.width = width
        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums
        self.normalization_type = normalization_type
        self.norm_kwargs = norm_kwargs
        self.use_spatial_ffn = use_spatial_ffn
        self.ffn_type = ffn_type
        self.ffn_kwargs = ffn_kwargs

        self.intro = keras.layers.Conv2D(
            width,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="intro_conv"
        )

        self.encoder_level1 = [
            self._create_block(width, f"enc_l1_blk_{i}")
            for i in range(enc_blk_nums[0])
        ]
        self.down1 = PWFNetDownsample(width * 2, name="down1")

        self.encoder_level2 = [
            self._create_block(width * 2, f"enc_l2_blk_{i}")
            for i in range(enc_blk_nums[1])
        ]
        self.down2 = PWFNetDownsample(width * 4, name="down2")

        self.bottleneck = [
            self._create_block(width * 4, f"middle_blk_{i}")
            for i in range(middle_blk_num)
        ]

        self.up2 = PWFNetUpsample(width * 2, name="up2")
        self.reduce_conv2 = keras.layers.Conv2D(
            width * 2,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name="reduce2"
        )
        self.decoder_level2 = [
            self._create_block(width * 2, f"dec_l2_blk_{i}")
            for i in range(dec_blk_nums[0])
        ]

        self.up1 = PWFNetUpsample(width, name="up1")
        self.reduce_conv1 = keras.layers.Conv2D(
            width,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name="reduce1"
        )
        self.decoder_level1 = [
            self._create_block(width, f"dec_l1_blk_{i}")
            for i in range(dec_blk_nums[1])
        ]

        self.output_l2 = keras.layers.Conv2D(
            img_channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="output_l2"
        )
        self.output_l1 = keras.layers.Conv2D(
            img_channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="output_l1"
        )
        self.output_l0 = keras.layers.Conv2D(
            img_channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="output_l0"
        )

    def _create_block(self, dim: int, name: str) -> PW_FNet_Block:
        """Create a PW-FNet block with the model's shared configuration.

        :param dim: Number of channels for the block.
        :param name: Name for the block.

        :return: Configured PW_FNet_Block instance.
        """
        return PW_FNet_Block(
            dim=dim,
            normalization_type=self.normalization_type,
            norm1_kwargs=self.norm_kwargs,
            norm2_kwargs=self.norm_kwargs,
            use_spatial_ffn=self.use_spatial_ffn,
            ffn_type=self.ffn_type,
            ffn_kwargs=self.ffn_kwargs,
            name=name
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method PW_FNet inherits ``Layer.build``, which marks the
        model built while its sub-layers are still unbuilt, and Keras warns
        about it. The shared helper traces ``call()`` on symbolic inputs, so
        what gets built matches what gets called.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """Run the encoder, bottleneck and decoder and return all three scales.

        :param inputs: Input tensor of shape (batch, height, width, img_channels).
        :param training: Training-mode flag forwarded to every block.

        :return: List of three restored images at different scales:
            [full_resolution, half_resolution, quarter_resolution].
        """
        # A stateless op, not a pooling layer: a layer built here would be
        # untracked and rebuilt on every trace.
        input_l1 = ops.average_pool(
            inputs, pool_size=2, strides=2, padding="valid")
        input_l2 = ops.average_pool(
            input_l1, pool_size=2, strides=2, padding="valid")

        feat = self.intro(inputs)

        feat_l1 = feat
        for blk in self.encoder_level1:
            feat_l1 = blk(feat_l1, training=training)
        skip1 = feat_l1

        feat_l2 = self.down1(skip1)
        for blk in self.encoder_level2:
            feat_l2 = blk(feat_l2, training=training)
        skip2 = feat_l2

        bottleneck_feat = self.down2(skip2)
        for blk in self.bottleneck:
            bottleneck_feat = blk(bottleneck_feat, training=training)

        res_l2 = self.output_l2(bottleneck_feat)
        out_l2 = input_l2 + res_l2

        dec_feat_l2 = self.up2(bottleneck_feat)
        dec_feat_l2 = ops.concatenate([dec_feat_l2, skip2], axis=-1)
        dec_feat_l2 = self.reduce_conv2(dec_feat_l2)
        for blk in self.decoder_level2:
            dec_feat_l2 = blk(dec_feat_l2, training=training)

        res_l1 = self.output_l1(dec_feat_l2)
        out_l1 = input_l1 + res_l1

        dec_feat_l1 = self.up1(dec_feat_l2)
        dec_feat_l1 = ops.concatenate([dec_feat_l1, skip1], axis=-1)
        dec_feat_l1 = self.reduce_conv1(dec_feat_l1)
        for blk in self.decoder_level1:
            dec_feat_l1 = blk(dec_feat_l1, training=training)

        res_l0 = self.output_l0(dec_feat_l1)
        out_l0 = inputs + res_l0

        return [out_l0, out_l1, out_l2]

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this model.

        :return: Dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            "img_channels": self.img_channels,
            "width": self.width,
            "middle_blk_num": self.middle_blk_num,
            "enc_blk_nums": self.enc_blk_nums,
            "dec_blk_nums": self.dec_blk_nums,
            "normalization_type": self.normalization_type,
            "norm_kwargs": self.norm_kwargs,
            "use_spatial_ffn": self.use_spatial_ffn,
            "ffn_type": self.ffn_type,
            "ffn_kwargs": self.ffn_kwargs,
        })
        return config


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_pw_fnet(
        img_channels: int = 3,
        width: int = 32,
        middle_blk_num: int = 4,
        enc_blk_nums: Optional[List[int]] = None,
        dec_blk_nums: Optional[List[int]] = None,
        normalization_type: str = 'layer_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        use_spatial_ffn: bool = True,
        ffn_type: Optional[str] = None,
        ffn_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
) -> PW_FNet:
    """Create a PW-FNet image-restoration model.

    Size is set by ``width`` and the per-level block counts; there is no named
    variant table.

    :param img_channels: Number of channels in input/output images. Must be positive.
    :param width: Base channel width. Must be positive.
    :param middle_blk_num: Number of bottleneck blocks. Must be non-negative.
    :param enc_blk_nums: Per-level encoder block counts; ``None`` resolves to
        ``[2, 2]``.
    :param dec_blk_nums: Per-level decoder block counts; ``None`` resolves to
        ``[2, 2]``. Must match ``enc_blk_nums`` in length.
    :param normalization_type: Normalization type from the normalization factory.
    :param norm_kwargs: Optional arguments forwarded to every normalization layer.
    :param use_spatial_ffn: If True use the original spatial FFN; if False use the
        factory FFN named by ``ffn_type``.
    :param ffn_type: Factory FFN type; required when ``use_spatial_ffn`` is False.
    :param ffn_kwargs: Optional arguments for the factory FFN.
    :param **kwargs: Additional arguments forwarded to the model constructor.

    :return: A configured PW_FNet instance. Calling it returns the three multi-scale
        outputs ``[full, half, quarter]``.

    :raises ValueError: If any argument is invalid, or the encoder/decoder block
        lists disagree in length or do not have exactly two entries each.

    Example:
        >>> model = create_pw_fnet(width=16, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])
        >>> outputs = model(keras.random.normal((1, 32, 32, 3)))
        >>> len(outputs)
        3
    """
    return PW_FNet(
        img_channels=img_channels,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums,
        normalization_type=normalization_type,
        norm_kwargs=norm_kwargs,
        use_spatial_ffn=use_spatial_ffn,
        ffn_type=ffn_type,
        ffn_kwargs=ffn_kwargs,
        **kwargs
    )

# ---------------------------------------------------------------------
