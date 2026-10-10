"""DarkIR low-light image restoration: a convolutional U-Net with parallel
dilated branches and a Fourier-domain modulation path.

Defines :class:`DarkIREncoderBlock`, :class:`DarkIRDecoderBlock`, :class:`DarkIR`,
and the ``create_darkir_model`` builder that assembles them into a U-Net.

Low-light photographs need both local repair (denoising, deblurring) and a
global adjustment (relighting), but self-attention over a full-resolution
feature map costs ``O(H^2 W^2)``. DarkIR replaces attention with two cheap
mechanisms instead: a bank of parallel dilated depthwise convolutions summed
together for local receptive field, and :class:`FreMLP` (in ``components.py``)
for the global view, which processes only the FFT magnitude of a feature map
through a small MLP and reattaches the original phase. Every block scales its
two residual branches by zero-initialized per-channel weights (``beta``,
``gamma``), so a fresh block starts as an identity and the network trains
without warmup.

The :class:`DarkIR` class is a ``keras.Model`` subclass with an ``include_top``
parameter: when ``True`` it uses :class:`EnhancementHead` from the vision heads
factory for the output head (denoising/super-resolution mode); when ``False``
it returns the encoder-decoder features for custom heads. The functional
``create_darkir_model`` is retained for backward compatibility and wraps the
subclass.

References:
    - Feijoo et al., 2025. DarkIR: Robust Low-Light Image Restoration. CVPR 2025.
    - Chen et al., 2022. Simple Baselines for Image Restoration.
      (https://arxiv.org/abs/2204.04676)
      Source of SimpleGate and the sigmoid-free simplified channel attention.
    - Yu & Koltun, 2016. Multi-Scale Context Aggregation by Dilated Convolutions.
      (https://arxiv.org/abs/1511.07122)
    - Shi et al., 2016. Real-Time Single Image and Video Super-Resolution Using an
      Efficient Sub-Pixel Convolutional Neural Network.
      (https://arxiv.org/abs/1609.05158)
    - Ronneberger et al., 2015. U-Net: Convolutional Networks for Biomedical Image
      Segmentation. (https://arxiv.org/abs/1505.04597)
    - Bachlechner et al., 2020. ReZero is All You Need: Fast Convergence at Large
      Depth. (https://arxiv.org/abs/2003.04887)
      The zero-initialized per-branch scale used by both blocks.
"""

import keras
from keras import layers, ops
from typing import List, Optional, Tuple, Dict, Any, Union, Literal

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.pooling.pixel_unshuffle import PixelShuffle2D
from dl_techniques.layers.heads.vision import (
    create_vision_head,
    VisionTaskType,
    HeadConfiguration,
)

from .components import FreMLP, DilatedBranch, SimpleGate, _add_list
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.darkir.model")
class DarkIREncoderBlock(keras.layers.Layer):
    """Encoder block (EBlock): parallel dilated branches plus a FreMLP modulator.

    Two residual paths. The first is the multi-scale spatial path: normalize,
    optional depthwise convolution, expand with a 1x1, run the parallel dilated
    bank and sum it, gate, rescale per channel with NAFNet's sigmoid-free
    simplified channel attention, project back. The second is
    :class:`FreMLP`, applied multiplicatively: the frequency output scales the
    running signal rather than being added alongside it. Both branch scales
    (``beta``, ``gamma``) are zero-initialized, so a fresh block is an exact
    identity.

    Block structure:

    .. code-block:: text

        Input x [B, H, W, C]
              │
        ┌─────┴────────────────────────────────────────────┐
        │  path 1: multi-scale spatial                     │
        │                                                  │
        │  LayerNorm                                       │
        │       ▼                                          │
        │  [DWConv 3×3]  <- extra_depth_wise, before the   │
        │       ▼            1×1 in the encoder            │
        │  Conv1×1 → C·dw_expand                           │
        │       ▼                                          │
        │  Σ over dilated branches [d₁, d₂, ..., dₙ]       │
        │       ▼                                          │
        │  SimpleGate → C·dw_expand/2                      │
        │       ▼                                          │
        │  GAP -> Conv1x1 -> multiply   (no sigmoid:       │
        │       ▼                      unbounded rescale)  │
        │  Conv1×1 → C                                     │
        └─────┬────────────────────────────────────────────┘
              ▼
        y = x + β ⊙ (path 1)          β zero-init, per channel
              │
        ┌─────┴────────────────────────────────────────────┐
        │  path 2: frequency modulation                    │
        │                                                  │
        │  LayerNorm → FreMLP → multiply by y              │
        │  (multiplicative, not additive: the spectrum     │
        │   scales the spatial features)                   │
        └─────┬────────────────────────────────────────────┘
              ▼
        out = y + γ ⊙ (y ⊙ FreMLP(Norm(y)))
              ▼
        Output [B, H, W, C]

    :param channels: Number of input and output channels, maintained throughout
        the block. Must be positive.
    :type channels: int
    :param dw_expand: Expansion factor for the depthwise path; intermediate
        width is ``channels * dw_expand``, which must be even for SimpleGate.
        Must be positive. Defaults to 2.
    :type dw_expand: int
    :param dilations: Dilation rates, one parallel branch per entry. ``None``
        resolves to ``[1]``. Must be non-empty and all positive. Commonly
        ``[1, 4, 9]``.
    :type dilations: List[int]
    :param extra_depth_wise: Whether to insert an extra depthwise convolution
        before the channel-expanding 1x1. Note this is the opposite ordering
        from :class:`DarkIRDecoderBlock`. Defaults to False.
    :type extra_depth_wise: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``channels`` or ``dw_expand`` is not positive, or if
        ``dilations`` is empty or contains a non-positive value.

    Input shape:
        4D tensor ``(batch, height, width, channels)``.

    Output shape:
        4D tensor ``(batch, height, width, channels)``; the residual structure
        preserves dimensionality.

    :ivar beta: Learnable per-channel scale of shape ``(1, 1, 1, channels)`` on
        the spatial path, zero-initialized. Not a scalar, despite how the
        equations read: the multiply broadcasts over N, H, W only.
    :vartype beta: keras.Variable
    :ivar gamma: The same, for the frequency path.
    :vartype gamma: keras.Variable
    :ivar branches: The parallel :class:`DilatedBranch` bank.
    :vartype branches: List[DilatedBranch]
    :ivar freq: The frequency-domain modulator.
    :vartype freq: FreMLP

    Example:
        .. code-block:: python

            block = DarkIREncoderBlock(
                channels=64, dw_expand=2, dilations=[1, 4, 9]
            )
            x = ops.random.normal((2, 32, 32, 64))
            y = block(x)  # (2, 32, 32, 64)

    Note:
        Zero-initialized ``beta`` and ``gamma`` mean the tower begins training
        as its global residual connection alone, which is what removes the need
        for warmup tricks.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        dilations: List[int] = None,
        extra_depth_wise: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize the encoder block and create every sub-layer.

        :param channels: Number of input and output channels.
        :type channels: int
        :param dw_expand: Depthwise-path expansion factor.
        :type dw_expand: int
        :param dilations: Dilation rates for the parallel bank.
        :type dilations: List[int]
        :param extra_depth_wise: Whether to add the extra depthwise convolution.
        :type extra_depth_wise: bool
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :raises ValueError: If any configuration value is invalid.
        """
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if dw_expand <= 0:
            raise ValueError(f"dw_expand must be positive, got {dw_expand}")
        if dilations is None:
            dilations = [1]
        if not dilations:
            raise ValueError("dilations cannot be empty")
        if any(d <= 0 for d in dilations):
            raise ValueError(f"All dilations must be positive, got {dilations}")

        self.channels = channels
        self.dw_expand = dw_expand
        self.dilations = dilations
        self.extra_depth_wise = extra_depth_wise
        self.expanded_channels = channels * dw_expand

        # Create all sub-layers in __init__
        # Normalization layers
        self.norm1 = create_normalization_layer("layer_norm", axis=-1, epsilon=1e-6)
        self.norm2 = create_normalization_layer("layer_norm", axis=-1, epsilon=1e-6)

        # Extra DW Conv (Optional)
        if self.extra_depth_wise:
            self.extra_conv = keras.layers.Conv2D(
                self.channels,
                kernel_size=3,
                padding="same",
                groups=self.channels,
                use_bias=True,
                name='extra_dw_conv'
            )
        else:
            self.extra_conv = keras.layers.Identity(name='identity_extra')

        # Projection to expanded channels
        self.conv1 = keras.layers.Conv2D(
            self.expanded_channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv1'
        )

        # Parallel Dilated Branches
        self.branches = [
            DilatedBranch(self.channels, self.dw_expand, d, name=f'branch_d{d}')
            for d in self.dilations
        ]

        # Channel Attention / Aggregation
        self.sca_avg = keras.layers.GlobalAveragePooling2D(
            keepdims=True,
            name='channel_attn_pool'
        )
        self.sca_conv = keras.layers.Conv2D(
            self.expanded_channels // 2,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='channel_attn_conv'
        )

        # SimpleGate activation
        self.sg1 = SimpleGate(name='simple_gate')

        # Projection back to original channels
        self.conv3 = keras.layers.Conv2D(
            self.channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv3'
        )

        # Frequency MLP Block
        self.freq = FreMLP(self.channels, expansion=2, name='freq_mlp')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer and create the two per-channel branch scales.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build normalization layers
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        # Build first path components
        self.extra_conv.build(input_shape)
        extra_shape = self.extra_conv.compute_output_shape(input_shape)

        self.conv1.build(extra_shape)
        conv1_shape = self.conv1.compute_output_shape(extra_shape)

        # Build all branches
        for branch in self.branches:
            branch.build(conv1_shape)

        # After branches, shape should match conv1_shape
        # SimpleGate halves the channels
        sg_input_shape = conv1_shape
        self.sg1.build(sg_input_shape)
        sg_output_shape = self.sg1.compute_output_shape(sg_input_shape)

        # Build channel attention
        self.sca_avg.build(sg_output_shape)
        sca_avg_shape = self.sca_avg.compute_output_shape(sg_output_shape)
        self.sca_conv.build(sca_avg_shape)

        # Build final projection
        self.conv3.build(sg_output_shape)

        # Build frequency path (operates on input_shape)
        self.freq.build(input_shape)

        # Create learnable scale parameters
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, self.channels),
            initializer="zeros",
            trainable=True
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, self.channels),
            initializer="zeros",
            trainable=True
        )

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass over the spatial path then the frequency modulation.

        :param inputs: Input tensor of shape
            ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Output tensor of the same shape.
        :rtype: keras.KerasTensor
        """
        # Store original input for residual
        y = inputs

        # === Path 1: Multi-scale Dilated Processing ===
        x = self.norm1(inputs)
        x = self.extra_conv(x)
        x = self.conv1(x)

        # Sum all parallel branches
        z = _add_list([branch(x) for branch in self.branches])

        # SimpleGate activation
        z = self.sg1(z)

        # Channel Attention
        attn = self.sca_avg(z)
        attn = self.sca_conv(attn)
        x = attn * z

        # Project back to original channels
        x = self.conv3(x)

        # First residual with learnable scale
        y = inputs + self.beta * x

        # === Path 2: Frequency Domain Processing ===
        x_step2 = self.norm2(y)
        x_freq = self.freq(x_step2)
        x = y * x_freq  # Element-wise modulation

        # Final residual with learnable scale
        out = y + x * self.gamma

        return out

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape, which is identical to the input shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "dw_expand": self.dw_expand,
            "dilations": self.dilations,
            "extra_depth_wise": self.extra_depth_wise
        })
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.darkir.model")
class DarkIRDecoderBlock(keras.layers.Layer):
    """Decoder block (DBlock): the same dilated path, with a gated FFN instead.

    Shares the encoder's multi-scale spatial path but replaces :class:`FreMLP`
    with an ordinary gated inverted FFN (expand 1x1, SimpleGate, project 1x1)
    added in the usual way, and moves the optional extra convolution after the
    channel-expanding 1x1. Because that convolution keeps ``groups=channels``
    while operating on ``channels * dw_expand`` maps, it is a grouped
    convolution with ``dw_expand`` channels per group, not a true depthwise
    convolution as its name suggests.

    Block structure:

    .. code-block:: text

        Input x [B, H, W, C]
              │
        ┌─────┴────────────────────────────────────────────┐
        │  path 1: multi-scale spatial                     │
        │                                                  │
        │  LayerNorm                                       │
        │       ▼                                          │
        │  Conv1×1 → C·dw_expand                           │
        │       ▼                                          │
        │  [grouped Conv 3×3]  <- extra_depth_wise, after  │
        │       ▼                  the 1×1 here; groups=C  │
        │                          over C·dw_expand maps   │
        │  Σ over dilated branches [d₁, d₂, ..., dₙ]       │
        │       ▼                                          │
        │  SimpleGate₁ → C·dw_expand/2                     │
        │       ▼                                          │
        │  GAP → Conv1×1 → multiply   (NO sigmoid)         │
        │       ▼                                          │
        │  Conv1×1 → C                                     │
        └─────┬────────────────────────────────────────────┘
              ▼
        y = x + β ⊙ (path 1)          β zero-init, per channel
              │
        ┌─────┴────────────────────────────────────────────┐
        │  path 2: gated inverted FFN                      │
        │                                                  │
        │  LayerNorm → Conv1×1 → C·ffn_expand              │
        │       ▼                                          │
        │  SimpleGate₂ → C·ffn_expand/2                    │
        │       ▼                                          │
        │  Conv1x1 -> C           (additive, unlike the    │
        └─────┬─────────────────  encoder's multiplicative ┘
              ▼                   frequency path)
        out = y + γ ⊙ (path 2)
              ▼
        Output [B, H, W, C]

    :param channels: Number of input and output channels, maintained throughout
        the block. Must be positive.
    :type channels: int
    :param dw_expand: Expansion factor for the depthwise path; intermediate
        width is ``channels * dw_expand``, which must be even for SimpleGate.
        Must be positive. Defaults to 2.
    :type dw_expand: int
    :param ffn_expand: Expansion factor for the FFN path; intermediate width is
        ``channels * ffn_expand``, which must be even for SimpleGate. Must be
        positive. Defaults to 2.
    :type ffn_expand: int
    :param dilations: Dilation rates, one parallel branch per entry. ``None``
        resolves to ``[1]``. Must be non-empty and all positive.
    :type dilations: List[int]
    :param extra_depth_wise: Whether to insert the extra grouped convolution
        after the channel-expanding 1x1. Note this is the opposite ordering
        from :class:`DarkIREncoderBlock`. Defaults to False.
    :type extra_depth_wise: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``channels``, ``dw_expand`` or ``ffn_expand`` is not
        positive, or if ``dilations`` is empty or contains a non-positive value.

    Input shape:
        4D tensor ``(batch, height, width, channels)``.

    Output shape:
        4D tensor ``(batch, height, width, channels)``.

    :ivar beta: Learnable per-channel scale of shape ``(1, 1, 1, channels)`` on
        the spatial path, zero-initialized. Not a scalar: the multiply
        broadcasts over N, H, W only.
    :vartype beta: keras.Variable
    :ivar gamma: The same, for the FFN path.
    :vartype gamma: keras.Variable
    :ivar sg1: SimpleGate on the spatial path.
    :vartype sg1: SimpleGate
    :ivar sg2: A separate SimpleGate instance on the FFN path.
    :vartype sg2: SimpleGate

    Example:
        .. code-block:: python

            block = DarkIRDecoderBlock(
                channels=64, dw_expand=2, ffn_expand=2, dilations=[1, 4, 9]
            )
            x = ops.random.normal((2, 32, 32, 64))
            y = block(x)  # (2, 32, 32, 64)
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dilations: List[int] = None,
        extra_depth_wise: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize the decoder block and create every sub-layer.

        :param channels: Number of input and output channels.
        :type channels: int
        :param dw_expand: Depthwise-path expansion factor.
        :type dw_expand: int
        :param ffn_expand: FFN-path expansion factor.
        :type ffn_expand: int
        :param dilations: Dilation rates for the parallel bank.
        :type dilations: List[int]
        :param extra_depth_wise: Whether to add the extra grouped convolution.
        :type extra_depth_wise: bool
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :raises ValueError: If any configuration value is invalid.
        """
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if dw_expand <= 0:
            raise ValueError(f"dw_expand must be positive, got {dw_expand}")
        if ffn_expand <= 0:
            raise ValueError(f"ffn_expand must be positive, got {ffn_expand}")
        if dilations is None:
            dilations = [1]
        if not dilations:
            raise ValueError("dilations cannot be empty")
        if any(d <= 0 for d in dilations):
            raise ValueError(f"All dilations must be positive, got {dilations}")

        self.channels = channels
        self.dw_expand = dw_expand
        self.ffn_expand = ffn_expand
        self.dilations = dilations
        self.extra_depth_wise = extra_depth_wise
        self.dw_channels = channels * dw_expand
        self.ffn_channels = channels * ffn_expand

        # Create all sub-layers in __init__
        # Normalization layers
        self.norm1 = create_normalization_layer("layer_norm", axis=-1, epsilon=1e-6)
        self.norm2 = create_normalization_layer("layer_norm", axis=-1, epsilon=1e-6)

        # First projection
        self.conv1 = keras.layers.Conv2D(
            self.dw_channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv1'
        )

        # Extra DW Conv (Optional) - NOTE: Applied AFTER conv1 in decoder
        if self.extra_depth_wise:
            self.extra_conv = keras.layers.Conv2D(
                self.dw_channels,
                kernel_size=3,
                padding="same",
                groups=self.channels,  # Groups based on original channels
                use_bias=True,
                name='extra_dw_conv'
            )
        else:
            self.extra_conv = keras.layers.Identity(name='identity_extra')

        # Parallel Dilated Branches
        # Note: In decoder, branches work with dw_channels and expansion=1
        self.branches = [
            DilatedBranch(self.dw_channels, expansion=1, dilation=d, name=f'branch_d{d}')
            for d in self.dilations
        ]

        # Channel Attention
        self.sca_avg = keras.layers.GlobalAveragePooling2D(
            keepdims=True,
            name='channel_attn_pool'
        )
        self.sca_conv = keras.layers.Conv2D(
            self.dw_channels // 2,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='channel_attn_conv'
        )

        # SimpleGate activations (two separate instances)
        self.sg1 = SimpleGate(name='simple_gate_1')
        self.sg2 = SimpleGate(name='simple_gate_2')

        # Projection back to original channels
        self.conv3 = keras.layers.Conv2D(
            self.channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv3'
        )

        # FFN Path projections
        self.conv4 = keras.layers.Conv2D(
            self.ffn_channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv4_ffn_expand'
        )
        self.conv5 = keras.layers.Conv2D(
            self.channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv5_ffn_project'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer and create the two per-channel branch scales.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build normalization layers
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        # Build first path components
        self.conv1.build(input_shape)
        conv1_shape = self.conv1.compute_output_shape(input_shape)

        self.extra_conv.build(conv1_shape)
        extra_shape = self.extra_conv.compute_output_shape(conv1_shape)

        # Build all branches
        for branch in self.branches:
            branch.build(extra_shape)

        # After branches, shape should match extra_shape
        # SimpleGate halves the channels
        self.sg1.build(extra_shape)
        sg1_output_shape = self.sg1.compute_output_shape(extra_shape)

        # Build channel attention
        self.sca_avg.build(sg1_output_shape)
        sca_avg_shape = self.sca_avg.compute_output_shape(sg1_output_shape)
        self.sca_conv.build(sca_avg_shape)

        # Build projection back to original channels
        self.conv3.build(sg1_output_shape)

        # Build FFN path (operates on input_shape after norm2)
        self.conv4.build(input_shape)
        conv4_shape = self.conv4.compute_output_shape(input_shape)

        self.sg2.build(conv4_shape)
        sg2_output_shape = self.sg2.compute_output_shape(conv4_shape)

        self.conv5.build(sg2_output_shape)

        # Create learnable scale parameters
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, self.channels),
            initializer="zeros",
            trainable=True
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, self.channels),
            initializer="zeros",
            trainable=True
        )

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass over the spatial path then the gated FFN.

        :param inputs: Input tensor of shape
            ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Output tensor of the same shape.
        :rtype: keras.KerasTensor
        """
        # Store original input for residual
        y = inputs

        # === Path 1: Multi-scale Dilated Processing ===
        x = self.norm1(inputs)

        # Note: Order differs from encoder (conv1 before extra_conv)
        x = self.conv1(x)
        x = self.extra_conv(x)

        # Sum all parallel branches
        z = _add_list([branch(x) for branch in self.branches])

        # First SimpleGate activation
        z = self.sg1(z)

        # Channel Attention
        attn = self.sca_avg(z)
        attn = self.sca_conv(attn)
        x = attn * z

        # Project back to original channels
        x = self.conv3(x)

        # First residual with learnable scale
        y = inputs + self.beta * x

        # === Path 2: Gated FFN Processing ===
        x = self.norm2(y)
        x = self.conv4(x)

        # Second SimpleGate activation
        x = self.sg2(x)

        x = self.conv5(x)

        # Final residual with learnable scale
        out = y + x * self.gamma

        return out

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape, which is identical to the input shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "dw_expand": self.dw_expand,
            "ffn_expand": self.ffn_expand,
            "dilations": self.dilations,
            "extra_depth_wise": self.extra_depth_wise
        })
        return config


# ---------------------------------------------------------------------
# DarkIR Model (keras.Model subclass with include_top)
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.darkir.model")
class DarkIR(keras.Model):
    """DarkIR low-light image restoration model with configurable output head.

    A U-Net architecture with parallel dilated branches and frequency-domain
    modulation, wrapped as a keras.Model subclass supporting the ``include_top``
    pattern. When ``include_top=True`` (default), the model uses an
    :class:`EnhancementHead` from the vision heads factory for image restoration
    (denoising mode). When ``include_top=False``, it returns the encoder-decoder
    features for custom heads.

    Architecture:
        - Intro 3x3 convolution
        - Encoder stages with DarkIREncoderBlock + stride-2 downsampling
        - Middle section with encoder/decoder blocks and residual
        - Decoder stages with PixelShuffle upsampling + DarkIRDecoderBlock
        - Optional EnhancementHead (denoising) or feature output
        - Global residual connection (input added to output)

    :param img_channels: Number of input and output image channels. Defaults to 3.
    :type img_channels: int
    :param width: Base feature width, doubled at each downsampling stage. Defaults to 32.
    :type width: int
    :param middle_blk_num_enc: Number of encoder blocks in the middle section. Defaults to 2.
    :type middle_blk_num_enc: int
    :param middle_blk_num_dec: Number of decoder blocks in the middle section. Defaults to 2.
    :type middle_blk_num_dec: int
    :param enc_blk_nums: Blocks per encoder stage. Defaults to [1, 2, 3].
    :type enc_blk_nums: List[int]
    :param dec_blk_nums: Blocks per decoder stage. Defaults to [3, 1, 1].
    :type dec_blk_nums: List[int]
    :param dilations: Dilation rates for all blocks. Defaults to [1, 4, 9].
    :type dilations: List[int]
    :param extra_depth_wise: Whether blocks use extra depthwise convolution. Defaults to True.
    :type extra_depth_wise: bool
    :param include_top: Whether to include the EnhancementHead for image restoration.
        When False, returns encoder-decoder features. Defaults to True.
    :type include_top: bool
    :param head_config_preset: Enhancement head preset: 'default', 'efficient',
        or 'high_performance'. Defaults to 'default'.
    :type head_config_preset: Literal['default', 'efficient', 'high_performance']
    :param head_config_overrides: Optional dict to override head configuration.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param use_side_loss: Whether to return an intermediate output for deep
        supervision. When True the model returns ``[main_output, side_output]``
        and the side output is at bottleneck resolution
        (``H / 2**len(enc_blk_nums)``), not full resolution. A caller that
        compiles a single full-resolution loss against both outputs will build
        fine and die at the first ``fit()`` step on a shape mismatch; the target
        for the side output must be downsampled by the same factor.
        Defaults to False.
    :type use_side_loss: bool
    :param kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        4D tensor ``(batch, height, width, img_channels)``. Height and width
        must be multiples of ``2 ** len(enc_blk_nums)``. Values expected in ``[0, 1]``.

    Output shape:
        - ``include_top=True``: ``(batch, height, width, img_channels)`` restored image.
        - ``include_top=False``: ``(batch, height, width, feature_channels)`` features.

    Example:
        >>> # Full restoration model
        >>> model = DarkIR(img_channels=3, width=32, include_top=True)
        >>>
        >>> # Feature extractor for custom head
        >>> backbone = DarkIR(img_channels=3, width=32, include_top=False)
        >>> features = backbone(inputs)
    """

    def __init__(
        self,
        img_channels: int = 3,
        width: int = 32,
        middle_blk_num_enc: int = 2,
        middle_blk_num_dec: int = 2,
        enc_blk_nums: Optional[List[int]] = None,
        dec_blk_nums: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        extra_depth_wise: bool = True,
        include_top: bool = True,
        head_config_preset: Literal['default', 'efficient', 'high_performance'] = 'default',
        head_config_overrides: Optional[Dict[str, Any]] = None,
        use_side_loss: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Set defaults
        if enc_blk_nums is None:
            enc_blk_nums = [1, 2, 3]
        if dec_blk_nums is None:
            dec_blk_nums = [3, 1, 1]
        if dilations is None:
            dilations = [1, 4, 9]

        # Validation
        if img_channels <= 0:
            raise ValueError(f"img_channels must be positive, got {img_channels}")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if middle_blk_num_enc < 0:
            raise ValueError(f"middle_blk_num_enc must be non-negative, got {middle_blk_num_enc}")
        if middle_blk_num_dec < 1:
            raise ValueError(
                f"middle_blk_num_dec must be >= 1, got {middle_blk_num_dec}: "
                f"at 0 the middle residual degenerates to 2 * x_light"
            )
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError(
                f"enc_blk_nums and dec_blk_nums must have same length, "
                f"got {len(enc_blk_nums)} and {len(dec_blk_nums)}"
            )
        if not enc_blk_nums or any(n <= 0 for n in enc_blk_nums):
            raise ValueError(f"All values in enc_blk_nums must be positive, got {enc_blk_nums}")
        if not dec_blk_nums or any(n <= 0 for n in dec_blk_nums):
            raise ValueError(f"All values in dec_blk_nums must be positive, got {dec_blk_nums}")
        if not dilations or any(d <= 0 for d in dilations):
            raise ValueError(f"All dilations must be positive, got {dilations}")

        # Store all configuration
        self.img_channels = img_channels
        self.width = width
        self.middle_blk_num_enc = middle_blk_num_enc
        self.middle_blk_num_dec = middle_blk_num_dec
        self.enc_blk_nums = list(enc_blk_nums)
        self.dec_blk_nums = list(dec_blk_nums)
        self.dilations = list(dilations)
        self.extra_depth_wise = extra_depth_wise
        self.include_top = include_top
        self.use_side_loss = use_side_loss
        self.head_config_preset = str(head_config_preset)
        self.head_config_overrides = dict(head_config_overrides) if head_config_overrides else None

        self.num_stages = len(enc_blk_nums)

        # Build the network
        self._build_network()

        logger.info(
            f"Created DarkIR: width={width}, stages={self.num_stages}, "
            f"enc_blk_nums={enc_blk_nums}, dec_blk_nums={dec_blk_nums}, "
            f"include_top={include_top}"
        )

    def _build_network(self) -> None:
        """Build the encoder-decoder U-Net architecture."""
        # Intro convolution
        self.intro_conv = layers.Conv2D(
            self.width, kernel_size=3, padding="same", name="intro"
        )

        # Encoder stages
        self.encoder_stages = []
        self.downsample_convs = []

        chan = self.width
        for i, num_blocks in enumerate(self.enc_blk_nums):
            stage_blocks = []
            for j in range(num_blocks):
                stage_blocks.append(
                    DarkIREncoderBlock(
                        channels=chan,
                        dilations=self.dilations,
                        extra_depth_wise=self.extra_depth_wise,
                        name=f"enc_stage_{i}_block_{j}"
                    )
                )
            self.encoder_stages.append(stage_blocks)

            # Downsample conv (except after last stage, handled in middle)
            down_conv = layers.Conv2D(
                chan * 2,
                kernel_size=2,
                strides=2,
                padding="valid",
                name=f"down_{i}"
            )
            self.downsample_convs.append(down_conv)
            chan = chan * 2

        # Middle section
        self.middle_encoder_blocks = []
        for i in range(self.middle_blk_num_enc):
            self.middle_encoder_blocks.append(
                DarkIREncoderBlock(
                    channels=chan,
                    dilations=self.dilations,
                    extra_depth_wise=self.extra_depth_wise,
                    name=f"mid_enc_{i}"
                )
            )

        self.middle_decoder_blocks = []
        for i in range(self.middle_blk_num_dec):
            self.middle_decoder_blocks.append(
                DarkIRDecoderBlock(
                    channels=chan,
                    dilations=self.dilations,
                    extra_depth_wise=self.extra_depth_wise,
                    name=f"mid_dec_{i}"
                )
            )

        self.middle_residual = layers.Add(name="middle_residual")

        # Side output head (for deep supervision / use_side_loss)
        self.side_out_head = None
        if self.use_side_loss:
            self.side_out_head = layers.Conv2D(
                self.img_channels,
                kernel_size=3,
                padding="same",
                name="side_out"
            )

        # Decoder stages
        self.decoder_stages = []
        self.upsample_convs = []
        self.upsample_pixelshuffles = []
        self.skip_adds = []

        for i, num_blocks in enumerate(self.dec_blk_nums):
            # Upsample conv (chan * 2 for PixelShuffle)
            up_conv = layers.Conv2D(
                chan * 2,
                kernel_size=1,
                use_bias=False,
                name=f"up_conv_{i}"
            )
            self.upsample_convs.append(up_conv)

            # PixelShuffle upsampling
            up_ps = PixelShuffle2D(block_size=2, name=f"pixel_shuffle_{i}")
            self.upsample_pixelshuffles.append(up_ps)

            # Skip connection add
            skip_add = layers.Add(name=f"skip_add_{i}")
            self.skip_adds.append(skip_add)

            # Halve channels after upsampling
            chan = chan // 2

            # Decoder blocks
            stage_blocks = []
            for j in range(num_blocks):
                stage_blocks.append(
                    DarkIRDecoderBlock(
                        channels=chan,
                        dilations=self.dilations,
                        extra_depth_wise=self.extra_depth_wise,
                        name=f"dec_stage_{i}_block_{j}"
                    )
                )
            self.decoder_stages.append(stage_blocks)

        # Ending convolution
        self.ending_conv = layers.Conv2D(
            self.img_channels,
            kernel_size=3,
            padding="same",
            name="ending"
        )

        self.final_residual = layers.Add(name="final_residual")

        # Enhancement head (when include_top=True)
        self.enhancement_head = None
        if self.include_top:
            head_config = HeadConfiguration.get_default_config(VisionTaskType.DENOISING)
            head_config.update({
                'output_channels': self.img_channels,
                'scale_factor': 1,  # Denoising: no upscaling
                'hidden_dim': self.width,  # Match base feature width
                'normalization_type': 'layer_norm',
                'activation_type': 'gelu',
            })
            if self.head_config_overrides:
                head_config.update(self.head_config_overrides)

            self.enhancement_head = create_vision_head(
                VisionTaskType.DENOISING, **head_config
            )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass through the DarkIR network.

        :param inputs: Input images of shape (batch, H, W, img_channels).
        :param training: Whether the call is in training mode.
        :return: Restored images (include_top=True) or features (include_top=False).
        """
        # Intro convolution
        x = self.intro_conv(inputs)

        # Encoder path
        skips = []
        for i, stage_blocks in enumerate(self.encoder_stages):
            for block in stage_blocks:
                x = block(x, training=training)
            skips.append(x)
            x = self.downsample_convs[i](x)

        # Middle encoder blocks
        for block in self.middle_encoder_blocks:
            x = block(x, training=training)

        x_light = x

        # Middle decoder blocks
        for block in self.middle_decoder_blocks:
            x = block(x, training=training)

        # Middle residual
        x = self.middle_residual([x, x_light])

        # Decoder path
        for i, stage_blocks in enumerate(self.decoder_stages):
            # Upsample
            x = self.upsample_convs[i](x)
            x = self.upsample_pixelshuffles[i](x)

            # Skip connection
            skip = skips.pop()
            x = self.skip_adds[i]([x, skip])

            # Decoder blocks
            for block in stage_blocks:
                x = block(x, training=training)

        # Ending convolution
        x = self.ending_conv(x)

        # Global residual
        outputs = self.final_residual([x, inputs])

        if self.include_top:
            # Apply enhancement head for denoising
            head_output = self.enhancement_head(outputs, training=training)
            main_output = head_output['enhanced']
        else:
            # Return features for custom heads
            main_output = outputs

        if self.use_side_loss:
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-044: the tap stays at
            # bottleneck resolution; the trainer downsamples the target to match, not
            # the other way around (`train_darkir.py --side-loss`). See decisions.md.
            side_out = self.side_out_head(x_light, training=training)
            return [main_output, side_out]

        return main_output

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        if self.use_side_loss:
            # Return list of shapes: [main_output_shape, side_output_shape]
            main_shape = input_shape
            factor = 2 ** self.num_stages
            side_shape = (input_shape[0],
                          input_shape[1] // factor if input_shape[1] else None,
                          input_shape[2] // factor if input_shape[2] else None,
                          self.img_channels)
            return [main_shape, side_shape]
        elif self.include_top:
            return input_shape
        else:
            # Feature output shape matches input spatial dims, channels = width
            return (input_shape[0], input_shape[1], input_shape[2], self.width)

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            "img_channels": self.img_channels,
            "width": self.width,
            "middle_blk_num_enc": self.middle_blk_num_enc,
            "middle_blk_num_dec": self.middle_blk_num_dec,
            "enc_blk_nums": self.enc_blk_nums,
            "dec_blk_nums": self.dec_blk_nums,
            "dilations": self.dilations,
            "extra_depth_wise": self.extra_depth_wise,
            "include_top": self.include_top,
            "use_side_loss": self.use_side_loss,
            "head_config_preset": self.head_config_preset,
            "head_config_overrides": self.head_config_overrides,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DarkIR":
        """Create model from configuration."""
        return cls(**config)


# ---------------------------------------------------------------------
# Functional Factory (backward compatibility)
# ---------------------------------------------------------------------


def create_darkir_model(
    img_channels: int = 3,
    width: int = 32,
    middle_blk_num_enc: int = 2,
    middle_blk_num_dec: int = 2,
    enc_blk_nums: List[int] = None,
    dec_blk_nums: List[int] = None,
    dilations: List[int] = None,
    extra_depth_wise: bool = True,
    include_top: bool = True,
    head_config_preset: Literal['default', 'efficient', 'high_performance'] = 'default',
    head_config_overrides: Optional[Dict[str, Any]] = None,
    use_side_loss: bool = False
) -> DarkIR:
    """Build the DarkIR model for low-light image restoration.

    A functional builder, not a subclass: it returns
    ``keras.Model(inputs, outputs)`` over a ``(None, None, img_channels)``
    input, so the model is fully convolutional and resolution-agnostic at call
    time. Encoder stages downsample with a stride-2 2x2 convolution; decoder
    stages upsample with a 1x1 channel expansion followed by pixel shuffle and
    add the matching encoder skip. A global residual carries the input to the
    output, so the network learns only the correction to apply.

    Architecture Overview:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, H, W, img_channels]       │
        │  H, W must be multiples of 2^stages  │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Intro Conv 3×3 → width              │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────┐
        │ Enc 0: n₀ × EBlock   │──skip 0──────────────────┐
        └──────────┬───────────┘                          │
              Conv 2×2 /2 'valid'   width → width·2       │
        ┌──────────▼───────────┐                          │
        │ Enc 1: n₁ × EBlock   │──skip 1───────────┐      │
        └──────────┬───────────┘                   │      │
              Conv 2×2 /2       width·2 → width·4  │      │
        ┌──────────▼───────────┐                   │      │
        │ Enc 2: n₂ × EBlock   │──skip 2────┐      │      │
        └──────────┬───────────┘            │      │      │
              Conv 2×2 /2                   │      │      │
        ┌──────────▼───────────────────────┐│      │      │
        │  middle                          ││      │      │
        │   middle_blk_num_enc × EBlock    ││      │      │
        │        ▼                         ││      │      │
        │      x_light  ──────────────────────► side_out  │
        │        ▼        (use_side_loss)  ││      │      │
        │   middle_blk_num_dec × DBlock    ││      │      │
        │        ▼                         ││      │      │
        │   Add([x, x_light])              ││      │      │
        └──────────┬───────────────────────┘│      │      │
                   ▼                        │      │      │
        ┌──────────────────────┐            │      │      │
        │ Conv1×1 → chan·2     │            │      │      │
        │ PixelShuffle(2)      │◄───────────┘      │      │
        │ Add(skip 2)          │                   │      │
        │ Dec 0: m₀ × DBlock   │                   │      │
        └──────────┬───────────┘                   │      │
        ┌──────────▼───────────┐                   │      │
        │ Dec 1: m₁ × DBlock   │◄──────────────────┘      │
        └──────────┬───────────┘                          │
        ┌──────────▼───────────┐                          │
        │ Dec 2: m₂ × DBlock   │◄─────────────────────────┘
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────────────────────┐
        │  Ending Conv 3×3 → img_channels      │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Add(input)   global residual        │
        │  → Output [B, H, W, img_channels]    │
        └──────────────────────────────────────┘

    Upsample channel arithmetic (why chan·2, not chan·4):

    .. code-block:: text

        PixelShuffle(block_size=2):  channels ÷ 4, H and W × 2

        pre-shuffle Conv1×1 emits  chan · 2
                    post-shuffle:  chan · 2 / 4  =  chan / 2
                     encoder skip:              chan / 2     ✓ match

        the original chan·4 left chan channels post-shuffle and
        mismatched the chan/2 skip; never caught because the model
        was dead-on-forward via the nonexistent DepthToSpace (D-002)

    Size constraint:

    .. code-block:: text

        the stride-2 downsample uses padding='valid', so a dimension
        that is not a multiple of 2^len(enc_blk_nums) truncates on the
        way down and the skip Add fails on mismatched shapes

        enc_blk_nums=[1,2,3]  →  8× downsampling  →  H, W % 8 == 0

    :param img_channels: Number of input and output image channels; typically 3
        for RGB. Must be positive. Defaults to 3.
    :type img_channels: int
    :param width: Base feature width, doubled at each downsampling stage. Must
        be positive. Defaults to 32.
    :type width: int
    :param middle_blk_num_enc: Number of encoder blocks in the middle section.
        Must be non-negative; 0 is genuinely safe here because ``x_light`` is
        then just the encoder output and the middle residual is still a real
        one. Defaults to 2.
    :type middle_blk_num_enc: int
    :param middle_blk_num_dec: Number of decoder blocks in the middle section.
        Must be at least 1: at 0 the middle residual degenerates to exactly
        ``2 * x_light``. See the D-126 anchor in the body. Defaults to 2.
    :type middle_blk_num_dec: int
    :param enc_blk_nums: Blocks per encoder stage; its length is the number of
        downsampling operations. ``None`` resolves to ``[1, 2, 3]``. All values
        must be positive.
    :type enc_blk_nums: List[int]
    :param dec_blk_nums: Blocks per decoder stage; must have the same length as
        ``enc_blk_nums``. ``None`` resolves to ``[3, 1, 1]``. All values must be
        positive.
    :type dec_blk_nums: List[int]
    :param dilations: Dilation rates applied to every encoder and decoder
        block. ``None`` resolves to ``[1, 4, 9]``. All values must be positive.
    :type dilations: List[int]
    :param extra_depth_wise: Whether every block uses its extra convolution
        (before the 1x1 in the encoder, after it in the decoder). Defaults to
        True.
    :type extra_depth_wise: bool
    :param include_top: Whether to include the EnhancementHead for image restoration.
        When False, returns encoder-decoder features for custom heads. Defaults to True.
    :type include_top: bool
    :param head_config_preset: Enhancement head preset: 'default', 'efficient',
        or 'high_performance'. Defaults to 'default'.
    :type head_config_preset: Literal['default', 'efficient', 'high_performance']
    :param head_config_overrides: Optional dict to override head configuration.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param use_side_loss: DEPRECATED. Use a custom training loop or model subclass
        for deep supervision. This parameter is ignored.
    :type use_side_loss: bool

    :return: The constructed DarkIR model (keras.Model subclass).
        With ``include_top=True``: outputs restored image or dict with 'enhanced' key.
        With ``include_top=False``: outputs encoder-decoder features.
    :rtype: DarkIR

    :raises ValueError: If ``img_channels`` or ``width`` is not positive, if
        ``middle_blk_num_enc`` is negative, if ``middle_blk_num_dec`` is less
        than 1, if ``enc_blk_nums`` and ``dec_blk_nums`` differ in length, or if
        any block count or dilation is not positive.

    Input shape:
        4D tensor ``(batch, height, width, img_channels)``. Height and width
        must be multiples of ``2 ** len(enc_blk_nums)``. Values are expected in
        ``[0, 1]``.

    Output shape:
        - ``include_top=True``: ``(batch, height, width, img_channels)`` or dict with 'enhanced'.
        - ``include_top=False``: ``(batch, height, width, feature_channels)`` features.

    Example:
        .. code-block:: python

            # Small model for testing
            model = create_darkir_model(
                img_channels=3,
                width=16,
                enc_blk_nums=[1, 1],
                dec_blk_nums=[1, 1],
                dilations=[1]
            )

            # Paper default
            model = create_darkir_model(
                img_channels=3,
                width=32,
                enc_blk_nums=[1, 2, 3],
                dec_blk_nums=[3, 1, 1],
                dilations=[1, 4, 9],
                extra_depth_wise=True
            )

            # Feature extractor for custom head
            backbone = create_darkir_model(
                img_channels=3,
                width=32,
                include_top=False,
                enc_blk_nums=[1, 2, 3],
                dec_blk_nums=[3, 1, 1],
                dilations=[1, 4, 9]
            )

            x = ops.random.normal((1, 256, 256, 3))
            y = model(x)  # (1, 256, 256, 3) or {'enhanced': ...}

    Note:
        Channel progression is ``width -> 2*width -> 4*width -> ...``; the
        global residual means the tower learns only the correction, and the
        zero-initialized block scales mean it starts as that residual alone.
    """
    return DarkIR(
        img_channels=img_channels,
        width=width,
        middle_blk_num_enc=middle_blk_num_enc,
        middle_blk_num_dec=middle_blk_num_dec,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums,
        dilations=dilations,
        extra_depth_wise=extra_depth_wise,
        include_top=include_top,
        use_side_loss=use_side_loss,
        head_config_preset=head_config_preset,
        head_config_overrides=head_config_overrides,
    )


# ---------------------------------------------------------------------