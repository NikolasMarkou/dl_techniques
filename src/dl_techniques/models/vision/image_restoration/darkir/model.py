"""
DarkIR low-light image restoration, a fully convolutional U-Net with parallel
dilated branches and a Fourier-domain modulation path.

Low-light photographs arrive with three degradations entangled at once: the scene
is under-exposed, the sensor noise that was always present is now comparable to
the signal, and the long exposure that produced the image also produced motion
blur. Restoring them separately does not work, because brightening amplifies the
noise and deblurring sharpens it. The architectural problem this creates is one
of receptive field under a compute budget: blur removal needs to see far enough
to find the blur kernel's support, and exposure correction needs a *global* view
of the image to decide how much to lift, yet self-attention over a full-resolution
restoration feature map costs `O(H^2 W^2)`.

This model resolves the two halves with two different cheap mechanisms rather
than one expensive one. Spatially, a set of parallel depthwise convolutions with
different dilation rates is summed:

`z = sum_d DWConv3x3_d(x)`

Since a 3x3 kernel at dilation `d` spans `3 + 2*(d - 1)` pixels, the default
rates `[1, 4, 9]` give 3x3, 11x11 and 19x19 supports from the same 3x3 parameter
count, and summing rather than concatenating keeps the channel width fixed so
the cost is linear in the number of branches. Globally, the Fourier transform is
used in place of attention: every output bin of an FFT already depends on every
input pixel, so a pointwise operation on the spectrum is a global operation in
`O(HW log HW)`, with no `H*W`-by-`H*W` matrix anywhere.

`FreMLP` is where that idea is implemented, and it is deliberately narrower than
"an MLP on the spectrum". It transforms to the frequency domain, processes only
the *magnitude* through a two-layer 1x1-conv MLP, and reattaches the original
phase by rescaling the unit phasor `freq / |freq|`. Magnitude carries the
illumination and contrast envelope while phase carries the structure, so editing
magnitude alone is what lets the layer relight an image without displacing its
edges. Two implementation facts about it are not obvious from the concept.
`keras.ops` has no `rfft2`, `angle` or `complex`, so the layer runs the
full complex `fft2` over the last two axes and therefore transposes NHWC to NCHW
and back around the transform; and the real output is valid only because a 1x1
convolution acts identically on the two members of every conjugate-pair bin, so
the processed spectrum retains the Hermitian symmetry of a real signal and the
imaginary part of the inverse transform can be discarded rather than needing a
projection.

Everything else in a block is parameter-frugal by the same instinct. `SimpleGate`
replaces the activation function with a channel split and an element-wise
product, `x1 * x2`, contributing a multiplicative nonlinearity at zero parameter
cost; the channel-attention branch is NAFNet's *simplified* channel attention,
a global average pool followed by a single 1x1 convolution with **no sigmoid**,
so it is an unbounded per-channel rescaling rather than a gate in `[0, 1]`.

The encoder block and the decoder block share the dilated first path but differ
in their second path and in one ordering detail. The encoder's second path is
`FreMLP`, and it is applied *multiplicatively*: the frequency output modulates
the running signal, `out = y + gamma * (y * FreMLP(Norm(y)))`, so the frequency
branch scales the spatial features rather than being added alongside them. The
decoder's second path is an ordinary gated inverted FFN (expand 1x1, SimpleGate,
project 1x1) added in the usual way. The ordering detail is that the optional
extra depthwise convolution sits *before* the channel-expanding 1x1 in the
encoder and *after* it in the decoder -- and because the decoder's version keeps
`groups=channels` while operating on `channels * dw_expand` maps, that layer is
a grouped convolution with `dw_expand` channels per group, not a true depthwise
convolution as its name suggests.

Both blocks scale each of their two residual branches by a learnable per-channel
weight, `beta` for the spatial path and `gamma` for the second path, and both are
initialized to **zeros**. Every block therefore begins training as an exact
identity and the whole tower starts as the global residual connection alone.
This is the LayerScale/ReZero idea and it is what allows a deep restoration
tower to be trained without warmup tricks: the network cannot destroy its input
before it has learned anything worth adding.

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
from typing import List, Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.pixel_unshuffle import PixelShuffle2D

from .components import FreMLP, DilatedBranch, SimpleGate, _add_list
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.darkir.model")
class DarkIREncoderBlock(keras.layers.Layer):
    """Encoder block (EBlock): parallel dilated branches plus a FreMLP modulator.

    Two residual paths. The first is the multi-scale spatial path: normalize,
    optional depthwise convolution, expand with a 1x1, run the parallel dilated
    bank and SUM it, gate, rescale per channel with NAFNet's sigmoid-free
    simplified channel attention, project back. The second is
    :class:`FreMLP`, applied MULTIPLICATIVELY -- the frequency output scales the
    running signal rather than being added alongside it. Both branch scales
    (``beta``, ``gamma``) are zero-initialized, so a fresh block is an exact
    identity.

    **Block structure:**

    .. code-block:: text

        Input x [B, H, W, C]
              │
        ┌─────┴────────────────────────────────────────────┐
        │  PATH 1: multi-scale spatial                     │
        │                                                  │
        │  LayerNorm                                       │
        │       ▼                                          │
        │  [DWConv 3×3]  ◄── extra_depth_wise, BEFORE the  │
        │       ▼            1×1 in the encoder            │
        │  Conv1×1 → C·dw_expand                           │
        │       ▼                                          │
        │  Σ over dilated branches [d₁, d₂, ..., dₙ]       │
        │       ▼                                          │
        │  SimpleGate → C·dw_expand/2                      │
        │       ▼                                          │
        │  GAP → Conv1×1 → multiply   (NO sigmoid:         │
        │       ▼                      unbounded rescale)  │
        │  Conv1×1 → C                                     │
        └─────┬────────────────────────────────────────────┘
              ▼
        y = x + β ⊙ (path 1)          β zero-init, per channel
              │
        ┌─────┴────────────────────────────────────────────┐
        │  PATH 2: frequency modulation                    │
        │                                                  │
        │  LayerNorm → FreMLP → multiply by y              │
        │  (MULTIPLICATIVE, not additive: the spectrum     │
        │   SCALES the spatial features)                   │
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
        BEFORE the channel-expanding 1x1. Note this is the opposite ordering
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

    :ivar beta: Learnable PER-CHANNEL scale of shape ``(1, 1, 1, channels)`` on
        the spatial path, zero-initialized. NOT a scalar, despite how the
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
    added in the usual way, and moves the optional extra convolution AFTER the
    channel-expanding 1x1. Because that convolution keeps ``groups=channels``
    while operating on ``channels * dw_expand`` maps, it is a GROUPED
    convolution with ``dw_expand`` channels per group, not a true depthwise
    convolution as its name suggests.

    **Block structure:**

    .. code-block:: text

        Input x [B, H, W, C]
              │
        ┌─────┴────────────────────────────────────────────┐
        │  PATH 1: multi-scale spatial                     │
        │                                                  │
        │  LayerNorm                                       │
        │       ▼                                          │
        │  Conv1×1 → C·dw_expand                           │
        │       ▼                                          │
        │  [grouped Conv 3×3]  ◄── extra_depth_wise, AFTER │
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
        │  PATH 2: gated inverted FFN                      │
        │                                                  │
        │  LayerNorm → Conv1×1 → C·ffn_expand              │
        │       ▼                                          │
        │  SimpleGate₂ → C·ffn_expand/2                    │
        │       ▼                                          │
        │  Conv1×1 → C          (ADDITIVE, unlike the      │
        └─────┬─────────────────  encoder's multiplicative │
              ▼                   frequency path)          │
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
        AFTER the channel-expanding 1x1. Note this is the opposite ordering
        from :class:`DarkIREncoderBlock`. Defaults to False.
    :type extra_depth_wise: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``channels``, ``dw_expand`` or ``ffn_expand`` is not
        positive, or if ``dilations`` is empty or contains a non-positive value.

    Input shape:
        4D tensor ``(batch, height, width, channels)``.

    Output shape:
        4D tensor ``(batch, height, width, channels)``.

    :ivar beta: Learnable PER-CHANNEL scale of shape ``(1, 1, 1, channels)`` on
        the spatial path, zero-initialized. NOT a scalar: the multiply
        broadcasts over N, H, W only.
    :vartype beta: keras.Variable
    :ivar gamma: The same, for the FFN path.
    :vartype gamma: keras.Variable
    :ivar sg1: SimpleGate on the spatial path.
    :vartype sg1: SimpleGate
    :ivar sg2: A SEPARATE SimpleGate instance on the FFN path.
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


def create_darkir_model(
    img_channels: int = 3,
    width: int = 32,
    middle_blk_num_enc: int = 2,
    middle_blk_num_dec: int = 2,
    enc_blk_nums: List[int] = None,
    dec_blk_nums: List[int] = None,
    dilations: List[int] = None,
    extra_depth_wise: bool = True,
    use_side_loss: bool = False
) -> keras.Model:
    """Build the DarkIR model for low-light image restoration.

    A functional builder, NOT a subclass: it returns
    ``keras.Model(inputs, outputs)`` over a ``(None, None, img_channels)``
    input, so the model is fully convolutional and resolution-agnostic at call
    time. Encoder stages downsample with a stride-2 2x2 convolution; decoder
    stages upsample with a 1x1 channel expansion followed by pixel shuffle and
    add the matching encoder skip. A global residual carries the input to the
    output, so the network learns only the correction to apply.

    **Architecture Overview:**

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
        │  MIDDLE                          ││      │      │
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
        │  Add(input)   GLOBAL RESIDUAL        │
        │  → Output [B, H, W, img_channels]    │
        └──────────────────────────────────────┘

    **Upsample channel arithmetic (why chan·2, not chan·4):**

    .. code-block:: text

        PixelShuffle(block_size=2):  channels ÷ 4, H and W × 2

        pre-shuffle Conv1×1 emits  chan · 2
                    post-shuffle:  chan · 2 / 4  =  chan / 2
                     encoder skip:              chan / 2     ✓ match

        the original chan·4 left chan channels post-shuffle and
        mismatched the chan/2 skip; never caught because the model
        was dead-on-forward via the nonexistent DepthToSpace (D-002)

    **Size constraint:**

    .. code-block:: text

        the stride-2 downsample uses padding='valid', so a dimension
        that is not a multiple of 2^len(enc_blk_nums) TRUNCATES on the
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
    :param use_side_loss: Whether to return an intermediate output for deep
        supervision. When True the model returns ``[main_output, side_output]``
        and the side output is at BOTTLENECK resolution
        (``H / 2**len(enc_blk_nums)``), not full resolution. A caller that
        compiles a single full-resolution loss against both outputs will build
        fine and die at the first ``fit()`` step on a shape mismatch; the target
        for the side output must be downsampled by the same factor.
        ``src/train/darkir/train_darkir.py --side-loss`` is the in-repo
        reference for that wiring. Defaults to False.
    :type use_side_loss: bool

    :return: The constructed DarkIR model. With ``use_side_loss=False`` a single
        output of shape ``(B, H, W, img_channels)``; with ``use_side_loss=True``
        two outputs, ``[main, side]``, of shapes ``(B, H, W, img_channels)`` and
        ``(B, H // 2**stages, W // 2**stages, img_channels)``.
    :rtype: keras.Model

    :raises ValueError: If ``img_channels`` or ``width`` is not positive, if
        ``middle_blk_num_enc`` is negative, if ``middle_blk_num_dec`` is less
        than 1, if ``enc_blk_nums`` and ``dec_blk_nums`` differ in length, or if
        any block count or dilation is not positive.

    Input shape:
        4D tensor ``(batch, height, width, img_channels)``. Height and width
        must be multiples of ``2 ** len(enc_blk_nums)``. Values are expected in
        ``[0, 1]``.

    Output shape:
        4D tensor ``(batch, height, width, img_channels)``, the restored image
        in the same range as the input.

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

            # Large model with deep supervision
            model = create_darkir_model(
                img_channels=3,
                width=48,
                enc_blk_nums=[2, 4, 6],
                dec_blk_nums=[6, 4, 2],
                dilations=[1, 4, 9],
                use_side_loss=True
            )

            x = ops.random.normal((1, 256, 256, 3))
            y = model(x)  # (1, 256, 256, 3)

    Note:
        Channel progression is ``width -> 2*width -> 4*width -> ...``; the
        global residual means the tower learns only the correction, and the
        zero-initialized block scales mean it starts as that residual alone.
    """
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
    # DECISION plan-2026-08-19T163559-499b6f0e/D-126
    # `>= 1`, not `>= 0`: at 0 the middle decoder loop never runs, so `x` is
    # still `x_light` when `layers.Add(name="middle_residual")([x, x_light])`
    # fires and the "residual" is EXACTLY `2 * x_light` -- MEASURED 2026-08-21,
    # max|middle_residual - 2*x_light| = 0.0 against a 2*x_light of magnitude
    # 4.427. Do NOT relax this back to non-negative for symmetry with
    # `middle_blk_num_enc`, which is genuinely 0-safe (at 0, `x_light` is just
    # the encoder output and the residual is still a real one). See
    # decisions.md D-126.
    if middle_blk_num_dec < 1:
        raise ValueError(
            "middle_blk_num_dec must be >= 1, got "
            f"{middle_blk_num_dec}: at 0 the middle residual degenerates to "
            "2 * x_light"
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

    # === Input ===
    inputs = keras.Input(shape=(None, None, img_channels), name="input_image")

    # === Intro Convolution ===
    x = keras.layers.Conv2D(width, kernel_size=3, padding="same", name="intro")(inputs)

    # === Encoder Path ===
    skips = []
    chan = width

    for i, num_blocks in enumerate(enc_blk_nums):
        # Apply encoder blocks
        for j in range(num_blocks):
            x = DarkIREncoderBlock(
                channels=chan,
                dilations=dilations,
                extra_depth_wise=extra_depth_wise,
                name=f"enc_stage_{i}_block_{j}"
            )(x)

        # Save skip connection
        skips.append(x)

        # Downsample (stride 2 convolution)
        chan = chan * 2
        x = keras.layers.Conv2D(
            chan,
            kernel_size=2,
            strides=2,
            padding="valid",
            name=f"down_{i}"
        )(x)

    # === Middle Section ===
    # Middle Encoder blocks
    for i in range(middle_blk_num_enc):
        x = DarkIREncoderBlock(
            channels=chan,
            dilations=dilations,
            extra_depth_wise=extra_depth_wise,
            name=f"mid_enc_{i}"
        )(x)

    # Store for optional side loss
    x_light = x

    # Middle Decoder blocks
    for i in range(middle_blk_num_dec):
        x = DarkIRDecoderBlock(
            channels=chan,
            dilations=dilations,
            extra_depth_wise=extra_depth_wise,
            name=f"mid_dec_{i}"
        )(x)

    # Residual connection in middle section
    x = keras.layers.Add(name="middle_residual")([x, x_light])

    # === Decoder Path ===
    for i, num_blocks in enumerate(dec_blk_nums):
        # Upsample using PixelShuffle (depth->space).
        # PixelShuffle2D(block_size=2) divides channels by 4 while doubling
        # H,W. The decoder halves channels per stage (chan -> chan//2), so the
        # pre-shuffle 1x1 conv must produce (chan//2)*4 == chan*2 filters so the
        # post-shuffle channel count matches the popped encoder skip. (The
        # original chan*4 left chan channels post-shuffle, mismatching the
        # chan//2-channel skip in the Add below; never caught because the model
        # was dead-on-forward via the nonexistent DepthToSpace. See D-002.)
        x = keras.layers.Conv2D(
            chan * 2,
            kernel_size=1,
            use_bias=False,
            name=f"up_conv_{i}"
        )(x)
        # DECISION plan_2026-06-15_00924f53/D-002: keras.layers.DepthToSpace
        # does not exist in Keras 3.8; PixelShuffle2D is the NHWC depth->space
        # replacement (inverse of PixelUnshuffle2D). See decisions.md D-002.
        x = PixelShuffle2D(block_size=2, name=f"pixel_shuffle_{i}")(x)

        # Halve channels (due to 2x spatial increase)
        chan = chan // 2

        # Add skip connection
        skip = skips.pop()
        x = keras.layers.Add(name=f"skip_add_{i}")([x, skip])

        # Apply decoder blocks
        for j in range(num_blocks):
            x = DarkIRDecoderBlock(
                channels=chan,
                dilations=dilations,
                extra_depth_wise=extra_depth_wise,
                name=f"dec_stage_{i}_block_{j}"
            )(x)

    # === Ending Convolution ===
    x = keras.layers.Conv2D(
        img_channels,
        kernel_size=3,
        padding="same",
        name="ending"
    )(x)

    # === Global Residual ===
    outputs = keras.layers.Add(name="final_residual")([x, inputs])

    # === Optional Side Loss ===
    if use_side_loss:
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-044: the tap stays at
        # bottleneck resolution and the TRAINER downsamples the target to meet
        # it. Do NOT "fix" this by upsampling `side_out` to full resolution so
        # that a single full-resolution loss happens to apply: that would make
        # the auxiliary gradient pass through an interpolation the main path
        # does not use, and it would still not be the paper's mechanism (a
        # separate full-resolution low-light branch, see the module docstring).
        # A caller wiring this flag must produce a matching downsampled target;
        # `src/train/darkir/train_darkir.py --side-loss` does exactly that.
        side_out = keras.layers.Conv2D(
            img_channels,
            kernel_size=3,
            padding="same",
            name="side_out"
        )(x_light)
        return keras.Model(
            inputs=inputs,
            outputs=[outputs, side_out],
            name="DarkIR"
        )

    return keras.Model(inputs=inputs, outputs=outputs, name="DarkIR")


# ---------------------------------------------------------------------