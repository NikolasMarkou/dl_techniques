"""Building blocks for DarkIR: gating, frequency-domain MLP, and dilated convolution.

Defines three layers used by ``darkir/model.py``: :class:`SimpleGate`, a
parameter-free multiplicative gate that halves channel width; :class:`FreMLP`,
which processes the FFT magnitude of a feature map through a small MLP while
leaving phase untouched; and :class:`DilatedBranch`, one scale of a parallel
dilated-convolution bank.

``FreMLP`` runs its FFT and inverse FFT in float32 regardless of the layer's
compute dtype, because TensorFlow has no float16 kernel for ``fft2``/``ifft2``;
only the 1x1-conv MLP between them runs at ``compute_dtype``.
"""

import keras
from typing import List, Optional, Tuple, Dict, Any
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------


def _add_list(tensors: List[Any]) -> Any:
    """Element-wise sum of a non-empty list of tensors.

    Backend-agnostic replacement for the nonexistent ``keras.ops.add_n``
    (absent in Keras 3.8). Folds the list with ``keras.ops.add``.

    :param tensors: Non-empty list of broadcast-compatible tensors.
    :type tensors: List[Any]
    :return: Their element-wise sum.
    :rtype: Any
    """
    acc = tensors[0]
    for t in tensors[1:]:
        acc = keras.ops.add(acc, t)
    return acc

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.darkir.components")
class SimpleGate(keras.layers.Layer):
    """Parameter-free multiplicative gating by channel split and product.

    Replaces the activation function: the input is split in half along the
    channel axis and the two halves are multiplied element-wise, contributing a
    multiplicative nonlinearity at ZERO parameter cost. This is NAFNet's gate,
    and it is what makes the surrounding blocks parameter-frugal.

    Operation:

    .. code-block:: text

        Input [B, H, W, 2C]
                │
                ▼
        ┌───────────────────────────────┐
        │  split(axis=-1)               │
        │    x₁ [B, H, W, C]            │
        │    x₂ [B, H, W, C]            │
        └───────────────┬───────────────┘
                        ▼
                  x₁ ⊙ x₂   (element-wise)
                        ▼
        ┌───────────────────────────────┐
        │  Output [B, H, W, C]          │
        │  channels are halved          │
        └───────────────────────────────┘

    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        4D tensor ``(batch, height, width, channels)``. ``channels`` must be
        even.

    Output shape:
        4D tensor ``(batch, height, width, channels // 2)``.

    Example:
        .. code-block:: python

            gate = SimpleGate()
            x = ops.random.normal((2, 32, 32, 64))
            y = gate(x)  # (2, 32, 32, 32)

    Note:
        Ensure the preceding layer produces an even channel count; expansion
        factors that yield odd widths will not split cleanly.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the gate.

        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        """
        super().__init__(**kwargs)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Split the channel axis in half and multiply the two halves.

        :param x: Input tensor of shape ``(batch, height, width, 2*channels)``.
        :type x: keras.KerasTensor
        :return: Gated output of shape ``(batch, height, width, channels)``.
        :rtype: keras.KerasTensor
        """
        # Split along the channel axis (last axis in Keras NHWC format)
        x1, x2 = keras.ops.split(x, 2, axis=-1)
        return x1 * x2

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape, with the channel axis halved.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape tuple with ``channels // 2``.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = input_shape[-1] // 2 if input_shape[-1] is not None else None
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        return super().get_config()


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.darkir.components")
class FreMLP(keras.layers.Layer):
    """
    Global feature modeling in the frequency domain, magnitude only.

    Every output bin of an FFT depends on every input pixel, so a pointwise
    operation on the spectrum is a global operation at ``O(HW log HW)`` with no
    ``HW x HW`` matrix anywhere. This layer transforms to the frequency domain,
    passes only the magnitude through a two-layer 1x1-conv MLP, and reattaches
    the original phase by rescaling the unit phasor ``freq / |freq|``.
    Magnitude carries the illumination and contrast envelope while phase carries
    structure, so editing magnitude alone relights an image without displacing
    its edges.

    Operation:

    .. code-block:: text

        Input [B, H, W, C]  (real)
                │
                ▼
        ┌──────────────────────────────────────┐
        │  cast float32, NHWC → NCHW           │
        │  fft2 over the last two axes         │
        │  (keras.ops has no rfft2 / angle /   │
        │   complex, so the full complex       │
        │   transform is used)                 │
        └───────────────┬──────────────────────┘
                        │  freq_real, freq_imag [B, C, H, W]
                        ▼
        ┌──────────────────────────────────────┐
        │  mag = √(real² + imag²)              │
        │  phase is not extracted; it survives │
        │  as the unit phasor freq / |freq|    │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  NCHW → NHWC, cast to compute_dtype  │
        │  Conv1×1(C → expansion·C)            │
        │  LeakyReLU(0.1)                      │
        │  Conv1×1(expansion·C → C)            │
        │  cast float32, NHWC → NCHW           │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  out = mag' · (freq / max(|freq|, ε))│
        │  original phase, new magnitude       │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  ifft2, keep the real part          │
        │  valid because a 1×1 conv acts       │
        │  identically on both members of      │
        │  every conjugate-pair bin, so        │
        │  Hermitian symmetry is preserved     │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output [B, H, W, C]  (shape kept)   │
        └──────────────────────────────────────┘

    Mixed-precision island:

    .. code-block:: text

        float32 ───────► compute_dtype ───────► float32
        fft2, |·|        the 1×1-conv MLP        phasor, ifft2

        TF has no float16 kernel for fft2/ifft2, so the spectral
        steps are pinned to float32 while the layer's only matmuls
        stay in mixed precision. See the D-054 anchor in `call`.

    :param channels: Number of input and output channels; the layer preserves
        channel dimensionality. Must be positive.
    :type channels: int
    :param expansion: Expansion factor for the internal MLP hidden dimension,
        which is ``channels * expansion``. Must be positive. Defaults to 2.
    :type expansion: int
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``channels`` or ``expansion`` is not positive.

    Input shape:
        4D tensor ``(batch, height, width, channels)``.

    Output shape:
        4D tensor ``(batch, height, width, channels)``; shape is preserved and
        only feature values change.

    :ivar conv1: First 1x1 convolution, ``channels -> expansion*channels``.
    :vartype conv1: keras.layers.Conv2D
    :ivar act: LeakyReLU with negative slope 0.1.
    :vartype act: keras.layers.LeakyReLU
    :ivar conv2: Second 1x1 convolution, ``expansion*channels -> channels``.
    :vartype conv2: keras.layers.Conv2D

    Example:
        .. code-block:: python

            freq_mlp = FreMLP(channels=64, expansion=2)
            x = ops.random.normal((2, 32, 32, 64))
            y = freq_mlp(x)  # (2, 32, 32, 64)

    Note:
        Cost is ``O(HW log HW)`` and independent of the channel count, which is
        what makes this cheaper than self-attention's ``O(H²W²)`` for the same
        global receptive field.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        **kwargs: Any
    ) -> None:
        """Initialize the frequency MLP.

        :param channels: Number of input and output channels.
        :type channels: int
        :param expansion: Hidden-dimension expansion factor.
        :type expansion: int
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :raises ValueError: If either argument is not positive.
        """
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")

        self.channels = channels
        self.expansion = expansion

        # Create sub-layers in __init__
        hidden_dim = int(channels * expansion)
        self.conv1 = keras.layers.Conv2D(
            hidden_dim,
            kernel_size=1,
            strides=1,
            padding='valid',
            name='conv1'
        )
        self.act = keras.layers.LeakyReLU(negative_slope=0.1, name='leaky_relu')
        self.conv2 = keras.layers.Conv2D(
            channels,
            kernel_size=1,
            strides=1,
            padding='valid',
            name='conv2'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the sub-layers explicitly for correct serialization.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build sub-layers explicitly
        self.conv1.build(input_shape)

        # Compute intermediate shape after conv1
        conv1_output_shape = self.conv1.compute_output_shape(input_shape)
        self.act.build(conv1_output_shape)
        self.conv2.build(conv1_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Forward pass: FFT, process the magnitude, reattach phase, inverse FFT.

        :param x: Input tensor of shape ``(batch, height, width, channels)``.
        :type x: keras.KerasTensor
        :return: Output tensor of the same shape.
        :rtype: keras.KerasTensor
        """
        # keras.ops has no rfft2/irfft2/angle/complex, so this uses full-spectrum
        # fft2 as a (real, imag) tuple, moving the spatial dims last: NHWC -> NCHW.

        # 1. FFT over spatial dimensions (H, W). Input is real -> zero imag part.
        #
        # DECISION plan-2026-08-19T163559-499b6f0e/D-054: steps 1, 2, 4, 5 run in
        # float32 because TensorFlow has no float16 kernel for fft2/ifft2. Only
        # step 3's conv runs at compute_dtype. See decisions.md.
        x_t = keras.ops.transpose(keras.ops.cast(x, "float32"), (0, 3, 1, 2))  # (B, C, H, W)
        freq_real, freq_imag = keras.ops.fft2((x_t, keras.ops.zeros_like(x_t)))

        # 2. Magnitude (phase is retained implicitly via the unit phasor below).
        mag = keras.ops.sqrt(keras.ops.square(freq_real) + keras.ops.square(freq_imag))  # (B, C, H, W)

        # 3. Process the magnitude spectrum through the 1x1-conv MLP, which
        # operates channels-last; transpose to NHWC and back.
        mag_nhwc = keras.ops.cast(
            keras.ops.transpose(mag, (0, 2, 3, 1)), self.compute_dtype
        )                                              # (B, H, W, C)
        mag_nhwc = self.conv1(mag_nhwc)
        mag_nhwc = self.act(mag_nhwc)
        mag_nhwc = self.conv2(mag_nhwc)
        mag_proc = keras.ops.cast(
            keras.ops.transpose(mag_nhwc, (0, 3, 1, 2)), "float32"
        )                                              # (B, C, H, W)

        # 4. Reconstruct with the original phase: scale the unit phasor
        # (freq / |freq|) by the processed magnitude. Guard the divide.
        denom = keras.ops.maximum(mag, keras.config.epsilon())
        out_real = mag_proc * (freq_real / denom)
        out_imag = mag_proc * (freq_imag / denom)

        # 5. Inverse FFT back to the spatial domain. The processed spectrum
        # keeps the Hermitian symmetry of a real signal (the 1x1 conv acts
        # identically on conjugate-pair bins), so the real part is the output.
        spatial_real, _ = keras.ops.ifft2((out_real, out_imag))  # (B, C, H, W)
        return keras.ops.cast(
            keras.ops.transpose(spatial_real, (0, 2, 3, 1)), self.compute_dtype
        )                                                  # (B, H, W, C)

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
            "expansion": self.expansion
        })
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.darkir.components")
class DilatedBranch(keras.layers.Layer):
    """One dilated depthwise convolution: a single scale of a parallel bank.

    A 3x3 depthwise convolution at dilation ``d``, used as one branch of the
    multi-scale bank the encoder and decoder blocks sum. Dilation buys receptive
    field at no parameter cost, and the depthwise grouping keeps the branch
    cheap enough that several can run in parallel.

    Receptive field per branch (3x3 kernel):

    .. code-block:: text

        span = 3 + 2·(d − 1)

        d = 1   ███          3 × 3     local texture
        d = 4   █···█···█   11 × 11    mid-range structure
        d = 9   █········█  19 × 19    blur-kernel support

        all three share the same 3x3 parameter count; the caller
        sums them, so the channel width does not grow and the
        cost is linear in the number of branches

    :param channels: Number of input channels. Must be positive. Output width
        is ``channels * expansion``.
    :type channels: int
    :param expansion: Channel expansion factor. Must be positive. Defaults to
        1 (no expansion).
    :type expansion: int
    :param dilation: Dilation rate controlling the spatial receptive field.
        Must be positive. Defaults to 1.
    :type dilation: int
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``channels``, ``expansion`` or ``dilation`` is not
        positive.

    Input shape:
        4D tensor ``(batch, height, width, channels)``.

    Output shape:
        4D tensor ``(batch, height, width, channels * expansion)``; spatial
        dimensions are preserved by ``padding='same'`` regardless of dilation.

    :ivar dw_channels: ``channels * expansion``, also the group count.
    :vartype dw_channels: int
    :ivar conv: The dilated depthwise convolution.
    :vartype conv: keras.layers.Conv2D

    Example:
        .. code-block:: python

            x = ops.random.normal((2, 32, 32, 64))
            branches = [
                DilatedBranch(channels=64, expansion=2, dilation=d)
                for d in [1, 4, 9]
            ]
            combined = _add_list([branch(x) for branch in branches])
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 1,
        dilation: int = 1,
        **kwargs: Any
    ) -> None:
        """Initialize the dilated branch.

        :param channels: Number of input channels.
        :type channels: int
        :param expansion: Channel expansion factor.
        :type expansion: int
        :param dilation: Dilation rate.
        :type dilation: int
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :raises ValueError: If any argument is not positive.
        """
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")

        self.channels = channels
        self.expansion = expansion
        self.dilation = dilation
        self.dw_channels = int(channels * expansion)

        # Create depthwise convolution in __init__
        self.conv = keras.layers.Conv2D(
            filters=self.dw_channels,
            kernel_size=3,
            padding="same",
            dilation_rate=self.dilation,
            groups=self.dw_channels,  # Depthwise: each filter processes one channel
            use_bias=True,
            name=f"dilated_conv_d{self.dilation}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the convolution explicitly.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build sub-layer explicitly
        self.conv.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Apply the dilated depthwise convolution.

        :param x: Input tensor of shape ``(batch, height, width, channels)``.
        :type x: keras.KerasTensor
        :return: Output of shape ``(batch, height, width, channels*expansion)``.
        :rtype: keras.KerasTensor
        """
        return self.conv(x)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape, with the expanded channel count.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape tuple with ``channels * expansion`` channels.
        :rtype: Tuple[Optional[int], ...]
        """
        return self.conv.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "expansion": self.expansion,
            "dilation": self.dilation
        })
        return config


# ---------------------------------------------------------------------
