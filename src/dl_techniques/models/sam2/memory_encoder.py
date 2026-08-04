"""SAM 2 memory encoder.

Compresses a predicted high-resolution mask together with the image encoder's
pixel features into the narrow ``mem_dim``-wide spatial memory that memory
attention later reads as keys and values.

Three mechanisms in this file are SILENT when ported wrong -- the model builds,
forward-passes, trains and serializes either way:

1. **The mask is passed through** ``20 * sigmoid(x) - 10``\\ **, i.e. the affine
   is applied AFTER the sigmoid, not before it.** The transform rescales a
   probability in ``(0, 1)`` into the wide SIGNED range ``(-10, +10)``. Two
   wrong readings produce the same shapes and a plausible loss: a bare
   ``sigmoid(x)`` (range ``(0, 1)``), and the affine-then-sigmoid
   ``sigmoid(20 * x - 10)``, which is a near-step function also in ``(0, 1)``
   and therefore ~20x narrower with no negative half at all.
2. **The downsampler's layer COUNT comes from the shipped configuration, not
   from the reference class signature.** At ``k=3, s=2, p=1`` it is four
   convolutions; the signature default ``k=4, s=4, p=0`` is two. **Both give a
   total stride of 16**, so an assertion on the output resolution alone cannot
   tell them apart.
3. **The fusion is additive.** The projected pixel features and the downsampled
   mask are ADDED, never concatenated, so the fuser sees ``in_dim`` channels
   rather than ``2 * in_dim``.

All three are guarded behaviourally in
``tests/test_models/test_sam2/test_memory_encoder.py``.
"""

import math
import keras
from keras import ops
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.embedding.positional_embedding_sine_2d import (
    PositionEmbeddingSine2D,
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SAM2MaskDownSampler(keras.layers.Layer):
    """Strided convolutional stack compressing a mask to the feature grid.

    Repeats ``ZeroPadding -> Conv2D -> LayerNormalization -> activation``
    until the accumulated stride reaches ``total_stride``, then applies a single
    bare ``1x1`` convolution to ``embed_dim`` with **no** trailing normalization
    and **no** trailing activation.

    **Layer count and channel growth are derived, not configured.** The number
    of strided stages is ``log(total_stride) / log(stride)`` and each stage
    multiplies the channel count by ``stride ** 2``, starting from
    ``mask_in_chans``. At the shipped ``stride=2, total_stride=16`` that is four
    stages widening ``1 -> 4 -> 16 -> 64 -> 256``; at the reference class's
    signature default ``stride=4`` it would be two stages widening
    ``1 -> 16 -> 256``. **Both reach total stride 16**, which is why
    :attr:`num_layers` and :attr:`channel_sequence` -- not the output
    resolution -- are the observable that distinguishes them.

    **Padding is explicit, not ``'same'``.** ``padding='same'`` at ``k=3,
    s=2`` pads asymmetrically (nothing on the top/left, one row on the
    bottom/right), whereas the reference pads one row on **both** sides. The
    two agree on output SHAPE and disagree on values at the borders, so this
    layer pads explicitly and convolves with ``padding='valid'``.

    :param embed_dim: Output channel width of the final ``1x1`` projection.
    :type embed_dim: int
    :param kernel_size: Kernel size of every strided convolution.
    :type kernel_size: int
    :param stride: Stride of every strided convolution. Must be at least 2.
    :type stride: int
    :param padding: Symmetric zero padding applied before every strided
        convolution.
    :type padding: int
    :param total_stride: Accumulated stride of the whole stack.
    :type total_stride: int
    :param mask_in_chans: Channel width of the incoming mask. The channel
        growth sequence is derived from it.
    :type mask_in_chans: int
    :param activation: Activation applied after each normalization.
    :type activation: str
    :param norm_epsilon: Epsilon of every ``LayerNormalization``.
    :type norm_epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``stride`` is less than 2, if ``total_stride`` is not
        an exact integer power of ``stride``, or if any of ``embed_dim``,
        ``kernel_size``, ``mask_in_chans`` is not positive, or if ``padding`` is
        negative.

    Example:
        >>> import numpy as np
        >>> down = SAM2MaskDownSampler(embed_dim=32)
        >>> down.num_layers
        4
        >>> down.channel_sequence
        (4, 16, 64, 256)
        >>> tuple(down(np.zeros((1, 64, 64, 1), dtype="float32")).shape)
        (1, 4, 4, 32)
    """

    def __init__(
            self,
            embed_dim: int = 256,
            kernel_size: int = 3,
            stride: int = 2,
            padding: int = 1,
            total_stride: int = 16,
            mask_in_chans: int = 1,
            activation: str = "gelu",
            norm_epsilon: float = 1e-6,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if stride < 2:
            raise ValueError(
                f"stride must be at least 2 -- a stride of 1 never reaches "
                f"total_stride, got {stride}"
            )
        if padding < 0:
            raise ValueError(f"padding must not be negative, got {padding}")
        if mask_in_chans <= 0:
            raise ValueError(
                f"mask_in_chans must be positive, got {mask_in_chans}")

        num_layers = int(round(
            math.log(float(total_stride)) / math.log(float(stride))))
        if num_layers < 1 or stride ** num_layers != int(total_stride):
            raise ValueError(
                f"total_stride must be an exact positive integer power of "
                f"stride, got total_stride={total_stride}, stride={stride}"
            )

        # Store ALL configuration parameters.
        self.embed_dim = int(embed_dim)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.total_stride = int(total_stride)
        self.mask_in_chans = int(mask_in_chans)
        self.activation = activation
        self.norm_epsilon = float(norm_epsilon)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-016
        # The channel ladder is DERIVED from `stride ** 2`, never listed. Do
        # NOT replace this with a hardcoded (4, 16, 64, 256): that literal is
        # only correct at stride 2, and the whole point of the guard on this
        # layer is that a wrong `stride` keeps the total stride at 16 while
        # changing both the layer count and the ladder. See decisions.md D-016.
        channels: List[int] = []
        width = self.mask_in_chans
        for _ in range(num_layers):
            width = width * self.stride * self.stride
            channels.append(width)
        self._channel_sequence: Tuple[int, ...] = tuple(channels)

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.pads = [
            keras.layers.ZeroPadding2D(
                padding=self.padding, name=f"pad_{index}")
            for index in range(num_layers)
        ]
        self.convs = [
            keras.layers.Conv2D(
                filters=width,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding="valid",
                name=f"down_conv_{index}",
            )
            for index, width in enumerate(self._channel_sequence)
        ]
        self.norms = [
            keras.layers.LayerNormalization(
                axis=-1, epsilon=self.norm_epsilon, name=f"down_norm_{index}")
            for index in range(num_layers)
        ]
        self.activations = [
            keras.layers.Activation(self.activation, name=f"down_act_{index}")
            for index in range(num_layers)
        ]
        #: Bare projection: no normalization and no activation follow it.
        self.final_conv = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name="final_proj",
        )

    @property
    def num_layers(self) -> int:
        """Number of strided convolution stages.

        :return: ``log(total_stride) / log(stride)``.
        :rtype: int
        """
        return len(self.convs)

    @property
    def channel_sequence(self) -> Tuple[int, ...]:
        """Output channel width of each strided stage, in order.

        :return: ``mask_in_chans * stride**2`` compounded per stage.
        :rtype: Tuple[int, ...]
        """
        return self._channel_sequence

    def _stage_spatial(self, size: Optional[int]) -> Optional[int]:
        """Apply one strided stage to a single spatial dimension.

        :param size: Input extent, or ``None`` when dynamic.
        :type size: Optional[int]
        :return: Output extent, or ``None``.
        :rtype: Optional[int]
        """
        if size is None:
            return None
        return (int(size) + 2 * self.padding - self.kernel_size) // self.stride + 1

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every stage against its own propagated shape.

        :param input_shape: ``(batch, height, width, mask_in_chans)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4, or if its channel width
            disagrees with ``mask_in_chans``.
        """
        if self.built:
            return

        shape = tuple(input_shape)
        if len(shape) != 4:
            raise ValueError(
                f"SAM2MaskDownSampler expects a rank-4 channels-last input, "
                f"got {shape}"
            )
        if shape[-1] is not None and int(shape[-1]) != self.mask_in_chans:
            raise ValueError(
                f"input carries {shape[-1]} channels but mask_in_chans is "
                f"{self.mask_in_chans}; the channel growth ladder "
                f"{self._channel_sequence} is derived from it"
            )

        height, width = shape[1], shape[2]
        for index in range(self.num_layers):
            padded = (
                shape[0],
                None if height is None else height + 2 * self.padding,
                None if width is None else width + 2 * self.padding,
                self.mask_in_chans if index == 0
                else self._channel_sequence[index - 1],
            )
            self.pads[index].build(
                (shape[0], height, width, padded[-1]))
            self.convs[index].build(padded)
            height = self._stage_spatial(height)
            width = self._stage_spatial(width)
            stage_shape = (
                shape[0], height, width, self._channel_sequence[index])
            self.norms[index].build(stage_shape)
            self.activations[index].build(stage_shape)

        self.final_conv.build(
            (shape[0], height, width, self._channel_sequence[-1]))

        logger.debug(
            "SAM2MaskDownSampler built: %d stages, channels %s, embed_dim %d",
            self.num_layers, self._channel_sequence, self.embed_dim,
        )
        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Downsample a mask to the feature grid.

        :param inputs: ``(batch, height, width, mask_in_chans)``.
        :type inputs: Any
        :param training: Keras training flag, forwarded to the sub-layers.
        :type training: Optional[bool]
        :return: ``(batch, height // total_stride, width // total_stride,
            embed_dim)``.
        :rtype: Any
        """
        x = inputs
        for index in range(self.num_layers):
            x = self.pads[index](x)
            x = self.convs[index](x, training=training)
            x = self.norms[index](x, training=training)
            x = self.activations[index](x, training=training)
        # Bare 1x1: no normalization, no activation. See the class docstring.
        return self.final_conv(x, training=training)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the downsampled shape.

        :param input_shape: ``(batch, height, width, mask_in_chans)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, height', width', embed_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        shape = tuple(input_shape)
        height, width = shape[1], shape[2]
        for _ in range(self.num_layers):
            height = self._stage_spatial(height)
            width = self._stage_spatial(width)
        return (shape[0], height, width, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "total_stride": self.total_stride,
            "mask_in_chans": self.mask_in_chans,
            "activation": self.activation,
            "norm_epsilon": self.norm_epsilon,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SAM2Fuser(keras.layers.Layer):
    """Stack of residual ConvNeXt V1 blocks fusing mask and pixel features.

    Each block is a :class:`~dl_techniques.layers.convnext_v1_block.ConvNextV1Block`
    -- depthwise ``KxK`` convolution, ``LayerNormalization``, ``1x1`` expansion
    to ``4 * dim``, GELU, ``1x1`` reduction, learnable per-channel ``gamma``
    -- which reproduces the reference fuser block exactly.

    .. important::
        ``ConvNextV1Block`` is the residual **branch only**: it does not add the
        skip connection. The reference block does. **This layer therefore adds
        the residual itself** (``x = x + block(x)``). Removing that addition
        leaves every shape unchanged and silently deletes the residual path.

    :param dim: Channel width, unchanged by the stack.
    :type dim: int
    :param num_layers: Number of blocks.
    :type num_layers: int
    :param kernel_size: Depthwise kernel size of each block.
    :type kernel_size: int
    :param gamma_initial_value: Initial value of each block's layer-scale
        ``gamma``. The reference calls this ``layer_scale_init_value``.
    :type gamma_initial_value: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``dim``, ``num_layers`` or ``kernel_size`` is not
        positive.

    Example:
        >>> import numpy as np
        >>> fuser = SAM2Fuser(dim=16, num_layers=2)
        >>> tuple(fuser(np.zeros((1, 4, 4, 16), dtype="float32")).shape)
        (1, 4, 4, 16)
    """

    def __init__(
            self,
            dim: int = 256,
            num_layers: int = 2,
            kernel_size: int = 7,
            gamma_initial_value: float = 1e-6,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        # Store ALL configuration parameters.
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.gamma_initial_value = float(gamma_initial_value)

        # Sub-layers -- created unconditionally, built explicitly in build().
        #
        # REUSE, not reimplementation: the repo block's computation order was
        # verified against a float64 NumPy transcription of the reference
        # block's forward pass (see the A-6 oracle in the step-5 tests).
        self.blocks = [
            ConvNextV1Block(
                kernel_size=self.kernel_size,
                filters=self.dim,
                gamma_initial_value=self.gamma_initial_value,
                name=f"fuser_block_{index}",
            )
            for index in range(self.num_layers)
        ]

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every block against the unchanged input shape.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel width is
            not ``dim``.
        """
        if self.built:
            return

        shape = tuple(input_shape)
        if len(shape) != 4:
            raise ValueError(
                f"SAM2Fuser expects a rank-4 channels-last input, got {shape}")
        if shape[-1] is not None and int(shape[-1]) != self.dim:
            raise ValueError(
                f"SAM2Fuser is configured for {self.dim} channels but received "
                f"{shape[-1]}; the memory encoder fuses ADDITIVELY, so a width "
                f"of 2 * dim means the fusion was concatenated"
            )

        for block in self.blocks:
            block.build(shape)

        logger.debug(
            "SAM2Fuser built: %d blocks, dim %d, kernel %d",
            self.num_layers, self.dim, self.kernel_size,
        )
        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Apply the residual block stack.

        :param inputs: ``(batch, height, width, dim)``.
        :type inputs: Any
        :param training: Keras training flag, forwarded to the blocks.
        :type training: Optional[bool]
        :return: Same shape as ``inputs``.
        :rtype: Any
        """
        x = inputs
        for block in self.blocks:
            # ConvNextV1Block is the BRANCH only -- the residual is ours.
            x = x + block(x, training=training)
        return x

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the (unchanged) output shape.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``input_shape``.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_layers": self.num_layers,
            "kernel_size": self.kernel_size,
            "gamma_initial_value": self.gamma_initial_value,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SAM2MemoryEncoder(keras.layers.Layer):
    """Encode a predicted mask plus pixel features into spatial memory.

    Forward order, in full:

    .. code-block:: text

        masks  = sigmoid(masks) * sigmoid_scale + sigmoid_bias
        masks  = mask_downsampler(masks)          # -> (H', W', in_dim)
        x      = pix_feat_proj(pix_feat)          # 1x1, NOT identity
        x      = x + masks                        # ADDITIVE, never concat
        x      = fuser(x)
        x      = out_proj(x)                      # -> (H', W', out_dim)
        pos    = position_encoding(x)             # on the POST-out_proj x

    **The affine on the mask logits, and its ORDER.** The sigmoid comes FIRST
    and the affine SECOND: ``20 * sigmoid(x) - 10`` maps a probability onto the
    signed range ``(-10, +10)``. Reversing the two -- ``sigmoid(20 * x - 10)``
    -- yields a near-step function in ``(0, 1)``: same shape, same dtype, a
    plausible loss, a ~20x smaller dynamic range and no negative half. A bare
    ``sigmoid(x)`` is the third plausible reading. At ``x = 0`` the three give
    ``0.0``, ``4.54e-5`` and ``0.5``; the guard in
    ``test_memory_encoder.py::TestMaskAffine`` discriminates all three at once.
    The transform is evaluated in the layer's **variable** dtype so that under
    ``mixed_float16`` neither the sigmoid nor the ``* 20`` rescale is taken at
    reduced precision.

    **The positional-encoding width.** The reference config's
    ``num_pos_feats: 64`` belongs to a class that halves its argument
    internally; this repo's :class:`PositionEmbeddingSine2D` does not, and emits
    ``2 * num_pos_feats``. The default here is therefore ``out_dim // 2``, and
    the invariant to assert is the OUTPUT width
    (:attr:`pos_enc_channels` ``== out_dim``), never the constructor argument.

    **A KNOWN, ACCEPTED half-pixel deviation in that encoding.** The reused
    :class:`PositionEmbeddingSine2D` normalizes coordinates to pixel CENTRES
    (``(i + 0.5) / H``, the DETR / SAM 1 convention); the reference SAM 2 sine
    encoding uses pixel EDGES (``(i + 1) / H``). Everything else matches,
    including the interleaved sin/cos layout. The position argument therefore
    differs by ``pi / H`` radians, MEASURED as a max absolute difference of
    0.098 / 0.049 / 0.012 at grids 32 / 64 / 256 against an amplitude of 1.
    This is accepted rather than corrected: the layer is shared repo-wide and
    both encodings feed projections trained alongside them, so it is a fixed
    reparametrization rather than an error -- but it WOULD matter to a port
    loading released upstream weights. The magnitude is pinned by
    ``test_neck.py::TestPositionEncodingWidth::
    test_the_half_pixel_offset_deviation_is_pinned_at_its_MEASURED_size``.

    :param in_dim: Channel width of the incoming pixel features and of the
        fused stream.
    :type in_dim: int
    :param out_dim: Channel width of the produced memory.
    :type out_dim: int
    :param mask_kernel_size: Downsampler kernel size.
    :type mask_kernel_size: int
    :param mask_stride: Downsampler stride per stage.
    :type mask_stride: int
    :param mask_padding: Downsampler symmetric padding per stage.
    :type mask_padding: int
    :param mask_total_stride: Downsampler accumulated stride.
    :type mask_total_stride: int
    :param mask_in_chans: Channel width of the incoming mask.
    :type mask_in_chans: int
    :param num_fuser_layers: Number of ConvNeXt blocks in the fuser.
    :type num_fuser_layers: int
    :param fuser_kernel_size: Depthwise kernel size of each fuser block.
    :type fuser_kernel_size: int
    :param gamma_initial_value: Fuser layer-scale initial value.
    :type gamma_initial_value: float
    :param sigmoid_scale: Multiplier applied to the mask logits.
    :type sigmoid_scale: float
    :param sigmoid_bias: Offset added to the scaled mask logits.
    :type sigmoid_bias: float
    :param skip_mask_sigmoid: When ``True`` the mask is consumed as-is, with
        neither the affine nor the sigmoid. Used when the caller already
        binarized the mask.
    :type skip_mask_sigmoid: bool
    :param num_pos_feats: Half the positional-encoding output width. ``None``
        defers to ``out_dim // 2``.
    :type num_pos_feats: Optional[int]
    :param pos_enc_temperature: Sine positional-encoding temperature.
    :type pos_enc_temperature: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``in_dim`` or ``out_dim`` is not positive, if
        ``out_dim`` is odd while ``num_pos_feats`` is defaulted, or if
        ``num_pos_feats`` is not positive.

    Example:
        >>> import numpy as np
        >>> encoder = SAM2MemoryEncoder(in_dim=32, out_dim=8)
        >>> memory, position = encoder([
        ...     np.zeros((1, 4, 4, 32), dtype="float32"),
        ...     np.zeros((1, 64, 64, 1), dtype="float32"),
        ... ])
        >>> tuple(memory.shape), tuple(position.shape)
        ((1, 4, 4, 8), (1, 4, 4, 8))
    """

    def __init__(
            self,
            in_dim: int = 256,
            out_dim: int = 64,
            mask_kernel_size: int = 3,
            mask_stride: int = 2,
            mask_padding: int = 1,
            mask_total_stride: int = 16,
            mask_in_chans: int = 1,
            num_fuser_layers: int = 2,
            fuser_kernel_size: int = 7,
            gamma_initial_value: float = 1e-6,
            sigmoid_scale: float = 20.0,
            sigmoid_bias: float = -10.0,
            skip_mask_sigmoid: bool = False,
            num_pos_feats: Optional[int] = None,
            pos_enc_temperature: float = 10000.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if num_pos_feats is None and out_dim % 2 != 0:
            raise ValueError(
                f"out_dim must be even when num_pos_feats is defaulted so the "
                f"sine encoding can split it between the two spatial axes, got "
                f"{out_dim}"
            )

        # Store ALL configuration parameters.
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.mask_kernel_size = int(mask_kernel_size)
        self.mask_stride = int(mask_stride)
        self.mask_padding = int(mask_padding)
        self.mask_total_stride = int(mask_total_stride)
        self.mask_in_chans = int(mask_in_chans)
        self.num_fuser_layers = int(num_fuser_layers)
        self.fuser_kernel_size = int(fuser_kernel_size)
        self.gamma_initial_value = float(gamma_initial_value)
        self.sigmoid_scale = float(sigmoid_scale)
        self.sigmoid_bias = float(sigmoid_bias)
        self.skip_mask_sigmoid = bool(skip_mask_sigmoid)
        self.pos_enc_temperature = float(pos_enc_temperature)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-013
        # Do NOT "fix" this to `out_dim` to match the reference config's
        # literal `num_pos_feats: 64`. That literal belongs to a class which
        # halves it internally; this repo's `PositionEmbeddingSine2D` does not.
        # Passing 64 here yields a 128-wide encoding against a 64-wide memory
        # stream -- and on a square grid it BROADCASTS rather than raising, so
        # unlike the neck (D-013, where 512-vs-256 fails loudly) this site is
        # silent. Assert `pos_enc_channels == out_dim`. See decisions.md D-013.
        self.num_pos_feats = (
            self.out_dim // 2 if num_pos_feats is None else int(num_pos_feats)
        )
        if self.num_pos_feats <= 0:
            raise ValueError(
                f"num_pos_feats must be positive, got {self.num_pos_feats}")

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.mask_downsampler = SAM2MaskDownSampler(
            embed_dim=self.in_dim,
            kernel_size=self.mask_kernel_size,
            stride=self.mask_stride,
            padding=self.mask_padding,
            total_stride=self.mask_total_stride,
            mask_in_chans=self.mask_in_chans,
            name="mask_downsampler",
        )
        #: A real 1x1 convolution, NOT an identity.
        self.pix_feat_proj = keras.layers.Conv2D(
            filters=self.in_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name="pix_feat_proj",
        )
        self.fuser = SAM2Fuser(
            dim=self.in_dim,
            num_layers=self.num_fuser_layers,
            kernel_size=self.fuser_kernel_size,
            gamma_initial_value=self.gamma_initial_value,
            name="fuser",
        )
        self.out_proj = keras.layers.Conv2D(
            filters=self.out_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            name="out_proj",
        )
        self.position_encoding = PositionEmbeddingSine2D(
            num_pos_feats=self.num_pos_feats,
            temperature=self.pos_enc_temperature,
            normalize=True,
            name="position_encoding",
        )

    @property
    def pos_enc_channels(self) -> int:
        """Channel width of the returned positional encoding.

        :return: ``2 * num_pos_feats``.
        :rtype: int
        """
        return 2 * self.num_pos_feats

    def build(
            self, input_shape: Sequence[Tuple[Optional[int], ...]]
    ) -> None:
        """Build the downsampler, projections and fuser.

        :param input_shape: ``(pix_feat_shape, mask_shape)``.
        :type input_shape: Sequence[Tuple[Optional[int], ...]]
        :raises ValueError: If two shapes are not supplied, if either is not
            rank 4, if the pixel features do not carry ``in_dim`` channels, or
            if the downsampled mask grid does not match the pixel-feature grid.
        """
        if self.built:
            return

        shapes = [tuple(shape) for shape in input_shape]
        if len(shapes) != 2:
            raise ValueError(
                f"SAM2MemoryEncoder expects (pix_feat, masks), got "
                f"{len(shapes)} inputs"
            )
        pix_shape, mask_shape = shapes
        for name, shape in (("pix_feat", pix_shape), ("masks", mask_shape)):
            if len(shape) != 4:
                raise ValueError(
                    f"{name} must be a rank-4 channels-last shape, got {shape}")
        if pix_shape[-1] is not None and int(pix_shape[-1]) != self.in_dim:
            raise ValueError(
                f"pix_feat carries {pix_shape[-1]} channels but in_dim is "
                f"{self.in_dim}"
            )

        self.mask_downsampler.build(mask_shape)
        down_shape = self.mask_downsampler.compute_output_shape(mask_shape)
        for axis in (1, 2):
            if (down_shape[axis] is not None and pix_shape[axis] is not None
                    and int(down_shape[axis]) != int(pix_shape[axis])):
                raise ValueError(
                    f"the downsampled mask grid {down_shape[1:3]} does not "
                    f"match the pixel-feature grid {pix_shape[1:3]}; the "
                    f"fusion is ADDITIVE and needs both on the same grid"
                )

        self.pix_feat_proj.build(pix_shape)
        fused_shape = (
            pix_shape[0], pix_shape[1], pix_shape[2], self.in_dim)
        self.fuser.build(fused_shape)
        self.out_proj.build(fused_shape)
        memory_shape = (
            pix_shape[0], pix_shape[1], pix_shape[2], self.out_dim)
        self.position_encoding.build(memory_shape)

        logger.debug(
            "SAM2MemoryEncoder built: in_dim %d, out_dim %d, downsampler "
            "stages %d, fuser blocks %d",
            self.in_dim, self.out_dim, self.mask_downsampler.num_layers,
            self.num_fuser_layers,
        )
        super().build(input_shape)

    def _affine_sigmoid(self, masks: Any) -> Any:
        """Apply ``sigmoid(masks) * sigmoid_scale + sigmoid_bias``.

        # DECISION plan-2026-08-04T044628-4c240b4c/D-033
        # SIGMOID FIRST, AFFINE SECOND. Do NOT "simplify" this to
        # `sigmoid(scale * masks + bias)`: that is a different function
        # (a near-step map into `(0, 1)` instead of an affine rescale into
        # `(-10, +10)`), it produces the same shapes, the same dtype and a
        # plausible loss, and it is exactly the defect this round exists to
        # repair. The order is fixed by the reference implementation, which
        # sigmoids the mask itself and then calls the encoder with
        # `skip_mask_sigmoid=True`. See decisions.md D-033.

        # DECISION plan-2026-08-04T044628-4c240b4c/D-045
        # Do NOT delete the `variable_dtype` cast below. Its ORIGINAL
        # justification (float16 overflow of `20 * x`) was made false by the
        # D-033 order fix, and an adversarial review correctly identified the
        # guard on it as vacuous and proposed removing the machinery. The cast
        # stays for a DIFFERENT, measured reason -- resolution, stated below --
        # and it now has a guard that can actually go red. See decisions.md
        # D-045.

        Evaluated in the layer's VARIABLE dtype rather than its compute dtype.
        The reason is PRECISION, not overflow: with the sigmoid taken first
        (D-033) the product is bounded by 20 and cannot overflow float16, so an
        overflow probe here is vacuous. What the cast buys is resolution. Under
        ``mixed_float16`` the intermediate ``sigmoid(x) * 20`` lives near
        ``+-10``, where float16's spacing is ``7.8e-3``; subtracting 10 then
        leaves an output near 0 quantized at that coarse spacing rather than at
        the ``4.9e-4`` its own magnitude would allow. MEASURED over logits in
        ``[-1, 1]`` against a float64 oracle, with float16 INPUT in both arms:
        max error ``2.4e-3`` computing in float32, ``1.3e-2`` computing in
        float16 -- a 5.3x loss, on 81% of the probed logits. The result is cast
        back to the compute dtype, so this costs the caller nothing.

        :param masks: Mask logits.
        :type masks: Any
        :return: Transformed mask, in the compute dtype.
        :rtype: Any
        """
        if self.skip_mask_sigmoid:
            return ops.cast(masks, self.compute_dtype)
        work = ops.cast(masks, self.variable_dtype)
        scale = ops.cast(self.sigmoid_scale, self.variable_dtype)
        bias = ops.cast(self.sigmoid_bias, self.variable_dtype)
        return ops.cast(
            ops.sigmoid(work) * scale + bias, self.compute_dtype)

    def call(
            self,
            inputs: Sequence[Any],
            training: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Encode one frame's mask and pixel features into memory.

        :param inputs: ``(pix_feat, masks)`` -- pixel features
            ``(batch, h, w, in_dim)`` and high-resolution mask LOGITS
            ``(batch, H, W, mask_in_chans)``.
        :type inputs: Sequence[Any]
        :param training: Keras training flag, forwarded to the sub-layers.
        :type training: Optional[bool]
        :return: ``(memory, position)``, both
            ``(batch, h, w, out_dim)``.
        :rtype: Tuple[Any, Any]
        """
        pix_feat, masks = inputs[0], inputs[1]

        masks = self._affine_sigmoid(masks)
        masks = self.mask_downsampler(masks, training=training)

        x = self.pix_feat_proj(pix_feat, training=training)
        # ADDITIVE fusion. A concatenation here would double the width and the
        # fuser's build-time check would raise -- deliberately.
        x = x + masks
        x = self.fuser(x, training=training)
        x = self.out_proj(x, training=training)

        # `PositionEmbeddingSine2D` emits (batch, channels, height, width) in
        # float32 at every dtype policy. Both the transpose and the cast are
        # required; neither is cosmetic.
        position = self.position_encoding(x)
        position = ops.transpose(position, (0, 2, 3, 1))
        position = ops.cast(position, x.dtype)
        return x, position

    def compute_output_shape(
            self, input_shape: Sequence[Tuple[Optional[int], ...]]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return the memory and positional-encoding shapes.

        :param input_shape: ``(pix_feat_shape, mask_shape)``.
        :type input_shape: Sequence[Tuple[Optional[int], ...]]
        :return: ``(memory_shape, position_shape)``.
        :rtype: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
        """
        pix_shape = tuple(input_shape[0])
        memory_shape = (
            pix_shape[0], pix_shape[1], pix_shape[2], self.out_dim)
        position_shape = (
            pix_shape[0], pix_shape[1], pix_shape[2], self.pos_enc_channels)
        return memory_shape, position_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "mask_kernel_size": self.mask_kernel_size,
            "mask_stride": self.mask_stride,
            "mask_padding": self.mask_padding,
            "mask_total_stride": self.mask_total_stride,
            "mask_in_chans": self.mask_in_chans,
            "num_fuser_layers": self.num_fuser_layers,
            "fuser_kernel_size": self.fuser_kernel_size,
            "gamma_initial_value": self.gamma_initial_value,
            "sigmoid_scale": self.sigmoid_scale,
            "sigmoid_bias": self.sigmoid_bias,
            "skip_mask_sigmoid": self.skip_mask_sigmoid,
            "num_pos_feats": self.num_pos_feats,
            "pos_enc_temperature": self.pos_enc_temperature,
        })
        return config

# ---------------------------------------------------------------------
