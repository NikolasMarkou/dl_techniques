"""Segmentation head: per-pixel class logits, upsampled toward input size."""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...conv_blocks.conv_block import ConvBlock
from dl_techniques.utils.tensors import is_power_of_two, is_single_shape
from ..validation import positive_int
from dl_techniques.utils.keras_registration import register_dl_technique
from .base import BaseVisionHead
from .utils import _require_hidden_dim_at_least


@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class SegmentationHead(BaseVisionHead):
    """
    One map of per-pixel class logits, upsampled towards the input resolution.

    Three ``ConvBlock`` stages refine the features, then transposed convs
    double the resolution once per stage, then a linear 1x1 conv emits one
    channel per class. The output is pre-softmax: compile a per-pixel
    cross-entropy with ``from_logits=True``, or apply
    ``keras.ops.softmax(out, axis=-1)`` for class probabilities.

    This head accepts a single feature map or a list of them. With a list and
    ``use_skip_connections``, the last map is the one refined and the rest
    become skips in reverse order. One skip is concatenated after each
    refine block, while skips remain.

    Two skip channel modes are supported:
    - ``"auto"`` (default): refine blocks halve channels each step
      (``hidden_dim``, ``hidden_dim//2``, ``hidden_dim//4``), skip
      concatenation dynamically widens the tensor.
    - ``"explicit"``: caller provides ``explicit_skip_channels`` list
      matching the encoder feature channels (e.g., ``[32, 64, 128, 256]``
      for AccUNet). Refine block output channels are derived from the
      corresponding skip channels, and the upsample convs keep the width the
      refine stack ends at. ``hidden_dim`` is then read only by the
      attention and FFN stages: with both off it is refused, and stored as
      ``None``.

    This is the only head in the module that returns a bare tensor. Every
    other one returns a dict.

    **Architecture Overview (auto mode):**

    .. code-block:: text

        inputs: (B, H, W, C), or a list of feature maps
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ list input: x = the last map,   │
        │ skips = the rest, reversed      │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ refine_blocks: 3 x ConvBlock    │
        │ concat skips[i] after block i   │
        └────────────────┬────────────────┘
                         ▼ channels: hidden_dim, then halved
        ┌─────────────────────────────────┐
        │ upsample_layers                 │
        │ Conv2DTranspose stride 2        │
        └────────────────┬────────────────┘
                         ▼ log2(upsampling_factor) of them
        ┌─────────────────────────────────┐
        │ seg_head: Conv2D 1x1     linear │
        └────────────────┬────────────────┘
                         ▼
        (B, H*s, W*s, num_classes), s = 2 ** the count above

    In auto mode the refine blocks halve their channel count as they go:
    ``hidden_dim``, then ``hidden_dim // 2``, then ``hidden_dim // 4``. Every
    upsample layer uses ``hidden_dim // 8``, one halving further still.
    Concatenating a skip
    widens the tensor again, so the built shapes follow ``call`` rather than
    the block widths alone.

    The output resolution is ``upsampling_factor`` times the input: one
    stride-2 transposed conv per doubling, ``log2(upsampling_factor)`` of
    them. With the default ``upsampling_factor=4`` that is 2 layers. A factor
    that is not a power of two raises ``ValueError``.

    Input shape:
        ``(batch, height, width, channels)``, or a list of such shapes.

    Output shape:
        ``(batch, out_height, out_width, num_classes)``.

    :param num_classes: Number of segmentation classes.
    :type num_classes: int
    :param upsampling_factor: Output resolution over input resolution, a
        power of two. ``log2`` of it stride-2 transposed convs are built.
    :type upsampling_factor: int
    :param use_skip_connections: Concatenate list inputs as skips. Ignored
        when the input is a single tensor.
    :type use_skip_connections: bool
    :param skip_channel_mode: ``"auto"`` (default) or ``"explicit"``. In
        explicit mode, ``explicit_skip_channels`` must be provided.
    :type skip_channel_mode: Literal["auto", "explicit"]
    :param explicit_skip_channels: List of channel counts for each skip
        connection, ordered from deepest to shallowest (matching the reversed
        encoder features). Required when ``skip_channel_mode="explicit"``.
    :type explicit_skip_channels: Optional[List[int]]
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar refine_blocks: The three refinement ``ConvBlock`` layers.
    :vartype refine_blocks: List[ConvBlock]
    :ivar upsample_layers: The stride-2 ``Conv2DTranspose`` layers.
    :vartype upsample_layers: List[keras.layers.Conv2DTranspose]
    :ivar seg_head: Output ``Conv2D(num_classes)``, linear: it emits the
        class logits.
    :vartype seg_head: keras.layers.Conv2D
    """

    def __init__(
            self,
            num_classes: int,
            upsampling_factor: int = 4,
            use_skip_connections: bool = True,
            skip_channel_mode: Literal["auto", "explicit"] = "auto",
            explicit_skip_channels: Optional[List[int]] = None,
            input_format: Literal["spatial", "sequence"] = "spatial",
            patch_grid_size: Optional[Tuple[int, int]] = None,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the segmentation layers.

        :param num_classes: Number of segmentation classes.
        :type num_classes: int
        :param upsampling_factor: Output over input resolution, a power of
            two.
        :type upsampling_factor: int
        :param use_skip_connections: Concatenate list inputs as skips.
        :type use_skip_connections: bool
        :param skip_channel_mode: ``"auto"`` or ``"explicit"``.
        :type skip_channel_mode: Literal["auto", "explicit"]
        :param explicit_skip_channels: Channel counts for explicit skip mode.
        :type explicit_skip_channels: Optional[List[int]]
        :param input_format: Input format, ``"spatial"`` for (B,H,W,C) or
            ``"sequence"`` for (B,N,D) ViT patch tokens. Defaults to ``"spatial"``.
        :type input_format: Literal["spatial", "sequence"]
        :param patch_grid_size: Tuple ``(H, W)`` of patch grid dimensions for
            ``input_format="sequence"``. Required when using sequence format.
        :type patch_grid_size: Optional[Tuple[int, int]]
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        :raises ValueError: In explicit mode, if ``hidden_dim`` is given while
            neither the attention nor the FFN stage is on (no layer reads it).
        """
        given_hidden_dim = kwargs.get('hidden_dim')
        if given_hidden_dim is None:
            kwargs.pop('hidden_dim', None)
        super().__init__(**kwargs)

        self.num_classes = positive_int(
            num_classes, 'num_classes', type(self).__name__)
        self.upsampling_factor = upsampling_factor
        self.use_skip_connections = use_skip_connections
        self.skip_channel_mode = skip_channel_mode
        self.explicit_skip_channels = explicit_skip_channels
        self.input_format = input_format
        self.patch_grid_size = patch_grid_size

        if self.input_format == "sequence" and self.patch_grid_size is None:
            raise ValueError("patch_grid_size is required when input_format='sequence'")

        if self.skip_channel_mode == "explicit" and self.explicit_skip_channels is None:
            raise ValueError("explicit_skip_channels is required when skip_channel_mode='explicit'")

        # DECISION plan-2026-10-14T062616-81938e81/D-028
        # One stride-2 transposed conv per doubling: log2(factor) of them. Do
        # NOT go back to int(sqrt(factor)): it equals log2 only at 2, 4, 16
        # and 32, so factor 8 gave 4x and 64 gave 256x, and FFTNet (factor =
        # patch_size) returned a map that was not image-sized for patch 8.
        # See decisions.md D-028.
        if (isinstance(upsampling_factor, bool)
                or not isinstance(upsampling_factor, int)
                or not is_power_of_two(upsampling_factor)):
            raise ValueError(
                f"upsampling_factor must be a power of two (1, 2, 4, 8, ...), "
                f"got {upsampling_factor!r}"
            )
        self._num_upsamples = upsampling_factor.bit_length() - 1

        # Auto mode: every upsample conv takes hidden_dim // 8 channels and
        # the refine blocks hidden_dim, // 2 and // 4. Explicit mode sizes
        # both from explicit_skip_channels, so hidden_dim is read only by the
        # attention and FFN stages.
        if self.skip_channel_mode == "explicit":
            self._drop_unread_hidden_dim(
                given_hidden_dim,
                "skip_channel_mode='explicit' sizes the refine blocks and the "
                "upsample convs from explicit_skip_channels")
        elif self._num_upsamples:
            _require_hidden_dim_at_least(
                self.hidden_dim, 8, type(self).__name__,
                "its upsample convs are created at hidden_dim // 8 channels")
        else:
            _require_hidden_dim_at_least(
                self.hidden_dim, 4, type(self).__name__,
                "its third refine block takes hidden_dim // 4 channels")

        self._create_segmentation_layers()

    def _create_segmentation_layers(self) -> None:
        """
        Build the refine stack, the upsample stack and the output conv.

        In "auto" mode, refine channel counts halve each step.
        In "explicit" mode, refine block output channels match the
        corresponding skip channels (from explicit_skip_channels).

        :return: None.
        :rtype: None
        """

        # Feature refinement blocks
        self.refine_blocks = []

        if self.skip_channel_mode == "explicit" and self.explicit_skip_channels is not None:
            # Explicit mode: refine blocks output channels to match skip channels
            # We have 3 refine blocks, explicit_skip_channels has 4 entries (L0-L3)
            # Refine block i output should match skip_channels[-(i+2)] for concatenation
            for i, out_channels in enumerate(self._explicit_refine_widths()):
                self.refine_blocks.append(
                    ConvBlock(
                        filters=out_channels,
                        kernel_size=3,
                        normalization_type=self.normalization_type,
                        activation_type=self.activation_type,
                        dropout_rate=self.dropout_rate,
                        name=f'refine_block_{i}'
                    )
                )
        else:
            # Auto mode: halve channels each step
            channels = self.hidden_dim
            for i in range(3):
                self.refine_blocks.append(
                    ConvBlock(
                        filters=channels,
                        kernel_size=3,
                        normalization_type=self.normalization_type,
                        activation_type=self.activation_type,
                        dropout_rate=self.dropout_rate,
                        name=f'refine_block_{i}'
                    )
                )
                channels = channels // 2

        # Upsampling layers. Each keeps the width it is fed: in auto mode the
        # third halving of hidden_dim; in explicit mode the last refine width
        # plus the skip concatenated after it (_explicit_upsample_width).
        if self.skip_channel_mode == "explicit" and self.explicit_skip_channels is not None:
            filters = self._explicit_upsample_width()
        else:
            filters = self.hidden_dim // 8
        self.upsample_layers = []
        for i in range(self._num_upsamples):
            self.upsample_layers.append(
                keras.layers.Conv2DTranspose(
                    filters=filters,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    activation=self.activation_type,
                    name=f'upsample_{i}'
                )
            )

        # Final segmentation layer.
        # DECISION plan-2026-10-14T062616-81938e81/D-003
        # Linear: the class map is logits. Do NOT put activation='softmax'
        # back: a from_logits=True loss would double-softmax, and a
        # from_logits=False one hides the probabilities inside the head where
        # no consumer can choose. Range maps (the depth sigmoid) are the only
        # squashed outputs in this module. See decisions.md D-003.
        self.seg_head = keras.layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            padding='same',
            name='seg_head'
        )

    def _explicit_refine_widths(self) -> List[int]:
        """
        Output widths of the three refine blocks in explicit mode.

        Block ``i`` emits half the channels of the skip two levels up,
        ``explicit_skip_channels[-(i + 2)] // 2``.

        :return: The three widths, in block order.
        :rtype: List[int]
        """
        skip_ch = self.explicit_skip_channels
        return [skip_ch[-(i + 2)] // 2 for i in range(3)]

    def _explicit_upsample_width(self) -> int:
        """
        Width the explicit-mode upsample convs are fed and keep.

        The same arithmetic ``build`` and ``call`` follow: the last refine
        width, plus the skip concatenated after the last refine block when
        there is one (skip ``i`` follows block ``i``, for ``i < 3``).

        :return: The channel count after the refine stack.
        :rtype: int
        """
        width = self._explicit_refine_widths()[-1]
        if self.use_skip_connections and len(self.explicit_skip_channels) > 2:
            width += self.explicit_skip_channels[2]
        return width

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> None:
        """
        Build every sub-layer on the shape ``call`` will hand it.

        This repeats the skip-concatenation arithmetic of ``call``, so a
        multi-scale input builds the same widths it will run with. The common
        layers are built on the single highest-level map.

        In explicit mode, validates that skip shapes match explicit_skip_channels.

        For sequence input format, the input is reshaped to spatial in call(),
        so we build the layers for the spatial shape.

        :param input_shape: One feature-map shape, or a list of shapes for a
            multi-scale input.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :return: None.
        :rtype: None
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-027
        # Decide "list of shapes" by the ELEMENT type (is_single_shape), NOT
        # by isinstance(input_shape, list): load_model rebuilds this layer
        # from a JSON build_config, where ONE shape arrives as a list of ints,
        # and the list test then read it as several one-int shapes, so every
        # saved model holding this head failed to load. See decisions.md D-027.
        shape_list = not is_single_shape(input_shape)

        # For sequence input, compute the spatial shape after reshaping
        if self.input_format == "sequence":
            if self.patch_grid_size is None:
                raise ValueError("patch_grid_size must be set for sequence input format")
            if shape_list:
                spatial_shapes = []
                for shape in input_shape:
                    batch, _, dim = shape
                    h, w = self.patch_grid_size
                    spatial_shapes.append((batch, h, w, dim))
                input_shape = spatial_shapes
            else:
                batch, _, dim = input_shape
                h, w = self.patch_grid_size
                input_shape = (batch, h, w, dim)

        if shape_list and self.use_skip_connections:
            base_shape = input_shape[-1]
            skip_shapes = input_shape[:-1][::-1]
        else:
            base_shape = input_shape[-1] if shape_list else input_shape
            skip_shapes = []
        self._check_skip_resolution(base_shape, skip_shapes)

        # In explicit mode, validate skip channels
        if self.skip_channel_mode == "explicit" and self.explicit_skip_channels is not None:
            if len(skip_shapes) != len(self.explicit_skip_channels):
                raise ValueError(
                    f"explicit_skip_channels has {len(self.explicit_skip_channels)} entries "
                    f"but {len(skip_shapes)} skip inputs provided"
                )
            for i, (skip_shape, exp_ch) in enumerate(zip(skip_shapes, self.explicit_skip_channels)):
                if skip_shape[-1] != exp_ch:
                    raise ValueError(
                        f"Skip {i} channels {skip_shape[-1]} != explicit_skip_channels[{i}]={exp_ch}"
                    )

        shape = self._common_processed_shape(base_shape)

        for i, refine_block in enumerate(self.refine_blocks):
            refine_block.build(shape)
            shape = tuple(refine_block.compute_output_shape(shape))
            if self.use_skip_connections and i < len(skip_shapes):
                skip_channels = skip_shapes[i][-1]
                merged = (
                    shape[-1] + skip_channels
                    if shape[-1] is not None and skip_channels is not None
                    else None
                )
                shape = shape[:-1] + (merged,)

        for upsample_layer in self.upsample_layers:
            upsample_layer.build(shape)
            shape = tuple(upsample_layer.compute_output_shape(shape))

        self.seg_head.build(shape)

        # Common layers operate on the single highest-level feature map.
        super().build(base_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Refine, concatenate skips, upsample, then classify each pixel.

        :param inputs: One feature map, or a list of them with the
            highest-level map last. For sequence format, input is (B, N, D).
        :type inputs: Union[keras.KerasTensor, List[keras.KerasTensor]]
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Per-pixel class logits, pre-softmax. This head returns no
            dict.
        :rtype: keras.KerasTensor
        """

        # Handle sequence input format (ViT patch tokens)
        x = inputs
        if self.input_format == "sequence":
            if self.patch_grid_size is None:
                raise ValueError("patch_grid_size must be set for sequence input format")
            if isinstance(x, list):
                # Convert each element in the list
                x = [self._reshape_sequence_to_spatial(xi) for xi in x]
            else:
                x = self._reshape_sequence_to_spatial(x)

        # Handle multi-scale inputs if skip connections are used
        if isinstance(x, list) and self.use_skip_connections:
            # The last entry is the highest-level map. The rest are skips,
            # reversed so the deepest one is consumed first.
            x_main = x[-1]
            skip_features = x[:-1][::-1]
        else:
            x_main = x if not isinstance(x, list) else x[-1]
            skip_features = []
        self._check_skip_resolution(
            tuple(x_main.shape), [tuple(s.shape) for s in skip_features])

        # Apply common processing
        if self.use_attention:
            x_main = self._apply_attention(x_main, training=training)
        if self.use_ffn:
            x_main = self.ffn(x_main, training=training)

        # Refinement and upsampling
        for i, refine_block in enumerate(self.refine_blocks):
            x_main = refine_block(x_main, training=training)

            # Add skip connections if available
            if self.use_skip_connections and i < len(skip_features):
                x_main = keras.ops.concatenate([x_main, skip_features[i]], axis=-1)

        # Upsample to original resolution
        for upsample_layer in self.upsample_layers:
            x_main = upsample_layer(x_main)

        # Final segmentation output
        seg_output = self.seg_head(x_main)

        return seg_output

    @staticmethod
    def _check_skip_resolution(
            base_shape: Tuple[Optional[int], ...],
            skip_shapes: List[Tuple[Optional[int], ...]]
    ) -> None:
        """
        Refuse a skip whose height or width differs from the feature map's.

        The refine blocks keep the resolution and the upsampling runs only
        after them, so each skip is concatenated at the feature map's own
        height and width. A dimension unknown on either side (``None``) is
        not checked here.

        :param base_shape: Shape of the highest-level feature map.
        :type base_shape: Tuple[Optional[int], ...]
        :param skip_shapes: Shapes of the skips, in concatenation order.
        :type skip_shapes: List[Tuple[Optional[int], ...]]
        :return: None.
        :rtype: None
        :raises ValueError: If a known skip height or width differs.
        """
        base_hw = tuple(base_shape[1:3])
        for i, skip_shape in enumerate(skip_shapes):
            skip_hw = tuple(skip_shape[1:3])
            if any(a is not None and b is not None and a != b
                   for a, b in zip(base_hw, skip_hw)):
                raise ValueError(
                    f"SegmentationHead skip {i} is {skip_hw}, but every skip "
                    f"must have the feature map's height and width "
                    f"{base_hw}: the skips are concatenated after the refine "
                    f"blocks, before any upsampling."
                )

    def _reshape_sequence_to_spatial(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape sequence (B, N, D) to spatial (B, H, W, D)."""
        batch_size = keras.ops.shape(x)[0]
        h, w = self.patch_grid_size
        return keras.ops.reshape(x, (batch_size, h, w, -1))

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """
        Report the output shape.

        Each of the ``log2(upsampling_factor)`` transposed convs doubles
        height and width. The output conv emits ``num_classes``
        channels. A list input is measured by its last entry.

        For sequence input format, the input shape (B, N, D) is reshaped
        to spatial (B, H, W, D) using patch_grid_size before computing.

        :param input_shape: One feature-map shape, or a list of shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :return: Shape ``(batch, out_height, out_width, num_classes)``.
        :rtype: Tuple[Optional[int], ...]
        """
        base_shape = (
            input_shape if is_single_shape(input_shape) else input_shape[-1]
        )
        batch = base_shape[0]

        # Handle sequence input format
        if self.input_format == "sequence":
            if self.patch_grid_size is None:
                raise ValueError("patch_grid_size must be set for sequence input format")
            # base_shape is (B, N, D) -> reshape to (B, H, W, D)
            height, width = self.patch_grid_size
        else:
            height, width = base_shape[1], base_shape[2]

        scale = self.upsampling_factor
        out_height = height * scale if height is not None else None
        out_width = width * scale if width is not None else None
        return (batch, out_height, out_width, self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``num_classes``, ``upsampling_factor``,
            ``use_skip_connections``, ``skip_channel_mode``,
            ``explicit_skip_channels``, ``input_format``, and ``patch_grid_size``,
            on top of the base configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'upsampling_factor': self.upsampling_factor,
            'use_skip_connections': self.use_skip_connections,
            'skip_channel_mode': self.skip_channel_mode,
            'explicit_skip_channels': self.explicit_skip_channels,
            'input_format': self.input_format,
            'patch_grid_size': self.patch_grid_size
        })
        return config
