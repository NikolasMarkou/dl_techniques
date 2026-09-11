"""SCUNet: Swin-Conv-UNet for image restoration.

This file holds :class:`SCUNet`, a U-Net for denoising, deblurring and similar
restoration tasks, and the ``create_scunet`` factory. Each stage stacks
``SwinConvBlock``s that split their channels between a convolutional branch and
a windowed-attention transformer branch, alternating plain and shifted windows
for cross-window connectivity. The encoder-decoder is symmetric: 3 down stages,
a bottleneck and 3 up stages, joined by additive rather than concatenating skip
connections, with a stride product of 8. Inputs are reflect-padded to a
multiple of 64 and cropped back on output, 64 being the stride product times
the default window size of 8, so every stage's feature map divides by the
window. That makes ``SCUNet()(keras.Input((None, None, 3)))`` one model that
serves every concrete size. The pad target is a hardcoded 64 rather than a
function of ``window_size``. A reflect pad must be strictly smaller than the
extent it pads, so heights or widths below 33 raise, and no pretrained weights
ship with this port.

References:
    - Zhang et al., 2022. Practical Blind Denoising via Swin-Conv-UNet and
      Data Synthesis.
"""

import keras
from keras import ops
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.model_build import concretize_axes, materialize_sublayers
from dl_techniques.layers.transformers.swin_conv_block import SwinConvBlock
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.scunet.model")
class SCUNet(keras.Model):
    """Restore an image through a Swin-Conv U-Net.

    Three down stages, a bottleneck and three up stages, each stage a stack of
    ``SwinConvBlock``s. Skips are added, not concatenated, so a stage's output
    width equals the width of the tensor it rejoins.

    Architecture:

    .. code-block:: text

        input [B, H, W, C_in]
              │
              ▼
        reflect pad to a multiple of 64
              │
              ▼
        ┌──────────────────────────────┐
        │ head conv 3x3                │──► x1 [H, W, D]
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ down1: blocks, conv stride 2 │──► x2 [H/2, W/2, 2D]
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ down2: blocks, conv stride 2 │──► x3 [H/4, W/4, 4D]
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ down3: blocks, conv stride 2 │──► x4 [H/8, W/8, 8D]
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ body: blocks                 │
        └──────────────────────────────┘
              │
             (+)◄── x4
              ▼
        ┌──────────────────────────────┐
        │ up3: conv-t stride 2, blocks │
        └──────────────────────────────┘
              │
             (+)◄── x3
              ▼
        ┌──────────────────────────────┐
        │ up2: conv-t stride 2, blocks │
        └──────────────────────────────┘
              │
             (+)◄── x1 ... no: x2
              ▼
        ┌──────────────────────────────┐
        │ up1: conv-t stride 2, blocks │
        └──────────────────────────────┘
              │
             (+)◄── x1
              ▼
        ┌──────────────────────────────┐
        │ tail conv 3x3                │
        └──────────────────────────────┘
              │
              ▼
        crop to [H, W]
              │
              ▼
        output [B, H, W, C_in]

    Each labelled tensor is both the stage output and the skip added on the way
    back up, and every add sits on the input of the following stage.

    Stage internals:

    .. code-block:: text

        down stage                    up stage

        in                            in
        │                             │
        ▼                             ▼
        block 0, window W             conv transpose 2x2, stride 2
        block 1, window SW            block 0, window W
        ... alternating               block 1, window SW
        │                             ... alternating
        ▼                             │
        conv 2x2, stride 2            ▼
        │                             out
        ▼
        out

    Channel arithmetic, with D = ``dim``:

    .. code-block:: text

        stage   block width   conv_dim   trans_dim   stage output
        head    -             -          -           D
        down1   D             D/2        D/2         2D
        down2   2D            D          D           4D
        down3   4D            2D         2D          8D
        body    8D            4D         4D          8D
        up3     4D            2D         2D          4D
        up2     2D            D          D           2D
        up1     D             D/2        D/2         D
        tail    D             -          -           C_in

    A block's width is ``conv_dim + trans_dim``, so ``dim`` must be even for
    the outer stages, and the narrowest transformer branch is ``dim // 2``.

    Padding:

    .. code-block:: text

        H = 33   pad 31   ──►  64   ──►  /8  ──►  8, divides the window
        H = 64   pad  0   ──►  64
        H = 32   pad 32   ──►  rejected, pad not smaller than H

    :param in_nc: Number of input channels, and of output channels. Defaults
        to 3.
    :type in_nc: int
    :param config: Number of blocks per stage, 7 entries (3 down, 1 bottleneck,
        3 up). Defaults to ``[4, 4, 4, 4, 4, 4, 4]``.
    :type config: List[int]
    :param dim: Base dimension for feature channels. Must be positive and even.
        Defaults to 64.
    :type dim: int
    :param head_dim: Dimension of each attention head. The narrowest
        transformer branch is ``dim // 2``, and nothing here checks
        ``head_dim`` against it. Defaults to 32.
    :type head_dim: int
    :param window_size: Size of the attention window. Defaults to 8, which is
        what the hardcoded pad target of 64 assumes.
    :type window_size: int
    :param stochastic_depth_rate: Maximum stochastic depth rate, ramped
        linearly across all ``sum(config)`` blocks. Defaults to 0.0.
    :type stochastic_depth_rate: float
    :param input_resolution: Resolution hint forwarded to every
        ``SwinConvBlock`` (halved per stage), where it changes no attention
        geometry. ``build`` also traces the graph at this resolution, so give
        it a value a real input could have. Defaults to 256.
    :type input_resolution: int
    :param kwargs: Additional keyword arguments for the Model base class.

    :raises ValueError: If any constructor argument is outside its valid range;
        see :meth:`_validate_config`.
    """

    def __init__(
            self,
            in_nc: int = 3,
            config: List[int] = None,
            dim: int = 64,
            head_dim: int = 32,
            window_size: int = 8,
            stochastic_depth_rate: float = 0.0,
            input_resolution: int = 256,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if config is None:
            config = [4, 4, 4, 4, 4, 4, 4]

        self._validate_config(
            in_nc=in_nc,
            config=config,
            dim=dim,
            head_dim=head_dim,
            window_size=window_size,
            stochastic_depth_rate=stochastic_depth_rate,
            input_resolution=input_resolution,
        )

        self.in_nc = in_nc
        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.stochastic_depth_rate = stochastic_depth_rate
        self.input_resolution = input_resolution

        logger.info(f"Initializing SCUNet with config: {config}, dim: {dim}, "
                    f"window_size: {window_size}, input_resolution: {input_resolution}")

        # One rate per block, ramped across the whole network rather than per stage.
        dpr = linear_drop_path_rates(sum(config), stochastic_depth_rate)

        self._build_network(dpr)

    def _validate_config(
            self,
            in_nc: int,
            config: List[int],
            dim: int,
            head_dim: int,
            window_size: int,
            stochastic_depth_rate: float,
            input_resolution: int,
    ) -> None:
        """Validate constructor arguments, raising ``ValueError`` on invalid input.

        Checks each argument on its own. It does not check ``head_dim`` against
        ``dim // 2``, nor ``input_resolution`` or ``window_size`` against the
        pad target of 64.

        :param in_nc: Number of input channels.
        :type in_nc: int
        :param config: Per-stage block counts (must have exactly 7 entries).
        :type config: List[int]
        :param dim: Base feature dimension.
        :type dim: int
        :param head_dim: Attention head dimension.
        :type head_dim: int
        :param window_size: Attention window size.
        :type window_size: int
        :param stochastic_depth_rate: Maximum stochastic-depth drop rate.
        :type stochastic_depth_rate: float
        :param input_resolution: Expected input resolution.
        :type input_resolution: int
        :return: Nothing.
        :rtype: None

        :raises ValueError: If any argument is outside its valid range.
        """
        if in_nc <= 0:
            raise ValueError(f"in_nc must be positive, got {in_nc}")

        if not isinstance(config, (list, tuple)):
            raise ValueError(
                f"config must be a list/tuple, got {type(config).__name__}"
            )
        if len(config) != 7:
            raise ValueError(
                f"config must have exactly 7 stage entries "
                f"(3 down + bottleneck + 3 up), got {len(config)}"
            )
        if any((not isinstance(c, int)) or c <= 0 for c in config):
            raise ValueError(
                f"every config entry must be a positive int, got {config}"
            )

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim % 2 != 0:
            raise ValueError(
                f"dim must be even (the outer stages use dim // 2), got {dim}"
            )

        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        if not (0.0 <= stochastic_depth_rate <= 1.0):
            raise ValueError(
                f"stochastic_depth_rate must be in [0, 1], "
                f"got {stochastic_depth_rate}"
            )

        if input_resolution <= 0:
            raise ValueError(
                f"input_resolution must be positive, got {input_resolution}"
            )

    def _build_network(self, dpr: List[float]) -> None:
        """Create the head, the seven stages and the tail.

        Each down stage ends with its own strided convolution and each up stage
        begins with its transposed one, so a stage's Sequential carries the
        resampling with it.

        :param dpr: Drop path rate per block, consumed in stage order.
        :type dpr: List[float]
        :return: Nothing.
        :rtype: None
        """
        self.m_head = keras.layers.Conv2D(
            self.dim,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="head"
        )

        # Walks the flat dpr list one stage at a time.
        begin = 0

        self.m_down1 = self._create_stage_blocks(
            num_blocks=self.config[0],
            conv_dim=self.dim // 2,
            trans_dim=self.dim // 2,
            dpr=dpr[begin:begin + self.config[0]],
            input_res=self.input_resolution,
            stage_name="down1"
        )
        self.m_down1.append(
            keras.layers.Conv2D(
                2 * self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="down1_downsample"
            )
        )

        begin += self.config[0]

        self.m_down2 = self._create_stage_blocks(
            num_blocks=self.config[1],
            conv_dim=self.dim,
            trans_dim=self.dim,
            dpr=dpr[begin:begin + self.config[1]],
            input_res=self.input_resolution // 2,
            stage_name="down2"
        )
        self.m_down2.append(
            keras.layers.Conv2D(
                4 * self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="down2_downsample"
            )
        )

        begin += self.config[1]

        self.m_down3 = self._create_stage_blocks(
            num_blocks=self.config[2],
            conv_dim=2 * self.dim,
            trans_dim=2 * self.dim,
            dpr=dpr[begin:begin + self.config[2]],
            input_res=self.input_resolution // 4,
            stage_name="down3"
        )
        self.m_down3.append(
            keras.layers.Conv2D(
                8 * self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="down3_downsample"
            )
        )

        begin += self.config[2]

        # The bottleneck resamples nothing, so it holds blocks alone.
        self.m_body = self._create_stage_blocks(
            num_blocks=self.config[3],
            conv_dim=4 * self.dim,
            trans_dim=4 * self.dim,
            dpr=dpr[begin:begin + self.config[3]],
            input_res=self.input_resolution // 8,
            stage_name="body"
        )

        begin += self.config[3]

        self.m_up3 = [
            keras.layers.Conv2DTranspose(
                4 * self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="up3_upsample"
            )
        ]
        self.m_up3.extend(
            self._create_stage_blocks(
                num_blocks=self.config[4],
                conv_dim=2 * self.dim,
                trans_dim=2 * self.dim,
                dpr=dpr[begin:begin + self.config[4]],
                input_res=self.input_resolution // 4,
                stage_name="up3"
            )
        )

        begin += self.config[4]

        self.m_up2 = [
            keras.layers.Conv2DTranspose(
                2 * self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="up2_upsample"
            )
        ]
        self.m_up2.extend(
            self._create_stage_blocks(
                num_blocks=self.config[5],
                conv_dim=self.dim,
                trans_dim=self.dim,
                dpr=dpr[begin:begin + self.config[5]],
                input_res=self.input_resolution // 2,
                stage_name="up2"
            )
        )

        begin += self.config[5]

        self.m_up1 = [
            keras.layers.Conv2DTranspose(
                self.dim,
                kernel_size=2,
                strides=2,
                padding="valid",
                use_bias=False,
                name="up1_upsample"
            )
        ]
        self.m_up1.extend(
            self._create_stage_blocks(
                num_blocks=self.config[6],
                conv_dim=self.dim // 2,
                trans_dim=self.dim // 2,
                dpr=dpr[begin:begin + self.config[6]],
                input_res=self.input_resolution,
                stage_name="up1"
            )
        )

        self.m_tail = keras.layers.Conv2D(
            self.in_nc,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="tail"
        )

        self.m_down1 = keras.Sequential(self.m_down1, name="down1")
        self.m_down2 = keras.Sequential(self.m_down2, name="down2")
        self.m_down3 = keras.Sequential(self.m_down3, name="down3")
        self.m_body = keras.Sequential(self.m_body, name="body")
        self.m_up3 = keras.Sequential(self.m_up3, name="up3")
        self.m_up2 = keras.Sequential(self.m_up2, name="up2")
        self.m_up1 = keras.Sequential(self.m_up1, name="up1")

    def _create_stage_blocks(
            self,
            num_blocks: int,
            conv_dim: int,
            trans_dim: int,
            dpr: List[float],
            input_res: int,
            stage_name: str
    ) -> List[keras.layers.Layer]:
        """Create one stage's blocks, alternating plain and shifted windows.

        Even indices get window type ``W`` and odd ones ``SW``, so a stage with
        a single block never shifts.

        :param num_blocks: Number of blocks in the stage.
        :type num_blocks: int
        :param conv_dim: Channels routed to each block's convolutional branch.
        :type conv_dim: int
        :param trans_dim: Channels routed to each block's transformer branch.
        :type trans_dim: int
        :param dpr: Drop path rates for blocks in this stage.
        :type dpr: List[float]
        :param input_res: Resolution hint for this stage.
        :type input_res: int
        :param stage_name: Name of the stage for block naming.
        :type stage_name: str

        :return: List of layer blocks for the stage.
        :rtype: List[keras.layers.Layer]
        """
        blocks = []
        for i in range(num_blocks):
            block_type = "W" if i % 2 == 0 else "SW"
            blocks.append(
                SwinConvBlock(
                    conv_dim=conv_dim,
                    trans_dim=trans_dim,
                    head_dim=self.head_dim,
                    window_size=self.window_size,
                    drop_path_rate=dpr[i],
                    block_type=block_type,
                    input_resolution=input_res,
                    name=f"{stage_name}_block_{i}"
                )
            )
        return blocks

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from an explicit ``build`` call.

        Returns immediately if the model is already built. Real calls stay
        fully dynamic; only this trace uses a fixed resolution.

        :param input_shape: Shape of ``call``'s ``x``, ``(B, H, W, C)``.
        :type input_shape: Any
        :return: Nothing.
        :rtype: None
        """
        if self.built:
            return
        # DECISION plan-2026-08-23T091307-9a110062/D-422: coerce the spatial axes
        # to input_resolution, or `(-h) % 64` sees the placeholder's None.
        # See decisions.md.
        materialize_sublayers(
            self,
            concretize_axes(
                input_shape,
                {1: self.input_resolution, 2: self.input_resolution},
            ),
        )
        super().build(input_shape)

    def call(self,
             x: keras.KerasTensor,
             training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the forward pass, padding on entry and cropping on exit.

        :param x: Input tensor of shape (B, H, W, C).
        :type x: keras.KerasTensor
        :param training: Boolean indicating whether the model should behave in
            training mode or inference mode.
        :type training: Optional[bool]

        :return: Output tensor of shape (B, H, W, C), the same spatial size as
            the input.
        :rtype: keras.KerasTensor

        :raises ValueError: If a statically-known spatial extent is so small that
            the reflect pad up to the next multiple of 64 would be at least
            as large as the extent itself, which is every extent below 33. A
            dynamic extent is not checked here.
        """
        # DECISION plan-2026-07-31T210633-b63a35aa/D-004: check statically, since
        # MirrorPad needs every pad amount below its own dimension; a fallback
        # would invent pad content and a runtime check breaks the symbolic build.
        # See decisions.md.
        for axis_name, extent in (("height", x.shape[1]), ("width", x.shape[2])):
            if extent is None:
                continue
            pad_amount = (-extent) % 64
            if pad_amount >= extent:
                raise ValueError(
                    f"SCUNet reflect-pads {axis_name} up to the next multiple of "
                    f"64, but a reflect pad must be strictly smaller than the "
                    f"extent it pads: {axis_name}={extent} needs a pad of "
                    f"{pad_amount}, which is not less than {extent}. The "
                    f"smallest {axis_name} SCUNet accepts is 33. Pass a larger "
                    f"input, or pad it yourself before calling."
                )

        # Captured before the pad, so the crop below restores the input size.
        h, w = ops.shape(x)[1], ops.shape(x)[2]

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-045: compute the pad with
        # `ops`, never numpy, which raises on the symbolic None extent from
        # `SCUNet()(keras.Input((None, None, 3)))`. See decisions.md.
        padding_bottom = (-h) % 64
        padding_right = (-w) % 64

        paddings = [[0, 0], [0, padding_bottom], [0, padding_right], [0, 0]]
        x = ops.pad(x, paddings, mode="REFLECT")

        x1 = self.m_head(x)
        x2 = self.m_down1(x1, training=training)
        x3 = self.m_down2(x2, training=training)
        x4 = self.m_down3(x3, training=training)

        x = self.m_body(x4, training=training)

        # Each add lands on the next stage's input, so the skip is a residual
        # around the stage that produced it.
        x = self.m_up3(x + x4, training=training)
        x = self.m_up2(x + x3, training=training)
        x = self.m_up1(x + x2, training=training)
        x = self.m_tail(x + x1)

        # A no-op when nothing was padded.
        x = x[:, :h, :w, :]

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Dictionary containing every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "in_nc": self.in_nc,
            "config": self.config,
            "dim": self.dim,
            "head_dim": self.head_dim,
            "window_size": self.window_size,
            "input_resolution": self.input_resolution,
            "stochastic_depth_rate": self.stochastic_depth_rate,
        })
        return config

# ---------------------------------------------------------------------


def create_scunet(
        in_nc: int = 3,
        config: Optional[List[int]] = None,
        dim: int = 64,
        head_dim: int = 32,
        window_size: int = 8,
        stochastic_depth_rate: float = 0.0,
        input_resolution: int = 256,
        **kwargs: Any
) -> SCUNet:
    """Create an SCUNet image-restoration model.

    Zhang et al. publish a single network (``config=[4]*7``, ``dim=64``) and
    scale it by editing those two arguments, so there is no variant table to
    select from and this factory just forwards to the constructor with the
    paper's defaults.

    :param in_nc: Number of input (and output) channels. Defaults to 3.
    :type in_nc: int
    :param config: Per-stage block counts; exactly 7 entries (3 down + bottleneck
        + 3 up). ``None`` resolves to the paper's ``[4, 4, 4, 4, 4, 4, 4]``.
    :type config: Optional[List[int]]
    :param dim: Base feature dimension. Must be positive and even.
    :type dim: int
    :param head_dim: Attention head dimension. The narrowest transformer branch
        is ``dim // 2``.
    :type head_dim: int
    :param window_size: Swin attention window size. The pad target of 64 is
        fixed and assumes the default of 8.
    :type window_size: int
    :param stochastic_depth_rate: Maximum stochastic-depth drop rate, scheduled
        linearly across all blocks.
    :type stochastic_depth_rate: float
    :param input_resolution: Resolution hint forwarded to every
        ``SwinConvBlock``, where it changes no attention geometry, and used as
        the spatial size of ``build``'s trace.
    :type input_resolution: int
    :param kwargs: Additional arguments forwarded to the model constructor.

    :return: A configured SCUNet instance.
    :rtype: SCUNet

    :raises ValueError: If any argument is outside its valid range.

    Examples:
        >>> model = create_scunet(dim=32, head_dim=16, config=[1] * 7)
        >>> model(keras.random.normal((1, 64, 64, 3))).shape
        (1, 64, 64, 3)
    """
    return SCUNet(
        in_nc=in_nc,
        config=config,
        dim=dim,
        head_dim=head_dim,
        window_size=window_size,
        stochastic_depth_rate=stochastic_depth_rate,
        input_resolution=input_resolution,
        **kwargs
    )

# ---------------------------------------------------------------------
