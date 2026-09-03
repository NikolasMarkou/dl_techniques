"""THERA EDSR-baseline feature backbone as a Keras layer (no upsampling).

THERA uses an EDSR-baseline encoder as its low-resolution feature extractor:
a head convolution, a stack of residual blocks, and a body-tail convolution
with a long skip connection. THERA does not use EDSR's pixel-shuffle
upsampling tail, since arbitrary-scale upsampling is the job of the neural
heat field downstream, so this backbone only preserves spatial shape:
``(B, H, W, C_in) -> (B, H, W, num_feats)``.

THERA's reference residual block stores a ``res_scale`` but never applies
it. This port applies it (``x + res_scale * body(x)``) with a default of
``1.0``, numerically identical to THERA's reference, so a caller can pass
``res_scale=0.1`` to recover textbook EDSR. Padding uses Keras' ``'same'``
to match Flax's default ``'SAME'``. Data layout is NHWC.

Reference:
    Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
    Neural Heat Fields"; Lim et al., "Enhanced Deep Residual Networks for Single
    Image Super-Resolution" (EDSR), CVPRW 2017.
"""

import keras
from typing import Any, Dict, List, Optional, Tuple
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.thera.edsr_backbone")
class EDSRResidualBlock(keras.layers.Layer):
    """An EDSR residual block: ``x + res_scale * conv(act(conv(x)))``.

    Two 3x3 convolutions with an activation between them form the residual
    branch, which is added back to the input. Both convolutions preserve
    channel count and spatial size.

    Architecture:

    .. code-block:: text

        x [B, H, W, num_feats]
          |----------------------------+
          v                            |
        +----------------+             |
        | conv1 -> act   |             |
        +----------------+             |
          |                            |
          v                            |
        +----------------+             |
        | conv2          |             |
        +----------------+             |
          |                            |
          v (* res_scale)              |
         (+) <-------------------------+
          |
          v
        out [B, H, W, num_feats]

    :param num_feats: Channel count of both convolutions and of the input.
    :type num_feats: int
    :param kernel_size: Spatial size of both convolutions.
    :type kernel_size: int
    :param res_scale: Scalar multiplier applied to the residual branch
        before the skip add. The default 1.0 matches THERA's reference,
        which stores but never applies this scale; pass 0.1 for textbook EDSR.
    :type res_scale: float
    :param activation: Activation applied between the two convolutions, any
        value accepted by :func:`keras.activations.get`.
    :type activation: str
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    Input shape:
        ``(B, H, W, num_feats)``.

    Output shape:
        ``(B, H, W, num_feats)`` (identical to input).

    Example:
        >>> blk = EDSRResidualBlock(num_feats=64, res_scale=0.1)
        >>> y = blk(keras.random.normal((2, 24, 24, 64)))
        >>> y.shape
        (2, 24, 24, 64)
    """

    def __init__(
        self,
        num_feats: int,
        kernel_size: int = 3,
        res_scale: float = 1.0,
        activation: str = "relu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_feats <= 0:
            raise ValueError(f"num_feats must be positive, got {num_feats}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.num_feats = int(num_feats)
        self.kernel_size = int(kernel_size)
        self.res_scale = float(res_scale)
        # Stored as a resolved Keras activation object so it round-trips
        # through keras.activations.serialize/deserialize.
        self.activation = keras.activations.get(activation)
        self._activation_fn = self.activation

        self.conv1 = keras.layers.Conv2D(
            filters=self.num_feats,
            kernel_size=self.kernel_size,
            padding="same",
            name="conv1",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=self.num_feats,
            kernel_size=self.kernel_size,
            padding="same",
            name="conv2",
        )

    def build(self, input_shape: Any) -> None:
        # Build each sublayer explicitly with its propagated shape before
        # super().build, so a .keras reload restores their weights.
        self.conv1.build(input_shape)
        conv1_out_shape = self.conv1.compute_output_shape(input_shape)
        self.conv2.build(conv1_out_shape)
        super().build(input_shape)

    def call(self, x: Any, training: Optional[bool] = None) -> Any:
        # DECISION plan_2026-06-11_f662207d/D-005: apply res_scale to the residual
        # branch; THERA's reference stores it but never applies it. See decisions.md.
        residual = self.conv1(x, training=training)
        residual = self._activation_fn(residual)
        residual = self.conv2(residual, training=training)
        return x + self.res_scale * residual

    def compute_output_shape(self, input_shape: Any) -> Any:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_feats": self.num_feats,
                "kernel_size": self.kernel_size,
                "res_scale": self.res_scale,
                "activation": keras.activations.serialize(self.activation),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EDSRResidualBlock":
        config = dict(config)
        if "activation" in config:
            # Accepts a serialized dict or a bare string name; deserialize is
            # a no-op on an already-string value.
            config["activation"] = keras.activations.deserialize(config["activation"])
        return cls(**config)


@register_dl_technique("dl_techniques.models.thera.edsr_backbone")
class EDSRBackbone(keras.layers.Layer):
    """EDSR-baseline feature backbone for THERA (no upsampling tail).

    Extracts low-resolution spatial features for THERA's arbitrary-scale
    super-resolution pipeline: a head convolution, a deep residual-block
    stack, a body convolution, and a long skip, producing a
    ``num_feats``-channel feature map at the input resolution. The heat-field
    decoder downstream handles upsampling.

    Architecture:

    .. code-block:: text

        x [B, H, W, C_in]
          |
          v
        +----------------+
        | head_conv      |  -> [B, H, W, num_feats]
        +----------------+
          |------------------------------+
          v                              |
        +----------------------------+   |
        | res_block_1 ... res_block_N|   |
        +----------------------------+   |
          |                              |
          v                              |
        +----------------+               |
        | body_conv      |               |
        +----------------+               |
          |                              |
         (+) <---------------------------+   long skip
          |
          v
        features [B, H, W, num_feats]

    The input channel count is arbitrary (RGB, 3, is typical); the head
    convolution infers it. Defaults reproduce THERA's "edsr-baseline"
    encoder (``num_feats=64``, ``num_blocks=16``); ``res_scale`` defaults
    to 1.0, matching THERA's reference (see :class:`EDSRResidualBlock`).

    :param num_feats: Number of feature channels throughout the backbone,
        and the output channel count.
    :type num_feats: int
    :param num_blocks: Number of residual blocks in the body.
    :type num_blocks: int
    :param kernel_size: Spatial kernel size for every convolution.
    :type kernel_size: int
    :param res_scale: Residual-branch scale forwarded to each residual
        block. Use 0.1 for textbook EDSR behavior.
    :type res_scale: float
    :param activation: Activation used inside each residual block.
    :type activation: str
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    Input shape:
        ``(B, H, W, C_in)`` with arbitrary ``C_in`` (commonly 3).

    Output shape:
        ``(B, H, W, num_feats)``.

    Example:
        >>> backbone = EDSRBackbone(num_feats=64, num_blocks=16)
        >>> feats = backbone(keras.random.normal((2, 24, 24, 3)))
        >>> feats.shape
        (2, 24, 24, 64)
    """

    def __init__(
        self,
        num_feats: int = 64,
        num_blocks: int = 16,
        kernel_size: int = 3,
        res_scale: float = 1.0,
        activation: str = "relu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_feats <= 0:
            raise ValueError(f"num_feats must be positive, got {num_feats}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.num_feats = int(num_feats)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        self.res_scale = float(res_scale)
        # Stored as a resolved Keras activation object, forwarded to each
        # residual block, which re-resolves it (a no-op on a callable).
        self.activation = keras.activations.get(activation)

        self.head_conv = keras.layers.Conv2D(
            filters=self.num_feats,
            kernel_size=self.kernel_size,
            padding="same",
            name="head_conv",
        )
        self.res_blocks: List[EDSRResidualBlock] = [
            EDSRResidualBlock(
                num_feats=self.num_feats,
                kernel_size=self.kernel_size,
                res_scale=self.res_scale,
                activation=self.activation,
                name=f"res_block_{i}",
            )
            for i in range(self.num_blocks)
        ]
        self.body_conv = keras.layers.Conv2D(
            filters=self.num_feats,
            kernel_size=self.kernel_size,
            padding="same",
            name="body_conv",
        )

    def build(self, input_shape: Any) -> None:
        # Build head on the raw input shape, then propagate its output shape
        # through every residual block and the body conv before super().build,
        # so every conv kernel is restored on reload.
        self.head_conv.build(input_shape)
        feat_shape = self.head_conv.compute_output_shape(input_shape)
        for block in self.res_blocks:
            block.build(feat_shape)
            feat_shape = block.compute_output_shape(feat_shape)
        self.body_conv.build(feat_shape)
        super().build(input_shape)

    def call(
        self,
        x: Any,
        training: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # *args/**kwargs absorb THERA's backbone-protocol second positional
        # argument (the reference signature is __call__(self, x, _=None)).
        h = self.head_conv(x, training=training)
        b = h
        for block in self.res_blocks:
            b = block(b, training=training)
        b = self.body_conv(b, training=training)
        return h + b

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        input_shape = tuple(input_shape)
        return input_shape[:-1] + (self.num_feats,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_feats": self.num_feats,
                "num_blocks": self.num_blocks,
                "kernel_size": self.kernel_size,
                "res_scale": self.res_scale,
                "activation": keras.activations.serialize(self.activation),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EDSRBackbone":
        config = dict(config)
        if "activation" in config:
            # Accepts a serialized dict or a bare string name; deserialize is
            # a no-op on an already-string value.
            config["activation"] = keras.activations.deserialize(config["activation"])
        return cls(**config)
