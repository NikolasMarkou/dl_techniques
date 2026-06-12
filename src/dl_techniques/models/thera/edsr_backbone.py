"""THERA EDSR-baseline feature backbone as a Keras layer (no upsampling).

THERA (Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields)
uses an EDSR-baseline encoder as its low-resolution feature extractor. The
encoder is the *feature* part of EDSR only -- the head conv plus a stack of
residual blocks plus a body-tail conv with a long skip connection. THERA does
**not** use EDSR's pixel-shuffle upsampling tail (arbitrary-scale upsampling is
the job of the neural heat field downstream), so this backbone is purely
spatial-shape-preserving: ``(B, H, W, C_in) -> (B, H, W, num_feats)``.

The reference JAX/Flax encoder (THERA ``model/edsr.py``), instantiated as
``EDSR(None, num_blocks=16, num_feats=64)`` (= "edsr-baseline", no upsampling)::

    class ResidualBlock(nn.Module):
        channels; kernel_size; res_scale: float; activation
        def setup(self):
            self.body = Sequential([Conv(channels, k), activation, Conv(channels, k)])
        def __call__(self, x):
            return x + self.body(x)        # NOTE: res_scale stored but NOT applied

    class EDSR(nn.Module):
        scale_factor; channels=3; num_blocks=32; num_feats=256
        def setup(self):
            self.head = Sequential([Conv(num_feats, (3, 3))])
            res_blocks = [ResidualBlock(num_feats, (3, 3), res_scale=0.1, activation=relu)
                          for _ in range(num_blocks)]
            res_blocks.append(Conv(num_feats, (3, 3)))
            self.body = Sequential(res_blocks)
        def __call__(self, x, _=None):
            x = self.head(x)
            x = x + self.body(x)           # long skip; feature extractor, NO upsampling tail
            return x

Padding: Flax ``Conv`` defaults to ``padding='SAME'``; the Keras port uses
``keras.layers.Conv2D(..., padding='same')`` to match. Data layout is NHWC.

Reference:
    Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
    Neural Heat Fields"; Lim et al., "Enhanced Deep Residual Networks for Single
    Image Super-Resolution" (EDSR), CVPRW 2017.
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class EDSRResidualBlock(keras.layers.Layer):
    """An EDSR residual block: ``x + res_scale * conv(act(conv(x)))``.

    Two ``3x3`` convolutions with an activation between them form the residual
    branch; the block adds a (scaled) residual to its input. Both convolutions
    keep the channel count and spatial size fixed (``padding='same'``), so the
    block is shape-preserving.

    **Intent**: Provide a single THERA/EDSR residual unit -- a two-conv residual
    branch with a configurable scale -- that is fully serializable (the
    activation is stored as a resolved Keras activation object and round-trips
    via ``keras.activations.serialize``/``deserialize``) and shape-preserving.

    **Architecture**::

        x ->  Conv(k) -> act -> Conv(k) -> (* res_scale) ->  (+) -> out
        |                                                      ^
        +------------------------ skip ------------------------+

    res_scale note (THERA fidelity)
    -------------------------------
    THERA's reference residual block returns ``x + body(x)`` and *ignores* the
    ``res_scale`` it stores (a quirk inherited from jax-enhance). Textbook EDSR
    instead scales the residual branch by ``res_scale`` (typically ``0.1``) for
    training stability of very deep stacks. This port implements the textbook
    form ``x + res_scale * body(x)`` with **default ``res_scale=1.0``**, which is
    numerically identical to THERA's reference (1.0 * body == body) while still
    being a faithful superset: a caller may pass ``res_scale=0.1`` to recover
    canonical EDSR. See the ``D-005`` decision anchor in :meth:`call`.

    Args:
        num_feats: Channel count of both convolutions (and of the input). Must
            be a positive integer.
        kernel_size: Spatial size of both convolutions. Defaults to 3.
        res_scale: Scalar multiplier applied to the residual branch before the
            skip add. Defaults to 1.0 (THERA-faithful). Use 0.1 for textbook
            EDSR.
        activation: Activation applied between the two convolutions. Any value
            accepted by :func:`keras.activations.get`. Defaults to ``"relu"``.
        **kwargs: Forwarded to :class:`keras.layers.Layer`.

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
        # Resolve + store the activation as a Keras activation object (guide
        # §7/8: complex objects are serialized via keras.activations.serialize).
        # keras.activations.get accepts a string, a callable, or a serialized
        # dict and always returns the callable, which Conv/usage accepts.
        self.activation = keras.activations.get(activation)
        self._activation_fn = self.activation

        # Sublayers (created here, built explicitly in ``build`` -- four-strike
        # build-ordering discipline, LESSONS.md).
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
        # Explicitly build each sublayer with the correct propagated shape BEFORE
        # ``super().build`` so a ``.keras`` reload restores the conv weights
        # (Keras-3 four-strike build-ordering defect, LESSONS.md).
        self.conv1.build(input_shape)
        conv1_out_shape = self.conv1.compute_output_shape(input_shape)
        self.conv2.build(conv1_out_shape)
        super().build(input_shape)

    def call(self, x: Any, training: Optional[bool] = None) -> Any:
        # DECISION plan_2026-06-11_f662207d/D-005
        # res_scale is applied as `x + res_scale * body(x)`. Do NOT drop it to a
        # bare `x + body(x)`: THERA's reference block stores res_scale but never
        # applies it (a jax-enhance quirk). We default res_scale=1.0 so the math
        # is numerically identical to THERA, yet a caller can set res_scale=0.1
        # to obtain canonical/textbook EDSR. Dropping the multiply would silently
        # forbid the textbook variant. See decisions.md D-005.
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
            # keras.activations.deserialize accepts a serialized dict OR a bare
            # string name (back-compat with pre-iter-2 string configs), so this
            # is safe for both the already-string and already-serialized cases.
            config["activation"] = keras.activations.deserialize(config["activation"])
        return cls(**config)


@keras.saving.register_keras_serializable()
class EDSRBackbone(keras.layers.Layer):
    """EDSR-baseline feature backbone for THERA (no upsampling tail).

    **Intent**: Extract low-resolution spatial features for the THERA arbitrary-
    scale super-resolution pipeline -- a head conv, a deep residual-block stack,
    a body conv, and a long skip -- producing a ``num_feats``-channel feature map
    at the input resolution (the heat-field decoder downstream handles
    upsampling). Fully serializable (the residual blocks round-trip their
    activation via ``keras.activations.serialize``).

    **Architecture** (shape-preserving over spatial dims)::

        x -> head Conv -> [ res_block_1 -> ... -> res_block_N ] -> body Conv -> (+) -> features
                  |                                                              ^
                  +------------------------ long skip ---------------------------+

    Equivalently::

        h   = head_conv(x)                      # (B, H, W, num_feats)
        b   = res_block_1(h)
        ...
        b   = res_block_num_blocks(b)
        b   = body_conv(b)                      # (B, H, W, num_feats)
        out = h + b                             # long skip connection

    The input channel count is arbitrary (RGB = 3 is typical); the head
    convolution infers it. The output always has ``num_feats`` channels and the
    same spatial resolution as the input.

    Defaults reproduce THERA's "edsr-baseline" encoder (``num_feats=64``,
    ``num_blocks=16``). ``res_scale`` defaults to ``1.0`` (THERA-faithful, see
    :class:`EDSRResidualBlock`).

    Args:
        num_feats: Number of feature channels throughout the backbone (and the
            output channel count). Defaults to 64.
        num_blocks: Number of residual blocks in the body. Defaults to 16.
        kernel_size: Spatial kernel size for every convolution. Defaults to 3.
        res_scale: Residual-branch scale forwarded to each residual block.
            Defaults to 1.0. Use 0.1 for textbook EDSR behavior.
        activation: Activation used inside each residual block. Defaults to
            ``"relu"``.
        **kwargs: Forwarded to :class:`keras.layers.Layer`.

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
        # Resolve + store the activation as a Keras activation object (guide
        # §7/8). It is forwarded (as the callable) to each residual block, which
        # re-resolves it through keras.activations.get -- a no-op on a callable.
        self.activation = keras.activations.get(activation)

        # Sublayers (built explicitly in ``build`` -- four-strike discipline).
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
        # Build head on the raw input shape, then propagate the (num_feats)
        # head-output shape through every residual block and the body conv
        # BEFORE ``super().build`` so all conv kernels are restored on reload
        # (Keras-3 four-strike build-ordering defect, LESSONS.md).
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
        # *args/**kwargs absorb the THERA backbone-protocol 2nd positional arg
        # (the reference signature is ``__call__(self, x, _=None)``) gracefully.
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
            # Accepts a serialized dict OR a bare string name (back-compat with
            # pre-iter-2 string configs); deserialize is a no-op on a string.
            config["activation"] = keras.activations.deserialize(config["activation"])
        return cls(**config)
