"""
`DPTDecoder`, the decoder half of the Dense Prediction Transformer (DPT)
architecture, turning encoder features into a dense pixel-wise prediction
such as a depth map.

The "Transformer" in DPT names its encoder, not this decoder. This decoder is
a plain convolutional head: a sequence of `3x3 Conv2D` + `BatchNormalization`
+ activation blocks, one per entry in `dims`, each narrowing the channel
count. `upsample_factor` (a power of two, at most `2 ** len(dims)`) places a
`2x` bilinear `UpSampling2D` after each of the first `log2(upsample_factor)`
stages; the remaining stages keep the resolution. A final `3x3 Conv2D`
projects to `output_channels`.

The final layer's activation defaults to `linear`, not a bounded activation.
A pipeline whose target genuinely lives in `[0, 1]` should pass `sigmoid`;
multi-class segmentation should pass `softmax`; the reference Depth Anything
relative-depth head ends in a ReLU. `upsample_factor=1`, the default, assumes
the input features already carry the output resolution; a caller decoding
stride-`s` features passes `upsample_factor=s`.

`DepthAnything` (`model.py`) does not use `DPTDecoder`. Its decoder is
`DPTHead`, also defined here: the reference Depth Anything V1 head
(LiheYoung/Depth-Anything `depth_anything/dpt.py` `DPTHead`, with its
`blocks.py` `FeatureFusionBlock` and `ResidualConvUnit`), which reads four
encoder layers and fuses them coarse to fine. OmniPoint's dense heads use
`DPTDecoder`.
"""

import keras
import numpy as np
from typing import Dict, Tuple, Optional, Any, List, Sequence, Union
from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Variance epsilon for every `BatchNormalization` in this package.
#: Matches the torch reference's `nn.BatchNorm2d` default (1e-5), not
#: Keras' own default (1e-3, 100x larger).
REFERENCE_BN_EPSILON: float = 1e-5

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.depth_anything.components")
class DPTDecoder(keras.layers.Layer):
    """Convolutional decoder head for dense prediction (DPT-style).

    Architecture:

    .. code-block:: text

        Input [B, H, W, C]
              │
              ▼
        ┌──────────────────────────────┐
        │ per stage i in dims:         │
        │  Conv2D(dims[i], 3x3, same)  │
        │  → BatchNorm(eps=1e-5)       │
        │  → Activation                │
        │  → UpSampling2D(2x, bilinear)│  (only for the first
        └──────────────┬───────────────┘   log2(upsample_factor) stages)
                       ▼
        ┌─────────────────────────────┐
        │ Conv2D(output_channels, 3x3)│
        │ → output_activation         │  (linear by default)
        └──────────────┬──────────────┘
                       ▼
        Output [B, H*upsample_factor, W*upsample_factor, output_channels]

    :param dims: Channel dimension per decoder stage; its length sets the
        number of stages.
    :type dims: Optional[List[int]]
    :param output_channels: Number of channels in the final prediction.
    :type output_channels: int
    :param kernel_initializer: Initializer for every convolutional kernel.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every convolutional
        kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_bias: Whether the stage convolutions use a bias term. The
        final output convolution always uses one.
    :type use_bias: bool
    :param activation: Activation applied after each stage's batch norm.
    :type activation: Union[str, callable]
    :param output_activation: Activation applied to the final projection.
        Defaults to ``"linear"`` so the output stays unconstrained for
        affine-invariant and scale-shift depth losses; pass ``"sigmoid"``
        only when the target genuinely lives in ``[0, 1]``, or ``"softmax"``
        for multi-class segmentation.
    :type output_activation: Union[str, callable]
    :param upsample_factor: Total bilinear upsampling factor across every
        stage. Must be a power of 2, at most ``2 ** len(dims)``.
    :type upsample_factor: int
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``.

    Output shape:
        4D tensor ``(batch_size, height * upsample_factor,
        width * upsample_factor, output_channels)``.

    Example:
        .. code-block:: python

            decoder = DPTDecoder(dims=[256, 128, 64, 32], output_channels=1)
            x = keras.random.normal([2, 64, 64, 768])
            output = decoder(x)
            print(output.shape)  # (2, 64, 64, 1)
    """

    def __init__(
            self,
            dims: Optional[List[int]] = None,
            output_channels: int = 1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_bias: bool = False,
            activation: Union[str, callable] = "relu",
            output_activation: Union[str, callable] = "linear",
            upsample_factor: int = 1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration parameters
        self.dims = dims if dims is not None else [256, 128, 64, 32]
        self.output_channels = output_channels
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.activation = deserialize_activation(activation)
        self.output_activation = deserialize_activation(output_activation)
        self.upsample_factor = int(upsample_factor)

        # Validate upsample_factor: must be a power of 2 and <= 2**len(dims).
        if self.upsample_factor < 1 or (self.upsample_factor & (self.upsample_factor - 1)) != 0:
            raise ValueError(
                f"upsample_factor must be a positive power of 2, got {upsample_factor}"
            )
        # Number of 2x upsamples to insert.
        self._num_upsamples = 0
        uf = self.upsample_factor
        while uf > 1:
            uf //= 2
            self._num_upsamples += 1
        if self._num_upsamples > len(self.dims):
            raise ValueError(
                f"upsample_factor=2**{self._num_upsamples} requires at least "
                f"{self._num_upsamples} decoder stages, but len(dims)={len(self.dims)}"
            )

        # All sublayer dims are shape-independent (derived from self.dims /
        # self.output_channels / self._num_upsamples), so create them here in
        # __init__ for stable layer tracking across (de)serialization.
        self.conv_layers: List[keras.layers.Conv2D] = []
        self.batch_norm_layers: List[keras.layers.BatchNormalization] = []
        self.activation_layers: List[keras.layers.Layer] = []
        self.upsample_layers: List[Optional[keras.layers.Layer]] = []

        # Create convolutional layers for each dimension
        for i, dim in enumerate(self.dims):
            # Convolutional layer
            conv = keras.layers.Conv2D(
                filters=dim,
                kernel_size=3,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=self.use_bias,
                name=f'conv_{i}'
            )
            self.conv_layers.append(conv)

            # Batch normalization layer
            # DECISION plan-2026-08-17T183311-79c63e38/D-028: pass epsilon explicitly, never drop it.
            # Keras' 1e-3 default is 100x the torch reference this head follows. See REFERENCE_BN_EPSILON above.
            bn = keras.layers.BatchNormalization(
                epsilon=REFERENCE_BN_EPSILON, name=f'bn_{i}'
            )
            self.batch_norm_layers.append(bn)

            # Activation layer
            activation_layer = keras.layers.Activation(
                self.activation,
                name=f'activation_{i}'
            )
            self.activation_layers.append(activation_layer)

            # Upsample after the first _num_upsamples stages (one 2x per stage).
            if i < self._num_upsamples:
                up = keras.layers.UpSampling2D(
                    size=(2, 2), interpolation='bilinear', name=f'upsample_{i}'
                )
            else:
                up = None
            self.upsample_layers.append(up)

        # Final output layer
        self.output_conv = keras.layers.Conv2D(
            filters=self.output_channels,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            activation=self.output_activation,
            use_bias=True,  # Output layer typically uses bias
            name='output_conv'
        )

        self._build_input_shape: Optional[Tuple[int, ...]] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build every sublayer `call` runs, in `call`'s own shape order.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[int, ...]
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-124: sublayers build eagerly here, not lazily on first call.
        # `load_model` restores from the saved input_shape without a forward pass; lazy build left it 0-weight. See decisions.md.
        self._build_input_shape = input_shape
        shape = tuple(input_shape)
        for conv, bn, act, up in zip(
                self.conv_layers,
                self.batch_norm_layers,
                self.activation_layers,
                self.upsample_layers,
        ):
            conv.build(shape)
            shape = conv.compute_output_shape(shape)
            bn.build(shape)
            act.build(shape)
            if up is not None:
                shape = up.compute_output_shape(shape)
        self.output_conv.build(shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the decoder stages then the output projection.

        :param inputs: Input features, shape ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer runs in training or inference mode.
        :type training: Optional[bool]
        :return: Decoded output, shape ``(batch_size, height, width, output_channels)``.
        :rtype: keras.KerasTensor
        """
        x = inputs

        # Apply decoder layers sequentially
        for conv, bn, activation, up in zip(
                self.conv_layers,
                self.batch_norm_layers,
                self.activation_layers,
                self.upsample_layers,
        ):
            x = conv(x)
            x = bn(x, training=training)
            x = activation(x)
            if up is not None:
                x = up(x)

        # Apply final output layer
        x = self.output_conv(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[int, ...]
        :return: Output shape tuple.
        :rtype: Tuple[int, ...]
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)

        # Spatial dims are scaled by upsample_factor; channels become output_channels.
        h = input_shape_list[1]
        w = input_shape_list[2]
        if h is not None:
            h = h * self.upsample_factor
        if w is not None:
            w = w * self.upsample_factor
        output_shape_list = [input_shape_list[0], h, w, self.output_channels]

        # Return as tuple for consistency
        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dims": self.dims,
            "output_channels": self.output_channels,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "activation": serialize_activation(self.activation),
            "output_activation": serialize_activation(self.output_activation),
            "upsample_factor": self.upsample_factor,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        :return: Dictionary containing the build configuration.
        :rtype: Dict[str, Any]
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        :param config: Dictionary containing build configuration.
        :type config: Dict[str, Any]
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------


def _align_corners_taps(n_in: int, n_out: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two source indices and the weight of each output sample along one axis.

    Torch's ``F.interpolate(mode='bilinear', align_corners=True)`` maps output
    index ``i`` to source coordinate ``i * (n_in - 1) / (n_out - 1)`` (``0`` when
    ``n_out == 1``) and blends its two neighbours ``lo`` and ``hi = lo + 1``
    (clamped) as ``(1 - frac) * x[lo] + frac * x[hi]``.

    :param n_in: Source length along the axis.
    :type n_in: int
    :param n_out: Target length along the axis.
    :type n_out: int
    :return: ``(lo, hi, frac)``, int32, int32 and float64 arrays of length ``n_out``.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    scale = 0.0 if n_out == 1 else (n_in - 1) / (n_out - 1)
    src = np.arange(n_out, dtype=np.float64) * scale
    lo = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    hi = np.minimum(lo + 1, n_in - 1)
    return lo.astype(np.int32), hi.astype(np.int32), src - lo


def resize_bilinear_align_corners(
        x: keras.KerasTensor, height: int, width: int,
) -> keras.KerasTensor:
    """Bilinear resize with torch's ``align_corners=True`` convention.

    Interface contract: ``x`` is ``(batch, H, W, C)`` with static ``H`` and
    ``W``; returns ``(batch, height, width, C)`` in ``x``'s dtype, the
    separable form of ``F.interpolate(x, (height, width), mode='bilinear',
    align_corners=True)`` (corner pixels map to corner pixels). Equal sizes
    return ``x`` itself. Raises ``ValueError`` when ``H`` or ``W`` is not static.
    Keras' own bilinear resize (``UpSampling2D``, ``keras.ops.image.resize``)
    uses half-pixel centres, the ``align_corners=False`` convention, which
    samples different source positions.

    :param x: Feature map ``(batch, H, W, C)``.
    :type x: keras.KerasTensor
    :param height: Output height.
    :type height: int
    :param width: Output width.
    :type width: int
    :return: The resized feature map.
    :rtype: keras.KerasTensor
    :raises ValueError: If the spatial dimensions of ``x`` are not static.
    """
    in_h, in_w = x.shape[1], x.shape[2]
    if in_h is None or in_w is None:
        raise ValueError(
            f"resize_bilinear_align_corners needs static spatial dims, got {x.shape}")
    height, width = int(height), int(width)
    if (in_h, in_w) == (height, width):
        return x
    for axis, n_in, n_out in ((1, in_h, height), (2, in_w, width)):
        lo, hi, frac = _align_corners_taps(n_in, n_out)
        shape = [1, 1, 1, 1]
        shape[axis] = n_out
        weight = keras.ops.convert_to_tensor(frac.reshape(shape), dtype=x.dtype)
        a = keras.ops.take(x, lo, axis=axis)
        b = keras.ops.take(x, hi, axis=axis)
        x = a + (b - a) * weight
    return x


@register_dl_technique("dl_techniques.models.depth_anything.components")
class DPTResidualConvUnit(keras.layers.Layer):
    """``ResidualConvUnit`` of the reference DPT head, without batch norm.

    ``out = x + conv2(relu(conv1(relu(x))))``, both convolutions ``3x3``, stride
    1, padding 1, with bias (Depth-Anything ``depth_anything/blocks.py``, the
    ``bn=False`` path ``DPTHead(use_bn=False)`` builds).

    :param features: Channel count of the input and output.
    :type features: int
    :param kernel_initializer: Initializer for both kernels (cloned per conv).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for both kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    """

    def __init__(
            self,
            features: int,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.features = int(features)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.conv1, self.conv2 = (
            keras.layers.Conv2D(
                self.features, 3, padding="same", use_bias=True,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                kernel_regularizer=self.kernel_regularizer, name=name)
            for name in ("conv1", "conv2"))

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build both convolutions on the input shape (the unit keeps it).

        :param input_shape: ``(batch, H, W, features)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.conv1.build(input_shape)
        self.conv2.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """``inputs + conv2(relu(conv1(relu(inputs))))``.

        :param inputs: ``(batch, H, W, features)``.
        :type inputs: keras.KerasTensor
        :return: Same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        out = self.conv1(keras.ops.relu(inputs))
        out = self.conv2(keras.ops.relu(out))
        return out + inputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """The input shape: the unit is residual.

        :param input_shape: Input shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``input_shape``.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Constructor arguments.

        :return: The layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "features": self.features,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


@register_dl_technique("dl_techniques.models.depth_anything.components")
class DPTFeatureFusionBlock(keras.layers.Layer):
    """``FeatureFusionBlock`` of the reference DPT head, without batch norm.

    For ``[path, skip]``: ``out = res_conf_unit2(path + res_conf_unit1(skip))``;
    for ``[path]`` alone: ``out = res_conf_unit2(path)``. Then a bilinear resize
    (``align_corners=True``) to ``output_size``, or by 2 when it is ``None``, and
    a ``1x1`` convolution with bias (Depth-Anything ``depth_anything/blocks.py``,
    ``deconv=False, expand=False, align_corners=True``).

    The reference builds ``resConfUnit1`` in every block, also in
    ``refinenet4``, whose forward has no skip input and never calls it. A block
    built with ``has_skip=False`` creates no first unit: a layer no forward
    reads would be weights that never train.

    :param features: Channel count of every input and of the output.
    :type features: int
    :param has_skip: Whether the block takes a skip input (``[path, skip]``).
    :type has_skip: bool
    :param kernel_initializer: Initializer for every kernel (cloned per conv).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    """

    def __init__(
            self,
            features: int,
            has_skip: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.features = int(features)
        self.has_skip = bool(has_skip)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        def unit(name: str) -> DPTResidualConvUnit:
            return DPTResidualConvUnit(
                features=self.features,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                kernel_regularizer=self.kernel_regularizer, name=name)

        self.res_conf_unit1 = unit("res_conf_unit1") if self.has_skip else None
        self.res_conf_unit2 = unit("res_conf_unit2")
        self.out_conv = keras.layers.Conv2D(
            self.features, 1, use_bias=True,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer, name="out_conv")

    def _output_hw(self, shape: Sequence[Optional[int]], output_size: Optional[Tuple[int, int]]):
        if output_size is not None:
            return int(output_size[0]), int(output_size[1])
        return tuple(None if d is None else 2 * d for d in shape[1:3])

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build the units on the path shape and the ``1x1`` conv on the same width.

        :param input_shape: ``[path_shape]`` or ``[path_shape, skip_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        """
        path_shape = tuple(input_shape[0])
        if self.res_conf_unit1 is not None:
            self.res_conf_unit1.build(tuple(input_shape[1]))
        self.res_conf_unit2.build(path_shape)
        self.out_conv.build(path_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            output_size: Optional[Tuple[int, int]] = None,
    ) -> keras.KerasTensor:
        """Fuse, refine, resize, project.

        :param inputs: ``[path]``, or ``[path, skip]`` with equal shapes.
        :type inputs: List[keras.KerasTensor]
        :param output_size: Static ``(height, width)`` to resize to; ``None``
            doubles the input size.
        :type output_size: Optional[Tuple[int, int]]
        :return: ``(batch, height, width, features)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the number of inputs does not match ``has_skip``.
        """
        if len(inputs) != (2 if self.has_skip else 1):
            raise ValueError(
                f"{self.name}: has_skip={self.has_skip} takes "
                f"{2 if self.has_skip else 1} inputs, got {len(inputs)}")
        out = inputs[0]
        if self.res_conf_unit1 is not None:
            out = out + self.res_conf_unit1(inputs[1])
        out = self.res_conf_unit2(out)
        out = resize_bilinear_align_corners(out, *self._output_hw(out.shape, output_size))
        return self.out_conv(out)

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]],
            output_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[int], ...]:
        """``(batch, height, width, features)`` at ``output_size`` (or 2x).

        :param input_shape: ``[path_shape]`` or ``[path_shape, skip_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :param output_size: As in :meth:`call`.
        :type output_size: Optional[Tuple[int, int]]
        :return: The output shape.
        :rtype: Tuple[Optional[int], ...]
        """
        shape = tuple(input_shape[0])
        return (shape[0],) + tuple(self._output_hw(shape, output_size)) + (self.features,)

    def get_config(self) -> Dict[str, Any]:
        """Constructor arguments.

        :return: The layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "features": self.features,
            "has_skip": self.has_skip,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


@register_dl_technique("dl_techniques.models.depth_anything.components")
class DPTHead(keras.layers.Layer):
    """The Depth Anything V1 relative-depth head: a DPT neck over four encoder layers.

    A port of ``DPTHead`` in LiheYoung/Depth-Anything ``depth_anything/dpt.py``
    for one output map (``nclass == 1``) with ``use_bn=False`` and
    ``use_clstoken=False``, the configuration all three published checkpoints
    use (``config.json`` of ``LiheYoung/depth_anything_vit{s,b,l}14``).

    Architecture (``g`` = the encoder's patch grid, ``F`` = ``features``,
    ``C_i`` = ``out_channels[i]``):

    .. code-block:: text

        4 encoder layers, each [B, g, g, D]
          │ projects[i]: Conv 1x1, D -> C_i
          │ resize_layers: ConvTranspose 4x4/4 (4g), ConvTranspose 2x2/2 (2g),
          │                identity (g), Conv 3x3/2 pad 1 (g/2)
          │ layer{i}_rn: Conv 3x3, C_i -> F, no bias
          ▼
        refinenet4(l4)            -> resize to l3's size
        refinenet3(path4, l3)     -> resize to l2's size
        refinenet2(path3, l2)     -> resize to l1's size
        refinenet1(path2, l1)     -> 2x                          [B, 8g, 8g, F]
          │ output_conv1: Conv 3x3, F -> F/2
          │ bilinear (align_corners) to g * patch_size          [B, H, W, F/2]
          │ output_conv2: Conv 3x3 -> 32, ReLU, Conv 1x1 -> out, ReLU
          ▼
        [B, g * patch_size, g * patch_size, output_channels]  (non-negative)

    The reference's encoder patch is 14 (DINOv2); ``patch_size`` is this
    model's (16), so the head returns the input resolution. The reference's
    ``refinenet4.resConfUnit1`` is not built (see `DPTFeatureFusionBlock`).

    :param features: ``F``, the fusion width.
    :type features: int
    :param out_channels: ``C_1..C_4``, the four projection widths.
    :type out_channels: Sequence[int]
    :param patch_size: The encoder's patch size; the output is
        ``patch_size`` times the patch grid.
    :type patch_size: int
    :param output_channels: Channels of the predicted map (the reference: 1).
    :type output_channels: int
    :param kernel_initializer: Initializer for every kernel (cloned per conv).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        A list of four 4D tensors ``(batch, g_h, g_w, D)``, the encoder's last
        four layers in order (shallowest first), with static ``g_h``, ``g_w``.

    Output shape:
        4D tensor ``(batch, g_h * patch_size, g_w * patch_size, output_channels)``.
    """

    #: Width of ``output_conv2``'s hidden layer (``head_features_2`` in the reference).
    HIDDEN_FEATURES: int = 32

    def __init__(
            self,
            features: int = 64,
            out_channels: Sequence[int] = (48, 96, 192, 384),
            patch_size: int = 16,
            output_channels: int = 1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        out_channels = [int(c) for c in out_channels]
        if len(out_channels) != 4 or min(out_channels) < 1:
            raise ValueError(f"out_channels must be 4 positive ints, got {out_channels}")
        if int(features) < 2 or int(patch_size) < 1 or int(output_channels) < 1:
            raise ValueError(
                f"features must be >= 2, patch_size and output_channels >= 1; got "
                f"{features}, {patch_size}, {output_channels}")
        self.features = int(features)
        self.out_channels = out_channels
        self.patch_size = int(patch_size)
        self.output_channels = int(output_channels)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        def conv(filters, kernel, name, layer=keras.layers.Conv2D, **kw):
            return layer(filters, kernel, kernel_initializer=clone_initializer(self.kernel_initializer),
                         kernel_regularizer=self.kernel_regularizer, name=name, **kw)

        c = self.out_channels
        self.projects = [conv(c[i], 1, f"projects_{i}") for i in range(4)]
        # resize_layers[2] is the identity; [3] is torch's Conv2d(3, stride 2,
        # padding 1), whose symmetric pad Keras' 'same' does not reproduce for an
        # even input (it pads bottom/right only), hence the explicit pad.
        self.resize_layers_0 = conv(c[0], 4, "resize_layers_0", keras.layers.Conv2DTranspose, strides=4)
        self.resize_layers_1 = conv(c[1], 2, "resize_layers_1", keras.layers.Conv2DTranspose, strides=2)
        self.resize_layers_3_pad = keras.layers.ZeroPadding2D(1, name="resize_layers_3_pad")
        self.resize_layers_3 = conv(c[3], 3, "resize_layers_3", strides=2)
        self.layer_rn = [
            conv(self.features, 3, f"layer{i + 1}_rn", padding="same", use_bias=False)
            for i in range(4)]
        # DECISION plan-2026-10-14T062616-81938e81/D-141: each sublayer gets its own
        # clone_initializer(...), never one kwargs dict carrying the shared instance
        # (the shared-initializer census counts that dict as a role suspect).
        self.refinenets = [
            DPTFeatureFusionBlock(
                features=self.features, has_skip=i < 3,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                kernel_regularizer=self.kernel_regularizer, name=f"refinenet{i + 1}")
            for i in range(4)]
        self.output_conv1 = conv(self.features // 2, 3, "output_conv1", padding="same")
        # output_conv2 is nn.Sequential(Conv 3x3, ReLU, Conv 1x1, ReLU, Identity).
        self.output_conv2_0 = conv(self.HIDDEN_FEATURES, 3, "output_conv2_0", padding="same",
                                   activation="relu")
        self.output_conv2_2 = conv(self.output_channels, 1, "output_conv2_2", activation="relu")

    def _resize(self, i: int, x: keras.KerasTensor) -> keras.KerasTensor:
        if i == 0:
            return self.resize_layers_0(x)
        if i == 1:
            return self.resize_layers_1(x)
        if i == 2:
            return x
        return self.resize_layers_3(self.resize_layers_3_pad(x))

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build every sub-layer in `call`'s order on the shapes `call` produces.

        :param input_shape: Four shapes ``(batch, g_h, g_w, D)``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :raises ValueError: If not given four inputs.
        """
        if len(input_shape) != 4:
            raise ValueError(f"DPTHead takes 4 encoder layers, got {len(input_shape)}")
        levels = []
        for i, shape in enumerate(input_shape):
            shape = tuple(shape)
            self.projects[i].build(shape)
            shape = self.projects[i].compute_output_shape(shape)
            for layer in {0: [self.resize_layers_0], 1: [self.resize_layers_1], 2: [],
                          3: [self.resize_layers_3_pad, self.resize_layers_3]}[i]:
                layer.build(shape)
                shape = layer.compute_output_shape(shape)
            self.layer_rn[i].build(shape)
            levels.append(self.layer_rn[i].compute_output_shape(shape))
        path = levels[3]
        for i in (3, 2, 1, 0):
            inputs = [path] if i == 3 else [path, levels[i]]
            self.refinenets[i].build(inputs)
            size = None if i == 0 else levels[i - 1][1:3]
            path = self.refinenets[i].compute_output_shape(inputs, size)
        self.output_conv1.build(path)
        shape = self._output_shape(input_shape[0], self.features // 2)
        self.output_conv2_0.build(shape)
        self.output_conv2_2.build(self.output_conv2_0.compute_output_shape(shape))
        super().build(input_shape)

    def _output_shape(self, shape: Sequence[Optional[int]], channels: int) -> Tuple[Optional[int], ...]:
        h, w = (None if d is None else d * self.patch_size for d in shape[1:3])
        return (shape[0], h, w, channels)

    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """Reassemble the four layers, fuse them coarse to fine, predict.

        :param inputs: Four ``(batch, g_h, g_w, D)`` maps, shallowest first.
        :type inputs: List[keras.KerasTensor]
        :return: ``(batch, g_h * patch_size, g_w * patch_size, output_channels)``,
            non-negative.
        :rtype: keras.KerasTensor
        """
        levels = [
            self.layer_rn[i](self._resize(i, self.projects[i](x)))
            for i, x in enumerate(inputs)]
        path = self.refinenets[3]([levels[3]], output_size=levels[2].shape[1:3])
        path = self.refinenets[2]([path, levels[2]], output_size=levels[1].shape[1:3])
        path = self.refinenets[1]([path, levels[1]], output_size=levels[0].shape[1:3])
        path = self.refinenets[0]([path, levels[0]])
        out = self.output_conv1(path)
        _, h, w, _ = self._output_shape(inputs[0].shape, 0)
        out = resize_bilinear_align_corners(out, h, w)
        return self.output_conv2_2(self.output_conv2_0(out))

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """``(batch, g_h * patch_size, g_w * patch_size, output_channels)``.

        :param input_shape: Four shapes ``(batch, g_h, g_w, D)``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: The output shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return self._output_shape(input_shape[0], self.output_channels)

    def get_config(self) -> Dict[str, Any]:
        """Constructor arguments.

        :return: The layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "features": self.features,
            "out_channels": list(self.out_channels),
            "patch_size": self.patch_size,
            "output_channels": self.output_channels,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
