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
`DepthAnything` needs that: affine-invariant and scale-shift depth losses
require an unconstrained output so the network can pick its own scale (see
`model.py`'s module docstring). A pipeline whose target genuinely lives in
`[0, 1]` should pass `sigmoid`; multi-class segmentation should pass
`softmax`. `upsample_factor=1`, the default, assumes the input features
already carry the output resolution; `model.py` instead passes the encoder
stride (16), so an `H/16 x W/16` feature map is restored to full resolution.
"""

import keras
from typing import Dict, Tuple, Optional, Any, List, Union
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
        ┌─────────────────────────────┐
        │ per stage i in dims:        │
        │  Conv2D(dims[i], 3x3, same) │
        │  → BatchNorm(eps=1e-5)      │
        │  → Activation               │
        │  → UpSampling2D(2x, bilinear)│  (only for the first
        └──────────────┬──────────────┘   log2(upsample_factor) stages)
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
