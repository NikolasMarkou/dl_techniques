"""Spatial attention map for convolutional feature maps, the :class:`SpatialAttention` layer.

This is the spatial half of the Convolutional Block Attention Module
(CBAM): it scores where in a feature map the information sits, while its
sibling channel attention scores which features matter. The layer pools
the channel axis with both mean and max, concatenates the two resulting
`(H, W, 1)` maps into `(H, W, 2)`, and runs one large-kernel convolution
(7x7 by default) over that descriptor, since saliency is a neighbourhood
property a 1x1 convolution cannot see. A gate activation, sigmoid by
default, turns the convolution output into the final `(H, W, 1)` map:
`M_s(F) = sigma(f^k([AvgPool(F); MaxPool(F)]))`.

The layer returns the attention map itself, not the gated input — the
caller multiplies it back in. It takes no `channels` argument: pooling
fully reduces the channel axis before the convolution sees it, so one
instance works for any input width.

References:
    - Woo et al., 2018. CBAM: Convolutional Block Attention Module.
      (https://arxiv.org/abs/1807.06521)
"""

import keras
from typing import Optional, Union, Dict, Any, Tuple

from dl_techniques.layers.activations import resolve_activation_layer
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.attention.spatial_attention")
class SpatialAttention(keras.layers.Layer):
    """
    Spatial attention from CBAM: score where the information sits.

    Pools the input over its channel axis with mean and max, concatenates
    the two 2D maps, and runs one convolution plus a gate activation over
    the result. The layer returns the attention map, not the gated input:
    ``M_s(F) = sigma(f^{k x k}([AvgPool(F); MaxPool(F)]))``. The caller
    multiplies it back in; :class:`CBAM` is the caller that does so.

    Architecture:

    .. code-block:: text

        inputs [B, H, W, C]  (attention_mask accepted, ignored)
                    │
            ┌───────┴───────┐
            ▼                ▼
        mean(axis=-1)   max(axis=-1)
        keepdims=True   keepdims=True
        [B,H,W,1]        [B,H,W,1]
            └───────┬───────┘
                    ▼
            concatenate(axis=-1)
                    │  [B, H, W, 2]
                    ▼
        ┌────────────────────────┐
        │ Conv2D(filters=1,      │
        │   kernel_size=k)       │
        └───────────┬────────────┘
                    │  [B, H, W, 1]
                    ▼
        ┌────────────────────────┐
        │ gate_activation        │
        │ ('sigmoid' by default) │
        └───────────┬────────────┘
                    ▼
            output [B, H, W, 1]

    :param kernel_size: Size of the convolution kernel. Must be odd and
        positive. Defaults to 7 following the original CBAM paper.
    :type kernel_size: int
    :param kernel_initializer: Initializer for the convolution kernel
        weights. Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for the convolution
        kernel weights. Defaults to ``None``.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param use_bias: Whether to include bias in the convolution layer.
        Defaults to ``True``.
    :type use_bias: bool
    :param gate_activation_type: Activation producing the final spatial gate,
        resolved through
        :func:`~dl_techniques.layers.activations.resolve_activation_layer`.
        Defaults to ``'sigmoid'``. Sigmoid is what bounds the returned map to
        ``[0, 1]``; another choice changes that range.
    :type gate_activation_type: str
    :param gate_activation_args: Optional keyword arguments forwarded to the
        gate activation layer's constructor. Defaults to ``None``.
    :type gate_activation_args: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar kernel_size: The configured convolution kernel size.
    :vartype kernel_size: int
    :ivar use_bias: Whether the convolution carries a bias.
    :vartype use_bias: bool
    :ivar gate_activation_type: The configured gate activation name.
    :vartype gate_activation_type: str
    :ivar conv: The single-filter convolution over the 2-channel descriptor.
    :vartype conv: keras.layers.Conv2D
    :ivar gate_activation: The layer producing the final gate.
    :vartype gate_activation: keras.layers.Layer

    :raises ValueError: If ``kernel_size`` is not positive or not odd.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.

    Output shape:
        4D tensor with shape ``(batch_size, height, width, 1)``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.attention import SpatialAttention

        x = keras.random.normal((4, 32, 32, 64))
        gate = SpatialAttention(kernel_size=7)(x)
        refined = x * gate

    .. note::

       Unlike its CBAM sibling :class:`ChannelAttention`, this layer takes
       no ``channels`` argument. The two pooling ops fully reduce the
       channel axis before the convolution sees it, so one instance works
       for any ``C``, which is why :class:`CBAM` forwards ``channels`` to
       the channel branch only.
    """

    def __init__(
            self,
            kernel_size: int = 7,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_bias: bool = True,
            gate_activation_type: str = 'sigmoid',
            gate_activation_args: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Validate the kernel size and create the two sub-layers.

        :param kernel_size: Convolution kernel size. Must be odd and positive.
        :type kernel_size: int
        :param kernel_initializer: Initializer for the convolution kernel.
        :type kernel_initializer: str or keras.initializers.Initializer
        :param kernel_regularizer: Optional regularizer for the kernel.
        :type kernel_regularizer: keras.regularizers.Regularizer or None
        :param use_bias: Whether the convolution carries a bias.
        :type use_bias: bool
        :param gate_activation_type: Activation producing the final gate.
        :type gate_activation_type: str
        :param gate_activation_args: Optional kwargs for the gate activation.
        :type gate_activation_args: Optional[Dict[str, Any]]
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any

        :raises ValueError: If ``kernel_size`` is not positive or not odd.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for 'same' padding, got {kernel_size}")

        # Store configuration
        self.kernel_size = kernel_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.gate_activation_type = gate_activation_type
        self.gate_activation_args = gate_activation_args

        self.conv = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name='spatial_attention_conv'
        )

        self.gate_activation = resolve_activation_layer(
            self.gate_activation_type,
            name='spatial_attention_gate_activation',
            **(self.gate_activation_args or {}),
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and its sub-layers.

        The convolution never sees the input shape as given. It sees the shape
        after channel pooling and concatenation, which has 2 channels, so that
        is the shape it is built at. The gate is built at 1 channel.

        :param input_shape: Shape tuple of the input tensor. Expected to be
            ``(batch_size, height, width, channels)``.
        :type input_shape: tuple

        :raises ValueError: If ``input_shape`` is not rank 4.
        """
        if self.built:
            return

        # Without this check a non-4D input reaches self.conv.build below with
        # a confusing Keras-internal kernel-shape error instead of this one.
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        # Build the convolution at the post-concatenation shape: the avg_pool
        # and max_pool maps give it exactly 2 channels.
        conv_input_shape = list(input_shape)
        conv_input_shape[-1] = 2
        self.conv.build(tuple(conv_input_shape))

        # Gate activation operates on the conv output (B, H, W, 1)
        gate_input_shape = list(input_shape)
        gate_input_shape[-1] = 1
        self.gate_activation.build(tuple(gate_input_shape))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute the spatial attention map for the input tensor.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Accepted for signature compatibility with the
            package's masked attention layers, but ignored. This vision
            layer pools over channels and emits a per-location gate, so it
            has no token-masking semantics; the parameter stays in the
            signature so the layer remains drop-in compatible with masked
            attention layers and with existing serialized configs.
        :type attention_mask: keras.KerasTensor or None
        :param training: Whether the layer should behave in training mode
            or inference mode.
        :type training: bool or None
        :return: Spatial attention map of shape
            ``(batch_size, height, width, 1)``. Values lie in ``[0, 1]`` under
            the default ``'sigmoid'`` gate; another ``gate_activation_type``
            changes that range.
        :rtype: keras.KerasTensor
        """
        # Pool over the channel axis. Each op gives (B, H, W, 1).
        avg_pool = keras.ops.mean(inputs, axis=-1, keepdims=True)
        max_pool = keras.ops.max(inputs, axis=-1, keepdims=True)

        # Concatenate the two descriptors into (B, H, W, 2).
        concat = keras.ops.concatenate([avg_pool, max_pool], axis=-1)

        # Convolve to (B, H, W, 1), then gate.
        attention_logits = self.conv(concat, training=training)
        attention_map = self.gate_activation(attention_logits, training=training)

        return attention_map

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Spatial dimensions are preserved; the channel axis becomes 1.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple ``(batch_size, height, width, 1)``.
        :rtype: tuple
        """
        # One attention channel out, spatial dimensions unchanged.
        output_shape = list(input_shape)
        output_shape[-1] = 1
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        Every constructor argument is included, so a reloaded layer rebuilds
        both sub-layers from config.

        :return: Dictionary containing the layer configuration.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "gate_activation_type": self.gate_activation_type,
            "gate_activation_args": self.gate_activation_args,
        })
        return config
