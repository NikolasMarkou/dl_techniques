"""Per-channel attention weights for convolutional feature maps, the :class:`ChannelAttention` layer.

This is the channel half of the Convolutional Block Attention Module
(CBAM): it scores which features matter, while its sibling spatial
attention scores where in the map the information sits. The layer pools
the input over its spatial axes with both global average and global max
pooling, each giving a length-`C` vector, then runs both vectors through
one shared bottleneck MLP (reduce to `C // ratio`, activate, expand back
to `C`). The two MLP outputs are summed before the gate, so one strong
max response can lift a channel whose mean is unremarkable:
`M_c(F) = sigma(MLP(AvgPool(F)) + MLP(MaxPool(F)))`.

The layer returns the per-channel weights, not the gated input — the
caller multiplies them back in. Unlike its sibling `SpatialAttention`,
this layer needs a `channels` argument, since the shared MLP's weight
shapes are tied to one channel count.

References:
    - Woo et al., 2018. CBAM: Convolutional Block Attention Module.
      (https://arxiv.org/abs/1807.06521)
"""

import keras
from typing import Optional, Union, Dict, Any, Tuple

from dl_techniques.initializers import clone_initializer
from dl_techniques.layers.activations import resolve_activation_layer
from dl_techniques.utils.keras_registration import register_dl_technique

@register_dl_technique("dl_techniques.layers.attention.channel_attention")
class ChannelAttention(keras.layers.Layer):
    """
    Channel attention from CBAM: score which channels matter.

    Pools the input over its spatial axes with mean and max, runs both
    descriptors through one shared bottleneck MLP, sums the results and gates
    them. The layer returns the per-channel weights, not the gated input:
    ``M_c(F) = sigma(W_1(act(W_0(F_avg))) + W_1(act(W_0(F_max))))``. The caller
    multiplies them back in. :class:`CBAM` is the caller that does so.

    Architecture:

    .. code-block:: text

        inputs [B, H, W, C]
                │
        ┌───────┴────────┐
        ▼                 ▼
        mean(axes=1,2)  max(axes=1,2)
        keepdims=True   keepdims=True
        [B,1,1,C]        [B,1,1,C]
        ▼                 ▼
        reshape [B,C]   reshape [B,C]
        └───────┬────────┘
                ▼
        ┌────────────────────────┐
        │ shared weights:         │
        │  dense1: Dense(C//ratio)│
        │  intermediate_activation│
        │    ('relu' by default)  │
        │  dense2: Dense(C)       │
        └───────────┬─────────────┘
                    │  two [B, C] results
                    ▼
            elementwise sum
                    │  [B, C]
                    ▼
        ┌────────────────────────┐
        │ gate_activation         │
        │ ('sigmoid' by default) │
        └───────────┬─────────────┘
                    ▼
            reshape [B, 1, 1, C]
                    ▼
            output [B, 1, 1, C]

    Both paths call the same three sub-layer instances — that sharing is
    the point, forcing one relationship model to explain both descriptors.

    :param channels: Number of input channels. Must be positive and
        divisible by ``ratio``.
    :type channels: int
    :param ratio: Reduction ratio for the shared MLP bottleneck dimension
        (``channels // ratio``). Must be positive and divide evenly into
        ``channels``. Defaults to 8.
    :type ratio: int
    :param kernel_initializer: Initializer for the dense layer kernels.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for the dense layer
        kernels. Defaults to ``None``.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param use_bias: Whether to include bias in dense layers.
        Defaults to ``False``.
    :type use_bias: bool
    :param intermediate_activation_type: Activation applied inside the shared
        MLP bottleneck, resolved through
        :func:`~dl_techniques.layers.activations.resolve_activation_layer`.
        Defaults to ``'relu'`` (the CBAM paper's choice).
    :type intermediate_activation_type: str
    :param intermediate_activation_args: Optional keyword arguments forwarded
        to the intermediate activation layer's constructor. Defaults to
        ``None``.
    :type intermediate_activation_args: Optional[Dict[str, Any]]
    :param gate_activation_type: Activation producing the final channel gate,
        resolved through
        :func:`~dl_techniques.layers.activations.resolve_activation_layer`.
        Defaults to ``'sigmoid'``. Sigmoid is what bounds the returned weights
        to ``[0, 1]``; another choice changes that range.
    :type gate_activation_type: str
    :param gate_activation_args: Optional keyword arguments forwarded to the
        gate activation layer's constructor. Defaults to ``None``.
    :type gate_activation_args: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar channels: The configured channel count.
    :vartype channels: int
    :ivar ratio: The configured reduction ratio.
    :vartype ratio: int
    :ivar use_bias: Whether the dense layers carry a bias.
    :vartype use_bias: bool
    :ivar dense1: The bottleneck reduction, ``Dense(channels // ratio)``.
    :vartype dense1: keras.layers.Dense
    :ivar intermediate_activation: The layer between the two dense layers.
    :vartype intermediate_activation: keras.layers.Layer
    :ivar dense2: The expansion back to ``channels``.
    :vartype dense2: keras.layers.Dense
    :ivar gate_activation: The layer producing the final gate.
    :vartype gate_activation: keras.layers.Layer

    :raises ValueError: If ``channels`` is not positive.
    :raises ValueError: If ``ratio`` is not positive or does not divide
        evenly into ``channels``.
    :raises ValueError: From ``build()``, if the input is not 4D, or if its
        trailing dimension does not equal ``channels``.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.

    Output shape:
        4D tensor with shape ``(batch_size, 1, 1, channels)``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.attention import ChannelAttention

        x = keras.random.normal((4, 32, 32, 64))
        weights = ChannelAttention(channels=64, ratio=8)(x)
        refined = x * weights

    .. note::

       Unlike its CBAM sibling :class:`SpatialAttention`, this layer needs a
       ``channels`` argument. The shared MLP's weights have shape
       ``(C, C // ratio)`` and ``(C // ratio, C)``, so one instance is tied to
       one ``C``. That is why :class:`CBAM` forwards ``channels`` here and not
       to the spatial branch.
    """

    def __init__(
            self,
            channels: int,
            ratio: int = 8,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_bias: bool = False,
            intermediate_activation_type: str = 'relu',
            intermediate_activation_args: Optional[Dict[str, Any]] = None,
            gate_activation_type: str = 'sigmoid',
            gate_activation_args: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create the four sub-layers.

        :param channels: Number of input channels. Must be positive and
            divisible by ``ratio``.
        :type channels: int
        :param ratio: Reduction ratio for the MLP bottleneck.
        :type ratio: int
        :param kernel_initializer: Initializer for the dense kernels.
        :type kernel_initializer: str or keras.initializers.Initializer
        :param kernel_regularizer: Optional regularizer for the dense kernels.
        :type kernel_regularizer: keras.regularizers.Regularizer or None
        :param use_bias: Whether the dense layers carry a bias.
        :type use_bias: bool
        :param intermediate_activation_type: Activation inside the bottleneck.
        :type intermediate_activation_type: str
        :param intermediate_activation_args: Optional kwargs for it.
        :type intermediate_activation_args: Optional[Dict[str, Any]]
        :param gate_activation_type: Activation producing the final gate.
        :type gate_activation_type: str
        :param gate_activation_args: Optional kwargs for the gate activation.
        :type gate_activation_args: Optional[Dict[str, Any]]
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any

        :raises ValueError: If ``channels`` or ``ratio`` is not positive, or if
            ``ratio`` does not divide ``channels``.
        """
        super().__init__(**kwargs)

        # `channels` stays the CNN channel-count name here, not `dim` (a
        # frozen naming carve-out, GUIDE.md section 2). The check below is
        # not common.validate_head_divisibility(): this is an MLP width, not
        # a head count.
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if channels % ratio != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by ratio ({ratio})"
            )

        # Store configuration
        self.channels = channels
        self.ratio = ratio
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.intermediate_activation_type = intermediate_activation_type
        self.intermediate_activation_args = intermediate_activation_args
        self.gate_activation_type = gate_activation_type
        self.gate_activation_args = gate_activation_args

        self.dense1 = keras.layers.Dense(
            units=channels // ratio,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='channel_attention_dense_1'
        )

        self.intermediate_activation = resolve_activation_layer(
            self.intermediate_activation_type,
            name='channel_attention_intermediate_activation',
            **(self.intermediate_activation_args or {}),
        )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-070: dense2 takes a clone
        # of the initializer, not the shared instance -- sharing made both kernels element-for-element equal. See decisions.md.
        self.dense2 = keras.layers.Dense(
            units=channels,
            use_bias=use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name='channel_attention_dense_2'
        )

        self.gate_activation = resolve_activation_layer(
            self.gate_activation_type,
            name='channel_attention_gate_activation',
            **(self.gate_activation_args or {}),
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and its sub-layers.

        The MLP never sees the input shape as given. It sees the shape after
        spatial pooling and flattening, which is ``(batch, channels)``, so that
        is the shape the two dense layers are built at. Building each sub-layer
        explicitly is what guarantees every weight variable exists before
        weight restoration during model loading.

        :param input_shape: Shape tuple of the input tensor. Expected to be
            ``(batch_size, height, width, channels)``.
        :type input_shape: tuple

        :raises ValueError: If ``input_shape`` is not rank 4, or if its
            trailing dimension does not equal ``channels``.
        """
        if self.built:
            return

        # This rank check mirrors SpatialAttention.build in spatial_attention.py
        # and uses the same message shape, so both halves of CBAM fail the same
        # way.
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        if input_shape[-1] != self.channels:
            raise ValueError(
                f"Expected input channels ({input_shape[-1]}) to match "
                f"layer channels ({self.channels})"
            )

        # After pooling and flattening the MLP sees (batch_size, channels).
        mlp_input_shape = (input_shape[0], self.channels)

        # Build the sub-layers in call order.
        self.dense1.build(mlp_input_shape)

        # (batch_size, channels // ratio)
        dense1_output_shape = self.dense1.compute_output_shape(mlp_input_shape)

        self.intermediate_activation.build(dense1_output_shape)
        self.dense2.build(dense1_output_shape)

        # The gate runs on the sum of the two dense2 outputs, shape (B, C).
        self.gate_activation.build((input_shape[0], self.channels))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute the per-channel attention weights for the input tensor.

        Note this layer takes no ``attention_mask`` argument, unlike its CBAM
        sibling :class:`SpatialAttention`, which accepts one and ignores it.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode
            or inference mode.
        :type training: bool or None
        :return: Channel attention weights of shape
            ``(batch_size, 1, 1, channels)``. Values lie in ``[0, 1]`` under the
            default ``'sigmoid'`` gate; another ``gate_activation_type`` changes
            that range.
        :rtype: keras.KerasTensor
        """
        # Pool over the spatial axes. Each op gives (B, 1, 1, C).
        avg_pool = keras.ops.mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = keras.ops.max(inputs, axis=[1, 2], keepdims=True)

        # Flatten to (B, C) so the dense layers can consume them.
        avg_pool_flat = keras.ops.reshape(avg_pool, (-1, self.channels))
        max_pool_flat = keras.ops.reshape(max_pool, (-1, self.channels))

        # Both paths call the same three sub-layers, so one MLP explains
        # both descriptors.
        avg_out = self.dense1(avg_pool_flat, training=training)
        avg_out = self.intermediate_activation(avg_out, training=training)
        avg_out = self.dense2(avg_out, training=training)

        max_out = self.dense1(max_pool_flat, training=training)
        max_out = self.intermediate_activation(max_out, training=training)
        max_out = self.dense2(max_out, training=training)

        # Sum first, gate second. One strong max response can therefore lift a
        # channel whose mean is unremarkable.
        attention_weights = self.gate_activation(
            avg_out + max_out, training=training
        )

        # Back to (B, 1, 1, C) so the caller can broadcast-multiply.
        attention_weights = keras.ops.reshape(
            attention_weights, (-1, 1, 1, self.channels)
        )

        return attention_weights

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        The spatial axes collapse to 1; the channel axis is preserved.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple ``(batch_size, 1, 1, channels)``.
        :rtype: tuple
        """
        return (input_shape[0], 1, 1, self.channels)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        Every constructor argument is included, so a reloaded layer rebuilds all
        four sub-layers from config.

        :return: Dictionary containing the layer configuration.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "ratio": self.ratio,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "intermediate_activation_type": self.intermediate_activation_type,
            "intermediate_activation_args": self.intermediate_activation_args,
            "gate_activation_type": self.gate_activation_type,
            "gate_activation_args": self.gate_activation_args,
        })
        return config
