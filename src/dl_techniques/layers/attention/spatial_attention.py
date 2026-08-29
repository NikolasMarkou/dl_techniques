"""
A spatial attention map for convolutional feature maps.

This module implements the spatial attention stage of the Convolutional Block
Attention Module (CBAM). It scores WHERE in a feature map the information sits.
Its sibling, channel attention, scores WHAT features matter.

Architecture:
    The layer first squeezes every channel into two 2D descriptors, then turns
    those descriptors into one attention map. Two steps:

    1.  **Channel information aggregation.** Two pooling ops run along the
        channel axis, each producing a 2D map:

        -   **Average pooling** summarizes the mean response at every spatial
            location. This carries global context.
        -   **Max pooling** reports the strongest single response at every
            spatial location. This survives when a signal is concentrated
            rather than spread.

    2.  **Spatial map generation.** The two 2D maps are concatenated on the
        channel axis into a `(H, W, 2)` descriptor. One convolution with a
        single filter and a large kernel (7x7 by default) runs over it. A large
        kernel is used because saliency is a neighbourhood property. The result
        goes through the gate activation, sigmoid by default, which produces
        the final map.

Foundational Mathematics:
    The spatial attention map `M_s` for an input feature map `F` is:

        M_s(F) = σ( f^k ([AvgPool(F); MaxPool(F)]) )

    -   `AvgPool(F)` and `MaxPool(F)` pool along the channel axis. Each turns a
        tensor of shape `(H, W, C)` into `(H, W, 1)`.
    -   `[;]` concatenates the two maps on the channel axis, giving `(H, W, 2)`.
    -   `f^k` is a 2D convolution with one filter of kernel size `k x k`, for
        example 7x7. It acts as a spatial feature detector.
    -   `σ` is the gate activation. The default sigmoid bounds the output to
        `[0, 1]`, which is what makes it usable as a multiplicative mask. A
        different activation changes that range.

References:
    - The foundational paper for this module:
      Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). "CBAM:
      Convolutional Block Attention Module". European Conference on
      Computer Vision (ECCV).
"""

# ---------------------------------------------------------------------

import keras
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.activations import resolve_activation_layer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.attention.spatial_attention")
class SpatialAttention(keras.layers.Layer):
    """
    Spatial attention from CBAM: score where the information sits.

    Pools the input over its channel axis with mean and max, concatenates the
    two 2D maps, and runs one convolution plus a gate activation over the
    result. The layer returns the attention MAP, not the gated input:
    ``M_s(F) = sigma(f^{k x k}([AvgPool(F); MaxPool(F)]))``. The caller
    multiplies it back in. :class:`CBAM` is the caller that does so.

    **Architecture Overview:**

    .. code-block:: text

                    inputs  [B, H, W, C]
            attention_mask is accepted but IGNORED here
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌───────────────────┐ ┌───────────────────┐
          │ mean over axis -1 │ │ max over axis -1  │
          │ keepdims=True     │ │ keepdims=True     │
          └─────────┬─────────┘ └─────────┬─────────┘
            [B,H,W,1]                     [B,H,W,1]
                    └─────────┬───────────┘
                              ▼
                    concatenate on axis -1
                              │  [B, H, W, 2]
                              ▼
          ┌───────────────────────────────────────┐
          │ conv: Conv2D(filters=1,               │
          │   kernel_size=k, padding='same')      │
          └──────────────────┬────────────────────┘
                             │  [B, H, W, 1]
                             ▼
          ┌───────────────────────────────────────┐
          │ gate_activation                       │
          │ (gate_activation_type; 'sigmoid' by   │
          │  default, which bounds to [0, 1])     │
          └──────────────────┬────────────────────┘
                             ▼
                    output  [B, H, W, 1]

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
       **no** ``channels`` argument. The two pooling ops fully reduce the
       channel axis before the convolution sees it, so one instance works for
       any ``C``. That is why :class:`CBAM` forwards ``channels`` to the
       channel branch only.
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

        # CREATE sub-layer in __init__ following modern Keras 3 pattern
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

        # This rank check mirrors ChannelAttention.build in channel_attention.py
        # and uses the same message shape, so both halves of CBAM fail the same
        # way. Without it a non-4D input reached self.conv.build below, which
        # forces [-1] = 2 onto a shape of the wrong length. That surfaced as the
        # Keras-internal message "Kernel shape must have the same length as
        # input, but received kernel of shape (k, k, 2, 1) and input of shape
        # (...)", which names neither this layer nor the real problem.
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

        # Always call parent build at the end
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
            package's masked attention layers, but **ignored**. There is no
            masking code on this path and nothing is suppressed. This is a
            vision layer that pools over channels and emits a per-location
            gate; it has no token-masking semantics. The parameter stays in
            the signature so the layer remains drop-in compatible with masked
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

# ---------------------------------------------------------------------
