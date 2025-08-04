"""
Convolutional Block Attention Module (CBAM) Implementation.

This module implements the CBAM attention mechanism as described in:
'CBAM: Convolutional Block Attention Module' (Woo et al., 2018)
https://arxiv.org/abs/1807.06521v2

The implementation consists of three main components:
    - ChannelAttention: Implements channel-wise attention mechanism
    - SpatialAttention: Implements spatial attention mechanism
    - CBAM: Combines both attention mechanisms into a single module

Example:
    >>> from dl_techniques.layers.cbam import CBAM
    >>> cbam = CBAM(channels=64, ratio=8)
    >>> refined_features = cbam(input_features)
"""

import keras
from typing import Optional, Union, Dict, Any, Tuple
from dl_techniques.utils.logger import logger


class ChannelAttention(keras.layers.Layer):
    """Channel attention module of CBAM.

    This module applies channel-wise attention by using both max-pooling
    and average-pooling features, followed by a shared MLP network.

    Args:
        channels: Number of input channels.
        ratio: Reduction ratio for the shared MLP.
        kernel_initializer: Initializer for the dense layer kernels.
        kernel_regularizer: Regularizer function for the dense layer kernels.
        use_bias: Whether to include bias in dense layers.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        channels: int,
        ratio: int = 8,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        # Sublayers will be initialized in build()
        self.shared_mlp = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights based on input shape.

        Args:
            input_shape: Shape tuple indicating the input shape of the layer.
        """
        self._build_input_shape = input_shape

        # Create shared MLP layers
        self.shared_mlp = keras.Sequential([
            keras.layers.Dense(
                self.channels // self.ratio,
                activation='relu',
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='channel_attention_dense_1'
            ),
            keras.layers.Dense(
                self.channels,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='channel_attention_dense_2'
            )
        ], name='shared_mlp')

        # Build the sequential model
        dummy_input_shape = (1, 1, self.channels)
        self.shared_mlp.build(dummy_input_shape)

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Apply channel attention to input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Channel attention map of shape (batch_size, 1, 1, channels).
        """
        # Apply global average pooling and global max pooling
        avg_pool = keras.ops.mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = keras.ops.max(inputs, axis=[1, 2], keepdims=True)

        # Pass through shared MLP
        avg_out = self.shared_mlp(avg_pool, training=training)
        max_out = self.shared_mlp(max_pool, training=training)

        # Combine and apply sigmoid activation
        return keras.ops.sigmoid(avg_out + max_out)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        input_shape_list = list(input_shape)
        return tuple([input_shape_list[0], 1, 1, input_shape_list[-1]])

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "ratio": self.ratio,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


class SpatialAttention(keras.layers.Layer):
    """Spatial attention module of CBAM.

    This module applies spatial attention using channel-wise pooling
    followed by a convolution operation.

    Args:
        kernel_size: Size of the convolution kernel.
        kernel_initializer: Initializer for the convolution kernels.
        kernel_regularizer: Regularizer function for the convolution kernels.
        use_bias: Whether to include bias in convolution layer.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        # Sublayers will be initialized in build()
        self.conv = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights based on input shape.

        Args:
            input_shape: Shape tuple indicating the input shape of the layer.
        """
        self._build_input_shape = input_shape

        # Create convolution layer
        self.conv = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name='spatial_attention_conv'
        )

        # Build the convolution layer with concatenated pooling features (2 channels)
        conv_input_shape = list(input_shape)
        conv_input_shape[-1] = 2  # avg_pool + max_pool
        self.conv.build(tuple(conv_input_shape))

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Apply spatial attention to input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Spatial attention map of shape (batch_size, height, width, 1).
        """
        # Apply channel-wise pooling
        avg_pool = keras.ops.mean(inputs, axis=-1, keepdims=True)
        max_pool = keras.ops.max(inputs, axis=-1, keepdims=True)

        # Concatenate pooled features
        concat = keras.ops.concatenate([avg_pool, max_pool], axis=-1)

        # Apply convolution and sigmoid activation
        return self.conv(concat, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [1])

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


class CBAM(keras.layers.Layer):
    """Convolutional Block Attention Module.

    This module combines channel and spatial attention mechanisms to
    refine feature maps in a sequential manner.

    Args:
        channels: Number of input channels.
        ratio: Reduction ratio for the channel attention module.
        kernel_size: Kernel size for the spatial attention module.
        channel_kernel_initializer: Initializer for channel attention kernels.
        spatial_kernel_initializer: Initializer for spatial attention kernels.
        channel_kernel_regularizer: Regularizer for channel attention kernels.
        spatial_kernel_regularizer: Regularizer for spatial attention kernels.
        channel_use_bias: Whether to use bias in channel attention layers.
        spatial_use_bias: Whether to use bias in spatial attention layers.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        channels: int,
        ratio: int = 8,
        kernel_size: int = 7,
        channel_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        spatial_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        channel_kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        spatial_kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        channel_use_bias: bool = False,
        spatial_use_bias: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_kernel_initializer = keras.initializers.get(channel_kernel_initializer)
        self.spatial_kernel_initializer = keras.initializers.get(spatial_kernel_initializer)
        self.channel_kernel_regularizer = keras.regularizers.get(channel_kernel_regularizer)
        self.spatial_kernel_regularizer = keras.regularizers.get(spatial_kernel_regularizer)
        self.channel_use_bias = channel_use_bias
        self.spatial_use_bias = spatial_use_bias

        # Sublayers will be initialized in build()
        self.channel_attention = None
        self.spatial_attention = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights based on input shape.

        Args:
            input_shape: Shape tuple indicating the input shape of the layer.
        """
        self._build_input_shape = input_shape

        logger.info(f"Building CBAM layer with input shape: {input_shape}")

        # Create channel attention module
        self.channel_attention = ChannelAttention(
            channels=self.channels,
            ratio=self.ratio,
            kernel_initializer=self.channel_kernel_initializer,
            kernel_regularizer=self.channel_kernel_regularizer,
            use_bias=self.channel_use_bias,
            name='channel_attention'
        )
        self.channel_attention.build(input_shape)

        # Create spatial attention module
        self.spatial_attention = SpatialAttention(
            kernel_size=self.kernel_size,
            kernel_initializer=self.spatial_kernel_initializer,
            kernel_regularizer=self.spatial_kernel_regularizer,
            use_bias=self.spatial_use_bias,
            name='spatial_attention'
        )
        self.spatial_attention.build(input_shape)

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Apply CBAM attention to input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Refined feature map of shape (batch_size, height, width, channels).
        """
        # Apply channel attention
        channel_attention_map = self.channel_attention(inputs, training=training)
        channel_refined = inputs * channel_attention_map

        # Apply spatial attention to channel-refined features
        spatial_attention_map = self.spatial_attention(channel_refined, training=training)
        refined = channel_refined * spatial_attention_map

        return refined

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "ratio": self.ratio,
            "kernel_size": self.kernel_size,
            "channel_kernel_initializer": keras.initializers.serialize(self.channel_kernel_initializer),
            "spatial_kernel_initializer": keras.initializers.serialize(self.spatial_kernel_initializer),
            "channel_kernel_regularizer": keras.regularizers.serialize(self.channel_kernel_regularizer),
            "spatial_kernel_regularizer": keras.regularizers.serialize(self.spatial_kernel_regularizer),
            "channel_use_bias": self.channel_use_bias,
            "spatial_use_bias": self.spatial_use_bias,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


# Helper function for creating CBAM models
def create_cbam_enhanced_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    channels: int = 64,
    cbam_ratio: int = 8,
    cbam_kernel_size: int = 7
) -> keras.Model:
    """Create a simple CNN model enhanced with CBAM attention.

    Args:
        input_shape: Shape of input images (height, width, channels).
        num_classes: Number of output classes.
        channels: Number of channels in the feature extraction layers.
        cbam_ratio: Reduction ratio for CBAM channel attention.
        cbam_kernel_size: Kernel size for CBAM spatial attention.

    Returns:
        Keras model with CBAM attention modules.
    """
    inputs = keras.layers.Input(shape=input_shape)

    # Feature extraction with CBAM
    x = keras.layers.Conv2D(channels, 3, padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = CBAM(channels=channels, ratio=cbam_ratio, kernel_size=cbam_kernel_size)(x)

    x = keras.layers.Conv2D(channels * 2, 3, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = CBAM(channels=channels * 2, ratio=cbam_ratio, kernel_size=cbam_kernel_size)(x)
    x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Conv2D(channels * 4, 3, padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = CBAM(channels=channels * 4, ratio=cbam_ratio, kernel_size=cbam_kernel_size)(x)
    x = keras.layers.MaxPooling2D(2)(x)

    # Classification head
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='cbam_enhanced_model')
    return model