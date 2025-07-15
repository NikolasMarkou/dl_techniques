"""
ConvBlock layer implementation for FractalNet architecture.

This module provides the fundamental building block for FractalNet consisting of:
Conv2D → BatchNorm → ReLU → Dropout → Conv2D → BatchNorm → ReLU → Dropout
"""

import keras
from typing import Union, Tuple, Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvBlock(keras.layers.Layer):
    """Base convolutional block for FractalNet.

    Implements the fundamental building block B(x) consisting of:
    Conv2D → BatchNorm → ReLU → Dropout → Conv2D → BatchNorm → ReLU → Dropout

    This layer creates two sequential convolution operations with optional batch
    normalization, activation, and dropout. The first convolution can have custom
    strides for downsampling, while the second always uses stride=1 to maintain
    feature map dimensions.

    Args:
        filters: Integer, number of output filters for both convolutions.
        kernel_size: Integer or tuple of 2 integers, size of the convolution kernels.
            Defaults to 3.
        strides: Integer or tuple of 2 integers, strides of the first convolution.
            Defaults to 1.
        padding: String, padding mode for convolutions. Either "valid" or "same".
            Defaults to "same".
        use_batch_norm: Boolean, whether to use batch normalization after each
            convolution. Defaults to True.
        dropout_rate: Float between 0 and 1, dropout rate for regularization.
            Applied after each activation. Defaults to 0.0.
        kernel_initializer: String name or initializer instance, weight initializer
            for convolution kernels. Defaults to "he_normal".
        kernel_regularizer: String name or regularizer instance, weight regularizer
            for convolution kernels. Defaults to None.
        activation: String name or callable, activation function applied after
            each batch normalization. Defaults to "relu".
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, new_height, new_width, filters)`
        where `new_height` and `new_width` depend on the strides parameter.

    Example:
        >>> # Basic ConvBlock with 64 filters
        >>> conv_block = ConvBlock(filters=64)
        >>> x = keras.random.normal((2, 32, 32, 3))
        >>> y = conv_block(x)
        >>> print(y.shape)
        (2, 32, 32, 64)

        >>> # ConvBlock with stride=2 for downsampling
        >>> conv_block = ConvBlock(filters=128, strides=2)
        >>> x = keras.random.normal((2, 32, 32, 64))
        >>> y = conv_block(x)
        >>> print(y.shape)
        (2, 16, 16, 128)
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            strides: Union[int, Tuple[int, int]] = 1,
            padding: str = "same",
            use_batch_norm: bool = True,
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activation: Union[str, callable] = "relu",
            **kwargs: Any
    ) -> None:
        """Initialize the ConvBlock layer.

        Validates inputs and stores configuration. Sublayers are created in build().

        Raises:
            ValueError: If filters is not a positive integer.
            ValueError: If dropout_rate is not between 0 and 1.
            ValueError: If padding is not "valid" or "same".
        """
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(filters, int) or filters <= 0:
            raise ValueError(f"filters must be a positive integer, got {filters}")

        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        if padding not in ["valid", "same"]:
            raise ValueError(f"padding must be 'valid' or 'same', got {padding}")

        # Store configuration
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.activation = activation

        # Initialize layer attributes - will be set in build()
        self.conv1 = None
        self.bn1 = None
        self.act1 = None
        self.dropout1 = None
        self.conv2 = None
        self.bn2 = None
        self.act2 = None
        self.dropout2 = None

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.debug(f"Initialized ConvBlock with filters={filters}, "
                     f"kernel_size={kernel_size}, strides={strides}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the ConvBlock layers.

        Creates all sublayers with proper naming and configuration.
        The first convolution uses the specified strides, while the second
        always uses stride=1 to maintain feature map dimensions.

        Args:
            input_shape: Shape tuple of the input tensor, including batch dimension.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # First convolution layer
        self.conv1 = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=not self.use_batch_norm,
            name="conv1"
        )

        # First batch normalization (if enabled)
        if self.use_batch_norm:
            self.bn1 = keras.layers.BatchNormalization(name="bn1")

        # First activation
        self.act1 = keras.layers.Activation(self.activation, name="act1")

        # First dropout (if enabled)
        if self.dropout_rate > 0:
            self.dropout1 = keras.layers.Dropout(self.dropout_rate, name="dropout1")

        # Second convolution layer (always stride=1)
        self.conv2 = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,  # Always stride=1 for second conv
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=not self.use_batch_norm,
            name="conv2"
        )

        # Second batch normalization (if enabled)
        if self.use_batch_norm:
            self.bn2 = keras.layers.BatchNormalization(name="bn2")

        # Second activation
        self.act2 = keras.layers.Activation(self.activation, name="act2")

        # Second dropout (if enabled)
        if self.dropout_rate > 0:
            self.dropout2 = keras.layers.Dropout(self.dropout_rate, name="dropout2")

        super().build(input_shape)
        logger.debug(f"Built ConvBlock with input_shape={input_shape}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the ConvBlock.

        Applies the complete sequence: Conv2D → BatchNorm → ReLU → Dropout →
        Conv2D → BatchNorm → ReLU → Dropout

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode (adding dropout, updating batch norm statistics)
                or inference mode.

        Returns:
            Output tensor of shape (batch_size, new_height, new_width, filters).
        """
        # First convolution path
        x = self.conv1(inputs)

        if self.use_batch_norm:
            x = self.bn1(x, training=training)

        x = self.act1(x)

        if self.dropout_rate > 0:
            x = self.dropout1(x, training=training)

        # Second convolution path
        x = self.conv2(x)

        if self.use_batch_norm:
            x = self.bn2(x, training=training)

        x = self.act2(x)

        if self.dropout_rate > 0:
            x = self.dropout2(x, training=training)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "activation": self.activation,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        Args:
            config: Build configuration dictionary containing input_shape.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
