"""
Keras Implementation of the FractalNet Convolutional Block.

This file defines the `ConvBlock` layer, which serves as the fundamental
repeated computational unit in the FractalNet architecture, as described in
'FractalNet: Ultra-Deep Neural Networks without Residuals' by Larsson et al. (2016).
Link: https://arxiv.org/abs/1605.07648

The primary export is the `ConvBlock` class, which encapsulates a fixed sequence
of operations: Conv2D → BatchNorm → ReLU → Dropout, repeated twice.

Implementation Details:
-----------------------

1.  **Composite Layer Structure:**
    The `ConvBlock` is not a primitive Keras layer but a composite `keras.layers.Layer`
    that acts as a container. It internally manages a sequence of standard Keras
    layers (`Conv2D`, `BatchNormalization`, `Activation`, `Dropout`). This abstraction
    simplifies the construction of larger models like FractalNet by creating a
    reusable, self-contained unit.

2.  **Conditional Component Creation:**
    The inclusion of `BatchNormalization` and `Dropout` layers is conditional,
    controlled by the `use_batch_norm` and `dropout_rate` arguments respectively.
    - If `use_batch_norm` is `False`, the `BatchNormalization` layers are not
      created, and the `use_bias` argument for the `Conv2D` layers is set to `True`.
    - If `dropout_rate` is `0`, the `Dropout` layers are skipped entirely, avoiding
      unnecessary overhead during both training and inference.

3.  **Asymmetric Strides for Downsampling:**
    A key architectural choice is how strides are handled. The `strides` parameter
    passed during initialization *only* applies to the first `Conv2D` layer (`conv1`).
    This allows the block to perform spatial downsampling at the beginning. The
    second `Conv2D` layer (`conv2`) is hardcoded with `strides=1` to preserve the
    feature map dimensions within the second half of the block.

4.  **Modern Keras 3 Pattern:**
    Following modern best practices, all sub-layers are created in `__init__()` and
    explicitly built in `build()` to ensure robust serialization and deserialization.
"""

import keras
from typing import Union, Tuple, Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger

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
        filters: Integer, number of output filters for both convolutions. Must be positive.
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
        ```python
        # Basic ConvBlock with 64 filters
        conv_block = ConvBlock(filters=64)
        x = keras.random.normal((2, 32, 32, 3))
        y = conv_block(x)
        print(y.shape)  # (2, 32, 32, 64)

        # ConvBlock with stride=2 for downsampling
        conv_block = ConvBlock(filters=128, strides=2)
        x = keras.random.normal((2, 32, 32, 64))
        y = conv_block(x)
        print(y.shape)  # (2, 16, 16, 128)

        # ConvBlock without batch normalization
        conv_block = ConvBlock(
            filters=64,
            use_batch_norm=False,
            dropout_rate=0.1
        )
        ```

    Raises:
        ValueError: If filters is not a positive integer.
        ValueError: If dropout_rate is not between 0 and 1.
        ValueError: If padding is not "valid" or "same".
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

        Creates all sub-layers and stores configuration. Following modern Keras 3
        best practices, sub-layers are instantiated here but remain unbuilt.

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

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
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

        # First batch normalization (conditionally created)
        if self.use_batch_norm:
            self.bn1 = keras.layers.BatchNormalization(name="bn1")
        else:
            self.bn1 = None

        # First activation
        self.act1 = keras.layers.Activation(self.activation, name="act1")

        # First dropout (conditionally created)
        if self.dropout_rate > 0:
            self.dropout1 = keras.layers.Dropout(self.dropout_rate, name="dropout1")
        else:
            self.dropout1 = None

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

        # Second batch normalization (conditionally created)
        if self.use_batch_norm:
            self.bn2 = keras.layers.BatchNormalization(name="bn2")
        else:
            self.bn2 = None

        # Second activation
        self.act2 = keras.layers.Activation(self.activation, name="act2")

        # Second dropout (conditionally created)
        if self.dropout_rate > 0:
            self.dropout2 = keras.layers.Dropout(self.dropout_rate, name="dropout2")
        else:
            self.dropout2 = None

        logger.debug(f"Initialized ConvBlock with filters={filters}, "
                     f"kernel_size={kernel_size}, strides={strides}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the ConvBlock and all its sub-layers.

        Following modern Keras 3 best practices, this method explicitly builds
        each sub-layer in computational order to ensure robust serialization.
        This is critical for layers with sub-layers to prevent weight loading errors.

        Args:
            input_shape: Shape tuple of the input tensor, including batch dimension.
        """
        # Build sub-layers in computational order
        self.conv1.build(input_shape)

        # Compute intermediate shape after first convolution
        conv1_output_shape = self.conv1.compute_output_shape(input_shape)

        if self.bn1 is not None:
            self.bn1.build(conv1_output_shape)

        self.act1.build(conv1_output_shape)

        if self.dropout1 is not None:
            self.dropout1.build(conv1_output_shape)

        # Shape doesn't change through batch norm, activation, or dropout
        intermediate_shape = conv1_output_shape

        # Build second convolution path
        self.conv2.build(intermediate_shape)

        # Compute output shape after second convolution
        conv2_output_shape = self.conv2.compute_output_shape(intermediate_shape)

        if self.bn2 is not None:
            self.bn2.build(conv2_output_shape)

        self.act2.build(conv2_output_shape)

        if self.dropout2 is not None:
            self.dropout2.build(conv2_output_shape)

        # Always call parent build at the end
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

        if self.bn1 is not None:
            x = self.bn1(x, training=training)

        x = self.act1(x)

        if self.dropout1 is not None:
            x = self.dropout1(x, training=training)

        # Second convolution path
        x = self.conv2(x)

        if self.bn2 is not None:
            x = self.bn2(x, training=training)

        x = self.act2(x)

        if self.dropout2 is not None:
            x = self.dropout2(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the ConvBlock.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple after applying both convolutions.
        """
        # First convolution with custom strides
        intermediate_shape = self.conv1.compute_output_shape(input_shape)

        # Second convolution always has stride=1, so shape may change based on kernel size and padding
        output_shape = self.conv2.compute_output_shape(intermediate_shape)

        return output_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns all configuration parameters needed to recreate this layer,
        following modern Keras 3 serialization best practices.

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

# ---------------------------------------------------------------------