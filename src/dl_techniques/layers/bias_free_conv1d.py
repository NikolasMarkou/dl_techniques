"""
Bias-Free 1D Convolutional Layer

A building block for bias-free CNNs that removes all additive constants
to enable better generalization across noise levels in time series denoising tasks.

Based on "Robust and Interpretable Blind Image Denoising via Bias-Free
Convolutional Neural Networks" (Mohan et al., ICLR 2020), adapted for 1D signals.
"""

import keras
from keras import layers
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BiasFreeConv1D(keras.layers.Layer):
    """
    Bias-free 1D convolutional layer with batch normalization and activation.

    This layer implements a convolution without bias, followed by bias-free
    batch normalization (center=False) and activation. This ensures no
    additive constants are introduced at any stage, which is crucial for
    achieving scaling invariance and better generalization across noise levels
    in time series data.

    The key modifications from standard Conv1D layers:
    - Conv1D uses use_bias=False
    - BatchNormalization uses center=False (removes beta/bias parameter)
    - All sub-layers are created in __init__ and built explicitly for serialization

    Mathematical formulation:
        y = activation(BN(Conv1D(x)))
        where BN has no bias term (center=False) and Conv1D has no bias

    Args:
        filters: Integer, number of output filters for the convolution. Must be positive.
        kernel_size: Integer, size of the convolutional kernel along the time dimension.
            Must be positive and odd for symmetric filtering. Defaults to 3.
        activation: String name of activation function or callable. Common choices
            include 'relu', 'gelu', 'swish'. Defaults to 'relu'.
        kernel_initializer: String or Initializer instance for convolution weights.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: String or Regularizer instance for convolution weights.
            None means no regularization. Defaults to None.
        use_batch_norm: Boolean, whether to use batch normalization. When False,
            only convolution and activation are applied. Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, time_steps, features)`

    Output shape:
        3D tensor with shape: `(batch_size, time_steps, filters)`
        Time dimension is preserved due to 'same' padding.

    Example:
        ```python
        # Basic usage for time series denoising
        layer = BiasFreeConv1D(filters=64, kernel_size=3, activation='relu')

        # Use in a denoising model
        inputs = keras.Input(shape=(1000, 1))  # 1000 time steps, 1 feature
        x = BiasFreeConv1D(32, kernel_size=5)(inputs)
        x = BiasFreeConv1D(64, kernel_size=3)(x)
        outputs = BiasFreeConv1D(1, kernel_size=1, activation='linear')(x)
        model = keras.Model(inputs, outputs)

        # Without batch normalization
        layer_no_bn = BiasFreeConv1D(
            filters=32,
            kernel_size=3,
            use_batch_norm=False,
            kernel_regularizer='l2'
        )
        ```

    Note:
        This layer follows the bias-free principle essential for denoising tasks
        where scaling invariance is required. All additive constants are removed
        to prevent the network from learning to add noise patterns.

    References:
        - Mohan et al., "Robust and Interpretable Blind Image Denoising via
          Bias-Free Convolutional Neural Networks", ICLR 2020

    Raises:
        ValueError: If filters is not positive or kernel_size is invalid.
        TypeError: If activation, initializer or regularizer types are invalid.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        activation: Union[str, callable] = 'relu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_batch_norm: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(filters, int) or filters <= 0:
            raise ValueError(f"filters must be a positive integer, got {filters}")
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(f"kernel_size must be a positive integer, got {kernel_size}")
        if not isinstance(use_batch_norm, bool):
            raise TypeError(f"use_batch_norm must be boolean, got {type(use_batch_norm)}")

        # Store configuration parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        # Bias-free convolution
        self.conv = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False,  # Key: no bias terms for scaling invariance
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f'{self.name}_conv'
        )

        # Bias-free batch normalization (if enabled)
        if self.use_batch_norm:
            self.batch_norm = layers.BatchNormalization(
                center=False,  # Key: no bias/beta parameter
                scale=True,    # Keep gamma/scale parameter for feature scaling
                name=f'{self.name}_bn'
            )
        else:
            self.batch_norm = None

        # Activation layer (if specified)
        if self.activation is not None:
            self.activation_layer = layers.Activation(
                self.activation,
                name=f'{self.name}_activation'
            )
        else:
            self.activation_layer = None

        logger.debug(f"Initialized BiasFreeConv1D with {filters} filters, "
                    f"kernel_size={kernel_size}, batch_norm={use_batch_norm}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.

        Args:
            input_shape: Shape tuple indicating the input shape.
                Expected format: (batch_size, time_steps, features)
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input shape (batch_size, time_steps, features), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        if input_shape[-1] is None:
            raise ValueError("Last dimension (features) of input must be defined")

        # Build sub-layers in computational order
        self.conv.build(input_shape)
        logger.debug(f"Built conv layer with input shape {input_shape}")

        # Build batch norm if enabled
        if self.batch_norm is not None:
            conv_output_shape = self.conv.compute_output_shape(input_shape)
            self.batch_norm.build(conv_output_shape)
            logger.debug(f"Built batch norm layer with shape {conv_output_shape}")

        # Activation layers don't need explicit build() call
        if self.activation_layer is not None:
            logger.debug(f"Activation layer ready: {self.activation}")

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward computation through the bias-free 1D convolution layer.

        Args:
            inputs: Input tensor with shape (batch_size, time_steps, features).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Passed to batch normalization.

        Returns:
            Output tensor after bias-free convolution, normalization and activation
            with shape (batch_size, time_steps, filters).
        """
        # Apply convolution (no bias)
        x = self.conv(inputs)

        # Apply batch normalization if enabled (no center/beta)
        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)

        # Apply activation if specified
        if self.activation_layer is not None:
            x = self.activation_layer(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape tuple. Since we use 'same' padding, the time
            dimension is preserved and only the feature dimension changes.
        """
        # Use conv layer's output shape computation
        return self.conv.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration including all parameters
            needed to reconstruct the layer.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': keras.activations.serialize(
                keras.activations.get(self.activation)
            ),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'use_batch_norm': self.use_batch_norm,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BiasFreeResidualBlock1D(keras.layers.Layer):
    """
    Bias-free residual block for ResNet-style architecture with 1D convolutions.

    Implements a residual block using BiasFreeConv1D layers with a skip connection.
    The block performs: output = activation(input + F(input)), where F is the
    residual function computed by two bias-free convolution layers.

    Designed for time series data with shape [batch, time, features] where
    scaling invariance is crucial for robust denoising performance.

    Architecture:
        input -> BiasFreeConv1D -> BiasFreeConv1D -> Add(input) -> Activation

    Mathematical formulation:
        F(x) = BiasFreeConv1D(BiasFreeConv1D(x))
        output = activation(x + F(x))  [if input_filters == output_filters]
        output = activation(shortcut(x) + F(x))  [if dimensions don't match]

    Args:
        filters: Integer, number of filters in the convolutional layers. Must be positive.
        kernel_size: Integer, size of convolutional kernels along time dimension.
            Must be positive and typically odd for symmetric filtering. Defaults to 3.
        activation: String name of activation function or callable. Applied after
            the residual addition. Common choices: 'relu', 'gelu'. Defaults to 'relu'.
        kernel_initializer: String or Initializer instance for convolution weights.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: String or Regularizer instance for convolution weights.
            Applied to all convolutions in the block. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, time_steps, features)`

    Output shape:
        3D tensor with shape: `(batch_size, time_steps, filters)`

    Example:
        ```python
        # Basic residual block for time series
        block = BiasFreeResidualBlock1D(filters=64, kernel_size=3)

        # Use in a deep denoising network
        inputs = keras.Input(shape=(1000, 32))  # 1000 time steps, 32 features
        x = BiasFreeResidualBlock1D(64)(inputs)
        x = BiasFreeResidualBlock1D(64)(x)  # Same filters, pure residual
        x = BiasFreeResidualBlock1D(128)(x)  # Different filters, with shortcut conv
        outputs = BiasFreeConv1D(1, kernel_size=1, activation='linear')(x)
        model = keras.Model(inputs, outputs)

        # With regularization for better generalization
        regularized_block = BiasFreeResidualBlock1D(
            filters=32,
            kernel_size=5,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )
        ```

    Note:
        When input_filters != output_filters, a 1x1 convolution shortcut is added
        to match dimensions. This shortcut is also bias-free to maintain the
        scaling invariance property essential for robust denoising.

    References:
        - He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
        - Mohan et al., "Robust and Interpretable Blind Image Denoising via
          Bias-Free Convolutional Neural Networks", ICLR 2020

    Raises:
        ValueError: If filters is not positive or kernel_size is invalid.
        TypeError: If activation, initializer or regularizer types are invalid.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        activation: Union[str, callable] = 'relu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(filters, int) or filters <= 0:
            raise ValueError(f"filters must be a positive integer, got {filters}")
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(f"kernel_size must be a positive integer, got {kernel_size}")

        # Store configuration parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        # First conv layer with batch norm and activation
        self.conv1 = BiasFreeConv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_batch_norm=True,
            name=f'{self.name}_conv1'
        )

        # Second conv layer with batch norm but no activation (before addition)
        self.conv2 = BiasFreeConv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,  # No activation before addition
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_batch_norm=True,
            name=f'{self.name}_conv2'
        )

        # Shortcut convolution (created conditionally in build)
        self.shortcut_conv = None

        # Addition layer for residual connection
        self.add_layer = layers.Add(name=f'{self.name}_add')

        # Final activation after addition
        if self.activation is not None:
            self.final_activation = layers.Activation(
                self.activation,
                name=f'{self.name}_final_activation'
            )
        else:
            self.final_activation = None

        logger.debug(f"Initialized BiasFreeResidualBlock1D with {filters} filters, "
                    f"kernel_size={kernel_size}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the residual block components.

        CRITICAL: Explicitly build each sub-layer for robust serialization.

        Args:
            input_shape: Shape tuple indicating the input shape.
                Expected format: (batch_size, time_steps, features)
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input shape (batch_size, time_steps, features), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        if input_shape[-1] is None:
            raise ValueError("Last dimension (features) of input must be defined")

        input_filters = input_shape[-1]

        # Build main path layers
        self.conv1.build(input_shape)
        logger.debug(f"Built conv1 layer with input shape {input_shape}")

        conv1_output_shape = self.conv1.compute_output_shape(input_shape)
        self.conv2.build(conv1_output_shape)
        logger.debug(f"Built conv2 layer with input shape {conv1_output_shape}")

        # Create and build shortcut connection if needed (input filters != output filters)
        if input_filters != self.filters:
            self.shortcut_conv = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding='same',
                use_bias=False,  # Key: no bias for scaling invariance
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'{self.name}_shortcut'
            )
            self.shortcut_conv.build(input_shape)
            logger.debug(f"Built shortcut conv with input shape {input_shape}")

        # Addition and activation layers don't need explicit build
        logger.debug("Built residual block components")

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the residual block.

        Args:
            inputs: Input tensor with shape (batch_size, time_steps, features).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Passed to sub-layers.

        Returns:
            Output tensor with shape (batch_size, time_steps, filters).
        """
        # Main residual path: F(x)
        x = self.conv1(inputs, training=training)
        residual = self.conv2(x, training=training)

        # Shortcut path: either identity or 1x1 conv
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
        else:
            shortcut = inputs

        # Add residual and shortcut: shortcut + F(x)
        x = self.add_layer([shortcut, residual])

        # Final activation
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape tuple with updated feature dimension.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.filters  # Update feature dimension
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration including all parameters
            needed to reconstruct the layer.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': keras.activations.serialize(
                keras.activations.get(self.activation)
            ),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------