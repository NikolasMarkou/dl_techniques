"""
Bias-Free 1D Convolutional Layer

A building block for bias-free CNNs that removes all additive constants
to enable better generalization across noise levels in time series denoising tasks.

Based on "Robust and Interpretable Blind Image Denoising via Bias-Free
Convolutional Neural Networks" (Mohan et al., ICLR 2020), adapted for 1D signals.
"""

import keras
from keras import layers
from typing import Optional, Union, Tuple

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
    - All sublayers are explicitly built with correct shapes

    Args:
        filters: Integer, number of output filters for the convolution.
        kernel_size: Integer, size of the convolutional kernel along the time dimension.
        activation: String name of activation function or callable. Defaults to 'relu'.
        kernel_initializer: String or Initializer instance for convolution weights.
        kernel_regularizer: String or Regularizer instance for convolution weights.
        use_batch_norm: Boolean, whether to use batch normalization. Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, time_steps, features)`

    Output shape:
        3D tensor with shape: `(batch_size, new_time_steps, filters)`
        where new_time_steps depends on padding (same padding preserves time_steps).

    Example:
        >>> # Create a bias-free conv layer for time series
        >>> layer = BiasFreeConv1D(filters=64, kernel_size=3, activation='relu')
        >>>
        >>> # Use in a model for time series
        >>> inputs = keras.Input(shape=(100, 1))  # 100 time steps, 1 feature
        >>> x = layer(inputs)
        >>> model = keras.Model(inputs, x)
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        activation: Union[str, callable] = 'relu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_batch_norm: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Store configuration parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_batch_norm = use_batch_norm

        # Initialize sublayers to None - will be created in build()
        self.conv = None
        self.batch_norm = None
        self.activation_layer = None

        # Store the build shape for serialization
        self._build_input_shape = None

    def build(self, input_shape):
        """
        Create the layer's sublayers based on input shape.

        This method creates and builds all sublayers with the correct shapes,
        ensuring proper weight initialization and gradient flow.

        Args:
            input_shape: Shape tuple (tuple of integers) indicating the input shape.
                Expected format: (batch_size, time_steps, features)
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Create bias-free convolution sublayer
        self.conv = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False,  # Key: no bias terms for scaling invariance
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f'{self.name}_conv'
        )

        # Build the convolution layer with input shape
        self.conv.build(input_shape)
        logger.debug(f"Built conv layer {self.conv.name} with input shape {input_shape}")

        # Create bias-free batch normalization sublayer if requested
        if self.use_batch_norm:
            self.batch_norm = layers.BatchNormalization(
                center=False,  # Key: no bias/beta parameter
                scale=True,    # Keep gamma/scale parameter for feature scaling
                name=f'{self.name}_bn'
            )

            # Build batch norm layer with conv output shape
            conv_output_shape = self.conv.compute_output_shape(input_shape)
            self.batch_norm.build(conv_output_shape)
            logger.debug(f"Built batch norm layer {self.batch_norm.name} with shape {conv_output_shape}")

        # Create activation sublayer if specified
        if self.activation is not None:
            self.activation_layer = layers.Activation(
                self.activation,
                name=f'{self.name}_activation'
            )
            # Activation layers don't need explicit build() call
            logger.debug(f"Created activation layer {self.activation_layer.name}")

        super().build(input_shape)

    def call(self, inputs, training=None):
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

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape tuple. Since we use 'same' padding, the time
            dimension is preserved and only the feature dimension changes.
        """
        if self.conv is not None:
            return self.conv.compute_output_shape(input_shape)
        else:
            # If not built yet, compute expected shape
            input_shape_list = list(input_shape)
            input_shape_list[-1] = self.filters  # Change feature dimension
            return tuple(input_shape_list)

    def get_config(self):
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': keras.activations.serialize(
                keras.activations.get(self.activation)
            ),
            'kernel_initializer': keras.initializers.serialize(
                keras.initializers.get(self.kernel_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(
                keras.regularizers.get(self.kernel_regularizer)
            ),
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    def get_build_config(self):
        """
        Get the config needed to build the layer from a config.

        This method is needed for proper model saving and loading.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config):
        """
        Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BiasFreeResidualBlock1D(keras.layers.Layer):
    """
    Bias-free residual block for ResNet-style architecture with 1D convolutions.

    Implements a residual block using BiasFreeConv1D layers with a skip connection.
    The block performs: output = input + F(input), where F is the residual function.
    Designed for time series data with shape [batch, time, features].

    Args:
        filters: Integer, number of filters in the convolutional layers.
        kernel_size: Integer, size of convolutional kernels along time dimension.
        activation: String name of activation function or callable.
        kernel_initializer: String or Initializer instance for convolution weights.
        kernel_regularizer: String or Regularizer instance for convolution weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, time_steps, features)`

    Output shape:
        3D tensor with shape: `(batch_size, time_steps, filters)`

    Example:
        >>> # Create a bias-free residual block for time series
        >>> block = BiasFreeResidualBlock1D(filters=64, kernel_size=3)
        >>>
        >>> # Use in a time series model
        >>> inputs = keras.Input(shape=(100, 32))  # 100 time steps, 32 features
        >>> x = block(inputs)
        >>> model = keras.Model(inputs, x)
    """

    def __init__(
            self,
            filters: int,
            kernel_size: int = 3,
            activation: Union[str, callable] = 'relu',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Store configuration parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # Initialize sublayers to None - will be created in build()
        self.conv1 = None
        self.conv2 = None
        self.shortcut_conv = None
        self.add_layer = None
        self.final_activation = None

        # Store the build shape for serialization
        self._build_input_shape = None

    def build(self, input_shape):
        """
        Build the residual block components.

        Args:
            input_shape: Shape tuple indicating the input shape.
                Expected format: (batch_size, time_steps, features)
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        input_filters = input_shape[-1]

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
        self.conv1.build(input_shape)

        # Second conv layer with batch norm but no activation (before addition)
        conv1_output_shape = self.conv1.compute_output_shape(input_shape)
        self.conv2 = BiasFreeConv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,  # No activation before addition
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_batch_norm=True,
            name=f'{self.name}_conv2'
        )
        self.conv2.build(conv1_output_shape)

        # Shortcut connection (if input filters != output filters)
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

        # Addition layer for residual connection
        self.add_layer = layers.Add(name=f'{self.name}_add')

        # Final activation after addition
        if self.activation is not None:
            self.final_activation = layers.Activation(
                self.activation,
                name=f'{self.name}_final_activation'
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass through the residual block.

        Args:
            inputs: Input tensor with shape (batch_size, time_steps, features).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor with shape (batch_size, time_steps, filters).
        """
        # Main path
        x = self.conv1(inputs, training=training)
        residual = self.conv2(x, training=training)

        # Shortcut path
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
        else:
            shortcut = inputs

        # Add residual and shortcut
        x = self.add_layer([shortcut, residual])

        # Final activation
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def compute_output_shape(self, input_shape):
        """
        Compute output shape.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape tuple with updated feature dimension.
        """
        input_shape_list = list(input_shape)
        input_shape_list[-1] = self.filters  # Update feature dimension
        return tuple(input_shape_list)

    def get_config(self):
        """
        Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': keras.activations.serialize(
                keras.activations.get(self.activation)
            ),
            'kernel_initializer': keras.initializers.serialize(
                keras.initializers.get(self.kernel_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(
                keras.regularizers.get(self.kernel_regularizer)
            ),
        })
        return config

    def get_build_config(self):
        """
        Get build configuration.

        Returns:
            Dictionary containing the build configuration.
        """
        return {'input_shape': self._build_input_shape}

    def build_from_config(self, config):
        """
        Build from configuration.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

# ---------------------------------------------------------------------