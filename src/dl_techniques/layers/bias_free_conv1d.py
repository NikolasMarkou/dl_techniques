"""Bias-free 1D convolutional building block for scaling-invariant networks.

Removes all additive constants (convolution bias and batch normalization
center/beta) to enable better generalization across noise levels in time
series denoising tasks. The forward computation is
``y = activation(BN(Conv1D(x)))`` where both BN and Conv1D are bias-free,
ensuring that ``f(alpha * x) = alpha * f(x)`` for any positive scalar alpha.

Based on "Robust and Interpretable Blind Image Denoising via Bias-Free
Convolutional Neural Networks" (Mohan et al., ICLR 2020), adapted for 1D
signals.
"""

import keras
from keras import layers
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BiasFreeConv1D(keras.layers.Layer):
    """Bias-free 1D convolutional layer with optional batch normalization and activation.

    Implements a convolution without bias, followed by bias-free batch
    normalization (``center=False``) and activation. No additive constants are
    introduced at any stage, which is crucial for achieving scaling invariance
    and better generalization across noise levels. The mathematical formulation
    is ``y = activation(BN(Conv1D(x)))`` where BN has no beta term and Conv1D
    has ``use_bias=False``.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────┐
        │  Input (B, T, C_in)           │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  Conv1D (no bias, same pad)   │
        │  filters=F, kernel_size=K     │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  BatchNorm (center=False)     │  ← optional
        │  scale=True, no beta          │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  Activation (e.g. ReLU)       │  ← optional
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  Output (B, T, F)             │
        └───────────────────────────────┘

    :param filters: Number of output filters for the convolution. Must be positive.
    :type filters: int
    :param kernel_size: Size of the convolutional kernel along the time dimension.
        Must be positive. Defaults to 3.
    :type kernel_size: int
    :param activation: Activation function name or callable. Defaults to ``'relu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for convolution weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for convolution weights.
        Defaults to ``None``.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param use_batch_norm: Whether to use batch normalization. Defaults to ``True``.
    :type use_batch_norm: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
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
        """Build the layer and all sub-layers.

        :param input_shape: Shape tuple ``(batch_size, time_steps, features)``.
        :type input_shape: Tuple[Optional[int], ...]
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
        """Forward computation through the bias-free 1D convolution layer.

        :param inputs: Input tensor of shape ``(batch_size, time_steps, features)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, time_steps, filters)``.
        :rtype: keras.KerasTensor
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
        """Compute the output shape of the layer.

        :param input_shape: Shape of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        # Use conv layer's output shape computation
        return self.conv.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
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
    """Bias-free residual block for 1D convolutions with skip connections.

    Implements a residual block using two ``BiasFreeConv1D`` layers: the first
    applies activation, the second does not. A skip connection adds the input
    (or a 1x1 shortcut projection when dimensions differ) before a final
    activation. The formulation is ``output = activation(shortcut(x) + F(x))``
    where ``F(x) = BiasFreeConv1D(BiasFreeConv1D(x))``. All paths are bias-free
    to preserve scaling invariance.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────┐
        │  Input (B, T, C_in)           │
        └───────┬───────────┬───────────┘
                │           │
                │     ┌─────┴─────────────────┐
                │     │  BiasFreeConv1D + act  │
                │     └─────┬─────────────────┘
                │           ▼
                │     ┌───────────────────────┐
                │     │  BiasFreeConv1D (lin)  │
                │     └─────┬─────────────────┘
                │           │
                ▼           ▼
        ┌───────────┐  ┌────────┐
        │ shortcut  │  │  F(x)  │
        │ (1x1 or   │  │        │
        │  identity) │  │        │
        └─────┬─────┘  └───┬────┘
              └──────┬──────┘
                     ▼
              ┌──────────────┐
              │   Add + Act  │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │ Output (B,T,F)│
              └──────────────┘

    :param filters: Number of filters in the convolutional layers. Must be positive.
    :type filters: int
    :param kernel_size: Size of convolutional kernels. Defaults to 3.
    :type kernel_size: int
    :param activation: Activation function name or callable. Defaults to ``'relu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for convolution weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for convolution weights.
        Defaults to ``None``.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the Layer base class.
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
        """Build the residual block components.

        :param input_shape: Shape tuple ``(batch_size, time_steps, features)``.
        :type input_shape: Tuple[Optional[int], ...]
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
        """Forward pass through the residual block.

        :param inputs: Input tensor of shape ``(batch_size, time_steps, features)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, time_steps, filters)``.
        :rtype: keras.KerasTensor
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
        """Compute output shape.

        :param input_shape: Shape of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple with updated feature dimension.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.filters  # Update feature dimension
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
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
