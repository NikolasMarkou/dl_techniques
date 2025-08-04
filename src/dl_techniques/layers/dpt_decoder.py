"""
This module provides a `DPTDecoder` layer, which implements the decoder component of
the Dense Prediction Transformer (DPT) architecture.

While the "Transformer" part of DPT refers to its encoder (which uses a Vision
Transformer to extract features), the decoder's role is to take these powerful,
high-level features and translate them back into a dense, pixel-wise prediction, such
as a depth map or a segmentation mask.

This decoder is not a Transformer-based decoder. Instead, it is a lightweight and
effective convolutional head. It progressively refines the feature maps from the
encoder, gradually reducing the channel dimensionality while maintaining the spatial
resolution to produce the final dense output.

Architectural Design:

1.  **Sequential Convolutional Blocks:**
    -   The core of the decoder is a sequence of simple convolutional blocks.
    -   Each block consists of a `3x3 Conv2D` layer, followed by `BatchNormalization`
        and a non-linear `Activation` (e.g., ReLU).
    -   The number of blocks and the channel dimension of each block are defined by
        the `dims` parameter. For example, `dims=[256, 128, 64]` would create three
        such blocks, reducing the channel dimension from the input's size down to 64.

2.  **No Upsampling:**
    -   A key characteristic of this specific decoder design is that it **does not
        perform any spatial upsampling**. It assumes that the input features from the
        encoder already have the desired output spatial resolution.
    -   This is common in DPT variants where the encoder is designed to preserve
        spatial detail or where features from multiple scales are fused *before*
        being passed to this final decoder head.

3.  **Final Output Projection:**
    -   After the sequence of refining blocks, a final `3x3 Conv2D` layer is used to
        project the features into the desired number of `output_channels`.
    -   This final layer also applies an `output_activation` (e.g., `sigmoid` for
        depth estimation between 0 and 1, or `softmax` for multi-class segmentation)
        to format the output for the specific prediction task.

In summary, the `DPTDecoder` acts as a simple but effective "prediction head" that
takes a high-dimensional feature map from a powerful encoder and translates it into a
low-dimensional, interpretable, pixel-wise output map.
"""

import keras
from typing import Dict, Tuple, Optional, Any, List, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DPTDecoder(keras.layers.Layer):
    """DPT (Dense Prediction Transformer) decoder.

    Implements the decoder component of the DPT architecture for dense prediction tasks.
    The decoder processes multi-scale features through a series of convolutional layers
    with batch normalization and activation functions to produce dense predictions.

    Args:
        dims: List of integers, decoder layer dimensions for each stage.
            Defaults to [256, 128, 64, 32].
        output_channels: Integer, number of output channels for final prediction.
            Defaults to 1.
        kernel_initializer: String or Initializer, initializer for convolutional kernels.
            Defaults to "he_normal".
        kernel_regularizer: Regularizer or None, regularizer for convolutional kernels.
            Defaults to None.
        use_bias: Boolean, whether to use bias in convolutional layers.
            Defaults to False.
        activation: String or callable, activation function to use.
            Defaults to "relu".
        output_activation: String or callable, activation function for output layer.
            Defaults to "sigmoid".
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, height, width, output_channels)`

    Returns:
        A 4D tensor representing the decoded dense predictions.

    Example:
        >>> decoder = DPTDecoder(dims=[256, 128, 64, 32], output_channels=1)
        >>> x = keras.random.normal([2, 64, 64, 768])  # Input features
        >>> output = decoder(x)
        >>> print(output.shape)
        (2, 64, 64, 1)
    """

    def __init__(
            self,
            dims: Optional[List[int]] = None,
            output_channels: int = 1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_bias: bool = False,
            activation: Union[str, callable] = "relu",
            output_activation: Union[str, callable] = "sigmoid",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration parameters
        self.dims = dims if dims is not None else [256, 128, 64, 32]
        self.output_channels = output_channels
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.activation = activation
        self.output_activation = output_activation

        # Will be initialized in build()
        self.conv_layers: List[keras.layers.Conv2D] = []
        self.batch_norm_layers: List[keras.layers.BatchNormalization] = []
        self.activation_layers: List[keras.layers.Layer] = []
        self.output_conv: Optional[keras.layers.Conv2D] = None
        self._build_input_shape: Optional[Tuple[int, ...]] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build decoder layers based on input shape.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Clear any existing layers (in case of rebuild)
        self.conv_layers = []
        self.batch_norm_layers = []
        self.activation_layers = []

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
            bn = keras.layers.BatchNormalization(name=f'bn_{i}')
            self.batch_norm_layers.append(bn)

            # Activation layer
            activation_layer = keras.layers.Activation(
                self.activation,
                name=f'activation_{i}'
            )
            self.activation_layers.append(activation_layer)

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

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through decoder.

        Args:
            inputs: Input features tensor with shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Decoded output tensor with shape (batch_size, height, width, output_channels).
        """
        x = inputs

        # Apply decoder layers sequentially
        for conv, bn, activation in zip(
                self.conv_layers,
                self.batch_norm_layers,
                self.activation_layers
        ):
            x = conv(x)
            x = bn(x, training=training)
            x = activation(x)

        # Apply final output layer
        x = self.output_conv(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)

        # Output has same spatial dimensions but different channel count
        output_shape_list = input_shape_list[:-1] + [self.output_channels]

        # Return as tuple for consistency
        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "dims": self.dims,
            "output_channels": self.output_channels,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "activation": self.activation,
            "output_activation": self.output_activation,
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
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
