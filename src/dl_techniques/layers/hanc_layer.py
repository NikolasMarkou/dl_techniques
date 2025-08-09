"""
Hierarchical Aggregation of Neighborhood Context (HANC) Layer Implementation.

This module implements the HANC layer from ACC-UNet, which performs hierarchical
context aggregation by computing mean and max pooling at multiple scales and
concatenating them to provide long-range dependencies.
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any

from dl_techniques.utils.logger import logger


class HANCLayer(keras.layers.Layer):
    """
    Hierarchical Aggregation of Neighborhood Context (HANC) Layer.

    This layer implements hierarchical context aggregation by computing average
    and max pooling at multiple scales (2x2, 4x4, 8x8, 16x16) and concatenating
    them along the channel dimension. This provides an approximate version of
    self-attention by comparing pixels with neighborhood statistics at multiple scales.

    The layer concatenates:
    - Original features
    - Average pooled features at k different scales (upsampled back)
    - Max pooled features at k different scales (upsampled back)

    Total output channels = input_channels * (2*k - 1)

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels after final 1x1 convolution.
        k: Number of hierarchical levels. k=1 means no pooling (original only),
           k=2 adds 2x2 pooling, k=3 adds 2x2 and 4x4, etc.
        kernel_initializer: Initializer for the 1x1 convolution kernel.
        kernel_regularizer: Regularizer for the 1x1 convolution kernel.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, channels).

    Output shape:
        4D tensor with shape (batch_size, height, width, out_channels).

    Example:
        ```python
        # Basic usage
        hanc = HANCLayer(in_channels=64, out_channels=64, k=3)

        # With custom initialization
        hanc = HANCLayer(
            in_channels=128,
            out_channels=128,
            k=4,
            kernel_initializer='he_normal'
        )
        ```

    Note:
        Higher k values provide more contextual information but increase
        computational cost. k=3 (up to 4x4 patches) is recommended for
        most applications.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            k: int = 3,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Validate k
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        # Calculate total channels after concatenation
        self.total_channels = in_channels * (2 * k - 1)

        # Will be initialized in build()
        self.conv = None
        self.batch_norm = None
        self.activation = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights."""
        self._build_input_shape = input_shape

        # 1x1 convolution to reduce channels
        self.conv = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='hanc_conv'
        )

        # Batch normalization
        self.batch_norm = keras.layers.BatchNormalization(name='hanc_bn')

        # Activation
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='hanc_activation')

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation."""
        batch_size = ops.shape(inputs)[0]
        height = ops.shape(inputs)[1]
        width = ops.shape(inputs)[2]

        # Start with original features
        features_list = [inputs]

        if self.k == 1:
            # No pooling, just original features
            concatenated = inputs
        else:
            # Add average pooled features at different scales
            for scale in range(1, self.k):
                pool_size = 2 ** scale  # 2, 4, 8, 16, ...

                # Average pooling
                avg_pooled = keras.layers.AveragePooling2D(
                    pool_size=pool_size,
                    strides=pool_size,
                    padding='same'
                )(inputs)

                # Upsample back to original size
                avg_upsampled = keras.layers.UpSampling2D(
                    size=pool_size,
                    interpolation='nearest'
                )(avg_pooled)

                # Ensure correct spatial dimensions by cropping/padding if needed
                avg_upsampled = self._resize_to_match(avg_upsampled, inputs)
                features_list.append(avg_upsampled)

            # Add max pooled features at different scales
            for scale in range(1, self.k):
                pool_size = 2 ** scale  # 2, 4, 8, 16, ...

                # Max pooling
                max_pooled = keras.layers.MaxPooling2D(
                    pool_size=pool_size,
                    strides=pool_size,
                    padding='same'
                )(inputs)

                # Upsample back to original size
                max_upsampled = keras.layers.UpSampling2D(
                    size=pool_size,
                    interpolation='nearest'
                )(max_pooled)

                # Ensure correct spatial dimensions
                max_upsampled = self._resize_to_match(max_upsampled, inputs)
                features_list.append(max_upsampled)

            # Concatenate all features along channel dimension
            concatenated = keras.layers.Concatenate(axis=-1)(features_list)

        # Apply 1x1 convolution to reduce channels
        x = self.conv(concatenated)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)

        return x

    def _resize_to_match(self, tensor: keras.KerasTensor, target: keras.KerasTensor) -> keras.KerasTensor:
        """Resize tensor to match target spatial dimensions."""
        target_height = ops.shape(target)[1]
        target_width = ops.shape(target)[2]
        tensor_height = ops.shape(tensor)[1]
        tensor_width = ops.shape(tensor)[2]

        # If dimensions don't match, crop or pad
        if tensor_height != target_height or tensor_width != target_width:
            # Simple cropping/padding - in practice this should rarely be needed
            # with proper upsampling, but included for robustness
            if tensor_height > target_height:
                tensor = tensor[:, :target_height, :, :]
            if tensor_width > target_width:
                tensor = tensor[:, :, :target_width, :]

        return tensor

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return tuple(list(input_shape[:-1]) + [self.out_channels])

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'k': self.k,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])