"""
Image Patch Embedding Layer for Vision Transformers

This module provides a Keras layer for converting images into sequence of patch embeddings,
which is a core component of Vision Transformer (ViT) architectures. The implementation is
compatible with Keras 3.8.0 and TensorFlow 2.18.0 backend.

The PatchEmbed layer takes an image and:
1. Splits it into fixed-size patches using a strided convolution
2. Projects each patch into an embedding space
3. Reshapes the output to create a sequence of patch embeddings

Features:
- Configurable patch size (square or rectangular)
- Customizable embedding dimension
- Support for kernel regularization and initialization
- Optional activation function

"""

import keras
import tensorflow as tf
from keras import Layer
from typing import Optional, Union, Tuple
from keras.api.initializers import Initializer
from keras.api.regularizers import Regularizer


@keras.utils.register_keras_serializable()
class PatchEmbed(Layer):
    """2D Image to Patch Embedding Layer.

    Splits images into patches and linearly embeds each patch.

    Args:
        patch_size: Size of patches to split the input image into
        embed_dim: Embedding dimension for patches
        kernel_initializer: Initializer for the projection matrix
        kernel_regularizer: Regularizer function for the projection matrix
        name: Optional name for the layer
        **kwargs: Additional layer arguments
    """

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]],
            embed_dim: int,
            kernel_initializer: Union[str, Initializer] = "glorot_normal",
            kernel_regularizer: Optional[Regularizer] = None,
            activation: Union[str, None] = "linear",
            name: Optional[str] = None,
            **kwargs
    ):
        """Initialize the PatchEmbed layer.

        Args:
            patch_size: Size of patches to split the input image into
            embed_dim: Embedding dimension for patches
            kernel_initializer: Initializer for the projection matrix
            kernel_regularizer: Regularizer function for the projection matrix
            activation: Activation function to use (default is 'linear')
            name: Optional name for the layer
            **kwargs: Additional layer arguments
        """
        super().__init__(name=name, **kwargs)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.activation = keras.activations.get(activation)
        self.proj = None

    def build(self, input_shape: tuple) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor (batch_size, height, width, channels)
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input (got {len(input_shape)}D input)")

        # Create the projection layer
        self.proj = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            activation=self.activation,
            padding="valid",
            name=f"{self.name}_projection" if self.name else None
        )

        # Build the projection layer
        self.proj.build(input_shape)

        self.built = True

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, height, width, channels)

        Returns:
            Embedded patches tensor of shape (batch_size, n_patches, embed_dim)
        """
        # Apply convolution to extract patches
        x = self.proj(x)  # (batch_size, h', w', embed_dim)

        # Reshape to (batch_size, n_patches, embed_dim)
        # The batch dimension is preserved automatically
        x = keras.layers.Reshape((-1, self.embed_dim))(x)

        return x

    def get_config(self) -> dict:
        """Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'activation': keras.activations.serialize(self.activation),
        })
        return config
