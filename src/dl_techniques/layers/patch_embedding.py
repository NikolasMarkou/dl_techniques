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
from keras import ops
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PatchEmbedding2D(keras.layers.Layer):
    """2D Image to Patch Embedding Layer.

    Splits images into patches and linearly embeds each patch into a feature vector.
    This is commonly used as the first layer in Vision Transformers to convert
    images into sequences of patch embeddings.

    Args:
        patch_size: Size of patches to split the input image into. Can be an integer
            for square patches or a tuple (height, width) for rectangular patches.
        embed_dim: Embedding dimension for each patch.
        kernel_initializer: Initializer for the projection matrix. Defaults to
            "glorot_normal".
        kernel_regularizer: Optional regularizer function for the projection matrix.
        bias_initializer: Initializer for the bias vector. Defaults to "zeros".
        bias_regularizer: Optional regularizer function for the bias vector.
        activation: Activation function to apply after patch projection. Defaults
            to "linear" (no activation).
        use_bias: Boolean, whether to use bias in the projection layer.
        name: Optional name for the layer.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]],
            embed_dim: int,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activation: Optional[Union[str, callable]] = "linear",
            use_bias: bool = True,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # Store configuration parameters
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

        # Will be initialized in build()
        self.proj = None
        self._build_input_shape = None

        logger.info(f"Initialized PatchEmbedding2D with patch_size={self.patch_size}, "
                    f"embed_dim={self.embed_dim}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape of the input tensor (batch_size, height, width, channels).
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input (batch_size, height, width, channels), "
                             f"got {len(input_shape)}D input with shape {input_shape}")

        # Validate that height and width are divisible by patch size
        height, width = input_shape[1], input_shape[2]
        if height is not None and height % self.patch_size[0] != 0:
            raise ValueError(f"Input height ({height}) must be divisible by "
                             f"patch height ({self.patch_size[0]})")
        if width is not None and width % self.patch_size[1] != 0:
            raise ValueError(f"Input width ({width}) must be divisible by "
                             f"patch width ({self.patch_size[1]})")

        # Create the projection layer
        self.proj = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            use_bias=self.use_bias,
            activation=self.activation,
            padding="valid",
            name=f"{self.name}_projection" if self.name else "projection"
        )

        # Build the projection layer
        self.proj.build(input_shape)

        super().build(input_shape)

        logger.info(f"Built PatchEmbedding2D with input_shape={input_shape}")

    def call(self, inputs, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Embedded patches tensor of shape (batch_size, num_patches, embed_dim).
        """
        # Apply convolution to extract and embed patches
        x = self.proj(inputs, training=training)  # (batch_size, h', w', embed_dim)

        # Get the spatial dimensions after convolution
        batch_size = ops.shape(x)[0]
        h_patches = ops.shape(x)[1]
        w_patches = ops.shape(x)[2]

        # Calculate total number of patches
        num_patches = h_patches * w_patches

        # Reshape to (batch_size, num_patches, embed_dim)
        x = ops.reshape(x, (batch_size, num_patches, self.embed_dim))

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple (batch_size, num_patches, embed_dim).
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        batch_size = input_shape[0]
        height, width = input_shape[1], input_shape[2]

        # Calculate number of patches
        if height is not None and width is not None:
            h_patches = height // self.patch_size[0]
            w_patches = width // self.patch_size[1]
            num_patches = h_patches * w_patches
        else:
            num_patches = None

        return (batch_size, num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration for proper serialization.

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

    @property
    def num_patches(self) -> Optional[int]:
        """Get the number of patches this layer will produce.

        Returns:
            Number of patches if input shape is known, None otherwise.
        """
        if self._build_input_shape is None:
            return None

        height, width = self._build_input_shape[1], self._build_input_shape[2]
        if height is None or width is None:
            return None

        h_patches = height // self.patch_size[0]
        w_patches = width // self.patch_size[1]
        return h_patches * w_patches

    def get_patch_grid_shape(self) -> Optional[Tuple[int, int]]:
        """Get the grid shape of patches (height, width).

        Returns:
            Tuple of (h_patches, w_patches) if input shape is known, None otherwise.
        """
        if self._build_input_shape is None:
            return None

        height, width = self._build_input_shape[1], self._build_input_shape[2]
        if height is None or width is None:
            return None

        h_patches = height // self.patch_size[0]
        w_patches = width // self.patch_size[1]
        return (h_patches, w_patches)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PatchEmbedding1d(keras.layers.Layer):
    """
    Patch embedding layer for time series data.

    Converts time series into patches and embeds them into a higher dimensional space.
    Supports overlapping patches through stride parameter.

    Args:
        patch_size: Integer, size of each patch.
        embed_dim: Integer, embedding dimension.
        stride: Integer, stride for patch extraction. If None, uses patch_size (non-overlapping).
        padding: String, padding mode ('same', 'valid', or 'causal'). Defaults to 'causal'.
        use_bias: Boolean, whether to use bias in the embedding layer.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            patch_size: int,
            embed_dim: int,
            stride: Optional[int] = None,
            padding: str = 'causal',
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride if stride is not None else patch_size
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Embedding layer will be initialized in build()
        self.embedding = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the embedding layer."""
        self._build_input_shape = input_shape

        # Create 1D convolution layer for patch embedding
        self.embedding = keras.layers.Conv1D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="patch_embedding"
        )

        self.embedding.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Convert inputs to patches and embed them."""
        # Handle NaN values by replacing with zeros
        x = ops.where(ops.isnan(inputs), 0.0, inputs)

        # Apply patch embedding
        embedded = self.embedding(x, training=training)

        return embedded

    def compute_output_shape(self, input_shape):
        """Compute output shape after patch embedding."""
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        if self.padding == 'valid':
            output_len = (seq_len - self.patch_size) // self.stride + 1
        elif self.padding == 'same':
            output_len = (seq_len + self.stride - 1) // self.stride
        else:  # causal
            output_len = seq_len // self.stride

        return (batch_size, output_len, self.embed_dim)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "stride": self.stride,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
