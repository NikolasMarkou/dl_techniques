"""Convert a 2D image into a sequence of flattened patch embeddings.

This layer implements the input "tokenization" strategy for Vision
Transformers (ViT), bridging the gap between the grid-like structure of
images and the sequence-based architecture of Transformers. It transforms an
image into a sequence of vectors, where each vector represents a small
rectangular region, or "patch," of the original image.

Architecture:
    The core idea is to treat an image as an ordered sequence of smaller
    sub-images. This transformation is achieved elegantly and efficiently
    using a single `keras.layers.Conv2D` operation. The key insight lies in
    the specific configuration of this convolution:

    1.  **Kernel Size:** The `kernel_size` of the convolution is set to be
        identical to the desired `patch_size`. This ensures that the filter
        views exactly one patch at a time.

    2.  **Strides:** The `strides` of the convolution are also set equal to
        the `patch_size`. This makes the kernel slide across the image in
        non-overlapping steps, moving from one patch to the next adjacent one.

    3.  **Filters:** The number of convolution filters is set to the target
        `embed_dim`. This means that as the kernel processes each patch, it
        simultaneously performs a linear projection, transforming the patch
        into a vector of the desired embedding dimension.

    The output of this convolution is a 4D tensor of shape
    `(batch, num_patches_h, num_patches_w, embed_dim)`. A final reshape
    operation flattens the two spatial dimensions into a single sequence
    dimension, producing the required `(batch, num_patches, embed_dim)`
    output for the subsequent Transformer encoder.

Foundational Mathematics:
    Conceptually, the operation consists of two stages: patching and linear
    projection. For an input image `X ∈ R^(H x W x C)`, it is first divided
    into `N` flattened patches `x_p ∈ R^(P*P*C)`, where `P` is the patch size
    and `N = (H*W)/P^2` is the total number of patches.

    Each patch `x_p^(i)` is then linearly projected by a learnable weight
    matrix `E ∈ R^((P^2*C) x D)`, where `D` is the embedding dimension:

        z_i = x_p^(i) * E

    The `Conv2D` layer provides a computationally efficient implementation of
    this exact process. The convolutional kernel weights are equivalent to
    the projection matrix `E`, and the sliding window mechanism with a stride
    equal to the patch size performs both the patching and projection in a
    single, highly optimized operation.

References:
    - This patch embedding strategy is the cornerstone of the Vision
      Transformer (ViT) model, introduced in:
      Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
      Unterthiner, T., ... & Houlsby, N. (2020). "An Image is Worth 16x16
      Words: Transformers for Image Recognition at Scale".
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

    This layer follows the modern Keras 3 pattern where sub-layers are created in
    __init__() and explicitly built in build() for robust serialization.

    Args:
        patch_size: Size of patches to split the input image into. Can be an integer
            for square patches or a tuple (height, width) for rectangular patches.
            Must be positive.
        embed_dim: Embedding dimension for each patch. Must be positive.
        kernel_initializer: Initializer for the projection matrix. Defaults to
            "glorot_normal".
        kernel_regularizer: Optional regularizer function for the projection matrix.
        bias_initializer: Initializer for the bias vector. Defaults to "zeros".
        bias_regularizer: Optional regularizer function for the bias vector.
        activation: Activation function to apply after patch projection. Defaults
            to "linear" (no activation).
        use_bias: Boolean, whether to use bias in the projection layer. Defaults to True.
        **kwargs: Additional layer arguments.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)` where
        height and width must be divisible by the corresponding patch dimensions.

    Output shape:
        3D tensor with shape: `(batch_size, num_patches, embed_dim)` where
        num_patches = (height // patch_height) * (width // patch_width).

    Raises:
        ValueError: If patch_size or embed_dim are not positive.
        ValueError: If input dimensions are not divisible by patch dimensions during call.

    Example:
        ```python
        # Create patch embedding layer
        patch_embed = PatchEmbedding2D(patch_size=16, embed_dim=768)

        # Input image
        inputs = keras.Input(shape=(224, 224, 3))  # Standard ImageNet size

        # Convert to patches
        patches = patch_embed(inputs)
        print(patches.shape)  # (None, 196, 768) - 196 = 14*14 patches

        # In a model
        model = keras.Model(inputs, patches)
        ```
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
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if isinstance(patch_size, int):
            if patch_size <= 0:
                raise ValueError(f"patch_size must be positive, got {patch_size}")
            self.patch_size = (patch_size, patch_size)
        else:
            if len(patch_size) != 2 or any(p <= 0 for p in patch_size):
                raise ValueError(f"patch_size must be positive integer or tuple of 2 positive integers, got {patch_size}")
            self.patch_size = tuple(patch_size)

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")

        # Store ALL configuration parameters
        self.embed_dim = embed_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

        # CREATE sub-layer in __init__ (modern Keras 3 pattern)
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
            name="projection"
        )

        logger.info(f"Initialized PatchEmbedding2D with patch_size={self.patch_size}, "
                    f"embed_dim={self.embed_dim}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and its sub-layers.

        Args:
            input_shape: Shape of the input tensor (batch_size, height, width, channels).
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input (batch_size, height, width, channels), "
                             f"got {len(input_shape)}D input with shape {input_shape}")

        # Validate that height and width are divisible by patch size (if known)
        height, width = input_shape[1], input_shape[2]
        if height is not None and height % self.patch_size[0] != 0:
            raise ValueError(f"Input height ({height}) must be divisible by "
                             f"patch height ({self.patch_size[0]})")
        if width is not None and width % self.patch_size[1] != 0:
            raise ValueError(f"Input width ({width}) must be divisible by "
                             f"patch width ({self.patch_size[1]})")

        # CRITICAL: Explicitly build sub-layers for robust serialization
        self.proj.build(input_shape)

        # Always call parent build at the end
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
        x = self.proj(inputs, training=training)  # (batch_size, h_patches, w_patches, embed_dim)

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
            Dictionary containing ALL __init__ parameters for proper serialization.
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


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PatchEmbedding1D(keras.layers.Layer):
    """Patch embedding layer for time series data.

    Converts time series into patches and embeds them into a higher dimensional space.
    Supports overlapping patches through stride parameter.

    This layer follows the modern Keras 3 pattern where sub-layers are created in
    __init__() and explicitly built in build() for robust serialization.

    Args:
        patch_size: Integer, size of each patch. Must be positive.
        embed_dim: Integer, embedding dimension. Must be positive.
        stride: Integer, stride for patch extraction. If None, uses patch_size (non-overlapping).
            Must be positive if provided.
        padding: String, padding mode ('same', 'valid', or 'causal'). Defaults to 'causal'.
        use_bias: Boolean, whether to use bias in the embedding layer. Defaults to True.
        kernel_initializer: Initializer for the kernel weights. Defaults to "glorot_uniform".
        bias_initializer: Initializer for the bias vector. Defaults to "zeros".
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, seq_len, features)`.

    Output shape:
        3D tensor with shape: `(batch_size, output_len, embed_dim)` where
        output_len depends on padding mode and stride.

    Raises:
        ValueError: If patch_size, embed_dim, or stride are not positive.
        ValueError: If padding is not one of the allowed values.

    Example:
        ```python
        # Create 1D patch embedding layer
        patch_embed = PatchEmbedding1D(patch_size=16, embed_dim=256)

        # Input time series
        inputs = keras.Input(shape=(128, 64))  # seq_len=128, features=64

        # Convert to patches
        patches = patch_embed(inputs)
        print(patches.shape)  # (None, 8, 256) - 8 patches from 128 timesteps

        # With overlapping patches
        patch_embed_overlap = PatchEmbedding1D(patch_size=16, embed_dim=256, stride=8)
        patches_overlap = patch_embed_overlap(inputs)
        print(patches_overlap.shape)  # (None, 15, 256) - more patches due to overlap
        ```
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

        # Validate inputs
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if stride is not None and stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if padding not in ['same', 'valid', 'causal']:
            raise ValueError(f"padding must be one of ['same', 'valid', 'causal'], got {padding}")

        # Store ALL configuration parameters
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride if stride is not None else patch_size
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # CREATE sub-layer in __init__ (modern Keras 3 pattern)
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

        logger.info(f"Initialized PatchEmbedding1D with patch_size={self.patch_size}, "
                    f"embed_dim={self.embed_dim}, stride={self.stride}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and its sub-layers.

        Args:
            input_shape: Shape of the input tensor (batch_size, seq_len, features).
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch_size, seq_len, features), "
                             f"got {len(input_shape)}D input with shape {input_shape}")

        # CRITICAL: Explicitly build sub-layers for robust serialization
        self.embedding.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

        logger.info(f"Built PatchEmbedding1D with input_shape={input_shape}")

    def call(self, inputs, training: Optional[bool] = None) -> keras.KerasTensor:
        """Convert inputs to patches and embed them.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Embedded patches tensor of shape (batch_size, output_len, embed_dim).
        """
        # Handle NaN values by replacing with zeros
        x = ops.where(ops.isnan(inputs), 0.0, inputs)

        # Apply patch embedding
        embedded = self.embedding(x, training=training)

        return embedded

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape after patch embedding.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple (batch_size, output_len, embed_dim).
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {len(input_shape)}D")

        batch_size = input_shape[0]
        seq_len = input_shape[1]

        if seq_len is None:
            output_len = None
        else:
            if self.padding == 'valid':
                output_len = (seq_len - self.patch_size) // self.stride + 1
            elif self.padding == 'same':
                output_len = (seq_len + self.stride - 1) // self.stride
            else:  # causal
                output_len = seq_len // self.stride

        return (batch_size, output_len, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing ALL __init__ parameters for proper serialization.
        """
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


# ---------------------------------------------------------------------