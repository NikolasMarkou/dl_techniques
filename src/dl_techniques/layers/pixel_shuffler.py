"""
This module provides a `PixelShuffle` layer, implementing a space-to-depth
transformation tailored for Vision Transformer (ViT) architectures.

In architectures like Vision Transformers, the computational cost of self-attention
grows quadratically with the number of input tokens. This layer provides an efficient,
non-destructive method to reduce the number of spatial tokens at intermediate
stages of the model, thereby decreasing computational complexity while preserving
all spatial information.

Key Concept: Space-to-Depth Transformation

Instead of discarding information through pooling or strided convolutions, this
operation *rearranges* data from the spatial dimensions (height and width) into the
channel dimension (depth).

The process works as follows:

1.  **Assumes ViT Input:** The layer expects a standard ViT input format: a sequence
    of tokens where the first token is a special `[CLS]` token and the rest are
    flattened spatial tokens from an image grid.

2.  **Separate CLS Token:** The `[CLS]` token, which holds the global representation,
    is temporarily set aside and is not affected by the spatial rearrangement.

3.  **Reshape to 2D Grid:** The sequence of spatial tokens is conceptually reshaped
    back into its 2D grid format (e.g., `196 tokens -> 14x14 grid`).

4.  **Rearrange Blocks:** The grid is then broken down into non-overlapping blocks of
    `scale_factor x scale_factor`. The information within each of these small spatial
    blocks is "folded" or stacked into the channel dimension.

5.  **Result:** This creates a new, smaller spatial grid where each "pixel" or token
    now has a much larger channel dimension, as it contains all the information
    from the original block. For example, with a `scale_factor` of 2, a `2x2` block of
    4 pixels is transformed into a single pixel, and its channel dimension is
    multiplied by 4.

6.  **Re-assemble Sequence:** The new, smaller spatial grid is flattened back into a
    sequence of tokens, and the original `[CLS]` token is prepended.

This allows subsequent self-attention layers to operate on a much shorter sequence,
making the model more efficient.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PixelShuffle(keras.layers.Layer):
    """Pixel shuffle operation for reducing spatial tokens in vision transformers.

    Implements pixel shuffle to reduce the number of visual tokens by rearranging
    spatial information into channel dimensions, enabling more efficient processing
    in vision-language models. This operation is particularly useful for reducing
    computational complexity while preserving spatial information.

    The layer assumes input tokens are arranged as [CLS_token, spatial_tokens] where
    spatial tokens represent a square spatial grid.

    Args:
        scale_factor: Factor by which to reduce spatial dimensions. Must be a positive
            integer that evenly divides the spatial dimensions. Default: 2.
        validate_spatial_dims: Whether to validate that spatial dimensions are
            perfect squares and divisible by scale_factor. Default: True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Example:
        >>> # Input: [batch, 197, 768] (196 spatial tokens + 1 CLS token, 14x14 spatial)
        >>> pixel_shuffle = PixelShuffle(scale_factor=2)
        >>> # Output: [batch, 50, 3072] (49 spatial tokens + 1 CLS token, 7x7 spatial)

    Raises:
        ValueError: If scale_factor is not a positive integer.
        ValueError: If spatial dimensions are not compatible with scale_factor
            when validate_spatial_dims is True.
    """

    def __init__(
            self,
            scale_factor: int = 2,
            validate_spatial_dims: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(scale_factor, int) or scale_factor <= 0:
            raise ValueError(
                f"scale_factor must be a positive integer, got {scale_factor}"
            )

        self.scale_factor = scale_factor
        self.validate_spatial_dims = validate_spatial_dims
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input shape is invalid or incompatible with scale_factor.
        """
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, channels), got shape {input_shape}"
            )

        # Validate spatial dimensions if enabled and known
        if self.validate_spatial_dims and input_shape[1] is not None:
            seq_len = input_shape[1]
            spatial_len = seq_len - 1  # Subtract CLS token

            if spatial_len <= 0:
                raise ValueError(
                    f"Sequence length must be > 1 (need at least CLS + 1 spatial token), "
                    f"got {seq_len}"
                )

            # Check if spatial_len is a perfect square
            h_float = spatial_len ** 0.5
            h = int(h_float)
            if h * h != spatial_len:
                raise ValueError(
                    f"Spatial tokens ({spatial_len}) must form a perfect square, "
                    f"got {spatial_len} tokens"
                )

            # Check if spatial dimensions are divisible by scale_factor
            if h % self.scale_factor != 0:
                raise ValueError(
                    f"Spatial dimension ({h}) must be divisible by scale_factor "
                    f"({self.scale_factor})"
                )

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply pixel shuffle operation.

        Args:
            inputs: Input tensor of shape [batch, num_tokens, channels] where
                num_tokens = 1 + H*W (CLS token + spatial tokens).

        Returns:
            Shuffled tensor with reduced spatial tokens of shape
            [batch, 1 + (H//scale_factor)*(W//scale_factor), channels*scale_factor^2].

        Raises:
            ValueError: If runtime spatial dimensions are incompatible.
        """
        # Get dynamic shapes
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        channels = inputs.shape[-1]  # Known at build time

        # Separate CLS token and spatial tokens
        cls_token = inputs[:, 0:1, :]  # [batch, 1, channels]
        spatial_tokens = inputs[:, 1:, :]  # [batch, H*W, channels]

        # Calculate spatial dimensions (assuming square)
        spatial_len = seq_len - 1
        h_float = ops.sqrt(ops.cast(spatial_len, "float32"))
        h = ops.cast(h_float, "int32")
        w = h  # Assume square spatial arrangement

        # Validate spatial dimensions at runtime if needed
        if self.validate_spatial_dims:
            # This creates a runtime assertion
            ops.assert_equal(
                h * h, spatial_len,
                message="Spatial tokens must form a perfect square"
            )
            ops.assert_equal(
                h % self.scale_factor, 0,
                message=f"Spatial dimension must be divisible by {self.scale_factor}"
            )

        # Reshape spatial tokens to 2D spatial format
        spatial_tokens = ops.reshape(spatial_tokens, [batch_size, h, w, channels])

        # Apply pixel shuffle (space-to-depth operation)
        new_h = h // self.scale_factor
        new_w = w // self.scale_factor
        new_c = channels * (self.scale_factor ** 2)

        # Rearrange pixels: group scale_factor x scale_factor blocks into channels
        # [B, H, W, C] -> [B, H//s, s, W//s, s, C] -> [B, H//s, W//s, s, s, C] -> [B, H//s, W//s, s*s*C]
        shuffled = ops.reshape(spatial_tokens, [
            batch_size, new_h, self.scale_factor, new_w, self.scale_factor, channels
        ])
        shuffled = ops.transpose(shuffled, [0, 1, 3, 2, 4, 5])
        shuffled = ops.reshape(shuffled, [batch_size, new_h * new_w, new_c])

        # Concatenate CLS token back
        return ops.concatenate([cls_token, shuffled], axis=1)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple.
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)
        batch_size, seq_len, channels = input_shape_list

        if seq_len is None or channels is None:
            # If sequence length or channels are unknown, we can't compute exact output shape
            new_channels = channels * (self.scale_factor ** 2) if channels is not None else None
            return tuple([batch_size, None, new_channels])

        spatial_len = seq_len - 1  # Remove CLS token
        new_spatial_len = spatial_len // (self.scale_factor ** 2)
        new_seq_len = new_spatial_len + 1  # Add CLS token back
        new_channels = channels * (self.scale_factor ** 2)

        # Return as tuple for consistency
        return tuple([batch_size, new_seq_len, new_channels])

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "scale_factor": self.scale_factor,
            "validate_spatial_dims": self.validate_spatial_dims,
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        """Build layer from configuration.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


def create_pixel_shuffle_model(
        input_shape: Tuple[int, int, int],
        scale_factor: int = 2,
        num_classes: int = 1000
) -> keras.Model:
    """Create a simple model demonstrating PixelShuffle usage.

    Args:
        input_shape: Input shape (seq_len, channels) for the model.
        scale_factor: Scale factor for pixel shuffle operation.
        num_classes: Number of output classes.

    Returns:
        Keras model with PixelShuffle layer.

    Example:
        >>> model = create_pixel_shuffle_model((197, 768), scale_factor=2)
        >>> model.summary()
    """
    inputs = keras.Input(shape=input_shape, name="token_inputs")

    # Apply pixel shuffle to reduce spatial tokens
    x = PixelShuffle(scale_factor=scale_factor, name="pixel_shuffle")(inputs)

    # Extract CLS token for classification
    cls_token = x[:, 0, :]  # [batch, channels * scale_factor^2]

    # Classification head
    x = keras.layers.LayerNormalization(name="final_norm")(cls_token)
    x = keras.layers.Dropout(0.1, name="final_dropout")(x)
    outputs = keras.layers.Dense(num_classes, name="classifier")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pixel_shuffle_model")
    return model

# ---------------------------------------------------------------------

