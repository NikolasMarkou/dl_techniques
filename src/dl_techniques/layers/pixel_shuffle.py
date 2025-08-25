"""
Pixel Shuffle Layer Implementation for Vision Transformers.

This module implements a pixel shuffle operation specifically designed for reducing the number
of spatial tokens in vision transformers while preserving spatial information by rearranging
it into channel dimensions. This technique is crucial for efficient vision-language models
and multi-scale vision processing.

Mathematical Foundation
-----------------------

The pixel shuffle operation performs a space-to-depth transformation on vision transformer
tokens arranged as [CLS_token, spatial_tokens]. Given an input with spatial dimensions
H×W and C channels, the operation:

1. Separates the CLS token from spatial tokens
2. Reshapes spatial tokens from [B, H*W, C] to [B, H, W, C]
3. Groups scale_factor × scale_factor spatial blocks into channel dimensions
4. Reduces spatial dimensions by scale_factor in each direction
5. Increases channel dimensions by scale_factor²

Mathematical transformation:
- Input:  [B, 1 + H*W, C]
- Output: [B, 1 + (H/s)*(W/s), C*s²]

where s is the scale_factor and B is batch size.

Key Features
------------

- **Token-aware design**: Handles CLS tokens separately from spatial tokens
- **Efficient processing**: Reduces computational complexity for subsequent layers
- **Information preservation**: No information loss, only spatial rearrangement
- **Flexible scaling**: Configurable scale factors for different reduction needs
- **Runtime validation**: Optional validation of spatial dimension compatibility

Use Cases in Vision Transformers
---------------------------------

Vision-Language Models: Reducing spatial tokens before cross-attention with text
Hierarchical Processing: Creating multi-scale representations
Computational Efficiency: Reducing tokens before expensive operations

Implementation Details
----------------------

The layer assumes spatial tokens are arranged in row-major order representing
a square spatial grid. For scale_factor=2, each 2×2 spatial block becomes 4 channels.
The operation preserves all information while changing memory layout.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Any, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PixelShuffle(keras.layers.Layer):
    """
    Pixel shuffle operation for reducing spatial tokens in vision transformers.

    Implements pixel shuffle to reduce the number of visual tokens by rearranging
    spatial information into channel dimensions, enabling more efficient processing
    in vision-language models. This operation is particularly useful for reducing
    computational complexity while preserving spatial information.

    The layer assumes input tokens are arranged as [CLS_token, spatial_tokens] where
    spatial tokens represent a square spatial grid.

    Args:
        scale_factor: Integer, factor by which to reduce spatial dimensions. Must be a
            positive integer that evenly divides the spatial dimensions. Defaults to 2.
        validate_spatial_dims: Boolean, whether to validate that spatial dimensions are
            perfect squares and divisible by scale_factor. Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, num_tokens, channels)` where
        num_tokens = 1 + H*W (CLS token + spatial tokens).

    Output shape:
        3D tensor with shape: `(batch_size, 1 + (H//scale_factor)*(W//scale_factor),
        channels*scale_factor^2)`.

    Example:
        ```python
        # Input: [batch, 197, 768] (196 spatial tokens + 1 CLS token, 14x14 spatial)
        pixel_shuffle = PixelShuffle(scale_factor=2)
        # Output: [batch, 50, 3072] (49 spatial tokens + 1 CLS token, 7x7 spatial)

        # Basic usage
        inputs = keras.Input(shape=(197, 768))
        outputs = PixelShuffle(scale_factor=2)(inputs)

        # With validation disabled for performance
        fast_shuffle = PixelShuffle(scale_factor=2, validate_spatial_dims=False)
        ```

    Raises:
        ValueError: If scale_factor is not a positive integer.
        ValueError: If spatial dimensions are not compatible with scale_factor
            when validate_spatial_dims is True.

    Note:
        The layer assumes square spatial arrangements (H = W) and that CLS tokens
        are preserved and not affected by the shuffle operation. The operation is
        fully differentiable and suitable for end-to-end training.
    """

    def __init__(
        self,
        scale_factor: int = 2,
        validate_spatial_dims: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(scale_factor, int) or scale_factor <= 0:
            raise ValueError(
                f"scale_factor must be a positive integer, got {scale_factor}"
            )

        # Store ALL configuration
        self.scale_factor = scale_factor
        self.validate_spatial_dims = validate_spatial_dims

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Validate input shape and build the layer.

        This layer creates no weights, only validates input compatibility.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input shape is invalid or incompatible with scale_factor.
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, channels), got shape {input_shape}"
            )

        # Validate spatial dimensions if enabled and known at build time
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

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply pixel shuffle operation.

        Args:
            inputs: Input tensor of shape [batch, num_tokens, channels] where
                num_tokens = 1 + H*W (CLS token + spatial tokens).
            training: Boolean indicating training mode (unused for this layer).

        Returns:
            Shuffled tensor with reduced spatial tokens of shape
            [batch, 1 + (H//scale_factor)*(W//scale_factor), channels*scale_factor^2].
        """
        # Identity operation for scale_factor=1
        if self.scale_factor == 1:
            return inputs

        # Get dynamic shapes
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        channels = inputs.shape[-1]  # Known at build time

        # Separate CLS token and spatial tokens
        cls_token = inputs[:, 0:1, :]  # [batch, 1, channels]
        spatial_tokens = inputs[:, 1:, :]  # [batch, H*W, channels]

        # Calculate spatial dimensions (assuming square)
        spatial_len = seq_len - 1
        h_float = ops.sqrt(ops.cast(spatial_len, "float32"))
        h = ops.cast(h_float, "int32")
        w = h  # Assume square spatial arrangement

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

        # Pad CLS token to match the new channel dimension
        padding_amount = new_c - channels
        paddings = [[0, 0], [0, 0], [0, padding_amount]]
        cls_token_expanded = ops.pad(cls_token, paddings)

        # Concatenate CLS token back
        return ops.concatenate([cls_token_expanded, shuffled], axis=1)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple.
        """
        input_shape_list = list(input_shape)
        batch_size, seq_len, channels = input_shape_list

        if seq_len is None:
            new_seq_len = None
        else:
            spatial_len = seq_len - 1
            new_spatial_len = spatial_len // (self.scale_factor ** 2)
            new_seq_len = new_spatial_len + 1

        if channels is None:
            new_channels = None
        else:
            new_channels = channels * (self.scale_factor ** 2)

        return tuple([batch_size, new_seq_len, new_channels])

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters.
        """
        config = super().get_config()
        config.update({
            "scale_factor": self.scale_factor,
            "validate_spatial_dims": self.validate_spatial_dims,
        })
        return config

# ---------------------------------------------------------------------