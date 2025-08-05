"""
Pixel Shuffle Layer Implementation for Vision Transformers.

This module implements a pixel shuffle operation specifically designed for reducing the number
of spatial tokens in vision transformers while preserving spatial information by rearranging
it into channel dimensions. This technique is crucial for efficient vision-language models
and multi-scale vision processing.

## Mathematical Foundation

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

## Key Features

- **Token-aware design**: Handles CLS tokens separately from spatial tokens
- **Efficient processing**: Reduces computational complexity for subsequent layers
- **Information preservation**: No information loss, only spatial rearrangement
- **Flexible scaling**: Configurable scale factors for different reduction needs
- **Runtime validation**: Optional validation of spatial dimension compatibility

## Use Cases in Vision Transformers

### 1. Vision-Language Models
Reducing spatial tokens before cross-attention with text:
```python
# Input: [batch, 197, 768] (196 spatial + 1 CLS, 14×14 spatial)
pixel_shuffle = PixelShuffle(scale_factor=2)
# Output: [batch, 50, 3072] (49 spatial + 1 CLS, 7×7 spatial)
```

### 2. Hierarchical Processing
Creating multi-scale representations:
```python
# Stage 1: Full resolution
tokens_high = input_tokens  # [B, 197, 768]

# Stage 2: Reduced resolution with more channels
shuffle_2x = PixelShuffle(scale_factor=2)
tokens_mid = shuffle_2x(tokens_high)  # [B, 50, 3072]

# Stage 3: Further reduced resolution
shuffle_4x = PixelShuffle(scale_factor=2)  # Applied to already shuffled tokens
tokens_low = shuffle_4x(tokens_mid)  # [B, 14, 12288]
```

### 3. Computational Efficiency
Reducing tokens before expensive operations:
```python
# Before expensive cross-attention
shuffled_tokens = PixelShuffle(scale_factor=2)(vision_tokens)
cross_attention_output = cross_attention(shuffled_tokens, text_tokens)
```

## Implementation Details

### Spatial Token Arrangement
The layer assumes spatial tokens are arranged in row-major order representing
a square spatial grid:
```
Token indices for 3×3 spatial grid:
[CLS] [0] [1] [2] [3] [4] [5] [6] [7] [8]

Spatial arrangement:
[0] [1] [2]
[3] [4] [5]
[6] [7] [8]
```

### Pixel Shuffle Operation
For scale_factor=2, each 2×2 spatial block becomes 4 channels:
```
Input spatial (2×2):     Output channels:
[a] [b]          →       [a, b, c, d] (concatenated)
[c] [d]
```

### Memory Layout
The operation preserves all information while changing memory layout:
- **Before**: Many tokens, fewer channels per token
- **After**: Fewer tokens, more channels per token
- **Total parameters**: Unchanged (H*W*C = (H/s)*(W/s)*C*s²)

## Performance Considerations

### Computational Complexity
- **Spatial complexity**: Reduced by factor of s² for subsequent layers
- **Channel complexity**: Increased by factor of s² but affects fewer operations
- **Memory usage**: Identical total memory, different layout
- **Cache efficiency**: May improve due to spatial locality

### Recommended Scale Factors
- **scale_factor=2**: Most common, good balance of reduction and information density
- **scale_factor=4**: Aggressive reduction for very high-resolution inputs
- **scale_factor=1**: Identity operation, useful for architectural flexibility

## Integration Examples

### Basic Usage
```python
# Create layer
pixel_shuffle = PixelShuffle(scale_factor=2)

# Apply to vision transformer tokens
reduced_tokens = pixel_shuffle(vision_tokens)

# Use in model
model = keras.Sequential([
    # ... vision transformer layers ...
    PixelShuffle(scale_factor=2),
    # ... subsequent processing with fewer tokens ...
])
```

### With Validation
```python
# Strict validation for development
pixel_shuffle = PixelShuffle(
    scale_factor=2,
    validate_spatial_dims=True  # Validates perfect square and divisibility
)

# Relaxed validation for production
pixel_shuffle = PixelShuffle(
    scale_factor=2,
    validate_spatial_dims=False  # Faster, assumes valid inputs
)
```

## References

The pixel shuffle concept is adapted from super-resolution literature and extended
for vision transformer token processing:

1. Shi, W., et al. "Real-time single image and video super-resolution using an
   efficient sub-pixel convolutional neural network." CVPR 2016.
2. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using
   Shifted Windows." ICCV 2021.
3. Radford, A., et al. "Learning Transferable Visual Models From Natural Language
   Supervision." ICML 2021.

## Notes

- The layer assumes square spatial arrangements (H = W)
- CLS tokens are preserved and not affected by the shuffle operation
- Input validation can be disabled for performance in production environments
- The operation is fully differentiable and suitable for end-to-end training
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

# ---------------------------------------------------------------------

