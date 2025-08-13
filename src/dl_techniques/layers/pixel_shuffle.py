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

# ---------------------------------------------------------------------

import keras
from keras import ops
from typing import Optional, Tuple, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PixelShuffle(keras.layers.Layer):
    def __init__(
            self,
            scale_factor: int = 2,
            validate_spatial_dims: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(scale_factor, int) or scale_factor <= 0:
            raise ValueError(f"scale_factor must be a positive integer, got {scale_factor}")
        self.scale_factor = scale_factor
        self.validate_spatial_dims = validate_spatial_dims

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, channels), got shape {input_shape}")
        if self.validate_spatial_dims and input_shape[1] is not None:
            seq_len = input_shape[1]
            spatial_len = seq_len - 1
            if spatial_len <= 0:
                raise ValueError(f"Sequence length must be > 1, got {seq_len}")
            h_float = spatial_len ** 0.5
            h = int(h_float)
            if h * h != spatial_len:
                raise ValueError(f"Spatial tokens ({spatial_len}) must form a perfect square.")
            if h % self.scale_factor != 0:
                raise ValueError(f"Spatial dimension ({h}) must be divisible by scale_factor ({self.scale_factor})")
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        if self.scale_factor == 1:
            return inputs
        input_shape = ops.shape(inputs)
        batch_size, seq_len, channels = input_shape[0], input_shape[1], inputs.shape[-1]
        cls_token, spatial_tokens = inputs[:, 0:1, :], inputs[:, 1:, :]
        spatial_len = seq_len - 1
        h = ops.cast(ops.sqrt(ops.cast(spatial_len, "float32")), "int32")
        w = h
        spatial_tokens = ops.reshape(spatial_tokens, [batch_size, h, w, channels])
        new_h, new_w = h // self.scale_factor, w // self.scale_factor
        new_c = channels * (self.scale_factor ** 2)
        shuffled = ops.reshape(spatial_tokens, [batch_size, new_h, self.scale_factor, new_w, self.scale_factor, channels])
        shuffled = ops.transpose(shuffled, [0, 1, 3, 2, 4, 5])
        shuffled = ops.reshape(shuffled, [batch_size, new_h * new_w, new_c])
        padding_amount = new_c - channels
        paddings = [[0, 0], [0, 0], [0, padding_amount]]
        cls_token_expanded = ops.pad(cls_token, paddings)
        return ops.concatenate([cls_token_expanded, shuffled], axis=1)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        batch_size, seq_len, channels = input_shape
        if seq_len is None:
            new_seq_len = None
        else:
            new_seq_len = ((seq_len - 1) // (self.scale_factor ** 2)) + 1
        if channels is None:
            new_channels = None
        else:
            new_channels = channels * (self.scale_factor ** 2)
        return batch_size, new_seq_len, new_channels

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "scale_factor": self.scale_factor,
            "validate_spatial_dims": self.validate_spatial_dims,
        })
        return config