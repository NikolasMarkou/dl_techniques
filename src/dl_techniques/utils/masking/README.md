# Masking Utilities Module

The `dl_techniques.layers.masking` module provides a comprehensive collection of attention and segmentation masking utilities for deep learning, accessible through a unified factory interface for consistent mask creation and robust configuration management.

## Overview

This module includes twelve distinct mask types and a factory system for standardized creation and manipulation of masks. All utilities are designed for modern Keras 3.x compatibility and are backend-agnostic, making them suitable for transformer architectures, vision models, and instance segmentation pipelines.

## Available Mask Types

The following mask types are supported by the factory system with automated parameter validation and defaults:

| Type | Category | Description | Use Case |
|------|----------|-------------|----------|
| `causal` | Attention | Lower triangular mask preventing future attention | Autoregressive models, language generation |
| `sliding_window` | Attention | Causal mask with limited attention window | Efficient long-sequence processing |
| `global_local` | Attention | Dual masks for global and local attention patterns | Hierarchical attention mechanisms |
| `block_diagonal` | Attention | Block-wise attention within non-overlapping segments | Structured attention, hierarchical models |
| `random` | Attention | Random masking with specified probability | Regularization, attention pattern analysis |
| `banded` | Attention | Band around diagonal for local attention | Symmetric local attention patterns |
| `padding` | Attention | Attention mask from padding positions | Handling variable-length sequences |
| `valid_query` | Segmentation | Mask for valid object queries | Instance segmentation with variable objects |
| `spatial` | Segmentation | Spatial attention mask for image regions | Region-based attention focusing |
| `query_interaction` | Segmentation | Control interactions between object queries | Structured object relationships |
| `instance_separation` | Segmentation | Enforce separation between instance predictions | Non-overlapping instance masks |

## Factory Interface

### Basic Usage

```python
from dl_techniques.layers.masking import create_mask

# Create a standard causal mask
causal_mask = create_mask('causal', seq_len=128)

# Create sliding window mask with custom window size
window_mask = create_mask('sliding_window', seq_len=256, window_size=64)
```

### Configuration-Based Creation

```python
from dl_techniques.layers.masking import create_mask, MaskConfig

config = MaskConfig(
    mask_type='block_diagonal',
    seq_len=512,
    block_size=32,
    dtype='float32'
)

block_mask = create_mask(config=config)
```

### Mask Discovery

```python
from dl_techniques.layers.masking import get_mask_info

# Get information about all mask types
info = get_mask_info()

# Print requirements for a specific type
sliding_window_info = info['sliding_window']
print(f"Required parameters: {sliding_window_info['required_params']}")
print(f"Optional parameters: {sliding_window_info['optional_params']}")
```

### Validation

```python
from dl_techniques.layers.masking import MaskConfig

# Validate configuration before creation
try:
    config = MaskConfig(
        mask_type='sliding_window',
        seq_len=128,
        window_size=32
    )
    print("Configuration is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Mask-Specific Parameters

### Attention Masks

#### Causal
**Required:** `seq_len`  
**Optional:** `dtype` (default: 'bool')

```python
mask = create_mask('causal', seq_len=128, dtype='float32')
```

#### Sliding Window
**Required:** `seq_len`, `window_size`  
**Optional:** `dtype` (default: 'bool')

```python
mask = create_mask(
    'sliding_window',
    seq_len=256,
    window_size=64  # Each position attends to 64 previous positions
)
```

#### Global-Local
**Required:** `seq_len`, `sliding_window`  
**Optional:** `dtype` (default: 'bool')  
**Returns:** Tuple of (global_mask, local_mask)

```python
global_mask, local_mask = create_mask(
    'global_local',
    seq_len=512,
    sliding_window=128
)
```

#### Block Diagonal
**Required:** `seq_len`, `block_size`  
**Optional:** `dtype` (default: 'bool')

```python
mask = create_mask(
    'block_diagonal',
    seq_len=256,
    block_size=16  # Creates 16 independent attention blocks
)
```

#### Random
**Required:** `seq_len`  
**Optional:** `mask_probability` (default: 0.1), `seed` (default: None), `dtype` (default: 'bool')

```python
mask = create_mask(
    'random',
    seq_len=128,
    mask_probability=0.15,  # 15% of positions masked
    seed=42
)
```

#### Banded
**Required:** `seq_len`, `band_width`  
**Optional:** `dtype` (default: 'bool')

```python
mask = create_mask(
    'banded',
    seq_len=128,
    band_width=5  # Attention within ±2 positions from diagonal
)
```

#### Padding
**Required:** `padding_mask` (in extra_params)  
**Optional:** `dtype` (default: 'bool')

```python
padding_mask = keras.ops.array([[False, False, True, True]])  # Shape: (batch, seq_len)
config = MaskConfig(
    mask_type='padding',
    dtype='float32',
    extra_params={'padding_mask': padding_mask}
)
attention_mask = create_mask(config=config)  # Shape: (batch, seq_len, seq_len)
```

### Segmentation Masks

#### Valid Query
**Required:** `num_queries`  
**Optional:** `valid_queries` (default: None), `dtype` (default: 'bool')

```python
# Mark first 5 queries as valid out of 10
mask = create_mask(
    'valid_query',
    num_queries=10,
    valid_queries=5  # Scalar: number of valid queries
)
```

#### Spatial
**Required:** `height`, `width`  
**Optional:** `attention_regions` (default: None), `mask_mode` (default: 'inside'), `dtype` (default: 'bool')

```python
# Create spatial mask for 64x64 image regions
regions = keras.ops.ones((64, 64), dtype='bool')  # Binary mask of valid regions
mask = create_mask(
    'spatial',
    height=64,
    width=64,
    attention_regions=regions,
    mask_mode='outside'  # Mask outside the regions
)
```

#### Query Interaction
**Required:** `num_queries`  
**Optional:** `interaction_type` (default: 'self'), `hierarchy_levels` (default: None), `dtype` (default: 'bool')

```python
# Control query interactions
mask = create_mask(
    'query_interaction',
    num_queries=20,
    interaction_type='hierarchical',
    hierarchy_levels=keras.ops.array([0, 0, 1, 1, 2, 2, ...])  # Query hierarchy levels
)
```

#### Instance Separation
**Required:** `mask_predictions` (in extra_params)  
**Optional:** `separation_threshold` (default: 0.5), `dtype` (default: 'bool')

```python
predictions = keras.random.uniform((2, 10, 64, 64))  # (batch, queries, H, W)
config = MaskConfig(
    mask_type='instance_separation',
    separation_threshold=0.7,
    extra_params={'mask_predictions': predictions}
)
separation_mask = create_mask(config=config)
```

## Direct Factory Instantiation

While the unified interface is recommended, direct factory access is available:

```python
from dl_techniques.layers.masking import MaskFactory

# Direct factory usage (bypasses unified interface)
causal_mask = MaskFactory.create_causal_mask(seq_len=128)
window_mask = MaskFactory.create_sliding_window_mask(seq_len=256, window_size=64)
```

## Integration Patterns

### In Transformer Attention

```python
import keras
from dl_techniques.layers.masking import create_mask, apply_mask

@keras.saving.register_keras_serializable()
class MaskedAttention(keras.layers.Layer):
    def __init__(self, num_heads, key_dim, mask_type='causal', **mask_kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mask_type = mask_type
        self.mask_kwargs = mask_kwargs
        
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )
    
    def call(self, inputs, training=None):
        seq_len = keras.ops.shape(inputs)[1]
        
        # Create attention mask dynamically
        mask = create_mask(self.mask_type, seq_len=seq_len, **self.mask_kwargs)
        
        # Apply attention with mask
        attention_output = self.attention(
            inputs, inputs,
            attention_mask=mask,
            training=training
        )
        
        return attention_output
```

### In Instance Segmentation

```python
@keras.saving.register_keras_serializable()
class SegmentationHead(keras.layers.Layer):
    def __init__(self, num_queries, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
    
    def call(self, mask_logits, valid_queries=None):
        # Create query validity mask
        query_mask = create_mask(
            'valid_query',
            num_queries=self.num_queries,
            valid_queries=valid_queries
        )
        
        # Apply mask to logits
        masked_logits = apply_mask(
            mask_logits,
            query_mask,
            mask_type='segmentation'
        )
        
        return masked_logits
```

### Combining Multiple Masks

```python
from dl_techniques.layers.masking import create_mask, combine_masks

# Create individual masks
causal_mask = create_mask('causal', seq_len=128)
padding_mask = create_mask(
    config=MaskConfig(
        mask_type='padding',
        extra_params={'padding_mask': padding_tensor}
    )
)

# Combine masks with logical operations
combined_mask = combine_masks(
    causal_mask,
    padding_mask,
    combination='or'  # Mask if either condition is true
)
```

## Mask Application

### Universal Apply Function

```python
from dl_techniques.layers.masking import apply_mask

# Apply to attention logits
logits = keras.random.normal((2, 8, 128, 128))  # (batch, heads, seq, seq)
mask = create_mask('causal', seq_len=128)
masked_logits = apply_mask(logits, mask, mask_value=-1e9, mask_type='attention')

# Apply to segmentation predictions
predictions = keras.random.uniform((2, 10, 64, 64))  # (batch, queries, H, W)
query_mask = create_mask('valid_query', num_queries=10, valid_queries=5)
masked_preds = apply_mask(predictions, query_mask, mask_type='segmentation')
```

## Visualization

```python
from dl_techniques.layers.masking import create_mask, visualize_mask

# Create and visualize a mask
mask = create_mask('sliding_window', seq_len=64, window_size=16)

# Visualize with custom settings
visualize_mask(
    mask,
    title='Sliding Window Attention Pattern',
    figsize=(10, 8),
    cmap='Blues',
    save_path='attention_pattern.png',
    show=True
)
```

## Parameter Validation

The factory performs comprehensive validation:

### Required Parameter Checking
```python
# Will raise ValueError: seq_len required for causal mask
create_mask('causal')

# Will raise ValueError: window_size required for sliding window mask
create_mask('sliding_window', seq_len=128)
```

### Value Range Validation
```python
# Will raise ValueError: window_size must be positive
create_mask('sliding_window', seq_len=128, window_size=-1)

# Will raise ValueError: mask_probability must be in [0, 1]
create_mask('random', seq_len=128, mask_probability=1.5)
```

### Type Validation
```python
# Will raise ValueError: Invalid mask_type
create_mask('invalid_type', seq_len=128)
```

## Logging and Debugging

The module provides detailed logging for debugging:

### Debug Level Logging
Shows mask creation details:
```
DEBUG Created causal mask with shape (128, 128)
DEBUG Created sliding window mask: shape=(256, 256), window_size=64
DEBUG Combined 2 masks using or
```

### Info Level Logging
Provides visualization feedback:
```
INFO Saved mask visualization to attention_pattern.png
```

### Warning Level Logging
Alerts about potential issues:
```
WARNING matplotlib not available for visualization
WARNING Bounding box masking not fully implemented, returning no mask
```

## Best Practices

1. **Use the Unified Interface**: The `create_mask()` function provides consistent validation and logging across all mask types.

2. **Leverage Configuration Objects**: For complex setups, use `MaskConfig` for better organization and reusability:
   ```python
   config = MaskConfig(
       mask_type='sliding_window',
       seq_len=512,
       window_size=128,
       dtype='float32'
   )
   ```

3. **Cache Static Masks**: For masks that don't change during training, create them once and reuse:
   ```python
   class CachedMaskedAttention(keras.layers.Layer):
       def build(self, input_shape):
           seq_len = input_shape[1]
           self.mask = create_mask('causal', seq_len=seq_len)
           super().build(input_shape)
   ```

4. **Use Appropriate Data Types**: Use `bool` for memory efficiency, `float32` when needed for computation:
   ```python
   # Memory-efficient boolean mask
   bool_mask = create_mask('causal', seq_len=1024, dtype='bool')
   
   # Float mask for direct multiplication
   float_mask = create_mask('causal', seq_len=1024, dtype='float32')
   ```

5. **Validate Early**: Use `MaskConfig` validation in production pipelines to catch errors before training:
   ```python
   try:
       config = MaskConfig(mask_type=user_input_type, **user_params)
       mask = create_mask(config=config)
   except ValueError as e:
       logger.error(f"Invalid mask configuration: {e}")
       # Use fallback configuration
   ```

## API Reference

### Functions

#### `create_mask(mask_type=None, config=None, **kwargs)`
Universal interface for creating masks with validation.

#### `apply_mask(inputs, mask, mask_value=-1e9, mask_type=None)`
Apply a mask to inputs (attention logits or segmentation predictions).

#### `combine_masks(*masks, combination='or')`
Combine multiple masks using logical operations ('and', 'or', 'xor').

#### `visualize_mask(mask, title='Mask Visualization', **kwargs)`
Visualize a mask using matplotlib (if available).

#### `get_mask_info()`
Get comprehensive information about all available mask types.

### Classes

#### `MaskType`
Enum defining all valid mask type strings.

#### `MaskConfig`
Configuration dataclass for mask creation with validation.

#### `MaskFactory`
Factory class containing all mask creation methods.

### Types

#### `MaskType`
An `Enum` defining valid mask types: `'causal'`, `'sliding_window'`, `'global_local'`, `'block_diagonal'`, `'random'`, `'banded'`, `'padding'`, `'valid_query'`, `'spatial'`, `'query_interaction'`, `'instance_separation'`.

---