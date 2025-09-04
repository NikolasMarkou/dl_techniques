# Embedding Layers Module

The `dl_techniques.layers.embedding` module provides a comprehensive collection of embedding layer implementations for deep learning architectures, particularly for Transformers and models dealing with spatial data. It features a unified factory interface for consistent layer creation, configuration management, and robust parameter validation.

## Overview

This module includes seven distinct embedding layer types, covering patch tokenization, learned absolute positional encodings, and various forms of Rotary Position Embeddings (RoPE). All layers are designed for modern Keras 3.x compatibility with full serialization support and are accessible through a centralized factory system.

## Available Embedding Types

All layers in this module are supported by the factory system, which provides automated parameter validation, default value handling, and a consistent creation interface.

| Type | Class | Description | Use Case |
| :--- | :--- | :--- | :--- |
| `patch_1d` | `PatchEmbedding1D` | 1D patch embedding for time series data. | Tokenizing time series or other 1D sequential data for transformers. |
| `patch_2d` | `PatchEmbedding2D` | 2D image to patch embedding for Vision Transformers. | Converting images into a sequence of patch embeddings in ViT models. |
| `positional_learned` | `PositionalEmbedding`| Adds learned, trainable positional embeddings to a sequence. | Standard absolute positional encoding for transformer models. |
| `rope` | `RotaryPositionEmbedding` | Standard RoPE for relative position encoding. | Injecting relative positional information into Q/K vectors in attention. |
| `dual_rope` | `DualRotaryPositionEmbedding` | Dual RoPE with separate global/local configurations. | Gemma3-style models using both global and local attention patterns. |
| `continuous_rope`| `ContinuousRoPE` | RoPE extended for continuous multi-dimensional coordinates. | Positional encoding for data with spatial coordinates (e.g., 3D point clouds). |
| `continuous_sincos`| `ContinuousSinCosEmbed`| Embeds continuous coordinates using sine/cosine functions. | Creating fixed, smooth positional representations for continuous data. |

## Factory Interface

### Basic Usage

```python
from dl_techniques.layers.embedding import create_embedding_layer

# Create a patch embedding layer for a Vision Transformer
patch_embed = create_embedding_layer(
    'patch_2d',
    patch_size=16,
    embed_dim=768,
    name='vit_patch_embed'
)

# Create a standard Rotary Position Embedding layer
rope = create_embedding_layer(
    'rope',
    head_dim=64,
    max_seq_len=4096,
    rope_percentage=0.5
)
```

### Configuration-Based Creation

```python
from dl_techniques.layers.embedding import create_embedding_from_config

config = {
    'type': 'positional_learned',
    'max_seq_len': 1024,
    'dim': 512,
    'dropout': 0.1,
    'name': 'learned_pos_embed'
}

pos_embed = create_embedding_from_config(config)
```

### Parameter Discovery

```python
from dl_techniques.layers.embedding import get_embedding_info

# Get information about all embedding types
info = get_embedding_info()

# Print requirements for a specific type
rope_info = info['rope']
print(f"Required: {rope_info['required_params']}")
print(f"Optional: {list(rope_info['optional_params'].keys())}")
# Required: ['head_dim', 'max_seq_len']
# Optional: ['rope_theta', 'rope_percentage']
```

### Validation

```python
from dl_techniques.layers.embedding import validate_embedding_config

# Validate configuration before creation
try:
    validate_embedding_config('dual_rope', head_dim=256, max_seq_len=4096)
    print("Configuration is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Layer-Specific Parameters

### PatchEmbedding1D (`patch_1d`)
- **Required**: `patch_size`, `embed_dim`
- **Optional**: `stride` (default: `None`), `padding` (default: `'causal'`), `use_bias` (default: `True`)

```python
ts_embed = create_embedding_layer(
    'patch_1d',
    patch_size=16,
    embed_dim=128,
    stride=8  # Create overlapping patches
)
```

### PatchEmbedding2D (`patch_2d`)
- **Required**: `patch_size`, `embed_dim`
- **Optional**: `activation` (default: `'linear'`), `use_bias` (default: `True`)

```python
img_embed = create_embedding_layer(
    'patch_2d',
    patch_size=(16, 16),
    embed_dim=768
)
```

### PositionalEmbedding (`positional_learned`)
- **Required**: `max_seq_len`, `dim`
- **Optional**: `dropout` (default: `0.0`), `scale` (default: `0.02`)

```python
pos_embed = create_embedding_layer(
    'positional_learned',
    max_seq_len=512,
    dim=768,
    dropout=0.1
)
```

### RotaryPositionEmbedding (`rope`)
- **Required**: `head_dim`, `max_seq_len`
- **Optional**: `rope_theta` (default: `10000.0`), `rope_percentage` (default: `0.5`)

```python
rope_embed = create_embedding_layer(
    'rope',
    head_dim=64,
    max_seq_len=2048,
    rope_percentage=0.25 # Apply RoPE to first 25% of dimensions
)
```

### DualRotaryPositionEmbedding (`dual_rope`)
- **Required**: `head_dim`, `max_seq_len`
- **Optional**: `global_theta_base` (default: `1,000,000.0`), `local_theta_base` (default: `10,000.0`)

```python
dual_rope_embed = create_embedding_layer(
    'dual_rope',
    head_dim=256,
    max_seq_len=8192,
    global_theta_base=500000.0 # Custom base for long context
)
```

### ContinuousRoPE (`continuous_rope`)
- **Required**: `dim`, `ndim`
- **Optional**: `max_wavelength` (default: `10000.0`), `assert_positive` (default: `True`)

```python
# For 3D point cloud data
continuous_rope_3d = create_embedding_layer(
    'continuous_rope',
    dim=128, # Embedding dimension
    ndim=3   # 3D coordinates (x, y, z)
)
```

### ContinuousSinCosEmbed (`continuous_sincos`)
- **Required**: `dim`, `ndim`
- **Optional**: `max_wavelength` (default: `10000.0`), `assert_positive` (default: `True`)

```python
# For 2D spatial data
sincos_embed_2d = create_embedding_layer(
    'continuous_sincos',
    dim=256, # Embedding dimension
    ndim=2   # 2D coordinates (x, y)
)
```

## Direct Layer Instantiation
While the factory is recommended for consistency and validation, direct instantiation is always available.

```python
from dl_techniques.layers.embedding import PatchEmbedding2D, RotaryPositionEmbedding

# Direct instantiation (bypasses factory validation and defaults)
patch_embed = PatchEmbedding2D(patch_size=16, embed_dim=768)
rope_embed = RotaryPositionEmbedding(head_dim=64, max_seq_len=1024)
```

## Integration Patterns

### In a Vision Transformer (ViT)

```python
@keras.saving.register_keras_serializable()
class ViTInputBlock(keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, max_seq_len, **kwargs):
        super().__init__(**kwargs)
        # Create patch and positional embeddings using the factory
        self.patch_embed = create_embedding_layer(
            'patch_2d', patch_size=patch_size, embed_dim=embed_dim
        )
        # Assuming num_patches + 1 (for CLS token) fits in max_seq_len
        self.pos_embed = create_embedding_layer(
            'positional_learned', max_seq_len=max_seq_len, dim=embed_dim
        )

    def call(self, images):
        patches = self.patch_embed(images)
        # Here you would typically add a CLS token
        return self.pos_embed(patches)
```

### In an Attention Block with RoPE

```python
@keras.saving.register_keras_serializable()
class AttentionWithRoPE(keras.layers.Layer):
    def __init__(self, head_dim, max_seq_len, **kwargs):
        super().__init__(**kwargs)
        self.qkv_proj = keras.layers.Dense(head_dim * 3)
        self.rope = create_embedding_layer(
            'rope', head_dim=head_dim, max_seq_len=max_seq_len
        )

    def call(self, inputs):
        q, k, v = keras.ops.split(self.qkv_proj(inputs), 3, axis=-1)
        
        # Apply RoPE to queries and keys BEFORE attention
        q_rotated = self.rope(q)
        k_rotated = self.rope(k)
        
        # Perform attention with rotated Q and K
        # ... attention logic ...
        return # ... attention output
```

## Parameter Validation
The factory performs comprehensive validation, catching common errors before layer creation.

#### Required Parameter Checking
```python
# Raises ValueError: "Required parameters missing for patch_2d: ['embed_dim']"
create_embedding_layer('patch_2d', patch_size=16)
```

#### Value Range Validation
```python
# Raises ValueError: "rope_percentage must be in (0, 1], got 1.5"
create_embedding_layer('rope', head_dim=64, max_seq_len=512, rope_percentage=1.5)

# Raises ValueError: "head_dim must be positive, got -64"
create_embedding_layer('rope', head_dim=-64, max_seq_len=512)
```

#### Invalid String Value Validation
```python
# Raises ValueError: "padding must be 'same', 'valid', or 'causal', got 'invalid_padding'"
create_embedding_layer('patch_1d', patch_size=16, embed_dim=128, padding='invalid_padding')
```

## Logging and Debugging
The factory provides detailed logging to aid in debugging model configurations.

#### Info Level Logging
Shows all parameters passed to each layer, including resolved defaults.
```
INFO Creating 'rope' embedding layer with parameters:
INFO   'head_dim': 64
INFO   'max_seq_len': 4096
INFO   'name': None
INFO   'rope_percentage': 0.5
INFO   'rope_theta': 10000.0
```

#### Debug Level Logging
Confirms successful layer creation.
```
DEBUG Successfully created 'rope' layer: rotary_position_embedding
```

#### Error Logging
Provides detailed context when layer creation fails.
```
ERROR Failed to create 'patch_1d' embedding layer (PatchEmbedding1D).
  Required params: ['patch_size', 'embed_dim']
  Provided params: ['patch_size']
  Check parameter compatibility and types. Use get_embedding_info() for details.
  Original error: Required parameters missing for patch_1d: ['embed_dim']. Required: ['patch_size', 'embed_dim']
```

## API Reference

#### Functions
- `create_embedding_layer(embedding_type, name=None, **kwargs)`: Factory function for creating embedding layers with validation.
- `create_embedding_from_config(config)`: Create an embedding layer from a configuration dictionary.
- `validate_embedding_config(embedding_type, **kwargs)`: Validate configuration parameters before creation.
- `get_embedding_info()`: Get comprehensive information about all available embedding types.

#### Types
- `EmbeddingType`: A `Literal` type defining valid embedding type strings.