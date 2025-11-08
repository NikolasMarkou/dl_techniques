# Embedding Layers Module

The `dl_techniques.layers.embedding` module provides a comprehensive collection of embedding layer implementations for deep learning architectures, particularly for Transformers and models dealing with spatial data. It features a unified factory interface for consistent layer creation, configuration management, and robust parameter validation.

## Overview

This module includes eight distinct embedding layer types, covering patch tokenization, learned absolute positional encodings, various forms of Rotary Position Embeddings (RoPE), and BERT-style embeddings. All layers are designed for modern Keras 3.x compatibility with full serialization support and are accessible through a centralized factory system.

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
| `bert_embeddings` | `BertEmbeddings` | BERT embeddings combining word, position, and token type embeddings. | BERT-style language models with configurable normalization. |

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

# Create BERT embeddings for language modeling
bert_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=30522,
    hidden_size=768,
    max_position_embeddings=512,
    type_vocab_size=2,
    normalization_type='rms_norm'
)
```

### Configuration-Based Creation

```python
from dl_techniques.layers.embedding import create_embedding_from_config

config = {
    'type': 'bert_embeddings',
    'vocab_size': 30522,
    'hidden_size': 768,
    'max_position_embeddings': 512,
    'type_vocab_size': 2,
    'dropout_rate': 0.1,
    'normalization_type': 'layer_norm',
    'name': 'bert_embeddings'
}

bert_embed = create_embedding_from_config(config)
```

### Parameter Discovery

```python
from dl_techniques.layers.embedding import get_embedding_info

# Get information about all embedding types
info = get_embedding_info()

# Print requirements for a specific type
bert_info = info['bert_embeddings']
print(f"Required: {bert_info['required_params']}")
print(f"Optional: {list(bert_info['optional_params'].keys())}")
# Required: ['vocab_size', 'hidden_size', 'max_position_embeddings', 'type_vocab_size']
# Optional: ['initializer_range', 'layer_norm_eps', 'dropout_rate', 'normalization_type']
```

### Validation

```python
from dl_techniques.layers.embedding import validate_embedding_config

# Validate configuration before creation
try:
    validate_embedding_config(
        'bert_embeddings', 
        vocab_size=30522, 
        hidden_size=768,
        max_position_embeddings=512,
        type_vocab_size=2
    )
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
- **Optional**: `dropout_rate` (default: `0.0`), `scale` (default: `0.02`)

```python
pos_embed = create_embedding_layer(
    'positional_learned',
    max_seq_len=512,
    dim=768,
    dropout_rate=0.1
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

### BertEmbeddings (`bert_embeddings`)
- **Required**: `vocab_size`, `hidden_size`, `max_position_embeddings`, `type_vocab_size`
- **Optional**: `initializer_range` (default: `0.02`), `layer_norm_eps` (default: `1e-8`), `hidden_dropout_prob` (default: `0.0`), `normalization_type` (default: `'layer_norm'`)

```python
# Standard BERT embeddings
bert_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=30522,
    hidden_size=768,
    max_position_embeddings=512,
    type_vocab_size=2
)

# BERT embeddings with RMS normalization for efficiency
bert_rms_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=50257,  # GPT-2 vocab size
    hidden_size=1024,
    max_position_embeddings=1024,
    type_vocab_size=1,  # Only one segment type
    normalization_type='rms_norm',
    dropout_rate=0.1
)

# BERT embeddings with Band RMS for improved stability
bert_band_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=30522,
    hidden_size=768,
    max_position_embeddings=512,
    type_vocab_size=2,
    normalization_type='band_rms',
    layer_norm_eps=1e-12
)
```

## Direct Layer Instantiation
While the factory is recommended for consistency and validation, direct instantiation is always available.

```python
from dl_techniques.layers.embedding import PatchEmbedding2D, RotaryPositionEmbedding, BertEmbeddings

# Direct instantiation (bypasses factory validation and defaults)
patch_embed = PatchEmbedding2D(patch_size=16, embed_dim=768)
rope_embed = RotaryPositionEmbedding(head_dim=64, max_seq_len=1024)
bert_embed = BertEmbeddings(
    vocab_size=30522,
    hidden_size=768,
    max_position_embeddings=512,
    type_vocab_size=2
)
```

## Integration Patterns

### In a BERT-Style Language Model

```python
@keras.saving.register_keras_serializable()
class BertInputLayer(keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, 
                 type_vocab_size, normalization_type='layer_norm', **kwargs):
        super().__init__(**kwargs)
        
        # Create BERT embeddings using the factory
        self.embeddings = create_embedding_layer(
            'bert_embeddings',
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            normalization_type=normalization_type,
            name='bert_embeddings'
        )

    def call(self, input_ids, token_type_ids=None, position_ids=None, training=None):
        return self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training
        )
```

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

# Raises ValueError: "Required parameters missing for bert_embeddings: ['type_vocab_size']"
create_embedding_layer('bert_embeddings', vocab_size=30522, hidden_size=768, max_position_embeddings=512)
```

#### Value Range Validation
```python
# Raises ValueError: "rope_percentage must be in (0, 1], got 1.5"
create_embedding_layer('rope', head_dim=64, max_seq_len=512, rope_percentage=1.5)

# Raises ValueError: "vocab_size must be positive, got -30522"
create_embedding_layer('bert_embeddings', vocab_size=-30522, hidden_size=768, 
                      max_position_embeddings=512, type_vocab_size=2)

# Raises ValueError: "dropout_rate must be in [0, 1], got 1.5"
create_embedding_layer('bert_embeddings', vocab_size=30522, hidden_size=768,
                      max_position_embeddings=512, type_vocab_size=2, dropout_rate=1.5)
```

#### Invalid String Value Validation
```python
# Raises ValueError: "padding must be 'same', 'valid', or 'causal', got 'invalid_padding'"
create_embedding_layer('patch_1d', patch_size=16, embed_dim=128, padding='invalid_padding')

# Raises ValueError: "normalization_type must be one of ['layer_norm', 'rms_norm', 'band_rms', 'batch_norm'], got 'invalid_norm'"
create_embedding_layer('bert_embeddings', vocab_size=30522, hidden_size=768,
                      max_position_embeddings=512, type_vocab_size=2, normalization_type='invalid_norm')
```

## Logging and Debugging
The factory provides detailed logging to aid in debugging model configurations.

#### Info Level Logging
Shows all parameters passed to each layer, including resolved defaults.
```
INFO Creating 'bert_embeddings' embedding layer with parameters:
INFO   'dropout_rate': 0.1
INFO   'hidden_size': 768
INFO   'initializer_range': 0.02
INFO   'layer_norm_eps': 1e-08
INFO   'max_position_embeddings': 512
INFO   'name': 'bert_embeddings'
INFO   'normalization_type': 'rms_norm'
INFO   'type_vocab_size': 2
INFO   'vocab_size': 30522
```

#### Debug Level Logging
Confirms successful layer creation.
```
DEBUG Successfully created 'bert_embeddings' layer: bert_embeddings
```

#### Error Logging
Provides detailed context when layer creation fails.
```
ERROR Failed to create 'bert_embeddings' embedding layer (BertEmbeddings).
  Required params: ['vocab_size', 'hidden_size', 'max_position_embeddings', 'type_vocab_size']
  Provided params: ['vocab_size', 'hidden_size', 'max_position_embeddings']
  Check parameter compatibility and types. Use get_embedding_info() for details.
  Original error: Required parameters missing for bert_embeddings: ['type_vocab_size']. Required: ['vocab_size', 'hidden_size', 'max_position_embeddings', 'type_vocab_size']
```

## BERT Embeddings Advanced Usage

### Custom Normalization Configuration

```python
# Using different normalization types for different use cases
configs = {
    'standard_bert': {
        'type': 'bert_embeddings',
        'vocab_size': 30522,
        'hidden_size': 768,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'normalization_type': 'layer_norm',  # Standard BERT
        'layer_norm_eps': 1e-12
    },
    'efficient_bert': {
        'type': 'bert_embeddings',
        'vocab_size': 30522,
        'hidden_size': 768,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'normalization_type': 'rms_norm',  # Faster training
        'dropout_rate': 0.1
    },
    'stable_bert': {
        'type': 'bert_embeddings',
        'vocab_size': 30522,
        'hidden_size': 768,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'normalization_type': 'band_rms',  # Better stability
        'layer_norm_eps': 1e-8
    }
}

# Create different variants
embeddings = {name: create_embedding_from_config(config) 
              for name, config in configs.items()}
```

### Single Segment Models

```python
# For models that don't use segment embeddings (like GPT-style)
gpt_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=50257,  # GPT-2 vocab
    hidden_size=1024,
    max_position_embeddings=1024,
    type_vocab_size=1,  # Only one segment type
    normalization_type='layer_norm',
    dropout_rate=0.1
)
```

### Large Model Configuration

```python
# For large language models with extended context
large_bert_embed = create_embedding_layer(
    'bert_embeddings',
    vocab_size=100000,  # Large vocabulary
    hidden_size=1536,   # Large hidden size
    max_position_embeddings=4096,  # Long sequences
    type_vocab_size=8,  # Multiple segment types
    normalization_type='rms_norm',  # Efficient normalization
    initializer_range=0.01,  # Smaller init for stability
    layer_norm_eps=1e-6,
    dropout_rate=0.05  # Lower dropout for large models
)
```

## API Reference

#### Functions
- `create_embedding_layer(embedding_type, name=None, **kwargs)`: Factory function for creating embedding layers with validation.
- `create_embedding_from_config(config)`: Create an embedding layer from a configuration dictionary.
- `validate_embedding_config(embedding_type, **kwargs)`: Validate configuration parameters before creation.
- `get_embedding_info()`: Get comprehensive information about all available embedding types.

#### Types
- `EmbeddingType`: A `Literal` type defining valid embedding type strings.

#### BERT Embeddings Input/Output
- **Input shapes**:
  - `input_ids`: `(batch_size, sequence_length)` - Token IDs
  - `token_type_ids`: `(batch_size, sequence_length)` - Optional segment IDs
  - `position_ids`: `(batch_size, sequence_length)` - Optional position IDs
- **Output shape**: `(batch_size, sequence_length, hidden_size)` - Combined embeddings

#### Available Normalization Types for BERT Embeddings
- `'layer_norm'`: Standard LayerNormalization (BERT default)
- `'rms_norm'`: RMSNorm for efficiency
- `'band_rms'`: Band-constrained RMS for stability
- `'batch_norm'`: BatchNormalization (less common for transformers)