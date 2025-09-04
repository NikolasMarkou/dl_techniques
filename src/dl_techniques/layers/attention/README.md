# Attention Module

The `dl_techniques.layers.attention` module provides a comprehensive collection of attention mechanisms for deep learning, with a unified factory interface for consistent layer creation, configuration management, and parameter validation.

## Overview

This module includes seventeen different attention layer types, ranging from standard multi-head attention to specialized variants for vision, efficiency, and advanced modeling. All layers are built using Keras 3 for backend-agnostic compatibility and support full serialization. The factory system ensures a standardized, safe, and introspectable way to integrate any of these attention mechanisms into your models.

## Available Attention Types

The following layers are supported by the factory system with automated parameter validation and defaults:

| Type | Class | Description | Use Case |
|------|-------|-------------|----------|
| `adaptive_multi_head` | `AdaptiveMultiHeadAttention` | MHA with adaptive temperature softmax. | Transformers where attention sharpness needs to adapt to sequence length. |
| `anchor` | `AnchorAttention` | Hierarchical attention with anchor tokens. | Long-sequence models where full self-attention is too costly. |
| `attention_block` | `AttentionBlock` | Vision-specific attention block for 4D tensors. | Vision transformers and hybrid CNN-transformer models. |
| `capsule_routing` | `CapsuleRoutingSelfAttention` | Self-attention with capsule network dynamic routing. | Experimental models aiming for better contextualization. |
| `channel` | `ChannelAttention` | Channel attention module from CBAM. | CNNs to recalibrate channel-wise feature responses. |
| `cbam` | `CBAM` | Convolutional Block Attention Module (Channel + Spatial). | Plug-and-play attention module for any CNN to refine features. |
| `perceiver` | `PerceiverAttention` | Cross-attention from the Perceiver architecture. | Cross-modal attention (e.g., text to image) and latent bottleneck models. |
| `differential` | `DifferentialMultiHeadAttention` | Dual MHA to amplify signal and cancel noise. | Transformers requiring improved focus and reduced hallucination. |
| `fnet` | `FNetFourierTransform` | Parameter-free token mixing with Fourier Transforms. | Efficient replacement for self-attention in sequence models. |
| `group_query` | `GroupedQueryAttention` | GQA with shared K/V heads for efficiency. | Large language models where K/V cache size is a bottleneck. |
| `hopfield` | `HopfieldAttention` | Modern Hopfield Network for pattern retrieval. | Associative memory tasks; mimics standard attention with `update_steps_max=0`. |
| `mobile_mqa` | `MobileMQA` | Mobile-optimized Multi-Query Attention for vision. | Efficient attention in vision models for mobile and edge devices. |
| `multi_head` | `MultiHeadAttention` | Standard Multi-Head Self-Attention. | General-purpose self-attention in vision and sequence models. |
| `non_local` | `NonLocalAttention` | Non-local attention for capturing long-range dependencies in CNNs. | Augmenting CNNs with global context reasoning. |
| `shared_weights_cross` | `SharedWeightsCrossAttention`| Cross-attention between modalities with shared weights. | Efficient multi-modal learning where different data types exchange information. |
| `spatial` | `SpatialAttention` | Spatial attention module from CBAM. | CNNs to highlight spatially significant feature regions. |
| `window` | `WindowAttention` | Windowed Multi-Head Attention from Swin Transformer. | Vision transformers (e.g., Swin) for efficient local attention. |

## Factory Interface

### Basic Usage

```python
from dl_techniques.layers.attention import create_attention_layer

# Create a standard multi-head attention layer
mha = create_attention_layer('multi_head', embed_dim=256, num_heads=8)

# Create a CBAM block for a CNN
cbam = create_attention_layer('cbam', channels=128, ratio=16)
```

### Configuration-Based Creation

```python
from dl_techniques.layers.attention import create_attention_from_config

config = {
    'type': 'group_query',
    'd_model': 1024,
    'n_head': 16,
    'n_kv_head': 4,
    'name': 'gqa_block_1'
}

gqa_layer = create_attention_from_config(config)
```

### Parameter Discovery

```python
from dl_techniques.layers.attention import get_attention_info

# Get information about all attention types
info = get_attention_info()

# Print requirements for a specific type
gqa_info = info['group_query']
print(f"Required: {gqa_info['required_params']}")
print(f"Optional: {list(gqa_info['optional_params'].keys())}")
```

### Validation

```python
from dl_techniques.layers.attention import validate_attention_config

# Validate configuration before creation
try:
    validate_attention_config('window', dim=96, window_size=7, num_heads=4)
    print("Configuration is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Layer-Specific Parameters

### `adaptive_multi_head`
**Required:** `num_heads`, `key_dim`  
**Optional:** `dropout` (default: 0.0), `min_temp` (default: 0.1), `max_temp` (default: 1.0)
```python
attn = create_attention_layer(
    'adaptive_multi_head',
    num_heads=8,
    key_dim=64,
    dropout=0.1,
    min_temp=0.05
)
```

### `anchor`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `dropout` (default: 0.0)
```python
attn = create_attention_layer(
    'anchor',
    dim=512,
    num_heads=8,
    dropout=0.1
)
```

### `attention_block`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `attention_type` (default: 'multi_head_attention'), `use_layer_scale` (default: True)
```python
attn = create_attention_layer(
    'attention_block',
    dim=256,
    num_heads=4,
    attention_type='window_attention'
)
```

### `capsule_routing`
**Required:** `num_heads`  
**Optional:** `key_dim` (default: None), `routing_iterations` (default: 3)
```python
attn = create_attention_layer(
    'capsule_routing',
    num_heads=8,
    key_dim=64,
    routing_iterations=5
)
```

### `channel`
**Required:** `channels`  
**Optional:** `ratio` (default: 8), `use_bias` (default: False)
```python
attn = create_attention_layer(
    'channel',
    channels=256,
    ratio=16
)
```

### `cbam`
**Required:** `channels`  
**Optional:** `ratio` (default: 8), `kernel_size` (default: 7)
```python
attn = create_attention_layer(
    'cbam',
    channels=256,
    ratio=16,
    kernel_size=5
)
```

### `perceiver`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `dropout` (default: 0.0)
```python
attn = create_attention_layer(
    'perceiver',
    dim=256,
    num_heads=8,
    dropout=0.1
)
```

### `differential`
**Required:** `dim`, `num_heads`, `head_dim`  
**Optional:** `dropout` (default: 0.0), `attention_dropout` (default: 0.0), `lambda_init` (default: 0.8)
```python
attn = create_attention_layer(
    'differential',
    dim=512,
    num_heads=8,
    head_dim=64,
    attention_dropout=0.1
)
```

### `fnet`
**Required:** None  
**Optional:** `implementation` (default: 'matrix'), `normalize_dft` (default: True)
```python
attn = create_attention_layer(
    'fnet',
    implementation='fft'
)
```

### `group_query`
**Required:** `d_model`, `n_head`, `n_kv_head`  
**Optional:** `max_seq_len` (default: 2048), `dropout_rate` (default: 0.0)
```python
attn = create_attention_layer(
    'group_query',
    d_model=1024,
    n_head=16,
    n_kv_head=4
)
```

### `hopfield`
**Required:** `num_heads`, `key_dim`  
**Optional:** `update_steps_max` (default: 0), `update_steps_eps` (default: 1e-4)
```python
attn = create_attention_layer(
    'hopfield',
    num_heads=8,
    key_dim=64,
    update_steps_max=3
)
```

### `mobile_mqa`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `use_downsampling` (default: False)
```python
attn = create_attention_layer(
    'mobile_mqa',
    dim=256,
    num_heads=8,
    use_downsampling=True
)
```

### `multi_head`
**Required:** `embed_dim`  
**Optional:** `num_heads` (default: 8), `dropout_rate` (default: 0.0)
```python
attn = create_attention_layer(
    'multi_head',
    embed_dim=512,
    num_heads=8,
    dropout_rate=0.1
)
```

### `non_local`
**Required:** `attention_channels`  
**Optional:** `normalization` (default: 'batch'), `attention_mode` (default: 'gaussian')
```python
attn = create_attention_layer(
    'non_local',
    attention_channels=128,
    normalization='layer',
    attention_mode='dot_product'
)
```

### `shared_weights_cross`
**Required:** `dim`  
**Optional:** `num_heads` (default: 8), `dropout_rate` (default: 0.0)
```python
attn = create_attention_layer(
    'shared_weights_cross',
    dim=256,
    num_heads=4,
    dropout_rate=0.1
)
```

### `spatial`
**Required:** None  
**Optional:** `kernel_size` (default: 7), `use_bias` (default: True)
```python
attn = create_attention_layer(
    'spatial',
    kernel_size=5
)
```

### `window`
**Required:** `dim`, `window_size`, `num_heads`  
**Optional:** `attn_dropout_rate` (default: 0.0), `qkv_bias` (default: True)
```python
attn = create_attention_layer(
    'window',
    dim=96,
    window_size=7,
    num_heads=4,
    attn_dropout_rate=0.05
)
```

## Direct Layer Instantiation

While the factory is recommended, direct instantiation is always available.

```python
from dl_techniques.layers.attention import MultiHeadAttention, CBAM, WindowAttention

# Direct instantiation (bypasses factory validation and defaults)
mha = MultiHeadAttention(embed_dim=512, num_heads=8)
cbam = CBAM(channels=256, ratio=16)
window_attn = WindowAttention(dim=96, window_size=7, num_heads=4)
```

## Integration Patterns

### In a Custom Transformer Block

```python
@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, attention_type='multi_head', **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.attention_type = attention_type
        
        from dl_techniques.layers.attention import create_attention_layer
        
        # Create attention using the factory
        # Map generic 'dim' to layer-specific parameter name
        attn_params = {'num_heads': num_heads}
        if attention_type in ['multi_head']:
            attn_params['embed_dim'] = dim
        else:
            attn_params['dim'] = dim

        self.attn = create_attention_layer(attention_type, name='attention', **attn_params)
        # ... other layers like FFN, LayerNorm
    
    def call(self, inputs):
        x = self.attn(inputs)
        # ... rest of the block
        return x
```

### In Model Builders with Configuration Files

```python
import json
from dl_techniques.layers.attention import create_attention_from_config

# Load configuration from file
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Create attention layer from the 'attention' section of the config
attention_layer = create_attention_from_config(config['attention'])
```

**Example `model_config.json`:**
```json
{
    "model_name": "VisionTransformer-S",
    "attention": {
        "type": "attention_block",
        "dim": 384,
        "num_heads": 6,
        "dropout_rate": 0.1
    },
    "ffn": { "...ffn_config..." }
}
```

## Parameter Validation

The factory performs comprehensive validation on layer creation.

**Missing Required Parameters:**
```python
# Raises ValueError: "Required parameters for 'group_query' are missing: ['n_kv_head']"
create_attention_layer('group_query', d_model=512, n_head=8)
```

**Invalid Value Ranges:**
```python
# Raises ValueError: "Parameter 'num_heads' must be positive"
create_attention_layer('multi_head', embed_dim=256, num_heads=-8)

# Raises ValueError: "Parameter 'dropout_rate' must be between 0.0 and 1.0"
create_attention_layer('multi_head', embed_dim=256, dropout_rate=1.5)
```

**Unknown Attention Type:**
```python
# Raises ValueError: "Unknown attention type 'vanilla_attention'"
create_attention_layer('vanilla_attention', dim=512)
```

## Logging and Debugging

The factory provides detailed logging to aid development.

**INFO Level:** Shows parameters used for layer creation.
```
INFO Creating 'group_query' layer with parameters: {'d_model': 1024, 'n_head': 16, 'n_kv_head': 4, 'name': 'gqa_block_1', ...}
```

**ERROR Level:** Provides context for failed layer creation.
```
ERROR Failed to create 'group_query' layer (GroupedQueryAttention). Required: ['d_model', 'n_head', 'n_kv_head']. Provided: ['d_model', 'n_head']. Please check parameter compatibility. Original error: ...
```

## API Reference

### Functions

-   **`create_attention_layer(attention_type, name=None, **kwargs)`**: Factory for creating attention layers with validation.
-   **`create_attention_from_config(config)`**: Creates a layer from a configuration dictionary.
-   **`validate_attention_config(attention_type, **kwargs)`**: Validates parameters before creation.
-   **`get_attention_info()`**: Returns a dictionary with details about all available attention types.

### Types

-   **`AttentionType`**: A `Literal` type defining all valid attention type strings.