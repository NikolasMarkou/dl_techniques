# Feed-Forward Network (FFN) Module

The `dl_techniques.layers.ffn` module provides a comprehensive collection of feed-forward network implementations for deep learning architectures, with a unified factory interface for consistent layer creation and configuration management.

## Overview

This module includes nine different FFN layer types and a factory system for standardized creation and parameter validation. All layers are designed for modern Keras 3.x compatibility with full serialization support.

## Available FFN Types

### Factory-Supported Layers

The following layers are supported by the factory system with automated parameter validation and defaults:

| Type | Class | Description | Use Case |
|------|-------|-------------|----------|
| `mlp` | `MLPBlock` | Standard MLP with intermediate expansion | General purpose feed-forward processing |
| `swiglu` | `SwiGLUFFN` | SwiGLU with gating mechanism | Modern transformer architectures (LLaMa, Qwen) |
| `differential` | `DifferentialFFN` | Dual-pathway processing | Enhanced feature processing |
| `glu` | `GLUFFN` | Gated Linear Unit with configurable activation | Improved gradient flow |
| `geglu` | `GeGLUFFN` | GELU-based Gated Linear Unit | GELU-based gated processing |
| `residual` | `ResidualBlock` | Residual block with skip connections | Deep networks requiring gradient flow |
| `swin_mlp` | `SwinMLP` | Swin Transformer MLP variant | Vision models and windowed attention |

### Direct-Use Layers

These specialized layers are available for direct instantiation:

| Class | Description | Use Case |
|-------|-------------|----------|
| `LogicFFN` | Logic-based FFN with learnable soft logic operations | Reasoning tasks requiring logic operations |
| `CountingFFN` | FFN that learns to count events in sequences | Sequence processing with counting requirements |

## Factory Interface

### Basic Usage

```python
from dl_techniques.layers.ffn import create_ffn_layer

# Create standard MLP
mlp = create_ffn_layer('mlp', hidden_dim=512, output_dim=256)

# Create modern SwiGLU for transformers
swiglu = create_ffn_layer(
    'swiglu',
    d_model=768,
    ffn_expansion_factor=4,
    dropout_rate=0.1
)
```

### Configuration-Based Creation

```python
from dl_techniques.layers.ffn import create_ffn_from_config

config = {
    'type': 'differential',
    'hidden_dim': 1024,
    'output_dim': 512,
    'branch_activation': 'relu',
    'dropout_rate': 0.1,
    'name': 'diff_ffn_block'
}

ffn = create_ffn_from_config(config)
```

### Parameter Discovery

```python
from dl_techniques.layers.ffn import get_ffn_info

# Get information about all FFN types
info = get_ffn_info()

# Print requirements for specific type
mlp_info = info['mlp']
print(f"Required: {mlp_info['required_params']}")
print(f"Optional: {list(mlp_info['optional_params'].keys())}")
```

### Validation

```python
from dl_techniques.layers.ffn import validate_ffn_config

# Validate configuration before creation
try:
    validate_ffn_config('swiglu', d_model=768, ffn_expansion_factor=4)
    print("Configuration is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Layer-Specific Parameters

### MLPBlock
**Required:** `hidden_dim`, `output_dim`  
**Optional:** `activation` (default: 'gelu'), `dropout_rate` (default: 0.0), `use_bias` (default: True)

```python
mlp = create_ffn_layer(
    'mlp',
    hidden_dim=2048,
    output_dim=768,
    activation='relu',
    dropout_rate=0.1
)
```

### SwiGLUFFN
**Required:** `d_model`  
**Optional:** `ffn_expansion_factor` (default: 4), `ffn_multiple_of` (default: 256), `dropout_rate` (default: 0.0), `use_bias` (default: False)

```python
swiglu = create_ffn_layer(
    'swiglu',
    d_model=768,
    ffn_expansion_factor=8,
    ffn_multiple_of=128,
    dropout_rate=0.1
)
```

### DifferentialFFN
**Required:** `hidden_dim`, `output_dim`  
**Optional:** `branch_activation` (default: 'gelu'), `gate_activation` (default: 'sigmoid'), `dropout_rate` (default: 0.0)

```python
diff_ffn = create_ffn_layer(
    'differential',
    hidden_dim=1024,
    output_dim=768,
    branch_activation='relu',
    gate_activation='sigmoid'
)
```

### GLUFFN
**Required:** `hidden_dim`, `output_dim`  
**Optional:** `activation` (default: 'swish'), `dropout_rate` (default: 0.0), `use_bias` (default: True)

```python
glu = create_ffn_layer(
    'glu',
    hidden_dim=2048,
    output_dim=768,
    activation='sigmoid',
    dropout_rate=0.1
)
```

### GeGLUFFN
**Required:** `hidden_dim`, `output_dim`  
**Optional:** `activation` (default: 'gelu'), `dropout_rate` (default: 0.0), `use_bias` (default: True)

```python
geglu = create_ffn_layer(
    'geglu',
    hidden_dim=3072,
    output_dim=768,
    dropout_rate=0.05
)
```

### ResidualBlock
**Required:** `hidden_dim`, `output_dim`  
**Optional:** `activation` (default: 'relu'), `dropout_rate` (default: 0.0), `use_bias` (default: True)

```python
residual = create_ffn_layer(
    'residual',
    hidden_dim=1024,
    output_dim=768,
    activation='gelu',
    dropout_rate=0.2
)
```

### SwinMLP
**Required:** `hidden_dim`  
**Optional:** `output_dim` (default: None), `activation` (default: 'gelu'), `dropout_rate` (default: 0.0), `use_bias` (default: True)

```python
swin_mlp = create_ffn_layer(
    'swin_mlp',
    hidden_dim=1024,
    output_dim=768,
    dropout_rate=0.1,
    activation='gelu'
)
```

## Direct Layer Instantiation

While the factory is the recommended approach for standardized layers, direct instantiation remains available for all layer types. This can be useful for specific use cases or when bypassing the factory is desirable.

### For Factory-Supported Layers

```python
from dl_techniques.layers.ffn import MLPBlock, SwiGLUFFN, GeGLUFFN

# Direct instantiation (bypasses factory validation and defaults)
mlp = MLPBlock(hidden_dim=512, output_dim=256, activation='relu')
swiglu = SwiGLUFFN(d_model=768, ffn_expansion_factor=4)
geglu = GeGLUFFN(hidden_dim=2048, output_dim=768)
```

### For Specialized Layers

```python
from dl_techniques.layers.ffn import LogicFFN, CountingFFN

# Logic-based FFN with soft logic operations
logic_ffn = LogicFFN(
    output_dim=768,
    logic_dim=256,
    use_bias=True
)

# FFN that learns to count events in sequences
counting_ffn = CountingFFN(
    output_dim=512,
    count_dim=128,
    counting_scope='local'  # 'global', 'local', or 'causal'
)
```

## Integration Patterns

### In Custom Transformer Layers

```python
@keras.saving.register_keras_serializable()
class CustomTransformerLayer(keras.layers.Layer):
    def __init__(self, hidden_size, ffn_type='mlp', **ffn_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.ffn_type = ffn_type
        self.ffn_kwargs = ffn_kwargs
        
        # Create FFN using factory
        from dl_techniques.layers.ffn import create_ffn_layer
        self.ffn = create_ffn_layer(
            ffn_type,
            hidden_dim=hidden_size * 4,
            output_dim=hidden_size,
            name='ffn',
            **ffn_kwargs
        )
    
    def call(self, inputs):
        return self.ffn(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'ffn_type': self.ffn_type,
            'ffn_kwargs': self.ffn_kwargs
        })
        return config
```

### In Model Builders

```python
def create_transformer_model(config):
    """Create transformer model with configurable FFN."""
    from dl_techniques.layers.ffn import create_ffn_layer
    
    ffn_config = config.get('ffn', {'type': 'mlp'})
    ffn_type = ffn_config.pop('type')
    
    # Create FFN layer
    ffn = create_ffn_layer(
        ffn_type,
        hidden_dim=config['hidden_size'] * 4,
        output_dim=config['hidden_size'],
        **ffn_config
    )
    
    return ffn
```

### With Configuration Files

```python
import json
from dl_techniques.layers.ffn import create_ffn_from_config

# Load configuration from file
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Create FFN from configuration
ffn_config = config['ffn']
ffn = create_ffn_from_config(ffn_config)
```

Example configuration file:
```json
{
    "ffn": {
        "type": "swiglu",
        "d_model": 768,
        "ffn_expansion_factor": 4,
        "dropout_rate": 0.1,
        "name": "transformer_ffn"
    }
}
```

## Parameter Validation

The factory performs comprehensive validation:

### Required Parameter Checking
```python
# Will raise ValueError for missing required parameters
create_ffn_layer('mlp', hidden_dim=512)  # Missing output_dim
```

### Value Range Validation
```python
# Will raise ValueError for invalid ranges
create_ffn_layer('mlp', hidden_dim=-100, output_dim=256)  # Negative hidden_dim
create_ffn_layer('swiglu', d_model=768, dropout_rate=1.5)  # Invalid dropout rate
```

### Activation Function Validation
```python
# Will raise ValueError for unknown activation functions
create_ffn_layer('mlp', hidden_dim=512, output_dim=256, activation='unknown_activation')
```

## Logging and Debugging

The factory provides detailed logging for debugging:

### Info Level Logging
Shows all parameters passed to each layer:
```
INFO Creating swiglu FFN layer with parameters:
INFO   d_model: 768
INFO   dropout_rate: 0.1
INFO   ffn_expansion_factor: 4
INFO   name: 'ffn_layer'
INFO   use_bias: True
```

### Debug Level Logging
Shows layer creation success:
```
DEBUG Successfully created swiglu FFN layer: ffn_layer
```

### Error Logging
Detailed error context for failed layer creation:
```
ERROR Failed to create mlp FFN layer (MLPBlock). Required parameters: ['hidden_dim', 'output_dim']. 
      Provided parameters: ['hidden_dim']. Check parameter compatibility and types.
```

## Best Practices

### 1. Use Factory for Standard Layers
```python
# Preferred: Use factory for validation and consistency
ffn = create_ffn_layer('mlp', hidden_dim=512, output_dim=256)

# Avoid: Direct instantiation for factory-supported layers
ffn = MLPBlock(hidden_dim=512, output_dim=256)
```

### 2. Use Direct Instantiation for Specialized Layers
```python
# Correct: Direct instantiation for specialized layers
logic_ffn = LogicFFN(output_dim=768, logic_dim=256)
counting_ffn = CountingFFN(output_dim=512, count_dim=128)
```

### 3. Validate Configurations in Production
```python
def create_production_ffn(ffn_type, **params):
    """Create FFN with production-grade validation."""
    try:
        validate_ffn_config(ffn_type, **params)
        return create_ffn_layer(ffn_type, **params)
    except ValueError as e:
        logger.error(f"FFN creation failed: {e}")
        # Fallback to safe default
        return create_ffn_layer('mlp', hidden_dim=512, output_dim=256)
```

### 4. Store Configurations for Reproducibility
```python
# Store FFN configurations for experiment tracking
ffn_configs = {
    'baseline': {'type': 'mlp', 'hidden_dim': 2048, 'output_dim': 768},
    'modern': {'type': 'swiglu', 'd_model': 768, 'ffn_expansion_factor': 4},
    'efficient': {'type': 'glu', 'hidden_dim': 1024, 'output_dim': 768}
}

# Create layers from stored configurations
ffn_layers = {name: create_ffn_from_config(config) 
              for name, config in ffn_configs.items()}
```

## Error Handling

### Common Validation Errors

**Missing Required Parameters:**
```python
# Error: Missing output_dim for mlp
create_ffn_layer('mlp', hidden_dim=512)

# Fix: Provide all required parameters
create_ffn_layer('mlp', hidden_dim=512, output_dim=256)
```

**Invalid Parameter Values:**
```python
# Error: Negative dimension
create_ffn_layer('mlp', hidden_dim=-100, output_dim=256)

# Fix: Use positive values
create_ffn_layer('mlp', hidden_dim=512, output_dim=256)
```

**Unknown FFN Type:**
```python
# Error: Unsupported type
create_ffn_layer('unknown_type', hidden_dim=512)

# Fix: Use supported type
create_ffn_layer('mlp', hidden_dim=512, output_dim=256)
```

## Advanced Usage

### Custom Parameter Override

```python
# Override default parameters
ffn = create_ffn_layer(
    'mlp',
    hidden_dim=1024,
    output_dim=768,
    activation='swish',          # Override default 'gelu'
    dropout_rate=0.2,            # Override default 0.0
    kernel_initializer='he_normal'  # Override default 'glorot_uniform'
)
```

### Dynamic FFN Selection

```python
def create_adaptive_ffn(model_size, efficiency_priority=False):
    """Create FFN based on model requirements."""
    if efficiency_priority:
        return create_ffn_layer('glu', hidden_dim=512, output_dim=256)
    elif model_size == 'large':
        return create_ffn_layer('swiglu', d_model=1024, ffn_expansion_factor=8)
    else:
        return create_ffn_layer('mlp', hidden_dim=2048, output_dim=768)
```

### Multi-FFN Architectures

```python
class MultiFFNLayer(keras.layers.Layer):
    """Layer using multiple FFN types in parallel."""
    
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
        # Create multiple FFN branches
        self.ffn_standard = create_ffn_layer('mlp', hidden_dim=d_model*4, output_dim=d_model)
        self.ffn_gated = create_ffn_layer('geglu', hidden_dim=d_model*4, output_dim=d_model)
        self.ffn_residual = create_ffn_layer('residual', hidden_dim=d_model*2, output_dim=d_model)
        
        self.output_projection = keras.layers.Dense(d_model)
    
    def call(self, inputs):
        # Process through all FFN branches
        out1 = self.ffn_standard(inputs)
        out2 = self.ffn_gated(inputs)
        out3 = self.ffn_residual(inputs)
        
        # Combine outputs
        combined = keras.ops.concatenate([out1, out2, out3], axis=-1)
        return self.output_projection(combined)
```

## API Reference

### Functions

#### `create_ffn_layer(ffn_type, name=None, **kwargs)`
Factory function for creating FFN layers with validation.

#### `create_ffn_from_config(config)`
Create FFN layer from configuration dictionary.

#### `validate_ffn_config(ffn_type, **kwargs)`
Validate FFN configuration parameters before creation.

#### `get_ffn_info()`
Get comprehensive information about all available FFN types.

### Types

#### `FFNType`
Literal type defining valid FFN type strings: `'mlp'`, `'swiglu'`, `'differential'`, `'glu'`, `'geglu'`, `'residual'`, `'swin_mlp'`.

## Migration Guide

### From Direct Instantiation

**Before:**
```python
# Manual parameter handling and validation
if ffn_type == 'mlp':
    ffn = MLPBlock(hidden_dim=hidden_dim, output_dim=output_dim, activation=activation)
elif ffn_type == 'swiglu':
    ffn = SwiGLUFFN(d_model=d_model, ffn_expansion_factor=expansion)
# ... many more cases
```

**After:**
```python
# Unified factory interface
ffn = create_ffn_layer(ffn_type, **params)
```

### Benefits of Migration

1. **Reduced Code**: Eliminate repetitive if/elif chains
2. **Better Validation**: Automatic parameter validation and error handling
3. **Consistency**: Standardized parameter handling across all FFN types
4. **Maintainability**: Central location for FFN creation logic
5. **Type Safety**: Full type hints and IDE support

## Troubleshooting

### Enable Debug Logging
```python
import logging
logging.getLogger("dl").setLevel(logging.DEBUG)
```

### Common Issues

**TypeError during layer creation:**
- Check parameter names match layer's expected interface
- Verify parameter types (int, float, str, bool)
- Use `get_ffn_info()` to see expected parameters

**ValueError for unknown activation:**
- Use standard Keras activation names: 'relu', 'gelu', 'swish', 'tanh', etc.
- Or pass callable activation functions directly

**Missing specialized layers:**
- `LogicFFN` and `CountingFFN` must be imported and used directly
- These are not included in the factory registry system