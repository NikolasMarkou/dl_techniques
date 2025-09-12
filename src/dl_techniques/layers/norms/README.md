# Normalization Factory Utility Guide

## Overview

The Normalization Factory is a centralized utility for creating normalization layers in the dl_techniques framework. It provides a unified interface for accessing all available normalization techniques, making it easy to experiment with different approaches and maintain consistent code patterns.

The factory method is available at: `dl_techniques.layers.norms.create_normalization_layer`

## Key Benefits

1. **Unified Interface**: Single function call to create any supported normalization layer
2. **Type Safety**: Full type hints and validation for all parameters
3. **Comprehensive Coverage**: Supports all normalization layers in dl_techniques
4. **Easy Experimentation**: Switch between normalization types by changing a single parameter
5. **Consistent Configuration**: Standardized parameter handling across all layer types
6. **Robust Validation**: Built-in parameter validation and error handling

## Supported Normalization Types

### Standard Keras Layers
- `layer_norm`: LayerNormalization - Standard normalization with learnable scale and bias
- `batch_norm`: BatchNormalization - Batch-based normalization with moving statistics

### dl_techniques Specialized Layers
- `rms_norm`: Root Mean Square normalization without centering
- `zero_centered_rms_norm`: Zero-centered RMS normalization combining RMSNorm efficiency with LayerNorm stability
- `band_rms`: RMS normalization with bounded magnitude constraints
- `adaptive_band_rms`: Adaptive RMS with log-transformed scaling
- `band_logit_norm`: Band-constrained logit normalization for classification
- `global_response_norm`: Global Response Normalization from ConvNeXt
- `logit_norm`: Temperature-scaled normalization for classification
- `max_logit_norm`: MaxLogit normalization for out-of-distribution detection
- `decoupled_max_logit`: Decoupled MaxLogit (DML) with constant decoupling
- `dml_plus_focal`: DML+ focal model variant
- `dml_plus_center`: DML+ center model variant
- `dynamic_tanh`: Dynamic Tanh normalization for normalization-free transformers

## Basic Usage

### Simple Layer Creation

```python
from dl_techniques.layers.norms import create_normalization_layer

# Create a standard layer normalization
layer_norm = create_normalization_layer('layer_norm', name='my_norm')

# Create RMS normalization with custom epsilon
rms_norm = create_normalization_layer('rms_norm', epsilon=1e-8, use_scale=True)

# Create Zero-Centered RMS normalization for enhanced stability
zero_centered_rms = create_normalization_layer(
    'zero_centered_rms_norm', 
    epsilon=1e-5, 
    use_scale=True
)

# Create Band RMS with constraints
band_rms = create_normalization_layer(
    'band_rms', 
    max_band_width=0.1,
    epsilon=1e-7
)
```

### Integration with Transformer Layers

```python
@keras.saving.register_keras_serializable()
class MyTransformerLayer(keras.layers.Layer):
    def __init__(self, normalization_type='layer_norm', **norm_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.normalization_type = normalization_type
        self.norm_kwargs = norm_kwargs
        
        # Create normalization layers using the factory
        self.attention_norm = self._create_normalization_layer('attention_norm')
        self.ffn_norm = self._create_normalization_layer('ffn_norm')

    def _create_normalization_layer(self, name):
        return create_normalization_layer(
            normalization_type=self.normalization_type,
            name=name,
            **self.norm_kwargs
        )
```

## Advanced Configuration

### Layer-Specific Parameters

Each normalization type supports specific parameters. Use kwargs to pass them:

```python
# Standard RMS Norm with custom axis and scaling
rms_layer = create_normalization_layer(
    'rms_norm',
    axis=(-2, -1),  # Normalize over multiple axes
    use_scale=True,
    epsilon=1e-6
)

# Zero-Centered RMS Norm for large language models
zero_centered_layer = create_normalization_layer(
    'zero_centered_rms_norm',
    axis=-1,
    use_scale=True,
    epsilon=1e-5,  # Slightly larger for stability
    scale_initializer='ones'
)

# Band RMS with tight constraints
band_layer = create_normalization_layer(
    'band_rms',
    max_band_width=0.05,  # Tighter constraint
    axis=-1,
    epsilon=1e-8
)

# Global Response Normalization
grn_layer = create_normalization_layer(
    'global_response_norm',
    eps=1e-6,  # Note: GRN uses 'eps' not 'epsilon'
    gamma_initializer='ones',
    beta_initializer='zeros'
)

# Dynamic Tanh for normalization-free transformers
dyt_layer = create_normalization_layer(
    'dynamic_tanh',
    alpha_init_value=0.5,
    axis=[-1]
)
```

### Parameter Validation

Use the validation function to check configurations before creating layers:

```python
from dl_techniques.layers.norms import validate_normalization_config

# Validate configuration
try:
    validate_normalization_config(
        'zero_centered_rms_norm',
        axis=-1,
        use_scale=True,
        epsilon=1e-5
    )
    print("Configuration is valid")
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## Information and Discovery

### Get Available Normalization Types

```python
from dl_techniques.layers.norms import get_normalization_info

# Get information about all supported types
info = get_normalization_info()

# Print available types
for norm_type, details in info.items():
    print(f"{norm_type}: {details['description']}")
    print(f"  Parameters: {details['parameters']}")
    print(f"  Use case: {details['use_case']}")
    print()
```

### Type Hints and IDE Support

The utility provides full type hints for better IDE support:

```python
from dl_techniques.layers.norms import NormalizationType

def create_model_with_norm(norm_type: NormalizationType):
    """Function with type-safe normalization selection."""
    return create_normalization_layer(norm_type)

# IDE will suggest valid normalization types
layer = create_model_with_norm('rms_norm')  # ✓ Valid
layer = create_model_with_norm('zero_centered_rms_norm')  # ✓ Valid
layer = create_model_with_norm('invalid')   # ✗ Type error
```

## Common Usage Patterns

### 1. Configurable Model Architecture

```python
class ConfigurableModel(keras.Model):
    def __init__(self, normalization_type='layer_norm', **norm_kwargs):
        super().__init__()
        self.norm_type = normalization_type
        self.norm_kwargs = norm_kwargs
        
        # Create layers with configurable normalization
        self.dense1 = keras.layers.Dense(512)
        self.norm1 = create_normalization_layer(normalization_type, **norm_kwargs)
        
        self.dense2 = keras.layers.Dense(256)
        self.norm2 = create_normalization_layer(normalization_type, **norm_kwargs)
        
        self.output_layer = keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.norm1(x)
        x = keras.activations.relu(x)
        
        x = self.dense2(x)
        x = self.norm2(x)
        x = keras.activations.relu(x)
        
        return self.output_layer(x)

# Easy to experiment with different normalizations
model1 = ConfigurableModel('layer_norm')
model2 = ConfigurableModel('rms_norm', use_scale=True)
model3 = ConfigurableModel('zero_centered_rms_norm', epsilon=1e-5)
model4 = ConfigurableModel('band_rms', max_band_width=0.1)
```

### 2. Large Language Model Normalization

```python
class LLMBlock(keras.layers.Layer):
    """Language model block with advanced normalization options."""
    
    def __init__(
        self, 
        hidden_size=768,
        normalization_type='zero_centered_rms_norm',  # Default to enhanced stability
        **norm_kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Use zero-centered RMS norm for better stability in large models
        self.attention_norm = create_normalization_layer(
            normalization_type,
            name='attention_norm',
            **norm_kwargs
        )
        
        self.ffn_norm = create_normalization_layer(
            normalization_type,
            name='ffn_norm', 
            **norm_kwargs
        )
        
        # Other layers...
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=12, key_dim=64
        )
        self.ffn = keras.layers.Dense(hidden_size * 4, activation='gelu')
        self.output_proj = keras.layers.Dense(hidden_size)
    
    def call(self, inputs):
        # Pre-normalization pattern
        norm_inputs = self.attention_norm(inputs)
        attn_out = self.attention(norm_inputs, norm_inputs)
        x = inputs + attn_out
        
        norm_x = self.ffn_norm(x)
        ffn_out = self.output_proj(self.ffn(norm_x))
        return x + ffn_out

# Create LLM with different normalization strategies
stable_llm = LLMBlock(normalization_type='zero_centered_rms_norm', epsilon=1e-5)
fast_llm = LLMBlock(normalization_type='rms_norm', use_scale=True)
standard_llm = LLMBlock(normalization_type='layer_norm')
```

### 3. Hyperparameter Sweeps

```python
def create_model_variants():
    """Create multiple model variants for comparison."""
    
    normalizations = [
        ('layer_norm', {}),
        ('rms_norm', {'use_scale': True}),
        ('zero_centered_rms_norm', {'epsilon': 1e-5, 'use_scale': True}),
        ('band_rms', {'max_band_width': 0.1}),
        ('global_response_norm', {}),
        ('dynamic_tanh', {'alpha_init_value': 0.5})
    ]
    
    models = {}
    for norm_type, kwargs in normalizations:
        models[norm_type] = ConfigurableModel(norm_type, **kwargs)
    
    return models

# Create all variants for comparison
model_variants = create_model_variants()
```

### 4. Layer Factory Pattern

```python
class LayerFactory:
    """Factory for creating standardized layer combinations."""
    
    @staticmethod
    def create_norm_dense_block(
        units: int,
        normalization_type: str = 'zero_centered_rms_norm',  # Default to enhanced stability
        activation: str = 'relu',
        **norm_kwargs
    ):
        """Create a normalized dense block."""
        
        class NormDenseBlock(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dense = keras.layers.Dense(units)
                self.norm = create_normalization_layer(normalization_type, **norm_kwargs)
                self.activation = keras.layers.Activation(activation)
            
            def call(self, inputs):
                x = self.dense(inputs)
                x = self.norm(x)
                return self.activation(x)
        
        return NormDenseBlock()

# Use factory to create standardized blocks
stable_block = LayerFactory.create_norm_dense_block(256, 'zero_centered_rms_norm')
fast_block = LayerFactory.create_norm_dense_block(256, 'rms_norm')
constrained_block = LayerFactory.create_norm_dense_block(128, 'band_rms', max_band_width=0.1)
```

### 5. Configuration-Based Creation

```python
from dl_techniques.layers.norms import create_normalization_from_config

# Configuration dictionary approach
normalization_configs = {
    'stable_llm': {
        'type': 'zero_centered_rms_norm',
        'epsilon': 1e-5,
        'use_scale': True,
        'axis': -1
    },
    'fast_training': {
        'type': 'rms_norm',
        'use_scale': True,
        'epsilon': 1e-6
    },
    'constrained': {
        'type': 'band_rms',
        'max_band_width': 0.1,
        'epsilon': 1e-7
    }
}

def create_model_from_config(config_name):
    config = normalization_configs[config_name]
    norm_layer = create_normalization_from_config(config)
    return norm_layer

# Easy configuration switching
stable_norm = create_model_from_config('stable_llm')
fast_norm = create_model_from_config('fast_training')
```

## Best Practices

### 1. Configuration Management

Store normalization configurations in dictionaries for easy management:

```python
NORMALIZATION_CONFIGS = {
    'standard': {'type': 'layer_norm', 'epsilon': 1e-6},
    'fast': {'type': 'rms_norm', 'use_scale': True},
    'stable_llm': {'type': 'zero_centered_rms_norm', 'epsilon': 1e-5, 'use_scale': True},
    'constrained': {'type': 'band_rms', 'max_band_width': 0.1},
    'efficient': {'type': 'global_response_norm'},
    'normfree': {'type': 'dynamic_tanh', 'alpha_init_value': 0.5}
}

def create_model(config_name='standard'):
    config = NORMALIZATION_CONFIGS[config_name].copy()
    norm_type = config.pop('type')
    return create_normalization_layer(norm_type, **config)
```

### 2. Model-Specific Defaults

Choose appropriate defaults for different model types:

```python
# Transformer/LLM models - prioritize stability
def create_transformer_norm(**kwargs):
    defaults = {
        'type': 'zero_centered_rms_norm',
        'epsilon': 1e-5,
        'use_scale': True
    }
    defaults.update(kwargs)
    norm_type = defaults.pop('type')
    return create_normalization_layer(norm_type, **defaults)

# Vision models - prioritize efficiency
def create_vision_norm(**kwargs):
    defaults = {
        'type': 'global_response_norm',
        'eps': 1e-6
    }
    defaults.update(kwargs)
    norm_type = defaults.pop('type')
    return create_normalization_layer(norm_type, **defaults)

# Fast inference models - prioritize speed
def create_fast_norm(**kwargs):
    defaults = {
        'type': 'rms_norm',
        'use_scale': True,
        'epsilon': 1e-6
    }
    defaults.update(kwargs)
    norm_type = defaults.pop('type')
    return create_normalization_layer(norm_type, **defaults)
```

### 3. Error Handling

Always validate configurations in production code:

```python
def safe_create_normalization_layer(norm_type, **kwargs):
    """Safely create normalization layer with validation."""
    try:
        validate_normalization_config(norm_type, **kwargs)
        return create_normalization_layer(norm_type, **kwargs)
    except ValueError as e:
        logger.error(f"Invalid normalization config: {e}")
        # Fallback to safe default
        return create_normalization_layer('layer_norm')
```

### 4. Documentation

Document the normalization choices in your models:

```python
class DocumentedModel(keras.Model):
    """
    Model with configurable normalization.
    
    Args:
        normalization_type: Type of normalization to use. Options:
            - 'layer_norm': Standard normalization (default)
            - 'rms_norm': Faster RMS-based normalization
            - 'zero_centered_rms_norm': Enhanced stability RMS normalization (recommended for LLMs)
            - 'band_rms': Constrained RMS for stability
            - See get_normalization_info() for full list
    """
    
    def __init__(self, normalization_type='layer_norm', **kwargs):
        super().__init__()
        # Implementation...
```

## Migration Guide

### From Manual Layer Creation

**Before:**
```python
def _create_normalization_layer(self, name):
    if self.normalization_type == 'layer_norm':
        return keras.layers.LayerNormalization(epsilon=self.epsilon, name=name)
    elif self.normalization_type == 'rms_norm':
        return RMSNorm(epsilon=self.epsilon, name=name)
    elif self.normalization_type == 'zero_centered_rms_norm':
        return ZeroCenteredRMSNorm(epsilon=self.epsilon, name=name)
    # ... many more elif statements
    else:
        raise ValueError(f"Unknown type: {self.normalization_type}")
```

**After:**
```python
def _create_normalization_layer(self, name):
    return create_normalization_layer(
        normalization_type=self.normalization_type,
        name=name,
        epsilon=self.epsilon,
        **self.norm_kwargs
    )
```

### Benefits of Migration

1. **Reduced Code**: Eliminate repetitive layer creation logic
2. **Better Coverage**: Access to all dl_techniques normalization layers including Zero-Centered RMSNorm
3. **Type Safety**: Better IDE support and error checking
4. **Consistency**: Standardized parameter handling
5. **Maintainability**: Central location for normalization logic updates
6. **Enhanced Stability**: Easy access to advanced normalization techniques

### Recommended Migration Path for LLMs

For existing large language models, consider migrating to Zero-Centered RMSNorm:

```python
# Old approach - standard RMS norm
old_norm = RMSNorm(epsilon=1e-6, use_scale=True)

# New approach - enhanced stability
new_norm = create_normalization_layer(
    'zero_centered_rms_norm',
    epsilon=1e-5,  # Slightly larger for stability
    use_scale=True
)
```

## Troubleshooting

### Common Issues

1. **Invalid Parameter Error**: Check parameter names using `get_normalization_info()`
2. **Type Error**: Ensure normalization_type is one of the supported values
3. **Import Error**: Verify all dl_techniques normalization modules are available
4. **Stability Issues**: Consider using 'zero_centered_rms_norm' for better training stability

### Debug Example

```python
# Debug normalization layer creation
try:
    layer = create_normalization_layer('zero_centered_rms_norm', use_scale=True)
    print("Layer created successfully")
except Exception as e:
    print(f"Error: {e}")
    
    # Get valid parameters for debugging
    info = get_normalization_info()
    print(f"Valid parameters: {info['zero_centered_rms_norm']['parameters']}")
    
    # Try with minimal configuration
    fallback_layer = create_normalization_layer('zero_centered_rms_norm')
    print("Fallback layer created")
```

### Performance Considerations

Different normalization types have different computational costs:

```python
# Performance ranking (approximate, from fastest to slowest)
PERFORMANCE_RANKING = [
    'rms_norm',                    # Fastest - no mean computation
    'zero_centered_rms_norm',      # Fast - single mean computation  
    'dynamic_tanh',                # Fast - simple operations
    'layer_norm',                  # Standard - mean and variance
    'band_rms',                    # Moderate - constrained operations
    'global_response_norm',        # Moderate - additional operations
    'batch_norm'                   # Variable - depends on batch size
]

def choose_normalization_by_performance(priority='balanced'):
    """Choose normalization based on performance requirements."""
    if priority == 'speed':
        return 'rms_norm'
    elif priority == 'stability':
        return 'zero_centered_rms_norm'  
    elif priority == 'balanced':
        return 'zero_centered_rms_norm'  # Good balance of speed and stability
    else:
        return 'layer_norm'  # Safe default
```

This normalization factory utility provides a robust, type-safe, and comprehensive solution for managing normalization layers across the dl_techniques framework. The addition of Zero-Centered RMSNorm offers enhanced training stability while maintaining computational efficiency, making it particularly suitable for large language models and advanced transformer architectures.