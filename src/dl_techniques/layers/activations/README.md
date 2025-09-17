# Activation Layer Module

The `dl_techniques.layers.activations` module provides a comprehensive collection of standard, custom, and advanced activation functions for deep learning, all accessible through a unified factory interface for consistent layer creation and robust configuration management.

## Overview

This module includes fifteen distinct activation layer types and a factory system for standardized instantiation and parameter validation. All layers are designed for modern Keras 3.x compatibility and support full serialization, making them suitable for any deep learning pipeline.

## Available Activation Types

The following layers are supported by the factory system with automated parameter validation and defaults:

| Type | Class | Description | Use Case |
|------|-------|-------------|----------|
| `adaptive_softmax` | `AdaptiveTemperatureSoftmax` | Softmax with dynamic temperature based on input entropy. | Maintaining sharpness in softmax for large output spaces. |
| `basis_function` | `BasisFunction` | Implements `b(x) = x * sigmoid(x)`, same as SiLU/Swish. | PowerMLP architectures; smooth, self-gated activation. |
| `gelu` | `GELU` | Gaussian Error Linear Unit. | State-of-the-art activation for Transformer models. |
| `hard_sigmoid` | `HardSigmoid` | Piecewise linear approximation of the sigmoid function. | Efficient gating in mobile architectures like MobileNetV3. |
| `hard_swish` | `HardSwish` | Computationally efficient approximation of the Swish/SiLU function. | High-performance activation for mobile-optimized models. |
| `silu` | `SiLU` | Sigmoid Linear Unit (Swish). | Self-gated activation that often outperforms ReLU. |
| `xatlu` | `xATLU` | Expanded ArcTan Linear Unit with a trainable `alpha`. | Adaptable arctan-based gating for specialized tasks. |
| `xgelu` | `xGELU` | Expanded Gaussian Error Linear Unit with a trainable `alpha`. | Extends GELU with a customizable gating range. |
| `xsilu` | `xSiLU` | Expanded Sigmoid Linear Unit with a trainable `alpha`. | Extends SiLU with a customizable gating range. |
| `elu_plus_one` | `EluPlusOne` | Enhanced ELU: `ELU(x) + 1 + epsilon`. | Ensuring strictly positive outputs (e.g., for rate parameters). |
| `mish` | `Mish` | A self-regularized, non-monotonic activation function. | Smooth activation that can outperform ReLU in deep models. |
| `saturated_mish` | `SaturatedMish` | Mish variant that smoothly saturates for large inputs. | Preventing activation explosion in very deep networks. |
| `relu_k` | `ReLUK` | Powered ReLU activation: `max(0, x)^k`. | Creating more aggressive non-linearities than standard ReLU. |
| `squash` | `SquashLayer` | Squashing non-linearity for Capsule Networks. | Normalizing vector outputs in Capsule Networks. |
| `thresh_max` | `ThreshMax` | Sparse softmax variant using a differentiable step function. | Creating sparse, confident probability distributions. |

## Factory Interface

### Basic Usage

```python
from dl_techniques.layers.activations import create_activation_layer

# Create a standard GELU layer
gelu_layer = create_activation_layer('gelu')

# Create ReLUK with a custom power
relu_k_layer = create_activation_layer('relu_k', k=2, name='relu_squared')
```

### Configuration-Based Creation

```python
from dl_techniques.layers.activations import create_activation_from_config

config = {
    'type': 'thresh_max',
    'slope': 15.0,
    'name': 'sparse_output_activation'
}

activation_layer = create_activation_from_config(config)
```

### Parameter Discovery

```python
from dl_techniques.layers.activations import get_activation_info

# Get information about all activation types
info = get_activation_info()

# Print requirements for a specific type
xgelu_info = info['xgelu']
print(f"Optional parameters for xGELU: {list(xgelu_info['optional_params'].keys())}")
```

### Validation

```python
from dl_techniques.layers.activations import validate_activation_config

# Validate configuration before creation
try:
    validate_activation_config('relu_k', k=3)
    print("Configuration is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Layer-Specific Parameters

### Parameter-Free Activations
The following activations have no configurable parameters: `basis_function`, `elu_plus_one`, `gelu`, `hard_sigmoid`, `hard_swish`, `mish`, `silu`. They can be created simply by name.

```python
mish_layer = create_activation_layer('mish')
hard_swish_layer = create_activation_layer('hard_swish')
```

### AdaptiveTemperatureSoftmax
**Optional:** `min_temp` (default: 0.1), `max_temp` (default: 1.0), `entropy_threshold` (default: 0.5), `eps` (default: 1e-7)

```python
adaptive_softmax = create_activation_layer(
    'adaptive_softmax',
    min_temp=0.05,
    max_temp=1.5,
    entropy_threshold=0.4
)
```

### Expanded Activations (xATLU, xGELU, xSiLU)
**Optional:** `alpha_initializer` (default: 'zeros'), `alpha_regularizer` (default: None), `alpha_constraint` (default: None)

```python
xgelu_reg = create_activation_layer(
    'xgelu',
    alpha_initializer='ones',
    alpha_regularizer=keras.regularizers.L2(1e-4)
)
```

### SaturatedMish
**Optional:** `alpha` (default: 3.0), `beta` (default: 0.5)

```python
sat_mish = create_activation_layer(
    'saturated_mish',
    alpha=4.0,  # Saturation starts later
    beta=0.2    # Sharper transition
)
```

### ReLUK
**Optional:** `k` (int, default: 3)

```python
relu_squared = create_activation_layer('relu_k', k=2)
```

### SquashLayer
**Optional:** `axis` (int, default: -1), `epsilon` (float, default: 1e-7)

```python
squash = create_activation_layer('squash', axis=2) # Squash along the 3rd dimension
```

### ThreshMax
**Optional:** `axis` (int, default: -1), `slope` (float, default: 10.0), `epsilon` (float, default: 1e-12)

```python
sharp_threshmax = create_activation_layer(
    'thresh_max',
    slope=50.0 # Creates a very sparse distribution
)
```

## Direct Layer Instantiation

While the factory is recommended, direct instantiation is always available.

```python
from dl_techniques.layers.activations import ReLUK, SaturatedMish, ThreshMax

# Direct instantiation (bypasses factory validation and logging)
relu_k = ReLUK(k=2)
sat_mish = SaturatedMish(alpha=4.0, beta=0.2)
thresh_max = ThreshMax(slope=20.0)
```

## Integration Patterns

### In Sequential Models

```python
import keras
from dl_techniques.layers.activations import create_activation_layer

model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,)),
    create_activation_layer('mish'),
    keras.layers.Dense(64),
    create_activation_layer('hard_swish'),
    keras.layers.Dense(10),
    create_activation_layer('adaptive_softmax')
])
```

### As an Argument to Other Layers

```python
# The factory returns a layer instance, not a string or function
# So it can be used directly where a layer is expected
dense_layer = keras.layers.Dense(
    64,
    activation=create_activation_layer('gelu')
)
```

### In Custom Layers

```python
@keras.saving.register_keras_serializable()
class CustomBlock(keras.layers.Layer):
    def __init__(self, units, activation_type='relu', **act_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_type = activation_type
        self.act_kwargs = act_kwargs
        
        self.dense = keras.layers.Dense(units)
        self.activation = create_activation_layer(
            activation_type,
            **act_kwargs
        )
    
    def call(self, inputs):
        x = self.dense(inputs)
        return self.activation(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation_type': self.activation_type,
            'act_kwargs': self.act_kwargs
        })
        return config
```

## Parameter Validation

The factory performs comprehensive validation for layers with configurable parameters.

### Value Range and Type Validation
```python
# Will raise ValueError: k must be a positive integer
create_activation_layer('relu_k', k=-2)

# Will raise TypeError: k must be an integer
create_activation_layer('relu_k', k=2.5)

# Will raise ValueError: alpha must be positive
create_activation_layer('saturated_mish', alpha=0)
```

## Logging and Debugging

The factory provides detailed logging for easier debugging.

### Info Level Logging
Shows all parameters used to create each layer:
```
INFO Creating relu_k activation layer with parameters:
INFO   k: 2
INFO   name: 'relu_squared'
```

### Debug Level Logging
Confirms successful layer creation:
```
DEBUG Successfully created relu_k layer: relu_squared
```

### Error Logging
Provides detailed context when creation fails:
```
ERROR Failed to create relu_k layer (ReLUK). Provided parameters: ['k'].
      Check parameter compatibility and types.
      Original error: k must be a positive integer, got -2
```

## Best Practices

1.  **Use the Factory for Consistency**: The factory ensures all activation layers are created through a single, validated interface.
2.  **Leverage Configuration Files**: For complex models, define activations in JSON or YAML files and create them with `create_activation_from_config` for reproducibility.
3.  **Validate Configurations**: Use `validate_activation_config` in production pipelines to catch errors early.

## API Reference

### Functions

#### `create_activation_layer(activation_type, name=None, **kwargs)`
Factory function for creating activation layers with validation.

#### `create_activation_from_config(config)`
Create an activation layer from a configuration dictionary.

#### `validate_activation_config(activation_type, **kwargs)`
Validate activation configuration parameters before creation.

#### `get_activation_info()`
Get comprehensive information about all available activation types.

### Types

#### `ActivationType`
A `Literal` type defining all valid activation type strings, such as `'gelu'`, `'hard_swish'`, `'xsilu'`, etc.

---
### `factory.py` (Updated)
---