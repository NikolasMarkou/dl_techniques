# Activation Layer Module

The `dl_techniques.layers.activations` module provides a comprehensive collection of standard, custom, and advanced activation functions for deep learning, all accessible through a unified factory interface for consistent layer creation and robust configuration management.

## Overview

This module includes distinct activation layer types and a factory system for standardized instantiation and parameter validation. All layers are designed for modern Keras 3.x compatibility and support full serialization.

## Available Activation Types

The following layers are supported by the factory system:

| Type | Class | Description |
|------|-------|-------------|
| `adaptive_softmax` | `AdaptiveTemperatureSoftmax` | Softmax with dynamic temperature based on input entropy. |
| `basis_function` | `BasisFunction` | Implements `b(x) = x * sigmoid(x)`, same as SiLU/Swish. |
| `differentiable_step` | `DifferentiableStep` | Learnable, differentiable approximation of a step function. |
| `elu_plus_one` | `EluPlusOne` | Enhanced ELU ensuring strictly positive outputs (`ELU(x) + 1 + ε`). |
| `gelu` | `GELU` | Gaussian Error Linear Unit. |
| `golu` | `GoLU` | Gompertz Linear Unit, asymmetrical self-gated activation. |
| `hard_sigmoid` | `HardSigmoid` | Piecewise linear approximation of the sigmoid function. |
| `hard_swish` | `HardSwish` | Computationally efficient approximation of Swish/SiLU. |
| `hierarchical_routing` | `HierarchicalRoutingLayer` | Trainable probabilistic routing tree for `O(log N)` classification. |
| `mish` | `Mish` | Self-regularized, non-monotonic activation. |
| `monotonicity` | `MonotonicityLayer` | Enforces monotonic constraints (e.g., for quantile regression). |
| `relu` | `keras.layers.ReLU` | Standard Rectified Linear Unit. |
| `relu_k` | `ReLUK` | Powered ReLU activation: `max(0, x)^k`. |
| `routing_probabilities` | `RoutingProbabilitiesLayer` | Parameter-free hierarchical routing using cosine basis patterns. |
| `saturated_mish` | `SaturatedMish` | Mish variant that smoothly saturates for large inputs. |
| `silu` | `SiLU` | Sigmoid Linear Unit (Swish). |
| `sparsemax` | `Sparsemax` | Euclidean projection onto the probability simplex (sparse outputs). |
| `squash` | `SquashLayer` | Vector length normalization for Capsule Networks. |
| `thresh_max` | `ThreshMax` | Sparse softmax variant using differentiable confidence thresholding. |
| `xatlu` | `xATLU` | Expanded ArcTan Linear Unit with trainable alpha. |
| `xgelu` | `xGELU` | Expanded GELU with trainable alpha. |
| `xsilu` | `xSiLU` | Expanded SiLU with trainable alpha. |

## Factory Interface

### Basic Usage

```python
from dl_techniques.layers.activations import create_activation_layer

# Create standard layers
gelu_layer = create_activation_layer('gelu')

# Create parameterized layers
relu_k = create_activation_layer('relu_k', k=2, name='relu_squared')
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

### Parameter Validation and Discovery

```python
from dl_techniques.layers.activations import get_activation_info, validate_activation_config

# Get registry info
info = get_activation_info()
print(info['differentiable_step']['optional_params'])

# Validate before creation
try:
    validate_activation_config('monotonicity', method='invalid_method')
except ValueError as e:
    print(f"Config error: {e}")
```

## Layer-Specific Configuration

### DifferentiableStep
A learnable binary gate.
```python
step = create_activation_layer(
    'differentiable_step',
    axis=-1,
    slope_initializer='ones',  # Learnable steepness
    shift_initializer='zeros'  # Learnable threshold
)
```

### Hierarchical Routing (Trainable)
Efficient classification for massive vocabularies/classes.
```python
routing = create_activation_layer(
    'hierarchical_routing',
    output_dim=50000,
    axis=-1,
    use_bias=True
)
```

### Monotonicity Layer
Enforces constraints like $Q(0.1) \le Q(0.5) \le Q(0.9)$.
```python
mono = create_activation_layer(
    'monotonicity',
    method='cumulative_softplus',
    axis=-1
)

# For bounded outputs (e.g., probability cumulative sums)
mono_bounded = create_activation_layer(
    'monotonicity',
    method='sigmoid',
    value_range=(0.0, 1.0)
)
```

### AdaptiveTemperatureSoftmax
Softmax that sharpens distribution based on entropy.
```python
adaptive = create_activation_layer(
    'adaptive_softmax',
    min_temp=0.1,
    max_temp=1.0,
    entropy_threshold=0.5
)
```

### RoutingProbabilities (Parameter-Free)
Deterministic routing based on fixed cosine patterns.
```python
# Infers output_dim from input shape if not provided
fixed_routing = create_activation_layer(
    'routing_probabilities',
    output_dim=100
)
```

### ThreshMax
Sparse softmax variant.
```python
sparse_soft = create_activation_layer(
    'thresh_max',
    slope=20.0,
    trainable_slope=True
)
```

## Integration Patterns

### Custom Blocks
```python
import keras
from dl_techniques.layers.activations import create_activation_layer

@keras.saving.register_keras_serializable()
class GatedBlock(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(units)
        self.gate = create_activation_layer('differentiable_step')
        
    def call(self, inputs):
        x = self.dense(inputs)
        return x * self.gate(x)
```

### Sequential Models
```python
model = keras.Sequential([
    keras.layers.Input(shape=(128,)),
    keras.layers.Dense(64),
    create_activation_layer('mish'),
    keras.layers.Dense(10),
    create_activation_layer('sparsemax')
])
```