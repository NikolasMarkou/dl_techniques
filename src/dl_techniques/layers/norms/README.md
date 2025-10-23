# Normalization Module

The `dl_techniques.layers.norms` module provides a comprehensive collection of normalization mechanisms for deep learning, with a unified factory interface for consistent layer creation, configuration management, and parameter validation.

## Overview

This module includes fifteen different normalization layer types, ranging from standard Keras layers to specialized variants for stability, efficiency, and advanced modeling like out-of-distribution detection. All layers are built using Keras 3 for backend-agnostic compatibility and support full serialization. The factory system ensures a standardized, safe, and introspectable way to integrate any of these normalization mechanisms into your models.

## Available Normalization Types

The following layers are supported by the factory system with automated parameter validation and defaults:

| Type | Class | Description | Use Case | Input Shape |
|------|-------|-------------|----------|-------------|
| `layer_norm` | `LayerNormalization` | Standard Keras normalization with learnable scale and bias. | General purpose normalization for transformers and deep networks. | Arbitrary |
| `batch_norm` | `BatchNormalization` | Standard Keras normalization with moving batch statistics. | Convolutional networks and batch-based training. | Arbitrary |
| `rms_norm` | `RMSNorm` | Root Mean Square normalization without centering for efficiency. | Transformers, especially for faster training and inference. | Arbitrary |
| `zero_centered_rms_norm`| `ZeroCenteredRMSNorm` | Combines RMSNorm efficiency with LayerNorm's zero-mean stability. | Large language models (LLMs) requiring enhanced training stability. | Arbitrary |
| `zero_centered_band_rms_norm`| `ZeroCenteredBandRMSNorm` | Adds a learnable band constraint to Zero-Centered RMSNorm. | Advanced LLMs for maximum stability and controlled flexibility. | Arbitrary |
| `band_rms` | `BandRMS` | RMS normalization with a learnable, bounded magnitude constraint. | Imposing "thick spherical shell" constraints for stable training. | Arbitrary |
| `adaptive_band_rms` | `AdaptiveBandRMS` | Adaptive RMS with scaling based on log-transformed RMS statistics. | Advanced training stability with input-adaptive scaling. | Arbitrary |
| `band_logit_norm` | `BandLogitNorm` | L2 normalization with a learned scaling factor bounded in a band. | Classification tasks with constrained logit magnitude. | Arbitrary |
| `global_response_norm`| `GlobalResponseNormalization` | Global Response Normalization (GRN) from ConvNeXt V2. | ConvNeXt-style architectures to enhance inter-channel competition. | 2D, 3D, or 4D tensors |
| `logit_norm` | `LogitNorm` | Temperature-scaled L2 normalization for classification logits. | Classification with calibrated confidence estimates. | Arbitrary |
| `max_logit_norm` | `MaxLogitNorm` | L2 normalization on logits to separate magnitude and direction. | Out-of-distribution (OOD) detection and uncertainty estimation. | Arbitrary |
| `decoupled_max_logit` | `DecoupledMaxLogit` | Decouples MaxLogit into cosine similarity and L2 norm components. | Advanced OOD detection with component analysis. | Arbitrary |
| `dml_plus_focal` | `DMLPlus` | DML+ variant returning the MaxCosine component for OOD detection. | Specialized "focal" models in the DML+ framework. | Arbitrary |
| `dml_plus_center` | `DMLPlus` | DML+ variant returning the MaxNorm component for OOD detection. | Specialized "center" models in the DML+ framework. | Arbitrary |
| `dynamic_tanh` | `DynamicTanh` | Learnable scaled hyperbolic tangent as a LayerNorm alternative. | Normalization-free transformer architectures. | Arbitrary |

## Factory Interface

### Basic Usage

```python
from dl_techniques.layers.norms import create_normalization_layer

# Create a standard layer normalization
layer_norm = create_normalization_layer('layer_norm', name='my_norm')

# Create RMS normalization with custom epsilon and scaling enabled
rms_norm = create_normalization_layer('rms_norm', epsilon=1e-8, use_scale=True)

# Create Zero-Centered Band RMS for maximum stability and control
zc_band_rms = create_normalization_layer(
    'zero_centered_band_rms_norm',
    max_band_width=0.1,
    epsilon=1e-6
)
```

### Configuration-Based Creation

```python
from dl_techniques.layers.norms import create_normalization_from_config

config = {
    'type': 'zero_centered_rms_norm',
    'epsilon': 1e-5,
    'use_scale': True,
    'axis': -1,
    'name': 'llm_norm_block_1'
}

zc_rms_layer = create_normalization_from_config(config)
```

### Parameter Discovery

```python
from dl_techniques.layers.norms import get_normalization_info

# Get information about all normalization types
info = get_normalization_info()

# Print details for a specific type
zc_band_rms_info = info['zero_centered_band_rms_norm']
print(f"Description: {zc_band_rms_info['description']}")
print(f"Parameters: {zc_band_rms_info['parameters']}")
```

### Validation

```python
from dl_techniques.layers.norms import validate_normalization_config

# Validate configuration before creation
try:
    validate_normalization_config(
        'band_rms',
        axis=-1,
        max_band_width=0.1,
        epsilon=1e-5
    )
    print("Configuration is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Layer-Specific Parameters

### `layer_norm`
**Optional:** `axis` (default: -1), `epsilon` (default: 1e-3), `center` (default: True), `scale` (default: True)
```python
norm = create_normalization_layer(
    'layer_norm',
    epsilon=1e-5,
    axis=-1
)
```

### `batch_norm`
**Optional:** `axis` (default: -1), `momentum` (default: 0.99), `epsilon` (default: 1e-3), `center` (default: True), `scale` (default: True)
```python
norm = create_normalization_layer(
    'batch_norm',
    momentum=0.9,
    axis=-1
)
```

### `rms_norm`
**Optional:** `axis` (default: -1), `epsilon` (default: 1e-6), `use_scale` (default: True), `scale_initializer` (default: 'ones')
```python
norm = create_normalization_layer(
    'rms_norm',
    epsilon=1e-5,
    use_scale=True
)
```

### `zero_centered_rms_norm`
**Optional:** `axis` (default: -1), `epsilon` (default: 1e-6), `use_scale` (default: True), `scale_initializer` (default: 'ones')
```python
norm = create_normalization_layer(
    'zero_centered_rms_norm',
    epsilon=1e-5,
    use_scale=True
)
```

### `zero_centered_band_rms_norm`
**Optional:** `max_band_width` (default: 0.1), `axis` (default: -1), `epsilon` (default: 1e-7), `band_initializer` (default: 'zeros'), `band_regularizer` (default: L2(1e-5))
```python
norm = create_normalization_layer(
    'zero_centered_band_rms_norm',
    max_band_width=0.08,
    epsilon=1e-6
)
```

### `band_rms`
**Optional:** `max_band_width` (default: 0.1), `axis` (default: -1), `epsilon` (default: 1e-7), `band_initializer` (default: 'zeros'), `band_regularizer` (default: L2(1e-5))
```python
norm = create_normalization_layer(
    'band_rms',
    max_band_width=0.2,
    band_initializer='ones'
)
```

### `adaptive_band_rms`
**Optional:** `max_band_width` (default: 0.1), `axis` (default: -1), `epsilon` (default: 1e-7), `band_initializer` (default: 'zeros'), `band_regularizer` (default: None)
```python
norm = create_normalization_layer(
    'adaptive_band_rms',
    max_band_width=0.15,
    axis=(1, 2) # Spatial normalization
)
```

### `band_logit_norm`
**Optional:** `max_band_width` (default: 0.01), `axis` (default: -1), `epsilon` (default: 1e-7)
```python
norm = create_normalization_layer(
    'band_logit_norm',
    max_band_width=0.05
)
```

### `global_response_norm`
**Optional:** `eps` (default: 1e-6), `gamma_initializer` (default: 'ones'), `beta_initializer` (default: 'zeros')
```python
# Note: factory maps `epsilon` to `eps` if `eps` is not provided
norm = create_normalization_layer(
    'global_response_norm',
    eps=1e-5 
)
```

### `logit_norm`
**Optional:** `temperature` (default: 0.04), `axis` (default: -1), `epsilon` (default: 1e-7)
```python
norm = create_normalization_layer(
    'logit_norm',
    temperature=0.1
)
```

### `max_logit_norm`
**Optional:** `axis` (default: -1), `epsilon` (default: 1e-7)
```python
norm = create_normalization_layer('max_logit_norm')
```

### `decoupled_max_logit`
**Optional:** `constant` (default: 1.0), `axis` (default: -1), `epsilon` (default: 1e-7)
```python
norm = create_normalization_layer(
    'decoupled_max_logit',
    constant=0.8
)
```

### `dml_plus_focal` / `dml_plus_center`
These types create a `DMLPlus` layer with `model_type` set to `'focal'` or `'center'`.
**Optional:** `axis` (default: -1), `epsilon` (default: 1e-7)
```python
# Focal model variant
focal_norm = create_normalization_layer('dml_plus_focal')

# Center model variant
center_norm = create_normalization_layer('dml_plus_center')
```

### `dynamic_tanh`
**Optional:** `axis` (default: -1), `alpha_init_value` (default: 0.5), `kernel_initializer` (default: 'ones'), `bias_initializer` (default: 'zeros')
```python
norm = create_normalization_layer(
    'dynamic_tanh',
    alpha_init_value=0.7 # Recommended for attention
)
```

## Direct Layer Instantiation

While the factory is recommended for consistency and safety, direct instantiation is always available.

```python
from dl_techniques.layers.norms import RMSNorm, ZeroCenteredBandRMSNorm, GlobalResponseNormalization

# Direct instantiation (bypasses factory validation and defaults)
rms_norm = RMSNorm(epsilon=1e-5, use_scale=True)
zc_band_rms = ZeroCenteredBandRMSNorm(max_band_width=0.1)
grn = GlobalResponseNormalization(eps=1e-6)
```

## Integration Patterns

### In a Custom Transformer Block

```python
import keras
from dl_techniques.layers.norms import create_normalization_layer

@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, normalization_type='zero_centered_rms_norm', **norm_kwargs):
        super().__init__()
        self.normalization_type = normalization_type
        self.norm_kwargs = norm_kwargs
        
        # Create normalization layers using the factory for enhanced stability
        self.attention_norm = create_normalization_layer(
            self.normalization_type, name='attention_norm', **self.norm_kwargs
        )
        self.ffn_norm = create_normalization_layer(
            self.normalization_type, name='ffn_norm', **self.norm_kwargs
        )
        # ... other layers like Attention, FFN
    
    def call(self, inputs):
        # Pre-normalization pattern
        norm_inputs = self.attention_norm(inputs)
        # attn_out = self.attention(norm_inputs, norm_inputs)
        # ... rest of the block
        return inputs # Placeholder
```

### In Model Builders with Configuration Files

```python
import json
from dl_techniques.layers.norms import create_normalization_from_config

# Load configuration from file
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Create normalization layer from the 'normalization' section of the config
# config['normalization'] = {'type': 'zero_centered_band_rms_norm', 'max_band_width': 0.1}
norm_layer = create_normalization_from_config(config['normalization'])
```

## Parameter Validation

The factory performs comprehensive validation on layer creation.

**Unknown Normalization Type:**
```python
# Raises ValueError: "Unknown normalization type: 'vanilla_norm'"
create_normalization_layer('vanilla_norm')
```

**Invalid Value Ranges:**
```python
# Raises ValueError: "max_band_width must be between 0 and 1, got 1.5"
create_normalization_layer('band_rms', max_band_width=1.5)

# Raises ValueError: "epsilon must be positive, got -1e-06"
create_normalization_layer('rms_norm', epsilon=-1e-6)
```

**Invalid Parameters for a Type:**
```python
# Raises ValueError: "Invalid parameters for dynamic_tanh: {'epsilon'}"
# DynamicTanh does not use the epsilon parameter.
validate_normalization_config('dynamic_tanh', epsilon=1e-6)
```

## API Reference

### Functions

-   **`create_normalization_layer(normalization_type, name=None, epsilon=1e-6, **kwargs)`**: Factory for creating normalization layers with validation.
-   **`create_normalization_from_config(config)`**: Creates a layer from a configuration dictionary.
-   **`validate_normalization_config(normalization_type, **kwargs)`**: Validates parameters before creation, raising a `ValueError` on failure.
-   **`get_normalization_info()`**: Returns a dictionary with details about all available normalization types, including their descriptions, parameters, and use cases.