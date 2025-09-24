# Standard Blocks Usage Guide

## Overview

The standard blocks module (`dl_techniques/layers/standard_blocks.py`) provides three highly configurable building blocks for constructing neural networks with modern Keras 3:

- **ConvBlock**: Configurable convolutional block with normalization, activation, and optional pooling
- **DenseBlock**: Configurable dense (fully connected) block with normalization, activation, and dropout
- **ResidualDenseBlock**: Dense block with residual connections for deeper networks

These blocks use factory patterns for normalization and activation layers, providing flexibility while maintaining clean, readable code.

## Import Statement

```python
from dl_techniques.layers.standard_blocks import ConvBlock, DenseBlock, ResidualDenseBlock
```

## ConvBlock

### Architecture Flow
```
Input → Conv2D → Normalization → Activation → Optional Dropout → Optional Pooling → Output
```

### Basic Usage

```python
# Simple convolutional block
conv_block = ConvBlock(
    filters=64,
    kernel_size=3,
    normalization_type='batch_norm',
    activation_type='relu'
)

# Apply to input tensor
x = keras.Input(shape=(32, 32, 3))
y = conv_block(x)
print(y.shape)  # (None, 32, 32, 64)
```

### Advanced Configuration

```python
# Advanced conv block with modern components
advanced_block = ConvBlock(
    filters=128,
    kernel_size=3,
    strides=2,  # Downsampling
    normalization_type='layer_norm',
    activation_type='gelu',
    dropout_rate=0.1,
    use_pooling=True,
    pool_type='max',
    pool_size=2,
    kernel_regularizer=keras.regularizers.L2(1e-4),
    normalization_kwargs={'epsilon': 1e-6},
    activation_kwargs={'approximate': True}
)
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filters` | int | Required | Number of convolutional filters |
| `kernel_size` | int/tuple | 3 | Size of convolution kernel |
| `strides` | int/tuple | 1 | Convolution strides |
| `padding` | str | "same" | Padding mode ("same" or "valid") |
| `normalization_type` | str | "batch_norm" | Type of normalization layer |
| `activation_type` | str | "relu" | Type of activation function |
| `dropout_rate` | float | 0.0 | Dropout rate (0.0 to disable) |
| `use_pooling` | bool | False | Whether to apply pooling |
| `pool_size` | int/tuple | 2 | Pooling window size |
| `pool_type` | str | "max" | Pooling type ("max" or "avg") |

### Common Use Cases

#### Feature Extraction Backbone
```python
# Building blocks for CNN backbone
backbone_blocks = [
    ConvBlock(32, kernel_size=7, strides=2, normalization_type='batch_norm'),
    ConvBlock(64, strides=2, use_pooling=True),
    ConvBlock(128, strides=2, activation_type='gelu'),
    ConvBlock(256, strides=2, dropout_rate=0.1),
]
```

#### Vision Transformer Style
```python
# Transformer-style conv blocks
vit_block = ConvBlock(
    filters=768,
    kernel_size=1,  # Point-wise convolution
    normalization_type='layer_norm',
    activation_type='gelu',
    normalization_kwargs={'axis': -1}  # Channel-last normalization
)
```

## DenseBlock

### Architecture Flow
```
Input → Dense → Optional Normalization → Activation → Optional Dropout → Output
```

### Basic Usage

```python
# Simple dense block
dense_block = DenseBlock(
    units=512,
    normalization_type='layer_norm',
    activation_type='relu',
    dropout_rate=0.1
)

# Apply to flattened features
x = keras.Input(shape=(1024,))
y = dense_block(x)
print(y.shape)  # (None, 512)
```

### Advanced Configuration

```python
# Transformer-style MLP block
transformer_mlp = DenseBlock(
    units=2048,
    normalization_type='rms_norm',
    activation_type='gelu',
    dropout_rate=0.1,
    kernel_regularizer=keras.regularizers.L2(1e-4),
    normalization_kwargs={'use_scale': True, 'epsilon': 1e-5},
    activation_kwargs={'approximate': False}
)

# Dense block without normalization
simple_dense = DenseBlock(
    units=256,
    normalization_type=None,  # No normalization
    activation_type='mish',
    dropout_rate=0.2
)
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `units` | int | Required | Number of output units |
| `normalization_type` | str/None | "layer_norm" | Type of normalization (None to disable) |
| `activation_type` | str | "relu" | Type of activation function |
| `dropout_rate` | float | 0.0 | Dropout rate |
| `kernel_regularizer` | Regularizer | None | Weight regularizer |
| `bias_regularizer` | Regularizer | None | Bias regularizer |
| `use_bias` | bool | True | Whether to use bias |

### Common Use Cases

#### Classification Head
```python
# Multi-layer classification head
classifier = [
    DenseBlock(512, activation_type='relu', dropout_rate=0.2),
    DenseBlock(256, activation_type='relu', dropout_rate=0.1),
    keras.layers.Dense(10, activation='softmax')  # Final layer
]
```

#### Transformer Feed-Forward Network
```python
# Transformer FFN block
transformer_ffn = DenseBlock(
    units=2048,  # Expansion factor of 4
    normalization_type='rms_norm',
    activation_type='gelu',
    dropout_rate=0.1
)
```

## ResidualDenseBlock

### Architecture Flow
```
Input → Dense → Optional Normalization → Activation → Optional Dropout → Add(Input) → Output
```

### Basic Usage

```python
# Simple residual dense block
residual_block = ResidualDenseBlock(
    normalization_type='layer_norm',
    activation_type='relu',
    dropout_rate=0.1
)

# Apply to features (output shape matches input)
x = keras.Input(shape=(512,))
y = residual_block(x)
print(y.shape)  # (None, 512) - same as input
```

### Advanced Configuration

```python
# Advanced residual block for deep networks
deep_residual = ResidualDenseBlock(
    normalization_type='rms_norm',
    activation_type='gelu',
    dropout_rate=0.1,
    kernel_regularizer=keras.regularizers.L2(1e-4),
    normalization_kwargs={'epsilon': 1e-6}
)
```

### Key Features

- **Automatic Unit Matching**: The dense layer automatically uses the same number of units as the input features
- **Residual Connection**: Adds the input to the transformed output for gradient flow
- **Deep Network Ready**: Enables training of very deep networks without vanishing gradients

### Common Use Cases

#### Deep Dense Networks
```python
# Stack multiple residual blocks for deep networks
def create_deep_dense_network(input_dim, num_blocks=10):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    
    for i in range(num_blocks):
        x = ResidualDenseBlock(
            normalization_type='layer_norm',
            activation_type='gelu',
            dropout_rate=0.1,
            name=f'residual_block_{i}'
        )(x)
    
    return keras.Model(inputs, x)
```

#### Transformer-Style Processing
```python
# Transformer decoder-like structure
transformer_block = ResidualDenseBlock(
    normalization_type='rms_norm',
    activation_type='gelu',
    dropout_rate=0.1
)
```

## Factory-Based Configuration

### Normalization Types
The blocks support various normalization types through factory functions:

- `'batch_norm'`: Batch normalization
- `'layer_norm'`: Layer normalization  
- `'rms_norm'`: RMS normalization
- `'zero_centered_rms_norm'`: Zero-centered RMS normalization
- `None`: No normalization (DenseBlock and ResidualDenseBlock only)

### Activation Types
Multiple activation functions are supported:

- `'relu'`: ReLU activation
- `'gelu'`: GELU activation
- `'mish'`: Mish activation  
- `'hard_swish'`: Hard Swish activation
- `'leaky_relu'`: Leaky ReLU activation

### Custom Parameters
Use `normalization_kwargs` and `activation_kwargs` to pass custom parameters:

```python
block = ConvBlock(
    filters=64,
    normalization_type='layer_norm',
    activation_type='leaky_relu',
    normalization_kwargs={'epsilon': 1e-6, 'axis': -1},
    activation_kwargs={'negative_slope': 0.01}
)
```

## Complete Examples

### CNN Architecture
```python
def create_cnn_classifier(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Feature extraction
    x = ConvBlock(32, kernel_size=7, strides=2, 
                  normalization_type='batch_norm',
                  activation_type='relu')(inputs)
    
    x = ConvBlock(64, strides=2, use_pooling=True,
                  activation_type='relu')(x)
    
    x = ConvBlock(128, strides=2, dropout_rate=0.1,
                  activation_type='gelu')(x)
    
    x = ConvBlock(256, strides=2, dropout_rate=0.2,
                  activation_type='gelu')(x)
    
    # Global pooling and classification
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    x = DenseBlock(512, activation_type='gelu', 
                   dropout_rate=0.3)(x)
    x = DenseBlock(256, activation_type='gelu', 
                   dropout_rate=0.2)(x)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```

### Deep Dense Network
```python
def create_deep_mlp(input_dim, output_dim, depth=8):
    inputs = keras.Input(shape=(input_dim,))
    
    # Initial projection
    x = DenseBlock(512, normalization_type='layer_norm',
                   activation_type='gelu', dropout_rate=0.1)(inputs)
    
    # Deep residual processing
    for i in range(depth):
        x = ResidualDenseBlock(
            normalization_type='rms_norm',
            activation_type='gelu',
            dropout_rate=0.1
        )(x)
    
    # Final projection
    outputs = keras.layers.Dense(output_dim)(x)
    
    return keras.Model(inputs, outputs)
```

### Hybrid CNN-MLP
```python
def create_hybrid_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Convolutional feature extraction
    x = ConvBlock(64, normalization_type='batch_norm')(inputs)
    x = ConvBlock(128, strides=2, use_pooling=True)(x)
    x = ConvBlock(256, strides=2, dropout_rate=0.1)(x)
    
    # Transition to dense processing
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Deep dense processing with residuals
    for i in range(4):
        x = ResidualDenseBlock(
            normalization_type='layer_norm',
            activation_type='gelu',
            dropout_rate=0.1
        )(x)
    
    # Classification
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)
```
