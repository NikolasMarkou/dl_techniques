# Complete Guide to Keras 3 Custom Layers and Models

A comprehensive, authoritative guide for creating robust, serializable, and production-ready custom Layers and Models in Keras 3 for the dl-techniques framework.

---

## Table of Contents

1. [Core Design Principles](#1-core-design-principles)
2. [Essential Setup and Registration](#2-essential-setup-and-registration)
3. [Layer Implementation Patterns](#3-layer-implementation-patterns)
4. [Graph-Safe Operations in call()](#4-graph-safe-operations-in-call)
5. [Model Implementation Patterns](#5-model-implementation-patterns)
6. [Architecture Patterns for Reusable Models](#6-architecture-patterns-for-reusable-models)
7. [Configuration Management](#7-configuration-management)
8. [Serialization and Deserialization](#8-serialization-and-deserialization)
9. [Weight Compatibility](#9-weight-compatibility)
10. [Extension Points and Modularity](#10-extension-points-and-modularity)
11. [Factory Patterns](#11-factory-patterns)
12. [Testing and Validation](#12-testing-and-validation)
13. [Common Pitfalls and Solutions](#13-common-pitfalls-and-solutions)
14. [Troubleshooting Guide](#14-troubleshooting-guide)
15. [Complete Examples](#15-complete-examples)

---

## 1. Core Design Principles

### 1.1 The Golden Rule: Create vs. Build

This is the **most critical concept** in modern Keras 3. Understanding this separation eliminates 99% of build and serialization errors.

#### The Serialization Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                     SAVING (.keras format)                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Keras calls get_config() on each layer                      │
│  2. Config dict is serialized to JSON                           │
│  3. For each BUILT layer, weights are extracted                 │
│  4. Everything packaged into .keras archive                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LOADING (.keras format)                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Parse JSON config                                            │
│  2. For each layer class (using registry):                      │
│     a. Call __init__(**config) → Creates UNBUILT layer          │
│  3. Call build() to create weight variables                     │
│  4. Load saved weight VALUES into weight variables              │
└─────────────────────────────────────────────────────────────────┘
```

#### The Rules

| Method | What Happens | What's Allowed |
|--------|--------------|----------------|
| `__init__` | Once at instantiation | CREATE all sub-layers, store ALL configuration |
| `build` | Once when shapes known | CREATE weights via `add_weight()`, BUILD sub-layers |
| `call` | Every batch | **Must remain symbolic** - only `ops` operations |
| `compute_output_shape` | Shape inference | Return output shape given input shape (REQUIRED) |

**CRITICAL**: Every custom layer MUST implement `compute_output_shape()`. This method:
- Enables Keras to infer shapes without running forward pass
- Is essential for building sub-layers with correct shapes
- Required for functional API model construction
- Must work without the layer being built

**NEVER** in `__init__`:
- Create weights directly (`self.add_weight`)
- Inspect `input_shape` or call operations requiring shapes

**ALWAYS** in `build`:
- Create layer's own weights
- Explicitly call `build()` on each sub-layer for robust serialization
- Call `super().build(input_shape)` at the end

**ALWAYS** implement `compute_output_shape`:
- Return the correct output shape tuple given input shape
- Must work even before the layer is built
- Use stored configuration (e.g., `self.units`) not weight shapes

### 1.2 Separation of Layer Creation vs. Layer Usage

```python
class ProperModel(keras.Model):
    def __init__(self, use_feature_a=True, use_feature_b=True, **kwargs):
        super().__init__(**kwargs)
        
        # LAYER CREATION: Always create all layers
        self.feature_a = FeatureLayer()
        self.feature_b = FeatureLayer()
        self.output_layer = OutputLayer()
        
        # Store usage flags
        self.use_feature_a = use_feature_a
        self.use_feature_b = use_feature_b
    
    def call(self, inputs, training=None):
        x = inputs
        
        # LAYER USAGE: Conditionally use layers
        if self.use_feature_a:
            x = self.feature_a(x)
        if self.use_feature_b:
            x = self.feature_b(x)
        
        return self.output_layer(x)
```

**Why This Matters**:
- All weights exist with consistent names
- Weight loading works regardless of configuration
- Models are interchangeable for transfer learning
- Easy to switch between modes (training/inference/feature extraction)

### 1.3 Configuration as Data

All model configuration should be serializable data (not code):

```python
# ✓ GOOD: Configuration as data
config = {
    'depth': 4,
    'filters': [64, 128, 256, 512],
    'activation': 'relu',
    'dropout_rate': 0.1
}
model = Model(**config)

# ✗ BAD: Configuration as code
model = Model()
model.add_layer(...)
```

### 1.4 Explicit Over Implicit

Make architecture decisions explicit and visible:

```python
class ExplicitModel(keras.Model):
    def __init__(self, depth: int, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.filters = filters
        
        # Build structure explicitly
        self._build_encoder()
        self._build_decoder()
        self._build_heads()
    
    def _build_encoder(self):
        """Build encoder stages."""
        self.encoder_stages = []
        for level in range(self.depth):
            stage = self._create_stage(level)
            self.encoder_stages.append(stage)
```

---

## 2. Essential Setup and Registration

### 2.1 Core Imports

```python
# Core Keras imports - always use full paths
import keras
from keras import ops
from keras import layers
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations

# Type hints
from typing import Optional, Union, Tuple, List, Dict, Any, Callable, Literal

# Numerical computing
import numpy as np

# For testing (backend-specific)
import tensorflow as tf

# DL-Techniques framework imports
from dl_techniques.utils.logger import logger
```

### 2.2 Registration Decorator

**CRITICAL**: Every custom class MUST be registered for serialization:

```python
@keras.saving.register_keras_serializable()
class YourCustomLayer(keras.layers.Layer):
    pass

# With package name for organization
@keras.saving.register_keras_serializable(package='MyModels')
class YourCustomModel(keras.Model):
    pass
```

Without this decorator, you'll get "Unknown layer" errors during model loading.

### 2.3 Type Hints Reference

```python
from typing import (
    Optional,      # Optional[T] = T | None
    Union,         # Union[A, B] = A | B  
    Tuple,         # Tuple[int, int] for fixed, Tuple[int, ...] for variable
    List,          # List[T] for homogeneous lists
    Dict,          # Dict[K, V] for dictionaries
    Any,           # Any type (use sparingly)
    Callable,      # Callable[[ArgTypes], ReturnType]
    Literal,       # Literal["option1", "option2"] for exact values
)

# Common patterns in Keras layers:
def __init__(
    self,
    units: int,
    activation: Optional[Union[str, Callable]] = None,
    kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
    use_bias: bool = True,
    dropout_rate: float = 0.1,
    **kwargs: Any
) -> None:
    ...

def call(
    self,
    inputs: keras.KerasTensor,
    training: Optional[bool] = None,
    mask: Optional[keras.KerasTensor] = None,
) -> keras.KerasTensor:
    ...

def compute_output_shape(
    self, 
    input_shape: Tuple[Optional[int], ...]
) -> Tuple[Optional[int], ...]:
    ...
```

---

## 3. Layer Implementation Patterns

### 3.1 Pattern 1: Simple Layer (No Sub-layers)

For layers that only need their own weights:

```python
@keras.saving.register_keras_serializable()
class SimpleCustomLayer(keras.layers.Layer):
    """
    Custom linear transformation layer with optional bias and activation.
    
    Args:
        units: Dimensionality of the output space.
        activation: Activation function. Defaults to None.
        use_bias: Whether to add a learnable bias vector. Defaults to True.
        kernel_initializer: Initializer for the weight matrix. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the bias vector. Defaults to 'zeros'.
        **kwargs: Additional arguments for Layer base class.
    
    Input shape:
        N-D tensor: `(batch_size, ..., input_dim)`.
    
    Output shape:
        N-D tensor: `(batch_size, ..., units)`.
    """
    
    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, Callable]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        
        # Store ALL configuration
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
        # Initialize weight attributes - created in build()
        self.kernel = None
        self.bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's own weights."""
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        
        super().build(input_shape)

    def call(
        self, 
        inputs: keras.KerasTensor, 
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation."""
        outputs = ops.matmul(inputs, self.kernel)
        
        if self.use_bias:
            outputs = ops.add(outputs, self.bias)
        
        if self.activation is not None:
            outputs = self.activation(outputs)
        
        return outputs

    def compute_output_shape(
        self, 
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        })
        return config
```

### 3.2 Pattern 2: Composite Layer (With Sub-layers)

For layers that contain other layers and need explicit building:

```python
@keras.saving.register_keras_serializable()
class CompositeLayer(keras.layers.Layer):
    """
    Multi-stage neural network block with sub-layer composition.
    
    **Architecture**:
    ```
    Input → Dense₁(hidden_dim, 'gelu') → Dropout → LayerNorm → Dense₂(output_dim) → Output
    ```
    
    Args:
        hidden_dim: Dimensionality of the intermediate hidden layer.
        output_dim: Dimensionality of the final output.
        dropout_rate: Dropout rate. Defaults to 0.1.
        use_norm: Whether to apply layer normalization. Defaults to True.
        **kwargs: Additional arguments for Layer base class.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        use_norm: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        
        # Store configuration
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        
        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.dense1 = layers.Dense(hidden_dim, activation="gelu", name="dense1")
        self.dropout = layers.Dropout(dropout_rate, name="dropout")
        self.norm = layers.LayerNormalization(name="layer_norm") if use_norm else None
        self.dense2 = layers.Dense(output_dim, name="dense2")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.
        
        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        self.dense1.build(input_shape)
        dense1_output_shape = self.dense1.compute_output_shape(input_shape)
        
        self.dropout.build(dense1_output_shape)
        
        if self.norm is not None:
            self.norm.build(dense1_output_shape)
            
        self.dense2.build(dense1_output_shape)
        
        super().build(input_shape)

    def call(
        self, 
        inputs: keras.KerasTensor, 
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through sub-layers."""
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return self.dense2(x)

    def compute_output_shape(
        self, 
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate,
            'use_norm': self.use_norm,
        })
        return config
```

### 3.3 Managing Complex Layer Hierarchies

**Pattern: Nested Lists**

```python
class HierarchicalModel(keras.Model):
    def __init__(self, depth: int, blocks_per_level: int, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.blocks_per_level = blocks_per_level
        
        # Use nested lists for hierarchical structure
        self.encoder_stages = []
        for level in range(depth):
            stage_blocks = []
            for block_idx in range(blocks_per_level):
                block = Block(name=f'enc_L{level}_blk{block_idx}')
                stage_blocks.append(block)
            self.encoder_stages.append(stage_blocks)
        
        self.encoder_downsamples = []
        for level in range(depth - 1):
            downsample = layers.MaxPooling2D(name=f'down_{level}')
            self.encoder_downsamples.append(downsample)
    
    def call(self, inputs, training=None):
        skip_connections = []
        x = inputs
        
        for level, stage_blocks in enumerate(self.encoder_stages):
            for block in stage_blocks:
                x = block(x, training=training)
            
            skip_connections.append(x)
            
            if level < len(self.encoder_downsamples):
                x = self.encoder_downsamples[level](x, training=training)
        
        return x, skip_connections
```

### 3.4 Implementing compute_output_shape

**CRITICAL**: Every custom layer MUST implement `compute_output_shape()`. This enables shape inference, sub-layer building, and functional API compatibility.

**Pattern 1: Simple Output Shape**

```python
def compute_output_shape(
    self, 
    input_shape: Tuple[Optional[int], ...]
) -> Tuple[Optional[int], ...]:
    """
    Compute output shape from input shape.
    
    Args:
        input_shape: Shape tuple, may contain None for dynamic dimensions.
    
    Returns:
        Output shape tuple.
    """
    # For layers that change the last dimension
    output_shape = list(input_shape)
    output_shape[-1] = self.units
    return tuple(output_shape)
```

**Pattern 2: Multiple Inputs**

```python
def compute_output_shape(
    self, 
    input_shape: Union[Tuple, List[Tuple]]
) -> Tuple[Optional[int], ...]:
    """Handle layers with multiple inputs."""
    if isinstance(input_shape, list):
        # Multiple inputs - use the first one as reference
        primary_shape = input_shape[0]
    else:
        primary_shape = input_shape
    
    output_shape = list(primary_shape)
    output_shape[-1] = self.output_dim
    return tuple(output_shape)
```

**Pattern 3: Shape-Changing Operations**

```python
def compute_output_shape(
    self, 
    input_shape: Tuple[Optional[int], ...]
) -> Tuple[Optional[int], ...]:
    """For layers that reshape or pool."""
    batch, height, width, channels = input_shape
    
    # Example: 2x downsampling
    new_height = height // 2 if height is not None else None
    new_width = width // 2 if width is not None else None
    
    return (batch, new_height, new_width, self.filters)
```

**Pattern 4: Using Sub-layer Shapes**

```python
def compute_output_shape(
    self, 
    input_shape: Tuple[Optional[int], ...]
) -> Tuple[Optional[int], ...]:
    """Delegate to sub-layers for complex compositions."""
    shape = self.dense1.compute_output_shape(input_shape)
    shape = self.dense2.compute_output_shape(shape)
    return shape
```

**Pattern 5: Multiple Outputs**

```python
def compute_output_shape(
    self, 
    input_shape: Tuple[Optional[int], ...]
) -> List[Tuple[Optional[int], ...]]:
    """For layers returning multiple tensors."""
    batch = input_shape[0]
    
    main_shape = (batch, self.main_units)
    aux_shape = (batch, self.aux_units)
    
    return [main_shape, aux_shape]
```

**Common Mistakes to Avoid:**

```python
# ❌ WRONG: Accessing weight shapes (layer may not be built)
def compute_output_shape(self, input_shape):
    return (input_shape[0], self.kernel.shape[-1])  # FAILS if not built!

# ✅ CORRECT: Use stored configuration
def compute_output_shape(self, input_shape):
    return (input_shape[0], self.units)  # Works always
```

---

## 4. Graph-Safe Operations in call()

This section covers the **most common source of runtime bugs**: breaking the computational graph during tracing.

### 4.1 The Core Concept

When Keras compiles your model, it runs your Python code **once** with symbolic inputs to generate a graph.

| ❌ Never in `call()` | ✅ Always in `call()` |
|---------------------|----------------------|
| `list()`, `tuple()`, or `int()` on a tensor | Use `ops.shape(x)` for dynamic dimensions |
| `.numpy()` on a tensor | Use `ops.cast()` for type conversion |
| Python `if` on tensor values | Use `ops.where()` or `ops.cond()` |
| `x.shape` for dynamic manipulation | Use `ops.shape(x)` which returns a tensor |
| Python `for` over tensor dimensions | Use vectorized ops or `ops.scan` |

### 4.2 Handling Shapes (The #1 Source of Bugs)

**❌ BAD - Breaking the Graph:**

```python
def call(self, inputs):
    shape = ops.shape(inputs)
    shape_list = list(shape)     # ERROR: Tries to read symbolic values
    batch_size = shape_list[0]   # ERROR: Indexing a Python list of symbols
```

**✅ GOOD - Graph Safe:**

```python
def call(self, inputs):
    shape = ops.shape(inputs)    # Returns a Tensor, not a list
    batch_size = shape[0]        # Slicing a Tensor is fine
    new_shape = ops.stack([batch_size, self.target_dim])
    x = ops.reshape(inputs, new_shape)
    return x
```

### 4.3 Branching (If/Else)

Python `if` statements execute during **tracing**, not runtime.

**❌ BAD:**

```python
def call(self, inputs):
    if ops.sum(inputs) > 0:  # Checked once at compile time!
        return inputs * 2
    else:
        return inputs
```

**✅ GOOD:**

```python
def call(self, inputs):
    condition = ops.sum(inputs) > 0
    return ops.where(condition, inputs * 2, inputs)

# OR for different computation paths:
def call(self, inputs):
    condition = ops.sum(inputs) > 0
    return ops.cond(
        condition,
        true_fn=lambda: inputs * 2,
        false_fn=lambda: inputs
    )
```

### 4.4 Loops

**Static Loops (Safe):** Iterating over a fixed number known at trace time:

```python
def call(self, inputs):
    x = inputs
    for i in range(self.num_layers):  # num_layers is an int from __init__
        x = self.layers[i](x)
    return x
```

**Dynamic Loops (Risky):** Iterating over a tensor dimension:

```python
# ❌ BAD
def call(self, inputs):
    seq_len = ops.shape(inputs)[1]
    for i in range(seq_len):  # ERROR: seq_len is symbolic!
        ...

# ✅ GOOD - Use vectorized operations or ops.scan
def call(self, inputs):
    return ops.sum(inputs, axis=1)  # Vectorized
```

### 4.5 Complete Graph-Safe Example

```python
@keras.saving.register_keras_serializable()
class GraphSafeLayer(keras.layers.Layer):
    """Demonstrates graph-safe patterns for dynamic shape handling."""
    
    def __init__(self, units: int, use_scaling: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.use_scaling = use_scaling
        self.dense = layers.Dense(units)
        
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.dense.build(input_shape)
        self.scale = self.add_weight(
            name='scale',
            shape=(self.units,),
            initializer='ones',
            trainable=True,
        )
        super().build(input_shape)
    
    def call(
        self, 
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        # GRAPH-SAFE: Get dynamic shape as tensor
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        
        x = self.dense(inputs)
        
        # Static Python condition (known at trace time) - OK
        if self.use_scaling:
            x = x * self.scale
        
        # GRAPH-SAFE: Dynamic condition on tensor values
        mean_val = ops.mean(x)
        x = ops.where(mean_val > 0, x, x + 0.1)
        
        # GRAPH-SAFE: Dynamic reshape using tensor shapes
        flat_shape = ops.stack([batch_size, -1])
        x_flat = ops.reshape(x, flat_shape)
        
        original_shape = ops.stack([batch_size, self.units])
        x = ops.reshape(x_flat, original_shape)
        
        return x
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'units': self.units, 'use_scaling': self.use_scaling})
        return config
```

---

## 5. Model Implementation Patterns

### 5.1 Custom Model with Sub-models

```python
@keras.saving.register_keras_serializable()
class CustomModel(keras.Model):
    """
    Multi-component model demonstrating proper composition and serialization.
    
    Args:
        hidden_dim: Dimensionality of the encoder's output space.
        output_dim: Dimensionality of the final output.
        use_processor: Whether to include intermediate processing layer.
        **kwargs: Additional arguments for Model base class.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        use_processor: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_processor = use_processor
        
        # Create all sub-models/layers in __init__
        self.encoder = layers.Dense(hidden_dim, activation='relu', name='encoder')
        self.processor = CompositeLayer(
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            name='processor'
        ) if use_processor else None
        self.decoder = layers.Dense(output_dim, name='decoder')

    def call(
        self, 
        inputs: keras.KerasTensor, 
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        x = self.encoder(inputs)
        
        if self.processor is not None:
            x = self.processor(x, training=training)
            
        return self.decoder(x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'use_processor': self.use_processor,
        })
        return config
```

### 5.2 Functional Model Factory

For functional API flexibility:

```python
def create_model(
    input_shape: Tuple[int, ...],
    hidden_dim: int = 128,
    output_dim: int = 10,
    num_layers: int = 2,
    use_custom_layer: bool = True
) -> keras.Model:
    """
    Factory function creating a functional API model.
    
    Args:
        input_shape: Tuple specifying input dimensions (excluding batch).
        hidden_dim: Dimensionality of hidden layers.
        output_dim: Number of output classes/features.
        num_layers: Number of hidden layers.
        use_custom_layer: Whether to use custom layers.
    
    Returns:
        keras.Model: Compiled functional model.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    for i in range(num_layers):
        if use_custom_layer:
            x = SimpleCustomLayer(hidden_dim, activation='relu', name=f'custom_{i}')(x)
        else:
            x = layers.Dense(hidden_dim, activation='relu', name=f'dense_{i}')(x)
    
    outputs = layers.Dense(output_dim, activation='softmax', name='output')(x)
    
    return keras.Model(inputs, outputs, name='functional_model')
```

---

## 6. Architecture Patterns for Reusable Models

### 6.1 Staged Construction

Build complex models in logical stages:

```python
class StagedModel(keras.Model):
    def __init__(self, input_shape_config, depth, **kwargs):
        super().__init__(**kwargs)
        
        self.input_shape_config = input_shape_config
        self.depth = depth
        
        # Build in stages (Golden Rule: Create in __init__)
        self._build_stem()
        self._build_encoder()
        self._build_bottleneck()
        self._build_decoder()
        self._build_heads()
    
    def _build_stem(self):
        """Stage 1: Input processing."""
        self.stem = StemBlock(...)
    
    def _build_encoder(self):
        """Stage 2: Downsampling path."""
        self.encoder_stages = []
        for level in range(self.depth):
            stage = self._create_encoder_stage(level)
            self.encoder_stages.append(stage)
    
    def _build_bottleneck(self):
        """Stage 3: Bottleneck processing."""
        self.bottleneck = BottleneckBlock(...)
    
    def _build_decoder(self):
        """Stage 4: Upsampling path."""
        self.decoder_stages = []
        for level in reversed(range(self.depth)):
            stage = self._create_decoder_stage(level)
            self.decoder_stages.append(stage)
    
    def _build_heads(self):
        """Stage 5: Output heads."""
        self.main_head = layers.Conv2D(...)
        if self.enable_auxiliary:
            self.aux_heads = [...]
```

### 6.2 Configuration-Driven Architecture

```python
class ConfigurableModel(keras.Model):
    """
    Model architecture driven by configuration dictionary.
    
    Example:
        config = {
            'encoder': {'type': 'resnet', 'depth': 50},
            'decoder': {'type': 'unet', 'skip_connections': True},
            'heads': {
                'main': {'channels': 10, 'activation': 'softmax'},
                'auxiliary': [{'channels': 10, 'activation': 'softmax'}]
            }
        }
        model = ConfigurableModel(config)
    """
    
    ENCODER_TYPES = {
        'resnet': ResNetEncoder,
        'efficientnet': EfficientNetEncoder,
    }
    
    DECODER_TYPES = {
        'unet': UNetDecoder,
        'fpn': FPNDecoder,
    }
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        self.encoder = self._build_encoder(config['encoder'])
        self.decoder = self._build_decoder(config['decoder'])
        self.heads = self._build_heads(config['heads'])
    
    def _build_encoder(self, encoder_config):
        encoder_type = encoder_config['type']
        encoder_class = self.ENCODER_TYPES[encoder_type]
        return encoder_class(**encoder_config)
    
    def _build_decoder(self, decoder_config):
        decoder_type = decoder_config['type']
        decoder_class = self.DECODER_TYPES[decoder_type]
        return decoder_class(**decoder_config)
    
    def _build_heads(self, heads_config):
        heads = {}
        heads['main'] = self._create_head(heads_config['main'])
        if 'auxiliary' in heads_config:
            heads['auxiliary'] = [
                self._create_head(cfg) for cfg in heads_config['auxiliary']
            ]
        return heads
```

---

## 7. Configuration Management

### 7.1 Comprehensive Configuration Schema

```python
@keras.saving.register_keras_serializable()
class WellConfiguredModel(keras.Model):
    """Model with comprehensive configuration management."""
    
    def __init__(
        self,
        # Core architecture
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        depth: int = 4,
        initial_filters: int = 64,
        filter_multiplier: int = 2,
        blocks_per_level: int = 2,
        
        # Block configuration
        block_type: str = 'residual',
        activation: str = 'relu',
        normalization: str = 'batch_norm',
        
        # Regularization
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        kernel_regularizer: Optional[str] = None,
        
        # Initialization
        kernel_initializer: str = 'he_normal',
        
        # Output configuration
        output_channels: Optional[int] = None,
        final_activation: str = 'linear',
        
        # Mode switches
        include_top: bool = True,
        enable_deep_supervision: bool = False,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store all configuration (critical for serialization)
        self.input_shape_config = input_shape
        self.depth = depth
        self.initial_filters = initial_filters
        self.filter_multiplier = filter_multiplier
        self.blocks_per_level = blocks_per_level
        self.block_type = block_type
        self.activation = activation
        self.normalization = normalization
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.output_channels = output_channels
        self.final_activation = final_activation
        self.include_top = include_top
        self.enable_deep_supervision = enable_deep_supervision
        
        self._validate_config()
        self._build_architecture()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.depth < 2:
            raise ValueError(f"depth must be >= 2, got {self.depth}")
        if self.initial_filters <= 0:
            raise ValueError(f"initial_filters must be positive")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1]")
        valid_block_types = ['residual', 'dense', 'mobile']
        if self.block_type not in valid_block_types:
            raise ValueError(f"block_type must be one of {valid_block_types}")
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_config,
            'depth': self.depth,
            'initial_filters': self.initial_filters,
            'filter_multiplier': self.filter_multiplier,
            'blocks_per_level': self.blocks_per_level,
            'block_type': self.block_type,
            'activation': self.activation,
            'normalization': self.normalization,
            'dropout_rate': self.dropout_rate,
            'drop_path_rate': self.drop_path_rate,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'output_channels': self.output_channels,
            'final_activation': self.final_activation,
            'include_top': self.include_top,
            'enable_deep_supervision': self.enable_deep_supervision,
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WellConfiguredModel':
        if 'kernel_regularizer' in config:
            config['kernel_regularizer'] = regularizers.deserialize(config['kernel_regularizer'])
        if 'kernel_initializer' in config:
            config['kernel_initializer'] = initializers.deserialize(config['kernel_initializer'])
        return cls(**config)
```

### 7.2 Configuration Presets Pattern

```python
class ModelWithPresets(keras.Model):
    """Model with predefined configuration presets."""
    
    PRESETS = {
        'tiny': {
            'depth': 3,
            'initial_filters': 32,
            'blocks_per_level': 2,
            'drop_path_rate': 0.0,
        },
        'base': {
            'depth': 4,
            'initial_filters': 64,
            'blocks_per_level': 3,
            'drop_path_rate': 0.1,
        },
        'large': {
            'depth': 4,
            'initial_filters': 96,
            'blocks_per_level': 4,
            'drop_path_rate': 0.2,
        }
    }
    
    def __init__(self, preset: Optional[str] = None, **kwargs):
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Available: {list(self.PRESETS.keys())}")
            config = self.PRESETS[preset].copy()
            config.update(kwargs)
            super().__init__(**config)
        else:
            super().__init__(**kwargs)
    
    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> 'ModelWithPresets':
        """Factory method to create model from preset."""
        return cls(preset=preset, **kwargs)
    
    @classmethod
    def list_presets(cls) -> List[str]:
        return list(cls.PRESETS.keys())
```

---

## 8. Serialization and Deserialization

### 8.1 Complete Serialization Pattern

```python
@keras.saving.register_keras_serializable(package='MyModels')
class SerializableModel(keras.Model):
    """
    Model with complete serialization support.
    
    Supports:
    1. Full model (.keras)
    2. Weights only (.h5, .weights.h5)
    3. Configuration (JSON)
    """
    
    def __init__(self, config_param1, config_param2, **kwargs):
        super().__init__(**kwargs)
        self.config_param1 = config_param1
        self.config_param2 = config_param2
        # ... build architecture
    
    def get_config(self) -> Dict[str, Any]:
        """Must include all parameters needed to reconstruct model."""
        config = super().get_config()
        config.update({
            'config_param1': self.config_param1,
            'config_param2': self.config_param2,
            'regularizer': regularizers.serialize(self.regularizer),
            'initializer': initializers.serialize(self.initializer)
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SerializableModel':
        """Must deserialize complex objects."""
        if 'regularizer' in config:
            config['regularizer'] = regularizers.deserialize(config['regularizer'])
        if 'initializer' in config:
            config['initializer'] = initializers.deserialize(config['initializer'])
        return cls(**config)
    
    def export_config(self, filepath: str):
        """Export configuration to JSON."""
        import json
        config = self.get_config()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_config_file(cls, filepath: str) -> 'SerializableModel':
        """Create model from JSON configuration."""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls.from_config(config)
```

### 8.2 The Critical Serialization Test

**Every custom component MUST pass this test:**

```python
import tempfile
import os

def test_serialization_cycle(layer_class, layer_config, sample_input):
    """
    Comprehensive serialization test.
    
    Verifies:
    1. Instantiation and forward pass
    2. Serialization to .keras format
    3. Deserialization back to functional layer
    4. Identical outputs after deserialization
    """
    original_layer = layer_class(**layer_config)
    original_output = original_layer(sample_input)
    
    # Wrap in model for serialization
    inputs = keras.Input(shape=sample_input.shape[1:])
    outputs = layer_class(**layer_config)(inputs)
    model = keras.Model(inputs, outputs)
    
    model_output = model(sample_input)
    
    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model.keras')
        model.save(model_path)
        loaded_model = keras.models.load_model(model_path)
    
    loaded_output = loaded_model(sample_input)
    
    np.testing.assert_allclose(
        ops.convert_to_numpy(model_output),
        ops.convert_to_numpy(loaded_output),
        rtol=1e-6, atol=1e-6,
        err_msg="Outputs should match after serialization"
    )
    
    logger.info("Serialization cycle test passed")
```

---

## 9. Weight Compatibility

### 9.1 Ensuring Weight Compatibility

**Rule: All Layers Always Created**

```python
class WeightCompatibleModel(keras.Model):
    """
    Model designed for weight compatibility across configurations.
    
    Key principle: Always create all layers, control usage via flags.
    """
    
    def __init__(
        self,
        output_channels: int,
        include_top: bool = True,
        enable_auxiliary_heads: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.output_channels = output_channels
        self.include_top = include_top
        self.enable_auxiliary_heads = enable_auxiliary_heads
        
        # CRITICAL: Always create all layers with consistent names
        self._build_backbone()
        self._build_main_head()
        self._build_auxiliary_heads()
    
    def _build_backbone(self):
        self.backbone = Backbone(name='backbone')
    
    def _build_main_head(self):
        self.main_head = layers.Conv2D(
            filters=self.output_channels,
            kernel_size=1,
            name='main_head'
        )
    
    def _build_auxiliary_heads(self):
        self.auxiliary_heads = []
        if self.enable_auxiliary_heads:
            for i in range(3):
                head = layers.Conv2D(
                    filters=self.output_channels,
                    kernel_size=1,
                    name=f'aux_head_{i}'
                )
                self.auxiliary_heads.append(head)
    
    def call(self, inputs, training=None):
        features = self.backbone(inputs, training=training)
        
        outputs = []
        
        if self.include_top:
            main_output = self.main_head(features[-1])
            outputs.append(main_output)
        else:
            outputs.append(features[-1])
        
        if self.enable_auxiliary_heads and self.include_top:
            for i, head in enumerate(self.auxiliary_heads):
                aux_output = head(features[-(i+2)])
                outputs.append(aux_output)
        
        return outputs if len(outputs) > 1 else outputs[0]
```

### 9.2 Weight Transfer Patterns

```python
def transfer_weights_layerwise(
    source_model: keras.Model,
    target_model: keras.Model,
    layer_mapping: Optional[Dict[str, str]] = None
):
    """Transfer weights layer by layer."""
    if layer_mapping is None:
        layer_mapping = {layer.name: layer.name for layer in source_model.layers}
    
    for source_name, target_name in layer_mapping.items():
        try:
            source_layer = source_model.get_layer(source_name)
            target_layer = target_model.get_layer(target_name)
            
            source_weights = source_layer.get_weights()
            target_shapes = [w.shape for w in target_layer.get_weights()]
            source_shapes = [w.shape for w in source_weights]
            
            if source_shapes == target_shapes:
                target_layer.set_weights(source_weights)
                logger.info(f"Transferred: {source_name} -> {target_name}")
            else:
                logger.warning(f"Shape mismatch for {source_name}")
        except ValueError as e:
            logger.warning(f"Failed to transfer {source_name}: {e}")


def transfer_backbone_weights(
    source_model: keras.Model,
    target_model: keras.Model,
    freeze_backbone: bool = True
):
    """Transfer only backbone weights."""
    target_model.backbone.set_weights(source_model.backbone.get_weights())
    
    if freeze_backbone:
        target_model.backbone.trainable = False
    
    logger.info("Backbone weights transferred")
```

---

## 10. Extension Points and Modularity

### 10.1 Hook Methods Pattern

```python
class ExtensibleModel(keras.Model):
    """Model with extension points via hook methods."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_architecture()
    
    def _build_architecture(self):
        self._build_stem()
        self._build_encoder()
        self._build_bottleneck()
        self._build_decoder()
        self._build_heads()
    
    # Extension points: Override these in subclasses
    def _build_stem(self):
        self.stem = DefaultStem()
    
    def _build_encoder(self):
        self.encoder = DefaultEncoder()
    
    def _build_bottleneck(self):
        self.bottleneck = DefaultBottleneck()
    
    def _build_decoder(self):
        self.decoder = DefaultDecoder()
    
    def _build_heads(self):
        self.main_head = DefaultHead()


class CustomModel(ExtensibleModel):
    """Extended model with custom encoder."""
    
    def _build_encoder(self):
        self.encoder = EfficientNetEncoder(version='b0', pretrained=True)
```

### 10.2 Plugin System Pattern

```python
class PluggableModel(keras.Model):
    """Model with plugin system for components."""
    
    STEM_PLUGINS = {}
    ENCODER_PLUGINS = {}
    DECODER_PLUGINS = {}
    
    @classmethod
    def register_stem(cls, name: str):
        def decorator(plugin_class):
            cls.STEM_PLUGINS[name] = plugin_class
            return plugin_class
        return decorator
    
    @classmethod
    def register_encoder(cls, name: str):
        def decorator(plugin_class):
            cls.ENCODER_PLUGINS[name] = plugin_class
            return plugin_class
        return decorator
    
    def __init__(
        self,
        stem_type: str = 'default',
        encoder_type: str = 'default',
        decoder_type: str = 'default',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.stem = self._create_stem(stem_type)
        self.encoder = self._create_encoder(encoder_type)
        self.decoder = self._create_decoder(decoder_type)
    
    def _create_stem(self, stem_type: str):
        if stem_type not in self.STEM_PLUGINS:
            raise ValueError(f"Unknown stem type: {stem_type}")
        return self.STEM_PLUGINS[stem_type]()


# Register plugins:
@PluggableModel.register_stem('conv')
class ConvStem(keras.layers.Layer):
    ...

@PluggableModel.register_stem('vision_transformer')
class ViTStem(keras.layers.Layer):
    ...

# Use:
model = PluggableModel(stem_type='conv', encoder_type='resnet')
```

### 10.3 Mixin Classes Pattern

```python
class AttentionMixin:
    """Mixin to add attention mechanisms."""
    
    def _add_attention(self, features, name_prefix=''):
        if not hasattr(self, '_attention_modules'):
            self._attention_modules = []
        
        attention = SelfAttention(name=f'{name_prefix}_attention')
        self._attention_modules.append(attention)
        return attention(features)


class DropoutMixin:
    """Mixin to add dropout layers."""
    
    def _add_dropout(self, features, rate=0.5, spatial=False, name_prefix=''):
        if spatial:
            dropout = layers.SpatialDropout2D(rate, name=f'{name_prefix}_dropout')
        else:
            dropout = layers.Dropout(rate, name=f'{name_prefix}_dropout')
        return dropout(features)


class EnhancedModel(AttentionMixin, DropoutMixin, keras.Model):
    """Model with attention and dropout features."""
    
    def call(self, inputs, training=None):
        x = self.encoder(inputs)
        x = self._add_attention(x, name_prefix='encoder')
        x = self._add_dropout(x, rate=0.3, name_prefix='encoder')
        return self.decoder(x)
```

---

## 11. Factory Patterns

### 11.1 Basic Factory Function

```python
def create_model(
    variant: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs
) -> keras.Model:
    """
    Factory function to create models.
    
    Args:
        variant: Model variant ('tiny', 'small', 'base', 'large')
        input_shape: Input dimensions
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        **kwargs: Additional model arguments
    
    Returns:
        Configured model instance
    """
    configs = {
        'tiny': {'depth': 3, 'filters': 32},
        'small': {'depth': 3, 'filters': 48},
        'base': {'depth': 4, 'filters': 64},
        'large': {'depth': 4, 'filters': 96}
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}")
    
    config = configs[variant].copy()
    config.update(kwargs)
    
    model = Model(input_shape=input_shape, num_classes=num_classes, **config)
    
    if pretrained:
        weights_path = download_weights(variant)
        model.load_weights(weights_path)
    
    return model
```

### 11.2 Builder Pattern

```python
class ModelBuilder:
    """
    Builder pattern for complex model construction.
    
    Example:
        model = (ModelBuilder()
            .set_backbone('resnet50')
            .set_decoder('fpn')
            .add_attention('se')
            .add_head('segmentation', num_classes=21)
            .build())
    """
    
    def __init__(self):
        self._config = {
            'backbone': None,
            'decoder': None,
            'attention': [],
            'heads': [],
            'input_shape': (224, 224, 3)
        }
    
    def set_backbone(self, backbone_type: str, **kwargs):
        self._config['backbone'] = {'type': backbone_type, **kwargs}
        return self
    
    def set_decoder(self, decoder_type: str, **kwargs):
        self._config['decoder'] = {'type': decoder_type, **kwargs}
        return self
    
    def add_attention(self, attention_type: str, **kwargs):
        self._config['attention'].append({'type': attention_type, **kwargs})
        return self
    
    def add_head(self, head_type: str, **kwargs):
        self._config['heads'].append({'type': head_type, **kwargs})
        return self
    
    def set_input_shape(self, input_shape: Tuple[int, int, int]):
        self._config['input_shape'] = input_shape
        return self
    
    def build(self) -> keras.Model:
        if self._config['backbone'] is None:
            raise ValueError("Backbone must be set")
        return ConfigurableModel(self._config)
```

---

## 12. Testing and Validation

### 12.1 Comprehensive Test Suite

```python
import pytest
import tempfile
import os

import numpy as np
import keras
from keras import ops


class TestCustomLayer:
    """Comprehensive test suite for custom Keras layers."""
    
    @pytest.fixture
    def layer_config(self):
        return {'units': 64, 'activation': 'relu', 'use_bias': True}
    
    @pytest.fixture
    def sample_input(self):
        return np.random.randn(8, 32).astype(np.float32)
    
    def test_instantiation(self, layer_config):
        """Test layer can be instantiated with valid config."""
        layer = SimpleCustomLayer(**layer_config)
        assert layer.units == layer_config['units']
    
    def test_invalid_config(self):
        """Test layer rejects invalid configuration."""
        with pytest.raises(ValueError, match="units must be positive"):
            SimpleCustomLayer(units=-1)
    
    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass produces correct output shape."""
        layer = SimpleCustomLayer(**layer_config)
        output = layer(sample_input)
        
        expected_shape = (sample_input.shape[0], layer_config['units'])
        assert output.shape == expected_shape
    
    def test_build_creates_weights(self, layer_config, sample_input):
        """Test that build() creates expected weights."""
        layer = SimpleCustomLayer(**layer_config)
        layer(sample_input)  # Triggers build
        
        assert layer.kernel is not None
        assert layer.kernel.shape == (sample_input.shape[-1], layer_config['units'])
        
        if layer_config['use_bias']:
            assert layer.bias is not None
            assert layer.bias.shape == (layer_config['units'],)
    
    def test_training_vs_inference(self, layer_config, sample_input):
        """Test layer behaves correctly in training vs inference mode."""
        layer = SimpleCustomLayer(**layer_config)
        
        train_output = layer(sample_input, training=True)
        infer_output = layer(sample_input, training=False)
        
        np.testing.assert_allclose(
            ops.convert_to_numpy(train_output),
            ops.convert_to_numpy(infer_output),
            rtol=1e-6, atol=1e-6
        )
    
    def test_serialization_cycle(self, layer_config, sample_input):
        """Test full save/load cycle preserves functionality."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = SimpleCustomLayer(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)
        
        original_output = model(sample_input)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
        
        loaded_output = loaded_model(sample_input)
        
        np.testing.assert_allclose(
            ops.convert_to_numpy(original_output),
            ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should match after serialization"
        )
    
    def test_get_config_complete(self, layer_config):
        """Test get_config returns all constructor arguments."""
        layer = SimpleCustomLayer(**layer_config)
        config = layer.get_config()
        
        assert 'units' in config
        assert 'activation' in config
        assert 'use_bias' in config
        assert 'kernel_initializer' in config
        assert 'bias_initializer' in config
    
    def test_from_config_reconstruction(self, layer_config, sample_input):
        """Test layer can be reconstructed from config."""
        original = SimpleCustomLayer(**layer_config)
        original(sample_input)
        
        config = original.get_config()
        reconstructed = SimpleCustomLayer.from_config(config)
        
        assert reconstructed.units == original.units
        assert reconstructed.use_bias == original.use_bias
    
    def test_compute_output_shape(self, layer_config, sample_input):
        """Test compute_output_shape matches actual output."""
        layer = SimpleCustomLayer(**layer_config)
        
        computed_shape = layer.compute_output_shape(sample_input.shape)
        actual_output = layer(sample_input)
        
        assert computed_shape == actual_output.shape
    
    def test_compute_output_shape_before_build(self, layer_config):
        """Test compute_output_shape works before layer is built."""
        layer = SimpleCustomLayer(**layer_config)
        
        # Should work even without calling layer
        input_shape = (None, 32)
        computed_shape = layer.compute_output_shape(input_shape)
        
        assert computed_shape == (None, layer_config['units'])
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_variable_batch_size(self, layer_config, batch_size):
        """Test layer handles various batch sizes."""
        layer = SimpleCustomLayer(**layer_config)
        
        inputs = np.random.randn(batch_size, 32).astype(np.float32)
        output = layer(inputs)
        
        assert output.shape[0] == batch_size
    
    @pytest.mark.parametrize("activation", [None, 'relu', 'gelu', 'tanh'])
    def test_different_activations(self, activation, sample_input):
        """Test layer works with various activation functions."""
        layer = SimpleCustomLayer(units=64, activation=activation)
        output = layer(sample_input)
        
        assert output.shape == (sample_input.shape[0], 64)
```

### 12.2 Integration Testing

```python
def test_complete_pipeline():
    """Test complete training pipeline."""
    model = create_model('base', num_classes=10)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    x_train = np.random.randn(100, 224, 224, 3).astype('float32')
    y_train = np.random.randint(0, 10, size=(100, 224, 224, 1))
    
    model.fit(x_train, y_train, epochs=2, validation_split=0.2, verbose=0)
    
    model.save('test_model.keras')
    loaded_model = keras.models.load_model('test_model.keras')
    
    pred_original = model.predict(x_train[:5], verbose=0)
    pred_loaded = loaded_model.predict(x_train[:5], verbose=0)
    
    np.testing.assert_allclose(pred_original, pred_loaded, rtol=1e-5)
    
    print("✓ Complete pipeline test passed")
```

---

## 13. Common Pitfalls and Solutions

### Pitfall 1: Conditional Layer Creation

**❌ WRONG:**
```python
def __init__(self, use_feature_a=True):
    super().__init__()
    if use_feature_a:
        self.feature_a = FeatureLayer()  # WRONG!
```

**✅ CORRECT:**
```python
def __init__(self, use_feature_a=True):
    super().__init__()
    self.use_feature_a = use_feature_a
    self.feature_a = FeatureLayer()  # Always create

def call(self, inputs):
    x = inputs
    if self.use_feature_a:
        x = self.feature_a(x)  # Conditionally use
    return x
```

### Pitfall 2: Wrong Build Pattern

**❌ WRONG:**
```python
def build(self, input_shape):
    self.dense = layers.Dense(self.units)  # WRONG: Creating in build()!
    super().build(input_shape)
```

**✅ CORRECT:**
```python
def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.dense = layers.Dense(units)  # Create in __init__

def build(self, input_shape):
    self.dense.build(input_shape)  # Build sub-layers
    super().build(input_shape)
```

### Pitfall 3: Missing Registration

**❌ WRONG:**
```python
class MyLayer(keras.layers.Layer):  # No decorator
    pass
# Results in: ValueError: Unknown layer: MyLayer
```

**✅ CORRECT:**
```python
@keras.saving.register_keras_serializable()
class MyLayer(keras.layers.Layer):
    pass
```

### Pitfall 4: Incomplete Configuration

**❌ WRONG:**
```python
def get_config(self):
    return {'units': self.units}  # Missing other parameters!
```

**✅ CORRECT:**
```python
def get_config(self):
    config = super().get_config()
    config.update({
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        # Include ALL __init__ parameters
    })
    return config
```

### Pitfall 5: Graph-Breaking Shape Operations

**❌ WRONG:**
```python
def call(self, inputs):
    shape = ops.shape(inputs)
    shape_list = list(shape)  # Breaks graph!
    batch = int(shape[0])     # Breaks graph!
```

**✅ CORRECT:**
```python
def call(self, inputs):
    shape = ops.shape(inputs)
    batch = shape[0]  # Keep as tensor
    new_shape = ops.stack([batch, self.units])
    return ops.reshape(inputs, new_shape)
```

### Pitfall 6: Python Conditionals on Tensors

**❌ WRONG:**
```python
def call(self, inputs):
    if ops.mean(inputs) > 0:  # Evaluated at trace time only!
        return inputs * 2
    return inputs
```

**✅ CORRECT:**
```python
def call(self, inputs):
    condition = ops.mean(inputs) > 0
    return ops.where(condition, inputs * 2, inputs)
```

### Pitfall 7: Using .numpy() in call()

**❌ WRONG:**
```python
def call(self, inputs):
    values = inputs.numpy()  # Cannot execute in graph!
    return ops.convert_to_tensor(values + 1)
```

**✅ CORRECT:**
```python
def call(self, inputs):
    return inputs + 1.0  # Stay in graph
```

### Pitfall 8: Mutable Default Arguments

**❌ WRONG:**
```python
def __init__(self, layer_sizes=[64, 128]):  # Mutable default!
    self.layer_sizes = layer_sizes
```

**✅ CORRECT:**
```python
def __init__(self, layer_sizes: Optional[List[int]] = None):
    self.layer_sizes = layer_sizes if layer_sizes is not None else [64, 128]
```

### Pitfall 9: Not Building Before Weight Loading

**❌ WRONG:**
```python
model = Model(...)
model.load_weights('weights.h5')  # May fail - model not built
```

**✅ CORRECT:**
```python
model = Model(...)
model.build((None, 224, 224, 3))  # Explicit build
model.load_weights('weights.h5')

# OR
model = Model(...)
dummy_input = np.zeros((1, 224, 224, 3))
model(dummy_input)  # Implicit build via forward pass
model.load_weights('weights.h5')
```

### Pitfall 10: Inconsistent Layer Names

**❌ WRONG:**
```python
for i in range(depth):
    layer = Block()  # No name - gets auto-generated names
    # Names change if depth changes
```

**✅ CORRECT:**
```python
for i in range(depth):
    layer = Block(name=f'block_{i}')  # Explicit names
    # Consistent across configurations
```

### Pitfall 11: Missing compute_output_shape

**❌ WRONG:**
```python
class MyLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units)
    
    def call(self, inputs):
        return self.dense(inputs)
    
    # Missing compute_output_shape!
    # Breaks functional API and sub-layer building
```

**✅ CORRECT:**
```python
class MyLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units)
    
    def call(self, inputs):
        return self.dense(inputs)
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
```

### Pitfall 12: Accessing Weights in compute_output_shape

**❌ WRONG:**
```python
def compute_output_shape(self, input_shape):
    # Fails if layer not built yet!
    return (input_shape[0], self.kernel.shape[-1])
```

**✅ CORRECT:**
```python
def compute_output_shape(self, input_shape):
    # Use stored config, works before build
    return (input_shape[0], self.units)
```

---

## 14. Troubleshooting Guide

### Debug Checklist

When encountering issues, verify in this order:

1. **✅ Registration**: `@keras.saving.register_keras_serializable()` decorator present?
2. **✅ Sub-layer Creation**: All sub-layers created in `__init__()`?
3. **✅ Configuration**: `get_config()` returns ALL `__init__` parameters?
4. **✅ Build Logic**: `build()` method handles sub-layer building if needed?
5. **✅ Output Shape**: `compute_output_shape()` implemented and uses config (not weights)?
6. **✅ Graph Safety**: `call()` uses only `ops` functions, no Python primitives on tensors?
7. **✅ Serialization**: Full save/load test passes?

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Unknown layer: MyCustomLayer` | Missing registration decorator | Add `@keras.saving.register_keras_serializable()` |
| `Layer was never built and thus has no variables` | Sub-layer not built before weight loading | Add explicit `sub_layer.build()` calls in parent's `build()` |
| `RecursionError during serialization` | Circular references in configuration | Store config parameters explicitly, not as `locals()` |
| `Cannot convert symbolic tensor to numpy` | Using `.numpy()` or Python type conversions in `call()` | Use `ops` functions only; keep all operations symbolic |
| `Graph execution error` / `InaccessibleTensorError` | Python conditionals on tensor values in `call()` | Use `ops.where()` or `ops.cond()` for runtime conditionals |
| `Could not compute output shape` | Missing or broken `compute_output_shape()` | Implement method using stored config, not weight shapes |
| `AttributeError: 'NoneType' has no attribute 'shape'` | Accessing weights in `compute_output_shape()` before build | Use `self.units` etc. instead of `self.kernel.shape` |

### Debug Helper

```python
def debug_layer_serialization(layer_class, layer_config, sample_input):
    """Debug helper for layer serialization issues."""
    try:
        layer = layer_class(**layer_config)
        output = layer(sample_input)
        logger.info(f"Forward pass successful: {output.shape}")
        
        config = layer.get_config()
        logger.info(f"Configuration keys: {list(config.keys())}")
        
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = layer_class(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(os.path.join(tmpdir, 'test.keras'))
            loaded = keras.models.load_model(os.path.join(tmpdir, 'test.keras'))
            logger.info("Serialization test passed")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
```

---

## 15. Complete Examples

### 15.1 Framework-Integrated Layer

```python
@keras.saving.register_keras_serializable()
class FrameworkIntegratedLayer(keras.layers.Layer):
    """
    Custom layer demonstrating integration with dl-techniques framework.
    
    Args:
        hidden_size: Dimensionality of the model.
        num_heads: Number of attention heads.
        use_transformer: Whether to include transformer processing.
        **kwargs: Additional arguments for Layer base class.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        use_transformer: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_transformer = use_transformer
        
        # Use framework components
        if use_transformer:
            from dl_techniques.layers.transformer import TransformerLayer
            self.transformer = TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=hidden_size * 4
            )
        else:
            self.transformer = None
        
        from dl_techniques.layers.norms.rms_norm import RMSNorm
        self.norm = RMSNorm()

    def call(
        self, 
        inputs: keras.KerasTensor, 
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        if self.transformer is not None:
            x = self.transformer(inputs, training=training)
        else:
            x = inputs
            
        return self.norm(x, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'use_transformer': self.use_transformer,
        })
        return config
```

### 15.2 Complete Reusable Model

```python
@keras.saving.register_keras_serializable(package='DLTechniques')
class WellStructuredModel(keras.Model):
    """
    Complete example of a well-structured, reusable model.
    
    Demonstrates:
    - Comprehensive configuration management
    - Complete serialization support
    - Weight compatibility across configurations
    - Clear extension points
    
    Args:
        input_shape: Input dimensions (height, width, channels)
        depth: Number of encoder/decoder stages
        initial_filters: Number of filters in first stage
        blocks_per_stage: Number of blocks per stage
        output_channels: Number of output channels
        include_top: Whether to include prediction head
        enable_deep_supervision: Whether to enable auxiliary outputs
        use_bias: Whether to use bias in convolutions
        kernel_initializer: Weight initializer
        kernel_regularizer: Weight regularizer
        **kwargs: Additional arguments for Model base class
    """
    
    PRESETS = {
        'tiny': {'depth': 3, 'initial_filters': 32, 'blocks_per_stage': 2},
        'base': {'depth': 4, 'initial_filters': 64, 'blocks_per_stage': 3},
        'large': {'depth': 5, 'initial_filters': 96, 'blocks_per_stage': 4}
    }
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        depth: int = 4,
        initial_filters: int = 64,
        blocks_per_stage: int = 2,
        output_channels: Optional[int] = None,
        include_top: bool = True,
        enable_deep_supervision: bool = False,
        use_bias: bool = True,
        kernel_initializer: str = 'he_normal',
        kernel_regularizer: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._validate_config(depth, initial_filters, blocks_per_stage)
        
        # Store all configuration (critical for serialization)
        self.input_shape_config = input_shape
        self.depth = depth
        self.initial_filters = initial_filters
        self.blocks_per_stage = blocks_per_stage
        self.output_channels = output_channels if output_channels is not None else input_shape[-1]
        self.include_top = include_top
        self.enable_deep_supervision = enable_deep_supervision
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
        # Build architecture (all layers always created)
        self._build_architecture()
        
        logger.info(f"Model created: depth={depth}, filters={initial_filters}")
    
    def _validate_config(self, depth, initial_filters, blocks_per_stage):
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        if initial_filters <= 0:
            raise ValueError(f"initial_filters must be positive, got {initial_filters}")
        if blocks_per_stage <= 0:
            raise ValueError(f"blocks_per_stage must be positive, got {blocks_per_stage}")
    
    def _build_architecture(self):
        self._build_encoder()
        self._build_decoder()
        self._build_heads()
    
    def _build_encoder(self):
        self.encoder_stages = []
        for level in range(self.depth):
            filters = self.initial_filters * (2 ** level)
            stage_blocks = []
            
            for block_idx in range(self.blocks_per_stage):
                block = layers.Conv2D(
                    filters=filters,
                    kernel_size=3,
                    padding='same',
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'enc_L{level}_blk{block_idx}'
                )
                stage_blocks.append(block)
            
            self.encoder_stages.append(stage_blocks)
    
    def _build_decoder(self):
        self.decoder_stages = []
        for level in reversed(range(self.depth)):
            filters = self.initial_filters * (2 ** level)
            stage_blocks = []
            
            for block_idx in range(self.blocks_per_stage):
                block = layers.Conv2D(
                    filters=filters,
                    kernel_size=3,
                    padding='same',
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'dec_L{level}_blk{block_idx}'
                )
                stage_blocks.append(block)
            
            self.decoder_stages.append(stage_blocks)
    
    def _build_heads(self):
        # Main head - always created
        self.main_head = layers.Conv2D(
            filters=self.output_channels,
            kernel_size=1,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='main_head'
        )
        
        # Auxiliary heads - always created if enabled
        self.auxiliary_heads = []
        if self.enable_deep_supervision:
            for i in range(self.depth - 1):
                head = layers.Conv2D(
                    filters=self.output_channels,
                    kernel_size=1,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'aux_head_{i}'
                )
                self.auxiliary_heads.append(head)
    
    def build(self, input_shape: Tuple[Optional[int], ...]):
        current_shape = input_shape
        
        for stage_blocks in self.encoder_stages:
            for block in stage_blocks:
                block.build(current_shape)
                current_shape = block.compute_output_shape(current_shape)
        
        for stage_blocks in self.decoder_stages:
            for block in stage_blocks:
                block.build(current_shape)
                current_shape = block.compute_output_shape(current_shape)
        
        self.main_head.build(current_shape)
        for head in self.auxiliary_heads:
            head.build(current_shape)
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        x = inputs
        skip_connections = []
        
        for stage_blocks in self.encoder_stages:
            for block in stage_blocks:
                x = block(x)
            skip_connections.append(x)
        
        outputs = []
        for idx, stage_blocks in enumerate(self.decoder_stages):
            for block in stage_blocks:
                x = block(x)
            
            if self.enable_deep_supervision and idx < len(self.auxiliary_heads):
                if self.include_top:
                    aux_out = self.auxiliary_heads[idx](x)
                else:
                    aux_out = x
                outputs.append(aux_out)
        
        if self.include_top:
            main_out = self.main_head(x)
        else:
            main_out = x
        
        if self.enable_deep_supervision:
            return [main_out] + outputs
        return main_out
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_config,
            'depth': self.depth,
            'initial_filters': self.initial_filters,
            'blocks_per_stage': self.blocks_per_stage,
            'output_channels': self.output_channels,
            'include_top': self.include_top,
            'enable_deep_supervision': self.enable_deep_supervision,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer)
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WellStructuredModel':
        if 'kernel_initializer' in config:
            config['kernel_initializer'] = initializers.deserialize(config['kernel_initializer'])
        if 'kernel_regularizer' in config:
            config['kernel_regularizer'] = regularizers.deserialize(config['kernel_regularizer'])
        return cls(**config)
    
    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> 'WellStructuredModel':
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(cls.PRESETS.keys())}")
        
        config = cls.PRESETS[preset].copy()
        config.update(kwargs)
        return cls(**config)
```

---

## Summary Checklist

When designing custom Keras components:

**Configuration Management**
- [ ] All parameters stored as instance attributes
- [ ] Complete `get_config()` implementation
- [ ] Working `from_config()` classmethod
- [ ] Predefined presets for common configurations

**Layer Creation**
- [ ] All layers created in `__init__` (not conditionally)
- [ ] Systematic, consistent layer naming
- [ ] Explicit `build()` method with sub-layer building
- [ ] Usage flags separate from creation

**Shape Handling**
- [ ] `compute_output_shape()` implemented for ALL custom layers
- [ ] Uses stored config (e.g., `self.units`), not weight shapes
- [ ] Works before layer is built
- [ ] Handles `None` dimensions correctly

**Serialization**
- [ ] `@keras.saving.register_keras_serializable()` decorator
- [ ] Complex objects properly serialized/deserialized
- [ ] Model can be saved and loaded completely

**Graph Safety**
- [ ] No Python type conversions on tensors in `call()`
- [ ] Use `ops.where()`/`ops.cond()` for conditionals
- [ ] Use `ops.shape()` for dynamic shapes
- [ ] No `.numpy()` calls in `call()`

**Weight Compatibility**
- [ ] Same layers exist regardless of configuration
- [ ] Consistent layer names across configurations
- [ ] Support for `skip_mismatch` weight loading

**Testing**
- [ ] Unit tests for creation, forward pass, shapes
- [ ] Serialization tests
- [ ] `compute_output_shape()` matches actual output
- [ ] Weight compatibility tests
- [ ] Integration tests

---

This guide provides a complete framework for building maintainable, reusable Keras components that integrate seamlessly with the dl-techniques framework. The modern approach is actually simpler than outdated patterns - let Keras handle the complexity while you focus on the layer logic.