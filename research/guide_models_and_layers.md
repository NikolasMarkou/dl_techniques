# Complete Guide to Modern Keras 3 Custom Layers and Models

A comprehensive, authoritative guide for creating robust, serializable, and production-ready custom Layers and Models in Keras 3, following modern best practices that eliminate common build and serialization errors.

## Table of Contents

1. [The Golden Rule: Create vs. Build](#1-the-golden-rule-create-vs-build)
2. [Essential Setup and Registration](#2-essential-setup-and-registration)
3. [Layer Implementation Patterns](#3-layer-implementation-patterns)
4. [Model Implementation Patterns](#4-model-implementation-patterns)
5. [Serialization Lifecycle](#5-serialization-lifecycle)
6. [Documentation and Type Safety](#6-documentation-and-type-safety)
7. [Comprehensive Testing](#7-comprehensive-testing)
8. [Framework Integration](#8-framework-integration)
9. [Common Pitfalls and Solutions](#9-common-pitfalls-and-solutions)
10. [Advanced Patterns](#10-advanced-patterns)
11. [Troubleshooting Guide](#11-troubleshooting-guide)

---

## 1. The Golden Rule: Create vs. Build

This is the **most critical concept** in modern Keras 3. Understanding this separation eliminates 99% of build and serialization errors.

### The Fundamental Principle

The key insight is that Keras serialization and deserialization follow a predictable lifecycle:

1. **Saving**: Keras calls `get_config()` to get reconstruction arguments and saves weights of all built layers
2. **Loading**: Keras calls `__init__()` with saved config to recreate layers (everything is **unbuilt**)
3. **Building**: Keras calls `build()` to create weight variables for both parent and child layers
4. **Restoring**: With weight variables existing, Keras loads saved weight values

### The Rules

- **`__init__(self, ...)`: CREATE sub-layers and store configuration**
  - Instantiate ALL sub-layers that your layer will use
  - Store ALL configuration arguments as instance attributes
  - **NEVER** create weights directly (`self.add_weight`) here
  - **NEVER** inspect `input_shape` or call operations that require shapes

- **`build(self, input_shape)`: CREATE weights and BUILD sub-layers**
  - Create the layer's own weights using `self.add_weight()`
  - **CRITICAL**: For layers with sub-layers, explicitly call `build()` on each sub-layer
  - Always call `super().build(input_shape)` at the end
  - This ensures all weight variables exist before weight restoration

### Why This Matters

If you don't build sub-layers in your `build()` method, Keras will try to load weights into unbuilt layers that have no weight variables, causing:

```
ValueError: Layer 'dense_1' was never built and thus it doesn't have any variables.
```

---

## 2. Essential Setup and Registration

### Core Imports

```python
# Core Keras imports - always use full paths
import keras
from keras import ops, layers, initializers, regularizers, constraints, activations
from typing import Optional, Union, Tuple, List, Dict, Any, Callable, Literal
import numpy as np

# For testing (backend-specific)
import tensorflow as tf

# DL-Techniques framework imports
from dl_techniques.utils.logger import logger
```

### Registration Decorator

**CRITICAL**: Every custom class MUST be registered for serialization:

```python
@keras.saving.register_keras_serializable()
class YourCustomLayer(keras.layers.Layer):
    # Your implementation
    pass
```

Without this decorator, you'll get "Unknown layer" errors during model loading.

---

## 3. Layer Implementation Patterns

### Pattern 1: Simple Layer (No Sub-layers)

For layers that only need their own weights:

```python
@keras.saving.register_keras_serializable()
class SimpleCustomLayer(keras.layers.Layer):
    """
    Custom linear transformation layer with optional bias and activation.
    
    This layer performs a learnable linear transformation similar to Dense layer,
    demonstrating the fundamental pattern for layers that create their own weights
    but contain no sub-layers. It showcases proper weight initialization, forward
    pass computation, and serialization handling.
    
    **Intent**: Educational example showing basic custom layer patterns with
    weight management, while being functionally equivalent to keras.layers.Dense.
    
    **Architecture**:
    ```
    Input(shape=[..., input_dim])
           ↓
    Linear Transform: X @ W  
           ↓
    Add Bias: + b (if use_bias=True)
           ↓  
    Activation: f(·) (if activation provided)
           ↓
    Output(shape=[..., units])
    ```
    
    **Mathematical Operation**:
        output = activation(inputs @ kernel + bias)
    
    Where:
    - @ denotes matrix multiplication
    - kernel has shape (input_dim, units) 
    - bias has shape (units,) if use_bias=True
    - activation is applied element-wise
    
    Args:
        units: Integer, dimensionality of the output space. Must be positive.
            This determines the number of neurons/outputs in the layer.
        activation: Optional activation function. Can be string name ('relu', 'gelu')
            or callable. None means linear activation (no transformation). 
            Defaults to None.
        use_bias: Boolean, whether to add a learnable bias vector to outputs.
            When True, adds bias term after linear transformation. Defaults to True.
        kernel_initializer: Initializer for the weight matrix. Accepts string names
            ('glorot_uniform', 'he_normal') or Initializer instances.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the bias vector. Only used when use_bias=True.
            Defaults to 'zeros'.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).
    
    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        Most common: 2D tensor with shape `(batch_size, input_dim)`.
    
    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        Same rank as input, but last dimension changed to `units`.
    
    Attributes:
        kernel: Weight matrix of shape (input_dim, units). Created in build().
        bias: Bias vector of shape (units,) if use_bias=True. Created in build().
    
    Example:
        ```python
        # Basic linear layer (no activation)
        layer = SimpleCustomLayer(64)
        inputs = keras.Input(shape=(784,))
        outputs = layer(inputs)  # Shape: (batch, 64)
        
        # With activation function
        layer = SimpleCustomLayer(128, activation='relu')
        
        # Without bias (linear transformation only)
        layer = SimpleCustomLayer(64, use_bias=False)
        ```
    
    Note:
        This implementation demonstrates proper weight creation in build() method,
        which is essential for Keras serialization. For production code, prefer
        using keras.layers.Dense which is optimized and battle-tested.
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
        """
        Create the layer's own weights.
        
        This is called automatically when the layer first processes input.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")
        
        # Create layer's own weights
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

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
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

### Pattern 2: Composite Layer (With Sub-layers)

For layers that contain other layers and need explicit building:

```python
@keras.saving.register_keras_serializable()
class CompositeLayer(keras.layers.Layer):
    """
    Multi-stage neural network block demonstrating sub-layer composition patterns.
    
    This layer implements a standard neural network building block consisting of
    two linear transformations with dropout and optional normalization in between.
    It demonstrates the critical pattern for layers containing sub-layers, showing
    proper initialization, building, and serialization handling.
    
    **Intent**: Educational example showcasing composite layer patterns for building
    complex neural network blocks from simpler components, while handling Keras
    serialization requirements correctly.
    
    **Architecture**:
    ```
    Input(shape=[..., input_dim])
           ↓
    Dense₁(hidden_dim, activation='gelu')
           ↓
    Dropout(rate=dropout_rate) 
           ↓
    LayerNorm() ← (optional, if use_norm=True)
           ↓
    Dense₂(output_dim, activation=None)
           ↓
    Output(shape=[..., output_dim])
    ```
    
    **Data Flow**: 
    1. Linear transformation with GELU activation to expand/contract dimensionality
    2. Dropout for regularization during training
    3. Layer normalization for training stability (optional)
    4. Final linear transformation to target output dimension
    
    This pattern is commonly found in transformer feed-forward networks,
    MLP blocks, and general deep learning architectures.
    
    Args:
        hidden_dim: Integer, dimensionality of the intermediate hidden layer.
            Controls the expressiveness of the transformation. Should be positive.
        output_dim: Integer, dimensionality of the final output. Must be positive.
        dropout_rate: Float between 0 and 1, fraction of input units to randomly
            set to 0 during training. Applied between first dense and normalization.
            Defaults to 0.1.
        use_norm: Boolean, whether to apply layer normalization after dropout.
            Helps with training stability in deep networks. Defaults to True.
        **kwargs: Additional arguments for Layer base class.
    
    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        
    Output shape:
        N-D tensor with shape: `(batch_size, ..., output_dim)`.
        Same rank as input, but last dimension becomes output_dim.
    
    Attributes:
        dense1: First Dense layer for feature transformation.
        dropout: Dropout layer for regularization.
        norm: LayerNormalization layer (if use_norm=True), else None.
        dense2: Final Dense layer for output projection.
    
    Example:
        ```python
        # Standard MLP block
        layer = CompositeLayer(hidden_dim=128, output_dim=64)
        inputs = keras.Input(shape=(32,))
        outputs = layer(inputs)  # Shape: (batch, 64)
        
        # Without normalization, higher dropout
        layer = CompositeLayer(
            hidden_dim=256, 
            output_dim=128, 
            dropout_rate=0.3,
            use_norm=False
        )
        
        # Transformer-style FFN block
        layer = CompositeLayer(
            hidden_dim=2048,  # 4x expansion 
            output_dim=512,   # back to model dim
            dropout_rate=0.1,
            use_norm=True
        )
        ```
        
    Note:
        This pattern demonstrates CRITICAL sub-layer building in build() method.
        Without explicit sub-layer building, serialization will fail when loading
        the model due to missing weight variables.
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
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        
        # Store configuration
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        
        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.dense1 = layers.Dense(hidden_dim, activation="gelu", name="dense1")
        self.dropout = layers.Dropout(dropout_rate, name="dropout")
        
        if use_norm:
            self.norm = layers.LayerNormalization(name="layer_norm")
        else:
            self.norm = None
            
        self.dense2 = layers.Dense(output_dim, name="dense2")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.
        
        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        # Build sub-layers in computational order
        self.dense1.build(input_shape)
        
        # Compute intermediate shapes
        dense1_output_shape = self.dense1.compute_output_shape(input_shape)
        
        self.dropout.build(dense1_output_shape)
        
        if self.norm is not None:
            # Dropout doesn't change shape
            self.norm.build(dense1_output_shape)
            
        self.dense2.build(dense1_output_shape)
        
        # Always call parent build at the end
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
            
        x = self.dense2(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate,
            'use_norm': self.use_norm,
        })
        return config
```

### Pattern 3: Sub-layers Only (No Custom Weights)

For layers that only orchestrate other layers:

```python
@keras.saving.register_keras_serializable()
class OrchestrationLayer(keras.layers.Layer):
    """
    Sequential multi-layer perceptron builder from configurable layer dimensions.
    
    This layer creates a stack of Dense layers with specified hidden dimensions
    and consistent activation functions. It demonstrates the pattern for layers
    that only orchestrate sub-layers without creating custom weights, where Keras
    can handle sub-layer building automatically.
    
    **Intent**: Provide a flexible way to create deep MLPs with configurable
    architecture from a simple list of dimensions, useful for rapid prototyping
    and hyperparameter experiments.
    
    **Architecture**:
    ```
    Input(shape=[..., input_dim])
           ↓
    Dense₁(hidden_dims[0], activation)
           ↓  
    Dense₂(hidden_dims[1], activation)
           ↓
          ...
           ↓
    Denseₙ(hidden_dims[n-1], activation) 
           ↓
    Output(shape=[..., hidden_dims[-1]])
    ```
    
    **Example Flow** (hidden_dims=[128, 64, 32]):
    ```
    Input([batch, 784]) → Dense(128) → Dense(64) → Dense(32) → Output([batch, 32])
    ```
    
    Each Dense layer applies: activation(X @ W + b)
    
    Args:
        hidden_dims: List of integers specifying the output dimension of each
            Dense layer in sequence. Must be non-empty with positive integers.
            Example: [512, 256, 128] creates 3 layers with those output sizes.
        activation: String name of activation function to apply to all Dense layers.
            Common choices: 'relu', 'gelu', 'tanh', 'sigmoid'. Defaults to 'relu'.
        **kwargs: Additional arguments for Layer base class.
    
    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        
    Output shape:
        N-D tensor with shape: `(batch_size, ..., hidden_dims[-1])`.
        Final dimension becomes the last value in hidden_dims list.
    
    Attributes:
        layers_list: List of Dense layers created from hidden_dims specification.
    
    Example:
        ```python
        # Create 3-layer MLP: input → 512 → 256 → 128
        layer = OrchestrationLayer([512, 256, 128], activation='relu')
        inputs = keras.Input(shape=(784,))
        outputs = layer(inputs)  # Shape: (batch, 128)
        
        # Single layer transformation
        layer = OrchestrationLayer([64], activation='gelu')
        
        # Deep network with GELU activations  
        layer = OrchestrationLayer([1024, 512, 256, 128, 64], activation='gelu')
        ```
        
    Note:
        This pattern doesn't require custom build() method since the layer only
        contains standard Keras layers. Keras automatically handles building
        of sub-layers on first call. Useful for rapid architecture prototyping.
    """
    
    def __init__(
        self,
        hidden_dims: List[int],
        activation: str = 'relu',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        # Validate inputs
        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All dimensions in hidden_dims must be positive")
        
        self.hidden_dims = hidden_dims
        self.activation = activation
        
        # Create all sub-layers
        self.layers_list = []
        for i, dim in enumerate(hidden_dims):
            self.layers_list.append(
                layers.Dense(dim, activation=activation, name=f'dense_{i}')
            )

    def call(
        self, 
        inputs: keras.KerasTensor, 
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass - Keras builds sub-layers automatically."""
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
        })
        return config
```

---

## 4. Model Implementation Patterns

### Basic Custom Model

For custom models, Keras handles sub-layer building automatically:

```python
@keras.saving.register_keras_serializable()
class CustomModel(keras.Model):
    """
    Multi-layer perceptron classifier with configurable architecture and dropout.
    
    This model implements a standard deep neural network for classification tasks,
    consisting of multiple hidden layers with dropout regularization and a final
    softmax classification layer. It demonstrates proper model-level patterns for
    organizing layers and configuration management.
    
    **Intent**: Provide a flexible, configurable MLP classifier that can be easily
    adapted for different classification tasks by adjusting layer dimensions,
    dropout rates, and activation functions.
    
    **Architecture**:
    ```
    Input(shape=[..., input_features])
           ↓
    Hidden₁(hidden_dims[0], activation) → Dropout(dropout_rate)
           ↓
    Hidden₂(hidden_dims[1], activation) → Dropout(dropout_rate) 
           ↓
          ...
           ↓
    Hiddenₙ(hidden_dims[n-1], activation) → Dropout(dropout_rate)
           ↓
    Classifier(num_classes, activation='softmax')
           ↓
    Output(shape=[..., num_classes]) # Probability distribution
    ```
    
    **Example Architecture** (hidden_dims=[512, 256], num_classes=10):
    ```
    Input([batch, 784])
         ↓
    Dense(512, 'relu') → Dropout(0.1)
         ↓  
    Dense(256, 'relu') → Dropout(0.1)
         ↓
    Dense(10, 'softmax')
         ↓
    Output([batch, 10])  # Probabilities for 10 classes
    ```
    
    Each hidden layer performs: dropout(activation(X @ W + b), training)
    Final layer performs: softmax(X @ W + b) for probability distribution
    
    Args:
        num_classes: Integer, number of output classes for classification.
            Must be positive. This determines the final layer size and should
            match the number of target classes in your dataset.
        hidden_dims: List of integers specifying hidden layer dimensions.
            Each integer creates a Dense layer with that many units.
            Defaults to [512, 256].
        dropout_rate: Float between 0 and 1, fraction of hidden layer outputs
            randomly set to 0 during training. Applied after each hidden layer
            but not after the final classifier. Defaults to 0.1.
        activation: String name of activation function for hidden layers.
            The final layer always uses softmax. Defaults to 'relu'.
        **kwargs: Additional arguments for Model base class.
    
    Input shape:
        2D tensor with shape: `(batch_size, input_features)`.
        
    Output shape:
        2D tensor with shape: `(batch_size, num_classes)`.
        Values represent class probabilities (sum to 1 due to softmax).
    
    Attributes:
        hidden_layers: List of Dense layers for feature learning.
        dropout_layers: List of Dropout layers for regularization.
        classifier: Final Dense layer with softmax for classification.
    
    Example:
        ```python
        # MNIST classifier (10 classes, 784 input features)
        model = CustomModel(num_classes=10, hidden_dims=[512, 256])
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Binary classifier with deeper network
        model = CustomModel(
            num_classes=2,
            hidden_dims=[1024, 512, 256, 128],
            dropout_rate=0.2,
            activation='gelu'
        )
        
        # Simple single hidden layer
        model = CustomModel(num_classes=3, hidden_dims=[128])
        ```
        
    Note:
        For models, Keras automatically handles sub-layer building, so no custom
        build() method is needed. The model can be compiled and trained directly
        after instantiation.
    """
    
    def __init__(
        self,
        num_classes: int,
        hidden_dims: List[int] = [512, 256],
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        # Validate inputs
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All dimensions in hidden_dims must be positive")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        
        # Store configuration
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # CREATE all sub-layers in __init__
        self.hidden_layers = []
        self.dropout_layers = []
        
        for i, dim in enumerate(hidden_dims):
            self.hidden_layers.append(
                layers.Dense(dim, activation=activation, name=f'hidden_{i}')
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_{i}')
            )
        
        self.classifier = layers.Dense(num_classes, activation='softmax', name='classifier')

    def call(
        self, 
        inputs: keras.KerasTensor, 
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the model."""
        x = inputs
        
        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x, training=training)
            x = dropout_layer(x, training=training)
        
        return self.classifier(x, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
        })
        return config
```

---

## 5. Serialization Lifecycle

Understanding the serialization lifecycle is key to robust implementations:

### The Save Process

```python
# 1. Model is trained and ready
model = CustomModel(num_classes=10)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# 2. Save triggers get_config() on all layers/models
model.save('my_model.keras')  # Calls get_config() + saves weights
```

### The Load Process

```python
# 1. Keras reads config and recreates layers with __init__()
# 2. All layers are unbuilt at this point
# 3. When model is first used, build() is called to create weight variables
# 4. Saved weights are loaded into the weight variables

loaded_model = keras.models.load_model('my_model.keras')  # Uses registered layers
```

### Critical Serialization Test

```python
def test_serialization_cycle(layer_class, layer_config, sample_input):
    """The most important test - full serialization cycle."""
    # 1. Create original layer in a model
    inputs = keras.Input(shape=sample_input.shape[1:])
    layer_output = layer_class(**layer_config)(inputs)
    model = keras.Model(inputs, layer_output)
    
    # 2. Get prediction from original
    original_prediction = model(sample_input)
    
    # 3. Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_model.keras')
        model.save(filepath)
        
        loaded_model = keras.models.load_model(filepath)
        loaded_prediction = loaded_model(sample_input)
        
        # 4. Verify identical outputs
        np.testing.assert_allclose(
            ops.convert_to_numpy(original_prediction),
            ops.convert_to_numpy(loaded_prediction),
            rtol=1e-6, atol=1e-6
        )
```

---

## 6. Documentation and Type Safety

### Comprehensive Docstring Template

```python
@keras.saving.register_keras_serializable()
class DocumentedLayer(keras.layers.Layer):
    """
    Feature scaling layer with learnable parameters and statistical normalization.
    
    This layer applies learnable feature scaling combined with standardization,
    demonstrating comprehensive documentation practices for custom Keras layers.
    It computes running statistics during training and applies both normalization
    and learnable scaling/shifting parameters.
    
    **Intent**: Provide a template for comprehensive layer documentation while
    implementing a practical feature scaling layer that combines normalization
    with learnable parameters for improved training dynamics.
    
    **Architecture**:
    ```
    Input(shape=[..., features])
           ↓
    Compute: μ = mean(x), σ² = var(x)  
           ↓
    Normalize: x_norm = (x - μ) / √(σ² + ε)
           ↓
    Scale & Shift: output = γ * x_norm + β
           ↓
    Output(shape=[..., features])
    ```
    
    **Mathematical Operations**:
    1. **Statistics**: μ = E[x], σ² = Var[x]
    2. **Normalization**: x̂ = (x - μ) / √(σ² + ε) 
    3. **Affine Transform**: y = γ ⊙ x̂ + β
    
    Where:
    - μ, σ² are computed per feature across batch dimension
    - γ (scale) and β (shift) are learnable parameters
    - ε is a small constant for numerical stability
    - ⊙ denotes element-wise multiplication
    
    Args:
        units: Integer, number of features to normalize. Must be positive.
            Should match the last dimension of input tensors.
        epsilon: Float, small constant added to variance for numerical stability.
            Prevents division by zero when variance is very small. Defaults to 1e-5.
        center: Boolean, if True, add learnable bias/shift parameter β.
            When False, no bias is applied after normalization. Defaults to True.
        scale: Boolean, if True, add learnable scale parameter γ. 
            When False, normalized features are not scaled. Defaults to True.
        beta_initializer: Initializer for bias parameter β (if center=True).
            Defaults to 'zeros'.
        gamma_initializer: Initializer for scale parameter γ (if scale=True).
            Defaults to 'ones'.
        **kwargs: Additional Layer base class arguments (name, trainable, etc.).
    
    Input shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        The last dimension must equal the `units` parameter.
        Most common: 2D with shape `(batch_size, units)`.
    
    Output shape:
        N-D tensor with same shape as input: `(batch_size, ..., units)`.
        Shape is preserved through the normalization process.
    
    Attributes:
        gamma: Scale parameter of shape (units,) if scale=True, else None.
        beta: Bias parameter of shape (units,) if center=True, else None.
    
    Example:
        ```python
        # Basic usage with learnable scale and bias
        layer = DocumentedLayer(units=64)
        inputs = keras.Input(shape=(784,))
        normalized = layer(inputs)
        
        # Only normalization, no learnable parameters
        layer = DocumentedLayer(units=128, center=False, scale=False)
        
        # Custom initialization for specific distributions
        layer = DocumentedLayer(
            units=256,
            gamma_initializer='he_normal',
            beta_initializer='normal'
        )
        
        # In a model with other layers
        inputs = keras.Input(shape=(784,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = DocumentedLayer(units=256)(x)  # Normalize features
        outputs = layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        ```
    
    References:
        - Layer Normalization: https://arxiv.org/abs/1607.06450
        - Batch Normalization: https://arxiv.org/abs/1502.03167
    
    Raises:
        ValueError: If units is not positive.
        ValueError: If epsilon is not positive.
        
    Note:
        This layer computes statistics per batch during training. For inference
        or when using in evaluation mode, consider implementing moving averages
        similar to BatchNormalization for more stable behavior.
    """
    
    def __init__(
        self,
        units: int,
        epsilon: float = 1e-5,
        center: bool = True,
        scale: bool = True,
        beta_initializer: Union[str, initializers.Initializer] = 'zeros',
        gamma_initializer: Union[str, initializers.Initializer] = 'ones',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        
        # Store configuration
        self.units = units
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        
        # Weight attributes (created in build)
        self.gamma = None
        self.beta = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create learnable parameters."""
        if input_shape[-1] != self.units:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) must match units ({self.units})"
            )
        
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=(self.units,),
                initializer=self.gamma_initializer,
                trainable=True
            )
            
        if self.center:
            self.beta = self.add_weight(
                name='beta', 
                shape=(self.units,),
                initializer=self.beta_initializer,
                trainable=True
            )
            
        super().build(input_shape)

    def call(
        self, 
        inputs: keras.KerasTensor, 
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply feature normalization and scaling."""
        # Compute statistics along all axes except the last (feature) dimension
        reduction_axes = list(range(len(inputs.shape) - 1))
        
        mean = ops.mean(inputs, axis=reduction_axes, keepdims=True)
        variance = ops.var(inputs, axis=reduction_axes, keepdims=True)
        
        # Normalize
        normalized = (inputs - mean) / ops.sqrt(variance + self.epsilon)
        
        # Apply learnable parameters
        if self.scale:
            normalized = normalized * self.gamma
        if self.center:
            normalized = normalized + self.beta
            
        return normalized

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
        })
        return config
```

### Type Hints Best Practices

```python
from typing import Optional, Union, Tuple, List, Dict, Any, Callable, Literal

def __init__(
    self,
    units: int,  # Specific types
    activation: Optional[Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]]] = None,
    dropout_rate: float = 0.1,
    layer_type: Literal['encoder', 'decoder'] = 'encoder',  # Constrained choices
    **kwargs: Any  # Generic for **kwargs
) -> None:

def compute_output_shape(
    self, 
    input_shape: Tuple[Optional[int], ...]  # Shape tuples
) -> Tuple[Optional[int], ...]:

def call(
    self, 
    inputs: keras.KerasTensor, 
    training: Optional[bool] = None
) -> keras.KerasTensor:
```

---

## 7. Comprehensive Testing

### Essential Test Suite

```python
import pytest
import tempfile
import os
from typing import Any, Dict

class TestCustomLayer:
    """Comprehensive test suite for custom layers."""
    
    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'units': 64,
            'activation': 'relu',
            'use_bias': True
        }
    
    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing."""
        return ops.random.normal(shape=(4, 32))
    
    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = CompositeLayer(**layer_config)
        
        assert hasattr(layer, 'hidden_dim')
        assert not layer.built
        assert layer.dense1 is not None  # Sub-layers created
    
    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass and building."""
        layer = CompositeLayer(**layer_config)
        
        output = layer(sample_input)
        
        assert layer.built
        assert output.shape[0] == sample_input.shape[0]  # Batch size preserved
    
    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = CompositeLayer(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)
        
        # Get original prediction
        original_pred = model(sample_input)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)
            
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)
            
            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )
    
    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = CompositeLayer(**layer_config)
        config = layer.get_config()
        
        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"
    
    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        layer = CompositeLayer(**layer_config)
        
        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = ops.mean(ops.square(output))
        
        gradients = tape.gradient(loss, layer.trainable_variables)
        
        assert all(g is not None for g in gradients)
        assert len(gradients) > 0
    
    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = CompositeLayer(**layer_config)
        
        output = layer(sample_input, training=training)
        assert output.shape[0] == sample_input.shape[0]
    
    def test_edge_cases(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            CompositeLayer(hidden_dim=0, output_dim=32)  # Invalid hidden_dim
        
        with pytest.raises(ValueError):
            CompositeLayer(hidden_dim=32, output_dim=-5)  # Invalid output_dim

# Run tests with: pytest test_custom_layer.py -v
```

---

## 8. Framework Integration

### Using DL-Techniques Components

```python
@keras.saving.register_keras_serializable()
class FrameworkIntegratedLayer(keras.layers.Layer):
    """
    Hybrid layer integrating dl-techniques framework components with standard Keras layers.
    
    This layer demonstrates how to properly integrate specialized components from the
    dl-techniques framework (such as TransformerLayer and RMSNorm) with standard 
    Keras layers, while maintaining proper serialization and configuration management.
    
    **Intent**: Show best practices for combining framework-specific components with
    custom logic, enabling modular architecture development while preserving all
    Keras functionality including serialization, compilation, and training.
    
    **Architecture**:
    ```
    Input(shape=[..., hidden_size])
           ↓
    TransformerLayer(hidden_size, num_heads) ← (if use_transformer=True)
           ↓                                      
    RMSNorm(normalized features)               ← (always applied)
           ↓
    Output(shape=[..., hidden_size])
    ```
    
    **Component Details**:
    - **TransformerLayer**: Advanced transformer block with configurable attention
    - **RMSNorm**: Root Mean Square normalization for stable training
    - **Conditional Architecture**: Can disable transformer for simpler processing
    
    This pattern enables sophisticated architectures while maintaining clean interfaces.
    
    Args:
        hidden_size: Integer, feature dimension throughout the layer. Must be positive.
            This size is maintained through all transformations.
        num_heads: Integer, number of attention heads in transformer layer.
            Must be positive and divide evenly into hidden_size. Defaults to 8.
        use_transformer: Boolean, whether to include transformer processing.
            When False, only applies normalization. Useful for ablation studies. 
            Defaults to True.
        **kwargs: Additional arguments for Layer base class.
    
    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.
        
    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.
        Dimensions are preserved through processing.
    
    Attributes:
        transformer: TransformerLayer instance if use_transformer=True, else None.
        norm: RMSNorm layer for feature normalization.
    
    Example:
        ```python
        # Full transformer + normalization
        layer = FrameworkIntegratedLayer(hidden_size=768, num_heads=12)
        inputs = keras.Input(shape=(128, 768))  # seq_len=128
        outputs = layer(inputs)
        
        # Normalization only (no transformer)
        layer = FrameworkIntegratedLayer(
            hidden_size=512, 
            use_transformer=False
        )
        
        # Custom head count
        layer = FrameworkIntegratedLayer(
            hidden_size=1024, 
            num_heads=16
        )
        ```
    
    Note:
        When integrating framework components, ensure they are properly imported
        and available. This pattern allows for modular replacement of components
        while maintaining consistent interfaces.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        use_transformer: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        # Validate inputs
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
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
            
        # Framework normalization
        from dl_techniques.layers.norms.rms_norm import RMSNorm
        self.norm = RMSNorm()

    def call(self, inputs, training=None):
        """Forward pass using framework components."""
        if self.transformer is not None:
            x = self.transformer(inputs, training=training)
        else:
            x = inputs
            
        return self.norm(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'use_transformer': self.use_transformer,
        })
        return config
```

### Model with Framework Optimization

```python
def create_model_with_framework_optimization(config: Dict[str, Any]) -> keras.Model:
    """
    Create model using dl-techniques optimization components.
    """
    from dl_techniques.optimization import (
        optimizer_builder,
        learning_rate_schedule_builder
    )
    
    # Create model
    model = CustomModel(**config['model'])
    
    # Framework optimization
    lr_config = config.get('learning_rate', {
        "type": "cosine_decay",
        "learning_rate": 0.001,
        "decay_steps": 10000
    })
    
    optimizer_config = config.get('optimizer', {
        "type": "adamw",
        "gradient_clipping_by_norm": 1.0
    })
    
    # Build optimization
    lr_schedule = learning_rate_schedule_builder(lr_config)
    optimizer = optimizer_builder(optimizer_config, lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

---

## 9. Common Pitfalls and Solutions

### Pitfall 1: Wrong Build Pattern

**❌ WRONG:**
```python
def build(self, input_shape):
    # DON'T create sub-layers in build()
    self.dense = layers.Dense(self.units)  # WRONG!
    super().build(input_shape)
```

**✅ CORRECT:**
```python
def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.dense = layers.Dense(units)  # Create in __init__

def build(self, input_shape):
    # Build sub-layers if needed for serialization
    self.dense.build(input_shape)
    super().build(input_shape)
```

### Pitfall 2: Missing Registration

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

### Pitfall 3: Incomplete Configuration

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

### Pitfall 4: Deprecated Methods

**❌ WRONG:**
```python
def get_build_config(self):  # Deprecated!
    return {"input_shape": self._build_input_shape}

def build_from_config(self, config):  # Deprecated!
    self.build(config["input_shape"])
```

**✅ CORRECT:**
```python
# Simply delete these methods entirely
# Modern Keras handles building automatically
```

---

## 10. Advanced Patterns

### Pattern 1: Dynamic Architecture

```python
@keras.saving.register_keras_serializable()
class DynamicLayer(keras.layers.Layer):
    """
    Configurable layer builder supporting multiple layer types from specification.
    
    This layer creates a heterogeneous sequence of different layer types (Dense, Conv2D, etc.)
    based on a configuration list, demonstrating dynamic architecture construction while
    maintaining proper serialization. Each layer's configuration is preserved and can be
    reconstructed during model loading.
    
    **Intent**: Enable flexible model architectures defined by configuration files or
    hyperparameter search, allowing rapid experimentation with different layer combinations
    without hard-coding specific architectures.
    
    **Architecture** (example configuration):
    ```
    config = [
        {'type': 'dense', 'units': 128, 'activation': 'relu'},
        {'type': 'dense', 'units': 64, 'activation': 'gelu'},
        {'type': 'conv2d', 'filters': 32, 'kernel_size': 3}
    ]
    
    Input → Dense(128, 'relu') → Dense(64, 'gelu') → Conv2D(32, 3) → Output
    ```
    
    **Configuration Format**:
    Each dict in layer_configs must contain:
    - 'type': Layer type ('dense', 'conv2d', etc.)  
    - Additional keys: Parameters for that layer type
    
    Args:
        layer_configs: List of dictionaries specifying layer configurations.
            Each dict must contain 'type' key and appropriate parameters for that layer.
            Example: [{'type': 'dense', 'units': 64, 'activation': 'relu'}]
        **kwargs: Additional arguments for Layer base class.
    
    Input shape:
        Variable, depends on the first layer in the configuration.
        
    Output shape:
        Variable, depends on the final layer in the configuration.
    
    Attributes:
        dynamic_layers: List of created layers based on configuration.
    
    Example:
        ```python
        # Mixed architecture
        config = [
            {'type': 'dense', 'units': 256, 'activation': 'gelu'},
            {'type': 'dense', 'units': 128, 'activation': 'relu'},
            {'type': 'dense', 'units': 64}
        ]
        layer = DynamicLayer(config)
        
        # For image processing (if input is 4D)
        config = [
            {'type': 'conv2d', 'filters': 32, 'kernel_size': 3},
            {'type': 'conv2d', 'filters': 64, 'kernel_size': 3}
        ]
        layer = DynamicLayer(config)
        ```
        
    Note:
        Configuration reconstruction during serialization requires careful handling
        of layer types. This pattern is powerful for hyperparameter optimization
        and architecture search applications.
    """
    
    def __init__(
        self,
        layer_configs: List[Dict[str, Any]],
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        if not layer_configs:
            raise ValueError("layer_configs cannot be empty")
        
        self.layer_configs = layer_configs
        
        # Create layers from configs
        self.dynamic_layers = []
        for i, config in enumerate(layer_configs):
            # Make a copy to avoid modifying original
            config_copy = config.copy()
            layer_type = config_copy.pop('type')
            
            if layer_type == 'dense':
                layer = layers.Dense(**config_copy, name=f'dynamic_dense_{i}')
            elif layer_type == 'conv2d':
                layer = layers.Conv2D(**config_copy, name=f'dynamic_conv2d_{i}')
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            self.dynamic_layers.append(layer)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.dynamic_layers:
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        
        # Reconstruct configs with type information
        configs_with_types = []
        for layer, original_config in zip(self.dynamic_layers, self.layer_configs):
            layer_config = layer.get_config()
            
            if isinstance(layer, layers.Dense):
                layer_config['type'] = 'dense'
            elif isinstance(layer, layers.Conv2D):
                layer_config['type'] = 'conv2d'
            
            configs_with_types.append(layer_config)
        
        config.update({'layer_configs': configs_with_types})
        return config
```

### Pattern 2: Stateful Layer

```python
@keras.saving.register_keras_serializable()
class StatefulLayer(keras.layers.Layer):
    """
    Memory-augmented layer with persistent internal state across forward passes.
    
    This layer maintains an internal memory state that gets updated with each forward
    pass, creating a form of neural memory that persists across batches. The state
    influences the current output and gets updated based on the current computation,
    enabling temporal dependencies and memory-based processing.
    
    **Intent**: Demonstrate stateful processing where the layer maintains information
    across different inputs, useful for online learning, adaptive processing, or 
    recurrent-like behavior without explicit sequence structure.
    
    **Architecture & State Flow**:
    ```
    Input(shape=[..., input_dim])
           ↓
    Dense Transform: h = Dense(inputs)  
           ↓
    Memory Integration: output = h + memory_state
           ↓
    State Update: memory_state ← mean(output, batch_axis)
           ↓  
    Output(shape=[..., units])
    
    Memory State: [units] (persistent across calls)
    ```
    
    **State Dynamics**:
    - **Initialization**: memory_state = zeros(units)
    - **Forward Pass**: output = dense(input) + memory_state
    - **State Update**: memory_state = mean(output) across batch dimension
    - **Reset**: memory_state can be manually reset to zeros
    
    Args:
        units: Integer, output dimension and memory state size. Must be positive.
        **kwargs: Additional arguments for Layer base class.
    
    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        
    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
    
    Attributes:
        dense: Dense layer for input transformation.
        memory_state: Persistent state vector, shape (units,).
    
    Methods:
        reset_states(): Manually reset memory state to zeros.
    
    Example:
        ```python
        # Create stateful layer
        layer = StatefulLayer(units=128)
        
        # Process multiple inputs (state accumulates)
        input1 = keras.random.normal((32, 64))
        output1 = layer(input1)  # State updated
        
        input2 = keras.random.normal((32, 64)) 
        output2 = layer(input2)  # Uses updated state from input1
        
        # Reset state manually
        layer.reset_states()
        output3 = layer(input1)  # Same as output1 (state reset)
        
        # Force state reset during forward pass
        output = layer(inputs, reset_state=True)
        ```
        
    Note:
        The state is not trainable (trainable=False) but affects training through
        the forward pass. State persists across batches during training, which
        may require careful handling for reproducibility.
    """
    
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        
        self.units = units
        self.stateful = True
        
        # Sub-layers
        self.dense = layers.Dense(units)
        
        # State created in build()
        self.memory_state = None

    def build(self, input_shape):
        # Create state variable
        self.memory_state = self.add_weight(
            name='memory_state',
            shape=(self.units,),
            initializer='zeros',
            trainable=False,  # State is not trainable
        )
        
        # Build sub-layers
        self.dense.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None, reset_state=False):
        if reset_state:
            self.memory_state.assign(ops.zeros_like(self.memory_state))
        
        x = self.dense(inputs, training=training)
        
        # Use memory state
        x = x + self.memory_state
        
        # Update memory state
        new_state = ops.mean(x, axis=0)
        self.memory_state.assign(new_state)
        
        return x

    def reset_states(self):
        """Reset memory state to zeros."""
        if self.memory_state is not None:
            self.memory_state.assign(ops.zeros_like(self.memory_state))

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config
```

---

## 11. Troubleshooting Guide

### Debug Checklist

When encountering issues, verify in this order:

1. **✅ Registration**: `@keras.saving.register_keras_serializable()` decorator present?
2. **✅ Sub-layer Creation**: All sub-layers created in `__init__()`?
3. **✅ Configuration**: `get_config()` returns ALL `__init__` parameters?
4. **✅ Build Logic**: `build()` method handles sub-layer building if needed?
5. **✅ Serialization**: Full save/load test passes?

### Common Errors and Solutions

#### "Unknown layer: MyCustomLayer"
- **Cause**: Missing registration decorator
- **Solution**: Add `@keras.saving.register_keras_serializable()`

#### "Layer was never built and thus has no variables"
- **Cause**: Sub-layer not built before weight loading
- **Solution**: Add explicit `sub_layer.build()` calls in parent's `build()` method

#### "RecursionError during serialization"
- **Cause**: Circular references in configuration
- **Solution**: Store config parameters explicitly, not as `locals()`

### Debug Helper

```python
def debug_layer_serialization(layer_class, layer_config, sample_input):
    """Debug helper for layer serialization issues."""
    from dl_techniques.utils.logger import logger
    
    try:
        # Test basic functionality
        layer = layer_class(**layer_config)
        output = layer(sample_input)
        logger.info(f"✅ Forward pass successful: {output.shape}")
        
        # Test configuration
        config = layer.get_config()
        logger.info(f"✅ Configuration keys: {list(config.keys())}")
        
        # Test serialization
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = layer_class(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(os.path.join(tmpdir, 'test.keras'))
            loaded = keras.models.load_model(os.path.join(tmpdir, 'test.keras'))
            logger.info("✅ Serialization test passed")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise
```

---

## Conclusion

This guide establishes the correct patterns for modern Keras 3 custom components. The key principles are:

1. **Create in `__init__`**: Instantiate all sub-layers during initialization
2. **Build appropriately**: For layers with sub-layers, explicitly build them in `build()`
3. **Register everything**: Use `@keras.saving.register_keras_serializable()` 
4. **Complete configuration**: `get_config()` must include all constructor parameters
5. **Test thoroughly**: The serialization cycle test is non-negotiable

Following these patterns ensures robust, serializable components that integrate seamlessly with Keras and the dl-techniques framework. The modern approach is actually simpler than outdated patterns - let Keras handle the complexity while you focus on the layer logic.