# Creating a Proper Custom Layer in Keras 3.x

This guide covers the essential steps to create robust custom Keras layers that follow best practices, ensuring they're properly serializable, backend-agnostic, and reliable in production environments.

## Basic Layer Structure

A proper Keras layer needs to implement these key methods:

```python
import keras
from keras import ops
from typing import Optional, Union, Any, Dict, Tuple, List

class MyCustomLayer(keras.layers.Layer):
    """Custom layer with configurable units.
    
    Args:
        units: Integer, dimensionality of the output space.
        activation: Activation function to use.
        use_bias: Boolean, whether the layer uses a bias vector.
        **kwargs: Additional keyword arguments to pass to the Layer base class.
    """
    def __init__(
        self, 
        units: int,
        activation: Optional[Union[str, callable]] = None,
        use_bias: bool = True,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        
        # These will be initialized in build()
        self.kernel = None
        self.bias = None
        
        # Store the build shape for serialization
        self._build_input_shape = None
        
    def build(self, input_shape):
        """Create the layer's weights based on input shape.
        
        Args:
            input_shape: Shape tuple (tuple of integers) or a list of shape tuples,
                indicating the input shape of the layer.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape
        
        # Create weights when input shape is known
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias"
            )
            
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """Forward computation with optional activation.
        
        Args:
            inputs: Input tensor or list/tuple of input tensors.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.
                
        Returns:
            A tensor containing the computation result after optional activation.
        """
        outputs = ops.matmul(inputs, self.kernel)
        
        if self.use_bias:
            outputs = outputs + self.bias
            
        if self.activation is not None:
            outputs = self.activation(outputs)
            
        return outputs
        
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.
        
        Args:
            input_shape: Shape of the input.
            
        Returns:
            Output shape.
        """
        # Convert input_shape to a list for consistent manipulation
        input_shape_list = list(input_shape)
        
        # Return as tuple for consistency
        return tuple(input_shape_list[:-1] + [self.units])
        
    def get_config(self):
        """Returns the layer configuration for serialization.
        
        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
        })
        return config
        
    def get_build_config(self):
        """Get the config needed to build the layer from a config.
        
        This method is needed for proper model saving and loading.
        
        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }
        
    def build_from_config(self, config):
        """Build the layer from a config created with get_build_config.
        
        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])
```

## Key Components Explained

### 1. Initialization (`__init__`)

- Always call `super().__init__(**kwargs)` to properly pass base layer parameters (like `name`, `dtype`, etc.)
- Store configuration parameters as instance attributes
- Initialize weight attributes to `None`
- Use typing hints for better documentation
- Avoid creating weights here - defer to `build()`

### 2. Building Weights (`build`)

- Create weights lazily when input shape is known
- Store `input_shape` for serialization with `self._build_input_shape = input_shape`
- Use `self.add_weight()` to create and track weights
- Always provide meaningful names for weights
- Call `super().build(input_shape)` at the end
- This approach makes your layer flexible and able to adapt to different input shapes

### 3. Forward Computation (`call`)

- Implement the layer's computation logic
- Use operations from `keras.ops` namespace for backend compatibility
- Include `training` parameter even if not used directly
- Always pass the `training` parameter to sublayers
- Can include other parameters like `mask` if needed

### 4. Shape Computation (`compute_output_shape`)

- Always implement this method for proper shape inference
- Convert input shapes to lists before manipulation and back to tuples when returning
- Handle shape manipulations consistently
- Match the logic in your `call` method

### 5. Serialization (`get_config`, `get_build_config`, `build_from_config`)

- Implement `get_config()` to capture all constructor parameters
- Implement `get_build_config()` to save build-specific information
- Implement `build_from_config()` to rebuild the layer after loading
- Don't include weights - they're handled automatically
- Use `serialize` methods for complex objects (initializers, regularizers, etc.)

## Proper Serialization and Deserialization

Proper serialization is crucial for production environments where models are saved, loaded, and deployed. Here's a comprehensive approach:

### Registering Your Layer

Always register your custom layer with Keras:

```python
@keras.saving.register_keras_serializable()
class MyCustomLayer(keras.layers.Layer):
    # layer implementation
```

### Handling Shape Information

When working with shapes, be consistent and avoid serialization issues:

```python
def compute_output_shape(self, input_shape):
    # Convert to list for consistent manipulation
    input_shape_list = list(input_shape)
    
    # Manipulate as list
    output_shape_list = input_shape_list[:-1] + [self.units]
    
    # Return as tuple
    return tuple(output_shape_list)
```

This approach handles all shape types that Keras might use internally.

### Storing Build Information

Store the input shape during build and implement the necessary methods:

```python
def build(self, input_shape):
    # Store for serialization
    self._build_input_shape = input_shape
    # Rest of build method...

def get_build_config(self):
    return {"input_shape": self._build_input_shape}

def build_from_config(self, config):
    if config.get("input_shape") is not None:
        self.build(config["input_shape"])
```

### Loading Models with Custom Layers

When loading models with custom layers, always provide the custom objects:

```python
model = keras.models.load_model(
    "path/to/model.keras",
    custom_objects={"MyCustomLayer": MyCustomLayer}
)
```

## Handling Sublayers and Dependencies

### Managing Sublayers

For complex layers with sublayers:

```python
def build(self, input_shape):
    self._build_input_shape = input_shape
    
    # Create sublayers in build
    self.dense1 = keras.layers.Dense(64)
    self.dense2 = keras.layers.Dense(32)
    
    # Build sublayers explicitly
    self.dense1.build(input_shape)
    
    # Calculate intermediate shape
    intermediate_shape = list(input_shape)
    intermediate_shape[-1] = 64
    self.dense2.build(tuple(intermediate_shape))
    
    super().build(input_shape)

def call(self, inputs, training=None):
    x = self.dense1(inputs, training=training)
    x = keras.activations.relu(x)
    return self.dense2(x, training=training)
```

### Internal Implementation of Dependencies 

```python
# Implement functions internally rather than importing them
def _get_activation_fn(self, activation_name):
    if activation_name == "relu":
        return keras.activations.relu
    elif activation_name == "gelu":
        return keras.activations.gelu
    else:
        return keras.activations.get(activation_name)
        
def build(self, input_shape):
    # ...
    self.activation_fn = self._get_activation_fn(self.activation_name)
```

## Backend Compatibility

Use Keras backend-agnostic operations for compatibility across TensorFlow, JAX, and PyTorch:

```python
# Good - works with any backend
from keras import ops
result = ops.matmul(x, y)

# Avoid - TensorFlow specific
import tensorflow as tf
result = tf.matmul(x, y)
```

## Weight Management

```python
# Properly define weights with clear names and initializers
self.kernel = self.add_weight(
    name="kernel",
    shape=(input_dim, self.units),
    initializer=self.kernel_initializer,
    regularizer=self.kernel_regularizer,
    constraint=self.kernel_constraint,
    trainable=True,
)
```

## Training & Inference Behavior

Some layers behave differently during training versus inference:

```python
def call(self, inputs, training=None):
    """Forward computation with conditional behavior based on training mode."""
    # Different behavior during training vs inference
    if training:
        # Apply training-specific logic (e.g., dropout)
        return some_training_operation(inputs)
    else:
        # Apply inference-specific logic
        return some_inference_operation(inputs)
```

Always propagate the training flag to sublayers:

```python
def call(self, inputs, training=None):
    """Forward computation with training flag propagation to sublayers."""
    # Pass training to all sublayers
    x = self.sublayer1(inputs, training=training)
    x = self.sublayer2(x, training=training)
    return x
```

## Common Pitfalls and Solutions

### Shape Handling Errors

**Problem**: Mixing tuple and list operations with shapes can cause serialization errors.

**Solution**: Convert shapes to lists before manipulation, then back to tuples when returning:

```python
def compute_output_shape(self, input_shape):
    input_shape_list = list(input_shape)
    return tuple(input_shape_list[:-1] + [self.filters])
```

### Missing Sublayers After Loading

**Problem**: Sublayers are None after loading a saved model.

**Solution**: Implement proper build methods and store build input shape:

```python
def get_build_config(self):
    return {"input_shape": self._build_input_shape}

def build_from_config(self, config):
    if config.get("input_shape") is not None:
        self.build(config["input_shape"])
```

### Function Dependencies

**Solution**: Implement necessary functionality as internal methods:

```python
def _get_activation(self, name):
    # Internal implementation
    return keras.activations.get(name)
```

## Example Test Methods

```python
def test_serialization(self):
    """Test serialization of the layer."""
    # Create and build the layer
    original_layer = MyCustomLayer(units=64)
    original_layer.build((None, 32))
    
    # Create some test data
    x = tf.random.normal((2, 32))
    original_output = original_layer(x)
    
    # Get config and recreate layer
    config = original_layer.get_config()
    build_config = original_layer.get_build_config()
    
    new_layer = MyCustomLayer.from_config(config)
    new_layer.build_from_config(build_config)
    
    # Check output matches
    new_output = new_layer(x)
    assert tf.reduce_all(tf.equal(original_output, new_output))
    
def test_model_save_load(self):
    """Test saving and loading a model with the custom layer."""
    # Create a model with the layer
    inputs = keras.Input(shape=(32,))
    x = MyCustomLayer(units=64)(inputs)
    model = keras.Model(inputs=inputs, outputs=x)
    
    # Test data
    test_input = tf.random.normal((2, 32))
    original_output = model(test_input)
    
    # Save and load model
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, "model.keras")
        model.save(model_path)
        
        loaded_model = keras.models.load_model(
            model_path,
            custom_objects={"MyCustomLayer": MyCustomLayer}
        )
        
        # Check output matches
        loaded_output = loaded_model(test_input)
        assert tf.reduce_all(tf.equal(original_output, loaded_output))
```

## Complete Example: Advanced Custom Layer

Here's a complete example of a robust custom layer:

```python
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os
from typing import Optional, Union, Any, Dict

@keras.saving.register_keras_serializable()
class AdvancedCustomLayer(keras.layers.Layer):
    """An advanced custom layer with robust serialization.
    
    Args:
        units: Integer, dimensionality of the output space.
        activation: String name of activation function or callable.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer for the kernel weights.
        bias_regularizer: Regularizer for the bias vector.
        **kwargs: Additional keyword arguments for the Layer parent class.
    """
    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, callable]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        
        # Store config parameters
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        
        # Will be set up in build()
        self.kernel = None
        self.bias = None
        self.activation_fn = None
        self._build_input_shape = None
        
    def _get_activation_fn(self, activation):
        """Internal method to get activation function."""
        if activation is None:
            return None
        return keras.activations.get(activation)
        
    def build(self, input_shape):
        """Build the layer weights and transformations.
        
        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store for serialization
        self._build_input_shape = input_shape
        
        # Get last dimension for weight shape
        input_dim = input_shape[-1]
        
        # Create weights
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
            )
            
        # Set up activation function
        self.activation_fn = self._get_activation_fn(self.activation)
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """Forward computation.
        
        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.
                
        Returns:
            Output tensor after computation.
        """
        outputs = ops.matmul(inputs, self.kernel)
        
        if self.use_bias:
            outputs = outputs + self.bias
            
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
            
        return outputs
        
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.
        
        Args:
            input_shape: Shape of the input.
            
        Returns:
            Output shape.
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)
        
        # Return as tuple for consistency
        return tuple(input_shape_list[:-1] + [self.units])
        
    def get_config(self):
        """Returns configuration for serialization.
        
        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config
        
    def get_build_config(self):
        """Get build configuration.
        
        Returns:
            Dictionary containing build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }
        
    def build_from_config(self, config):
        """Build from configuration.
        
        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])
```

## Conclusion

Creating robust custom Keras layers requires careful implementation of serialization methods, consistent shape handling, and proper management of sublayers and dependencies. By following these best practices, you can create layers that work reliably across different backends and are safe to use in production environments.

Remember to:
- Register your layer with `@keras.saving.register_keras_serializable()`
- Implement `get_config()`, `get_build_config()`, and `build_from_config()`
- Handle shapes consistently by converting to lists during manipulation
- Store build input shape for serialization
- Implement functionality internally as needed
- Test serialization thoroughly

Following these guidelines will help you create custom layers that integrate seamlessly with Keras's serialization system and can be reliably saved, loaded, and deployed in production.
