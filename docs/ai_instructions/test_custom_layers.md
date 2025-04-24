# Comprehensive Guide to Testing Custom Keras Layers

When creating custom layers for Keras, thorough testing is essential to ensure reliability, particularly for serialization and edge cases. This guide outlines a structured approach to testing custom layers.

## Setting Up Testing Infrastructure

Using pytest for structured testing provides organization and reusability:

```python
import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from your_module import YourCustomLayer

class TestYourCustomLayer:
    """Test suite for YourCustomLayer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return tf.random.normal([4, 32, 32, 64])
        
    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return YourCustomLayer(filters=32)
```

## Essential Test Categories

### 1. Initialization Tests

```python
def test_initialization_defaults(self):
    """Test initialization with default parameters."""
    layer = YourCustomLayer(filters=128)
    
    # Check default values
    assert layer.filters == 128
    assert layer.use_bias is True
    assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
    assert layer.bias_regularizer is None
    assert layer.activation == "relu"
    # ...etc

def test_initialization_custom(self):
    """Test initialization with custom parameters."""
    custom_regularizer = keras.regularizers.L2(1e-4)
    
    layer = YourCustomLayer(
        filters=64,
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=custom_regularizer,
        activation="gelu",
    )
    
    # Check custom values
    assert layer.filters == 64
    assert layer.use_bias is False
    assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
    assert layer.kernel_regularizer == custom_regularizer
    assert layer.activation == "gelu"

def test_invalid_parameters(self):
    """Test that invalid parameters raise appropriate errors."""
    with pytest.raises(ValueError):
        YourCustomLayer(filters=-10)  # Negative filters
        
    with pytest.raises(ValueError):
        YourCustomLayer(filters=64, data_format="invalid_format")
```

### 2. Build Process Tests

```python
def test_build_process(self, input_tensor):
    """Test that the layer builds properly."""
    layer = YourCustomLayer(filters=64)
    layer(input_tensor)  # Forward pass triggers build
    
    # Check that weights were created
    assert layer.built is True
    assert len(layer.weights) > 0
    assert hasattr(layer, "kernel")
    assert layer.kernel.shape == (input_tensor.shape[-1], 64)
    
    if hasattr(layer, "sublayers"):
        assert all(sublayer.built for sublayer in layer.sublayers)
```

### 3. Output Shape Tests

```python
def test_output_shapes(self, input_tensor):
    """Test that output shapes are computed correctly."""
    filters_to_test = [32, 64, 128]
    
    for filters in filters_to_test:
        layer = YourCustomLayer(filters=filters)
        output = layer(input_tensor)
        
        # Check output shape
        expected_shape = list(input_tensor.shape)
        expected_shape[-1] = filters
        expected_shape = tuple(expected_shape)
        assert output.shape == expected_shape
        
        # Test compute_output_shape separately
        computed_shape = layer.compute_output_shape(input_tensor.shape)
        assert computed_shape == expected_shape
```

### 4. Forward Pass Tests

```python
def test_forward_pass(self, input_tensor):
    """Test that forward pass produces expected values."""
    layer = YourCustomLayer(filters=64)
    output = layer(input_tensor)
    
    # Basic sanity checks
    assert not np.any(np.isnan(output.numpy()))
    assert not np.any(np.isinf(output.numpy()))
    
    # Test with controlled inputs for deterministic output
    controlled_input = tf.ones([1, 2, 2, 3])
    deterministic_layer = YourCustomLayer(
        filters=2,
        kernel_initializer="ones",
        bias_initializer="zeros",
        activation="linear"
    )
    result = deterministic_layer(controlled_input)
    
    # With these settings, we can predict the exact output values
    expected = np.ones([1, 2, 2, 2]) * 3  # Each output is sum of inputs (3)
    assert np.allclose(result.numpy(), expected)
```

### 5. Activation and Configuration Tests

```python
def test_different_activations(self, input_tensor):
    """Test layer with different activation functions."""
    activations = ["relu", "gelu", "swish", "linear"]
    
    for act in activations:
        layer = YourCustomLayer(filters=64, activation=act)
        output = layer(input_tensor)
        
        # Check output is valid
        assert not np.any(np.isnan(output.numpy()))
```

### 6. Serialization Tests

```python
def test_serialization(self):
    """Test serialization and deserialization of the layer."""
    original_layer = YourCustomLayer(
        filters=128,
        use_bias=True,
        kernel_initializer="he_normal",
        activation="gelu"
    )
    
    # Build the layer
    input_shape = (32, 32, 64)
    original_layer.build((None,) + input_shape)
    
    # Get configs
    config = original_layer.get_config()
    build_config = original_layer.get_build_config()
    
    # Recreate the layer
    recreated_layer = YourCustomLayer.from_config(config)
    recreated_layer.build_from_config(build_config)
    
    # Check configuration matches
    assert recreated_layer.filters == original_layer.filters
    assert recreated_layer.use_bias == original_layer.use_bias
    assert recreated_layer.activation == original_layer.activation
    
    # Check weights match (shapes should be the same at minimum)
    for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
        assert w1.shape == w2.shape
```

### 7. Model Integration Tests

```python
def test_model_integration(self, input_tensor):
    """Test the layer in a model context."""
    # Create a simple model with the custom layer
    inputs = keras.Input(shape=input_tensor.shape[1:])
    x = YourCustomLayer(filters=32)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = YourCustomLayer(filters=64)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(10)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
    )
    
    # Test forward pass
    y_pred = model(input_tensor, training=False)
    assert y_pred.shape == (input_tensor.shape[0], 10)
```

### 8. Model Save/Load Tests

```python
def test_model_save_load(self, input_tensor):
    """Test saving and loading a model with the custom layer."""
    # Create a model with the custom layer
    inputs = keras.Input(shape=input_tensor.shape[1:])
    x = YourCustomLayer(filters=32, name="custom_layer1")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = YourCustomLayer(filters=64, name="custom_layer2")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(10)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Generate a prediction before saving
    original_prediction = model.predict(input_tensor)
    
    # Create temporary directory for model
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, "model.keras")
        
        # Save the model
        model.save(model_path)
        
        # Load the model
        loaded_model = keras.models.load_model(
            model_path,
            custom_objects={"YourCustomLayer": YourCustomLayer}
        )
        
        # Generate prediction with loaded model
        loaded_prediction = loaded_model.predict(input_tensor)
        
        # Check predictions match
        assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)
        
        # Check layer types are preserved
        assert isinstance(loaded_model.get_layer("custom_layer1"), YourCustomLayer)
        assert isinstance(loaded_model.get_layer("custom_layer2"), YourCustomLayer)
```

### 9. Training and Gradient Tests

```python
def test_gradient_flow(self, input_tensor):
    """Test gradient flow through the layer."""
    layer = YourCustomLayer(filters=64)
    
    # Watch the variables
    with tf.GradientTape() as tape:
        inputs = tf.Variable(input_tensor)
        outputs = layer(inputs)
        loss = tf.reduce_mean(tf.square(outputs))
    
    # Get gradients
    grads = tape.gradient(loss, layer.trainable_variables)
    
    # Check gradients exist and are not None
    assert all(g is not None for g in grads)
    
    # Check gradients have values (not all zeros)
    assert all(np.any(g.numpy() != 0) for g in grads)

def test_training_loop(self, input_tensor):
    """Test training loop with the custom layer."""
    # Create a model with the custom layer
    model = keras.Sequential([
        keras.layers.InputLayer(input_tensor.shape[1:]),
        YourCustomLayer(32),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(),
        YourCustomLayer(64),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(10)
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Create mock data
    x_train = tf.random.normal([32] + list(input_tensor.shape[1:]))
    y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)
    
    # Initial loss
    initial_loss = model.evaluate(x_train, y_train, verbose=0)[0]
    
    # Train for a few epochs
    model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)
    
    # Final loss
    final_loss = model.evaluate(x_train, y_train, verbose=0)[0]
    
    # Loss should decrease
    assert final_loss < initial_loss
```

### 10. Edge Case and Robustness Tests

```python
def test_numerical_stability(self):
    """Test layer stability with extreme input values."""
    layer = YourCustomLayer(filters=16)
    
    # Create inputs with different magnitudes
    batch_size = 2
    height, width = 8, 8
    channels = 4
    
    test_cases = [
        tf.zeros((batch_size, height, width, channels)),  # Zeros
        tf.ones((batch_size, height, width, channels)) * 1e-10,  # Very small values
        tf.ones((batch_size, height, width, channels)) * 1e10,  # Very large values
        tf.random.normal((batch_size, height, width, channels)) * 1e5  # Large random values
    ]
    
    for test_input in test_cases:
        output = layer(test_input)
        
        # Check for NaN/Inf values
        assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
        assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

def test_regularization(self, input_tensor):
    """Test that regularization losses are properly applied."""
    # Create layer with regularization
    layer = YourCustomLayer(
        filters=32,
        kernel_regularizer=keras.regularizers.L2(0.1),
        bias_regularizer=keras.regularizers.L1(0.1)
    )
    
    # No regularization losses before calling the layer
    assert len(layer.losses) == 0
    
    # Apply the layer
    _ = layer(input_tensor)
    
    # Should have regularization losses now
    assert len(layer.losses) > 0
```

## Important Testing Considerations

### Shape Handling Testing

Test that your layer correctly handles different shape inputs and conversions:

```python
def test_shape_handling(self):
    """Test shape handling with different input formats."""
    layer = YourCustomLayer(filters=32)
    
    # Test with tuple shape
    tuple_shape = (None, 16, 16, 8)
    output_shape = layer.compute_output_shape(tuple_shape)
    assert output_shape == (None, 16, 16, 32)
    
    # Test with list shape
    list_shape = [None, 16, 16, 8]
    output_shape = layer.compute_output_shape(list_shape)
    assert output_shape == (None, 16, 16, 32)
    
    # Test with channels_first format if supported
    if hasattr(layer, "data_format"):
        layer.data_format = "channels_first"
        output_shape = layer.compute_output_shape((None, 8, 16, 16))
        assert output_shape == (None, 32, 16, 16)
```

### Serialization Edge Cases

Test serialization with different parameter types:

```python
def test_serialization_edge_cases(self):
    """Test serialization with various parameter types."""
    # Test with different initializer types
    layers_to_test = [
        YourCustomLayer(filters=32, kernel_initializer="glorot_uniform"),
        YourCustomLayer(filters=32, kernel_initializer=keras.initializers.GlorotUniform()),
        YourCustomLayer(filters=32, kernel_regularizer="l2"),
        YourCustomLayer(filters=32, kernel_regularizer=keras.regularizers.L2(0.01)),
    ]
    
    for original_layer in layers_to_test:
        # Build the layer
        original_layer.build((None, 16, 16, 8))
        
        # Get config and recreate
        config = original_layer.get_config()
        recreated_layer = YourCustomLayer.from_config(config)
        
        # Check key aspects match
        assert recreated_layer.filters == original_layer.filters
        # Additional checks specific to each parameter type
```

## Running the Test Suite

To ensure comprehensive testing, run all tests in the suite:

```python
if __name__ == '__main__':
    pytest.main([__file__])
```

## Essential Checks for Any Custom Layer

Every custom layer test suite should include:

1. **Serialization Tests**: Verify that the layer can be properly serialized and deserialized
2. **Model Save/Load Tests**: Ensure the layer works correctly after saving and loading a model
3. **Shape Handling Tests**: Confirm that the layer handles different shape formats properly
4. **Edge Case Tests**: Test with extreme values to ensure numerical stability
5. **Integration Tests**: Verify that the layer works correctly in a model context

By following this comprehensive testing approach, you can ensure your custom layers will be reliable in production environments.
