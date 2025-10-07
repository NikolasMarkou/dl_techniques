import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile


import math
from keras import ops
from typing import Optional, Union, Tuple, Dict, Any
from dl_techniques.layers.hierarchical_routing import HierarchicalRoutingLayer


@keras.saving.register_keras_serializable()
class HierarchicalRoutingLayer(keras.layers.Layer):
    def __init__(
            self,
            output_dim: int,
            epsilon: float = 1e-7,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            use_bias: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(output_dim, int) or output_dim <= 1:
            raise ValueError(
                f"The 'output_dim' must be an integer greater than 1, "
                f"but received: {output_dim}"
            )
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.padded_output_dim = 1 << (output_dim - 1).bit_length()
        self.num_decisions = int(math.log2(self.padded_output_dim))

        self.decision_dense = keras.layers.Dense(
            units=self.num_decisions,
            use_bias=self.use_bias,
            activation='sigmoid',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='decision_dense'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.decision_dense.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        decision_probs = self.decision_dense(inputs, training=training)
        batch_size = ops.shape(inputs)[0]
        padded_probs = ops.ones((batch_size, 1), dtype=self.compute_dtype)
        for i in range(self.num_decisions):
            p_go_right = decision_probs[:, i:i + 1]
            p_go_left = 1.0 - p_go_right
            probs_for_left_branches = padded_probs * p_go_left
            probs_for_right_branches = padded_probs * p_go_right
            combined = ops.stack(
                [probs_for_left_branches, probs_for_right_branches], axis=2
            )
            padded_probs = ops.reshape(combined, (batch_size, 2 ** (i + 1)))
        if self.output_dim == self.padded_output_dim:
            return padded_probs
        else:
            unnormalized_probs = padded_probs[:, :self.output_dim]
            prob_sum = ops.sum(unnormalized_probs, axis=-1, keepdims=True)
            final_probs = unnormalized_probs / (prob_sum + self.epsilon)
            return final_probs

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
            'output_dim': self.output_dim,
            'epsilon': self.epsilon,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


# --- End of Layer Definition ---


class TestHierarchicalRoutingLayer:
    """Test suite for HierarchicalRoutingLayer implementation."""

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        """Create a test input tensor of shape (batch_size, features)."""
        return tf.random.normal([4, 16])

    @pytest.fixture(params=[8, 10])
    def output_dim(self, request):
        """Provides both power-of-2 and non-power-of-2 output dimensions."""
        return request.param

    def test_initialization(self):
        """Test initialization of the layer."""
        # Test a non-power-of-2 dimension
        layer = HierarchicalRoutingLayer(output_dim=10)
        assert isinstance(layer, keras.layers.Layer)
        assert layer.output_dim == 10
        assert layer.padded_output_dim == 16
        assert layer.num_decisions == 4
        assert isinstance(layer.decision_dense, keras.layers.Dense)
        assert layer.decision_dense.units == 4

        # Test a power-of-2 dimension
        layer = HierarchicalRoutingLayer(output_dim=8)
        assert layer.output_dim == 8
        assert layer.padded_output_dim == 8
        assert layer.num_decisions == 3

    def test_invalid_initialization(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            HierarchicalRoutingLayer(output_dim=1)
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            HierarchicalRoutingLayer(output_dim=0)

    def test_build_process(self, input_tensor):
        """Test that the layer and its sub-layers build properly."""
        layer = HierarchicalRoutingLayer(output_dim=10)
        assert not layer.built
        assert not layer.decision_dense.built

        # Forward pass triggers build
        layer(input_tensor)

        assert layer.built
        assert layer.decision_dense.built
        # The dense layer should have one weight (kernel) since use_bias=False
        assert len(layer.weights) == 1
        assert len(layer.trainable_weights) == 1
        # Kernel shape: (input_features, num_decisions)
        assert layer.weights[0].shape == (16, 4)

    def test_output_shapes(self, input_tensor, output_dim):
        """Test that output shapes are computed correctly."""
        layer = HierarchicalRoutingLayer(output_dim=output_dim)
        output = layer(input_tensor)

        # Check output shape from the call
        expected_shape = (input_tensor.shape[0], output_dim)
        assert output.shape == expected_shape

        # Test compute_output_shape separately
        computed_shape = layer.compute_output_shape(input_tensor.shape)
        assert computed_shape == expected_shape

    def test_output_is_normalized(self, input_tensor, output_dim):
        """Test that the output is a valid probability distribution (sums to 1)."""
        layer = HierarchicalRoutingLayer(output_dim=output_dim)
        output = layer(input_tensor)

        # The sum of probabilities for each sample in the batch should be 1.0
        sums = tf.reduce_sum(output, axis=1).numpy()
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_probability_calculation_manual(self):
        """Test the core probability routing logic with deterministic weights."""
        input_dim = 2
        output_dim = 6  # Non-power-of-2, padded to 8 (3 decisions)

        layer = HierarchicalRoutingLayer(output_dim=output_dim, use_bias=True)
        inputs = tf.ones((1, input_dim))  # Simple input

        # Build layer to create weights
        layer(inputs)

        # Manually set weights for predictable sigmoid outputs
        # We want sigmoid outputs of [1.0, 0.0, 0.5]
        # To get sigmoid(x) -> 1, x must be large positive (e.g., 20)
        # To get sigmoid(x) -> 0, x must be large negative (e.g., -20)
        # To get sigmoid(x) -> 0.5, x must be 0
        kernel = np.zeros((input_dim, layer.num_decisions), dtype=np.float32)
        bias = np.array([20.0, -20.0, 0.0], dtype=np.float32)
        layer.decision_dense.set_weights([kernel, bias])

        # Expected decision probabilities: d1=1.0, d2=0.0, d3=0.5
        # Expected padded probabilities (for 8 leaves):
        # Leaf 0 (000): (1-d1)(1-d2)(1-d3) = 0 * 1 * 0.5 = 0
        # Leaf 1 (001): (1-d1)(1-d2)(d3)   = 0 * 1 * 0.5 = 0
        # Leaf 2 (010): (1-d1)(d2)(1-d3)   = 0 * 0 * 0.5 = 0
        # Leaf 3 (011): (1-d1)(d2)(d3)     = 0 * 0 * 0.5 = 0
        # Leaf 4 (100): (d1)(1-d2)(1-d3)   = 1 * 1 * 0.5 = 0.5
        # Leaf 5 (101): (d1)(1-d2)(d3)     = 1 * 1 * 0.5 = 0.5
        # Leaf 6 (110): (d1)(d2)(1-d3)     = 1 * 0 * 0.5 = 0
        # Leaf 7 (111): (d1)(d2)(d3)       = 1 * 0 * 0.5 = 0
        expected_padded = np.array([0, 0, 0, 0, 0.5, 0.5, 0, 0])

        # The layer slices this to output_dim=6
        unnormalized = expected_padded[:6]  # [0, 0, 0, 0, 0.5, 0.5]
        # The sum is 1.0, so renormalization doesn't change it.
        expected_final = unnormalized

        output = layer(inputs).numpy().flatten()
        assert np.allclose(output, expected_final, atol=1e-6)

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = HierarchicalRoutingLayer(
            output_dim=12,
            epsilon=1e-6,
            name="test_routing"
        )

        config = original_layer.get_config()
        recreated_layer = HierarchicalRoutingLayer.from_config(config)

        assert recreated_layer.name == original_layer.name
        assert recreated_layer.output_dim == original_layer.output_dim
        assert recreated_layer.epsilon == original_layer.epsilon
        assert recreated_layer.padded_output_dim == 2 ** 4

    def test_model_save_load(self, input_tensor, output_dim):
        """Test saving and loading a model with the HierarchicalRoutingLayer."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = keras.layers.Dense(32, activation='relu')(inputs)
        outputs = HierarchicalRoutingLayer(output_dim=output_dim, name="routing_output")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_prediction = model.predict(input_tensor)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"HierarchicalRoutingLayer": HierarchicalRoutingLayer}
            )

            loaded_prediction = loaded_model.predict(input_tensor)

            assert np.allclose(original_prediction, loaded_prediction, atol=1e-6)
            assert isinstance(loaded_model.get_layer("routing_output"), HierarchicalRoutingLayer)

    def test_gradient_flow(self, input_tensor):
        """Test gradient flow through the layer."""
        layer = HierarchicalRoutingLayer(output_dim=10)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            outputs = layer(input_tensor)
            # Use a simple loss that depends on all outputs
            loss = tf.reduce_mean(tf.square(outputs))

        grads = tape.gradient(loss, layer.trainable_weights)

        assert len(grads) > 0, "No gradients were computed."
        for grad in grads:
            assert grad is not None, "A gradient is None."
            assert np.any(grad.numpy() != 0), "Gradients are all zero."

    def test_training_loop(self):
        """Test that the layer can be used in a training loop."""
        batch_size, input_dim, output_dim = 16, 8, 5

        model = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(16, activation='relu'),
            HierarchicalRoutingLayer(output_dim=output_dim)
        ])

        # Since the layer outputs probabilities, CategoricalCrossentropy is a suitable loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Create mock data
        x_train = tf.random.normal([batch_size, input_dim])
        # Labels must be one-hot encoded for CategoricalCrossentropy
        y_train_indices = tf.random.uniform([batch_size], 0, output_dim, dtype=tf.int32)
        y_train = tf.one_hot(y_train_indices, depth=output_dim)

        # Get initial loss
        initial_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Train for a few steps
        model.fit(x_train, y_train, epochs=3, batch_size=4, verbose=0)

        # Get final loss
        final_loss = model.evaluate(x_train, y_train, verbose=0)[0]

        # Loss should decrease as the model learns
        assert final_loss < initial_loss