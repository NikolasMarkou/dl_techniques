import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models, initializers, regularizers
import tempfile
import os
from typing import Any, Dict

# --- Import Layer to be Tested ---
from dl_techniques.layers.transformer import TransformerLayer


# This test file assumes all custom sub-layers are installed and available,
# as requested. This simplifies the testing logic by removing conditional skips.

# --- Test Class ---
class TestTransformerLayer:
    """
    Comprehensive and modern test suite for the TransformerLayer.
    This suite follows modern Keras 3 testing best practices.
    """

    # --- Fixtures for Reusability ---
    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for a small, testable layer."""
        return {
            'hidden_size': 64,
            'num_heads': 4,
            'intermediate_size': 256,
        }

    @pytest.fixture
    def sample_input(self) -> tf.Tensor:
        """Provides a standard sample input tensor for testing."""
        # FIX: Use tf.random for tensor generation.
        # seq_len=16 is a perfect square (4*4), making it compatible
        # with WindowAttention(window_size=4).
        return tf.random.normal(shape=(4, 16, 64))

    # ===============================================
    # 1. Initialization and Build Tests
    # ===============================================
    def test_initialization_defaults(self, layer_config):
        """Tests layer initialization with default parameters."""
        layer = TransformerLayer(**layer_config)

        assert not layer.built
        assert layer.attention_type == 'multi_head_attention'
        assert layer.normalization_type == 'layer_norm'
        assert layer.normalization_position == 'post'
        assert layer.ffn_type == 'mlp'
        assert layer.use_stochastic_depth is False

    def test_initialization_custom_parameters(self):
        """Tests initialization with a wide range of custom parameters."""
        custom_regularizer = regularizers.L2(1e-4)
        layer = TransformerLayer(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
            normalization_position='pre',
            dropout_rate=0.2,
            use_stochastic_depth=True,
            stochastic_depth_rate=0.05,
            activation='relu',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=custom_regularizer
        )
        assert layer.normalization_position == 'pre'
        assert layer.use_bias is False
        assert layer.use_stochastic_depth is True
        assert isinstance(layer.kernel_initializer, initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer

    def test_initialization_error_handling(self):
        """Tests that invalid initialization parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            TransformerLayer(hidden_size=0, num_heads=4, intermediate_size=256)

        with pytest.raises(ValueError, match="must be divisible by"):
            TransformerLayer(hidden_size=64, num_heads=7, intermediate_size=256)

        with pytest.raises(ValueError, match="Unknown attention type"):
            TransformerLayer(hidden_size=64, num_heads=4, intermediate_size=256, attention_type="invalid_type")

    def test_build_process(self, layer_config, sample_input):
        """Tests that the layer and all its sub-layers are built after the first forward pass."""
        layer = TransformerLayer(**layer_config)
        assert not layer.built
        layer(sample_input)
        assert layer.built
        assert layer.attention.built and layer.ffn_layer.built

    def test_build_with_invalid_shape(self, layer_config):
        """Tests that build() raises an error for mismatched input shapes."""
        layer = TransformerLayer(**layer_config)
        with pytest.raises(ValueError, match="Input feature dimension"):
            layer.build(input_shape=(4, 16, 32))

    # ===============================================
    # 2. Forward Pass and Core Behavior Tests
    # ===============================================
    @pytest.mark.parametrize("attention_type", ['multi_head_attention', 'window_attention', 'group_query_attention',
                                                'differential_attention'])
    @pytest.mark.parametrize("ffn_type", ['mlp', 'swiglu', 'differential', 'glu', 'residual', 'swin_mlp'])
    @pytest.mark.parametrize("normalization_position", ['pre', 'post'])
    def test_forward_pass_combinations(self, attention_type, ffn_type, normalization_position, sample_input):
        """Tests the forward pass across a comprehensive matrix of configurations."""
        config = {
            'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256,
            'attention_type': attention_type,
            'ffn_type': ffn_type,
            'normalization_position': normalization_position
        }
        if attention_type == 'window_attention':
            config['window_size'] = 4
        if attention_type == 'group_query_attention':
            config['n_kv_head'] = 2

        layer = TransformerLayer(**config)
        output = layer(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_dropout_and_stochastic_depth_behavior(self, sample_input):
        """Verifies that dropout/stochastic depth are active only during training."""
        layer = TransformerLayer(
            hidden_size=64, num_heads=4, intermediate_size=256,
            dropout_rate=0.5, use_stochastic_depth=True, stochastic_depth_rate=0.5
        )

        output_train1 = layer(sample_input, training=True)
        output_train2 = layer(sample_input, training=True)
        assert not np.allclose(ops.convert_to_numpy(output_train1), ops.convert_to_numpy(output_train2))

        output_infer1 = layer(sample_input, training=False)
        output_infer2 = layer(sample_input, training=False)
        np.testing.assert_allclose(ops.convert_to_numpy(output_infer1), ops.convert_to_numpy(output_infer2))

    def test_attention_masking(self, layer_config, sample_input):
        """Tests that an attention mask influences the output."""
        layer = TransformerLayer(**layer_config, dropout_rate=0.0)

        seq_len = sample_input.shape[1]
        mask = np.ones((seq_len, seq_len))
        mask[:, seq_len // 2:] = 0
        attention_mask = tf.convert_to_tensor(mask, dtype='float32')

        output_unmasked = layer(sample_input, attention_mask=None, training=False)
        output_masked = layer(sample_input, attention_mask=attention_mask, training=False)

        assert not np.allclose(ops.convert_to_numpy(output_unmasked), ops.convert_to_numpy(output_masked))

    def test_differential_attention_layer_idx(self, sample_input):
        """Tests that the `layer_idx` argument affects differential attention."""
        config = {
            'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256,
            'attention_type': 'differential_attention', 'dropout_rate': 0.0
        }
        layer = TransformerLayer(**config)

        output_idx0 = layer(sample_input, layer_idx=0, training=False)
        output_idx5 = layer(sample_input, layer_idx=5, training=False)

        assert not np.allclose(ops.convert_to_numpy(output_idx0), ops.convert_to_numpy(output_idx5))

    # ===============================================
    # 3. Serialization Test (The Gold Standard)
    # ===============================================
    @pytest.mark.parametrize("attention_type", ['multi_head_attention', 'window_attention', 'group_query_attention'])
    @pytest.mark.parametrize("ffn_type", ['mlp', 'swiglu'])
    @pytest.mark.parametrize("normalization_position", ['pre', 'post'])
    def test_full_serialization_cycle(self, attention_type, ffn_type, normalization_position, sample_input):
        """Performs a full model save/load cycle, the most reliable test for serialization."""
        layer_config = {
            'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256,
            'attention_type': attention_type, 'ffn_type': ffn_type,
            'normalization_position': normalization_position,
            'use_stochastic_depth': True,
        }
        if attention_type == 'window_attention':
            layer_config['window_size'] = 4

        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = TransformerLayer(**layer_config)(inputs)
        model = models.Model(inputs, outputs)
        original_prediction = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Predictions differ after serialization for config: {layer_config}"
            )

    # ===============================================
    # 4. Gradient and Training Integration Tests
    # ===============================================
    def test_gradient_flow(self, layer_config, sample_input):
        """Tests that gradients are computed and flow through all trainable variables."""
        layer = TransformerLayer(**layer_config)
        x_var = tf.Variable(sample_input)

        # FIX: Use tf.GradientTape
        with tf.GradientTape() as tape:
            output = layer(x_var, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0, "No gradients were computed."
        assert all(g is not None for g in gradients), "A gradient is None."
        assert any(ops.max(ops.abs(g)) > 0 for g in gradients if g is not None), "All gradients are zero."

    def test_model_training_loop_integration(self, layer_config):
        """Ensures the layer can be used in a standard model.fit() training loop."""
        model = models.Sequential([
            layers.InputLayer(shape=(16, 64)),
            TransformerLayer(**layer_config),
            layers.GlobalAveragePooling1D(),
            layers.Dense(10)
        ])

        model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        # FIX: Use tf.random
        x_train = tf.random.normal((32, 16, 64))
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        history = model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)

        assert 'loss' in history.history
        assert not np.isnan(history.history['loss'][0]), "Loss became NaN during training."

    def test_stacked_layers_in_model(self, sample_input):
        """Tests stacking multiple TransformerLayers in a model."""
        config1 = {'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256, 'normalization_position': 'pre'}
        config2 = {'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256, 'normalization_position': 'post'}

        inputs = layers.Input(shape=sample_input.shape[1:])
        x = TransformerLayer(**config1)(inputs)
        x = TransformerLayer(**config2)(x)
        outputs = layers.GlobalAveragePooling1D()(x)
        model = models.Model(inputs, outputs)

        prediction = model(sample_input, training=False)

        assert prediction.shape == (sample_input.shape[0], 64)
        assert not np.any(np.isnan(ops.convert_to_numpy(prediction)))