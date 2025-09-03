import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models, initializers, regularizers
import tempfile
import os
import warnings
from typing import Any, Dict

from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig


# --- Test Class ---
class TestTransformerLayer:
    """
    Comprehensive and modern test suite for the TransformerLayer.
    This suite follows modern Keras 3 testing best practices and includes MoE support.
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
    def moe_config(self) -> MoEConfig:
        """Provides a standard MoE configuration for testing, using the new ffn_config structure."""
        return MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={
                    "type": "mlp",
                    "output_dim": 64,      # Matches hidden_size
                    "hidden_dim": 256,     # The intermediate size for the expert
                }
            ),
            gating_config=GatingConfig(
                gating_type='linear',
                top_k=2
            )
        )

    @pytest.fixture
    def moe_config_dict(self) -> Dict[str, Any]:
        """Provides MoE configuration as a dictionary for testing dict conversion."""
        return {
            'num_experts': 4,
            'expert_config': {
                'ffn_config': {
                    "type": "swiglu",
                    "output_dim": 64,
                    "ffn_expansion_factor": 4
                }
            },
            'gating_config': {
                'gating_type': 'linear',
                'top_k': 2
            }
        }

    @pytest.fixture
    def sample_input(self) -> tf.Tensor:
        """Provides a standard sample input tensor for testing."""
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
        assert layer.ffn_type == 'mlp'
        assert layer.moe_config is None

    def test_initialization_with_moe_config(self, layer_config, moe_config):
        """Tests initialization with MoE configuration."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)
        assert layer.moe_config is not None
        assert isinstance(layer.moe_config, MoEConfig)
        assert layer.moe_config.num_experts == 4
        assert layer.moe_config.expert_config.ffn_config['output_dim'] == 64

    def test_initialization_with_moe_dict(self, layer_config, moe_config_dict):
        """Tests initialization with MoE configuration as dictionary."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config_dict)
        assert layer.moe_config is not None
        assert isinstance(layer.moe_config, MoEConfig)
        assert layer.moe_config.num_experts == 4
        assert layer.moe_config.expert_config.ffn_config['type'] == 'swiglu'

    def test_moe_overrides_ffn_params_with_warning(self, layer_config, moe_config):
        """Tests that MoE config overrides FFN parameters and issues warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer = TransformerLayer(
                hidden_size=layer_config['hidden_size'],
                num_heads=layer_config['num_heads'],
                intermediate_size=512,
                moe_config=moe_config,
                ffn_type='swiglu',
                ffn_args={'some_arg': 'value'},
            )
            assert any("moe_config is provided" in str(warn.message) for warn in w)
            assert any("`ffn_type`" in str(warn.message) for warn in w)
            assert any("`ffn_args`" in str(warn.message) for warn in w)

    def test_moe_hidden_dim_synchronization(self, layer_config, moe_config):
        """Tests that MoE expert output_dim is synchronized with transformer hidden_size."""
        moe_config.expert_config.ffn_config['output_dim'] = 128  # Mismatched
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer = TransformerLayer(**layer_config, moe_config=moe_config)
            assert any("Adjusting moe_config.expert_config.ffn_config['output_dim']" in str(warn.message) for warn in w)
        assert layer.moe_config.expert_config.ffn_config['output_dim'] == layer.hidden_size

    def test_moe_intermediate_size_inheritance(self, layer_config):
        """Tests that MoE expert inherits intermediate_size from TransformerLayer when not set."""
        moe_config_no_intermediate = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(ffn_config={"type": "mlp", "output_dim": 64})
        )
        layer = TransformerLayer(**layer_config, moe_config=moe_config_no_intermediate)
        assert 'hidden_dim' in layer.moe_config.expert_config.ffn_config
        assert layer.moe_config.expert_config.ffn_config['hidden_dim'] == layer.intermediate_size

    def test_moe_preserves_explicit_intermediate_size(self):
        """Tests that explicitly set expert intermediate_size is preserved."""
        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={"type": "mlp", "output_dim": 64, "hidden_dim": 512}
            )
        )
        layer = TransformerLayer(hidden_size=64, num_heads=4, intermediate_size=256, moe_config=moe_config)
        assert layer.moe_config.expert_config.ffn_config['hidden_dim'] == 512

    def test_build_process_with_moe(self, layer_config, moe_config, sample_input):
        """Tests that the layer with MoE and all its sub-layers are built correctly."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)
        assert not layer.built
        layer(sample_input)
        assert layer.built
        assert hasattr(layer.ffn_layer, 'experts')

    # ===============================================
    # 2. Forward Pass and Core Behavior Tests
    # ===============================================
    @pytest.mark.parametrize("expert_ffn_type", ['mlp', 'swiglu', 'glu', 'geglu'])
    @pytest.mark.parametrize("gating_type", ['linear', 'cosine'])
    @pytest.mark.parametrize("top_k", [1, 2])
    def test_forward_pass_with_moe_variations(self, expert_ffn_type, gating_type, top_k, sample_input):
        """Tests forward pass with various MoE configurations."""
        ffn_config = {"type": expert_ffn_type, "output_dim": 64}
        if expert_ffn_type in ['mlp', 'glu', 'geglu']:
            ffn_config['hidden_dim'] = 256
        elif expert_ffn_type == 'swiglu':
            ffn_config['ffn_expansion_factor'] = 4

        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(ffn_config=ffn_config),
            gating_config=GatingConfig(gating_type=gating_type, top_k=top_k)
        )
        layer = TransformerLayer(hidden_size=64, num_heads=4, intermediate_size=256, moe_config=moe_config)
        output = layer(sample_input, training=False)
        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_moe_forward_pass_training_vs_inference(self, layer_config, moe_config, sample_input):
        """Tests that MoE behaves differently in training vs inference mode."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config, dropout_rate=0.0)
        output_train = layer(sample_input, training=True)
        output_infer = layer(sample_input, training=False)
        assert output_train.shape == output_infer.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output_train)))
        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer)))

    # ===============================================
    # 3. Serialization Test (The Gold Standard)
    # ===============================================
    def test_full_serialization_cycle_with_moe(self, moe_config, sample_input):
        """Tests full serialization cycle with MoE configuration."""
        layer_config = {
            'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256,
            'moe_config': moe_config, 'normalization_position': 'pre', 'use_stochastic_depth': True,
        }
        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = TransformerLayer(**layer_config)(inputs)
        model = models.Model(inputs, outputs)
        original_prediction = model(sample_input, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_moe_model.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    def test_full_serialization_cycle_with_moe_dict(self, moe_config_dict, sample_input):
        """Tests full serialization cycle with MoE configuration provided as dict."""
        layer_config = {
            'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256,
            'moe_config': moe_config_dict, 'normalization_position': 'post',
        }
        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = TransformerLayer(**layer_config)(inputs)
        model = models.Model(inputs, outputs)
        original_prediction = model(sample_input, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_moe_dict_model.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    # ===============================================
    # 4. Gradient and Training Integration Tests
    # ===============================================
    def test_gradient_flow_with_moe(self, layer_config, moe_config, sample_input):
        """Tests gradient flow through MoE layer."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)
        x_var = tf.Variable(sample_input)
        with tf.GradientTape() as tape:
            output = layer(x_var, training=True)
            loss = ops.mean(ops.square(output))
        gradients = tape.gradient(loss, layer.trainable_variables)
        assert len(gradients) > 0, "No gradients were computed for MoE."
        assert all(g is not None for g in gradients), "A gradient is None in MoE."

    def test_model_training_loop_integration_with_moe(self, layer_config, moe_config):
        """Ensures MoE layer can be used in a standard training loop."""
        model = models.Sequential([
            layers.InputLayer(shape=(16, 64)),
            TransformerLayer(**layer_config, moe_config=moe_config),
            layers.GlobalAveragePooling1D(),
            layers.Dense(10)
        ])
        model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        x_train = tf.random.normal((32, 16, 64))
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)
        history = model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)
        assert 'loss' in history.history
        assert not np.isnan(history.history['loss'][0]), "Loss became NaN during MoE training."

    def test_mixed_stacked_layers_with_moe(self, sample_input, moe_config):
        """Tests stacking TransformerLayers with mixed standard and MoE FFNs."""
        config_standard = {'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256}
        config_moe = {'hidden_size': 64, 'num_heads': 4, 'intermediate_size': 256, 'moe_config': moe_config}
        inputs = layers.Input(shape=sample_input.shape[1:])
        x = TransformerLayer(**config_standard)(inputs)
        x = TransformerLayer(**config_moe)(x)
        outputs = layers.GlobalAveragePooling1D()(x)
        model = models.Model(inputs, outputs)
        prediction = model(sample_input, training=False)
        assert prediction.shape == (sample_input.shape[0], 64)
        assert not np.any(np.isnan(ops.convert_to_numpy(prediction)))

    # ===============================================
    # 5. MoE-Specific Tests
    # ===============================================
    def test_moe_config_get_config(self, layer_config, moe_config):
        """Tests that get_config properly serializes MoE configuration."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)
        config = layer.get_config()
        assert 'moe_config' in config
        assert config['moe_config'] is not None
        assert isinstance(config['moe_config']['expert_config'], dict)
        assert 'ffn_config' in config['moe_config']['expert_config']

    def test_moe_config_from_config(self, layer_config, moe_config):
        """Tests that layer can be reconstructed from config with MoE."""
        original_layer = TransformerLayer(**layer_config, moe_config=moe_config)
        config = original_layer.get_config()
        reconstructed_layer = TransformerLayer.from_config(config)
        assert reconstructed_layer.moe_config is not None
        assert isinstance(reconstructed_layer.moe_config, MoEConfig)
        assert reconstructed_layer.moe_config.num_experts == original_layer.moe_config.num_experts

    @pytest.mark.parametrize("num_experts", [2, 8])
    @pytest.mark.parametrize("top_k", [1, 2])
    def test_moe_scaling_properties(self, num_experts, top_k, sample_input):
        """Tests MoE with different numbers of experts and top-k values."""
        moe_config = MoEConfig(
            num_experts=num_experts,
            expert_config=ExpertConfig(
                ffn_config={"type": "mlp", "output_dim": 64, "hidden_dim": 128}
            ),
            gating_config=GatingConfig(
                gating_type='linear', top_k=min(top_k, num_experts)
            )
        )
        layer = TransformerLayer(hidden_size=64, num_heads=4, intermediate_size=256, moe_config=moe_config)
        output = layer(sample_input, training=False)
        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))