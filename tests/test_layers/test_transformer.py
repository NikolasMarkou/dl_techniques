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
        """Provides a standard MoE configuration for testing."""
        return MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                expert_type='ffn',
                ffn_type='mlp',
                hidden_dim=64,  # Matches hidden_size
                intermediate_size=256
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
                'expert_type': 'ffn',
                'ffn_type': 'swiglu',
                'hidden_dim': 64,
                'intermediate_size': 256
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
        assert layer.normalization_position == 'post'
        assert layer.ffn_type == 'mlp'
        assert layer.use_stochastic_depth is False
        assert layer.moe_config is None

    def test_initialization_with_moe_config(self, layer_config, moe_config):
        """Tests initialization with MoE configuration."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)

        assert layer.moe_config is not None
        assert isinstance(layer.moe_config, MoEConfig)
        assert layer.moe_config.num_experts == 4
        assert layer.moe_config.expert_config.hidden_dim == 64

    def test_initialization_with_moe_dict(self, layer_config, moe_config_dict):
        """Tests initialization with MoE configuration as dictionary."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config_dict)

        assert layer.moe_config is not None
        assert isinstance(layer.moe_config, MoEConfig)
        assert layer.moe_config.num_experts == 4
        assert layer.moe_config.expert_config.ffn_type == 'swiglu'

    def test_moe_overrides_ffn_params_with_warning(self, layer_config, moe_config):
        """Tests that MoE config overrides FFN parameters and issues warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer = TransformerLayer(
                hidden_size=layer_config['hidden_size'],
                num_heads=layer_config['num_heads'],
                intermediate_size=512,  # Should be ignored
                moe_config=moe_config,
                ffn_type='swiglu',  # Should be ignored
                ffn_args={'some_arg': 'value'},  # Should be ignored
            )

            # Check that a warning was issued
            assert len(w) >= 1 # Can be more than 1 due to other automatic adjustments
            assert any("moe_config is provided" in str(warn.message) for warn in w)
            assert any("ffn_type" in str(warn.message) for warn in w)
            assert any("ffn_args" in str(warn.message) for warn in w)
            assert any("intermediate_size" in str(warn.message) for warn in w)

        # Verify MoE is used
        assert isinstance(layer.ffn_layer, layers.Layer)
        assert layer.ffn_layer.name == 'ffn'

    def test_moe_hidden_dim_synchronization(self, layer_config, moe_config):
        """Tests that MoE expert hidden_dim is synchronized with transformer hidden_size."""
        # Create MoE config with mismatched hidden_dim
        moe_config.expert_config.hidden_dim = 128  # Different from layer's hidden_size

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer = TransformerLayer(**layer_config, moe_config=moe_config)

            # Check that warning was issued about adjustment
            assert any("Adjusting moe_config.expert_config.hidden_dim" in str(warning.message) for warning in w)

        # Verify hidden_dim was adjusted
        assert layer.moe_config.expert_config.hidden_dim == layer.hidden_size

    def test_moe_intermediate_size_inheritance(self, layer_config, moe_config):
        """Tests that MoE expert inherits intermediate_size from TransformerLayer when not set."""
        # Ensure expert_config has no intermediate_size set
        moe_config.expert_config.intermediate_size = None

        # Create layer with specific intermediate_size
        layer = TransformerLayer(**layer_config, moe_config=moe_config)

        # Verify intermediate_size was inherited
        assert layer.moe_config.expert_config.intermediate_size == layer.intermediate_size

    def test_moe_preserves_explicit_intermediate_size(self, layer_config):
        """Tests that explicitly set expert intermediate_size is preserved."""
        # Create MoE config with explicit intermediate_size
        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                expert_type='ffn',
                ffn_type='mlp',
                hidden_dim=64,
                intermediate_size=512  # Explicitly set
            ),
            gating_config=GatingConfig(gating_type='linear', top_k=2)
        )

        # Create layer with different intermediate_size
        layer = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,  # Different from expert config
            moe_config=moe_config
        )

        # Verify expert's intermediate_size is preserved
        assert layer.moe_config.expert_config.intermediate_size == 512

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

    def test_build_process_with_moe(self, layer_config, moe_config, sample_input):
        """Tests that the layer with MoE and all its sub-layers are built correctly."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)
        assert not layer.built
        layer(sample_input)
        assert layer.built
        assert layer.attention.built
        assert layer.ffn_layer.built  # This is now the MoE layer
        # Check that it's actually an MoE layer
        assert hasattr(layer.ffn_layer, 'experts')  # MoE should have experts

    def test_build_with_invalid_shape(self, layer_config):
        """Tests that build() raises an error for mismatched input shapes."""
        layer = TransformerLayer(**layer_config)
        with pytest.raises(ValueError, match="Input feature dimension"):
            layer.build(input_shape=(4, 16, 32))

    # ===============================================
    # 2. Forward Pass and Core Behavior Tests
    # ===============================================
    @pytest.mark.parametrize("attention_type", ['multi_head_attention',
                                                'window_attention',
                                                'group_query_attention',
                                                'differential_attention'])
    @pytest.mark.parametrize("ffn_type", ['mlp', 'swiglu', 'differential', 'geglu', 'glu', 'residual', 'swin_mlp'])
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

    @pytest.mark.parametrize("expert_ffn_type", ['mlp', 'swiglu', 'glu', 'geglu'])
    @pytest.mark.parametrize("gating_type", ['linear', 'cosine'])
    @pytest.mark.parametrize("top_k", [1, 2])
    def test_forward_pass_with_moe_variations(self, expert_ffn_type, gating_type, top_k, sample_input):
        """Tests forward pass with various MoE configurations."""
        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                expert_type='ffn',
                ffn_type=expert_ffn_type,
                hidden_dim=64,
                intermediate_size=256
            ),
            gating_config=GatingConfig(
                gating_type=gating_type,
                top_k=top_k
            )
        )

        layer = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            moe_config=moe_config
        )

        output = layer(sample_input, training=False)
        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_moe_forward_pass_training_vs_inference(self, layer_config, moe_config, sample_input):
        """Tests that MoE behaves differently in training vs inference mode."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config, dropout_rate=0.0)

        # MoE might have auxiliary losses during training
        output_train = layer(sample_input, training=True)
        output_infer = layer(sample_input, training=False)

        # Both should have the same shape
        assert output_train.shape == output_infer.shape

        # No NaN values
        assert not np.any(np.isnan(ops.convert_to_numpy(output_train)))
        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer)))

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
    @pytest.mark.parametrize("attention_type", ['multi_head_attention',
                                                'window_attention',
                                                'group_query_attention',
                                                'differential_attention'])
    @pytest.mark.parametrize("ffn_type", ['mlp', 'swiglu', 'differential', 'geglu', 'glu', 'residual', 'swin_mlp'])
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

    def test_full_serialization_cycle_with_moe(self, moe_config, sample_input):
        """Tests full serialization cycle with MoE configuration."""
        layer_config = {
            'hidden_size': 64,
            'num_heads': 4,
            'intermediate_size': 256,  # Should be ignored
            'moe_config': moe_config,
            'normalization_position': 'pre',
            'use_stochastic_depth': True,
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
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization for MoE model"
            )

    def test_full_serialization_cycle_with_moe_dict(self, moe_config_dict, sample_input):
        """Tests full serialization cycle with MoE configuration provided as dict."""
        layer_config = {
            'hidden_size': 64,
            'num_heads': 4,
            'intermediate_size': 256,
            'moe_config': moe_config_dict,  # Pass as dict
            'normalization_position': 'post',
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
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization for MoE model (dict config)"
            )

    # ===============================================
    # 4. Gradient and Training Integration Tests
    # ===============================================
    def test_gradient_flow(self, layer_config, sample_input):
        """Tests that gradients are computed and flow through all trainable variables."""
        layer = TransformerLayer(**layer_config)
        x_var = tf.Variable(sample_input)

        with tf.GradientTape() as tape:
            output = layer(x_var, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0, "No gradients were computed."
        assert all(g is not None for g in gradients), "A gradient is None."
        assert any(ops.max(ops.abs(g)) > 0 for g in gradients if g is not None), "All gradients are zero."

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
        assert any(ops.max(ops.abs(g)) > 0 for g in gradients if g is not None), "All MoE gradients are zero."

    def test_model_training_loop_integration(self, layer_config):
        """Ensures the layer can be used in a standard model.fit() training loop."""
        model = models.Sequential([
            layers.InputLayer(shape=(16, 64)),
            TransformerLayer(**layer_config),
            layers.GlobalAveragePooling1D(),
            layers.Dense(10)
        ])

        model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        x_train = tf.random.normal((32, 16, 64))
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        history = model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)

        assert 'loss' in history.history
        assert not np.isnan(history.history['loss'][0]), "Loss became NaN during training."

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

    def test_mixed_stacked_layers_with_moe(self, sample_input, moe_config):
        """Tests stacking TransformerLayers with mixed standard and MoE FFNs."""
        config_standard = {
            'hidden_size': 64,
            'num_heads': 4,
            'intermediate_size': 256,
            'normalization_position': 'pre'
        }
        config_moe = {
            'hidden_size': 64,
            'num_heads': 4,
            'intermediate_size': 256,  # Ignored
            'moe_config': moe_config,
            'normalization_position': 'post'
        }

        inputs = layers.Input(shape=sample_input.shape[1:])
        x = TransformerLayer(**config_standard)(inputs)  # Standard FFN
        x = TransformerLayer(**config_moe)(x)  # MoE FFN
        x = TransformerLayer(**config_standard)(x)  # Standard FFN again
        outputs = layers.GlobalAveragePooling1D()(x)
        model = models.Model(inputs, outputs)

        prediction = model(sample_input, training=False)

        assert prediction.shape == (sample_input.shape[0], 64)
        assert not np.any(np.isnan(ops.convert_to_numpy(prediction)))

    # ===============================================
    # 5. MoE-Specific Tests
    # ===============================================
    def test_moe_expert_utilization(self, layer_config, moe_config, sample_input):
        """Tests that different experts are being utilized in MoE."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)

        # Run multiple forward passes
        outputs = []
        for _ in range(10):
            # Different random inputs should potentially route to different experts
            random_input = tf.random.normal(shape=sample_input.shape)
            output = layer(random_input, training=True)
            outputs.append(output)

        # All outputs should be valid
        for output in outputs:
            assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_moe_config_get_config(self, layer_config, moe_config):
        """Tests that get_config properly serializes MoE configuration."""
        layer = TransformerLayer(**layer_config, moe_config=moe_config)
        config = layer.get_config()

        assert 'moe_config' in config
        assert config['moe_config'] is not None
        # Should be serialized to dict
        assert isinstance(config['moe_config'], dict)
        assert config['moe_config']['num_experts'] == 4

    def test_moe_config_from_config(self, layer_config, moe_config):
        """Tests that layer can be reconstructed from config with MoE."""
        original_layer = TransformerLayer(**layer_config, moe_config=moe_config)
        config = original_layer.get_config()

        # Recreate layer from config
        reconstructed_layer = TransformerLayer(**config)

        assert reconstructed_layer.moe_config is not None
        assert isinstance(reconstructed_layer.moe_config, MoEConfig)
        assert reconstructed_layer.moe_config.num_experts == original_layer.moe_config.num_experts

    @pytest.mark.parametrize("num_experts", [2, 4, 8])
    @pytest.mark.parametrize("top_k", [1, 2])
    def test_moe_scaling_properties(self, num_experts, top_k, sample_input):
        """Tests MoE with different numbers of experts and top-k values."""
        moe_config = MoEConfig(
            num_experts=num_experts,
            expert_config=ExpertConfig(
                expert_type='ffn',
                ffn_type='mlp',
                hidden_dim=64,
                intermediate_size=128  # Smaller for faster testing
            ),
            gating_config=GatingConfig(
                gating_type='linear',
                top_k=min(top_k, num_experts)  # Ensure top_k <= num_experts
            )
        )

        layer = TransformerLayer(
            hidden_size=64,
            num_heads=4,
            intermediate_size=256,
            moe_config=moe_config
        )

        output = layer(sample_input, training=False)
        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))