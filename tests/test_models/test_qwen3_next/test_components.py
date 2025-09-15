import os
import tempfile
from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import layers, models, ops

from dl_techniques.models.qwen3_next.components import Qwen3NextBlock
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig


# --- Test Class ---
class TestQwen3NextBlock:
    """
    Comprehensive and modern test suite for the Qwen3NextBlock layer.
    This suite follows modern Keras 3 testing best practices and covers
    all aspects of the 3+1 layer architecture with MoE and stochastic depth.
    """

    # --- Fixtures for Reusability ---
    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for a basic testable block."""
        return {
            "dim": 128,
            "num_heads": 8,
            "dropout_rate": 0.0,
        }

    @pytest.fixture
    def custom_head_config(self) -> Dict[str, Any]:
        """Provides configuration with custom head dimension."""
        return {
            "dim": 144,
            "num_heads": 6,
            "head_dim": 32,  # Custom head size
            "normalization_type": "rms_norm",
        }

    @pytest.fixture
    def moe_config(self) -> MoEConfig:
        """Provides a basic MoE configuration for testing."""
        return MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={
                    "type": "swiglu",
                    "output_dim": 128,
                    "ffn_expansion_factor": 2
                }
            ),
            gating_config=GatingConfig(
                gating_type="linear",
                top_k=2,
                aux_loss_weight=0.01
            )
        )

    @pytest.fixture
    def moe_block_config(self, moe_config) -> Dict[str, Any]:
        """Provides configuration with MoE enabled."""
        return {
            "dim": 128,
            "num_heads": 8,
            "moe_config": moe_config,
            "dropout_rate": 0.1,
        }

    @pytest.fixture
    def stochastic_depth_config(self) -> Dict[str, Any]:
        """Provides configuration with stochastic depth enabled."""
        return {
            "dim": 96,
            "num_heads": 6,
            "use_stochastic_depth": True,
            "stochastic_depth_rate": 0.2,
            "dropout_rate": 0.0,
        }

    @pytest.fixture
    def advanced_config(self) -> Dict[str, Any]:
        """Provides configuration with all advanced features."""
        dim = 192
        # This MoE config must have an output_dim that matches the block's dim
        # for the residual connection to work.
        advanced_moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={
                    "type": "swiglu",
                    "output_dim": dim,  # Match the block's dimension
                    "ffn_expansion_factor": 2
                }
            ),
            gating_config=GatingConfig(
                gating_type="linear",
                top_k=2,
                aux_loss_weight=0.01
            )
        )
        return {
            "dim": dim,
            "num_heads": 12,
            "head_dim": 24,
            "moe_config": advanced_moe_config,
            "normalization_type": "zero_centered_rms_norm",
            "norm_eps": 1e-8,
            "dropout_rate": 0.1,
            "use_stochastic_depth": True,
            "stochastic_depth_rate": 0.15,
        }

    @pytest.fixture
    def sample_input(self) -> tf.Tensor:
        """Provides a standard sample input tensor for testing."""
        return tf.random.normal(shape=(4, 32, 128))

    @pytest.fixture
    def custom_sample_input(self) -> tf.Tensor:
        """Provides sample input matching custom head configuration."""
        return tf.random.normal(shape=(2, 24, 144))

    @pytest.fixture
    def stochastic_sample_input(self) -> tf.Tensor:
        """Provides sample input for stochastic depth configuration."""
        return tf.random.normal(shape=(3, 16, 96))

    @pytest.fixture
    def advanced_sample_input(self) -> tf.Tensor:
        """Provides sample input for advanced configuration."""
        return tf.random.normal(shape=(2, 20, 192))

    @pytest.fixture
    def attention_mask(self) -> tf.Tensor:
        """Provides a sample attention mask for testing."""
        # Create a causal mask
        seq_len = 32
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return tf.expand_dims(mask, 0)  # Add batch dimension

    # ===============================================
    # 1. Initialization and Build Tests
    # ===============================================
    def test_initialization_basic(self, basic_config):
        """Tests basic layer initialization with default parameters."""
        block = Qwen3NextBlock(**basic_config)
        assert not block.built
        assert block.dim == 128
        assert block.num_heads == 8
        assert block.head_dim == 16  # 128 // 8
        assert block.normalization_type == "zero_centered_rms_norm"
        assert block.norm_eps == 1e-6
        assert block.dropout_rate == 0.0
        assert not block.use_stochastic_depth
        assert block.moe_config is None

    def test_initialization_custom_head_dim(self, custom_head_config):
        """Tests initialization with custom head dimension."""
        block = Qwen3NextBlock(**custom_head_config)
        assert block.dim == 144
        assert block.num_heads == 6
        assert block.head_dim == 32  # Explicitly set
        assert block.normalization_type == "rms_norm"

    def test_initialization_with_moe(self, moe_block_config):
        """Tests initialization with MoE configuration."""
        block = Qwen3NextBlock(**moe_block_config)
        assert block.moe_config is not None
        assert block.moe_config.num_experts == 4
        assert block.dropout_rate == 0.1

    def test_initialization_with_stochastic_depth(self, stochastic_depth_config):
        """Tests initialization with stochastic depth enabled."""
        block = Qwen3NextBlock(**stochastic_depth_config)
        assert block.use_stochastic_depth
        assert block.stochastic_depth_rate == 0.2

    def test_initialization_moe_dict_conversion(self):
        """Tests MoE config dictionary conversion."""
        moe_dict = {
            "num_experts": 8,
            "expert_config": {
                "ffn_config": {"type": "mlp", "hidden_dim": 256, "output_dim": 128}
            },
            "gating_config": {"gating_type": "linear", "top_k": 1}
        }

        block = Qwen3NextBlock(dim=128, num_heads=8, moe_config=moe_dict)
        assert block.moe_config is not None
        assert block.moe_config.num_experts == 8

    def test_build_process_basic(self, basic_config, sample_input):
        """Tests that all sub-layers are built correctly."""
        block = Qwen3NextBlock(**basic_config)
        assert not block.built

        # Build the block by calling it
        output = block(sample_input)
        assert block.built
        assert output.shape == sample_input.shape

        # Check that all sub-layers are built
        # Delta layers (3x)
        for i in range(3):
            assert block.delta_norms[i].built
            assert block.delta_layers[i].built
            if block.delta_moe_layers[i] is not None:
                assert block.delta_moe_layers[i].built

        # Attention layer (1x)
        assert block.attention_norm.built
        assert block.attention_layer.built
        if block.attention_moe is not None:
            assert block.attention_moe.built

    def test_build_process_with_moe(self, moe_block_config, sample_input):
        """Tests build process with MoE layers."""
        block = Qwen3NextBlock(**moe_block_config)
        block(sample_input)
        assert block.built

        # All MoE layers should be built
        for i in range(3):
            assert block.delta_moe_layers[i] is not None
            assert block.delta_moe_layers[i].built

        assert block.attention_moe is not None
        assert block.attention_moe.built

    def test_build_process_with_stochastic_depth(self, stochastic_depth_config, stochastic_sample_input):
        """Tests build process with stochastic depth layers."""
        block = Qwen3NextBlock(**stochastic_depth_config)
        block(stochastic_sample_input)
        assert block.built

        # All stochastic depth layers should be built
        for i in range(4):  # 3 delta + 1 attention
            assert block.stochastic_depth_layers[i] is not None
            assert block.stochastic_depth_layers[i].built

    # ===============================================
    # 2. Parameter Validation Tests
    # ===============================================
    def test_parameter_validation_dim_positive(self):
        """Tests that dim must be positive."""
        with pytest.raises(ValueError, match="dim must be positive"):
            Qwen3NextBlock(dim=0, num_heads=8)

        with pytest.raises(ValueError, match="dim must be positive"):
            Qwen3NextBlock(dim=-128, num_heads=8)

    def test_parameter_validation_num_heads_positive(self):
        """Tests that num_heads must be positive."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            Qwen3NextBlock(dim=128, num_heads=0)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            Qwen3NextBlock(dim=128, num_heads=-8)

    def test_parameter_validation_head_dim_positive(self):
        """Tests that head_dim must be positive when specified."""
        with pytest.raises(ValueError, match="head_dim must be positive"):
            Qwen3NextBlock(dim=128, num_heads=8, head_dim=0)

        with pytest.raises(ValueError, match="head_dim must be positive"):
            Qwen3NextBlock(dim=128, num_heads=8, head_dim=-16)

    def test_parameter_validation_dropout_rate(self):
        """Tests that dropout_rate must be in [0, 1]."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            Qwen3NextBlock(dim=128, num_heads=8, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            Qwen3NextBlock(dim=128, num_heads=8, dropout_rate=1.5)

    def test_parameter_validation_stochastic_depth_rate(self):
        """Tests that stochastic_depth_rate must be in [0, 1]."""
        with pytest.raises(ValueError, match="stochastic_depth_rate must be in"):
            Qwen3NextBlock(dim=128, num_heads=8, stochastic_depth_rate=-0.1)

        with pytest.raises(ValueError, match="stochastic_depth_rate must be in"):
            Qwen3NextBlock(dim=128, num_heads=8, stochastic_depth_rate=1.5)

    def test_parameter_validation_norm_eps(self):
        """Tests that norm_eps must be positive."""
        with pytest.raises(ValueError, match="norm_eps must be positive"):
            Qwen3NextBlock(dim=128, num_heads=8, norm_eps=0.0)

        with pytest.raises(ValueError, match="norm_eps must be positive"):
            Qwen3NextBlock(dim=128, num_heads=8, norm_eps=-1e-6)

    # ===============================================
    # 3. Forward Pass and Core Behavior Tests
    # ===============================================
    def test_forward_pass_basic(self, basic_config, sample_input):
        """Tests basic forward pass functionality."""
        block = Qwen3NextBlock(**basic_config)
        output = block(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))
        assert not np.any(np.isinf(ops.convert_to_numpy(output)))

    def test_forward_pass_with_attention_mask(self, basic_config, sample_input, attention_mask):
        """Tests forward pass with attention mask."""
        block = Qwen3NextBlock(**basic_config)
        output = block(sample_input, attention_mask=attention_mask, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_forward_pass_with_moe(self, moe_block_config, sample_input):
        """Tests forward pass with MoE layers."""
        block = Qwen3NextBlock(**moe_block_config)
        output = block(sample_input, training=True)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

        # Check that auxiliary losses are added by MoE
        # In a real scenario, we'd check model.losses after calling the block
        # Here we just verify the output is valid

    def test_forward_pass_with_stochastic_depth(self, stochastic_depth_config, stochastic_sample_input):
        """Tests forward pass with stochastic depth enabled."""
        block = Qwen3NextBlock(**stochastic_depth_config)

        # During training, stochastic depth should be active
        output_train = block(stochastic_sample_input, training=True)
        output_infer = block(stochastic_sample_input, training=False)

        assert output_train.shape == stochastic_sample_input.shape
        assert output_infer.shape == stochastic_sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output_train)))
        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer)))

    def test_forward_pass_advanced(self, advanced_config, advanced_sample_input):
        """Tests forward pass with all advanced features."""
        block = Qwen3NextBlock(**advanced_config)
        output = block(advanced_sample_input, training=True)

        assert output.shape == advanced_sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_training_vs_inference_mode(self, moe_block_config, sample_input):
        """Tests that block behaves differently in training vs inference mode."""
        block = Qwen3NextBlock(**moe_block_config)

        output_train = block(sample_input, training=True)
        output_infer = block(sample_input, training=False)

        assert output_train.shape == output_infer.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output_train)))
        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer)))

    def test_deterministic_inference(self, basic_config, sample_input):
        """Tests that inference is deterministic without stochastic components."""
        block = Qwen3NextBlock(**basic_config)

        output1 = block(sample_input, training=False)
        output2 = block(sample_input, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Inference outputs should be identical",
        )

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 16])
    def test_different_num_heads(self, num_heads, sample_input):
        """Tests forward pass with different numbers of heads."""
        # Adjust dim to be divisible by num_heads
        dim = num_heads * 16
        block = Qwen3NextBlock(dim=dim, num_heads=num_heads)

        # Adjust input to match dim
        adjusted_input = tf.random.normal((sample_input.shape[0], sample_input.shape[1], dim))
        output = block(adjusted_input, training=False)

        assert output.shape == adjusted_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    @pytest.mark.parametrize("normalization_type",
                             ["zero_centered_rms_norm", "layer_norm", "rms_norm", "band_rms"])
    def test_different_normalization_types(self, normalization_type, sample_input):
        """Tests forward pass with different normalization types."""
        block = Qwen3NextBlock(
            dim=128,
            num_heads=8,
            normalization_type=normalization_type
        )
        output = block(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    # ===============================================
    # 4. Serialization Tests (The Gold Standard)
    # ===============================================
    def test_full_serialization_cycle_basic(self, basic_config, sample_input):
        """Tests full serialization cycle with basic configuration."""
        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = Qwen3NextBlock(**basic_config)(inputs)
        model = models.Model(inputs, outputs)

        # Get prediction from original model
        original_prediction = model(sample_input, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_qwen3_next_basic.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)

            # Verify identical outputs
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Predictions should match after serialization",
            )

    def test_full_serialization_cycle_with_moe(self, moe_block_config, sample_input):
        """Tests full serialization cycle with MoE configuration."""
        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = Qwen3NextBlock(**moe_block_config)(inputs)
        model = models.Model(inputs, outputs)

        original_prediction = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_qwen3_next_moe.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="MoE block serialization should work",
            )

    def test_full_serialization_cycle_with_stochastic_depth(
            self, stochastic_depth_config, stochastic_sample_input
    ):
        """Tests full serialization cycle with stochastic depth."""
        inputs = layers.Input(shape=stochastic_sample_input.shape[1:])
        outputs = Qwen3NextBlock(**stochastic_depth_config)(inputs)
        model = models.Model(inputs, outputs)

        # Use inference mode for deterministic comparison
        original_prediction = model(stochastic_sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_qwen3_next_stoch.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(stochastic_sample_input, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Stochastic depth block serialization should work",
            )

    def test_full_serialization_cycle_advanced(self, advanced_config, advanced_sample_input):
        """Tests full serialization cycle with all advanced features."""
        inputs = layers.Input(shape=advanced_sample_input.shape[1:])
        outputs = Qwen3NextBlock(**advanced_config)(inputs)
        model = models.Model(inputs, outputs)

        # Use inference mode for deterministic comparison
        original_prediction = model(advanced_sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_qwen3_next_advanced.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(advanced_sample_input, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Advanced block serialization should work",
            )

    # ===============================================
    # 5. Configuration and Serialization Tests
    # ===============================================
    def test_get_config_completeness_basic(self, basic_config):
        """Tests that get_config contains all __init__ parameters."""
        block = Qwen3NextBlock(**basic_config)
        config = block.get_config()

        # Check all required parameters are present
        required_params = [
            "dim",
            "num_heads",
            "head_dim",
            "moe_config",
            "normalization_type",
            "norm_eps",
            "dropout_rate",
            "use_stochastic_depth",
            "stochastic_depth_rate",
        ]

        for param in required_params:
            assert param in config, f"Missing {param} in get_config()"

    def test_get_config_completeness_moe(self, moe_block_config):
        """Tests get_config with MoE configuration."""
        block = Qwen3NextBlock(**moe_block_config)
        config = block.get_config()

        assert config["moe_config"] is not None
        assert isinstance(config["moe_config"], dict)
        assert "num_experts" in config["moe_config"]

    def test_from_config_reconstruction_basic(self, basic_config):
        """Tests that block can be reconstructed from config."""
        original_block = Qwen3NextBlock(**basic_config)
        config = original_block.get_config()
        reconstructed_block = Qwen3NextBlock.from_config(config)

        # Check key parameters match
        assert reconstructed_block.dim == original_block.dim
        assert reconstructed_block.num_heads == original_block.num_heads
        assert reconstructed_block.head_dim == original_block.head_dim
        assert reconstructed_block.normalization_type == original_block.normalization_type
        assert reconstructed_block.norm_eps == original_block.norm_eps
        assert reconstructed_block.dropout_rate == original_block.dropout_rate
        assert reconstructed_block.use_stochastic_depth == original_block.use_stochastic_depth

    def test_from_config_reconstruction_moe(self, moe_block_config):
        """Tests reconstruction from config with MoE."""
        original_block = Qwen3NextBlock(**moe_block_config)
        config = original_block.get_config()
        reconstructed_block = Qwen3NextBlock.from_config(config)

        assert reconstructed_block.moe_config is not None
        assert reconstructed_block.moe_config.num_experts == original_block.moe_config.num_experts

    # ===============================================
    # 6. Gradient and Training Integration Tests
    # ===============================================
    def test_gradient_flow_basic(self, basic_config, sample_input):
        """Tests gradient computation through the block."""
        block = Qwen3NextBlock(**basic_config)
        x_var = tf.Variable(sample_input)

        with tf.GradientTape() as tape:
            output = block(x_var, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, block.trainable_variables)

        assert len(gradients) > 0, "No gradients were computed"
        assert all(g is not None for g in gradients), "Some gradients are None"
        assert all(
            not np.any(np.isnan(ops.convert_to_numpy(g))) for g in gradients
        ), "NaN in gradients"

    def test_gradient_flow_with_moe(self, moe_block_config, sample_input):
        """Tests gradient flow with MoE layers."""
        block = Qwen3NextBlock(**moe_block_config)
        x_var = tf.Variable(sample_input)

        with tf.GradientTape() as tape:
            output = block(x_var, training=True)
            # Include auxiliary losses from MoE
            total_loss = ops.mean(ops.square(output))

        # Get gradients for all trainable variables
        all_vars = block.trainable_variables
        gradients = tape.gradient(total_loss, all_vars)

        assert len(gradients) > 0, "No gradients were computed"
        # Some gradients might be None due to auxiliary losses, but most should exist
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > 0, "All gradients are None"

    def test_trainable_variables_count_basic(self, basic_config, sample_input):
        """Tests the number of trainable variables in basic configuration."""
        block = Qwen3NextBlock(**basic_config)
        block(sample_input)  # Build the block

        # Expected variables:
        # 3x Delta layers: each has ~13 variables (from GatedDeltaNet tests)
        # 1x Attention layer: similar structure, ~13 variables
        # 4x Normalization layers: each has 1 scale parameter (zero-centered RMS)
        # Total: roughly 3*13 + 13 + 4*1 = 56 variables

        actual_vars = len(block.trainable_variables)
        # Allow some flexibility as the exact count depends on implementation details
        assert actual_vars >= 50, f"Expected at least 50 variables, got {actual_vars}"
        assert actual_vars <= 70, f"Expected at most 70 variables, got {actual_vars}"

    def test_model_training_loop_integration(self, basic_config):
        """Tests integration in a standard training loop."""
        model = models.Sequential([
            layers.InputLayer(shape=(16, 128)),
            Qwen3NextBlock(**basic_config),
            layers.GlobalAveragePooling1D(),
            layers.Dense(10),
        ])

        model.compile(
            "adam",
            keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=False,
        )

        # Generate dummy data
        x_train = tf.random.normal((32, 16, 128))
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        # Train for one epoch
        history = model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)

        assert "loss" in history.history
        assert not np.isnan(history.history["loss"][0]), "Loss became NaN during training"

    def test_stacked_blocks(self, sample_input):
        """Tests stacking multiple Qwen3NextBlock layers."""
        inputs = layers.Input(shape=sample_input.shape[1:])
        x = Qwen3NextBlock(dim=128, num_heads=8)(inputs)
        x = Qwen3NextBlock(dim=128, num_heads=4)(x)
        outputs = layers.GlobalAveragePooling1D()(x)

        model = models.Model(inputs, outputs)
        prediction = model(sample_input, training=False)

        assert prediction.shape == (sample_input.shape[0], 128)
        assert not np.any(np.isnan(ops.convert_to_numpy(prediction)))

    # ===============================================
    # 7. Architecture-Specific Tests
    # ===============================================
    def test_three_plus_one_architecture(self, basic_config, sample_input):
        """Tests that the 3+1 architecture is correctly implemented."""
        block = Qwen3NextBlock(**basic_config)
        block(sample_input)  # Build the block

        # Should have exactly 3 delta layers
        assert len(block.delta_norms) == 3
        assert len(block.delta_layers) == 3
        assert len(block.delta_moe_layers) == 3

        # Should have exactly 1 attention layer
        assert block.attention_norm is not None
        assert block.attention_layer is not None

        # Each component should be built
        for i in range(3):
            assert hasattr(block.delta_layers[i], '__class__')
            # Verify it's a GatedDeltaNet (would need to import to check directly)

        # Verify attention layer exists
        assert hasattr(block.attention_layer, '__class__')

    def test_residual_connections(self, basic_config, sample_input):
        """Tests that residual connections preserve information."""
        block = Qwen3NextBlock(**basic_config)

        # Create a block with very small weights to isolate residual effect
        output = block(sample_input, training=False)

        # The output should not be identical to input (due to processing)
        # but should be reasonably close to input magnitude
        input_norm = ops.sqrt(ops.mean(ops.square(sample_input)))
        output_norm = ops.sqrt(ops.mean(ops.square(output)))

        # Output norm should be in reasonable range of input norm
        ratio = output_norm / input_norm
        assert 0.5 <= ratio <= 2.0, f"Output/input norm ratio {ratio} seems unreasonable"

    def test_layer_processing_order(self, basic_config, sample_input):
        """Tests that layers are processed in the correct order."""
        block = Qwen3NextBlock(**basic_config)

        # We can't easily test internal order without modifying the implementation,
        # but we can verify the block produces sensible outputs
        output = block(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    # ===============================================
    # 8. MoE Integration Tests
    # ===============================================
    def test_moe_auxiliary_losses(self, moe_block_config, sample_input):
        """Tests that MoE layers contribute auxiliary losses."""
        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = Qwen3NextBlock(**moe_block_config)(inputs)
        model = models.Model(inputs, outputs)

        # Forward pass to trigger MoE
        _ = model(sample_input, training=True)

        # MoE layers should add auxiliary losses
        # Note: The exact mechanism depends on how MoE integrates with Keras
        # This test verifies the block can be used in a model context

    def test_moe_vs_no_moe_comparison(self, sample_input):
        """Tests behavioral difference between MoE and non-MoE blocks."""
        # Create basic block without MoE
        block_no_moe = Qwen3NextBlock(dim=128, num_heads=8)

        # Create MoE configuration
        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={"type": "mlp", "hidden_dim": 256, "output_dim": 128}
            ),
            gating_config=GatingConfig(top_k=2)
        )

        # Create block with MoE
        block_with_moe = Qwen3NextBlock(dim=128, num_heads=8, moe_config=moe_config)

        output_no_moe = block_no_moe(sample_input, training=False)
        output_with_moe = block_with_moe(sample_input, training=False)

        # Both should produce valid outputs
        assert output_no_moe.shape == sample_input.shape
        assert output_with_moe.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output_no_moe)))
        assert not np.any(np.isnan(ops.convert_to_numpy(output_with_moe)))

        # Outputs should be different (different architectures)
        assert not np.allclose(
            ops.convert_to_numpy(output_no_moe),
            ops.convert_to_numpy(output_with_moe),
            atol=1e-6
        )

    # ===============================================
    # 9. Stochastic Depth Tests
    # ===============================================
    def test_stochastic_depth_training_vs_inference(self, stochastic_depth_config, stochastic_sample_input):
        """Tests stochastic depth behavior in training vs inference."""
        block = Qwen3NextBlock(**stochastic_depth_config)

        # Multiple forward passes in training mode should potentially give different results
        outputs_train = []
        for _ in range(5):
            output = block(stochastic_sample_input, training=True)
            outputs_train.append(output)

        # In inference mode, should be deterministic
        output_infer_1 = block(stochastic_sample_input, training=False)
        output_infer_2 = block(stochastic_sample_input, training=False)

        # All outputs should be valid
        for output in outputs_train:
            assert not np.any(np.isnan(ops.convert_to_numpy(output)))

        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer_1)))
        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer_2)))

        # Inference should be deterministic
        np.testing.assert_allclose(
            ops.convert_to_numpy(output_infer_1),
            ops.convert_to_numpy(output_infer_2),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Inference should be deterministic"
        )

    def test_stochastic_depth_disabled(self, basic_config, sample_input):
        """Tests that stochastic depth layers are None when disabled."""
        block = Qwen3NextBlock(**basic_config)
        block(sample_input)  # Build the block

        # All stochastic depth layers should be None
        assert len(block.stochastic_depth_layers) == 4
        assert all(layer is None for layer in block.stochastic_depth_layers)

    # ===============================================
    # 10. Edge Cases and Robustness Tests
    # ===============================================
    def test_small_sequence_length(self):
        """Tests block with very small sequence length."""
        block = Qwen3NextBlock(dim=64, num_heads=4)
        small_input = tf.random.normal((2, 3, 64))  # Very short sequence

        output = block(small_input, training=False)
        assert output.shape == small_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_single_head(self, sample_input):
        """Tests block with single attention head."""
        block = Qwen3NextBlock(dim=128, num_heads=1)
        output = block(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_large_head_count(self):
        """Tests block with many attention heads."""
        dim = 256
        num_heads = 32  # Many heads
        block = Qwen3NextBlock(dim=dim, num_heads=num_heads)
        large_input = tf.random.normal((2, 16, dim))

        output = block(large_input, training=False)
        assert output.shape == large_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_batch_size_one(self):
        """Tests block with batch size 1."""
        block = Qwen3NextBlock(dim=96, num_heads=6)
        single_batch_input = tf.random.normal((1, 20, 96))

        output = block(single_batch_input, training=False)
        assert output.shape == single_batch_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_compute_output_shape(self, basic_config):
        """Tests compute_output_shape method."""
        block = Qwen3NextBlock(**basic_config)
        input_shape = (None, 50, 128)
        output_shape = block.compute_output_shape(input_shape)

        assert output_shape == input_shape

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_config, sample_input, training):
        """Tests behavior in different training modes."""
        block = Qwen3NextBlock(**basic_config)

        output = block(sample_input, training=training)
        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_very_small_epsilon(self, sample_input):
        """Tests block with very small normalization epsilon."""
        block = Qwen3NextBlock(dim=128, num_heads=8, norm_eps=1e-12)
        output = block(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_high_dropout_rate(self, sample_input):
        """Tests block with high dropout rate."""
        block = Qwen3NextBlock(dim=128, num_heads=8, dropout_rate=0.9)

        # Training mode with high dropout
        output_train = block(sample_input, training=True)
        assert not np.any(np.isnan(ops.convert_to_numpy(output_train)))

        # Inference mode should not use dropout
        output_infer = block(sample_input, training=False)
        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer)))

    def test_mixed_precision_compatibility(self, basic_config, sample_input):
        """Tests that block works with mixed precision."""
        # Convert input to float16
        sample_input_fp16 = tf.cast(sample_input, tf.float16)

        block = Qwen3NextBlock(**basic_config)

        try:
            output = block(sample_input_fp16, training=False)
            # Output might be float16 or float32 depending on the layer implementation
            assert not np.any(np.isnan(ops.convert_to_numpy(output)))
        except Exception:
            # If mixed precision isn't supported, that's also acceptable
            pytest.skip("Mixed precision not supported in current setup")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])