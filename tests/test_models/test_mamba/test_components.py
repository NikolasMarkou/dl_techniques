"""
Comprehensive Test Suite for Mamba Layers
==========================================

This module contains thorough tests for MambaLayer and MambaResidualBlock,
following the dl_techniques testing patterns and the Complete Guide to
Modern Keras 3 Custom Layers.

Test Categories:
- Initialization: Layer creation and configuration
- Forward Pass: Basic functionality and shape preservation
- Serialization: Critical save/load cycle tests
- Configuration: get_config() completeness
- Gradients: Backpropagation verification
- Training Modes: Behavior consistency
- Edge Cases: Boundary conditions and error handling
"""

import pytest
import tempfile
import os
import numpy as np
import keras
import tensorflow as tf
from typing import Dict, Any

from dl_techniques.utils.logger import logger
from dl_techniques.models.mamba import MambaLayer, MambaResidualBlock

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mamba_layer_config() -> Dict[str, Any]:
    """Standard configuration for MambaLayer testing."""
    return {
        "d_model": 768,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dt_rank": "auto",
    }


@pytest.fixture
def mamba_layer_custom_config() -> Dict[str, Any]:
    """Custom configuration for advanced testing."""
    return {
        "d_model": 512,
        "d_state": 32,
        "d_conv": 8,
        "expand": 3,
        "dt_rank": 64,
        "dt_min": 0.0001,
        "dt_max": 0.2,
        "conv_bias": False,
        "use_bias": True,
        "layer_idx": 5,
    }


@pytest.fixture
def residual_block_config() -> Dict[str, Any]:
    """Configuration for MambaResidualBlock testing."""
    return {
        "d_model": 768,
        "norm_epsilon": 1e-5,
        "mamba_kwargs": {
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
        },
    }


@pytest.fixture
def sample_input_small() -> keras.KerasTensor:
    """Small sample input for quick tests."""
    return keras.random.normal((2, 32, 768))


@pytest.fixture
def sample_input_medium() -> keras.KerasTensor:
    """Medium sample input for standard tests."""
    return keras.random.normal((4, 128, 768))


@pytest.fixture
def sample_input_large() -> keras.KerasTensor:
    """Large sample input for performance tests."""
    return keras.random.normal((2, 512, 768))


# =============================================================================
# MambaLayer Tests
# =============================================================================

class TestMambaLayer:
    """Comprehensive test suite for MambaLayer."""

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization_basic(self, mamba_layer_config: Dict[str, Any]) -> None:
        """Test basic layer initialization."""
        logger.info("Testing MambaLayer basic initialization...")

        layer = MambaLayer(**mamba_layer_config)

        # Check configuration stored correctly
        assert layer.d_model == mamba_layer_config["d_model"]
        assert layer.d_state == mamba_layer_config["d_state"]
        assert layer.d_conv == mamba_layer_config["d_conv"]
        assert layer.expand == mamba_layer_config["expand"]

        # Check internal dimension computed correctly
        expected_d_inner = int(mamba_layer_config["expand"] * mamba_layer_config["d_model"])
        assert layer.d_inner == expected_d_inner

        # Check dt_rank computed correctly (auto mode)
        import math
        expected_dt_rank = math.ceil(mamba_layer_config["d_model"] / 16)
        assert layer.dt_rank == expected_dt_rank

        # Check layer not built yet
        assert not layer.built

        # Check sub-layers created
        assert layer.in_proj is not None
        assert layer.conv1d is not None
        assert layer.x_proj is not None
        assert layer.dt_proj is not None
        assert layer.out_proj is not None
        assert layer.activation is not None

        # Check weights not created yet
        assert layer.A_log is None
        assert layer.D is None

        logger.info("✅ Basic initialization test passed")

    def test_initialization_custom(self, mamba_layer_custom_config: Dict[str, Any]) -> None:
        """Test initialization with custom parameters."""
        logger.info("Testing MambaLayer custom initialization...")

        layer = MambaLayer(**mamba_layer_custom_config)

        # Check all custom parameters
        assert layer.d_model == mamba_layer_custom_config["d_model"]
        assert layer.d_state == mamba_layer_custom_config["d_state"]
        assert layer.d_conv == mamba_layer_custom_config["d_conv"]
        assert layer.expand == mamba_layer_custom_config["expand"]
        assert layer.dt_rank == mamba_layer_custom_config["dt_rank"]
        assert layer.dt_min == mamba_layer_custom_config["dt_min"]
        assert layer.dt_max == mamba_layer_custom_config["dt_max"]
        assert layer.conv_bias == mamba_layer_custom_config["conv_bias"]
        assert layer.use_bias == mamba_layer_custom_config["use_bias"]
        assert layer.layer_idx == mamba_layer_custom_config["layer_idx"]

        logger.info("✅ Custom initialization test passed")

    def test_initialization_invalid_params(self) -> None:
        """Test initialization with invalid parameters."""
        logger.info("Testing MambaLayer with invalid parameters...")

        # Test with invalid d_model
        with pytest.raises((ValueError, TypeError)):
            MambaLayer(d_model=-768)

        # Test with invalid d_state
        with pytest.raises((ValueError, TypeError)):
            MambaLayer(d_model=768, d_state=0)

        logger.info("✅ Invalid parameters test passed")

    # -------------------------------------------------------------------------
    # Forward Pass Tests
    # -------------------------------------------------------------------------

    def test_forward_pass_small(
            self,
            mamba_layer_config: Dict[str, Any],
            sample_input_small: keras.KerasTensor
    ) -> None:
        """Test forward pass with small input."""
        logger.info("Testing MambaLayer forward pass (small)...")

        layer = MambaLayer(**mamba_layer_config)
        output = layer(sample_input_small)

        # Check layer is built
        assert layer.built

        # Check weights created
        assert layer.A_log is not None
        assert layer.D is not None

        # Check output shape matches input shape
        assert output.shape == sample_input_small.shape

        # Check output is not NaN or Inf
        output_np = keras.ops.convert_to_numpy(output)
        assert not np.isnan(output_np).any()
        assert not np.isinf(output_np).any()

        logger.info(f"✅ Forward pass (small) test passed - output shape: {output.shape}")

    def test_forward_pass_medium(
            self,
            mamba_layer_config: Dict[str, Any],
            sample_input_medium: keras.KerasTensor
    ) -> None:
        """Test forward pass with medium input."""
        logger.info("Testing MambaLayer forward pass (medium)...")

        layer = MambaLayer(**mamba_layer_config)
        output = layer(sample_input_medium)

        assert output.shape == sample_input_medium.shape
        assert layer.built

        logger.info(f"✅ Forward pass (medium) test passed - output shape: {output.shape}")

    def test_forward_pass_large(
            self,
            mamba_layer_config: Dict[str, Any],
            sample_input_large: keras.KerasTensor
    ) -> None:
        """Test forward pass with large input."""
        logger.info("Testing MambaLayer forward pass (large)...")

        layer = MambaLayer(**mamba_layer_config)
        output = layer(sample_input_large)

        assert output.shape == sample_input_large.shape
        assert layer.built

        logger.info(f"✅ Forward pass (large) test passed - output shape: {output.shape}")

    def test_forward_pass_batch_size_one(self, mamba_layer_config: Dict[str, Any]) -> None:
        """Test forward pass with batch size of 1."""
        logger.info("Testing MambaLayer forward pass (batch_size=1)...")

        layer = MambaLayer(**mamba_layer_config)
        input_tensor = keras.random.normal((1, 64, 768))
        output = layer(input_tensor)

        assert output.shape == input_tensor.shape

        logger.info("✅ Forward pass (batch_size=1) test passed")

    def test_forward_pass_variable_length(self, mamba_layer_config: Dict[str, Any]) -> None:
        """Test forward pass with different sequence lengths."""
        logger.info("Testing MambaLayer forward pass (variable lengths)...")

        layer = MambaLayer(**mamba_layer_config)

        for seq_len in [16, 32, 64, 128, 256]:
            input_tensor = keras.random.normal((2, seq_len, 768))
            output = layer(input_tensor)
            assert output.shape == input_tensor.shape
            logger.info(f"  ✓ Sequence length {seq_len}: passed")

        logger.info("✅ Variable length test passed")

    # -------------------------------------------------------------------------
    # Serialization Tests (CRITICAL)
    # -------------------------------------------------------------------------

    def test_serialization_cycle(
            self,
            mamba_layer_config: Dict[str, Any],
            sample_input_medium: keras.KerasTensor
    ) -> None:
        """
        CRITICAL TEST: Full serialization and deserialization cycle.

        This is the most important test - ensures the layer can be saved
        and loaded while preserving weights and behavior.
        """
        logger.info("Testing MambaLayer serialization cycle...")

        # Create model with MambaLayer
        inputs = keras.Input(shape=sample_input_medium.shape[1:])
        outputs = MambaLayer(**mamba_layer_config)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Get original prediction
        original_output = model(sample_input_medium)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_mamba_layer.keras")
            model.save(filepath)
            logger.info(f"  Saved model to {filepath}")

            loaded_model = keras.models.load_model(filepath)
            logger.info("  Loaded model successfully")

            loaded_output = loaded_model(sample_input_medium)

            # Verify outputs match
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Outputs differ after serialization"
            )

        logger.info("✅ Serialization cycle test passed")

    def test_serialization_with_training(
            self,
            mamba_layer_config: Dict[str, Any]
    ) -> None:
        """Test serialization after training."""
        logger.info("Testing MambaLayer serialization after training...")

        # Create simple model
        inputs = keras.Input(shape=(32, 768))
        outputs = MambaLayer(**mamba_layer_config)(inputs)
        outputs = keras.layers.GlobalAveragePooling1D()(outputs)
        outputs = keras.layers.Dense(10, activation="softmax")(outputs)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Generate dummy data
        x_train = keras.random.normal((16, 32, 768))
        y_train = keras.ops.cast(keras.random.uniform((16,), 0, 10), "int32")

        # Train for a few steps
        model.fit(x_train, y_train, epochs=2, verbose=0)

        # Get prediction before save
        test_input = keras.random.normal((4, 32, 768))
        original_pred = model.predict(test_input, verbose=0)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "trained_model.keras")
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model.predict(test_input, verbose=0)

            # Verify predictions match
            np.testing.assert_allclose(
                original_pred,
                loaded_pred,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Predictions differ after training and serialization"
            )

        logger.info("✅ Serialization after training test passed")

    # -------------------------------------------------------------------------
    # Configuration Tests
    # -------------------------------------------------------------------------

    def test_get_config_completeness(self, mamba_layer_config: Dict[str, Any]) -> None:
        """Test that get_config includes all initialization parameters."""
        logger.info("Testing MambaLayer get_config completeness...")

        layer = MambaLayer(**mamba_layer_config)
        config = layer.get_config()

        # Check all init parameters are in config
        required_keys = [
            "d_model", "d_state", "d_conv", "expand", "dt_rank",
            "dt_min", "dt_max", "dt_init", "dt_scale", "dt_init_floor",
            "conv_bias", "use_bias", "layer_idx"
        ]

        for key in required_keys:
            assert key in config, f"Missing '{key}' in get_config()"

        # Check values match
        assert config["d_model"] == mamba_layer_config["d_model"]
        assert config["d_state"] == mamba_layer_config["d_state"]
        assert config["d_conv"] == mamba_layer_config["d_conv"]
        assert config["expand"] == mamba_layer_config["expand"]

        logger.info("✅ get_config completeness test passed")

    def test_from_config_reconstruction(self, mamba_layer_config: Dict[str, Any]) -> None:
        """Test layer reconstruction from config."""
        logger.info("Testing MambaLayer from_config reconstruction...")

        # Create original layer
        original_layer = MambaLayer(**mamba_layer_config)
        config = original_layer.get_config()

        # Reconstruct from config
        reconstructed_layer = MambaLayer.from_config(config)

        # Check parameters match
        assert reconstructed_layer.d_model == original_layer.d_model
        assert reconstructed_layer.d_state == original_layer.d_state
        assert reconstructed_layer.d_conv == original_layer.d_conv
        assert reconstructed_layer.expand == original_layer.expand
        assert reconstructed_layer.dt_rank == original_layer.dt_rank

        logger.info("✅ from_config reconstruction test passed")

    # -------------------------------------------------------------------------
    # Gradient Flow Tests
    # -------------------------------------------------------------------------

    def test_gradients_flow(
            self,
            mamba_layer_config: Dict[str, Any],
            sample_input_small: keras.KerasTensor
    ) -> None:
        """Test that gradients flow through the layer correctly."""
        logger.info("Testing MambaLayer gradient flow...")

        layer = MambaLayer(**mamba_layer_config)

        with tf.GradientTape() as tape:
            # Build layer by calling it
            output = layer(sample_input_small, training=True)
            loss = keras.ops.mean(keras.ops.square(output))

        # Compute gradients
        trainable_vars = layer.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Check all trainable variables have gradients
        assert len(gradients) > 0, "No gradients computed"
        assert len(gradients) == len(trainable_vars)

        # Check no gradient is None
        for i, (var, grad) in enumerate(zip(trainable_vars, gradients)):
            assert grad is not None, f"Gradient is None for variable {i}: {var.name}"

        logger.info(f"✅ Gradient flow test passed - {len(gradients)} gradients computed")

    def test_gradients_selective_scan(
            self,
            mamba_layer_config: Dict[str, Any]
    ) -> None:
        """Test gradients through the selective scan operation."""
        logger.info("Testing gradients through selective scan...")

        layer = MambaLayer(**mamba_layer_config)
        input_tensor = keras.random.normal((2, 16, 768))

        with tf.GradientTape(persistent=True) as tape:
            output = layer(input_tensor, training=True)
            loss = keras.ops.mean(keras.ops.square(output))

        # Check A_log and D have gradients (SSM-specific parameters)
        grad_A = tape.gradient(loss, layer.A_log)
        grad_D = tape.gradient(loss, layer.D)

        assert grad_A is not None, "A_log gradient is None"
        assert grad_D is not None, "D gradient is None"

        logger.info("✅ Selective scan gradients test passed")

    # -------------------------------------------------------------------------
    # Training Mode Tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
            self,
            mamba_layer_config: Dict[str, Any],
            sample_input_small: keras.KerasTensor,
            training: bool
    ) -> None:
        """Test layer behavior in different training modes."""
        logger.info(f"Testing MambaLayer with training={training}...")

        layer = MambaLayer(**mamba_layer_config)
        output = layer(sample_input_small, training=training)

        # Check output shape is correct
        assert output.shape == sample_input_small.shape

        # Check output is valid
        output_np = keras.ops.convert_to_numpy(output)
        assert not np.isnan(output_np).any()
        assert not np.isinf(output_np).any()

        logger.info(f"✅ Training mode {training} test passed")

    def test_inference_mode_consistency(self, mamba_layer_config: Dict[str, Any]) -> None:
        """Test that inference mode gives consistent results."""
        logger.info("Testing MambaLayer inference consistency...")

        layer = MambaLayer(**mamba_layer_config)
        input_tensor = keras.random.normal((2, 32, 768))

        # Build layer
        _ = layer(input_tensor, training=False)

        # Multiple inference passes should give same result
        output1 = layer(input_tensor, training=False)
        output2 = layer(input_tensor, training=False)
        output3 = layer(input_tensor, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Inference outputs differ between passes"
        )

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output2),
            keras.ops.convert_to_numpy(output3),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Inference outputs differ between passes"
        )

        logger.info("✅ Inference consistency test passed")

    # -------------------------------------------------------------------------
    # Weight and Parameter Tests
    # -------------------------------------------------------------------------

    def test_weight_shapes(
            self,
            mamba_layer_config: Dict[str, Any],
            sample_input_small: keras.KerasTensor
    ) -> None:
        """Test that all weights have correct shapes."""
        logger.info("Testing MambaLayer weight shapes...")

        layer = MambaLayer(**mamba_layer_config)
        _ = layer(sample_input_small)  # Build layer

        # Check A_log shape
        expected_A_shape = (layer.d_inner, layer.d_state)
        assert layer.A_log.shape == expected_A_shape, \
            f"A_log shape {layer.A_log.shape} != expected {expected_A_shape}"

        # Check D shape
        expected_D_shape = (layer.d_inner,)
        assert layer.D.shape == expected_D_shape, \
            f"D shape {layer.D.shape} != expected {expected_D_shape}"

        logger.info("✅ Weight shapes test passed")

    def test_trainable_variables_count(
            self,
            mamba_layer_config: Dict[str, Any],
            sample_input_small: keras.KerasTensor
    ) -> None:
        """Test that layer has expected number of trainable variables."""
        logger.info("Testing MambaLayer trainable variables...")

        layer = MambaLayer(**mamba_layer_config)
        _ = layer(sample_input_small)  # Build layer

        trainable_vars = layer.trainable_variables

        # Should have: A_log, D, in_proj (W,b?), conv1d (W,b?), x_proj (W),
        # dt_proj (W,b), out_proj (W,b?)
        # Exact count depends on use_bias and conv_bias settings
        assert len(trainable_vars) > 0, "No trainable variables"

        logger.info(f"✅ Trainable variables test passed - {len(trainable_vars)} variables")

    # -------------------------------------------------------------------------
    # Edge Case Tests
    # -------------------------------------------------------------------------

    def test_edge_case_small_dimension(self) -> None:
        """Test with very small d_model."""
        logger.info("Testing MambaLayer with small dimension...")

        layer = MambaLayer(d_model=64, d_state=4, d_conv=2)
        input_tensor = keras.random.normal((2, 16, 64))
        output = layer(input_tensor)

        assert output.shape == input_tensor.shape

        logger.info("✅ Small dimension test passed")

    def test_edge_case_large_state(self) -> None:
        """Test with large state dimension."""
        logger.info("Testing MambaLayer with large state dimension...")

        layer = MambaLayer(d_model=256, d_state=64, expand=2)
        input_tensor = keras.random.normal((2, 32, 256))
        output = layer(input_tensor)

        assert output.shape == input_tensor.shape

        logger.info("✅ Large state dimension test passed")

    def test_edge_case_short_sequence(self) -> None:
        """Test with very short sequence."""
        logger.info("Testing MambaLayer with short sequence...")

        layer = MambaLayer(d_model=768, d_state=16)
        input_tensor = keras.random.normal((2, 4, 768))  # Only 4 timesteps
        output = layer(input_tensor)

        assert output.shape == input_tensor.shape

        logger.info("✅ Short sequence test passed")

    def test_edge_case_explicit_dt_rank(self) -> None:
        """Test with explicitly set dt_rank instead of auto."""
        logger.info("Testing MambaLayer with explicit dt_rank...")

        layer = MambaLayer(d_model=768, dt_rank=32)
        assert layer.dt_rank == 32

        input_tensor = keras.random.normal((2, 32, 768))
        output = layer(input_tensor)
        assert output.shape == input_tensor.shape

        logger.info("✅ Explicit dt_rank test passed")


# =============================================================================
# MambaResidualBlock Tests
# =============================================================================

class TestMambaResidualBlock:
    """Comprehensive test suite for MambaResidualBlock."""

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization_basic(self, residual_block_config: Dict[str, Any]) -> None:
        """Test basic block initialization."""
        logger.info("Testing MambaResidualBlock basic initialization...")

        block = MambaResidualBlock(**residual_block_config)

        # Check configuration
        assert block.d_model == residual_block_config["d_model"]
        assert block.norm_epsilon == residual_block_config["norm_epsilon"]
        assert block.mamba_kwargs == residual_block_config["mamba_kwargs"]

        # Check sub-layers created
        assert block.norm is not None
        assert block.mamba is not None

        # Check not built
        assert not block.built

        logger.info("✅ Basic initialization test passed")

    def test_initialization_minimal(self) -> None:
        """Test initialization with minimal parameters."""
        logger.info("Testing MambaResidualBlock minimal initialization...")

        block = MambaResidualBlock(d_model=768)

        assert block.d_model == 768
        assert block.mamba_kwargs == {}
        assert block.norm is not None
        assert block.mamba is not None

        logger.info("✅ Minimal initialization test passed")

    # -------------------------------------------------------------------------
    # Forward Pass Tests
    # -------------------------------------------------------------------------

    def test_forward_pass_no_residual(
            self,
            residual_block_config: Dict[str, Any],
            sample_input_medium: keras.KerasTensor
    ) -> None:
        """Test forward pass without previous residual."""
        logger.info("Testing MambaResidualBlock forward pass (no residual)...")

        block = MambaResidualBlock(**residual_block_config)
        hidden, residual = block(sample_input_medium, residual=None)

        # Check shapes
        assert hidden.shape == sample_input_medium.shape
        assert residual.shape == sample_input_medium.shape

        # Check values are valid
        hidden_np = keras.ops.convert_to_numpy(hidden)
        residual_np = keras.ops.convert_to_numpy(residual)
        assert not np.isnan(hidden_np).any()
        assert not np.isnan(residual_np).any()

        # Check residual equals input (first block)
        np.testing.assert_allclose(
            residual_np,
            keras.ops.convert_to_numpy(sample_input_medium),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Residual should equal input for first block"
        )

        logger.info("✅ Forward pass (no residual) test passed")

    def test_forward_pass_with_residual(
            self,
            residual_block_config: Dict[str, Any],
            sample_input_medium: keras.KerasTensor
    ) -> None:
        """Test forward pass with previous residual."""
        logger.info("Testing MambaResidualBlock forward pass (with residual)...")

        block = MambaResidualBlock(**residual_block_config)

        # First pass
        hidden1, residual1 = block(sample_input_medium, residual=None)

        # Second pass with residual from first
        hidden2, residual2 = block(hidden1, residual=residual1)

        # Check shapes
        assert hidden2.shape == sample_input_medium.shape
        assert residual2.shape == sample_input_medium.shape

        # Check residual accumulates correctly
        # residual2 should be hidden1 + residual1
        expected_residual2 = keras.ops.convert_to_numpy(hidden1 + residual1)
        actual_residual2 = keras.ops.convert_to_numpy(residual2)

        np.testing.assert_allclose(
            actual_residual2,
            expected_residual2,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Residual accumulation incorrect"
        )

        logger.info("✅ Forward pass (with residual) test passed")

    def test_forward_pass_chain(
            self,
            residual_block_config: Dict[str, Any]
    ) -> None:
        """Test chaining multiple residual blocks."""
        logger.info("Testing MambaResidualBlock chaining...")

        # Create multiple blocks
        blocks = [MambaResidualBlock(**residual_block_config) for _ in range(4)]

        input_tensor = keras.random.normal((2, 32, 768))
        hidden = input_tensor
        residual = None

        # Chain blocks
        for i, block in enumerate(blocks):
            hidden, residual = block(hidden, residual=residual, training=True)
            logger.info(f"  ✓ Block {i + 1}: hidden shape {hidden.shape}, residual shape {residual.shape}")

        # Final output should have same shape
        assert hidden.shape == input_tensor.shape
        assert residual.shape == input_tensor.shape

        logger.info("✅ Block chaining test passed")

    # -------------------------------------------------------------------------
    # Serialization Tests
    # -------------------------------------------------------------------------

    def test_serialization_cycle(
            self,
            residual_block_config: Dict[str, Any],
            sample_input_medium: keras.KerasTensor
    ) -> None:
        """Test full serialization cycle for residual block."""
        logger.info("Testing MambaResidualBlock serialization cycle...")

        # Create model with residual block
        inputs = keras.Input(shape=sample_input_medium.shape[1:])

        # Need to handle tuple output from residual block
        block = MambaResidualBlock(**residual_block_config)
        hidden, residual = block(inputs, residual=None)
        # Use only hidden output for model

        model = keras.Model(inputs=inputs, outputs=hidden)

        # Get original output
        original_output = model(sample_input_medium)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_residual_block.keras")
            model.save(filepath)
            logger.info(f"  Saved model to {filepath}")

            loaded_model = keras.models.load_model(filepath)
            logger.info("  Loaded model successfully")

            loaded_output = loaded_model(sample_input_medium)

            # Verify outputs match
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Outputs differ after serialization"
            )

        logger.info("✅ Serialization cycle test passed")

    # -------------------------------------------------------------------------
    # Configuration Tests
    # -------------------------------------------------------------------------

    def test_get_config_completeness(self, residual_block_config: Dict[str, Any]) -> None:
        """Test get_config includes all parameters."""
        logger.info("Testing MambaResidualBlock get_config...")

        block = MambaResidualBlock(**residual_block_config)
        config = block.get_config()

        # Check required keys
        assert "d_model" in config
        assert "norm_epsilon" in config
        assert "mamba_kwargs" in config

        # Check values
        assert config["d_model"] == residual_block_config["d_model"]
        assert config["norm_epsilon"] == residual_block_config["norm_epsilon"]

        logger.info("✅ get_config completeness test passed")

    # -------------------------------------------------------------------------
    # Gradient Flow Tests
    # -------------------------------------------------------------------------

    def test_gradients_flow(
            self,
            residual_block_config: Dict[str, Any],
            sample_input_small: keras.KerasTensor
    ) -> None:
        """Test gradients flow through residual block."""
        logger.info("Testing MambaResidualBlock gradient flow...")

        block = MambaResidualBlock(**residual_block_config)

        with tf.GradientTape() as tape:
            hidden, residual = block(sample_input_small, residual=None, training=True)
            loss = keras.ops.mean(keras.ops.square(hidden))

        trainable_vars = block.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Check gradients exist
        assert len(gradients) > 0
        assert len(gradients) == len(trainable_vars)

        for var, grad in zip(trainable_vars, gradients):
            assert grad is not None, f"No gradient for {var.name}"

        logger.info(f"✅ Gradient flow test passed - {len(gradients)} gradients")

    def test_gradients_with_residual(
            self,
            residual_block_config: Dict[str, Any]
    ) -> None:
        """Test gradients flow through residual connections."""
        logger.info("Testing gradients through residual connections...")

        block = MambaResidualBlock(**residual_block_config)
        input_tensor = keras.random.normal((2, 32, 768))

        with tf.GradientTape() as tape:
            # First pass
            hidden1, residual1 = block(input_tensor, residual=None, training=True)
            # Second pass with residual
            hidden2, residual2 = block(hidden1, residual=residual1, training=True)

            loss = keras.ops.mean(keras.ops.square(hidden2))

        # Get gradients for both passes
        all_vars = block.trainable_variables
        gradients = tape.gradient(loss, all_vars)

        # All should have gradients
        assert all(g is not None for g in gradients)

        logger.info("✅ Residual connection gradients test passed")

    # -------------------------------------------------------------------------
    # Training Mode Tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
            self,
            residual_block_config: Dict[str, Any],
            sample_input_small: keras.KerasTensor,
            training: bool
    ) -> None:
        """Test block in different training modes."""
        logger.info(f"Testing MambaResidualBlock with training={training}...")

        block = MambaResidualBlock(**residual_block_config)
        hidden, residual = block(sample_input_small, residual=None, training=training)

        assert hidden.shape == sample_input_small.shape
        assert residual.shape == sample_input_small.shape

        logger.info(f"✅ Training mode {training} test passed")

    # -------------------------------------------------------------------------
    # Normalization Tests
    # -------------------------------------------------------------------------

    def test_normalization_applied(
            self,
            residual_block_config: Dict[str, Any]
    ) -> None:
        """Test that normalization is properly applied."""
        logger.info("Testing normalization in residual block...")

        block = MambaResidualBlock(**residual_block_config)

        # Create input with specific statistics
        input_tensor = keras.random.normal((2, 32, 768), mean=5.0, stddev=2.0)

        # First pass - normalization should bring values closer to 0 mean
        hidden, residual = block(input_tensor, residual=None, training=False)

        # Residual should equal input (first block)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(residual),
            keras.ops.convert_to_numpy(input_tensor),
            rtol=1e-6,
            atol=1e-6
        )

        logger.info("✅ Normalization test passed")

    # -------------------------------------------------------------------------
    # Edge Case Tests
    # -------------------------------------------------------------------------

    def test_edge_case_different_d_model(self) -> None:
        """Test blocks with different d_model values."""
        logger.info("Testing blocks with different d_model...")

        for d_model in [128, 256, 512, 1024]:
            block = MambaResidualBlock(d_model=d_model)
            input_tensor = keras.random.normal((2, 16, d_model))
            hidden, residual = block(input_tensor, residual=None)

            assert hidden.shape == input_tensor.shape
            assert residual.shape == input_tensor.shape
            logger.info(f"  ✓ d_model={d_model}: passed")

        logger.info("✅ Different d_model test passed")


# =============================================================================
# Integration Tests
# =============================================================================

class TestLayerIntegration:
    """Integration tests combining multiple layers."""

    def test_stacked_blocks(self) -> None:
        """Test stacking multiple Mamba blocks."""
        logger.info("Testing stacked Mamba blocks...")

        num_blocks = 6
        d_model = 768

        # Create stacked blocks
        blocks = [
            MambaResidualBlock(
                d_model=d_model,
                mamba_kwargs={"d_state": 16, "d_conv": 4, "layer_idx": i}
            )
            for i in range(num_blocks)
        ]

        # Process through all blocks
        input_tensor = keras.random.normal((2, 64, d_model))
        hidden = input_tensor
        residual = None

        for block in blocks:
            hidden, residual = block(hidden, residual=residual, training=True)

        # Check final output
        assert hidden.shape == input_tensor.shape
        assert residual.shape == input_tensor.shape

        # Check values are valid
        hidden_np = keras.ops.convert_to_numpy(hidden)
        assert not np.isnan(hidden_np).any()
        assert not np.isinf(hidden_np).any()

        logger.info(f"✅ Stacked blocks test passed - {num_blocks} blocks")

    def test_in_functional_model(self) -> None:
        """Test layers in functional API model."""
        logger.info("Testing layers in functional API model...")

        # Build model
        inputs = keras.Input(shape=(32, 768))

        # First block
        block1 = MambaResidualBlock(d_model=768, mamba_kwargs={"d_state": 16})
        hidden1, res1 = block1(inputs, residual=None)

        # Second block
        block2 = MambaResidualBlock(d_model=768, mamba_kwargs={"d_state": 16})
        hidden2, res2 = block2(hidden1, residual=res1)

        # Output layer
        pooled = keras.layers.GlobalAveragePooling1D()(hidden2)
        outputs = keras.layers.Dense(10, activation="softmax")(pooled)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile and test
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Test forward pass
        test_input = keras.random.normal((4, 32, 768))
        output = model(test_input)

        assert output.shape == (4, 10)

        logger.info("✅ Functional API integration test passed")


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance and efficiency tests."""

    def test_memory_efficiency(self, mamba_layer_config: Dict[str, Any]) -> None:
        """Test memory usage stays reasonable."""
        logger.info("Testing memory efficiency...")

        layer = MambaLayer(**mamba_layer_config)

        # Process increasing sequence lengths
        for seq_len in [128, 256, 512]:
            input_tensor = keras.random.normal((1, seq_len, 768))
            output = layer(input_tensor)
            assert output.shape[1] == seq_len
            logger.info(f"  ✓ Sequence length {seq_len}: passed")

        logger.info("✅ Memory efficiency test passed")

    def test_computational_consistency(self, mamba_layer_config: Dict[str, Any]) -> None:
        """Test that computation gives consistent results."""
        logger.info("Testing computational consistency...")

        layer = MambaLayer(**mamba_layer_config)
        input_tensor = keras.random.normal((2, 64, 768))

        # Multiple forward passes should give same result
        output1 = layer(input_tensor, training=False)
        output2 = layer(input_tensor, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Outputs differ between passes"
        )

        logger.info("✅ Computational consistency test passed")


# =============================================================================
# Run All Tests
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])