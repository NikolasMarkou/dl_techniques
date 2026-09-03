import pytest
import numpy as np
import tempfile
import os
from typing import Any, Dict

import keras
from keras import ops

from dl_techniques.layers.embedding.patch_embed_3d import PatchEmbed3D


class TestPatchEmbed3D:
    """Comprehensive test suite for PatchEmbed3D following house Keras 3 patterns."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample video clip tensor, (batch, T, H, W, C)."""
        return keras.random.normal([2, 8, 32, 32, 3])

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            "patch_size": 16,
            "tubelet_size": 2,
            "embed_dim": 96,
        }

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = PatchEmbed3D(patch_size=16, tubelet_size=2, embed_dim=64)

        assert layer.patch_size == (16, 16)
        assert layer.tubelet_size == 2
        assert layer.embed_dim == 64
        assert layer.use_bias is True
        assert layer.flatten is True
        assert not layer.built

        # Sub-layer should be created in __init__
        assert hasattr(layer, "proj")
        assert layer.proj is not None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_config = {
            "patch_size": (8, 16),
            "tubelet_size": 4,
            "embed_dim": 32,
            "kernel_initializer": "he_normal",
            "kernel_regularizer": keras.regularizers.L2(1e-4),
            "bias_initializer": "ones",
            "bias_regularizer": keras.regularizers.L1(1e-5),
            "activation": "relu",
            "use_bias": False,
            "flatten": False,
        }

        layer = PatchEmbed3D(**custom_config)

        assert layer.patch_size == (8, 16)
        assert layer.tubelet_size == 4
        assert layer.embed_dim == 32
        assert layer.use_bias is False
        assert layer.flatten is False

    def test_forward_pass_shape(
        self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor
    ):
        """Test forward pass shape: (B, T, H, W, C) -> (B, N, embed_dim)."""
        layer = PatchEmbed3D(**layer_config)

        output = layer(sample_input)

        assert layer.built

        # N = (T // tubelet) * (H // patch) * (W // patch)
        expected_n = (8 // 2) * (32 // 16) * (32 // 16)
        assert output.shape == (2, expected_n, 96)

        output_np = ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))

    def test_flatten_false_returns_grid(
        self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor
    ):
        """flatten=False must return the raw 5D grid, not the flattened sequence."""
        config = dict(layer_config)
        config["flatten"] = False
        layer = PatchEmbed3D(**config)

        output = layer(sample_input)

        expected_shape = (2, 8 // 2, 32 // 16, 32 // 16, 96)
        assert output.shape == expected_shape

        # compute_output_shape must agree.
        computed = layer.compute_output_shape((2, 8, 32, 32, 3))
        assert computed == expected_shape

    def test_compute_output_shape_flatten_true(self):
        """compute_output_shape for flatten=True must match the forward pass."""
        layer = PatchEmbed3D(patch_size=16, tubelet_size=2, embed_dim=64)
        input_shape = (None, 8, 32, 32, 3)
        output_shape = layer.compute_output_shape(input_shape)

        expected_n = (8 // 2) * (32 // 16) * (32 // 16)
        assert output_shape == (None, expected_n, 64)

    def test_degenerate_tubelet_size_one(self):
        """tubelet_size=1 must run without error (Conv3D collapses cleanly)."""
        layer = PatchEmbed3D(patch_size=8, tubelet_size=1, embed_dim=32)
        clip = keras.random.normal([2, 4, 16, 16, 3])

        output = layer(clip)

        expected_n = (4 // 1) * (16 // 8) * (16 // 8)
        assert output.shape == (2, expected_n, 32)

        output_np = ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))

    def test_serialization_cycle(
        self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor
    ):
        """CRITICAL TEST: get_config()/from_config() round-trip, max|delta|==0.0."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = PatchEmbed3D(**layer_config)(inputs)
        model = keras.Model(inputs, layer_output)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            original_np = ops.convert_to_numpy(original_prediction)
            loaded_np = ops.convert_to_numpy(loaded_prediction)
            max_delta = float(np.max(np.abs(original_np - loaded_np)))
            assert max_delta == 0.0, f"Predictions differ after serialization: max|delta|={max_delta}"

    def test_config_completeness(self, layer_config: Dict[str, Any]):
        """Test that get_config contains all __init__ parameters."""
        layer = PatchEmbed3D(**layer_config)
        config = layer.get_config()

        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"

        essential_keys = [
            "patch_size", "tubelet_size", "embed_dim", "use_bias", "flatten",
        ]
        for key in essential_keys:
            assert key in config, f"Missing essential key {key} in get_config()"

    def test_from_config_round_trip(self, layer_config: Dict[str, Any]):
        """get_config() -> from_config() reproduces an equivalent layer."""
        layer = PatchEmbed3D(**layer_config)
        config = layer.get_config()
        rebuilt = PatchEmbed3D.from_config(config)

        assert rebuilt.patch_size == layer.patch_size
        assert rebuilt.tubelet_size == layer.tubelet_size
        assert rebuilt.embed_dim == layer.embed_dim
        assert rebuilt.use_bias == layer.use_bias
        assert rebuilt.flatten == layer.flatten

    def test_validation_errors(self):
        """Test error conditions and edge cases (non-positive sizes)."""
        with pytest.raises(ValueError):
            PatchEmbed3D(patch_size=0, tubelet_size=2, embed_dim=32)

        with pytest.raises(ValueError):
            PatchEmbed3D(patch_size=16, tubelet_size=0, embed_dim=32)

        with pytest.raises(ValueError):
            PatchEmbed3D(patch_size=16, tubelet_size=2, embed_dim=-5)

        with pytest.raises(ValueError):
            PatchEmbed3D(patch_size=(1, 2, 3), tubelet_size=2, embed_dim=32)

    def test_build_invalid_rank(self):
        """build() must raise on a non-5D input shape."""
        layer = PatchEmbed3D(patch_size=16, tubelet_size=2, embed_dim=32)

        with pytest.raises(ValueError, match="Expected 5D input"):
            layer.build((2, 32, 32, 3))

    def test_build_indivisible_dims(self):
        """build() must raise when a dimension is not divisible by its patch size."""
        layer = PatchEmbed3D(patch_size=16, tubelet_size=2, embed_dim=32)

        with pytest.raises(ValueError):
            layer.build((2, 8, 33, 32, 3))  # 33 not divisible by 16

        layer2 = PatchEmbed3D(patch_size=16, tubelet_size=3, embed_dim=32)
        with pytest.raises(ValueError):
            layer2.build((2, 8, 32, 32, 3))  # 8 not divisible by 3

    def test_gradients_flow(
        self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor
    ):
        """Test gradient computation."""
        import tensorflow as tf

        layer = PatchEmbed3D(**layer_config)

        with tf.GradientTape() as tape:
            tf_input = tf.Variable(ops.convert_to_numpy(sample_input))
            output = layer(tf_input)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
        self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor, training
    ):
        """Test behavior in different training modes."""
        layer = PatchEmbed3D(**layer_config)

        output = layer(sample_input, training=training)
        assert output.shape[0] == sample_input.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
