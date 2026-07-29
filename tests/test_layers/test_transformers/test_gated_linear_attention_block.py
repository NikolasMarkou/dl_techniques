# tests/test_layers/test_transformers/test_gated_linear_attention_block.py

import os
import tempfile
from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import layers, models, ops

# Make sure to import the NEW, refactored GatedLinearAttentionBlock
from dl_techniques.layers.transformers import GatedLinearAttentionBlock


# --- Test Class ---
class TestGatedLinearAttentionBlock:
    """
    Comprehensive test suite for the refactored and configurable GatedLinearAttentionBlock layer.
    """

    # --- Fixtures for Reusability ---
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for the layer with default settings."""
        return {
            "dim": 64,
            "num_heads": 4,
            "max_seq_len": 256,
            "conv_kernel_size": 4,
            "dropout_rate": 0.0,
        }

    @pytest.fixture
    def custom_config(self) -> Dict[str, Any]:
        """Provides a configuration with custom head dim, norm, and activation."""
        return {
            "dim": 72,
            "num_heads": 6,
            "max_seq_len": 128,
            "head_dim": 16,
            "conv_kernel_size": 3,
            "activation": "gelu",
            "normalization_type": "layer_norm",
        }

    @pytest.fixture
    def ffn_config(self) -> Dict[str, Any]:
        """Provides a configuration that uses a custom SwiGLU FFN output."""
        return {
            "dim": 64,
            "num_heads": 4,
            "max_seq_len": 128,
            "ffn_type": "swiglu",
            # NOT 256: SwiGLU's own 2/3-rule default for output_dim=64 IS 256, so a
            # fixture of 256 cannot distinguish "honored" from "silently ignored".
            "intermediate_size": 320,
        }

    def test_swiglu_actually_honors_intermediate_size(self, ffn_config) -> None:
        """`intermediate_size` must reach the FFN -- it used to be silently discarded.

        SwiGLUFFN had no `hidden_dim` (it sized itself via `ffn_expansion_factor`), so the
        factory's kwarg filter DROPPED the `hidden_dim` GatedLinearAttentionBlock passed, so the FFN was
        built at SwiGLU's 2/3-rule default and nothing failed. The old test only asserted
        `hasattr(layer, "output_ffn")`, which was true either way.

        The fixture deliberately asks for 320, NOT 256: SwiGLU's own default for
        output_dim=64 IS 256, so a 256 fixture passes whether or not the value is honored.
        """
        layer = GatedLinearAttentionBlock(**ffn_config)
        assert layer.output_ffn.hidden_dim == 320, (
            f"intermediate_size=320 did not reach the SwiGLU FFN "
            f"(hidden_dim={layer.output_ffn.hidden_dim})"
        )

    @pytest.fixture
    def regularized_config(self) -> Dict[str, Any]:
        """Provides a config with regularization and custom initializers."""
        return {
            "dim": 32,
            "num_heads": 2,
            "max_seq_len": 64,
            "dropout_rate": 0.1,
            "use_bias": True,
            "kernel_initializer": "he_normal",
            "kernel_regularizer": keras.regularizers.L2(1e-4),
            "bias_regularizer": keras.regularizers.L1(1e-5),
        }

    @pytest.fixture
    def sample_input_64(self) -> tf.Tensor:
        """Provides a standard sample input tensor (dim=64)."""
        return tf.random.normal(shape=(4, 16, 64))

    @pytest.fixture
    def sample_input_72(self) -> tf.Tensor:
        """Provides sample input matching the custom config (dim=72)."""
        return tf.random.normal(shape=(2, 12, 72))

    @pytest.fixture
    def sample_input_32(self) -> tf.Tensor:
        """Provides sample input for the regularized config (dim=32)."""
        return tf.random.normal(shape=(3, 8, 32))

    # ===============================================
    # 1. Initialization and Build Tests
    # ===============================================
    def test_initialization_defaults(self, default_config):
        """Tests layer initialization with default parameters."""
        layer = GatedLinearAttentionBlock(**default_config)
        assert not layer.built
        assert layer.dim == 64
        assert layer.num_heads == 4
        assert layer.head_dim == 16
        assert layer.activation == "silu"
        assert layer.normalization_type == "zero_centered_rms_norm"
        assert layer.ffn_type is None
        assert layer.use_default_ffn

    def test_initialization_custom_config(self, custom_config):
        """Tests initialization with custom norm, activation, and head dim."""
        layer = GatedLinearAttentionBlock(**custom_config)
        assert layer.dim == 72
        assert layer.num_heads == 6
        assert layer.head_dim == 16
        assert layer.activation == "gelu"
        assert layer.normalization_type == "layer_norm"

    def test_initialization_with_custom_ffn(self, ffn_config):
        """Tests initialization with a custom FFN output."""
        layer = GatedLinearAttentionBlock(**ffn_config)
        assert layer.ffn_type == "swiglu"
        assert not layer.use_default_ffn
        assert hasattr(layer, "output_ffn")
        assert not hasattr(layer, "output_proj")

    def test_build_process_default(self, default_config, sample_input_64):
        """Tests that the layer builds correctly with default FFN."""
        layer = GatedLinearAttentionBlock(**default_config)
        output = layer(sample_input_64)
        assert layer.built
        assert output.shape == sample_input_64.shape
        # Check that default FFN layers are built
        assert layer.output_proj.built
        assert layer.output_gate_linear.built
        assert not hasattr(layer, "output_ffn")

    def test_build_process_custom_ffn(self, ffn_config, sample_input_64):
        """Tests that the layer builds correctly with a custom FFN."""
        layer = GatedLinearAttentionBlock(**ffn_config)
        output = layer(sample_input_64)
        assert layer.built
        assert output.shape == sample_input_64.shape
        # Check that the custom FFN layer is built
        assert layer.output_ffn.built
        assert not hasattr(layer, "output_proj")

    # ===============================================
    # 2. Parameter Validation Tests (Largely Unchanged)
    # ===============================================
    @pytest.mark.parametrize(
        "invalid_params, match_str",
        [
            ({"dim": 0}, "dim must be positive"),
            ({"num_heads": 0}, "num_heads must be positive"),
            ({"max_seq_len": -1}, "max_seq_len must be positive"),
            ({"head_dim": 0}, "head_dim must be positive"),
            ({"dim": 65, "num_heads": 4}, "dim .* must be divisible by num_heads"),
            ({"conv_kernel_size": 0}, "conv_kernel_size must be positive"),
            ({"dropout_rate": -0.1}, "dropout_rate must be in"),
            ({"intermediate_size": 0}, "intermediate_size must be positive"),
            ({"intermediate_size": -128}, "intermediate_size must be positive"),
        ],
    )
    def test_parameter_validation(self, invalid_params, match_str):
        """Tests various parameter validation checks."""
        config = {"dim": 64, "num_heads": 4, "max_seq_len": 256}
        config.update(invalid_params)
        with pytest.raises(ValueError, match=match_str):
            GatedLinearAttentionBlock(**config)

    def test_intermediate_size_none_is_valid(self):
        """`intermediate_size=None` means "derive from dim" and must NOT raise.

        Control for `test_parameter_validation`: without it, a validator that
        rejected *every* `intermediate_size` (including the default `None`)
        would still look green.
        """
        layer = GatedLinearAttentionBlock(
            dim=64, num_heads=4, max_seq_len=256, intermediate_size=None
        )
        assert layer.intermediate_size is None

    # ===============================================
    # 2b. max_seq_len Overflow Guard
    # ===============================================
    def test_overflow_raises_on_static_seq_len(self):
        """A statically known seq_len > max_seq_len must raise, not zero-fill.

        Regression for the probed defect: `max_seq_len=4` on a `(1, 10, 16)`
        input returned the CORRECT shape with timesteps 4-9 silently all-zero,
        because `ops.while_loop(..., maximum_iterations=max_seq_len)` capped the
        loop while the output buffer stayed zero-initialized. No exception, no
        warning.

        The match pattern names both concrete numbers, so an unrelated
        `ValueError` (e.g. a shape or dtype complaint) cannot satisfy it.
        """
        layer = GatedLinearAttentionBlock(dim=16, num_heads=2, max_seq_len=4)
        over_long = tf.random.normal(shape=(1, 10, 16))
        with pytest.raises(
            ValueError, match=r"Sequence length \(10\) exceeds max_seq_len \(4\)"
        ):
            layer(over_long)

    def test_overflow_raises_from_build(self):
        """The same guard fires from `build()` on a static shape tuple.

        `build()` and `call()` are guarded independently: a layer built at one
        length can be re-called at another, so neither site alone is sufficient.
        """
        layer = GatedLinearAttentionBlock(dim=16, num_heads=2, max_seq_len=4)
        with pytest.raises(
            ValueError, match=r"Sequence length \(10\) exceeds max_seq_len \(4\)"
        ):
            layer.build((1, 10, 16))

    def test_overflow_control_at_exactly_max_seq_len(self):
        """CONTROL: seq_len == max_seq_len must NOT raise and must do real work.

        Without this control, a layer that raised unconditionally -- or that
        raised at the wrong comparison (`>=` instead of `>`) -- would pass the
        two tests above. The output must additionally be finite and not
        all-zero, so a "guard passes but the scan is dead" state also fails.
        """
        layer = GatedLinearAttentionBlock(dim=16, num_heads=2, max_seq_len=4)
        exact = tf.random.normal(shape=(1, 4, 16))
        output = layer(exact, training=False)

        assert output.shape == exact.shape
        out_np = ops.convert_to_numpy(output)
        assert np.all(np.isfinite(out_np)), "output must be finite"
        assert np.any(out_np != 0.0), "output must not be all-zero"
        # Every timestep must carry signal -- the defect zeroed a *suffix*.
        per_step_absmax = np.max(np.abs(out_np), axis=(0, 2))
        assert np.all(per_step_absmax > 0.0), (
            f"some timestep is entirely zero: {per_step_absmax}"
        )

    def test_overflow_guard_does_not_fire_on_symbolic_seq_len(self):
        """A `None` sequence axis must build (with a warning), never raise.

        This is D-002's documented gap: the guard is static-only. A functional
        model with `keras.Input(shape=(None, dim))` is legitimate and must keep
        working, so `build()` warns rather than raising.
        """
        inputs = keras.Input(shape=(None, 16))
        outputs = GatedLinearAttentionBlock(dim=16, num_heads=2, max_seq_len=4)(inputs)
        model = keras.models.Model(inputs, outputs)

        # Within the cap, the dynamic path still computes normally.
        prediction = model(tf.random.normal(shape=(1, 4, 16)), training=False)
        assert prediction.shape == (1, 4, 16)

    def test_build_validation_input_shape(self, default_config):
        """Tests build validation for input shape."""
        layer = GatedLinearAttentionBlock(**default_config)
        with pytest.raises(ValueError, match="Expected 3D input shape"):
            layer.build((32, 64))
        with pytest.raises(ValueError, match="Input feature dim .* must match layer dim"):
            layer.build((4, 16, 32))

    # ===============================================
    # 3. Forward Pass and Core Behavior Tests
    # ===============================================
    @pytest.mark.parametrize(
        "config_fixture, input_fixture",
        [
            ("default_config", "sample_input_64"),
            ("custom_config", "sample_input_72"),
            ("ffn_config", "sample_input_64"),
            ("regularized_config", "sample_input_32"),
        ],
    )
    def test_forward_pass_various_configs(self, config_fixture, input_fixture, request):
        """Tests forward pass with various configurations."""
        config = request.getfixturevalue(config_fixture)
        sample_input = request.getfixturevalue(input_fixture)
        layer = GatedLinearAttentionBlock(**config)
        output = layer(sample_input, training=False)

        assert output.shape == sample_input.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_training_vs_inference_mode(self, regularized_config, sample_input_32):
        """Tests that layer behaves differently in training vs inference due to dropout."""
        layer = GatedLinearAttentionBlock(**regularized_config)
        output_train = layer(sample_input_32, training=True)
        output_infer = layer(sample_input_32, training=False)
        assert not np.allclose(
            ops.convert_to_numpy(output_train), ops.convert_to_numpy(output_infer)
        )

    def test_deterministic_inference(self, default_config, sample_input_64):
        """Tests that inference is deterministic."""
        layer = GatedLinearAttentionBlock(**default_config)
        output1 = layer(sample_input_64, training=False)
        output2 = layer(sample_input_64, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(output1), ops.convert_to_numpy(output2)
        )

    # ===============================================
    # 4. Serialization and Configuration Tests
    # ===============================================
    def test_get_config_completeness(self, custom_config):
        """Tests that get_config contains all new and old __init__ parameters."""
        layer = GatedLinearAttentionBlock(**custom_config)
        config = layer.get_config()
        for param in custom_config.keys():
            assert param in config
        # Check other important defaults
        assert "activation" in config
        assert "normalization_type" in config

    def test_from_config_reconstruction(self, regularized_config):
        """Tests that a layer can be fully reconstructed from its config."""
        original_layer = GatedLinearAttentionBlock(**regularized_config)
        config = original_layer.get_config()
        reconstructed_layer = GatedLinearAttentionBlock.from_config(config)
        new_config = reconstructed_layer.get_config()
        assert config == new_config

    @pytest.mark.parametrize(
        "config_fixture, input_fixture",
        [
            ("default_config", "sample_input_64"),
            ("custom_config", "sample_input_72"),
            ("ffn_config", "sample_input_64"),
            ("regularized_config", "sample_input_32"),
        ],
    )
    def test_full_serialization_cycle(self, config_fixture, input_fixture, request):
        """Tests the full save/load cycle for various configurations."""
        config = request.getfixturevalue(config_fixture)
        sample_input = request.getfixturevalue(input_fixture)

        inputs = layers.Input(shape=sample_input.shape[1:])
        outputs = GatedLinearAttentionBlock(**config)(inputs)
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
                rtol=1e-6, atol=1e-6
            )

    # ===============================================
    # 5. Gradient and Training Integration Tests
    # ===============================================
    def test_gradient_flow(self, default_config, sample_input_64):
        """Tests that gradients can be computed through the layer."""
        layer = GatedLinearAttentionBlock(**default_config)
        x_var = tf.Variable(sample_input_64)
        with tf.GradientTape() as tape:
            output = layer(x_var, training=True)
            loss = ops.mean(ops.square(output))
        gradients = tape.gradient(loss, layer.trainable_variables)
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)

    def test_model_training_loop_integration(self, default_config):
        """Tests integration into a standard training loop."""
        model = models.Sequential([
            layers.InputLayer(shape=(16, 64)),
            GatedLinearAttentionBlock(**default_config),
            layers.GlobalAveragePooling1D(),
            layers.Dense(10),
        ])
        model.compile("adam", "sparse_categorical_crossentropy")
        x_train = tf.random.normal((32, 16, 64))
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)
        history = model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)
        assert "loss" in history.history

    # ===============================================
    # 6. Dynamic Shape Handling Tests (Crucial Check)
    # ===============================================
    @pytest.mark.parametrize(
        "input_shape",
        [
            (None, 16, 64),  # Dynamic batch
            (4, None, 64),  # Dynamic sequence length
            (None, None, 64),  # Fully dynamic
        ],
    )
    def test_functional_model_with_dynamic_shapes(self, default_config, input_shape):
        """Tests that the layer works in a functional model with dynamic shapes."""
        try:
            inputs = keras.Input(shape=input_shape[1:])
            outputs = GatedLinearAttentionBlock(**default_config)(inputs)
            model = keras.models.Model(inputs, outputs)
        except Exception as e:
            pytest.fail(f"Failed to build model with dynamic shape {input_shape}. Error: {e}")

        # Test forward pass with a concrete shape
        concrete_input = tf.random.normal(shape=(4, 16, 64))
        prediction = model(concrete_input, training=False)
        assert prediction.shape == concrete_input.shape

    def test_dynamic_sequence_length_in_training_loop(self, default_config):
        """Tests a model with dynamic sequence length can be compiled and trained."""
        inputs = keras.Input(shape=(None, 64))
        outputs = GatedLinearAttentionBlock(**default_config)(inputs)
        pooled = keras.layers.GlobalAveragePooling1D()(outputs)
        logits = keras.layers.Dense(10)(pooled)
        model = keras.models.Model(inputs, logits)
        model.compile("adam", "sparse_categorical_crossentropy")

        x_train = tf.random.normal((8, 20, 64))
        y_train = tf.random.uniform([8], 0, 10, dtype=tf.int32)

        try:
            history = model.fit(x_train, y_train, epochs=1, verbose=0)
            assert "loss" in history.history
        except Exception as e:
            pytest.fail(f"Training failed with dynamic sequence length. Error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])