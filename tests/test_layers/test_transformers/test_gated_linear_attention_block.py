# tests/test_layers/test_transformers/test_gated_linear_attention_block.py

import contextlib
import os
import sys
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
    def test_neither_branch_truncates_past_max_seq_len(self):
        """`max_seq_len` must not truncate EITHER branch. It is advisory only.

        History, because this test has now been rewritten twice and the reason
        matters:

        1. Originally it asserted that `build()`/`call()` RAISE on a static length
           over `max_seq_len` -- correct at the time, because `_sequential_scan`
           ran under `maximum_iterations=max_seq_len` and returned zeros past the
           cap.
        2. D-010 moved that raise into `_sequential_scan`, on the grounds that the
           chunked branch does not truncate. That was true but insufficient: with
           `keras.Input(shape=(None, dim))` TF eventually relaxes the trace
           signature and dispatches to the SEQUENTIAL branch with a symbolic
           length, where a static guard cannot fire. Measured at that revision:
           `max_seq_len=8` returned **52 of 60 timesteps all-zero**, silently.
        3. D-018 removed the truncation itself -- `maximum_iterations=seq_len` --
           so there is nothing to guard on either branch, and the raise was
           deleted rather than relocated again.

        Restoring `maximum_iterations=self.max_seq_len` fails this test.
        """
        seq_len, max_seq_len = 40, 8
        num_heads, head_dim = 2, 4
        with global_dtype_policy("float64"):
            args = _scan_inputs(seq_len, num_heads, head_dim, "float64", seed=91)
            oracle = numpy_gated_linear_recurrence(
                *[keras.ops.convert_to_numpy(a) for a in args]
            )
            tight = GatedLinearAttentionBlock(
                dim=num_heads * head_dim, num_heads=num_heads, head_dim=head_dim,
                max_seq_len=max_seq_len, chunk_size=64,
            )
            roomy = GatedLinearAttentionBlock(
                dim=num_heads * head_dim, num_heads=num_heads, head_dim=head_dim,
                max_seq_len=1024, chunk_size=64,
            )
            seq_tight = keras.ops.convert_to_numpy(tight._sequential_scan(*args))
            seq_roomy = keras.ops.convert_to_numpy(roomy._sequential_scan(*args))
            chunked = keras.ops.convert_to_numpy(
                tight._chunked_scan(*args, seq_len)
            )

        # The declared cap must not change the answer at all.
        assert np.array_equal(seq_tight, seq_roomy), (
            f"max_seq_len changed the sequential result: max|diff|="
            f"{float(np.abs(seq_tight - seq_roomy).max()):.3e}. It is advisory and "
            f"must not affect the computation."
        )
        for label, got in (("sequential", seq_tight), ("chunked", chunked)):
            err = float(np.abs(got - oracle).max())
            assert err <= 1e-10, (
                f"{label} branch disagrees with the NumPy oracle past "
                f"max_seq_len={max_seq_len}: max|diff|={err:.3e}"
            )
            per_step = np.max(np.abs(got), axis=(0, 2, 3))
            assert np.all(per_step > 0.0), (
                f"{label} branch returned dead timesteps past "
                f"max_seq_len={max_seq_len} — it is still truncating: {per_step}"
            )

    def test_symbolic_input_model_does_not_truncate_past_max_seq_len(self):
        """The END-TO-END scenario CRITICAL-2 was about, not a private-method proxy.

        `test_neither_branch_truncates_past_max_seq_len` calls the scan methods
        directly with static shapes. That is a proxy: it never reaches the state
        that actually broke, which needs (a) a symbolic sequence axis, and (b) a
        SECOND distinct length, so TF relaxes the trace signature and dispatches
        to `_sequential_scan` with a symbolic length where no static guard can
        fire.

        Measured before D-018, at exactly this configuration: L=40 was fine
        (chunked) and L=60 returned **52 of 60 timesteps all-zero**, silently.
        Restoring `maximum_iterations=self.max_seq_len` fails HERE, and this is
        the only test in the file that would catch it end-to-end.
        """
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, head_dim=8, max_seq_len=8, chunk_size=64
        )
        inputs = keras.Input(shape=(None, 32))
        model = keras.Model(inputs, layer(inputs))

        # Order matters: the SECOND, longer length is what triggers relaxation.
        for seq_len in (40, 60):
            x = np.random.default_rng(seq_len).normal(
                size=(2, seq_len, 32)
            ).astype("float32")
            out = model.predict(x, verbose=0)
            assert out.shape == (2, seq_len, 32)
            assert np.isfinite(out).all()
            per_step = np.max(np.abs(out), axis=(0, 2))
            dead = int((per_step == 0.0).sum())
            assert dead == 0, (
                f"L={seq_len} with max_seq_len={layer.max_seq_len}: {dead} of "
                f"{seq_len} timesteps came back ALL-ZERO. The scan is truncating "
                f"at the declared cap and returning the zero-initialized buffer "
                f"as if it were output."
            )

    def test_exceeding_declared_max_seq_len_warns_but_does_not_raise(self, caplog):
        """The advisory replaces the raise: a warning, and a correct answer."""
        import logging
        layer = GatedLinearAttentionBlock(
            dim=16, num_heads=2, max_seq_len=4, chunk_size=64
        )
        with caplog.at_level(logging.WARNING):
            layer.build((1, 10, 16))
        assert any(
            "exceeds the declared max_seq_len" in r.getMessage()
            for r in caplog.records
        ), (
            "building at seq_len=10 with max_seq_len=4 produced no advisory "
            f"warning; got {[r.getMessage()[:60] for r in caplog.records]}"
        )

    def test_chunked_path_is_correct_past_max_seq_len(self):
        """`call()` at seq_len > max_seq_len must now WORK, not raise.

        The complement of the test above, and the actual F-12 fix. `max_seq_len`
        was a spurious limit on the path that runs: measured at seq_len=32 with
        max_seq_len=8, `_chunked_scan` matched a NumPy oracle to 2.2e-15 while
        `_sequential_scan` returned zeros past index 8 (error 6.13). Restoring
        the `build()`/`call()` raise fails HERE.

        Asserting a live suffix is what makes this more than a smoke test: the
        original defect produced exactly this shape with a zeroed tail.
        """
        layer = GatedLinearAttentionBlock(dim=16, num_heads=2, max_seq_len=4)
        over_long = tf.random.normal(shape=(1, 10, 16))

        layer.build((1, 10, 16))          # must not raise either
        output = layer(over_long, training=False)

        assert output.shape == over_long.shape
        out_np = ops.convert_to_numpy(output)
        assert np.all(np.isfinite(out_np)), "output must be finite"
        per_step_absmax = np.max(np.abs(out_np), axis=(0, 2))
        assert np.all(per_step_absmax > 0.0), (
            f"timesteps past max_seq_len=4 came back dead, i.e. the chunked "
            f"path is truncating after all: {per_step_absmax}"
        )

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

    # --- Per-head Q/K normalization (plan step 4, decision D-003) ---

    @staticmethod
    def _spy_on(norm_layer, sink) -> None:
        """Record every ``(input, output)`` pair a built norm sublayer sees.

        The sublayer is observed through its OWN call, so the test consumes the
        bits the shipped ``call()`` actually feeds it and never re-derives what
        the shape "should" be. That distinction matters here: the rank of that
        tensor is precisely what is being measured, so a test that constructed
        the norm's input itself would only be verifying its own assumption.

        :param norm_layer: A built normalization sublayer to instrument.
        :param sink: List that receives ``(input_np, output_np)`` per call.
        """
        original = norm_layer.call

        def spy(inputs, *args, **kwargs):
            outputs = original(inputs, *args, **kwargs)
            sink.append(
                (ops.convert_to_numpy(inputs), ops.convert_to_numpy(outputs))
            )
            return outputs

        norm_layer.call = spy

    @pytest.mark.parametrize("which", ["q", "k"])
    def test_qk_norm_statistic_is_per_head(self, which):
        """Scaling head 0's projected Q/K must not move any other head's norm output.

        This is the discriminating test for D-003. Before the fix, ``q_norm`` /
        ``k_norm`` saw ``(batch, seq, num_heads * head_dim)`` and their RMS
        denominator mixed every head, so scaling one head's slice moved all of
        them (measured on the pre-fix code: max|diff| ~ 1e0 on the other heads).

        Non-vacuity guards, so this cannot pass for the wrong reason:
          (a) the perturbation is asserted to have REACHED the norm's input, on
              head 0 and on no other head;
          (b) head 0's normalized output is asserted to have CHANGED -- a layer
              emitting a constant would otherwise pass the main assertion.

        The tolerance is exactly ``0.0``: the heads are independent slices of the
        same elementwise op, so a correct per-head statistic leaves the other
        heads' bits untouched, not merely close.
        """
        num_heads, head_dim = 4, 8
        layer = GatedLinearAttentionBlock(
            dim=32,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=16,
            dropout_rate=0.0,
        )
        x = tf.random.stateless_normal((2, 6, 32), seed=(7, 11))
        layer(x, training=False)  # build

        captured: list = []
        self._spy_on(getattr(layer, f"{which}_norm"), captured)

        layer(x, training=False)

        # Scale ONLY head 0's output columns of the projection. Head h owns
        # kernel columns [h*head_dim, (h+1)*head_dim) because `call()` reshapes
        # qk_dim -> (num_heads, head_dim) in that (heads-major) order.
        projection = getattr(layer, f"{which}_proj")
        kernel = ops.convert_to_numpy(projection.kernel)
        perturbed_kernel = kernel.copy()
        perturbed_kernel[:, 0:head_dim] *= 5.0
        projection.kernel.assign(perturbed_kernel)

        layer(x, training=False)

        assert len(captured) == 2, f"expected 2 recorded norm calls, got {len(captured)}"
        (in_base, out_base), (in_pert, out_pert) = captured

        def as_heads(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(arr.shape[0], arr.shape[1], num_heads, head_dim)

        in_base, out_base = as_heads(in_base), as_heads(out_base)
        in_pert, out_pert = as_heads(in_pert), as_heads(out_pert)

        # (a) the perturbation reached the norm's input, on head 0 only
        assert not np.array_equal(in_base[:, :, 0, :], in_pert[:, :, 0, :]), (
            f"setup failed: scaling {which}_proj's head-0 columns did not change "
            f"{which}_norm's input"
        )
        assert np.array_equal(in_base[:, :, 1:, :], in_pert[:, :, 1:, :]), (
            f"setup failed: the perturbation leaked into other heads' "
            f"{which}_norm INPUT"
        )
        # (b) head 0's normalized output moved -- rules out a constant output
        assert not np.array_equal(out_base[:, :, 0, :], out_pert[:, :, 0, :]), (
            f"vacuous: head 0's normalized {which.upper()} did not change at all"
        )
        # THE ASSERTION UNDER TEST
        other_delta = float(
            np.abs(out_base[:, :, 1:, :] - out_pert[:, :, 1:, :]).max()
        )
        assert other_delta == 0.0, (
            f"{which}_norm's statistic is NOT per-head: scaling head 0 moved "
            f"heads 1..{num_heads - 1} by max|diff|={other_delta:.6e} "
            f"(expected exactly 0.0)"
        )

    def test_qk_norm_scale_weight_is_head_dim_and_v_norm_is_whole_tensor(self):
        """Pin the intentional Q/K-vs-V asymmetry as a shape fact, not a comment.

        Q/K normalize per head, so their scale weight is ``(head_dim,)`` shared
        across heads (the standard QK-Norm convention). V is deliberately left
        whole-tensor over ``v_dim = 2 * num_heads * head_dim``. If someone later
        "unifies" the three, this test fails and points at D-003.
        """
        num_heads, head_dim = 4, 8
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=num_heads, head_dim=head_dim, max_seq_len=16
        )
        layer(tf.random.stateless_normal((2, 6, 32), seed=(3, 5)), training=False)

        for norm in (layer.q_norm, layer.k_norm):
            shapes = [tuple(w.shape) for w in norm.weights]
            assert shapes == [(head_dim,)], (
                f"{norm.name} scale weight should be (head_dim,)={(head_dim,)}, "
                f"got {shapes}"
            )
        v_shapes = [tuple(w.shape) for w in layer.v_norm.weights]
        assert v_shapes == [(2 * num_heads * head_dim,)], (
            f"v_norm is deliberately whole-tensor; expected "
            f"[{(2 * num_heads * head_dim,)}], got {v_shapes}"
        )

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

        # `len(gradients) > 0` plus not-None is satisfied by an ALL-ZERO
        # gradient, i.e. by a layer whose parameters cannot learn. Assert
        # magnitude, and per-variable rather than in aggregate, so one dead
        # sub-layer cannot hide behind a healthy one.
        dead = [
            v.name for v, g in zip(layer.trainable_variables, gradients)
            if float(ops.max(ops.abs(g))) == 0.0
        ]
        assert not dead, (
            f"{len(dead)} of {len(gradients)} trainable variables received an "
            f"exactly-zero gradient, so they cannot learn: {dead}"
        )
        total = float(ops.max(ops.abs(ops.concatenate(
            [ops.reshape(g, (-1,)) for g in gradients], axis=0
        ))))
        assert np.isfinite(total) and total > 0.0, (
            f"gradient magnitude is {total} — no signal reaches the parameters"
        )

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


# =========================================================================
# 8. Shared helpers for the coverage-gap groups below
# =========================================================================


@contextlib.contextmanager
def global_dtype_policy(name: str):
    """Set the PROCESS-GLOBAL Keras dtype policy, then always restore it.

    The ``dtype_policy`` fixture in ``tests/test_layers/conftest.py`` is the
    house instrument for *parametrized* policy sweeps and is used as such
    below. This context manager exists for the tests that need ONE specific
    policy (the float64 oracle) rather than a sweep, and it keeps the same
    invariant that fixture documents: a leaked global policy poisons every
    later test in the session, so the restore lives in a ``finally``.

    :param name: Policy name, e.g. ``'float64'``.
    :type name: str
    :yield: The policy name now in force.
    """
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(name)
    try:
        yield name
    finally:
        keras.mixed_precision.set_global_policy(previous)


def capture_scan_io(layer: GatedLinearAttentionBlock, x, training: bool = False):
    """Capture the tensors the scan ACTUALLY receives, plus what it returns.

    This wraps ``gated_linear_scan`` itself, so the recorded ``q, k, v,
    alpha, beta`` are the layer's own post-projection / post-norm /
    post-conv / post-activation / post-reshape bits at the exact call
    boundary -- not values re-derived by the test from what it *thinks*
    the pipeline should produce. Re-deriving them would verify the test's
    assumptions instead of the layer (a failure mode this repo has already
    paid for once).

    :param layer: A ``GatedLinearAttentionBlock`` (built or unbuilt).
    :type layer: GatedLinearAttentionBlock
    :param x: Input tensor of shape ``(batch, seq, dim)``.
    :param training: Training flag forwarded to ``call``.
    :type training: bool
    :return: Dict with numpy ``q``, ``k``, ``v``, ``alpha``, ``beta``,
        ``scan_out`` (the scan's return value) and ``block_out``.
    :rtype: Dict[str, np.ndarray]
    """
    captured: Dict[str, Any] = {}
    original = layer.gated_linear_scan

    def spy(q, k, v, alpha, beta, training=None):
        out = original(q, k, v, alpha, beta, training=training)
        captured.update(
            q=ops.convert_to_numpy(q),
            k=ops.convert_to_numpy(k),
            v=ops.convert_to_numpy(v),
            alpha=ops.convert_to_numpy(alpha),
            beta=ops.convert_to_numpy(beta),
            scan_out=ops.convert_to_numpy(out),
        )
        return out

    layer.gated_linear_scan = spy
    try:
        captured["block_out"] = ops.convert_to_numpy(layer(x, training=training))
    finally:
        del layer.gated_linear_scan
    assert "scan_out" in captured, "gated_linear_scan was never called"
    return captured


def numpy_gated_linear_recurrence(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Independent float64 NumPy reference, written FROM THE DEFINITION.

    Transcribed from the two recurrence lines in the module docstring, not
    from ``gated_linear_scan``'s body -- an oracle copied from the code
    under test shares that code's bugs and proves nothing::

        S_t   = alpha_t * S_{t-1} + beta_t * (k_t v_t^(1)T)
        out_t = q_t^T S_t + v_t^(2)

    ``S`` starts at zero; the read-out uses ``S_t``, i.e. the state AFTER
    step ``t``'s write (inclusive in ``j = t``, verified independently by an
    ``alpha=0, beta=1`` probe during plan step 5). ``v`` is split in half
    along its LAST axis, i.e. within each head -- head ``h``'s write half is
    that head's own first ``head_dim`` channels, NOT the leading
    ``num_heads * head_dim`` block of the flat ``v_dim`` axis.

    :param q: Queries, shape ``(batch, seq, heads, head_dim)``.
    :param k: Keys, same shape as ``q``.
    :param v: Values, shape ``(batch, seq, heads, 2 * head_dim)``.
    :param alpha: Persistence gate, shape ``(batch, seq, heads)``.
    :param beta: Write gate, shape ``(batch, seq, heads)``.
    :return: Read-out of shape ``(batch, seq, heads, head_dim)``, float64.
    :rtype: np.ndarray
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    batch, seq, heads, head_dim = q.shape
    assert v.shape == (batch, seq, heads, 2 * head_dim), f"bad v shape {v.shape}"

    state = np.zeros((batch, heads, head_dim, head_dim), dtype=np.float64)
    out = np.zeros((batch, seq, heads, head_dim), dtype=np.float64)
    for t in range(seq):
        v_write = v[:, t, :, :head_dim]
        v_skip = v[:, t, :, head_dim:]
        outer = k[:, t][:, :, :, None] * v_write[:, :, None, :]
        a_t = alpha[:, t][:, :, None, None]
        b_t = beta[:, t][:, :, None, None]
        state = a_t * state + b_t * outer
        out[:, t] = np.einsum("bhi,bhij->bhj", q[:, t], state) + v_skip
    return out


def oracle_tolerance(seq_len: int, head_dim: int, expected: np.ndarray) -> float:
    """Roundoff bound for the float64 oracle-vs-layer comparison.

    Derivation (so this is a bound, not a pasted magic number). Per timestep
    and head the recurrence performs, per state entry, one multiply by
    ``alpha``, one multiply by ``beta`` and one add -- ~3 rounded float64
    operations. A state entry surviving ``T`` timesteps therefore carries a
    relative error of order ``3 * T * u`` with ``u = eps / 2`` the unit
    roundoff. The read-out adds a length-``head_dim`` dot product, worth a
    further ``head_dim * u``. The layer and the oracle also associate their
    sums differently (``ops.matmul`` on GPU vs ``np.einsum``), so charge that
    bound twice::

        rel_err <= 2 * (3 * T + head_dim) * u  =  (3 * T + head_dim) * eps

    The error is relative, so scale by the output magnitude, and take a 16x
    safety factor for reduction reassociation inside the backend's matmul::

        tol = 16 * (3 * T + head_dim) * eps * max(1, max|expected|)

    Calibration at ``T=12, head_dim=8``: this yields ~1.6e-13 while the
    MEASURED error is 2.8e-17 (0.31 eps-units of output scale) -- roughly
    four orders of magnitude inside the bound. It is not slack that hides a
    bug: perturbing ONE ``beta`` entry by 1% moves the comparison to 4.6e-05,
    eight orders of magnitude OUTSIDE it (pinned by
    ``test_oracle_rejects_a_one_percent_beta_perturbation``).

    :param seq_len: Number of timesteps the state accumulated over.
    :type seq_len: int
    :param head_dim: Per-head dimension (the dot-product length).
    :type head_dim: int
    :param expected: The oracle's output, used for its magnitude.
    :type expected: np.ndarray
    :return: Absolute tolerance.
    :rtype: float
    """
    eps = float(np.finfo(np.float64).eps)
    scale = max(1.0, float(np.abs(expected).max()))
    return 16.0 * (3 * seq_len + head_dim) * eps * scale


# Deliberately includes shapes where num_heads != head_dim in BOTH directions,
# so an axis/transpose confusion cannot hide behind a symmetric shape.
ORACLE_SHAPES = [
    # (dim, num_heads, head_dim, seq_len)
    (8, 1, 8, 7),      # single head, prime-length sequence
    (16, 3, 8, 12),    # multi-head, num_heads < head_dim
    (24, 4, 6, 17),    # multi-head, num_heads < head_dim, non-round dims
    (12, 8, 3, 11),    # multi-head, num_heads > head_dim
]


# =========================================================================
# 9. Numerical-correctness oracle (plan success criterion 6)
# =========================================================================
class TestRecurrenceOracle:
    """The scan must reproduce the recurrence it documents, not merely run."""

    @pytest.mark.parametrize("dim, num_heads, head_dim, seq_len", ORACLE_SHAPES)
    def test_scan_matches_numpy_oracle_float64(self, dim, num_heads, head_dim, seq_len):
        """``gated_linear_scan`` == an independent NumPy recurrence, at float64.

        The oracle is fed the layer's OWN pre-scan tensors (see
        ``capture_scan_io``), so this compares the executed scan against the
        documented math on identical bits. The whole layer runs under a
        float64 global policy, so the captured tensors and the scan output are
        genuinely float64 -- comparing an fp32 scan against a float64 oracle
        would measure the policy, not the algorithm.
        """
        with global_dtype_policy("float64"):
            keras.utils.set_random_seed(1234)
            layer = GatedLinearAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                max_seq_len=64,
                dropout_rate=0.0,
            )
            x = np.asarray(
                np.random.default_rng(0).normal(size=(2, seq_len, dim)),
                dtype="float64",
            )
            cap = capture_scan_io(layer, x, training=False)

        assert cap["scan_out"].dtype == np.float64, (
            f"the scan did not run at float64 (got {cap['scan_out'].dtype}); "
            "the comparison below would be measuring the dtype policy"
        )
        assert cap["q"].shape == (2, seq_len, num_heads, head_dim)
        assert cap["v"].shape == (2, seq_len, num_heads, 2 * head_dim)

        expected = numpy_gated_linear_recurrence(
            cap["q"], cap["k"], cap["v"], cap["alpha"], cap["beta"]
        )
        # Non-vacuity: a scan returning zeros (or an oracle that never ran)
        # must not be able to pass.
        assert float(np.abs(expected).max()) > 1e-6, "oracle output is ~zero"
        assert np.isfinite(expected).all()

        tol = oracle_tolerance(seq_len, head_dim, expected)
        err = float(np.abs(expected - cap["scan_out"]).max())
        assert err <= tol, (
            f"scan disagrees with the NumPy recurrence: max|diff|={err:.3e} "
            f"> tol={tol:.3e} (seq_len={seq_len}, head_dim={head_dim}, "
            f"output scale={np.abs(expected).max():.3e})"
        )

    def test_oracle_rejects_a_one_percent_beta_perturbation(self):
        """The oracle comparison must be ABLE to fail -- pinned, not asserted once.

        Same setup as the test above, but ONE element of the captured ``beta``
        is scaled by 1.01 before the oracle runs. If the comparison stayed
        within tolerance here, the tolerance (or the oracle) would be
        insensitive to the recurrence's gating and the green result above
        would mean nothing.
        """
        dim, num_heads, head_dim, seq_len = 16, 3, 8, 12
        with global_dtype_policy("float64"):
            keras.utils.set_random_seed(1234)
            layer = GatedLinearAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                max_seq_len=64,
                dropout_rate=0.0,
            )
            x = np.asarray(
                np.random.default_rng(0).normal(size=(2, seq_len, dim)),
                dtype="float64",
            )
            cap = capture_scan_io(layer, x, training=False)

        clean = numpy_gated_linear_recurrence(
            cap["q"], cap["k"], cap["v"], cap["alpha"], cap["beta"]
        )
        tol = oracle_tolerance(seq_len, head_dim, clean)
        assert float(np.abs(clean - cap["scan_out"]).max()) <= tol  # control

        perturbed_beta = cap["beta"].copy()
        perturbed_beta[0, seq_len // 2, 1] *= 1.01
        wrong = numpy_gated_linear_recurrence(
            cap["q"], cap["k"], cap["v"], cap["alpha"], perturbed_beta
        )
        err = float(np.abs(wrong - cap["scan_out"]).max())
        assert err > tol, (
            f"a 1% perturbation of one beta entry stayed WITHIN tolerance "
            f"(max|diff|={err:.3e} <= tol={tol:.3e}): the oracle comparison "
            f"cannot detect a wrong gate and proves nothing"
        )

    def test_oracle_rejects_an_exclusive_readout(self):
        """A read-out using ``S_{t-1}`` instead of ``S_t`` must be caught.

        Inclusivity in ``j = t`` is the single easiest thing to get wrong when
        the recurrence is re-derived (and plan step 7's chunked scan depends on
        it). This pins that the oracle comparison discriminates the two
        conventions rather than being blind to a one-step shift.
        """
        dim, num_heads, head_dim, seq_len = 16, 3, 8, 12
        with global_dtype_policy("float64"):
            keras.utils.set_random_seed(1234)
            layer = GatedLinearAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                max_seq_len=64,
                dropout_rate=0.0,
            )
            x = np.asarray(
                np.random.default_rng(0).normal(size=(2, seq_len, dim)),
                dtype="float64",
            )
            cap = capture_scan_io(layer, x, training=False)

        q, k, v = cap["q"], cap["k"], cap["v"]
        alpha, beta = cap["alpha"], cap["beta"]
        batch, seq, heads, hd = q.shape
        state = np.zeros((batch, heads, hd, hd))
        exclusive = np.zeros((batch, seq, heads, hd))
        for t in range(seq):
            # READ FIRST (state before this step's write) -- the wrong convention.
            exclusive[:, t] = (
                np.einsum("bhi,bhij->bhj", q[:, t], state) + v[:, t, :, hd:]
            )
            outer = k[:, t][:, :, :, None] * v[:, t, :, :hd][:, :, None, :]
            state = (
                alpha[:, t][:, :, None, None] * state
                + beta[:, t][:, :, None, None] * outer
            )

        tol = oracle_tolerance(seq_len, head_dim, exclusive)
        err = float(np.abs(exclusive - cap["scan_out"]).max())
        assert err > tol, (
            f"an EXCLUSIVE read-out matched the shipped scan within tolerance "
            f"(max|diff|={err:.3e} <= tol={tol:.3e}); the comparison cannot "
            f"tell S_t from S_{{t-1}}"
        )


# =========================================================================
# 10. Non-default normalization arguments
# =========================================================================
class TestNormalizationArgs:
    """Both branches of the ``q/k/v_norm_args`` default and the explicit path."""

    def test_empty_default_branch_for_non_zero_centered_norm(self):
        """A non-``zero_centered_rms_norm`` type takes the ``{}`` else-branch.

        The constructor's default is ``{'epsilon': 1e-5, 'use_scale': True}``
        ONLY for ``zero_centered_rms_norm``; every other type gets ``{}``. The
        observable consequence is that ``create_normalization_layer``'s own
        default ``epsilon=1e-6`` reaches the sublayer -- which is also a
        discriminator against Keras's ``LayerNormalization`` default of 1e-3,
        so this cannot pass by accident if the args were dropped entirely.
        """
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, max_seq_len=16, normalization_type="layer_norm"
        )
        assert layer.q_norm_args == {}
        assert layer.k_norm_args == {}
        assert layer.v_norm_args == {}
        for norm in (layer.q_norm, layer.k_norm, layer.v_norm):
            assert norm.get_config()["epsilon"] == pytest.approx(1e-6), (
                f"{norm.name} did not receive the factory default epsilon; "
                f"got {norm.get_config()['epsilon']}"
            )

    def test_explicit_norm_args_reach_the_built_sublayers(self):
        """Explicit per-tensor args must arrive at the constructed norm layers.

        Three DIFFERENT epsilons are used on purpose: identical values could
        not distinguish "each dict reached its own norm" from "one dict was
        broadcast to all three". The extra ``center=False`` on Q proves a
        non-shared, type-specific kwarg also survives the factory.
        """
        layer = GatedLinearAttentionBlock(
            dim=32,
            num_heads=4,
            max_seq_len=16,
            normalization_type="layer_norm",
            q_norm_args={"epsilon": 0.25, "center": False},
            k_norm_args={"epsilon": 0.125},
            v_norm_args={"epsilon": 0.0625},
        )
        layer(tf.random.stateless_normal((2, 8, 32), seed=(1, 2)), training=False)

        assert layer.q_norm.get_config()["epsilon"] == pytest.approx(0.25)
        assert layer.q_norm.get_config()["center"] is False
        assert layer.k_norm.get_config()["epsilon"] == pytest.approx(0.125)
        assert layer.k_norm.get_config()["center"] is True, (
            "q_norm_args leaked into k_norm"
        )
        assert layer.v_norm.get_config()["epsilon"] == pytest.approx(0.0625)

    def test_explicit_norm_args_survive_a_config_round_trip(self):
        """``get_config``/``from_config`` must preserve the explicit dicts."""
        original = GatedLinearAttentionBlock(
            dim=32,
            num_heads=4,
            max_seq_len=16,
            normalization_type="layer_norm",
            q_norm_args={"epsilon": 0.25},
            k_norm_args={"epsilon": 0.125},
            v_norm_args={"epsilon": 0.0625},
        )
        restored = GatedLinearAttentionBlock.from_config(original.get_config())
        assert restored.q_norm_args == {"epsilon": 0.25}
        assert restored.k_norm_args == {"epsilon": 0.125}
        assert restored.v_norm_args == {"epsilon": 0.0625}
        assert restored.q_norm.get_config()["epsilon"] == pytest.approx(0.25)

    def test_invalid_norm_args_fail_loudly(self):
        """An unknown norm kwarg must raise the layer's wrapping ValueError."""
        with pytest.raises(ValueError, match="Failed to create .* norm layer"):
            GatedLinearAttentionBlock(
                dim=32,
                num_heads=4,
                max_seq_len=16,
                normalization_type="layer_norm",
                q_norm_args={"definitely_not_a_layer_norm_argument": 1},
            )


# =========================================================================
# 11. ffn_args override
# =========================================================================
class TestFFNArgs:
    """``ffn_args`` is applied last and must reach the constructed FFN."""

    def test_ffn_args_override_reaches_the_built_ffn(self):
        """``ffn_args`` wins over this layer's own derived defaults.

        ``intermediate_size=320`` becomes the ``hidden_dim`` this layer
        derives; ``ffn_args={'hidden_dim': 192}`` must override it. 192 is
        neither 320 nor SwiGLU's own 2/3-rule default of 256 for
        ``output_dim=64``, so all three cases are distinguishable -- the
        precise trap that once let a SILENTLY DROPPED ``hidden_dim`` ship
        green (see ``_create_ffn_layer``'s comment trail).

        ``use_bias`` is overridden in the opposite direction from the block's
        own value, so this also proves the override is not merely echoing the
        layer's defaults.
        """
        layer = GatedLinearAttentionBlock(
            dim=64,
            num_heads=4,
            max_seq_len=32,
            ffn_type="swiglu",
            intermediate_size=320,
            use_bias=False,
            ffn_args={"hidden_dim": 192, "use_bias": True},
        )
        ffn_config = layer.output_ffn.get_config()
        assert ffn_config["hidden_dim"] == 192, (
            f"ffn_args['hidden_dim'] did not reach the FFN "
            f"(got {ffn_config['hidden_dim']}; 320 = intermediate_size wins, "
            f"256 = SwiGLU's own default)"
        )
        assert ffn_config["use_bias"] is True, (
            "ffn_args['use_bias'] did not override the block's use_bias=False"
        )
        assert layer.use_bias is False, "ffn_args must not mutate the block itself"

    def test_ffn_args_survive_a_config_round_trip_and_rebuild_the_same_ffn(self):
        """The override must be serialized, not just applied at construction."""
        original = GatedLinearAttentionBlock(
            dim=64,
            num_heads=4,
            max_seq_len=32,
            ffn_type="swiglu",
            intermediate_size=320,
            ffn_args={"hidden_dim": 192},
        )
        restored = GatedLinearAttentionBlock.from_config(original.get_config())
        assert restored.ffn_args == {"hidden_dim": 192}
        assert restored.output_ffn.get_config()["hidden_dim"] == 192

    def test_unknown_ffn_args_key_is_SILENTLY_DROPPED_not_rejected(self):
        """MEASURED: an unrecognized ``ffn_args`` key is discarded without error.

        **This pins behaviour that contradicts the shipped documentation.** The
        class docstring's ``:param ffn_args:`` says the dict is "passed through
        unfiltered -- the factory rejects an unknown key loudly", and
        ``_create_ffn_layer``'s comment repeats it ("unlike ``ffn_args``, which
        the factory now rejects loudly if unknown"). Neither is true:
        ``ffn/factory.py``'s ``create_ffn_layer`` ends with

            final_params = {k: v for k, v in params.items()
                            if k in valid_param_names}

        which filters the CALLER's keys exactly as it filters this block's own
        generic defaults. A typo in ``ffn_args`` is therefore silent.

        This test records the fact rather than asserting the claim, so the suite
        stays honest; the docstring/comment repair is reported to the plan, not
        made here (plan step 6 is test-only). If someone later makes the factory
        strict, THIS test goes red and points at the two prose sites to update.
        """
        layer = GatedLinearAttentionBlock(
            dim=64,
            num_heads=4,
            max_seq_len=32,
            ffn_type="swiglu",
            intermediate_size=320,
            ffn_args={"definitely_not_a_swiglu_argument": 1},
        )
        assert "definitely_not_a_swiglu_argument" not in layer.output_ffn.get_config()
        # the surrounding, valid configuration still arrived
        assert layer.output_ffn.get_config()["hidden_dim"] == 320


# =========================================================================
# 12. use_bias
# =========================================================================
# The seven Dense sublayers this layer owns under the DEFAULT (built-in gated)
# output stage. The three Conv1D layers are deliberately NOT listed: they are
# constructed without a `use_bias` argument, so they always carry a bias
# regardless of this flag -- counting "every bias in the layer" would blur that.
_DENSE_SUBLAYER_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "alpha_proj",
    "beta_proj",
    "output_proj",
    "output_gate_linear",
)


class TestUseBias:
    """``use_bias`` is asserted by naming variables, not by "it constructed"."""

    @staticmethod
    def _build(use_bias: bool) -> GatedLinearAttentionBlock:
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, max_seq_len=16, use_bias=use_bias
        )
        layer(tf.random.stateless_normal((2, 8, 32), seed=(4, 9)), training=False)
        return layer

    def test_default_is_biasless_on_every_dense_sublayer(self):
        layer = self._build(use_bias=False)
        assert layer.use_bias is False
        for name in _DENSE_SUBLAYER_NAMES:
            sub = getattr(layer, name)
            assert sub.bias is None, f"{name} has a bias under use_bias=False"

    def test_use_bias_true_adds_exactly_one_bias_per_dense_sublayer(self):
        layer = self._build(use_bias=True)
        assert layer.use_bias is True
        for name in _DENSE_SUBLAYER_NAMES:
            sub = getattr(layer, name)
            assert sub.bias is not None, f"{name} has no bias under use_bias=True"
            assert tuple(sub.bias.shape) == (sub.units,), (
                f"{name} bias shape {tuple(sub.bias.shape)} != ({sub.units},)"
            )

    def test_bias_variable_count_delta_is_exactly_the_dense_sublayer_count(self):
        """A count, not a spot check: nothing else may gain or lose a bias."""
        biasless = self._build(use_bias=False)
        biased = self._build(use_bias=True)

        def bias_paths(layer):
            # Drop the auto-numbered top-level layer name ('..._block_1/') so
            # two instances built in the same session are comparable; everything
            # that identifies the OWNING sublayer is kept.
            return sorted(
                w.path.split("/", 1)[1]
                for w in layer.weights
                if w.path.endswith("bias")
            )

        without, with_ = bias_paths(biasless), bias_paths(biased)
        assert len(with_) - len(without) == len(_DENSE_SUBLAYER_NAMES), (
            f"expected exactly {len(_DENSE_SUBLAYER_NAMES)} new bias variables, "
            f"got {len(with_) - len(without)}\nwithout={without}\nwith={with_}"
        )
        # The biasless set must be a strict subset (the convs' biases), so the
        # flag only ADDS variables and never renames or removes one.
        assert set(without) < set(with_)


# =========================================================================
# 13. Mixed precision (uses the house dtype_policy fixture)
# =========================================================================
_POLICY_TO_OUTPUT_DTYPE = {
    "float32": "float32",
    "mixed_float16": "float16",
    "float64": "float64",
}


class TestMixedPrecision:
    """float32 / mixed_float16 / float64, short AND long sequences."""

    def test_forward_pass_under_each_policy(self, dtype_policy):
        """Shape, dtype and finiteness under each global policy.

        The sublayers are constructed inside ``__init__``, so the layer MUST
        be created while the fixture's policy is in force -- constructing it
        earlier would silently pin every sublayer to the previous policy and
        make this test a float32 test wearing three names.
        """
        keras.utils.set_random_seed(11)
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, max_seq_len=64, dropout_rate=0.0
        )
        x = np.asarray(
            np.random.default_rng(3).normal(size=(2, 16, 32)), dtype="float32"
        )
        y = ops.convert_to_numpy(layer(x, training=False))
        assert y.shape == (2, 16, 32)
        assert y.dtype.name == _POLICY_TO_OUTPUT_DTYPE[dtype_policy], (
            f"policy {dtype_policy} produced {y.dtype}"
        )
        assert np.isfinite(y.astype("float64")).all(), (
            f"non-finite output under {dtype_policy}"
        )

    def test_long_sequence_accumulator_stays_finite(self, dtype_policy):
        """512 timesteps with the state gate driven to alpha = 1 (no decay).

        Why not just a short sequence: this repo has already been bitten by a
        toy ``N=7`` reduction that hid an fp16 failure appearing only at
        ``N >= 512``. And why the gate surgery: at random init ``alpha`` is
        ~sigmoid(small) ~ 0.5, so the state's memory horizon is ~1 step and a
        512-step run would exercise no more accumulation than a 2-step one.
        Setting ``alpha_proj``/``beta_proj`` to a large constant bias drives
        ``alpha, beta -> 1`` (MEASURED: alpha_min = 0.999994 at float32 and
        exactly 1.0 at fp16), so the state genuinely accumulates 512
        unrenormalized rank-1 writes -- the case the scan is unguarded for.

        MEASURED RESULT (2026-07-29, plan step 6): finite at all three
        policies; scan output max|.| = 3.363 (float32) / 3.367 (fp16). No fp16
        overflow was found, so nothing is xfail-pinned here. This test is the
        instrument that would catch it if the accumulator ever grows.
        """
        seq_len = 512
        keras.utils.set_random_seed(7)
        layer = GatedLinearAttentionBlock(
            dim=32,
            num_heads=4,
            max_seq_len=seq_len,
            dropout_rate=0.0,
            use_bias=True,  # alpha_proj/beta_proj need a bias to be driven
        )
        x = np.asarray(
            np.random.default_rng(1).normal(size=(2, seq_len, 32)), dtype="float32"
        )
        layer(x[:, :4], training=False)  # build

        for proj in (layer.alpha_proj, layer.beta_proj):
            proj.kernel.assign(np.zeros(proj.kernel.shape, dtype="float32"))
            proj.bias.assign(np.full(proj.bias.shape, 12.0, dtype="float32"))

        cap = capture_scan_io(layer, x, training=False)
        alpha = cap["alpha"].astype("float64")
        assert alpha.min() > 0.999, (
            f"setup failed: alpha was not driven to ~1 (min={alpha.min()}), so "
            f"the state still decays and this is not a long-accumulation test"
        )
        scan_out = cap["scan_out"].astype("float64")
        assert np.isfinite(scan_out).all(), (
            f"the {seq_len}-step state accumulator produced "
            f"{int((~np.isfinite(scan_out)).sum())} non-finite scan outputs "
            f"under {dtype_policy}"
        )
        assert np.isfinite(cap["block_out"].astype("float64")).all(), (
            f"non-finite block output at seq_len={seq_len} under {dtype_policy}"
        )


# =========================================================================
# 14. XLA
# =========================================================================
class TestXLACompilation:
    """``ops.while_loop`` + ``ops.scatter_update`` under ``jit_compile=True``.

    MEASURED (2026-07-29, plan step 6): both compile and run on this backend,
    so no xfail is needed. Recorded as decision D-010.
    """

    def test_jit_compiled_forward_matches_eager(self):
        keras.utils.set_random_seed(5)
        inputs = keras.Input(shape=(16, 32))
        block = GatedLinearAttentionBlock(dim=32, num_heads=4, max_seq_len=32)
        model = keras.models.Model(inputs, block(inputs))
        x = np.asarray(
            np.random.default_rng(2).normal(size=(4, 16, 32)), dtype="float32"
        )
        eager = ops.convert_to_numpy(model(x, training=False)).astype("float64")

        model.compile(loss="mse", optimizer="adam", jit_compile=True)
        jitted = model.predict(x, verbose=0).astype("float64")

        assert jitted.shape == eager.shape
        assert np.isfinite(jitted).all()

        # Tolerance, derived rather than guessed. Same weights, same input, so
        # only the arithmetic differs -- and on this GPU the difference is TF32,
        # not XLA: measured max|diff| = 6.6e-05 with TF32 enabled (the default)
        # and 2.2e-08 with `enable_tensor_float_32_execution(False)`, on an
        # output whose scale is 0.179. TF32 keeps 10 explicit mantissa bits, so
        # one TF32 ulp is 2**-11 relative; the measured gap is 0.76 ulp. Allow 4
        # ulps of output scale -- ~5x above the measured value, still ~4 orders
        # of magnitude below anything a miscompiled scan would produce.
        # (This matters because TF32 can be disabled process-globally by an
        # unrelated test module, which swings this number by ~3000x; the bound
        # must hold in BOTH regimes, and it does.)
        err = float(np.abs(jitted - eager).max())
        scale = float(np.abs(eager).max())
        tol = 4.0 * (2.0 ** -11) * scale
        assert err <= tol, (
            f"jit_compile=True changed the result beyond TF32 roundoff: "
            f"max|diff|={err:.3e} > tol={tol:.3e} (output scale {scale:.3e})"
        )

# ---------------------------------------------------------------------------
# Step 7: the chunked (blockwise) scan.
#
# `_chunked_scan` and `_sequential_scan` must compute the same function. They
# cannot agree bitwise -- they associate the same sums in a different order --
# so every comparison below is a tolerance, and each tolerance is derived or
# measured, never guessed.
#
# One TF32 ulp is 2**-11 = 4.88e-04 relative. That is a hard floor on how
# closely two differently-associated float32 matmul chains can agree on this
# GPU, and it sits ABOVE the 1e-4 float32 bound the plan pre-committed before
# TF32 was known to be in play. So the per-point float32 assertions below are
# expressed in TF32 ulps, exactly as the step-6 XLA test does, and hold in both
# TF32 regimes (an unrelated test module can disable TF32 process-wide).
#
# The plan's original 1e-4 is NOT abandoned: it is asserted directly, over the
# same grid, by `test_chunked_matches_sequential_float32_without_tf32` below,
# which turns TF32 off for the duration. Measured worst case there is 2.91e-06,
# i.e. ~34x tighter than the plan required, versus 7.73e-04 with TF32 left on --
# so the toggle is load-bearing when this file is run on its own or as part of
# this directory. The two tests are complementary: one pins the bound the plan
# committed to, the other pins what is achievable on the hardware's default
# matmul path.
#
# CAVEAT, measured: that "load-bearing" claim does NOT hold under a whole-suite
# run. `tests/test_layers/test_attention/test_linear_attention.py` disables TF32
# AT IMPORT for the entire pytest process, and it collects before this file, so
# under `make test` TF32 is already off and the toggle below is a no-op. TF32 is
# also GPU-only, so on a CPU-only run neither test is measuring a TF32 effect at
# all. Do not read a green result here as evidence that the toggle did anything;
# check the invocation scope first.
# ---------------------------------------------------------------------------

_TF32_ULP = 2.0 ** -11

# (seq_len, num_heads, head_dim). Non-multiples of chunk_size (7, 65, 257) and
# the exact 64 boundary are all deliberate: a powers-of-two-only grid can be
# fully green on a chunked scan that mishandles its own edges.
#
# MEASURED, and the reason 257 is load-bearing: dropping the inter-chunk carry
# decay (`state = chunk_decay * state + chunk_state` -> `state + chunk_state`)
# is detected by seq_len=257 ONLY. It is invisible at every shorter length here,
# and NOT because those cases are weak -- it is arithmetically invisible below
# THREE chunks. With `n_chunks == 1` there is no carry at all; with
# `n_chunks == 2` the only entry state that gets consumed is the one after chunk
# 0, and `chunk_decay * 0 + chunk_state[0]` equals `0 + chunk_state[0]`, so the
# two forms agree exactly. The paths first diverge at chunk 2, i.e.
# `seq_len > 2 * chunk_size`.
#
# So the grid needs at least one length with `ceil(seq_len/64) >= 3`, and 257
# (5 chunks) is it. `test_equiv_grid_can_detect_an_inter_chunk_carry_bug` below
# pins that requirement mechanically, so a future trim cannot silently drop the
# only cell that can fail.
#
# Trimmed from 7 lengths to 5: 63 was redundant with 7 (both single-chunk at
# chunk_size=64) and 128 was redundant with 65 (both exactly 2 chunks, both
# provably unable to detect a carry bug). 28 -> 20 combinations per test.
_EQUIV_GRID = [
    (seq_len, num_heads, head_dim)
    for seq_len in (1, 7, 64, 65, 257)
    for num_heads in (1, 4)
    for head_dim in (8, 32)
]

_EQUIV_CHUNK_SIZE = 64

# (label, alpha_value) -- `None` means `_scan_inputs`' default random draw.
#
# The near-1 regime is LOAD-BEARING, not a nicety. At the default
# `alpha ~ U(0.05, 0.95)` the per-chunk carry factor is ~1.8e-26, so the
# inter-chunk state contribution is numerically zero and every carry bug is
# invisible regardless of sequence length. At alpha=0.97 the factor is ~0.14, so
# the carry actually reaches the next chunk and can be checked.
# (label, alpha_value, alpha_range)
_ALPHA_REGIMES = [
    ("mixed", None, None),
    # RANDOM near-one, not constant. Two independent reasons this band is needed:
    #   * carry ALIVE: exp(64*log(~0.95)) is ~1e-1..1e-2, versus ~1.8e-26 at the
    #     default draw -- so the inter-chunk state actually reaches the next chunk
    #     and a DROPPED carry is visible;
    #   * carry VARYING per chunk -- so a MIS-INDEXED carry (`[:, :, 0]` for
    #     `[:, :, c]`) is visible too. A constant alpha makes those two
    #     expressions bit-identical and is blind to that bug by construction.
    ("near_one_random", None, (0.90, 0.99)),
]


def test_equiv_grid_covers_a_regime_where_the_inter_chunk_carry_MATTERS():
    """The grid must contain a case whose inter-chunk carry is numerically alive.

    A "no silent caps" guard on the grid itself, and the harder half of it is the
    GATE VALUE, not the sequence length.

    An earlier version of this test asserted only `max(chunk_counts) >= 3` and was
    named "...can_detect_an_inter_chunk_carry_bug". That name claimed a capability
    the grid did not have. `_scan_inputs` draws `alpha ~ U(0.05, 0.95)`, so over a
    64-wide chunk the carry factor `exp(cum_last) = exp(sum(log alpha))` is about
    **1.8e-26** -- the state entering chunk c+1 is numerically annihilated, and a
    carry bug (dropped, mis-indexed, or wrong sign) changes an output term whose
    true weight is ~1e-26. It is invisible at EVERY sequence length.

    Two requirements, therefore:
      * a length spanning >= 3 chunks -- below that the paths agree exactly even
        with the carry deleted, because the only consumed entry state is
        `chunk_decay * 0 + chunk_state[0]`; and
      * at least one gate regime whose per-chunk carry factor is materially
        non-zero, which is what `_ALPHA_REGIMES` provides.
    """
    chunk_counts = {
        -(-seq_len // _EQUIV_CHUNK_SIZE) for seq_len, _, _ in _EQUIV_GRID
    }
    assert max(chunk_counts) >= 3, (
        f"_EQUIV_GRID's longest case spans only {max(chunk_counts)} chunk(s) at "
        f"chunk_size={_EQUIV_CHUNK_SIZE}; the two forms agree exactly below 3 "
        f"chunks. Keep a seq_len > {2 * _EQUIV_CHUNK_SIZE}."
    )
    assert 1 in chunk_counts, "grid lost its single-chunk degenerate case"
    assert any(
        seq_len % _EQUIV_CHUNK_SIZE for seq_len, _, _ in _EQUIV_GRID
    ), "grid lost every length that is not a multiple of chunk_size"

    # The gate-value half. A regime is useful only if the carry survives a chunk.
    alive = []
    varying = []
    for label, alpha_value, alpha_range in _ALPHA_REGIMES:
        if alpha_range is not None:
            # geometric-mean gate over the band, as a representative
            typical = float(np.sqrt(alpha_range[0] * alpha_range[1]))
            varying.append(label)
        elif alpha_value is not None:
            typical = float(alpha_value)
        else:
            continue
        carry = float(np.exp(_EQUIV_CHUNK_SIZE * np.log(typical)))
        if carry > 1e-3:
            alive.append((label, carry))
    assert varying, (
        "no _ALPHA_REGIMES entry uses a RANDOM alpha_range, so every regime has a "
        "constant per-chunk decay. A mis-indexed carry (`chunk_decay[:, :, 0]` for "
        "`[:, :, c]`) is then bit-identical to the correct code and undetectable "
        "at any length. Keep a random near-one band."
    )
    assert alive, (
        f"no _ALPHA_REGIMES entry keeps the inter-chunk carry alive over "
        f"{_EQUIV_CHUNK_SIZE} steps. Without one, every equivalence cell is blind "
        f"to a carry bug: at the default alpha ~ U(0.05, 0.95) the carry factor is "
        f"~1.8e-26. Add a near-1 fixed alpha."
    )


def _scan_inputs(seq_len, num_heads, head_dim, dtype, alpha_value=None, seed=0,
                 alpha_range=None):
    """Random scan inputs.

    ``alpha_value`` pins the gate to a CONSTANT for adversarial runs.
    ``alpha_range`` draws it randomly from ``(lo, hi)`` instead.

    The distinction is load-bearing, not cosmetic. A constant alpha gives every
    chunk the SAME per-chunk decay `exp(cum_last)`, so a mis-INDEXED carry
    (`chunk_decay[:, :, 0]` in place of `[:, :, c]`) is arithmetically identical
    to the correct code and cannot be detected at any length or tolerance. Only a
    per-chunk-VARYING decay separates them -- hence the random near-one band.
    """
    rng = np.random.default_rng(seed)
    batch = 2
    q = rng.standard_normal((batch, seq_len, num_heads, head_dim)).astype(dtype)
    k = rng.standard_normal((batch, seq_len, num_heads, head_dim)).astype(dtype)
    v = rng.standard_normal((batch, seq_len, num_heads, 2 * head_dim)).astype(dtype)
    if alpha_range is not None:
        alpha = rng.uniform(*alpha_range, (batch, seq_len, num_heads)).astype(dtype)
    elif alpha_value is None:
        alpha = rng.uniform(0.05, 0.95, (batch, seq_len, num_heads)).astype(dtype)
    else:
        alpha = np.full((batch, seq_len, num_heads), alpha_value, dtype=dtype)
    beta = rng.uniform(0.1, 1.0, (batch, seq_len, num_heads)).astype(dtype)
    return [keras.ops.convert_to_tensor(x) for x in (q, k, v, alpha, beta)]

class TestChunkedScanEquivalence:
    """The chunked scan must agree with the sequential reference."""

    @pytest.mark.parametrize("alpha_label,alpha_value,alpha_range", _ALPHA_REGIMES)
    @pytest.mark.parametrize("seq_len,num_heads,head_dim", _EQUIV_GRID)
    def test_chunked_matches_sequential_float64(
        self, seq_len, num_heads, head_dim, alpha_label, alpha_value, alpha_range
    ):
        """float64: absolute agreement within the plan's pre-committed 1e-10.

        Swept over `_ALPHA_REGIMES` because the default random gate annihilates
        the inter-chunk carry (~1.8e-26 per chunk); the `near_one` regime is what
        makes this test able to see a carry bug at all.
        """
        with global_dtype_policy("float64"):
            layer = GatedLinearAttentionBlock(
                dim=num_heads * head_dim, num_heads=num_heads,
                head_dim=head_dim, max_seq_len=512, chunk_size=64,
            )
            args = _scan_inputs(seq_len, num_heads, head_dim, "float64",
                                alpha_value=alpha_value,
                                alpha_range=alpha_range)
            reference = keras.ops.convert_to_numpy(layer._sequential_scan(*args))
            chunked = keras.ops.convert_to_numpy(
                layer._chunked_scan(*args, seq_len)
            )

        assert np.isfinite(reference).all(), "sequential path produced non-finite values"
        assert np.isfinite(chunked).all(), "chunked path produced non-finite values"
        err = float(np.abs(reference - chunked).max())
        assert err <= 1e-10, (
            f"chunked scan disagrees with the sequential reference at "
            f"seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}: "
            f"max|diff|={err:.3e} > 1e-10"
        )

    @pytest.mark.parametrize("seq_len,num_heads,head_dim", _EQUIV_GRID)
    def test_chunked_matches_sequential_float32(self, seq_len, num_heads, head_dim):
        """float32: agreement within 4 TF32 ulps of output scale (see header)."""
        layer = GatedLinearAttentionBlock(
            dim=num_heads * head_dim, num_heads=num_heads,
            head_dim=head_dim, max_seq_len=512, chunk_size=64,
        )
        args = _scan_inputs(seq_len, num_heads, head_dim, "float32")
        reference = keras.ops.convert_to_numpy(layer._sequential_scan(*args))
        chunked = keras.ops.convert_to_numpy(layer._chunked_scan(*args, seq_len))

        assert np.isfinite(reference).all(), "sequential path produced non-finite values"
        assert np.isfinite(chunked).all(), "chunked path produced non-finite values"
        err = float(np.abs(reference - chunked).max())
        scale = float(np.abs(reference).max()) or 1.0
        tol = 4.0 * _TF32_ULP * scale
        assert err <= tol, (
            f"chunked scan disagrees with the sequential reference at "
            f"seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}: "
            f"max|diff|={err:.3e} > tol={tol:.3e} "
            f"({err / scale / _TF32_ULP:.2f} TF32 ulps of output scale)"
        )

    def test_chunked_matches_sequential_float32_without_tf32(self):
        """The plan's pre-committed float32 bound of 1e-4, honoured directly.

        The per-point float32 test above asserts 4 TF32 ulps because TF32 is the
        GPU's default matmul path and one ulp (4.88e-04) already exceeds 1e-4 --
        no correct implementation can meet the plan's number while TF32 is on.
        Turning TF32 off removes that floor and lets the original bound be
        asserted as written, which is the honest way to discharge the criterion
        rather than quietly relaxing it.

        Measured: 2.91e-06 with TF32 off, versus 7.73e-04 with it on. Removing
        the toggle makes this test fail, so the toggle is load-bearing.

        The whole grid runs inside ONE toggle. TF32 is process-global, and this
        repo has a standing lesson about a module that disabled it at import and
        swung unrelated measurements by ~3000x -- so the prior value is captured
        (not assumed to be True, since another module may already have disabled
        it), restored in a `finally`, and the restoration is asserted.
        """
        previous = tf.config.experimental.tensor_float_32_execution_enabled()
        worst = 0.0
        worst_case = None
        try:
            tf.config.experimental.enable_tensor_float_32_execution(False)
            for seq_len, num_heads, head_dim in _EQUIV_GRID:
                layer = GatedLinearAttentionBlock(
                    dim=num_heads * head_dim, num_heads=num_heads,
                    head_dim=head_dim, max_seq_len=512, chunk_size=64,
                )
                args = _scan_inputs(seq_len, num_heads, head_dim, "float32")
                reference = keras.ops.convert_to_numpy(layer._sequential_scan(*args))
                chunked = keras.ops.convert_to_numpy(
                    layer._chunked_scan(*args, seq_len)
                )
                assert np.isfinite(reference).all() and np.isfinite(chunked).all()
                rel = float(np.abs(reference - chunked).max()) / (
                    float(np.abs(reference).max()) or 1.0
                )
                if rel > worst:
                    worst, worst_case = rel, (seq_len, num_heads, head_dim)
        finally:
            tf.config.experimental.enable_tensor_float_32_execution(previous)

        # A leaked toggle would silently change every later precision assertion
        # in the session, which is exactly the failure this repo has recorded.
        assert (
            tf.config.experimental.tensor_float_32_execution_enabled() == previous
        ), "TF32 setting leaked out of this test"

        assert worst <= 1e-4, (
            f"chunked scan exceeds the plan's pre-committed float32 bound with "
            f"TF32 disabled: worst rel={worst:.3e} > 1e-4 at "
            f"(seq_len, num_heads, head_dim)={worst_case}"
        )

    def test_equivalence_gate_rejects_an_exclusive_intra_chunk_mask(self):
        """The gate must be ABLE to fail: feed it a deliberately wrong variant.

        The intra-chunk causal mask is ``tril`` INCLUSIVE -- the ``j = t`` term
        carries ``exp(0) = 1`` and is exactly the current step's own write,
        which the read-out sees because it reads the state AFTER the write.
        Dropping that diagonal is the single most plausible off-by-one in a
        chunked rewrite, so this pins that the gate catches it rather than
        waving it through as roundoff.
        """
        module = sys.modules[GatedLinearAttentionBlock.__module__]
        real_mask = module._inclusive_causal_mask

        def exclusive_mask(size, dtype):
            """Same mask with the diagonal dropped -- the off-by-one to catch."""
            idx = keras.ops.arange(size)
            return keras.ops.cast(
                keras.ops.greater(
                    keras.ops.expand_dims(idx, -1), keras.ops.expand_dims(idx, 0)
                ),
                dtype,
            )

        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, head_dim=8, max_seq_len=512, chunk_size=64
        )
        args = _scan_inputs(128, 4, 8, "float32", seed=5)
        reference = keras.ops.convert_to_numpy(layer._sequential_scan(*args))

        module._inclusive_causal_mask = exclusive_mask
        try:
            wrong = keras.ops.convert_to_numpy(layer._chunked_scan(*args, 128))
        finally:
            module._inclusive_causal_mask = real_mask

        assert module._inclusive_causal_mask is real_mask, "failed to restore the mask"

        err = float(np.abs(reference - wrong).max())
        scale = float(np.abs(reference).max()) or 1.0
        tol = 4.0 * _TF32_ULP * scale
        assert err > tol, (
            "the equivalence gate did NOT reject an exclusive intra-chunk mask "
            f"(max|diff|={err:.3e} <= tol={tol:.3e}); the gate is vacuous"
        )

    def test_chunked_matches_the_independent_numpy_oracle(self):
        """Agreement with the sequential path is not sufficient on its own.

        If both scans shared a bug they would agree with each other and the
        equivalence test would pass. So compare the chunked path against the
        step-6 oracle -- written from the recurrence definition, not from
        either implementation -- fed the layer's own captured pre-scan bits.
        """
        with global_dtype_policy("float64"):
            layer = GatedLinearAttentionBlock(
                dim=32, num_heads=4, head_dim=8, max_seq_len=128, chunk_size=64
            )
            x = keras.ops.convert_to_tensor(
                np.random.default_rng(11).standard_normal((2, 70, 32)).astype("float64")
            )
            captured = capture_scan_io(layer, x)
            expected = numpy_gated_linear_recurrence(
                captured["q"], captured["k"], captured["v"],
                captured["alpha"], captured["beta"],
            )
            chunked = keras.ops.convert_to_numpy(
                layer._chunked_scan(
                    keras.ops.convert_to_tensor(captured["q"]),
                    keras.ops.convert_to_tensor(captured["k"]),
                    keras.ops.convert_to_tensor(captured["v"]),
                    keras.ops.convert_to_tensor(captured["alpha"]),
                    keras.ops.convert_to_tensor(captured["beta"]),
                    70,
                )
            )

        assert chunked.shape == expected.shape
        tol = oracle_tolerance(70, 8, expected)
        err = float(np.abs(chunked - expected).max())
        assert err <= tol, (
            f"the CHUNKED scan disagrees with the independent NumPy "
            f"recurrence: max|diff|={err:.3e} > tol={tol:.3e}"
        )

    @pytest.mark.parametrize("alpha_value", [0.5, 0.1, 0.01, 1e-4, 1e-7])
    def test_small_alpha_stays_finite_and_equivalent(self, alpha_value):
        """Guard the numerical reason this factorization was chosen.

        The textbook two-vector form (``k * exp(-D)``) is algebraically the
        same but materializes an unbounded reciprocal: measured, it overflows
        float32 at chunk_size=64 once alpha reaches 0.1, and random-init
        sigmoid alpha already reaches 0.0111 -- so it fails at initialization,
        not in a contrived corner. Without this test the stability argument
        would be unguarded and a future "simplification" back to the
        reciprocal form would ship green.
        """
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, head_dim=8, max_seq_len=512, chunk_size=64
        )
        args = _scan_inputs(257, 4, 8, "float32", alpha_value=alpha_value)
        reference = keras.ops.convert_to_numpy(layer._sequential_scan(*args))
        chunked = keras.ops.convert_to_numpy(layer._chunked_scan(*args, 257))

        assert np.isfinite(chunked).all(), (
            f"chunked scan produced non-finite values at alpha={alpha_value}"
        )
        err = float(np.abs(reference - chunked).max())
        scale = float(np.abs(reference).max()) or 1.0
        assert err <= 4.0 * _TF32_ULP * scale, (
            f"chunked scan diverged from the reference at alpha={alpha_value}: "
            f"max|diff|={err:.3e}"
        )

    def test_padding_is_causally_invisible_to_real_steps(self):
        """Truncating to a length that needs padding reproduces the prefix exactly.

        RENAMED and REPAIRED. The old name and docstring asserted the padding was
        safe BECAUSE the padded values are "gate-neutral". That mechanism claim is
        false and the test could not detect it: the pad occupies the TAIL of the
        last chunk, causally downstream of every real step, so it is invisible no
        matter what is padded in. Measured -- padding alpha with 1.0, 0.5 or 0.0
        gives bit-identical output, and padding q/k/v/beta with 1e30 does too, so
        this assertion held under mutations it appeared to forbid.

        What it really pins is the CAUSAL claim, which is worth keeping: the
        prefix of a longer run must match a shorter padded run bit-exactly. 65
        needs 63 padded steps; 128 needs none. The value-independence claim is
        tested separately below, and the constraint that actually exists (pads
        must be FINITE) is tested after that.
        """
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, head_dim=8, max_seq_len=512, chunk_size=64
        )
        full = _scan_inputs(128, 4, 8, "float32", seed=3)
        out_full = keras.ops.convert_to_numpy(layer._chunked_scan(*full, 128))
        prefix = [x[:, :65] for x in full]
        out_prefix = keras.ops.convert_to_numpy(layer._chunked_scan(*prefix, 65))

        assert np.array_equal(out_full[:, :65], out_prefix), (
            "a padded chunk perturbed the real timesteps: "
            f"max|diff|={float(np.abs(out_full[:, :65] - out_prefix).max()):.3e} "
            "(expected exactly 0)"
        )

    def test_pad_values_are_irrelevant_but_must_be_finite(self):
        """The REAL padding contract, replacing the false gate-neutral claim.

        Two halves, both measured:

        * Any FINITE pad value gives bit-identical output. This is what makes the
          "gate-neutral" framing wrong -- correctness comes from the causal mask
          plus the final slice, not from alpha=1.
        * A NaN or inf pad DOES reach real rows, because the mask is applied
          multiplicatively and `NaN * 0 = NaN`. This is the constraint that
          genuinely exists, and it is the reason the decay exponent is masked
          BEFORE the exp rather than clamped after it (D-009).

        Driven through `_chunked_scan` on a length that needs padding (65 with
        chunk_size=64 pads 63 steps), by padding the inputs BY HAND to 128 with a
        chosen filler and comparing against the layer's own internal padding.
        """
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, head_dim=8, max_seq_len=512, chunk_size=64
        )
        seq = 65
        base = _scan_inputs(seq, 4, 8, "float64", seed=17)
        with global_dtype_policy("float64"):
            reference = keras.ops.convert_to_numpy(
                layer._chunked_scan(*base, seq)
            )

            def run_with_pad(fill_qkvb, fill_alpha):
                """Hand-pad to a chunk multiple, then scan at the padded length."""
                q, k, v, alpha, beta = [
                    keras.ops.convert_to_numpy(x) for x in base
                ]
                pad = 128 - seq

                def padded(arr, fill):
                    width = [(0, 0), (0, pad)] + [(0, 0)] * (arr.ndim - 2)
                    return np.pad(arr, width, constant_values=fill)

                args = [
                    keras.ops.convert_to_tensor(padded(q, fill_qkvb)),
                    keras.ops.convert_to_tensor(padded(k, fill_qkvb)),
                    keras.ops.convert_to_tensor(padded(v, fill_qkvb)),
                    keras.ops.convert_to_tensor(padded(alpha, fill_alpha)),
                    keras.ops.convert_to_tensor(padded(beta, fill_qkvb)),
                ]
                out = keras.ops.convert_to_numpy(layer._chunked_scan(*args, 128))
                return out[:, :seq]

            # (a) finite pads are irrelevant, including deliberately hostile ones
            for fill_qkvb, fill_alpha in ((0.0, 1.0), (0.0, 0.5), (0.0, 0.0),
                                          (1e30, 1.0), (-1e30, 0.25)):
                got = run_with_pad(fill_qkvb, fill_alpha)
                assert np.array_equal(got, reference), (
                    f"pad values (qkvb={fill_qkvb}, alpha={fill_alpha}) changed "
                    f"the real rows by "
                    f"{float(np.abs(got - reference).max()):.3e}; the padding is "
                    f"NOT value-independent as documented"
                )

            # (b) a non-finite pad DOES poison real rows -- the real constraint
            poisoned = run_with_pad(np.nan, 1.0)
            assert not np.all(np.isfinite(poisoned)), (
                "a NaN pad did NOT reach the real rows — the documented "
                "`NaN * 0 = NaN` propagation no longer holds, so the source "
                "note explaining why pads must be finite is now wrong"
            )


class TestChunkSizeParameter:
    """``chunk_size`` must be validated, serialized, and behaviour-neutral."""

    @pytest.mark.parametrize("bad", [0, -1, -64])
    def test_non_positive_chunk_size_raises(self, bad):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            GatedLinearAttentionBlock(
                dim=32, num_heads=4, max_seq_len=64, chunk_size=bad
            )

    def test_chunk_size_round_trips_through_get_config(self):
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, max_seq_len=64, chunk_size=16
        )
        config = layer.get_config()
        assert config["chunk_size"] == 16
        assert GatedLinearAttentionBlock.from_config(config).chunk_size == 16

    def test_chunk_size_survives_a_full_keras_round_trip(self):
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, max_seq_len=64, chunk_size=16
        )
        inputs = keras.Input(shape=(20, 32))
        model = keras.Model(inputs, layer(inputs))
        x = np.random.default_rng(0).normal(size=(2, 20, 32)).astype("float32")
        before = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "chunked.keras")
            model.save(path)
            restored = keras.models.load_model(path)

        assert restored.layers[-1].chunk_size == 16
        np.testing.assert_allclose(
            before, restored.predict(x, verbose=0), rtol=1e-6, atol=1e-6
        )

    @pytest.mark.parametrize("chunk_size", [1, 7, 16, 64, 256])
    def test_result_is_independent_of_chunk_size(self, chunk_size):
        """chunk_size is a performance knob, not a semantic one.

        Includes chunk_size=1 (degenerate: every chunk is one timestep) and 256
        (larger than the sequence, so one heavily padded chunk).
        """
        with global_dtype_policy("float64"):
            layer = GatedLinearAttentionBlock(
                dim=32, num_heads=4, head_dim=8,
                max_seq_len=256, chunk_size=chunk_size,
            )
            args = _scan_inputs(100, 4, 8, "float64", seed=7)
            reference = keras.ops.convert_to_numpy(layer._sequential_scan(*args))
            chunked = keras.ops.convert_to_numpy(layer._chunked_scan(*args, 100))

        err = float(np.abs(reference - chunked).max())
        assert err <= 1e-10, (
            f"chunk_size={chunk_size} changed the result: max|diff|={err:.3e}"
        )


class TestScanDispatch:
    """``gated_linear_scan`` must route to the right implementation."""

    def test_static_length_takes_the_chunked_path(self):
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, head_dim=8, max_seq_len=128, chunk_size=64
        )
        args = _scan_inputs(70, 4, 8, "float32", seed=2)
        calls = []
        real_chunked = layer._chunked_scan

        def spy(*a, **kw):
            calls.append(1)
            return real_chunked(*a, **kw)

        layer._chunked_scan = spy
        try:
            layer.gated_linear_scan(*args)
        finally:
            del layer._chunked_scan
        assert calls, "a static sequence length did not dispatch to _chunked_scan"

    def test_a_symbolic_input_model_reaches_BOTH_branches(self):
        """RENAMED and REPAIRED, because the old name asserted a falsehood.

        The old test claimed a symbolic sequence axis "falls back to the
        sequential path", and then asserted only output shape and finiteness --
        never which branch ran. It passed with the symbolic branch replaced by
        `zeros`, which is how it was caught.

        MEASURED behaviour, which is more interesting than the claim: with
        `keras.Input(shape=(None, dim))`, the TF trainer RETRACES on the concrete
        data spec, so the first `predict` at one length reaches
        `_chunked_scan`. Only after a SECOND distinct length does TF relax the
        signature to a dynamic shape, and `_sequential_scan` appear. So a single
        model with a symbolic sequence axis exercises BOTH branches over its
        lifetime, depending on how many distinct lengths it has seen.

        That is exactly why the two branches must agree (see
        `TestChunkedScanEquivalence` and `TestScanBranchAgreementAboveUnitGate`):
        a user cannot predict from the model definition which one will run.
        """
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, head_dim=8, max_seq_len=128, chunk_size=64
        )
        seen = {"chunked": 0, "sequential": 0}
        real_chunked, real_sequential = layer._chunked_scan, layer._sequential_scan

        def chunked_spy(*a, **k):
            seen["chunked"] += 1
            return real_chunked(*a, **k)

        def sequential_spy(*a, **k):
            seen["sequential"] += 1
            return real_sequential(*a, **k)

        layer._chunked_scan = chunked_spy
        layer._sequential_scan = sequential_spy
        try:
            inputs = keras.Input(shape=(None, 32))
            model = keras.Model(inputs, layer(inputs))
            for seq_len in (30, 70):
                x = np.random.default_rng(seq_len).normal(
                    size=(2, seq_len, 32)
                ).astype("float32")
                out = model.predict(x, verbose=0)
                assert out.shape == (2, seq_len, 32)
                assert np.isfinite(out).all()
        finally:
            del layer._chunked_scan
            del layer._sequential_scan

        total = seen["chunked"] + seen["sequential"]
        assert total > 0, (
            "neither scan branch ran — the model produced output without "
            "executing the recurrence at all"
        )
        assert seen["sequential"] > 0, (
            f"the sequential fallback was never reached across two distinct "
            f"sequence lengths, so the symbolic path is untested here: {seen}"
        )

    def test_fully_symbolic_signature_takes_the_sequential_path(self):
        """An explicitly symbolic `TensorSpec` must dispatch to the fallback.

        The complement of the test above, and the unambiguous case: when the
        sequence axis is symbolic at trace time there IS no concrete chunk grid,
        so `_chunked_scan` cannot be used. Asserts the dispatch itself rather
        than inferring it from the output.
        """
        layer = GatedLinearAttentionBlock(
            dim=32, num_heads=4, head_dim=8, max_seq_len=128, chunk_size=64
        )
        layer.build((None, None, 32))

        seen = {"chunked": 0, "sequential": 0}
        real_chunked, real_sequential = layer._chunked_scan, layer._sequential_scan

        def chunked_spy(*a, **k):
            seen["chunked"] += 1
            return real_chunked(*a, **k)

        def sequential_spy(*a, **k):
            seen["sequential"] += 1
            return real_sequential(*a, **k)

        layer._chunked_scan = chunked_spy
        layer._sequential_scan = sequential_spy
        try:
            traced = tf.function(
                lambda t: layer(t),
                input_signature=[tf.TensorSpec([None, None, 32], tf.float32)],
            )
            out = traced(tf.zeros((2, 40, 32)))
        finally:
            del layer._chunked_scan
            del layer._sequential_scan

        assert tuple(out.shape) == (2, 40, 32)
        assert seen["sequential"] > 0, (
            f"a fully symbolic signature did not reach _sequential_scan: {seen}"
        )
        assert seen["chunked"] == 0, (
            f"_chunked_scan ran on a symbolic sequence axis, where it has no "
            f"concrete chunk grid: {seen}"
        )



class TestScanBranchAgreementAboveUnitGate:
    """The two branches must compute the same function for `alpha > 1` too.

    `_chunked_scan` clamped its gate exponents with `ops.minimum(., 0.0)`. That
    is a no-op while `alpha <= 1` keeps the cumulative log-gate non-increasing,
    but for `alpha > 1` the sign flips and the clamp silently saturated the
    exponent to `exp(0)=1` INSIDE the causal region, where the value matters.
    `_sequential_scan` has no such clamp, so the two branches disagreed on a
    domain the public `gated_linear_scan` never excluded: measured 2.59 at
    alpha=1.05 and 3.59e+04 at alpha=2.0.

    Compared in RELATIVE terms against the independent NumPy oracle, because
    `alpha > 1` makes the recurrence genuinely growing -- at alpha=2.0,
    seq_len=128 the true output magnitude is ~1e+24, so an absolute tolerance
    would be meaningless here.
    """

    @pytest.mark.parametrize("alpha_value", [1.05, 1.5, 2.0])
    @pytest.mark.parametrize("seq_len,chunk_size", [(32, 8), (65, 64), (128, 32)])
    def test_branches_agree_above_unit_gate(self, alpha_value, seq_len, chunk_size):
        """Restoring either `ops.minimum(., 0.0)` clamp fails here."""
        num_heads, head_dim = 2, 8
        with global_dtype_policy("float64"):
            layer = GatedLinearAttentionBlock(
                dim=num_heads * head_dim, num_heads=num_heads,
                head_dim=head_dim, max_seq_len=2048, chunk_size=chunk_size,
            )
            args = _scan_inputs(
                seq_len, num_heads, head_dim, "float64",
                alpha_value=alpha_value, seed=101,
            )
            reference = keras.ops.convert_to_numpy(layer._sequential_scan(*args))
            chunked = keras.ops.convert_to_numpy(
                layer._chunked_scan(*args, seq_len)
            )
            oracle = numpy_gated_linear_recurrence(
                *[keras.ops.convert_to_numpy(a) for a in args]
            )

        scale = max(float(np.abs(reference).max()), 1e-300)
        rel_branches = float(np.abs(reference - chunked).max()) / scale
        rel_oracle = float(np.abs(chunked - oracle).max()) / scale

        assert rel_branches <= 1e-10, (
            f"chunked and sequential disagree at alpha={alpha_value}, "
            f"seq_len={seq_len}, chunk_size={chunk_size}: "
            f"relative max|diff|={rel_branches:.3e} -- the gate exponent is "
            f"being clamped, saturating a growing gate inside the causal region"
        )
        assert rel_oracle <= 1e-10, (
            f"chunked disagrees with the independent NumPy oracle at "
            f"alpha={alpha_value}: relative max|diff|={rel_oracle:.3e}"
        )

    def test_unit_gate_boundary_is_not_a_discontinuity(self):
        """CONTROL: alpha slightly below/at/above 1 must all track the oracle.

        Without this, a fix that special-cased `alpha > 1` -- or that broke the
        `alpha <= 1` path while fixing the other side -- would still pass the
        parametrized test above.
        """
        num_heads, head_dim, seq_len = 2, 8, 65
        results = {}
        with global_dtype_policy("float64"):
            for alpha_value in (0.95, 1.0, 1.05):
                layer = GatedLinearAttentionBlock(
                    dim=num_heads * head_dim, num_heads=num_heads,
                    head_dim=head_dim, max_seq_len=2048, chunk_size=64,
                )
                args = _scan_inputs(
                    seq_len, num_heads, head_dim, "float64",
                    alpha_value=alpha_value, seed=202,
                )
                chunked = keras.ops.convert_to_numpy(
                    layer._chunked_scan(*args, seq_len)
                )
                oracle = numpy_gated_linear_recurrence(
                    *[keras.ops.convert_to_numpy(a) for a in args]
                )
                scale = max(float(np.abs(oracle).max()), 1e-300)
                results[alpha_value] = (
                    float(np.abs(chunked - oracle).max()) / scale
                )

        for alpha_value, rel in results.items():
            assert rel <= 1e-10, (
                f"alpha={alpha_value} deviates from the oracle by "
                f"{rel:.3e} (relative); all of {sorted(results)} must hold"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
