"""
Comprehensive pytest test suite for the Mamba v2 foundation model.

This module provides extensive testing for the Mamba v2 implementation,
mirroring the structure of the Mamba v1 test suite. It covers:
- Foundation model initialization and V2-specific parameter validation.
- Architecture building and consistent output shape.
- Forward pass functionality, including the parallel SSM/MLP paths.
- Model variant creation and configuration for V2 architectures.
- Serialization and deserialization of the foundation model.
- Error handling and edge cases.
- End-to-end integration testing for gradient flow and training.
- Advanced V2-specific features like RMSNorm toggling.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any
from unittest import mock

from dl_techniques.models.language.mamba.mamba_v2 import Mamba2
from dl_techniques.models.language.mamba.components_v2 import Mamba2Layer

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight
from tests.numerics import reassociation_atol

class TestMamba2ModelInitialization:
    """Test Mamba v2 model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic Mamba v2 model initialization as a pure encoder."""
        model = Mamba2(
            vocab_size=1000,
            d_model=256,
            num_layers=6,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=64
        )
        assert model.d_model == 256
        assert model.num_layers == 6
        assert model.d_state == 128
        assert model.headdim == 64
        assert not model.built
        assert len(model.encoder_layers) == 6

    def test_parameter_validation(self):
        """Test Mamba v2 parameter validation for invalid values."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Mamba2(vocab_size=0, d_model=256, num_layers=4)
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Mamba2(vocab_size=-1, d_model=256, num_layers=4)
        with pytest.raises(ValueError, match="d_model must be positive"):
            Mamba2(vocab_size=1000, d_model=0, num_layers=4)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            Mamba2(vocab_size=1000, d_model=256, num_layers=0)

        # V2 specific validation: headdim must divide d_ssm
        with pytest.raises(ValueError, match="d_ssm .* must be divisible by headdim"):
            d_ssm = 256 * 2  # d_inner
            # This layer will fail on build because 512 is not divisible by 60
            Mamba2Layer(d_model=256, expand=2, headdim=60)

    def test_initialization_with_custom_config(self):
        """Test Mamba v2 model initialization with custom configuration."""
        model = Mamba2(
            vocab_size=25000,
            d_model=512,
            num_layers=8,
            d_state=64,
            d_conv=8,
            expand=3,
            headdim=128,
            norm_epsilon=1e-6,
            rmsnorm=False
        )
        assert model.rmsnorm is False
        assert model.norm_epsilon == 1e-6
        # Check a property of the inner layer
        first_mamba2_layer = model.encoder_layers[0].mamba2
        assert first_mamba2_layer.rmsnorm is False


class TestMamba2ModelVariants:
    """Test Mamba v2 model variants and factory methods."""

    # These pin the released Mamba-2 checkpoint shapes
    # (https://huggingface.co/state-spaces/mamba2-<size>/raw/main/config.json,
    # fetched 2026-08-18). Until then this table asserted ("370m", 1024, 24) and
    # ("780m", 1536, 36) -- i.e. it pinned the defect, so `from_variant("370m")`
    # building half the advertised model was a GREEN test. Do not "simplify" these
    # back toward `mamba_v1`'s table: v1 is a different size series (790m/1.4b/2.8b
    # against v2's 780m/1.3b/2.7b). See the D-024 anchor in `mamba_v2.py`.
    @pytest.mark.parametrize("variant, d_model, num_layers", [
        ("130m", 768, 24),
        ("370m", 1024, 48),
        ("780m", 1536, 48),
        ("1.3b", 2048, 48),
        ("2.7b", 2560, 64),
    ])
    def test_variants(self, variant, d_model, num_layers):
        """Test all standard parameter variants."""
        model = Mamba2.from_variant(variant, vocab_size=50257)
        assert model.d_model == d_model
        assert model.num_layers == num_layers
        assert model.vocab_size == 50257

    @pytest.mark.parametrize("alias, canonical", [
        ("base", "130m"),
        ("1.4b", "1.3b"),
        ("2.8b", "2.7b"),
    ])
    def test_variant_aliases_resolve_to_canonical_row(self, alias, canonical):
        """Non-size spellings still build, and build the row they alias.

        `1.4b`/`2.8b` are Mamba-*1* size names that were this table's keys before
        2026-08-18. They are kept accepting so no existing caller breaks, but they
        must not drift away from the v2 row they resolve to.
        """
        assert alias not in Mamba2.MODEL_VARIANTS
        expected = Mamba2.MODEL_VARIANTS[canonical]
        model = Mamba2.from_variant(alias, vocab_size=50257)
        assert model.d_model == expected["d_model"]
        assert model.num_layers == expected["num_layers"]

    def test_mamba1_only_size_names_are_not_v2_sizes(self):
        """The v2 table must not re-grow rows the Mamba-2 series never shipped.

        `state-spaces/mamba2-1.4b`, `-2.8b` and `-790m` do not exist (HTTP 401 /
        no such repo); `state-spaces/mamba-*` does ship those names. Listing them
        as v2 *sizes* is what let v1's table get copied over v2's in the first
        place.
        """
        for v1_only in ("790m", "1.4b", "2.8b"):
            assert v1_only not in Mamba2.MODEL_VARIANTS, (
                f"'{v1_only}' is a Mamba-1 size name; it may be an alias, not a row"
            )
        assert set(Mamba2.MODEL_VARIANTS) == {"130m", "370m", "780m", "1.3b", "2.7b"}

    def test_invalid_variant(self):
        """Test error handling for invalid variant names."""
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            Mamba2.from_variant("invalid", vocab_size=50257)

    def test_variant_with_custom_params(self):
        """Test creating variant with custom parameter overrides."""
        model = Mamba2.from_variant(
            "base",
            vocab_size=50257,
            d_state=64,
            expand=3
        )
        assert model.d_model == 768
        assert model.num_layers == 24
        assert model.d_state == 64
        assert model.expand == 3


class TestMamba2ModelBuilding:
    """Test Mamba v2 model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": 1000, "d_model": 128, "num_layers": 2,
            "d_state": 32, "d_conv": 4, "expand": 2, "headdim": 32
        }

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality and output contract."""
        model = Mamba2(**basic_config)
        input_ids = keras.random.randint((2, 16), 0, 1000)
        outputs = model({"input_ids": input_ids})

        assert model.built
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (2, 16, 128)

    def test_encoder_layers_configuration(self, basic_config):
        """Test that encoder layers are properly configured."""
        model = Mamba2(**basic_config)
        _ = model(keras.random.randint((1, 8), 0, 1000))

        for i, block in enumerate(model.encoder_layers):
            assert block.d_model == 128
            assert block.mamba2.d_state == 32
            assert block.mamba2.headdim == 32


class TestMamba2ModelForwardPass:
    """Test Mamba v2 model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> Mamba2:
        model = Mamba2(
            vocab_size=1000, d_model=64, num_layers=2, d_state=16,
            d_conv=4, expand=2, headdim=32
        )
        _ = model(keras.random.randint((1, 8), 0, 1000))
        return model

    def test_forward_pass_with_tensor_input(self, built_model):
        input_ids = keras.random.randint((2, 16), 0, 1000)
        outputs = built_model(input_ids)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (2, 16, 64)

    def test_forward_pass_with_dict_input(self, built_model):
        inputs = {'input_ids': keras.random.randint((3, 12), 0, 1000)}
        outputs = built_model(inputs)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (3, 12, 64)

    def test_forward_pass_variable_sequence_lengths(self, built_model):
        for seq_length in [8, 16, 32]:
            outputs = built_model(keras.random.randint((2, seq_length), 0, 1000))
            assert outputs["last_hidden_state"].shape == (2, seq_length, 64)

    def test_invalid_dict_input(self, built_model):
        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            built_model({'invalid_key': keras.ops.ones((2, 16), 'int32')})

    def test_training_vs_inference_mode(self, built_model):
        input_ids = keras.random.randint((2, 16), 0, 1000)
        output_train = built_model(input_ids, training=True)
        assert output_train["last_hidden_state"].shape == (2, 16, 64)

        output_inference1 = built_model(input_ids, training=False)
        output_inference2 = built_model(input_ids, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_inference1["last_hidden_state"]),
            keras.ops.convert_to_numpy(output_inference2["last_hidden_state"])
        )

class TestMamba2ModelSerialization:
    """Test Mamba v2 model serialization and deserialization."""

    def test_config_serialization(self):
        model = Mamba2(
            vocab_size=1000, d_model=128, num_layers=2, d_state=32, headdim=64
        )
        config = model.get_config()
        assert config['vocab_size'] == 1000
        assert config['d_model'] == 128
        assert config['headdim'] == 64

    def test_model_save_load_cycle(self):
        model = Mamba2(vocab_size=1000, d_model=64, num_layers=2, d_state=16, headdim=32)
        input_ids = keras.random.randint((2, 16), 0, 1000)
        original_outputs = model(input_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mamba2.keras")
            model.save(path)
            loaded_model = keras.models.load_model(path)
            loaded_outputs = loaded_model(input_ids)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs['last_hidden_state']),
                keras.ops.convert_to_numpy(loaded_outputs['last_hidden_state']),
                rtol=1e-5
            )

class TestMamba2EdgeCases:
    """Test Mamba v2 model edge cases."""

    def test_minimum_sequence_length(self):
        model = Mamba2(vocab_size=1000, d_model=64, num_layers=2, d_state=16, headdim=32)
        outputs = model(keras.ops.array([[42]], "int32"))
        assert outputs['last_hidden_state'].shape == (1, 1, 64)

    def test_mlp_path_integration(self):
        """Test model where d_ssm is less than d_inner, activating MLP path."""
        d_model = 128
        expand = 2
        d_inner = d_model * expand
        d_ssm = d_inner // 2

        model = Mamba2(
            vocab_size=1000, d_model=d_model, num_layers=2, d_state=32,
            expand=expand, headdim=32, d_ssm=d_ssm
        )

        outputs = model(keras.random.randint((2, 16), 0, 1000))
        assert outputs['last_hidden_state'].shape == (2, 16, d_model)
        assert model.encoder_layers[0].mamba2.d_ssm == d_ssm

    def test_output_not_nan_or_inf(self):
        """Test that outputs don't contain NaN or Inf values."""
        model = Mamba2(vocab_size=1000, d_model=64, num_layers=2, d_state=16, headdim=32)
        outputs = model(keras.random.randint((4, 32), 0, 1000))
        hidden_states = keras.ops.convert_to_numpy(outputs['last_hidden_state'])
        assert not np.isnan(hidden_states).any()
        assert not np.isinf(hidden_states).any()


class TestMamba2Integration:
    """Integration tests for the complete Mamba v2 model."""

    @pytest.fixture
    def small_model(self) -> Mamba2:
        return Mamba2(
            vocab_size=1000, d_model=64, num_layers=2, d_state=16,
            d_conv=2, expand=2, headdim=32
        )

    def test_gradient_flow(self, small_model):
        """Test that gradients flow through the entire model."""
        input_ids = keras.random.randint((2, 16), 0, 1000)
        with tf.GradientTape() as tape:
            outputs = small_model(input_ids, training=True)
            loss = keras.ops.mean(outputs['last_hidden_state']**2)
        gradients = tape.gradient(loss, small_model.trainable_weights)

        assert all(g is not None for g in gradients)
        assert all(keras.ops.any(g != 0) for g in gradients)

    def test_gradient_flow_through_ssm_params(self, small_model):
        """Test gradients for A_log, D, and dt_bias."""
        input_ids = keras.random.randint((2, 16), 0, 1000)
        with tf.GradientTape() as tape:
            outputs = small_model(input_ids, training=True)
            loss = keras.ops.mean(outputs['last_hidden_state']**2)

        mamba_layer = small_model.encoder_layers[0].mamba2
        names = ["A_log", "D", "dt_bias"]
        ssm_params = [mamba_layer.A_log, mamba_layer.D, mamba_layer.dt_bias]
        grads = tape.gradient(loss, ssm_params)

        # The old assertion was `not np.allclose(grad, 1e-1)` -- a comparison
        # against the constant 0.1, which is TRUE of the all-zero gradient and
        # therefore green for an SSM whose defining parameters receive nothing.
        # Almost certainly a typo for 0.0, but `allclose(x, 0)` is the wrong
        # instrument too: its atol=1e-8 is an absolute floor and A_log's
        # gradient scales with dt in [1e-3, 1e-1] by construction (see the
        # sibling `test_mamba_v1.py::test_gradient_flow_through_ssm_params`,
        # which measured 3.17e-08 for a CORRECT model). Same scale-free form
        # here: finite, not identically zero, and not mostly zeros.
        for name, grad in zip(names, grads):
            assert grad is not None, f"{name} gradient is None"
            g = keras.ops.convert_to_numpy(grad)
            assert np.isfinite(g).all(), f"{name} gradient has non-finite entries"
            assert np.abs(g).max() > 0.0, f"{name} gradient is identically zero"
            assert np.count_nonzero(g) > g.size // 2, (
                f"{name} gradient is mostly zeros "
                f"({np.count_nonzero(g)}/{g.size} non-zero)"
            )

    def test_training_integration(self, small_model):
        """Test the model in a minimal training loop."""
        optimizer = keras.optimizers.Adam()
        input_ids = keras.random.randint((4, 16), 0, 1000)
        targets = keras.random.randint((4, 16), 0, 1000)

        with tf.GradientTape() as tape:
            outputs = small_model(input_ids, training=True)
            logits = keras.layers.Dense(1000)(outputs['last_hidden_state'])
            initial_loss = keras.losses.sparse_categorical_crossentropy(
                targets, logits, from_logits=True
            )
        grads = tape.gradient(initial_loss, small_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, small_model.trainable_weights))

        with tf.GradientTape() as tape:
            outputs = small_model(input_ids, training=True)
            logits = keras.layers.Dense(1000)(outputs['last_hidden_state'])
            final_loss = keras.losses.sparse_categorical_crossentropy(
                targets, logits, from_logits=True
            )


class TestMamba2LayerCheckpointedScanGradients:
    """`Mamba2Layer.call()` wraps `self._ssm_scan` in `tf.recompute_grad`
    (plan-2026-09-13T165751-bc5433cb, step 3, D-001) to trade recompute FLOPs
    for backward-pass memory. This class is the direct verification of
    Assumption A1 for v2: that `tf.recompute_grad` propagates gradients
    correctly to every trainable weight -- including `self.dt_bias`, which is
    only read inside `_ssm_scan`, not passed as an explicit tensor argument --
    and that the wrap changes only backward-pass memory/compute, never any
    computed value.

    Shapes here are deliberately tiny (CPU-feasible, no GPU needed) -- this
    class asserts correctness, not the memory reduction, which step 5 measures
    separately on GPU1.
    """

    def _build_layer_and_input(self):
        """A small, built `Mamba2Layer` and a fixed input, for CPU-only tests."""
        layer = Mamba2Layer(
            d_model=8, d_state=4, d_conv=2, expand=2, headdim=8, ngroups=1
        )
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((2, 6, 8)).astype("float32")
        )
        # First call builds the layer's weights.
        layer(x, training=True)
        return layer, x

    def _tape_gradients(self, layer, x):
        """One `GradientTape` step: gradients w.r.t. every trainable weight."""
        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = keras.ops.mean(keras.ops.square(y))
        return tape.gradient(loss, layer.trainable_weights)

    def test_checkpointed_and_noncheckpointed_gradients_agree_per_weight(self):
        """Checkpointed gradients (current `call()`) must match the gradients
        `tf.recompute_grad` would otherwise have replaced, for EVERY trainable
        weight -- not just an aggregate loss scalar. This also exercises
        `self.dt_bias` and `self.A_log`, both read from inside `_ssm_scan`
        rather than passed in as explicit tensor arguments, which is exactly
        the closure-over-`self` mechanism Assumption A1 is about.

        The non-checkpointed gradient set is obtained by monkeypatching
        `tensorflow.recompute_grad` to an identity passthrough
        (`lambda fn: fn`), rather than calling `self._ssm_scan` directly:
        `call()` decides `scan_fn` internally (backend-guarded), so patching
        the module-level `tf.recompute_grad` the wrap actually calls is the
        only way to exercise the SAME `call()` code path with the wrap
        neutralized, instead of hand-duplicating `call()`'s pre/post-scan
        tensor plumbing in the test.
        """
        layer, x = self._build_layer_and_input()
        seq_len = int(x.shape[1])

        checkpointed_grads = self._tape_gradients(layer, x)
        with mock.patch("tensorflow.recompute_grad", lambda fn: fn):
            noncheckpointed_grads = self._tape_gradients(layer, x)

        weights = layer.trainable_weights
        assert len(checkpointed_grads) == len(weights)
        assert len(noncheckpointed_grads) == len(weights)

        for w, g_ckpt, g_plain in zip(weights, checkpointed_grads, noncheckpointed_grads):
            assert g_ckpt is not None, f"{w.path}: checkpointed gradient is None"
            assert g_plain is not None, f"{w.path}: non-checkpointed gradient is None"

            g_ckpt_np = keras.ops.convert_to_numpy(g_ckpt)
            g_plain_np = keras.ops.convert_to_numpy(g_plain)

            # Dominant per-step reduction in `_ssm_scan` is the state-axis
            # contraction in `einsum('bhpn,bhn->bhp', h, C[:, t])`, applied
            # once per sequence step (the while_loop's recurrence depth) --
            # so reduction_lengths=[d_state], num_steps=seq_len, mirroring the
            # `test_mdta.py` per-op-chain convention.
            scale = float(max(np.abs(g_ckpt_np).max(), np.abs(g_plain_np).max()))
            atol = reassociation_atol([layer.d_state], seq_len, scale=scale)

            np.testing.assert_allclose(
                g_ckpt_np,
                g_plain_np,
                atol=atol,
                rtol=0,
                err_msg=f"gradient mismatch for weight {w.path}",
            )

    def test_gradient_flow_oracle_passes_with_the_wrap_in_place(self):
        """RED-proof: the wrap must not silently disconnect any weight from
        the backward graph. Runs the shared oracle against the checkpointed
        (current, shipped) `call()` path AND, for symmetry, against the
        non-checkpointed path -- both must reach every trainable weight.
        """
        layer, x = self._build_layer_and_input()

        assert_gradients_reach_every_trainable_weight(layer, x, training=True)

        with mock.patch("tensorflow.recompute_grad", lambda fn: fn):
            assert_gradients_reach_every_trainable_weight(layer, x, training=True)

    def test_forward_pass_bit_identical_with_and_without_gradient_tape(self):
        """The wrap only changes backward-pass behavior: `tf.recompute_grad`'s
        wrapped function degrades to a normal forward call whenever no tape is
        watching. A plain forward pass (no tape) must therefore be
        bit-identical to a forward pass made inside a `GradientTape` context,
        for the checkpointed (current, shipped) `call()`.
        """
        layer, x = self._build_layer_and_input()

        y_no_tape = layer(x, training=False)

        with tf.GradientTape():
            y_with_tape = layer(x, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_no_tape),
            keras.ops.convert_to_numpy(y_with_tape),
            atol=0,
            rtol=0,
            err_msg="forward output changed depending on GradientTape presence",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])