"""Tests for the CliffordNetLM causal language model.

Covers initialization, parameter validation, forward pass, causality,
serialization, variant creation, and gradient flow.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.models.cliffordnet.lm import CliffordNetLM


def _random_ids(shape, vocab_size):
    """Generate random integer token IDs."""
    return np.random.randint(0, vocab_size, shape).astype(np.int32)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    """Minimal CliffordNetLM config for fast testing."""
    return {
        "vocab_size": 256,
        "max_seq_length": 32,
        "channels": 64,
        "depth": 2,
        "shifts": [1, 2],
        "dropout_rate": 0.0,
        "stochastic_depth_rate": 0.0,
    }


@pytest.fixture
def tiny_model(tiny_config):
    """Pre-built tiny CliffordNetLM."""
    model = CliffordNetLM(**tiny_config)
    dummy = _random_ids((1, 16), tiny_config["vocab_size"])
    model(dummy, training=False)
    return model


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMInitialization:

    def test_basic_initialization(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        assert model.vocab_size == 256
        assert model.channels == 64
        assert model.depth == 2
        assert model.shifts == [1, 2]
        assert model.max_seq_length == 32

    def test_default_initialization(self):
        model = CliffordNetLM(vocab_size=1000)
        assert model.channels == 128
        assert model.depth == 12
        assert model.max_seq_length == 512
        assert model.cli_mode == "full"
        assert model.ctx_mode == "diff"

    def test_custom_initialization(self):
        model = CliffordNetLM(
            vocab_size=500,
            channels=96,
            depth=4,
            shifts=[1, 2, 4],
            cli_mode="wedge",
            ctx_mode="abs",
            use_global_context=True,
            layer_scale_init=1e-3,
        )
        assert model.cli_mode == "wedge"
        assert model.ctx_mode == "abs"
        assert model.use_global_context is True
        assert model.shifts == [1, 2, 4]


# ---------------------------------------------------------------------
# Forward Pass Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMForwardPass:

    def test_forward_pass_basic(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        batch_size, seq_len = 2, 16
        input_ids = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, tiny_config["vocab_size"])

    def test_forward_pass_batch_size_one(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        input_ids = _random_ids((1, 8), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (1, 8, tiny_config["vocab_size"])

    def test_forward_pass_full_sequence(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        seq_len = tiny_config["max_seq_length"]
        input_ids = _random_ids((1, seq_len), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (1, seq_len, tiny_config["vocab_size"])

    def test_forward_pass_with_global_context(self, tiny_config):
        config = {**tiny_config, "use_global_context": True}
        model = CliffordNetLM(**config)
        input_ids = _random_ids((2, 16), config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (2, 16, config["vocab_size"])


# ---------------------------------------------------------------------
# Causal Masking Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMCausality:

    def test_causality_future_does_not_affect_past(self, tiny_config):
        """Changing a future token must not change any earlier position's logits."""
        config = {**tiny_config, "layer_scale_init": 1.0}
        model = CliffordNetLM(**config)

        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 5, 6, 7, 99]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)

        logits1 = keras.ops.convert_to_numpy(out1["logits"])
        logits2 = keras.ops.convert_to_numpy(out2["logits"])

        for pos in range(7):
            np.testing.assert_allclose(
                logits1[0, pos], logits2[0, pos], atol=1e-5,
                err_msg=f"Position {pos} logits changed when only position 7 changed "
                        f"(causality violation)",
            )
        assert not np.allclose(logits1[0, 7], logits2[0, 7], atol=1e-3)

    def test_causality_with_global_context(self, tiny_config):
        """Global context branch must also be causal."""
        config = {**tiny_config, "use_global_context": True, "layer_scale_init": 1.0}
        model = CliffordNetLM(**config)

        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 5, 6, 7, 99]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)

        logits1 = keras.ops.convert_to_numpy(out1["logits"])
        logits2 = keras.ops.convert_to_numpy(out2["logits"])

        for pos in range(7):
            np.testing.assert_allclose(
                logits1[0, pos], logits2[0, pos], atol=1e-5,
                err_msg=f"Global context leaks future info at position {pos}",
            )

    def test_causality_middle_change(self, tiny_config):
        """Changing a middle token must not affect earlier positions."""
        config = {**tiny_config, "layer_scale_init": 1.0, "depth": 4}
        model = CliffordNetLM(**config)

        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 99, 6, 7, 8]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)

        logits1 = keras.ops.convert_to_numpy(out1["logits"])
        logits2 = keras.ops.convert_to_numpy(out2["logits"])

        # Positions 0-3 must be unaffected
        for pos in range(4):
            np.testing.assert_allclose(
                logits1[0, pos], logits2[0, pos], atol=1e-5,
                err_msg=f"Position {pos} affected by change at position 4",
            )
        # Position 4+ should differ
        assert not np.allclose(logits1[0, 4], logits2[0, 4], atol=1e-3)


# ---------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMSerialization:

    def test_get_config(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        config = model.get_config()
        assert config["vocab_size"] == tiny_config["vocab_size"]
        assert config["channels"] == tiny_config["channels"]
        assert config["depth"] == tiny_config["depth"]
        assert config["shifts"] == tiny_config["shifts"]

    def test_from_config_roundtrip(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        config = model.get_config()
        model2 = CliffordNetLM.from_config(config)
        assert model2.vocab_size == model.vocab_size
        assert model2.channels == model.channels
        assert model2.depth == model.depth
        assert model2.shifts == model.shifts

    def test_compute_output_shape(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        shapes = model.compute_output_shape((None, 16))
        assert shapes["logits"] == (None, 16, tiny_config["vocab_size"])


# ---------------------------------------------------------------------
# Variant Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMVariants:

    def test_from_variant_nano(self):
        model = CliffordNetLM.from_variant("nano", vocab_size=256)
        assert model.channels == 128
        assert model.depth == 12
        assert model.shifts == [1, 2]

    def test_from_variant_mini(self):
        model = CliffordNetLM.from_variant("mini", vocab_size=256)
        assert model.channels == 192
        assert model.depth == 12
        assert model.shifts == [1, 2, 4]

    def test_from_variant_base(self):
        model = CliffordNetLM.from_variant("base", vocab_size=256)
        assert model.channels == 384
        assert model.depth == 18
        assert model.shifts == [1, 2, 4, 8, 16]

    def test_from_variant_large(self):
        model = CliffordNetLM.from_variant("large", vocab_size=256)
        assert model.channels == 512
        assert model.depth == 20

    def test_from_variant_xl(self):
        model = CliffordNetLM.from_variant("xl", vocab_size=256)
        assert model.channels == 768
        assert model.depth == 28

    def test_from_variant_with_overrides(self):
        model = CliffordNetLM.from_variant("nano", vocab_size=256, dropout_rate=0.2)
        assert model.dropout_rate == 0.2
        assert model.channels == 128

    def test_from_variant_unknown(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            CliffordNetLM.from_variant("nonexistent", vocab_size=256)

    def test_all_variants_have_required_keys(self):
        required_keys = {"channels", "depth", "shifts"}
        for name, config in CliffordNetLM.MODEL_VARIANTS.items():
            for key in required_keys:
                assert key in config, f"Variant '{name}' missing key '{key}'"


# ---------------------------------------------------------------------
# Gradient Flow Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMGradientFlow:

    def test_gradient_flow(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        batch_size, seq_len = 2, 8
        input_ids = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])
        labels = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])

        with tf.GradientTape() as tape:
            outputs = model(input_ids, training=True)
            logits = outputs["logits"]
            loss = keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True,
            )
            loss = keras.ops.mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        for var, grad in zip(model.trainable_variables, gradients):
            assert grad is not None, f"No gradient for {var.name}"

    def test_training_step(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        batch_size, seq_len = 4, 8
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        input_ids = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])
        labels = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])

        _ = model(input_ids, training=False)
        initial_weights = [w.numpy().copy() for w in model.trainable_weights[:2]]

        with tf.GradientTape() as tape:
            outputs = model(input_ids, training=True)
            loss = loss_fn(labels, outputs["logits"])

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        updated_weights = [w.numpy() for w in model.trainable_weights[:2]]
        for w_init, w_updated in zip(initial_weights, updated_weights):
            assert not np.allclose(w_init, w_updated), "Weights did not update"


# ---------------------------------------------------------------------
# TestNormalizationTypePassthrough — block ctx-norm propagation
# ---------------------------------------------------------------------


class TestNormalizationTypePassthrough:
    """Verify ``normalization_type`` propagates from CliffordNetLM into the
    inner CausalCliffordNetBlock instances, and that ``get_config`` /
    ``from_config`` round-trip preserves the value.
    """

    def test_normalization_type_passthrough(self, tiny_config):
        cfg = dict(tiny_config)
        cfg["normalization_type"] = "layer_norm"
        model = CliffordNetLM(**cfg)
        block = model.clifford_blocks[0]
        assert block.normalization_type == "layer_norm"
        assert isinstance(block.ctx_norm, keras.layers.LayerNormalization)

    def test_normalization_type_default_is_zero_centered_rms(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        block = model.clifford_blocks[0]
        assert block.normalization_type == "zero_centered_rms_norm"
        assert type(block.ctx_norm).__name__ == "ZeroCenteredRMSNorm"

    def test_normalization_type_config_round_trip(self, tiny_config):
        cfg = dict(tiny_config)
        cfg["normalization_type"] = "rms_norm"
        model = CliffordNetLM(**cfg)
        config = model.get_config()
        assert config["normalization_type"] == "rms_norm"
        restored = CliffordNetLM.from_config(config)
        assert restored.normalization_type == "rms_norm"
        assert restored.clifford_blocks[0].normalization_type == "rms_norm"


# ---------------------------------------------------------------------
# TestGlobalContextPeriod — periodic global-context plumbing
# ---------------------------------------------------------------------


@pytest.fixture
def periodic_config():
    """Config with depth=10 so periodicity is testable."""
    return {
        "vocab_size": 256,
        "max_seq_length": 32,
        "channels": 64,
        "depth": 10,
        "shifts": [1, 2],
        "dropout_rate": 0.0,
        "stochastic_depth_rate": 0.0,
    }


class TestGlobalContextPeriod:
    """Verify periodic global-context behavior. Default (None) preserves byte-
    identical BC; positive int forces ``use_global_context=True`` at 1-indexed
    positions n, 2n, ...; ``-1`` is normalized to ``None``; invalid values
    raise; full ``get_config`` / ``from_config`` round-trip preserves the
    per-block pattern.
    """

    def test_default_is_none(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        assert model.global_context_period is None
        # All blocks reflect the model-level flag (default False).
        for block in model.clifford_blocks:
            assert block.use_global_context is False

    def test_periodic_positions_global(self, periodic_config):
        cfg = dict(periodic_config)
        cfg["use_global_context"] = False
        cfg["global_context_period"] = 5
        model = CliffordNetLM(**cfg)
        flags = [b.use_global_context for b in model.clifford_blocks]
        # depth=10, period=5: positions (i+1) % 5 == 0 => i=4, i=9 are True.
        expected = [False, False, False, False, True,
                    False, False, False, False, True]
        assert flags == expected

    def test_non_periodic_blocks_respect_model_flag(self, periodic_config):
        cfg = dict(periodic_config)
        cfg["use_global_context"] = True
        cfg["global_context_period"] = 5
        model = CliffordNetLM(**cfg)
        flags = [b.use_global_context for b in model.clifford_blocks]
        # All True: periodic positions forced True, non-periodic from model flag.
        assert flags == [True] * 10

    def test_period_one_makes_all_global(self):
        model = CliffordNetLM(
            vocab_size=256, max_seq_length=32, channels=64, depth=4,
            stochastic_depth_rate=0.0, use_global_context=False,
            global_context_period=1,
        )
        flags = [b.use_global_context for b in model.clifford_blocks]
        assert flags == [True, True, True, True]

    def test_period_greater_than_depth(self):
        model = CliffordNetLM(
            vocab_size=256, max_seq_length=32, channels=64, depth=4,
            stochastic_depth_rate=0.0, use_global_context=False,
            global_context_period=99,
        )
        flags = [b.use_global_context for b in model.clifford_blocks]
        assert flags == [False, False, False, False]

    def test_sentinel_minus_one_disables(self, periodic_config):
        cfg = dict(periodic_config)
        cfg["use_global_context"] = False
        cfg["global_context_period"] = -1
        model = CliffordNetLM(**cfg)
        assert model.global_context_period is None
        for block in model.clifford_blocks:
            assert block.use_global_context is False

    def test_invalid_period_raises(self, periodic_config):
        cfg = dict(periodic_config)
        for bad in (0, -2, -100, 1.5, "5"):
            cfg_bad = dict(cfg)
            cfg_bad["global_context_period"] = bad
            with pytest.raises(ValueError):
                CliffordNetLM(**cfg_bad)

    def test_config_round_trip(self, periodic_config):
        cfg = dict(periodic_config)
        cfg["use_global_context"] = False
        cfg["global_context_period"] = 5
        model = CliffordNetLM(**cfg)
        config = model.get_config()
        assert config["global_context_period"] == 5
        restored = CliffordNetLM.from_config(config)
        assert restored.global_context_period == 5
        flags = [b.use_global_context for b in restored.clifford_blocks]
        expected = [False, False, False, False, True,
                    False, False, False, False, True]
        assert flags == expected

    def test_default_round_trip_preserves_none(self, tiny_config):
        model = CliffordNetLM(**tiny_config)
        config = model.get_config()
        assert config["global_context_period"] is None
        restored = CliffordNetLM.from_config(config)
        assert restored.global_context_period is None
