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
