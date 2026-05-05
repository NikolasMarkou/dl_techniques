"""Tests for the CliffordNetLMRouting causal language model.

Mirrors test_cliffordnet_lm.py and adds routing-specific contract checks:
sum-to-1 output, [eps, 1-eps] range, save/load roundtrip, gradient flow on
the routing kernel (trainable mode), loss compatibility with
``from_logits=False``, and a realistic-vocab (50261 -> padded 65536) shape
check.
"""

import os
import tempfile

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.models.cliffordnet.lm_routing import CliffordNetLMRouting
from dl_techniques.layers.activations.routing_probabilities import (
    RoutingProbabilitiesLayer,
)
from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
)
from dl_techniques.losses import MaskedCausalLMLoss


def _random_ids(shape, vocab_size):
    """Generate random integer token IDs."""
    return np.random.randint(0, vocab_size, shape).astype(np.int32)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    """Minimal CliffordNetLMRouting config for fast testing (vocab=128)."""
    return {
        "vocab_size": 128,
        "max_seq_length": 32,
        "channels": 64,
        "depth": 2,
        "shifts": [1, 2],
        "dropout_rate": 0.0,
        "stochastic_depth_rate": 0.0,
    }


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMRoutingInitialization:

    def test_init_trainable_mode(self, tiny_config):
        model = CliffordNetLMRouting(routing_mode="trainable", **tiny_config)
        assert isinstance(model, CliffordNetLMRouting)
        assert model.routing_mode == "trainable"
        assert isinstance(model.output_routing, RoutingProbabilitiesLayer)
        assert model.output_routing.mode == "trainable"

    def test_init_deterministic_mode(self, tiny_config):
        model = CliffordNetLMRouting(
            routing_mode="deterministic", **tiny_config,
        )
        assert model.routing_mode == "deterministic"
        assert model.output_routing.mode == "deterministic"

    def test_init_invalid_mode_raises(self, tiny_config):
        with pytest.raises(ValueError, match="routing_mode"):
            CliffordNetLMRouting(routing_mode="bogus", **tiny_config)

    def test_default_routing_mode_is_trainable(self, tiny_config):
        model = CliffordNetLMRouting(**tiny_config)
        assert model.routing_mode == "trainable"


# ---------------------------------------------------------------------
# Forward Pass Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMRoutingForward:

    @pytest.mark.parametrize("mode", ["trainable", "deterministic"])
    def test_forward_shape(self, tiny_config, mode):
        model = CliffordNetLMRouting(routing_mode=mode, **tiny_config)
        ids = _random_ids((2, 16), tiny_config["vocab_size"])
        out = model(ids, training=False)
        assert "logits" in out
        assert out["logits"].shape == (2, 16, tiny_config["vocab_size"])

    @pytest.mark.parametrize("mode", ["trainable", "deterministic"])
    def test_output_sums_to_one(self, tiny_config, mode):
        model = CliffordNetLMRouting(routing_mode=mode, **tiny_config)
        ids = _random_ids((2, 16), tiny_config["vocab_size"])
        probs = keras.ops.convert_to_numpy(model(ids, training=False)["logits"])
        sums = probs.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4)

    @pytest.mark.parametrize("mode", ["trainable", "deterministic"])
    def test_output_in_unit_range(self, tiny_config, mode):
        model = CliffordNetLMRouting(routing_mode=mode, **tiny_config)
        ids = _random_ids((2, 16), tiny_config["vocab_size"])
        probs = keras.ops.convert_to_numpy(model(ids, training=False)["logits"])
        assert np.all(probs > 0.0)
        assert np.all(probs < 1.0)


# ---------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMRoutingSerialization:

    @pytest.mark.parametrize("mode", ["trainable", "deterministic"])
    def test_get_config_roundtrip(self, tiny_config, mode):
        model = CliffordNetLMRouting(routing_mode=mode, **tiny_config)
        config = model.get_config()
        assert config["routing_mode"] == mode
        assert config["vocab_size"] == tiny_config["vocab_size"]
        model2 = CliffordNetLMRouting.from_config(config)
        assert model2.routing_mode == mode
        assert model2.vocab_size == model.vocab_size
        assert model2.channels == model.channels

    @pytest.mark.parametrize("mode", ["trainable", "deterministic"])
    def test_save_load_roundtrip(self, tiny_config, mode):
        model = CliffordNetLMRouting(routing_mode=mode, **tiny_config)
        ids = _random_ids((1, 8), tiny_config["vocab_size"])
        out_before = keras.ops.convert_to_numpy(
            model(ids, training=False)["logits"]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.keras")
            model.save(path)
            loaded = keras.models.load_model(
                path,
                custom_objects={
                    "CliffordNetLMRouting": CliffordNetLMRouting,
                    "RoutingProbabilitiesLayer": RoutingProbabilitiesLayer,
                    "CausalCliffordNetBlock": CausalCliffordNetBlock,
                },
            )

        out_after = keras.ops.convert_to_numpy(
            loaded(ids, training=False)["logits"]
        )
        np.testing.assert_allclose(out_before, out_after, atol=1e-5)
        assert loaded.routing_mode == mode

    def test_compute_output_shape(self, tiny_config):
        model = CliffordNetLMRouting(**tiny_config)
        shapes = model.compute_output_shape((None, 16))
        assert shapes["logits"] == (None, 16, tiny_config["vocab_size"])


# ---------------------------------------------------------------------
# Variant Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMRoutingVariants:

    def test_from_variant_nano(self):
        model = CliffordNetLMRouting.from_variant(
            "nano", vocab_size=256, max_seq_length=32,
        )
        assert model.channels == 128
        assert model.depth == 12
        assert model.routing_mode == "trainable"

    def test_from_variant_forwards_routing_mode(self):
        model = CliffordNetLMRouting.from_variant(
            "nano", vocab_size=256, max_seq_length=32,
            routing_mode="deterministic",
        )
        assert model.routing_mode == "deterministic"

    def test_from_variant_unknown(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            CliffordNetLMRouting.from_variant("nonexistent", vocab_size=256)


# ---------------------------------------------------------------------
# Gradient Flow Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMRoutingGradientFlow:

    def test_gradient_flow_trainable(self, tiny_config):
        """Routing kernel must receive non-zero gradient in trainable mode."""
        model = CliffordNetLMRouting(routing_mode="trainable", **tiny_config)
        ids = _random_ids((2, 8), tiny_config["vocab_size"])
        labels = _random_ids((2, 8), tiny_config["vocab_size"])

        with tf.GradientTape() as tape:
            probs = model(ids, training=True)["logits"]
            # cross-entropy on probabilities
            loss = keras.losses.sparse_categorical_crossentropy(
                labels, probs, from_logits=False,
            )
            loss = keras.ops.mean(loss)

        # The routing layer's `kernel` attribute is the trainable projection.
        routing_kernel = model.output_routing.kernel
        assert routing_kernel is not None, "routing kernel not built"
        # Find by identity in trainable_variables
        kernel_grad = None
        for v, g in zip(model.trainable_variables, tape.gradient(
            loss, model.trainable_variables,
        )):
            if v is routing_kernel:
                kernel_grad = g
                break
        assert kernel_grad is not None, "No gradient on routing kernel"
        kernel_grad_np = keras.ops.convert_to_numpy(kernel_grad)
        assert np.any(kernel_grad_np != 0.0), "Routing kernel gradient is all zero"


# ---------------------------------------------------------------------
# Loss Compatibility Tests
# ---------------------------------------------------------------------


class TestCliffordNetLMRoutingLossCompat:

    def test_loss_compatibility(self, tiny_config):
        """MaskedCausalLMLoss(from_logits=False) yields finite scalar loss."""
        model = CliffordNetLMRouting(routing_mode="trainable", **tiny_config)
        ids = _random_ids((2, 8), tiny_config["vocab_size"])
        labels = _random_ids((2, 8), tiny_config["vocab_size"])

        probs = model(ids, training=False)["logits"]
        loss_fn = MaskedCausalLMLoss(from_logits=False)
        loss_val = loss_fn(labels, probs)
        loss_np = keras.ops.convert_to_numpy(loss_val)
        assert np.isfinite(loss_np).all(), f"Non-finite loss: {loss_np}"
        assert float(loss_np) > 0.0


# ---------------------------------------------------------------------
# Realistic-vocab Padded Dim Test
# ---------------------------------------------------------------------


class TestCliffordNetLMRoutingRealisticVocab:

    def test_realistic_vocab_padded_dim(self):
        """vocab=50261 -> padded=65536 path; small batch+seq forward."""
        model = CliffordNetLMRouting(
            vocab_size=50261,
            max_seq_length=16,
            channels=64,
            depth=2,
            shifts=[1, 2],
            dropout_rate=0.0,
            stochastic_depth_rate=0.0,
            routing_mode="trainable",
        )
        ids = _random_ids((1, 8), 50261)
        out = model(ids, training=False)
        assert out["logits"].shape == (1, 8, 50261)

        probs = keras.ops.convert_to_numpy(out["logits"])
        sums = probs.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-3)

        # Confirm padded_output_dim was set as expected
        assert model.output_routing.padded_output_dim == 65536
        assert model.output_routing.num_decisions == 16


# ---------------------------------------------------------------------
# Input-embedding mode tests
# ---------------------------------------------------------------------


class TestCliffordNetLMRoutingInputEmbeddings:
    """Coverage for input_embedding={hce,albert,dense}.

    HCE is the default — verifies the new wiring doesn't break forward
    pass, gradient flow, save/load, and parameter-count expectations.
    """

    @pytest.mark.parametrize("mode", ["hce", "albert", "dense"])
    def test_forward_per_mode(self, tiny_config, mode):
        model = CliffordNetLMRouting(
            input_embedding=mode, routing_mode="trainable", **tiny_config,
        )
        ids = _random_ids((2, 16), tiny_config["vocab_size"])
        out = model(ids, training=False)
        assert out["logits"].shape == (2, 16, tiny_config["vocab_size"])

    def test_default_is_hce(self, tiny_config):
        model = CliffordNetLMRouting(
            routing_mode="trainable", **tiny_config,
        )
        assert model.input_embedding == "hce"
        # token_embedding sub-layer name reflects HCE choice.
        assert model.token_embedding.name == "token_embedding_hce"

    def test_hce_compresses_at_realistic_vocab(self):
        """At vocab=50261, D=128, K=2: HCE total params << dense baseline."""
        common = dict(
            max_seq_length=16, channels=128, depth=2, shifts=[1, 2],
            dropout_rate=0.0, stochastic_depth_rate=0.0,
            routing_mode="trainable",
        )
        ids = _random_ids((1, 8), 50261)
        m_dense = CliffordNetLMRouting(
            vocab_size=50261, input_embedding="dense", **common,
        )
        m_dense(ids)
        m_hce = CliffordNetLMRouting(
            vocab_size=50261, input_embedding="hce", **common,
        )
        m_hce(ids)
        # HCE model should be at least 10x smaller overall (embedding
        # dominates at small depth/channels).
        assert m_hce.count_params() * 10 < m_dense.count_params(), (
            f"hce={m_hce.count_params():,} vs dense={m_dense.count_params():,}"
        )

    def test_albert_default_bottleneck_compresses(self):
        """ALBERT with default k = max(8, min(channels//2, 128)) compresses."""
        common = dict(
            vocab_size=50261, max_seq_length=16, channels=128, depth=2,
            shifts=[1, 2], dropout_rate=0.0, stochastic_depth_rate=0.0,
            routing_mode="trainable",
        )
        ids = _random_ids((1, 8), 50261)
        m_albert = CliffordNetLMRouting(input_embedding="albert", **common)
        m_albert(ids)
        m_dense = CliffordNetLMRouting(input_embedding="dense", **common)
        m_dense(ids)
        assert m_albert.count_params() < m_dense.count_params()

    @pytest.mark.parametrize("mode", ["hce", "albert", "dense"])
    def test_serialization_roundtrip(self, tiny_config, mode, tmp_path):
        model = CliffordNetLMRouting(
            input_embedding=mode, routing_mode="trainable", **tiny_config,
        )
        ids = _random_ids((1, 8), tiny_config["vocab_size"])
        model(ids)  # build
        original = keras.ops.convert_to_numpy(model(ids)["logits"])

        path = str(tmp_path / f"lmrouting_{mode}.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        loaded_out = keras.ops.convert_to_numpy(loaded(ids)["logits"])
        np.testing.assert_allclose(original, loaded_out, atol=1e-5)
        assert loaded.input_embedding == mode

    @pytest.mark.parametrize("mode", ["hce", "albert", "dense"])
    def test_gradient_flow_per_mode(self, tiny_config, mode):
        model = CliffordNetLMRouting(
            input_embedding=mode, routing_mode="trainable", **tiny_config,
        )
        ids = tf.constant(_random_ids((1, 8), tiny_config["vocab_size"]))
        with tf.GradientTape() as tape:
            out = model(ids, training=True)
            loss = tf.reduce_mean(out["logits"])
        grads = tape.gradient(loss, model.trainable_variables)
        # All trainable vars receive non-None gradients.
        assert all(g is not None for g in grads)

    def test_invalid_input_embedding_raises(self, tiny_config):
        with pytest.raises(ValueError, match="input_embedding"):
            CliffordNetLMRouting(input_embedding="garbage", **tiny_config)
