"""
Model-level tests for ``CliffordRBFProtoNet`` (construction, forward pass,
``pretrained=True`` refusal, serialization round-trip, and gradient flow),
mirroring ``tests/test_models/test_rbf_protonet/test_model.py``'s conventions
for the sibling ``RBFProtoNet`` class.

``CliffordRBFProtoNet`` composes an isotropic Clifford-algebra backbone
(``CliffordNetBlock``) with the same RBF prototype-classification head design
used by ``RBFProtoNet`` (``output_mode='normalized'``, see D-002 in this
plan's ``decisions.md``). The head's defining contract is that its output is
already a per-class probability vector -- each row sums to 1.0 -- so that is
asserted here directly.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.rbf_protonet.model import (
    CliffordRBFProtoNet,
    create_clifford_rbf_protonet,
)

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _small_clifford_rbf_protonet(**overrides):
    """A CIFAR-shaped CliffordRBFProtoNet with small num_classes/depth for
    fast CPU tests (depth reduced from the production default of 12 to 4)."""
    cfg = dict(
        input_shape=(32, 32, 3),
        num_classes=10,
        depth=4,
    )
    cfg.update(overrides)
    return CliffordRBFProtoNet(**cfg)


def _images(batch: int = 4) -> np.ndarray:
    return np.random.default_rng(0).random((batch, 32, 32, 3)).astype("float32")


# ---------------------------------------------------------------------
# Construction + forward pass
# ---------------------------------------------------------------------


class TestCliffordRBFProtoNetConstruction:
    def test_constructs_without_error(self):
        model = _small_clifford_rbf_protonet()
        assert isinstance(model, CliffordRBFProtoNet)
        assert model.num_classes == 10
        assert model.depth == 4

    def test_forward_pass_shape(self):
        model = _small_clifford_rbf_protonet()
        x = _images(4)
        out = model(x, training=False)
        assert tuple(out.shape) == (4, 10)

    def test_forward_pass_no_nan(self):
        model = _small_clifford_rbf_protonet()
        out = model(_images(4), training=False).numpy()
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_output_rows_sum_to_one(self):
        """``output_mode='normalized'`` contract: every row is a probability
        distribution over `num_classes` prototypes."""
        model = _small_clifford_rbf_protonet()
        out = keras.ops.convert_to_numpy(model(_images(8), training=False))
        np.testing.assert_allclose(
            out.sum(axis=-1), np.ones(8), atol=1e-4, rtol=0,
            err_msg="CliffordRBFProtoNet output rows must sum to 1.0 under output_mode='normalized'",
        )

    def test_create_factory_matches_direct_construction(self):
        model = create_clifford_rbf_protonet(num_classes=10, input_shape=(32, 32, 3), depth=4)
        out = model(_images(2), training=False)
        assert tuple(out.shape) == (2, 10)


# ---------------------------------------------------------------------
# pretrained=True refusal
# ---------------------------------------------------------------------


class TestCliffordRBFProtoNetPretrainedRefusal:
    def test_pretrained_true_raises_not_implemented_on_class(self):
        with pytest.raises(NotImplementedError):
            CliffordRBFProtoNet(input_shape=(32, 32, 3), num_classes=10, depth=4, pretrained=True)

    def test_pretrained_true_raises_not_implemented_on_factory(self):
        with pytest.raises(NotImplementedError):
            create_clifford_rbf_protonet(
                num_classes=10, input_shape=(32, 32, 3), depth=4, pretrained=True,
            )


# ---------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------


class TestCliffordRBFProtoNetSerialization:
    def test_get_config_from_config_round_trip(self):
        model = _small_clifford_rbf_protonet()
        config = model.get_config()
        restored = CliffordRBFProtoNet.from_config(config)
        assert restored.get_config() == config

    def test_keras_save_load_round_trip(self, tmp_path):
        model = _small_clifford_rbf_protonet()
        x = _images(4)
        original_output = keras.ops.convert_to_numpy(model(x, training=False))

        save_path = tmp_path / "clifford_rbf_protonet.keras"
        model.save(save_path)
        loaded = keras.models.load_model(save_path)

        loaded_output = keras.ops.convert_to_numpy(loaded(x, training=False))
        np.testing.assert_allclose(
            loaded_output, original_output, atol=1e-5, rtol=0,
            err_msg="CliffordRBFProtoNet output must match exactly after a .keras save/load round trip",
        )


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------


class TestCliffordRBFProtoNetGradientFlow:
    def test_gradients_reach_every_trainable_weight(self):
        model = _small_clifford_rbf_protonet()
        x = _images(4)

        # Build the model and take one real optimizer step first, mirroring
        # the oracle's own documented usage (a built model with a live
        # optimizer state), before handing it to the oracle for the
        # per-weight gradient-reach assertion.
        import tensorflow as tf

        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        with tf.GradientTape() as tape:
            outputs = model(x, training=True)
            loss = keras.ops.mean(keras.ops.square(outputs))
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        assert_gradients_reach_every_trainable_weight(model, x)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-vvv"]))
