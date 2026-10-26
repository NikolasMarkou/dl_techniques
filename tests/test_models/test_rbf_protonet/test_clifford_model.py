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

from unittest import mock

import keras
import numpy as np
import pytest

from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.models.vision.rbf_protonet.model import (
    CliffordRBFProtoNet,
    create_clifford_rbf_protonet,
)

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
)


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


# ---------------------------------------------------------------------
# The D-005 smoke, kept as a standing test
# ---------------------------------------------------------------------

# ``rbf_protonet/model.py`` carries ``# DECISION plan-2026-09-16T064227-b83f88ad/D-005``:
# the stem composition was "validated by a standalone smoke test" and may not
# be swapped without re-running it. That smoke was a one-off probe. This class
# is the same probe as a test that can fail: gradient liveness through the stem
# and every downstream block, NaN/Inf-clean output, and the trainable-weight
# count. D-005 recorded 160 trainable weights (4 stem + 13 x 12 block) for the
# DEFAULT config (channels 128, depth 12, shifts (1, 2), patch_size 2, no
# global context, 32x32x3); the default-config test below is that config.
#
# ``drop_path_rate=0.0`` throughout this class, on purpose: ``StochasticDepth``
# drops a block's residual branch PER SAMPLE under ``training=True``, so with
# the default 0.1 and a batch of 2 a whole block is dropped on a few percent of
# draws and its weights then receive an exactly-zero gradient (measured: 4 of
# 40 draws of the default config, blocks 8-11). That is the layer working, not
# a dead component, and it made this guard flaky until the rate was pinned.
_RECORDED_STEM_AND_BLOCK_WEIGHTS = 160
_STEM_WEIGHTS = 4  # kernel, bias, BN gamma, BN beta


def _standalone_trainable_counts(model: CliffordRBFProtoNet) -> dict:
    """Trainable-weight counts of layers built STANDALONE from ``model``'s
    constructor arguments, so the expected total is not read back from the
    model's own sub-layers."""
    side = -(-model.input_height // model.patch_size)
    conv = keras.layers.Conv2D(
        model.channels, 3, strides=model.patch_size, padding="same", use_bias=True
    )
    norm = keras.layers.BatchNormalization(momentum=0.9)
    feature_map = conv(np.zeros((1, model.input_height, model.input_width, 3), "float32"))
    norm(feature_map)
    block = CliffordNetBlock(
        channels=model.channels,
        shifts=list(model.shifts),
        use_global_context=model.use_global_context,
    )
    block(np.zeros((1, side, side, model.channels), "float32"))
    return {
        "stem": len(conv.trainable_weights) + len(norm.trainable_weights),
        "block": len(block.trainable_weights),
    }


def _assert_the_d005_smoke(model: CliffordRBFProtoNet, x: np.ndarray) -> dict:
    """One live step: output shape/finiteness, derived weight count, and a
    finite non-zero gradient on every trainable weight."""
    outputs = keras.ops.convert_to_numpy(model(x, training=True))
    assert outputs.shape == (x.shape[0], model.num_classes)
    assert np.isfinite(outputs).all(), "output holds NaN/Inf"

    standalone = _standalone_trainable_counts(model)
    assert standalone["stem"] == _STEM_WEIGHTS
    expected = (
        standalone["stem"]
        + model.depth * standalone["block"]
        + (len(model.feature_proj.trainable_weights) if model.feature_proj else 0)
        + len(model.rbf_head.trainable_weights)
    )
    assert len(model.trainable_weights) == expected, (
        f"{len(model.trainable_weights)} trainable weights, structure says {expected}"
    )

    report = assert_gradients_reach_every_trainable_weight(model, x)
    assert len(report) == expected
    stem_paths = {w.path for w in model.stem.trainable_weights}
    assert len(stem_paths) == _STEM_WEIGHTS
    assert stem_paths <= set(report), "the stem weights are not in the gradient report"
    return standalone


class TestCliffordRBFProtoNetD005StandingSmoke:
    def test_the_recorded_default_config_has_160_stem_and_block_weights_all_live(self):
        model = CliffordRBFProtoNet(drop_path_rate=0.0)
        x = np.random.default_rng(1).random((2, 32, 32, 3)).astype("float32")
        standalone = _assert_the_d005_smoke(model, x)
        recorded = len(model.stem.trainable_weights) + sum(
            len(block.trainable_weights) for block in model.blocks
        )
        assert recorded == _RECORDED_STEM_AND_BLOCK_WEIGHTS
        assert (
            standalone["stem"] + model.depth * standalone["block"]
            == _RECORDED_STEM_AND_BLOCK_WEIGHTS
        )

    @pytest.mark.parametrize("patch_size", [1, 2, 4])
    def test_a_small_config_is_live_and_counted_at_every_patch_size(self, patch_size):
        model = _small_clifford_rbf_protonet(patch_size=patch_size, drop_path_rate=0.0)
        _assert_the_d005_smoke(model, _images(2))

    def test_the_smoke_sees_a_stem_whose_gradient_is_dead(self):
        """Positive control: a stem output cut from the graph is reported by
        name, so the liveness check above can fail."""
        model = _small_clifford_rbf_protonet(drop_path_rate=0.0)
        x = _images(2)
        model(x, training=True)
        stem_paths = {w.path for w in model.stem.trainable_weights}
        original_call = model.stem.call

        def _detached(*args, **kwargs):
            return keras.ops.stop_gradient(original_call(*args, **kwargs))

        with mock.patch.object(model.stem, "call", _detached):
            report = gradient_report(model, x)
            assert {p for p, g in report.items() if g is None} == stem_paths
            assert all(g for p, g in report.items() if p not in stem_paths)
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, x)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-vvv"]))
