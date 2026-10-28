"""
Model-level tests for ``CliffordRBFProtoNet`` (construction, arguments, layer names,
forward pass, ``pretrained`` refusal, serialization round-trip, and gradient flow),
mirroring ``tests/test_models/test_rbf_protonet/test_model.py``'s conventions
for the sibling ``RBFProtoNet`` class.

``CliffordRBFProtoNet`` composes an isotropic Clifford-algebra backbone
(``CliffordNetBlock``) with the same RBF prototype-classification head design
used by ``RBFProtoNet`` (``output_mode='normalized'``). The head's defining
contract is that its output is already a per-class probability vector -- each row
sums to 1.0 -- so that is asserted here directly.
"""

import inspect
import json
import re
import math
from unittest import mock

import keras
import numpy as np
import pytest

from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.models.vision.rbf_protonet.model import (
    WARM_START_HINT,
    CliffordRBFProtoNet,
    create_clifford_rbf_protonet,
)

from dl_techniques.utils.weight_transfer import refuse_pretrained

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


_IMAGE_SIDE = 32


def _small_clifford_rbf_protonet(**overrides):
    """A CliffordRBFProtoNet with small num_classes/depth for fast CPU tests
    (depth reduced from the production default of 12 to 4)."""
    cfg = dict(
        num_classes=10,
        depth=4,
    )
    cfg.update(overrides)
    return CliffordRBFProtoNet(**cfg)


def _images(batch: int = 4, height: int = _IMAGE_SIDE, width: int = _IMAGE_SIDE) -> np.ndarray:
    return np.random.default_rng(0).random((batch, height, width, 3)).astype("float32")


def _refusal(label: str, pretrained) -> str:
    """The message ``refuse_pretrained`` raises for this pair: the shared text and the pair's hint."""
    with pytest.raises(NotImplementedError) as raised:
        refuse_pretrained(pretrained, label, WARM_START_HINT)
    return str(raised.value)


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
        model = create_clifford_rbf_protonet(num_classes=10, depth=4)
        out = model(_images(2), training=False)
        assert tuple(out.shape) == (2, 10)

    @pytest.mark.parametrize("height, width", [(16, 16), (33, 20), (48, 64)])
    def test_the_model_runs_at_any_spatial_size(self, height, width):
        """The blocks are isotropic and the pooling is global, which is why the model
        takes no ``input_shape``: nothing in it would read one."""
        out = _small_clifford_rbf_protonet(depth=2)(_images(2, height, width), training=False)
        assert tuple(out.shape) == (2, 10)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------


class TestCliffordRBFProtoNetArguments:
    def test_num_classes_is_the_first_parameter(self):
        names = list(inspect.signature(CliffordRBFProtoNet.__init__).parameters)
        assert names[1] == "num_classes"
        assert names[-1] == "kwargs"

    def test_the_constructor_has_no_pretrained_or_input_shape_knob(self):
        names = inspect.signature(CliffordRBFProtoNet.__init__).parameters
        assert "pretrained" not in names and "input_shape" not in names
        config = _small_clifford_rbf_protonet().get_config()
        assert "pretrained" not in config and "input_shape" not in config

    def test_a_pretrained_keyword_on_the_class_is_refused_by_keras(self):
        with pytest.raises(ValueError, match="pretrained"):
            CliffordRBFProtoNet(num_classes=10, depth=4, pretrained=True)

    @pytest.mark.parametrize("make", [np.int64, np.int32, np.uint8])
    def test_a_numpy_integer_num_classes_is_accepted_and_stored_as_an_int(self, make):
        model = _small_clifford_rbf_protonet(num_classes=make(10), depth=1)
        assert type(model.num_classes) is int and model.num_classes == 10
        assert tuple(model(_images(2), training=False).shape) == (2, 10)
        json.dumps(model.get_config())  # an np.int64 in the config is not JSON-safe

    @pytest.mark.parametrize("value", [0, -3, 2.5, True, "10", None])
    def test_a_bad_num_classes_raises_naming_the_class_the_argument_and_the_value(self, value):
        """The one positive-integer rule (``utils.validation.positive_int``) with this class as owner."""
        with pytest.raises(
                ValueError,
                match=f"^CliffordRBFProtoNet: num_classes must be a positive integer, got .*{re.escape(str(value))}"):
            _small_clifford_rbf_protonet(num_classes=value)

    @pytest.mark.parametrize("keyword, value", [("input_shape", (32, 32, 3)), ("input_dim", 3)])
    def test_a_shape_keyword_is_refused_by_name(self, keyword, value):
        """Keras pops ``input_shape`` and ``input_dim`` and only warns; a caller written against
        the signature this class had before they were removed must not keep working silently."""
        with pytest.raises(ValueError, match=f"^CliffordRBFProtoNet takes no {keyword} .*any spatial size.*{keyword}="):
            CliffordRBFProtoNet(num_classes=10, depth=4, **{keyword: value})

    def test_an_unknown_keyword_is_refused_naming_it(self):
        with pytest.raises(ValueError, match="not_an_argument"):
            CliffordRBFProtoNet(num_classes=10, depth=4, not_an_argument=1)

    @pytest.mark.parametrize("rate", [1.0, 1.5, -0.1, float("nan")])
    def test_a_drop_path_rate_outside_the_unit_interval_raises_naming_the_value(self, rate):
        with pytest.raises(ValueError, match=f"^drop_path_rate must be in \\[0, 1\\), got {re.escape(repr(rate))}$"):
            _small_clifford_rbf_protonet(drop_path_rate=rate)

    def test_a_regularizer_given_by_name_is_resolved_serialized_and_rebuilt(self):
        model = _small_clifford_rbf_protonet(depth=1, feature_dim=24, kernel_regularizer="l2")
        assert isinstance(model.kernel_regularizer, keras.regularizers.L2)
        config = model.get_config()
        json.dumps(config)
        rebuilt = CliffordRBFProtoNet.from_config(config)
        assert isinstance(rebuilt.kernel_regularizer, keras.regularizers.L2)
        assert rebuilt.get_config() == config
        plain = _small_clifford_rbf_protonet(depth=1, feature_dim=24)
        x = _images(2)
        model(x)
        rebuilt(x)
        plain(x)
        assert len(model.losses) == len(rebuilt.losses) > len(plain.losses)

    def test_the_regularizer_reaches_the_stem_and_the_feature_projection(self):
        """``feature_proj`` exists only when ``channels`` differs from ``feature_dim``."""
        model = _small_clifford_rbf_protonet(depth=1, feature_dim=24, kernel_regularizer="l2")
        assert model.feature_proj is not None
        assert model.feature_proj.kernel_regularizer is not None
        assert model.stem.conv1.kernel_regularizer is not None


# ---------------------------------------------------------------------
# Layer names
# ---------------------------------------------------------------------


class TestCliffordRBFProtoNetLayerNames:
    def test_blocks_and_their_drop_paths_are_named_by_the_naming_table(self):
        model = _small_clifford_rbf_protonet(depth=3)
        assert [block.name for block in model.blocks] == ["block_0", "block_1", "block_2"]
        assert [path.name for path in model.drop_paths] == [
            "block_0_drop_path", "block_1_drop_path", "block_2_drop_path"]

    def test_the_block_weights_sit_under_the_block_name(self):
        model = _small_clifford_rbf_protonet(depth=2)
        model(_images(2))
        paths = [w.path for w in model.weights]
        assert any("/block_1/" in p for p in paths)
        assert not any("clifford_block_" in p for p in paths)


# ---------------------------------------------------------------------
# pretrained refusal
# ---------------------------------------------------------------------


class TestCliffordRBFProtoNetPretrainedRefusal:
    @pytest.mark.parametrize("pretrained", [True, "weights.keras"])
    def test_the_factory_refuses_with_the_one_shared_message(self, pretrained):
        with pytest.raises(NotImplementedError) as raised:
            create_clifford_rbf_protonet(num_classes=10, depth=4, pretrained=pretrained)
        assert str(raised.value) == _refusal("CliffordRBFProtoNet", pretrained)

    def test_the_message_gives_the_advice_that_is_true_for_this_model(self):
        message = _refusal("CliffordRBFProtoNet", "weights.keras")
        assert "model.load_weights(path)" in message
        assert "backbone/head split" not in message and "load_model" not in message

    @pytest.mark.parametrize("keyword, value", [("input_shape", (28, 28, 1)), ("input_dim", 3)])
    def test_the_factory_refuses_a_shape_keyword_by_name(self, keyword, value):
        """``create_clifford_rbf_protonet(input_shape=...)`` was the documented signature until it
        was removed; Keras alone only warns about it."""
        with pytest.raises(ValueError, match=f"CliffordRBFProtoNet takes no {keyword}"):
            create_clifford_rbf_protonet(num_classes=10, depth=4, **{keyword: value})

    def test_the_factory_refuses_an_unknown_keyword_naming_it(self):
        with pytest.raises(ValueError, match="not_an_argument"):
            create_clifford_rbf_protonet(num_classes=10, depth=4, not_an_argument=1)

    def test_the_refusal_comes_before_anything_is_built(self):
        """An invalid ``num_classes`` would raise ``ValueError`` if the model were built first."""
        with pytest.raises(NotImplementedError):
            create_clifford_rbf_protonet(num_classes=-1, pretrained=True)

    def test_the_default_is_a_random_initialization(self):
        model = create_clifford_rbf_protonet(num_classes=10, depth=1, pretrained=False)
        assert isinstance(model, CliffordRBFProtoNet)


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
    side = -(-_IMAGE_SIDE // model.patch_size)
    conv = keras.layers.Conv2D(
        model.channels, 3, strides=model.patch_size, padding="same", use_bias=True
    )
    norm = keras.layers.BatchNormalization(momentum=0.9)
    feature_map = conv(np.zeros((1, _IMAGE_SIDE, _IMAGE_SIDE, 3), "float32"))
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


# The stem's ``Conv2D`` bias feeds a ``BatchNormalization`` that is in TRAINING
# mode (batch statistics). The batch mean subtracts the bias again, so the true
# gradient of the loss with respect to that bias is identically zero; what a
# ``training=True`` tape returns for it is float32 rounding residue (measured
# 2026-10-26 over three seeds: max|grad| 5.2e-11 .. 6.8e-11, against 4.0e-06 ..
# 6.8e-06 for the kernel; 6.0e-13 against 4.3e-08 on the default config). A
# ``> 0.0`` assertion on it passes on that residue and would fail the day a
# device rounds it to exactly 0.0, so the bias is excluded from the
# ``training=True`` arm and asserted where its gradient is real: with
# ``training=False`` the BN normalizes with its moving statistics, the bias no
# longer cancels, and its gradient is of the kernel's order (measured 4.4e-05 ..
# 6.1e-05 against 2.2e-05 .. 3.2e-05). The relative floor below separates the
# two cases by about five orders of magnitude (measured ratio 1.9 .. 2.0 for a
# live bias, 1e-5 for the residue); the oracle's exact-zero floor cannot, which
# is why the oracle alone is not used for this one weight.
_LIVE_BIAS_MIN_RATIO_TO_KERNEL = 1e-3


def _stem_bias_and_kernel_paths(model: CliffordRBFProtoNet) -> tuple:
    """Return ``(bias_path, kernel_path)`` of the stem convolution, each found
    exactly once among the stem's trainable weights (a stem built without a
    bias fails here, by name, instead of passing on a smaller weight set)."""
    paths = [w.path for w in model.stem.trainable_weights]
    biases = [p for p in paths if p.endswith("stem_conv1/bias")]
    kernels = [p for p in paths if p.endswith("stem_conv1/kernel")]
    assert len(biases) == 1, f"expected exactly one stem conv bias, got {paths}"
    assert len(kernels) == 1, f"expected exactly one stem conv kernel, got {paths}"
    return biases[0], kernels[0]


def _assert_the_stem_bias_is_live_in_inference(
    model: CliffordRBFProtoNet, x: np.ndarray
) -> dict:
    """The stem conv bias receives a REAL gradient with ``training=False``.

    Real means: of the kernel's order (at least
    ``_LIVE_BIAS_MIN_RATIO_TO_KERNEL`` of it), not merely above 0.0, so rounding
    residue cannot satisfy it. Returns the ``training=False`` report."""
    bias_path, kernel_path = _stem_bias_and_kernel_paths(model)
    report = assert_gradients_reach_every_trainable_weight(model, x, training=False)
    assert report[bias_path] >= _LIVE_BIAS_MIN_RATIO_TO_KERNEL * report[kernel_path], (
        f"stem conv bias max|grad|={report[bias_path]:.3e} is not of the kernel's "
        f"order (kernel {report[kernel_path]:.3e}) with training=False: it is "
        "rounding residue, not a gradient"
    )
    return report


def _assert_the_d005_smoke(model: CliffordRBFProtoNet, x: np.ndarray) -> dict:
    """One live step: output shape/finiteness, derived weight count, and a
    finite non-zero gradient on every trainable weight.

    Two arms. ``training=True``: every trainable weight EXCEPT the stem conv bias
    must be on the graph, finite and not identically zero. The bias is left out
    of that arm because a training-mode BatchNorm cancels it (its true gradient
    is zero and the number the tape returns is float32 rounding residue, see the
    comment above ``_LIVE_BIAS_MIN_RATIO_TO_KERNEL``); it is checked to be on
    the graph and finite, and to be negligible next to its own inference
    gradient, which documents why it is excluded. ``training=False``: the BN uses
    its moving statistics, so EVERY weight, the bias included, must carry a real
    gradient (:func:`_assert_the_stem_bias_is_live_in_inference`)."""
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

    bias_path, _ = _stem_bias_and_kernel_paths(model)
    report = gradient_report(model, x, training=True)
    assert len(report) == expected
    stem_paths = {w.path for w in model.stem.trainable_weights}
    assert len(stem_paths) == _STEM_WEIGHTS
    assert stem_paths <= set(report), "the stem weights are not in the gradient report"
    disconnected = [p for p, g in report.items() if g is None]
    assert not disconnected, f"weights received NO gradient: {disconnected}"
    non_finite = [p for p, g in report.items() if not math.isfinite(g)]
    assert not non_finite, f"weights received a non-finite gradient: {non_finite}"
    dead = [p for p, g in report.items() if p != bias_path and g == 0.0]
    assert not dead, f"weights received an identically-zero gradient: {dead}"

    inference = _assert_the_stem_bias_is_live_in_inference(model, x)
    assert set(inference) == set(report)
    assert report[bias_path] <= _LIVE_BIAS_MIN_RATIO_TO_KERNEL * inference[bias_path], (
        f"the stem conv bias now carries a real gradient with training=True "
        f"({report[bias_path]:.3e} vs {inference[bias_path]:.3e} with "
        "training=False): the BN no longer cancels it, so it belongs in the "
        "training=True arm and the exclusion above is stale"
    )
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


    def test_the_smoke_sees_a_bias_that_a_training_mode_batch_norm_cancels(self):
        """Positive control for the ``training=False`` arm: a stem whose BN runs
        on batch statistics even at inference (the stem is called with
        ``training=True`` whatever the caller says) cancels the bias exactly as
        in training. The oracle's ``> 0.0`` floor still passes on the rounding
        residue that is left; the relative floor must not."""
        model = _small_clifford_rbf_protonet(drop_path_rate=0.0)
        x = _images(2)
        model(x, training=True)
        bias_path, kernel_path = _stem_bias_and_kernel_paths(model)
        live = _assert_the_stem_bias_is_live_in_inference(model, x)
        assert live[bias_path] >= _LIVE_BIAS_MIN_RATIO_TO_KERNEL * live[kernel_path]
        original_call = model.stem.call

        def _always_training(inputs, training=None, **kwargs):
            return original_call(inputs, training=True, **kwargs)

        with mock.patch.object(model.stem, "call", _always_training):
            with pytest.raises(AssertionError, match="rounding residue"):
                _assert_the_stem_bias_is_live_in_inference(model, x)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-vvv"]))
