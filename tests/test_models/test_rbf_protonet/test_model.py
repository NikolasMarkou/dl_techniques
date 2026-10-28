"""
Model-level tests for ``RBFProtoNet`` (construction, arguments, forward pass, and
the ``pretrained`` refusal contract), following ``tests/test_models/test_resnet/``
conventions.

``RBFProtoNet`` composes a small CIFAR-style CNN backbone with an RBF
prototype-classification head (``output_mode='normalized'``). The head's defining
contract is that its output is already a per-class probability vector -- each row
sums to 1.0 -- not a logits vector, so that is asserted here directly rather than
left to inference from the model's docstring.
"""

import inspect
import json
import re

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.rbf_protonet.model import (
    WARM_START_HINT,
    RBFProtoNet,
    create_rbf_protonet,
)
from dl_techniques.utils.weight_transfer import refuse_pretrained


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _small_rbf_protonet(**overrides):
    """An RBFProtoNet with a small num_classes for fast CPU tests."""
    cfg = dict(num_classes=10)
    cfg.update(overrides)
    return RBFProtoNet(**cfg)


def _images(batch: int = 4, height: int = 32, width: int = 32) -> np.ndarray:
    return np.random.default_rng(0).random((batch, height, width, 3)).astype("float32")


def _refusal(label: str, pretrained) -> str:
    """The message ``refuse_pretrained`` raises for this pair: the shared text and the pair's hint."""
    with pytest.raises(NotImplementedError) as raised:
        refuse_pretrained(pretrained, label, WARM_START_HINT)
    return str(raised.value)


# ---------------------------------------------------------------------
# Construction + forward pass
# ---------------------------------------------------------------------


class TestRBFProtoNetConstruction:
    def test_constructs_without_error(self):
        model = _small_rbf_protonet()
        assert isinstance(model, RBFProtoNet)
        assert model.num_classes == 10
        assert model.feature_dim == 128

    def test_forward_pass_shape(self):
        model = _small_rbf_protonet()
        x = _images(4)
        out = model(x, training=False)
        assert tuple(out.shape) == (4, 10)

    def test_forward_pass_no_nan(self):
        model = _small_rbf_protonet()
        out = model(_images(4), training=False).numpy()
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_output_rows_sum_to_one(self):
        """``output_mode='normalized'`` contract: every row is a probability
        distribution over `num_classes` prototypes."""
        model = _small_rbf_protonet()
        out = keras.ops.convert_to_numpy(model(_images(8), training=False))
        np.testing.assert_allclose(
            out.sum(axis=-1), np.ones(8), atol=1e-4, rtol=0,
            err_msg="RBFProtoNet output rows must sum to 1.0 under output_mode='normalized'",
        )

    def test_create_factory_matches_direct_construction(self):
        model = create_rbf_protonet(num_classes=10)
        out = model(_images(2), training=False)
        assert tuple(out.shape) == (2, 10)

    @pytest.mark.parametrize("height, width", [(16, 16), (33, 20), (48, 64)])
    def test_the_model_runs_at_any_spatial_size(self, height, width):
        """Global average pooling makes the model size-agnostic, which is why it
        takes no ``input_shape``: nothing in it would read one."""
        model = _small_rbf_protonet()
        out = model(_images(2, height, width), training=False)
        assert tuple(out.shape) == (2, 10)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------


class TestRBFProtoNetArguments:
    def test_num_classes_is_the_first_parameter(self):
        names = list(inspect.signature(RBFProtoNet.__init__).parameters)
        assert names[1] == "num_classes"
        assert names[-1] == "kwargs"

    def test_the_constructor_has_no_pretrained_or_input_shape_knob(self):
        """Neither is read by any forward path: ``pretrained`` only ever raised and
        ``input_shape`` only fed a log line and ``get_config``."""
        names = inspect.signature(RBFProtoNet.__init__).parameters
        assert "pretrained" not in names and "input_shape" not in names
        config = _small_rbf_protonet().get_config()
        assert "pretrained" not in config and "input_shape" not in config

    def test_a_pretrained_keyword_on_the_class_is_refused_by_keras(self):
        with pytest.raises(ValueError, match="pretrained"):
            RBFProtoNet(num_classes=10, pretrained=True)

    @pytest.mark.parametrize("make", [np.int64, np.int32, np.uint8])
    def test_a_numpy_integer_num_classes_is_accepted_and_stored_as_an_int(self, make):
        model = _small_rbf_protonet(num_classes=make(10))
        assert type(model.num_classes) is int and model.num_classes == 10
        assert tuple(model(_images(2), training=False).shape) == (2, 10)
        json.dumps(model.get_config())  # an np.int64 in the config is not JSON-safe

    @pytest.mark.parametrize("value", [0, -3, 2.5, True, "10", None])
    def test_a_bad_num_classes_raises_naming_the_class_the_argument_and_the_value(self, value):
        """The one positive-integer rule (``utils.validation.positive_int``) with this class as owner."""
        with pytest.raises(
                ValueError, match=f"^RBFProtoNet: num_classes must be a positive integer, got .*{re.escape(str(value))}"):
            _small_rbf_protonet(num_classes=value)

    @pytest.mark.parametrize("keyword, value", [("input_shape", (32, 32, 3)), ("input_dim", 3)])
    def test_a_shape_keyword_is_refused_by_name(self, keyword, value):
        """Keras pops ``input_shape`` and ``input_dim`` and only warns; a caller written against
        the signature this class had before they were removed must not keep working silently."""
        with pytest.raises(ValueError, match=f"^RBFProtoNet takes no {keyword} .*any spatial size.*{keyword}="):
            RBFProtoNet(num_classes=10, **{keyword: value})

    def test_an_unknown_keyword_is_refused_naming_it(self):
        with pytest.raises(ValueError, match="not_an_argument"):
            RBFProtoNet(num_classes=10, not_an_argument=1)

    @pytest.mark.parametrize("rate", [1.0, 1.5, -0.1, float("nan")])
    def test_a_dropout_rate_outside_the_unit_interval_raises_naming_the_value(self, rate):
        with pytest.raises(ValueError, match=f"^dropout_rate must be in \\[0, 1\\), got {re.escape(repr(rate))}$"):
            _small_rbf_protonet(dropout_rate=rate)

    def test_a_dropout_rate_just_below_one_is_accepted(self):
        assert _small_rbf_protonet(dropout_rate=0.99).dropout_rate == 0.99

    def test_a_regularizer_given_by_name_is_resolved_serialized_and_rebuilt(self):
        model = _small_rbf_protonet(
            stem_filters=8, filters_per_stage=[8, 16], feature_dim=24, kernel_regularizer="l2")
        assert isinstance(model.kernel_regularizer, keras.regularizers.L2)
        config = model.get_config()
        json.dumps(config)
        rebuilt = RBFProtoNet.from_config(config)
        assert isinstance(rebuilt.kernel_regularizer, keras.regularizers.L2)
        assert rebuilt.get_config() == config
        plain = _small_rbf_protonet(stem_filters=8, filters_per_stage=[8, 16], feature_dim=24)
        x = _images(2)
        model(x)
        rebuilt(x)
        plain(x)
        assert len(model.losses) == len(rebuilt.losses) > len(plain.losses)

    def test_the_regularizer_reaches_the_feature_projection_too(self):
        """``feature_proj`` exists only when the last stage width differs from ``feature_dim``."""
        model = _small_rbf_protonet(
            stem_filters=8, filters_per_stage=[8, 16], feature_dim=24, kernel_regularizer="l2")
        assert model.feature_proj is not None
        assert model.feature_proj.kernel_regularizer is not None

    def test_the_default_has_no_regularizer(self):
        model = _small_rbf_protonet()
        assert model.kernel_regularizer is None
        assert model.get_config()["kernel_regularizer"] is None


# ---------------------------------------------------------------------
# pretrained refusal
# ---------------------------------------------------------------------


class TestRBFProtoNetPretrainedRefusal:
    @pytest.mark.parametrize("pretrained", [True, "weights.keras"])
    def test_the_factory_refuses_with_the_one_shared_message(self, pretrained):
        with pytest.raises(NotImplementedError) as raised:
            create_rbf_protonet(num_classes=10, pretrained=pretrained)
        assert str(raised.value) == _refusal("RBFProtoNet", pretrained)

    def test_the_message_gives_the_advice_that_is_true_for_this_model(self):
        """The model has a backbone and a head and its pre-change archives are refused by
        ``load_model``; the standard text ("no backbone/head split ... load_model(path)")
        describes another kind of model."""
        message = _refusal("RBFProtoNet", "weights.keras")
        assert "model.load_weights(path)" in message
        assert "backbone/head split" not in message and "load_model" not in message
        assert "(got pretrained='weights.keras')" in message

    @pytest.mark.parametrize("keyword, value", [("input_shape", (64, 64, 1)), ("input_dim", 3)])
    def test_the_factory_refuses_a_shape_keyword_by_name(self, keyword, value):
        """``create_rbf_protonet(input_shape=...)`` was the documented signature until it was
        removed; Keras alone only warns about it."""
        with pytest.raises(ValueError, match=f"RBFProtoNet takes no {keyword}"):
            create_rbf_protonet(num_classes=10, **{keyword: value})

    def test_the_factory_refuses_an_unknown_keyword_naming_it(self):
        with pytest.raises(ValueError, match="not_an_argument"):
            create_rbf_protonet(num_classes=10, not_an_argument=1)

    def test_the_refusal_comes_before_anything_is_built(self):
        """An invalid ``num_classes`` would raise ``ValueError`` if the model were built first."""
        with pytest.raises(NotImplementedError):
            create_rbf_protonet(num_classes=-1, pretrained=True)

    def test_the_default_is_a_random_initialization(self):
        assert isinstance(create_rbf_protonet(num_classes=10, pretrained=False), RBFProtoNet)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-vvv"]))
