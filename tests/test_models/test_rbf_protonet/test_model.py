"""
Model-level tests for ``RBFProtoNet`` (construction, forward pass, and the
``pretrained=True`` refusal contract), following ``tests/test_models/test_resnet/``
conventions.

``RBFProtoNet`` composes a small CIFAR-style CNN backbone with an RBF
prototype-classification head (``output_mode='normalized'``, see D-002/D-006/
D-007 in this plan's ``decisions.md``). The head's defining contract is that
its output is already a per-class probability vector -- each row sums to 1.0
-- not a logits vector, so that is asserted here directly rather than left to
inference from the model's docstring.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.rbf_protonet.model import RBFProtoNet, create_rbf_protonet


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _small_rbf_protonet(**overrides):
    """A CIFAR-shaped RBFProtoNet with a small num_classes for fast CPU tests."""
    cfg = dict(
        input_shape=(32, 32, 3),
        num_classes=10,
    )
    cfg.update(overrides)
    return RBFProtoNet(**cfg)


def _images(batch: int = 4) -> np.ndarray:
    return np.random.default_rng(0).random((batch, 32, 32, 3)).astype("float32")


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
        model = create_rbf_protonet(num_classes=10, input_shape=(32, 32, 3))
        out = model(_images(2), training=False)
        assert tuple(out.shape) == (2, 10)


# ---------------------------------------------------------------------
# pretrained=True refusal
# ---------------------------------------------------------------------


class TestRBFProtoNetPretrainedRefusal:
    def test_pretrained_true_raises_not_implemented_on_class(self):
        with pytest.raises(NotImplementedError):
            RBFProtoNet(input_shape=(32, 32, 3), num_classes=10, pretrained=True)

    def test_pretrained_true_raises_not_implemented_on_factory(self):
        with pytest.raises(NotImplementedError):
            create_rbf_protonet(num_classes=10, input_shape=(32, 32, 3), pretrained=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-vvv"]))
