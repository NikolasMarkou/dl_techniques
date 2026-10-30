"""Both RBF prototype models build their head in ``output_mode='normalized'``.

D-002 (plan-2026-09-16T052349-7dfede94 and plan-2026-09-16T064227-b83f88ad): ``'basis'`` is
barely trainable at this scale and the trainer's ``from_logits=False`` loss needs a
probability vector, so ``'normalized'`` is the only supported mode. There is no shared helper
that pins it; each model's ``_build_rbf_head`` names the mode, and this file is the one place
that fails if either of them stops doing so.
"""

import numpy as np
import pytest

from dl_techniques.layers.mixtures.radial_basis_function import RBFLayer
from dl_techniques.models.vision.rbf_protonet.model import (
    CliffordRBFProtoNet,
    RBFProtoNet,
)


def _rbf_protonet() -> RBFProtoNet:
    return RBFProtoNet(num_classes=4, stem_filters=8, filters_per_stage=[8, 16], feature_dim=16,
                       repulsion_strength=0.3, min_distance=2.5)


def _clifford_rbf_protonet() -> CliffordRBFProtoNet:
    return CliffordRBFProtoNet(num_classes=4, channels=16, depth=1, feature_dim=16,
                              repulsion_strength=0.3, min_distance=2.5)


@pytest.mark.parametrize("make", [_rbf_protonet, _clifford_rbf_protonet],
                         ids=["RBFProtoNet", "CliffordRBFProtoNet"])
class TestTheHeadIsNormalized:
    def test_the_head_is_an_rbf_layer_in_normalized_mode(self, make):
        head = make().rbf_head
        assert isinstance(head, RBFLayer)
        assert head.output_mode == "normalized"
        assert head.name == "rbf_head"

    def test_the_head_forwards_its_arguments(self, make):
        model = make()
        assert model.rbf_head.units == model.num_classes == 4
        assert model.rbf_head.repulsion_strength == 0.3
        assert model.rbf_head.min_distance == 2.5

    def test_the_output_is_a_probability_vector(self, make):
        model = make()
        x = np.random.default_rng(0).random((3, 32, 32, 3)).astype("float32")
        y = np.asarray(model(x, training=False))
        assert y.shape == (3, 4)
        np.testing.assert_allclose(y.sum(axis=-1), 1.0, atol=1e-5)
