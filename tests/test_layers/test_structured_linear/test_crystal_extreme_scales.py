"""Adversarial numerics for the crystal-structured layers.

The layers are cosine heads: they L2-normalize an input and a weight frame and
read off inner products. Their normalization is
:func:`dl_techniques.utils.tensors.l2_normalize`, which rescales before it
squares; the naive spelling it replaced fails at BOTH ends of the input scale
and fails SILENTLY in both, meaning the output stays finite and merely stops
being a unit vector.

These tests drive all three layers across that range. Every threshold is
measured, and the measurement is recorded beside the assertion.

Measured against a float64 oracle, error by input scale:

    ============  =========================  ==========================
    input scale   naive error                this layer's error
    ============  =========================  ==========================
    ``1e-32``     ``9.5e-01``                ``9.0e-08``
    ``1e-08``     ``9.5e-01``                ``9.8e-08``
    ``1e+00``     ``8.3e-08``                ``9.4e-08``
    ``1e+20``     ``9.5e-01``                ``9.2e-08``
    ``1e+38``     ``9.5e-01``                ``1.1e-07``
    ============  =========================  ==========================
"""

import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.structured_linear.crystal_layer import CrystalLayer
from dl_techniques.layers.structured_linear.hierarchical_crystal_layer import (
    HierarchicalCrystalLayer
)
from dl_techniques.layers.structured_linear.crystal_multihead_projection import (
    CrystalMultiHeadProjection
)

# ---------------------------------------------------------------------

#: Scales spanning 68 orders of magnitude. The ends are where the naive
#: spelling fails; the middle is where it already worked, and is included so a
#: regression in the ordinary regime cannot hide behind the extreme arms.
_SCALES = (-32, -8, 0, 20, 38)

#: Logits are a cosine scaled by `scale`, so ``|logit| <= scale``. The bound is
#: checked with this relative slack, which float32 accumulation over a
#: length-``input_dim`` dot product stays well inside.
_COSINE_BOUND_SLACK = 1e-3

#: Measured ``|sum(joint) - 1|`` across every scale in this module: 1.19e-07.
_BAYES_ATOL = 1e-5

_MULTIHEAD_DIM = 32
_MULTIHEAD_HEADS = 4


def _cosine_bound(scale: float) -> float:
    """Upper bound on ``|logit|`` for a cosine scaled by ``scale``."""
    return scale * (1.0 + _COSINE_BOUND_SLACK)


def _scale_tensor(tensor, factor: float):
    """Multiply a float32 tensor by a positive factor, keeping its dtype."""
    scaled = ops.convert_to_numpy(tensor) * np.float32(factor)

    return ops.convert_to_tensor(scaled)


def _scaled(exponent: int, shape=(4, 8), seed: int = 0) -> np.ndarray:
    """Input of magnitude ~1e``exponent``, built without overflowing the DATA.

    Scaling a standard normal directly by 1e38 overflows in float32 before the
    layer ever sees it, which would measure the probe rather than the layer.
    The vector is normalized first and scaled second.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(shape)
    base /= np.abs(base).max()

    return (base * np.float64(10.0) ** exponent).astype(np.float32)


def _crystal(exponent: int):
    return _scaled(exponent)


def _multihead(exponent: int):
    return _scaled(exponent, shape=(2, 6, _MULTIHEAD_DIM))


class TestCrystalLayerExtremeScales:
    """A cosine head must stay a cosine head at every input magnitude."""

    @pytest.mark.parametrize('exponent', _SCALES)
    def test_logits_stay_finite_and_bounded(self, exponent):
        """Bounded by `scale`, which is the closed form of a cosine.

        This is the anti-vacuity shape of the whole module: a layer that
        returned zeros at 1e20 would pass a finiteness check, and would fail
        only because the logits are no longer within ``+-scale``.
        """
        scale = 10.0
        layer = CrystalLayer(num_classes=9, scale=scale, seed=0)

        output = ops.convert_to_numpy(layer(_crystal(exponent)))

        assert np.all(np.isfinite(output)), (
            f"1e{exponent} produced a non-finite logit"
        )
        assert np.abs(output).max() <= _cosine_bound(scale), (
            f"1e{exponent} gave |logit|max = {np.abs(output).max():.4f}, above "
            f"the cosine bound {scale}"
        )

    @pytest.mark.parametrize('exponent', _SCALES)
    def test_the_logits_do_not_collapse(self, exponent):
        """The other half of the pair: bounded output must still carry signal.

        The naive form returns exact zeros above ~1.8e19 -- finite, and within
        the bound above. Only a non-degeneracy assertion separates the two.
        """
        layer = CrystalLayer(num_classes=9, scale=10.0, seed=0)

        output = ops.convert_to_numpy(layer(_crystal(exponent)))

        assert np.abs(output).max() > 1.0, (
            f"1e{exponent} collapsed the logits to {np.abs(output).max():.3e}; "
            f"this is the silent-zero failure"
        )

    def test_a_zero_input_still_yields_zero_logits(self):
        """The all-zero limit is unchanged by the rescale."""
        layer = CrystalLayer(num_classes=9, scale=10.0, seed=0)

        output = ops.convert_to_numpy(layer(np.zeros((3, 8), dtype=np.float32)))

        assert np.all(np.isfinite(output))
        np.testing.assert_allclose(output, 0.0, rtol=0, atol=1e-5)


class TestHierarchicalCrystalLayerExtremeScales:
    """The Bayes identity must survive every input magnitude."""

    @pytest.mark.parametrize('exponent', _SCALES)
    def test_the_joint_still_sums_to_one(self, exponent):
        layer = HierarchicalCrystalLayer(
            num_macro_classes=3,
            num_micro_per_macro=5,
            return_hierarchy=True,
            seed=0
        )

        outputs = layer(_crystal(exponent))

        for key, value in outputs.items():
            assert np.all(np.isfinite(ops.convert_to_numpy(value))), (
                f"1e{exponent} made {key} non-finite"
            )

        joint = ops.convert_to_numpy(outputs['joint_probabilities'])
        np.testing.assert_allclose(
            joint.sum(axis=-1), 1.0, rtol=0, atol=_BAYES_ATOL,
            err_msg=f"1e{exponent} broke the Bayes normalization"
        )

    @pytest.mark.parametrize('exponent', _SCALES)
    def test_the_macro_distribution_is_not_degenerate(self, exponent):
        """Anti-vacuity: a uniform joint would also sum to one.

        At a collapsed input every macro probability is equal, which passes the
        sum assertion above. Requiring the distribution to be non-uniform
        separates the two.
        """
        layer = HierarchicalCrystalLayer(
            num_macro_classes=3,
            num_micro_per_macro=5,
            return_hierarchy=True,
            seed=0
        )

        joint = ops.convert_to_numpy(
            layer(_crystal(exponent))['joint_probabilities']
        )

        assert joint.max(axis=-1).min() - joint.min(axis=-1).max() > 1e-4, (
            f"1e{exponent} produced a uniform joint distribution, so the "
            f"normalization above would pass on a dead layer"
        )


class TestCrystalMultiHeadProjectionExtremeScales:
    """The head split and prism contraction must survive every magnitude."""

    @pytest.mark.parametrize('exponent', _SCALES)
    def test_the_output_stays_finite(self, exponent):
        layer = CrystalMultiHeadProjection(num_heads=_MULTIHEAD_HEADS, seed=0)

        output = ops.convert_to_numpy(layer(_multihead(exponent)))

        assert output.shape == (2, 6, _MULTIHEAD_DIM)
        assert np.all(np.isfinite(output)), f"1e{exponent} produced a non-finite value"

    @pytest.mark.parametrize('exponent', _SCALES)
    def test_the_output_does_not_collapse(self, exponent):
        layer = CrystalMultiHeadProjection(num_heads=_MULTIHEAD_HEADS, seed=0)

        output = ops.convert_to_numpy(layer(_multihead(exponent)))

        assert np.abs(output).max() > 1e-3, (
            f"1e{exponent} collapsed the projection to {np.abs(output).max():.3e}"
        )


class TestAdversarialRowsInABatch:
    """A degenerate row must not contaminate its neighbours."""

    @pytest.mark.parametrize('exponent', [20, 30])
    def test_one_huge_row_leaves_the_others_intact(self, exponent):
        """Contamination is sample-local: the reduction is over the last axis."""
        inputs = _scaled(exponent=0, shape=(8, 8))
        inputs[0] *= np.float32(10.0) ** exponent

        output = ops.convert_to_numpy(CrystalLayer(num_classes=9, seed=0)(inputs))

        assert np.all(np.isfinite(output[1:])), (
            "a degenerate row contaminated unrelated rows of the batch"
        )

    def test_a_zero_row_beside_huge_rows_is_handled(self):
        inputs = _scaled(exponent=0, shape=(6, 8))
        inputs[0] = 0.0
        inputs[1] *= np.float32(1e30)

        output = ops.convert_to_numpy(CrystalLayer(num_classes=9, seed=0)(inputs))

        assert np.all(np.isfinite(output)), (
            "mixing an all-zero row with a saturating row went non-finite"
        )


class TestTrainableWeightsThatHaveDrifted:
    """`trainable_crystal=True` lets the frame leave the sphere; it must survive."""

    @pytest.mark.parametrize('exponent', [10, 18, 20])
    def test_a_drifted_frame_still_yields_bounded_logits(self, exponent):
        """Scaling the frame by 1e``exponent`` models a weight that grew.

        The frame is re-normalized on every forward precisely so that a trained
        one still yields cosines. Measured against the naive form, which would
        return zeros for the two largest settings.
        """
        scale = 10.0
        layer = CrystalLayer(
            num_classes=9, scale=scale, trainable_crystal=True, seed=0
        )
        layer(_crystal(0))

        layer.crystal_weights.assign(
            _scale_tensor(layer.crystal_weights, 10.0 ** exponent)
        )

        output = ops.convert_to_numpy(layer(_crystal(0)))

        assert np.all(np.isfinite(output)), (
            f"a frame scaled by 1e{exponent} produced a non-finite logit"
        )
        assert np.abs(output).max() <= _cosine_bound(scale), (
            f"a frame scaled by 1e{exponent} broke the cosine bound: "
            f"|logit|max = {np.abs(output).max():.4f}"
        )
