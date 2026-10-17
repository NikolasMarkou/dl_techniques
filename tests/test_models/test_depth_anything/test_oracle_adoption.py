"""
Oracle adoption for ``models/vision/depth_anything`` -- Phase 5 batch A.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (CPU) on ``DepthAnything(encoder_kind='real',
encoder_type='vit_s', image_shape=(64, 64, 3))`` after one real optimizer step:
**164** trainable weights, **0** dead, **0** non-finite.

That 0 is a fact about this package worth having, because the architecture is an
encoder plus a DPT-style decoder that reads MULTIPLE encoder stages through
reassemble/fusion blocks. If one reassemble branch is built and then dropped
before the fusion sum, the model still returns a correctly-shaped, finite depth
map at full resolution, and every other test in this directory stays green --
the only visible symptom is a subtree of weights that never receives a gradient.
"""

import os

# Force CPU before keras / tensorflow are imported, matching this directory's
# `test_depth_anything.py`: the real-ViT encoder is the expensive arm here and
# the numbers pinned above were taken on CPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.losses.affine_invariant_loss import AffineInvariantLoss
from dl_techniques.models.vision.depth_anything import DepthAnything

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

IMAGE_SHAPE = (64, 64, 3)

#: Measured 2026-08-21 at encoder_kind='real', encoder_type='vit_s', 64x64.
#: Updated 2026-10-10 after fixing DepthEstimationHead to not create unused norm layer:
#: **164** trainable weights, **0** dead, **0** non-finite.
#: Updated 2026-10-17 (plan-2026-10-14T062616-81938e81 D-128), the decoder now
#: the reference DPTHead: **210** trainable weights (150 ViT + 60 head), **0**
#: dead, **0** non-finite.
GF_N_WEIGHTS = 210


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).random((batch,) + IMAGE_SHAPE).astype("float32")


def _model(**overrides):
    # Seeded: behind the reference head's final ReLU the share of exactly-zero
    # output pixels at init depends on the draw (MEASURED 2026-10-17, vit_s,
    # 10 seeds: 0.053 to 0.987 at 64x64), and an all-zero map has no gradient
    # anywhere, so an unseeded draw made the gradient-flow arm order-dependent
    # (0/210 live in one full-directory run). Seed 0 starts at 0.229.
    keras.utils.set_random_seed(0)
    kwargs = dict(encoder_kind="real", encoder_type="vit_s", image_shape=IMAGE_SHAPE)
    kwargs.update(overrides)
    model = DepthAnything(**kwargs)
    model(keras.ops.zeros((1,) + IMAGE_SHAPE))
    return model


def _one_adam_step(model, inputs) -> None:
    """One Adam step on the model's own objective, the affine-invariant loss.

    Not on ``default_loss``: the mean of squares pulls every output toward 0,
    and behind the reference head's final ReLU one such step leaves the map
    all zero (MEASURED 2026-10-17, vit_s 64x64: zero fraction 0.023 -> 0.995;
    at 256x256 1.000), a point with no gradient at all. The affine-invariant
    loss has no pull toward a constant (zero fraction 0.010 after the step).
    """
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    target = np.random.default_rng(1).random((inputs.shape[0],) + IMAGE_SHAPE[:2] + (1,))
    with tf.GradientTape() as tape:
        loss = AffineInvariantLoss()(target.astype("float32"), model(inputs, training=True))
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))


class TestDepthAnythingGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _images())


class TestDepthAnythingKnobSensitivity:

    def test_encoder_type_changes_the_parameterisation(self):
        """`encoder_type`, not `variant`, is this package's variant knob.

        `test_package_api_contract.py`'s MODEL_VARIANTS guard is structurally
        blind to it (it looks for `variant` / `from_variant`), so nothing else
        in the tree asserts that the slug reaches the encoder's width and depth.
        """
        builders = {
            slug: (lambda slug=slug: _model(encoder_type=slug))
            for slug in ("vit_s", "vit_b")
        }
        signatures = assert_structural_knob_changes_weights(
            builders, knob="encoder_type",
        )
        sizes = [
            sum(int(np.prod(s)) for s in signatures[slug])
            for slug in ("vit_s", "vit_b")
        ]
        assert sizes[0] < sizes[1], f"vit_b is not larger than vit_s: {sizes}"

    def test_dpt_features_changes_the_parameterisation(self):
        """A knob that reaches the DECODER only, so it cannot be met by the encoder."""
        builders = {
            f: (lambda f=f: _model(dpt_features=f))
            for f in (32, 48)
        }
        assert_structural_knob_changes_weights(builders, knob="dpt_features")

    def test_dpt_out_channels_changes_the_parameterisation(self):
        """The head's four projection widths, a second decoder-only knob."""
        builders = {
            tuple(c): (lambda c=c: _model(dpt_out_channels=list(c)))
            for c in ([16, 16, 32, 32], [24, 24, 48, 48])
        }
        assert_structural_knob_changes_weights(builders, knob="dpt_out_channels")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model(encoder_type="vit_s")),
                    "b": (lambda: _model(encoder_type="vit_s"))}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="encoder_type")


class TestDepthAnythingSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _images()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"DepthAnything returns one tensor, got {type(out)}"
            )
            assert tuple(out.shape) == (x.shape[0],) + IMAGE_SHAPE[:2] + (1,), (
                f"a depth map must keep the input resolution; got "
                f"{tuple(out.shape)}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_an_unsupported_encoder_type_is_rejected_at_construction(self):
        """The argument-validation half: a silent fallback would pass everything else."""
        with pytest.raises(ValueError, match="[Uu]nsupported encoder type"):
            DepthAnything(encoder_type="vit_xxl", image_shape=IMAGE_SHAPE)
