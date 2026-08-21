"""
Oracle adoption for ``models/dino`` -- Phase 5 batch A.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

The mask token, measured both ways
----------------------------------
Measured 2026-08-21 (CPU) on ``create_dino_v2('tiny', image_size=28,
patch_size=14, num_classes=10)`` after one real optimizer step: **177** trainable
weights, and the dead set depends ENTIRELY on the mask that is fed in.

===========================  =========  ==================================
mask                         n weights  identically-zero gradient
===========================  =========  ==================================
all-False (nothing masked)   177        ``mask_token/mask_token`` -- 1 of 177
mixed (2 of 4 patches)       177        none -- 0 of 177
===========================  =========  ==================================

So there is NO waiver in this file. The primary assertion runs with a mixed mask
and no ``expect_zero`` at all, over all 177 weights, and the companion test pins
the all-False reading as the EXACT one-element set. Together they say "the mask
token is dead precisely when nothing is masked", which is a claim about the
masking mechanism -- not the weaker "one weight is allowed to be dead", which a
future second dead weight would slide into unnoticed.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.dino.dino_v2 import create_dino_v2

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

IMAGE_SIZE = 28
PATCH_SIZE = 14
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 4
NUM_CLASSES = 10

#: Measured 2026-08-21 at the 'tiny' variant, 28px / patch 14.
GF_N_WEIGHTS = 177

#: The exact dead set under an ALL-FALSE mask. Pinned as a set, not a count.
# DECISION plan-2026-08-19T163559-499b6f0e/D-094
# Do NOT collapse these two tests into one `expect_zero=("mask_token",)` call,
# however much tidier that looks. A waiver says "this weight may be dead" and
# stops there; the pair says "the mask token is dead PRECISELY when nothing is
# masked" -- the primary test runs with NO waiver at all over all 177 weights,
# and this constant pins the all-False dead set as an exact SET, so a second
# dead weight cannot arrive inside an allowance. Measured 2026-08-21: 1 of 177
# under an all-False mask, 0 of 177 under a mixed one.
# See D-094 in plans/plan-2026-08-19T163559-499b6f0e/decisions.md.
DEAD_UNDER_EMPTY_MASK = ("mask_token/mask_token",)


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).random(
        (batch, IMAGE_SIZE, IMAGE_SIZE, 3)
    ).astype("float32")


def _empty_mask(batch: int = 2) -> np.ndarray:
    return np.zeros((batch, NUM_PATCHES), dtype=bool)


def _mixed_mask(batch: int = 2) -> np.ndarray:
    """Patches 0 and 2 masked in every row.

    A MIXED mask, not an all-True one: it keeps both sides of the
    `ops.where(mask, mask_token, patch_emb)` selection live, so the reading
    below covers the unmasked path as well.
    """
    mask = np.zeros((batch, NUM_PATCHES), dtype=bool)
    mask[:, 0] = True
    mask[:, 2] = True
    return mask


def _model(**overrides):
    kwargs = dict(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                  num_classes=NUM_CLASSES)
    variant = overrides.pop("variant", "tiny")
    kwargs.update(overrides)
    model = create_dino_v2(variant, **kwargs)
    model([_images(1), _empty_mask(1)], training=False)
    return model


def _one_adam_step(model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))


class TestDinoV2GradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        """All 177 weights, NO waiver -- because the mask actually masks."""
        model = _model()
        inputs = [_images(), _mixed_mask()]
        _one_adam_step(model, inputs)

        report = assert_gradients_reach_every_trainable_weight(model, inputs)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)

    def test_an_empty_mask_leaves_exactly_the_mask_token_dead(self):
        """The other half: the dead set under an all-False mask is EXACTLY one weight.

        This is what makes the test above a claim about masking rather than an
        accident of the input. `ops.where` with an all-False condition selects
        the mask token nowhere, so its gradient is identically zero -- and
        nothing ELSE may be dead, which is the part a waiver-shaped test would
        have stopped checking.
        """
        model = _model()
        inputs = [_images(), _empty_mask()]
        _one_adam_step(model, inputs)

        report = gradient_report(model, inputs)
        dead = sorted(p for p, v in report.items() if v is None or v == 0.0)

        assert dead == list(DEAD_UNDER_EMPTY_MASK), dead
        assert len(report) == GF_N_WEIGHTS

    def test_the_gradient_assertion_can_fail(self):
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, [_images(), _mixed_mask()],
                )


class TestDinoV2KnobSensitivity:

    def test_variant_changes_the_parameterisation(self):
        builders = {
            v: (lambda v=v: _model(variant=v)) for v in ("tiny", "small", "base")
        }
        signatures = assert_structural_knob_changes_weights(builders, knob="variant")
        sizes = [
            sum(int(np.prod(s)) for s in signatures[v])
            for v in ("tiny", "small", "base")
        ]
        assert sizes == sorted(sizes), f"variant is not monotone: {sizes}"

    def test_num_register_tokens_changes_the_parameterisation(self):
        """The DINOv2-with-registers knob, which `variant` alone cannot cover."""
        builders = {
            n: (lambda n=n: _model(num_register_tokens=n)) for n in (0, 2, 4)
        }
        assert_structural_knob_changes_weights(builders, knob="num_register_tokens")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model(variant="tiny")),
                    "b": (lambda: _model(variant="tiny"))}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="variant")


class TestDinoV2SmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        inputs = [_images(), _empty_mask()]

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"the classifier returns one tensor, got {type(out)}"
            )
            assert tuple(out.shape) == (2, NUM_CLASSES), (
                f"expected {(2, NUM_CLASSES)}, got {tuple(out.shape)}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, inputs, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
