"""
Oracle adoption for ``models/clip`` -- Phase 5 batch A.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (CPU) on a tiny two-tower CLIP (1 vision layer, 1 text
layer, width 32, embed_dim 32, context_length 8) after one real optimizer step:
**32** trainable weights, **0** dead, **0** non-finite.

Two things this adoption pins that a shape/finiteness smoke test cannot:

* ``logit_scale`` -- a single learnable scalar -- receives a live gradient. A
  temperature that is created, saved and never multiplied in is invisible to
  every structural check and changes nothing about the output shapes.
* BOTH towers appear in the trainable set. A contrastive model whose text tower
  is detached still returns a full, finite, correctly-shaped output dict.

``logit_scale`` is rank-0, which is why ``slice_leading_axis`` is dropped from
the smoke breakers below -- see that test's docstring.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.clip.model import CLIP

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
    collapse_to_scalar,
    append_trailing_axis,
)

IMAGE_SIZE = 32
PATCH_SIZE = 16
CONTEXT_LENGTH = 8
VOCAB_SIZE = 100
EMBED_DIM = 32

#: Measured 2026-08-21 at the tiny configuration in `_model`.
GF_N_WEIGHTS = 32

OUTPUT_KEYS = (
    "image_features", "text_features",
    "logits_per_image", "logits_per_text", "logit_scale",
)


def _model(**overrides):
    kwargs = dict(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
        vision_layers=1, vision_width=32, vision_heads=4, vision_kv_heads=2,
        vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH,
        text_layers=1, text_width=32, text_heads=4, text_kv_heads=2,
        embed_dim=EMBED_DIM,
    )
    kwargs.update(overrides)
    model = CLIP(**kwargs)
    model(_batch(1), training=False)
    return model


def _batch(batch: int = 2):
    rng = np.random.default_rng(0)
    return {
        "image": rng.random((batch, IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32"),
        "text": rng.integers(0, VOCAB_SIZE, (batch, CONTEXT_LENGTH)).astype("int32"),
    }


def _one_adam_step(model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))


class TestCLIPGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _batch()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)

        scale = [p for p in report if "logit_scale" in p]
        assert len(scale) == 1, f"expected one logit_scale weight, got {scale}"
        assert report[scale[0]] > 0.0

    def test_both_towers_are_in_the_trainable_set(self):
        """A contrastive model with one detached tower is otherwise well-shaped.

        Without this, the count above could be satisfied by a vision tower and a
        text tower that never contributes -- and every output key would still be
        present, finite and correctly shaped.
        """
        model = _model()
        report = assert_gradients_reach_every_trainable_weight(model, _batch())
        joined = " ".join(report)
        assert "visual" in joined or "vision" in joined, sorted(report)
        assert "text" in joined or "token_embedding" in joined, sorted(report)

    def test_the_gradient_assertion_can_fail(self):
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _batch())


class TestCLIPKnobSensitivity:

    def test_vision_layers_changes_the_parameterisation(self):
        builders = {n: (lambda n=n: _model(vision_layers=n)) for n in (1, 2, 3)}
        assert_structural_knob_changes_weights(builders, knob="vision_layers")

    def test_text_layers_changes_the_parameterisation(self):
        """The text tower has its own depth knob, and it must reach its own stack."""
        builders = {n: (lambda n=n: _model(text_layers=n)) for n in (1, 2, 3)}
        assert_structural_knob_changes_weights(builders, knob="text_layers")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model(vision_layers=1)),
                    "b": (lambda: _model(vision_layers=1))}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="vision_layers")


class TestCLIPSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        """`slice_leading_axis` is DROPPED, with the reason.

        `logit_scale` is a rank-0 leaf and that breaker raises on rank-0 by
        design -- it cannot break a scalar, so requiring a rejection would be an
        artefact of the breaker rather than a property of the contract. The
        remaining two still cover the container/key-set class and the full-shape
        class of under-assertion.
        """
        model = _model()
        x = _batch()
        batch = x["image"].shape[0]

        def contract(out):
            assert isinstance(out, dict), f"expected a dict, got {type(out)}"
            assert set(out) == set(OUTPUT_KEYS), f"key set drifted: {sorted(out)}"
            assert tuple(out["image_features"].shape) == (batch, EMBED_DIM)
            assert tuple(out["text_features"].shape) == (batch, EMBED_DIM)
            assert tuple(out["logits_per_image"].shape) == (batch, batch)
            assert tuple(out["logits_per_text"].shape) == (batch, batch)
            assert tuple(out["logit_scale"].shape) == ()
            assert_finite(out)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-094
        # Do NOT restore `slice_leading_axis` here, and do NOT "fix" it by
        # teaching it to skip rank-0 leaves. `logit_scale` is genuinely rank-0,
        # the breaker raises on rank-0 BY DESIGN (it cannot break a scalar, so a
        # rejection would be an artefact of the breaker), and a skip-rank-0
        # variant would silently turn this into a two-breaker call anyway while
        # reading as a three-breaker one. Dropping it explicitly, with the
        # reason, is the honest form. See D-094 in plans/plan-2026-08-19T163559-499b6f0e/decisions.md.
        rejections = assert_contract_rejects_a_broken_forward(
            model, x, contract, breakers=(collapse_to_scalar, append_trailing_axis),
        )
        assert set(rejections) == {"collapse_to_scalar", "append_trailing_axis"}
