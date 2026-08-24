"""
Oracle adoption for ``models/SAM/SAM2`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

WHAT IS MEASURED, AND WHY IT IS THE WRAPPER
--------------------------------------------
``SAM2.call`` is memory-free (an image-only inference at SAM 2's weights);
``SAM2TrainingModel`` is the traceable multi-frame path that actually trains,
unrolling a static frame loop over the submodules with a fresh local memory
bank. Every measurement here runs on the wrapper, at the ``tiny`` variant,
``multimask_output=False``, ``T = 3`` frames, batch 2.

Measured 2026-08-21 (GPU 1), one real Adam step, ``default_loss``: **344**
trainable weights, **304 live**, **40 not on the backward graph**. The 40 are
NOT a defect and NOT one story -- they are four separate, individually
attributable causes, and each is named in :data:`EXPECTED_OFF_GRAPH` so a later
reader cannot re-file the set as a dead model:

1. **``obj_ptr_proj`` (6).** A DOCUMENTED known limitation of the wrapper, in
   its own module docstring: the object pointer bypasses the memory encoder, so
   leaving it live would rebuild the T-deep recursion the wrapper exists to
   truncate. It ships frozen, deliberately.
2. **The three no-object / empty-memory embeddings (3):** ``no_mem_pos_enc``,
   ``no_obj_ptr``, ``no_obj_embed_spatial``. Reached only on a frame with no
   memory or no object; this clip has both on every frame.
3. **The prompt encoder's unused branches (13):** ``mask_downscaling`` (9, no
   mask prompt), ``point_embedding_2/3`` (2, no box prompt) and
   ``point_embedding_0`` (1, the background-label row -- every label in this
   clip is foreground). Input coverage, exactly as in ``test_sam``.
4. **Mask-tokens 1..3's hypernetwork MLPs (18).** The decoder holds
   ``num_multimask_outputs + 1`` mask tokens and returns index 0 at
   ``multimask_output=False``; the wrapper REFUSES ``multimask_output=True``
   outright (it would interleave the frame and mask axes), so on this path
   those three are structurally unreachable. This is the one entry of the four
   that a shipped configuration cannot flip.

``expect_zero`` is two-sided, so the assertion below proves every one of those
40 IS dead AND that all 304 others are live -- the waiver cannot rot into a
skip list that silently absorbs a new dead weight.

A HARD-WIRED DROPOUT RATE, AND WHY THE MEASUREMENT PINS IT
-----------------------------------------------------------
SAM 2's memory attention ships ``dropout_rate=0.1`` (the single home of the number
is ``memory_attention.DEFAULT_DROPOUT_RATE``). Since D-090 it IS reachable --
``SAM2.from_variant(dropout_rate=...)``, ``create_sam2(dropout_rate=...)`` and
a ``dropout_rate`` key in every variant table -- but the SHIPPED default is
unchanged at 0.1, which is what these models are built with. A training-mode
measurement on a stock SAM 2 is therefore still a DRAW -- the same hazard that made a BeiT arm in
batch A report a whole dead encoder block 1 run in 4. Every model measured here
runs through :func:`_pin_dropout_to_zero`, and
``test_the_hard_wired_memory_attention_dropout_is_really_there`` pins the
premise so the helper is deleted rather than left as ceremony if the rate ever
becomes configurable. (The off-graph SET happens to be the same either way --
dropout scales a gradient, it does not disconnect a weight -- but every
MAGNITUDE in an unpinned run is one sample of a distribution.)
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.SAM.SAM2.model import create_sam2
from dl_techniques.models.SAM.SAM2.training_model import (
    SAM2_IOU_SUPERVISION,
    SAM2_LOW_RES_LOGITS,
    SAM2_OBJECT_SCORE_LOGITS,
    SAM2TrainingModel,
)

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
from .test_training_model import BATCH, FRAMES, clip_inputs, trainer

#: Measured 2026-08-21 at the ``tiny`` variant, ``T = 3``.
GF_N_WEIGHTS = 344
GF_N_OFF_GRAPH = 40

#: Every weight NOT on the backward graph of the shipped clip path, matched by
#: path SUFFIX -- see the module docstring for the four causes. Suffixes rather
#: than absolute ``Variable.path`` strings because Keras uniquifies a model's
#: name per process, so an absolute pin is green alone and red behind any other
#: test that builds this class in the same session.
EXPECTED_OFF_GRAPH = tuple(sorted(
    # 1: the wrapper's documented truncation
    [f"obj_ptr_proj/obj_ptr_proj_dense{i}/{w}"
     for i in (1, 2, 3) for w in ("kernel", "bias")]
    # 2: no-object / empty-memory embeddings
    + ["/no_mem_pos_enc", "/no_obj_ptr", "/no_obj_embed_spatial"]
    # 3: prompt branches this clip does not supply
    + [f"mask_downscaling/conv{i}/{w}"
       for i in (1, 2, 3) for w in ("kernel", "bias")]
    + [f"mask_downscaling/norm{i}/{w}"
       for i in (1, 2) for w in ("gamma", "beta")]
    + [f"point_embedding_{i}/embeddings" for i in (0, 2, 3)]
    # 4: mask tokens the wrapper's single-mask path never returns
    + [f"hypernetwork_mlp_{m}/hyper_dense{i}_{m}/{w}"
       for m in (1, 2, 3) for i in (1, 2, 3) for w in ("kernel", "bias")]
))


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


#: MEASURED 2026-08-21 and worth stating plainly: SAM 2's memory attention
#: ships ``dropout_rate=0.1``. Since plan-2026-08-22T035419-a11304c8 D-090 the rate
#: IS configurable (``SAM2.from_variant(dropout_rate=...)``), but the shipped
#: default these models are built with is still 0.1, so a training-mode
#: measurement on a stock SAM 2 is a DRAW, which is the exact hazard that made a BeiT arm in batch A flaky 1 run
#: in 4. Since the rate cannot be configured, it is pinned here by assignment
#: on the built layers, and ``test_no_layer_is_stochastic`` then asserts that
#: nothing stochastic survived. Do NOT drop this: without it, every number in
#: this file's docstring is one sample of a distribution. (Building the fixture
#: with ``dropout_rate=0.0`` would be the tidier route now that the knob
#: exists, but the trainer fixture is constructed elsewhere; the pinning below
#: also covers stochastic depth, which no dropout knob reaches.)
def _pin_dropout_to_zero(model: keras.Model) -> int:
    pinned = 0
    for layer in model._flatten_layers(include_self=False):
        rate = getattr(layer, "rate", None)
        if isinstance(rate, float) and rate > 0.0:
            layer.rate = 0.0
            pinned += 1
    return pinned


def _built(**overrides) -> SAM2TrainingModel:
    # Since D-090 the rate is a CONSTRUCTOR argument, so it is killed at the
    # source rather than only on the built `Dropout` layers. This is not
    # cosmetic: `SAM2.dropout_rate` is a property over the memory-attention
    # stack's configured rate, so a model built at 0.1 and pinned afterwards
    # still REPORTS 0.1 and `test_no_layer_is_stochastic` correctly rejects it.
    # `_pin_dropout_to_zero` is still called below -- it covers the stochastic
    # depth in the Hiera trunk, which no dropout knob reaches.
    overrides.setdefault("dropout_rate", 0.0)
    keras.utils.set_random_seed(11)
    model = trainer(**overrides)
    _pin_dropout_to_zero(model)
    return model


class TestSAM2GradientFlow:

    def test_the_hard_wired_memory_attention_dropout_is_really_there(self):
        """The premise of ``_pin_dropout_to_zero``, asserted before it is used.

        What it pins is the SHIPPED RATE, not the absence of a knob: D-090 gave
        SAM 2 a ``dropout_rate`` knob and deliberately left the default at 0.1,
        so this stayed green through that change -- correctly, because the
        fixture is still built at the default and still draws. If the shipped
        default ever moves to 0.0, this goes red and the pinning helper can be
        deleted rather than left as dead ceremony.
        """
        keras.utils.set_random_seed(11)
        stock = trainer()
        rates = [(layer.name, layer.rate)
                 for layer in stock._flatten_layers(include_self=False)
                 if isinstance(getattr(layer, "rate", None), float)
                 and layer.rate > 0.0]
        assert rates, (
            "no non-zero dropout rate found on a stock SAM 2 -- the shipped "
            "default is now zero, so _pin_dropout_to_zero is obsolete"
        )
        assert all(rate == 0.1 for _, rate in rates), rates

    def test_no_layer_is_stochastic(self):
        """After pinning: nothing in the measured model draws."""
        model = _built()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], (
            f"a non-zero stochastic rate is live: {stochastic}. A gradient "
            f"report taken under one reports the DRAW, not the model"
        )

    def test_the_expected_off_graph_set_is_exactly_the_measured_one(self):
        """The count AND the membership, before any waiver is trusted."""
        model = _built()
        x = clip_inputs(model)
        _one_adam_step(model, x)
        report = gradient_report(model, x)
        dead = [p for p, v in report.items() if v is None or v == 0.0]

        assert len(dead) == GF_N_OFF_GRAPH, (
            f"expected {GF_N_OFF_GRAPH} off-graph weights, got {len(dead)}: "
            f"{sorted(dead)}"
        )
        unexplained = [p for p in dead
                       if not any(p.endswith(s) for s in EXPECTED_OFF_GRAPH)]
        assert not unexplained, (
            f"off-graph weights with NO attributed cause: {sorted(unexplained)}"
        )

    def test_gradients_reach_every_other_trainable_weight_after_one_step(self):
        model = _built()
        x = clip_inputs(model)
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, expect_zero=EXPECTED_OFF_GRAPH)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)
        live = [p for p, v in report.items() if v is not None and v > 0.0]
        assert len(live) == GF_N_WEIGHTS - GF_N_OFF_GRAPH

    def test_the_memory_pathway_is_among_the_live_weights(self):
        """The claim the wrapper exists to make.

        A count of 304 live weights would be met by an image-encoder-only
        model; the memory attention and the memory encoder are what make this a
        VIDEO training path, and they are named rather than assumed.
        """
        model = _built()
        x = clip_inputs(model)
        _one_adam_step(model, x)
        report = gradient_report(model, x)
        for fragment in ("memory_attention", "memory_encoder"):
            live = [p for p, v in report.items()
                    if fragment in p and v is not None and v > 0.0]
            assert live, f"no LIVE weight under {fragment!r}"

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, clip_inputs(model))


class TestSAM2KnobSensitivity:

    def test_mem_dim_changes_the_memory_parameterisation(self):
        """A knob that reaches ONLY the memory path.

        ``mem_dim`` sets the memory token width and is consumed by both
        ``memory_attention`` and ``memory_encoder``; a model with no memory at
        all would be insensitive to it.
        """
        # `hidden_dim` is 32 at `tiny` and `mem_dim` must divide it.
        builders = {
            d: (lambda d=d: _built(mem_dim=d)) for d in (8, 16, 32)
        }
        assert_structural_knob_changes_weights(builders, knob="mem_dim")

    def test_num_frames_is_NOT_a_parameterisation_knob(self):
        """The negative control, stated rather than left implicit.

        The frame loop is unrolled over SHARED submodules, so ``num_frames``
        must NOT change the weight-shape signature -- a T-dependent parameter
        count would mean the wrapper had grown per-frame weights, which is the
        opposite of what it is for. The oracle's own failure message
        (``is a no-op``) is the PASSING condition here.
        """
        builders = {t: (lambda t=t: _built(num_frames=t)) for t in (2, 3)}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="num_frames")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="mem_dim")


class TestSAM2SmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _built()
        x = clip_inputs(model)
        grid = model.sam2.feature_grid * 4

        def contract(out):
            assert isinstance(out, dict), (
                f"SAM2TrainingModel returns a dict, got {type(out)}"
            )
            assert set(out) >= {SAM2_LOW_RES_LOGITS, SAM2_OBJECT_SCORE_LOGITS,
                                SAM2_IOU_SUPERVISION}, (
                f"missing an output key: {sorted(set(out))}"
            )
            logits = out[SAM2_LOW_RES_LOGITS]
            assert tuple(logits.shape) == (BATCH, FRAMES, grid, grid), (
                f"{SAM2_LOW_RES_LOGITS}: expected "
                f"{(BATCH, FRAMES, grid, grid)}, got {tuple(logits.shape)}"
            )
            scores = out[SAM2_OBJECT_SCORE_LOGITS]
            assert tuple(scores.shape) == (BATCH, FRAMES, 1), (
                f"{SAM2_OBJECT_SCORE_LOGITS}: expected {(BATCH, FRAMES, 1)}, "
                f"got {tuple(scores.shape)}"
            )
            for tensor in out.values():
                assert_finite(tensor)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
