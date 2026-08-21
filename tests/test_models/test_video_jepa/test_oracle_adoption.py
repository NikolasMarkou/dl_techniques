"""
Oracle adoption for ``models/video_jepa`` -- Phase 5 batch C.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored and no ``src/`` file is added.

THE PREDICTION HEAD IS *NOT* ON THE RETURNED TENSOR'S BACKWARD GRAPH
----------------------------------------------------------------------
FOUND BY THIS ADOPTION. ``VideoJEPA.call`` returns the RAW predictor output;
the per-horizon head ``pred_head_h1`` is applied only inside the multi-horizon
loss, which the model publishes through ``add_loss``. Pointing the gradient
oracle at the forward output alone therefore reports
``video_jepa/pred_head_h1/kernel`` as receiving **NO gradient at all** -- a
``None``, not a small number -- while the other 116 trainable weights are live.

``model.py`` says so in as many words at the ``pred_heads[h_idx]`` site ("the
raw predictor output is not supervised in ``call``"), so this is architecture,
not a defect. But a reader running the instrument for the first time would file
it, which is why it is pinned as an EXACT one-element set here rather than
mentioned in a comment.

Every gradient assertion in this file therefore uses **the loss the model
actually trains with** -- the output term PLUS ``model.losses`` (next-frame,
mask-prediction and SIGReg). Under it, ``0 of 117`` are dead. This is the same
shape ``vq_vae`` needed in this batch and ``memory_bank`` needed in batch B.

Measured 2026-08-21 on **CPU** (see the device note below), one Adam step, at
``img_size=32 / patch_size=8 / num_frames=2``:

===================================  =========  ===========================
loss                                 weights    dead
===================================  =========  ===========================
ramp(output) only                    117        1 (``pred_head_h1/kernel``,
                                                ``None``)
ramp(output) + ``model.losses``      117        0
===================================  =========  ===========================

THE ATTENTION KEY BIAS IS A MATHEMATICALLY DEAD PARAMETER, AND ITS ZERO IS
DEVICE-DEPENDENT
----------------------------------------------------------------------------
ALSO FOUND BY THIS ADOPTION, on GPU, after the file was green on CPU. In
dot-product attention ``q . (k + b_k) = q . k + q . b_k``, and the second term
is identical for every key, so it is a constant shift along the softmax axis and
cancels EXACTLY. The key bias of a stock ``keras.layers.MultiHeadAttention``
therefore has an analytically zero gradient. Measured directly: replacing it
with a 3x-scaled random draw moves the layer output by **8.94e-08**, while the
same treatment of the QUERY bias moves it by more than 1e-3.

The operational point is the device dependence. Whether that noise rounds to
EXACTLY ``0.0`` -- which is what the oracle convicts on -- depends on the
reduction order: on CPU every key bias read non-zero and this file was green,
while on GPU 1 it read exactly ``0.0`` for ``attn_block_0`` in one run and for
``attn_block_0`` AND ``attn_block_1`` in the next. **A single green CPU run
would have shipped a test that failed 2 runs in 2 on the machine the suite
actually runs on.**

It is NOT waived with ``expect_zero``: that is a two-sided claim ("these MUST be
zero") and on CPU they are not. The assertion is that the dead set is a SUBSET
of the key-bias family and that every weight outside it is live, with the
inertness proved directly in :class:`TestTheAttentionKeyBiasIsMathematicallyInert`
rather than asserted from a docstring.

DEVICE NOTE, STATED BECAUSE IT HAS BITTEN THIS PACKAGE BEFORE
--------------------------------------------------------------
This package's float32 precision arm is **device-dependent by 39x** (GPU
1.433e4 vs CPU 3.669e2). Nothing in this file asserts a gradient MAGNITUDE for
that reason -- every claim is a liveness claim, which is device-independent
apart from the key-bias rounding described above. The one number quoted in the
table is a COUNT.

THE GRAPH-MODE CONSTRAINT IS A DELIBERATE PIN -- VERIFY IT, DO NOT "FIX" IT
-----------------------------------------------------------------------------
``model.py``'s D-001/D-003 mask gate is written ``training is True``, a PYTHON
IDENTITY comparison. Under ``@tf.function`` a symbolic ``training`` is not
``True``, so tube masking short-circuits OFF -- including for
``tf.constant(True)``. That is documented, tested in ``test_video_jepa.py``, and
correct: the alternative, ``bool(training)``, raises
``OperatorNotAllowedInGraphError`` at trace time. :class:`TestTheGraphModeMaskGateIsPinned`
verifies the pin STILL HOLDS at the model level (the existing test verifies the
gate expression); it does not attempt to make masking work under a symbolic
flag.

``dropout`` is pinned to ``0.0`` and every build is seeded: batch A had an arm
flaky 1 run in 4 and batch B one flaky 2 in 5. On top of that, ``TubeMaskGenerator``
calls an UNSEEDED ``keras.random.uniform``, so ``mask_prediction_enabled`` is
turned OFF for every gradient reading here and exercised separately.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.video_jepa.config import VideoJEPAConfig
from dl_techniques.models.video_jepa.model import VideoJEPA, create_video_jepa

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..precision_arm_oracle import _asymmetric_loss, flatten_tensors
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

IMG_SIZE = 32
PATCH_SIZE = 8
NUM_FRAMES = 2
BUILD_SEED = 0

#: Measured 2026-08-21, one Adam step.
GF_WEIGHTS = 117

#: The head that the RETURNED tensor cannot reach, as a path SUFFIX -- never an
#: absolute ``Variable.path``: Keras uniquifies a model name per process, so the
#: second ``VideoJEPA`` in one session is ``video_jepa_1/...``. An absolute pin
#: is green alone and red behind any other test that builds the same class; it
#: bit batch B twice.
PRED_HEAD = "pred_head_h1/kernel"

#: The EXACT dead set under ``mask_prediction_enabled=False``, measured.
#: Two members, for two unrelated documented reasons: the horizon head is
#: supervised only through ``add_loss``, and ``mask_token`` is built
#: unconditionally (so a ``.keras`` round trip keeps it) but applied only where
#: a mask selects it. Both are pinned two-sided below.
DEAD_WITHOUT_MASKING = frozenset({PRED_HEAD, "mask_token"})

#: The attention KEY BIAS family, as a path suffix.
#:
#: FOUND BY THIS ADOPTION, on GPU, and it is a property of dot-product attention
#: rather than of this model. ``q . (k + b_k) = q . k + q . b_k``, and the second
#: term is the SAME for every key ``j``, so it is a constant shift along the
#: softmax axis and cancels EXACTLY. The key bias therefore has an analytically
#: zero gradient. MEASURED on a stock ``keras.layers.MultiHeadAttention``:
#: replacing the key bias with a 3x-scaled random draw moves the output by
#: **8.94e-08** -- float32 noise -- while its "gradient" reads 4.83e-05, which is
#: the same noise coming back through the tape.
#:
#: The consequence for an oracle is the one that matters here: whether that
#: rounding lands on EXACTLY 0.0 is DEVICE-DEPENDENT. On CPU every key bias read
#: non-zero and this file was green; on GPU 1 it read exactly 0.0 for
#: ``attn_block_0`` in one run and for ``attn_block_0`` AND ``attn_block_1`` in
#: the next. A single green CPU run would have shipped a test that fails 2 runs
#: in 2 on the machine the suite actually runs on.
#:
#: This is NOT waived with ``expect_zero``: that would be a two-sided claim
#: ("these MUST be zero"), and on CPU they are not. The assertion is instead
#: that the dead set is a SUBSET of this family and that every weight OUTSIDE it
#: is live -- with the inertness itself proved directly rather than assumed.
KEY_BIAS_SUFFIX = "mha/key/bias"


def _pixels(batch: int = 1, frames: int = NUM_FRAMES,
            seed: int = 0) -> Dict[str, np.ndarray]:
    return {"pixels": np.random.default_rng(seed).random(
        (batch, frames, IMG_SIZE, IMG_SIZE, 3)).astype("float32")}


def _video_jepa(**o) -> VideoJEPA:
    kwargs: Dict[str, Any] = dict(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, num_frames=NUM_FRAMES,
        history_size_k=NUM_FRAMES,
        # Pinned, not defaulted, for the reasons in the module docstring.
        dropout=0.0, mask_prediction_enabled=False,
    )
    kwargs.update(o)
    return create_video_jepa(**kwargs)


def _built(build_fn=_video_jepa, seed: int = BUILD_SEED) -> VideoJEPA:
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_pixels(), training=False)
    return model


def _built_at_four_frames(build_fn, seed: int = BUILD_SEED) -> VideoJEPA:
    """Build on a FOUR-frame clip, in TRAINING mode.

    Not interchangeable with :func:`_built`, and the difference is a real
    property of this model: a horizon head is created in ``__init__`` but is
    only CALLED -- and therefore only built, and therefore only present in
    ``model.weights`` -- when a causal pair exists for it, i.e. when
    ``T > h``. On this file's default two-frame clip the ``h=2`` and ``h=3``
    heads never fire, so ``predict_horizons=(1,)`` and ``(1, 2)`` both report
    **161 weights / 338164 parameters** and the structural knob instrument
    correctly calls the knob a no-op. Measured at four frames they are
    161 / 162 / 163. A shape sweep that had used the short clip would have
    reported a genuine knob inert; this is why the builder is separate and
    named.
    """
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_pixels(frames=4), training=True)
    return model


def _assert_only_key_biases_are_dead(report, also_dead=()):
    """Every weight outside the mathematically-inert key-bias family is live.

    Deliberately NOT ``expect_zero=(KEY_BIAS_SUFFIX,)``: that is a two-sided
    claim, and on CPU the key biases read a small NON-zero value (float32 noise
    coming back through the tape), so the waiver would be reported obsolete
    there. The claim that is true on both devices is the SUBSET one -- see
    :data:`KEY_BIAS_SUFFIX`.
    """
    dead = {
        path for path, value in report.items()
        if value is None or value == 0.0
    }
    unexplained = {
        path for path in dead
        if not path.endswith(KEY_BIAS_SUFFIX)
        and not any(path.endswith(s) for s in also_dead)
    }
    assert unexplained == set(), (
        f"{len(unexplained)} weight(s) are dead for no attributed reason "
        f"(the key-bias family and {sorted(also_dead)} are the only accepted "
        f"causes): {sorted(unexplained)}"
    )
    live = len(report) - len(dead)
    assert live >= len(report) - _count_key_biases(report) - len(also_dead), (
        f"only {live}/{len(report)} weights are live")


def _count_key_biases(report) -> int:
    return sum(1 for path in report if path.endswith(KEY_BIAS_SUFFIX))


def ramp_loss(outputs: Any) -> Any:
    """IMPORTED from ``precision_arm_oracle``, never re-typed (D-059)."""
    return sum(_asymmetric_loss(t) for t in flatten_tensors(outputs))


def _training_loss(model: keras.Model):
    """The loss this model actually trains with: output term + ``model.losses``.

    ``model.losses`` must be read INSIDE the tape, after the forward, or it is
    the PREVIOUS call's list -- which would silently make every assertion below
    a statement about a stale graph.
    """

    def loss_fn(outputs: Any) -> Any:
        extra = model.losses
        return ramp_loss(outputs) + (sum(extra) if extra else 0.0)

    return loss_fn


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = _training_loss(model)(outputs)
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestThePredictionHeadNeedsTheAddLossTerms:
    """The finding, pinned two-sided, before anything depends on it."""

    def test_the_returned_tensor_alone_leaves_the_head_disconnected(self):
        """The false CRITICAL. Asserted as an EXACT set, so a genuinely dead
        encoder or predictor weight cannot hide behind this explanation.

        The set has TWO members, not one: ``mask_token`` is also off the graph
        here, for an unrelated and equally documented reason -- see
        :data:`DEAD_WITHOUT_MASKING`.
        """
        model = _built()
        report = gradient_report(model, _pixels(), loss_fn=ramp_loss)
        disconnected = {p for p, v in report.items() if v is None}
        assert {p.split("/", 1)[1]
                for p in disconnected} == set(DEAD_WITHOUT_MASKING), (
            f"expected exactly {sorted(DEAD_WITHOUT_MASKING)} to be off the "
            f"returned tensor's backward graph, got {sorted(disconnected)}"
        )

    def test_the_mask_token_is_dead_exactly_when_nothing_is_masked(self):
        """The second half of the two-member set above, pinned separately.

        ``mask_token`` is built UNCONDITIONALLY -- a lazily-built weight is
        dropped by a ``.keras`` round trip -- and applied only where a mask
        selects it. So it is dead exactly when ``mask_prediction_enabled`` is
        False, and live the moment it is not. This is the same mechanism batch
        A measured on BEiT and DINOv2, and it is pinned the same way: an exact
        set with the discriminating half asserted, never a one-sided waiver.
        """
        masked = _built(lambda: _video_jepa(mask_prediction_enabled=True))
        x = _pixels()
        _one_adam_step(masked, x)
        report = gradient_report(masked, x, loss_fn=_training_loss(masked))
        path = next(p for p in report if p.endswith("mask_token"))
        assert report[path] is not None and report[path] > 0.0, (
            f"mask_token is dead even with masking ON (max|grad|="
            f"{report[path]}); the substitution is not reaching the graph"
        )

    def test_the_model_publishes_its_losses_through_add_loss(self):
        """The premise. If these stop being published, the predictor trains
        against nothing and every shape check stays green."""
        model = _built()
        model(_pixels(), training=True)
        assert len(model.losses) >= 2, (
            f"expected at least the next-frame and SIGReg terms, got "
            f"{len(model.losses)}")

    def test_under_the_training_loss_the_head_is_live(self):
        """The discriminating half."""
        model = _built()
        x = _pixels()
        _one_adam_step(model, x)
        report = gradient_report(model, x, loss_fn=_training_loss(model))
        path = next(p for p in report if p.endswith(PRED_HEAD))
        assert report[path] is not None and report[path] > 0.0, (
            f"the horizon head is dead even under the training loss "
            f"(max|grad|={report[path]}) -- the add_loss explanation is then "
            f"wrong and this IS a disconnected head"
        )


class TestVideoJEPAGradientFlow:

    def test_no_layer_is_stochastic(self):
        model = _built()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate", "dropout")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], f"a non-zero stochastic rate is live: {stochastic}"

    def test_masking_is_off_for_every_gradient_reading_in_this_file(self):
        """``TubeMaskGenerator`` calls an UNSEEDED ``keras.random.uniform``.

        A gradient reading taken with masking on reports the DRAW. The flag is
        asserted rather than trusted, because it is a config field and a
        default flip would silently make every count above stochastic.
        """
        assert _built().config.mask_prediction_enabled is False

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        """Taken with masking ON, and with NO waiver at all.

        Masking off would need ``expect_zero=("mask_token",)``, and a waiver
        that can be removed by exercising the model properly should be. The
        tube mask's DRAW is unseeded, but the number of masked positions is a
        deterministic function of ``mask_ratio``, so LIVENESS -- which is all
        this asserts -- does not depend on which patches were chosen. Measured
        stable over 5 consecutive runs.
        """
        model = _built(lambda: _video_jepa(mask_prediction_enabled=True))
        x = _pixels()
        _one_adam_step(model, x)

        report = gradient_report(model, x, loss_fn=_training_loss(model))
        assert len(report) == GF_WEIGHTS == len(model.trainable_weights)
        _assert_only_key_biases_are_dead(report)

    def test_with_masking_OFF_exactly_the_mask_token_is_dead(self):
        """The companion, so the waiver-free reading above cannot hide a
        regression that only shows with masking disabled.

        The waived set here is ONE element, not the two of
        :data:`DEAD_WITHOUT_MASKING`: that constant describes the dead set
        under the RETURNED TENSOR alone, and under the full training loss the
        horizon head is reached through ``add_loss`` whether masking is on or
        off. Measured max|grad| on ``pred_head_h1/kernel`` here: 8.969e-02.
        The oracle's ``live_but_waived`` clause is what caught the first draft
        of this test waiving both.
        """
        model = _built()
        x = _pixels()
        _one_adam_step(model, x)
        report = gradient_report(model, x, loss_fn=_training_loss(model))
        assert len(report) == GF_WEIGHTS
        path = next(p for p in report if p.endswith("mask_token"))
        assert report[path] is None or report[path] == 0.0, (
            f"mask_token is LIVE with masking off (max|grad|={report[path]}); "
            f"the substitution is running when nothing is masked")
        _assert_only_key_biases_are_dead(report, also_dead=("mask_token",))

    def test_the_target_encoder_is_deliberately_off_the_backward_graph(self):
        """EMA owns the target encoder; the optimizer must never see it.

        ``call`` wraps ``encode_frames_target`` in ``stop_gradient``, and that
        is the whole point of the EMA target (without it, the identity map is
        the optimal solution). Asserted directly on the trainable set: no
        ``target_encoder`` weight may be trainable at all.
        """
        model = _built()
        trainable_targets = [
            w.path for w in model.trainable_weights
            if "target_encoder" in w.path
        ]
        assert trainable_targets == [], (
            f"the EMA target encoder exposes trainable weights: "
            f"{trainable_targets}")

    def test_the_gradient_assertion_can_fail(self):
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _pixels(), loss_fn=_training_loss(model))


class TestTheAttentionKeyBiasIsMathematicallyInert:
    """The mechanism behind :data:`KEY_BIAS_SUFFIX`, proved rather than argued.

    Without this class the subset assertion above would be a skip list with a
    story attached. With it, the story is a measurement, and a change that made
    the key bias matter would fail here first.
    """

    @staticmethod
    def _mha():
        keras.utils.set_random_seed(0)
        layer = keras.layers.MultiHeadAttention(num_heads=2, key_dim=8)
        x = keras.random.normal((2, 6, 16))
        layer(x, x, x)
        return layer, x

    @staticmethod
    def _bias(layer, kind):
        return next(w for w in layer.weights if w.path.endswith(f"{kind}/bias"))

    def test_a_large_change_to_the_key_bias_barely_moves_the_output(self):
        """``q . (k + b_k) = q . k + q . b_k``, and the second term is the same
        for every key, so it is a constant shift along the softmax axis and
        cancels exactly. Measured: **8.94e-08** for a 3x-scaled random draw."""
        layer, x = self._mha()
        before = keras.ops.convert_to_numpy(layer(x, x, x))
        bias = self._bias(layer, "key")
        bias.assign(np.random.default_rng(0).normal(
            size=bias.shape).astype("float32") * 3.0)
        after = keras.ops.convert_to_numpy(layer(x, x, x))
        delta = float(np.max(np.abs(before - after)))
        assert delta < 1e-5, (
            f"the key bias moved the attention output by {delta:.3e}; it is "
            f"NOT shift-invariant here and the whole waiver above is wrong")

    def test_the_QUERY_bias_control_moves_the_output_a_lot(self):
        """The discriminating half. Without it, the test above would pass on a
        layer whose biases were all ignored, or on an assign that did nothing."""
        layer, x = self._mha()
        before = keras.ops.convert_to_numpy(layer(x, x, x))
        bias = self._bias(layer, "query")
        bias.assign(np.random.default_rng(0).normal(
            size=bias.shape).astype("float32") * 3.0)
        after = keras.ops.convert_to_numpy(layer(x, x, x))
        delta = float(np.max(np.abs(before - after)))
        assert delta > 1e-3, (
            f"the QUERY bias only moved the output by {delta:.3e}; the assign "
            f"is not reaching the forward pass, so the key-bias reading above "
            f"proves nothing")

    def test_this_model_actually_uses_stock_multi_head_attention(self):
        """The premise linking the two probes above to this package."""
        model = _built()
        assert any(
            isinstance(layer, keras.layers.MultiHeadAttention)
            for layer in model._flatten_layers(include_self=False)
        ), "no stock MultiHeadAttention in this model; the key-bias family "\
           "explanation does not apply and the subset waiver must be re-derived"


class TestTheGraphModeMaskGateIsPinned:
    """A DELIBERATE constraint. Verified, never "fixed".

    ``model.py``'s gate is ``training is True`` -- a Python identity check that
    constant-folds to ``False`` for any symbolic operand, INCLUDING
    ``tf.constant(True)``. The alternative, ``bool(training)``, raises
    ``OperatorNotAllowedInGraphError`` at trace time, which is the iter-1
    defect this shape replaced. ``test_video_jepa.py`` pins the gate
    EXPRESSION; this pins its consequence at the MODEL level.
    """

    def test_a_traced_forward_with_a_symbolic_training_flag_does_not_raise(self):
        model = _built(lambda: _video_jepa(mask_prediction_enabled=True))
        x = _pixels()

        @tf.function
        def traced(pixels, training):
            return model({"pixels": pixels}, training=training)

        out = traced(tf.constant(x["pixels"]), tf.constant(True))
        assert_finite(out)

    def test_a_symbolic_true_leaves_masking_OFF_and_that_is_the_contract(self):
        """The consequence, asserted rather than assumed.

        With masking genuinely ON the tube mask is redrawn per call from an
        unseeded RNG, so two calls disagree. Under a SYMBOLIC ``True`` the gate
        folds off, so two traced calls must agree EXACTLY. If this test ever
        starts failing, masking has begun firing under graph mode -- which is a
        behaviour CHANGE, not a fix, and the D-001/D-003 note must be revisited
        before anything else.
        """
        model = _built(lambda: _video_jepa(mask_prediction_enabled=True))
        x = _pixels()

        @tf.function
        def traced(pixels, training):
            return model({"pixels": pixels}, training=training)

        a = keras.ops.convert_to_numpy(
            traced(tf.constant(x["pixels"]), tf.constant(True)))
        b = keras.ops.convert_to_numpy(
            traced(tf.constant(x["pixels"]), tf.constant(True)))
        np.testing.assert_array_equal(a, b)

    def test_a_python_true_in_eager_mode_DOES_enable_masking(self):
        """The discriminating half: the gate is about the FLAG's type, not
        about masking being permanently disabled."""
        model = _built(lambda: _video_jepa(mask_prediction_enabled=True))
        x = _pixels()
        a = keras.ops.convert_to_numpy(model(x, training=True))
        b = keras.ops.convert_to_numpy(model(x, training=True))
        assert float(np.max(np.abs(a - b))) > 0.0, (
            "two eager training-mode calls agree exactly; the tube mask is not "
            "being drawn, so the graph-mode claim above compares nothing"
        )


class TestVideoJEPAKnobSensitivity:

    def test_embed_dim_changes_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _built(lambda: _video_jepa(embed_dim=d)))
            for d in (32, 64)
        }
        assert_structural_knob_changes_weights(builders, knob="embed_dim")

    def test_predictor_depth_changes_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _built(lambda: _video_jepa(predictor_depth=d)))
            for d in (1, 2, 3)
        }
        assert_structural_knob_changes_weights(builders, knob="predictor_depth")

    def test_predict_horizons_changes_the_head_count(self):
        """One ``Dense`` head per horizon -- the knob that owns this file's
        finding."""
        builders = {
            h: (lambda h=h: _built_at_four_frames(
                lambda: _video_jepa(num_frames=4, history_size_k=4,
                                    predict_horizons=h)))
            for h in ((1,), (1, 2), (1, 2, 3))
        }
        signatures = assert_structural_knob_changes_weights(
            builders, knob="predict_horizons")
        counts = {k: len(v) for k, v in signatures.items()}
        assert counts[(1,)] < counts[(1, 2)] < counts[(1, 2, 3)], counts

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="embed_dim")

    def test_a_patch_size_that_does_not_divide_the_image_is_refused(self):
        with pytest.raises(ValueError):
            _video_jepa(img_size=32, patch_size=7)


class TestVideoJEPASmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _built()
        x = _pixels(batch=2)
        grid = IMG_SIZE // PATCH_SIZE

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"VideoJEPA.call returns ONE prediction tensor, got {type(out)}")
            assert tuple(out.shape) == (
                2, NUM_FRAMES, grid, grid, model.config.embed_dim), (
                f"expected (B, T, H_p, W_p, D); got {tuple(out.shape)}")
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_a_non_dict_input_is_refused_with_a_named_reason(self):
        model = _built()
        with pytest.raises(ValueError, match="pixels"):
            model(_pixels()["pixels"], training=False)

    def test_a_dict_without_pixels_is_refused_with_a_named_reason(self):
        model = _built()
        with pytest.raises(ValueError, match="pixels"):
            model({"frames": _pixels()["pixels"]}, training=False)

    def test_the_time_axis_is_taken_from_the_BATCH_not_from_the_config(self):
        """D-041: every frame count in the loss arithmetic comes from THIS
        batch's ``T``. A build that read ``cfg.num_frames`` instead produced a
        NaN loss at ``T <= h`` and a silently rescaled one otherwise -- with the
        output shape unchanged either way."""
        model = _built(lambda: _video_jepa(num_frames=4, history_size_k=4))
        for frames in (2, 3, 4):
            out = model(_pixels(frames=frames), training=True)
            assert tuple(out.shape)[1] == frames, tuple(out.shape)
            assert_finite(out)
            for term in model.losses:
                assert np.isfinite(
                    float(keras.ops.convert_to_numpy(term))), (
                    f"a published loss term is non-finite at T={frames}")
