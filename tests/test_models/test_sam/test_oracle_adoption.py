"""
Oracle adoption for ``models/SAM/SAM1`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

WHAT IS MEASURED, AND WHY IT IS THE WRAPPER
--------------------------------------------
``SAM`` itself cannot be traced (``TypeError: len is not well defined for a
symbolic Tensor``, at ``jit_compile`` both ``False`` and ``True``), and its
``fit()`` path is ``SAMTrainingModel``. Every measurement here therefore runs on
the wrapper, which is the object that actually trains. Weights are seeded
non-zero first (``seed_nonzero_weights``): many SAM weights initialize to
exactly zero, and a liveness probe on an all-zero weight can be structurally
unable to observe what it claims to measure.

PROMPT COVERAGE IS THE WHOLE MEASUREMENT
-----------------------------------------
Measured 2026-08-21 (GPU 1) at the reduced fixture (201 trainable weights), one
real Adam step, ``default_loss``:

===================================================  ====  ===================
input                                                dead  what the dead set is
===================================================  ====  ===================
1 foreground point, no box, no mask, 1 round,        31    9 mask-downscaling +
``multimask_output=False``                                 2 box-corner + 1
                                                           background-label +
                                                           18 hypernet 1-3
fg + bg + padding labels, a box, 2 refinement         6    hypernet 0 only
rounds, ``multimask_output=True``
===================================================  ====  ===================

**None of the 31 is a defect; all of them are input coverage.** Each prompt
branch of a promptable segmenter is unreachable unless that prompt is supplied,
which is precisely why an under-prompted smoke test can leave two thirds of the
prompt encoder unexercised and report green. The full-coverage arm is the main
assertion.

The residual 6 are the reference architecture, not a gap: the decoder holds
``num_multimask_outputs + 1`` mask tokens, index 0 is returned at
``multimask_output=False`` and indices 1..3 at ``True``, so NEITHER setting alone
reaches all four hypernetwork MLPs. ``test_both_settings_together_cover_every_
hypernetwork`` asserts the union does and the intersection is empty -- which is
a stronger statement than either arm could make, and the only honest one.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.SAM.SAM1 import SAMTrainingModel
from dl_techniques.models.SAM.SAM1.training_model import (
    INPUT_BOXES,
    INPUT_GT_MASK,
    INPUT_IMAGE,
    INPUT_POINT_COORDS,
    INPUT_POINT_LABELS,
    IOU_PREDICTIONS,
    LOW_RES_LOGITS,
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
from .test_correctness import (
    GRID_SIZE,
    IMG_SIZE,
    build_reduced_sam,
    seed_nonzero_weights,
)

BATCH = 2
POINTS = 3
LOW_RES = 4 * GRID_SIZE
#: Refinement rounds. Two, so the second round feeds a MASK prompt back in and
#: `mask_downscaling` is reachable at all.
ROUNDS = 2

#: Measured 2026-08-21 at the reduced fixture.
GF_N_WEIGHTS = 201

#: The six weights of mask-token 0's hypernetwork MLP, matched by path SUFFIX.
#: Suffixes rather than absolute ``Variable.path`` strings because Keras
#: uniquifies a model's name per process: the second ``SAMTrainingModel`` built
#: in one pytest session is ``sam_training_model_1/...``, so an absolute pin is
#: green alone and red behind any other test in the same session.
HYPERNET_0 = frozenset(
    f"hypernetwork_mlp_0/hyper_dense{i}_0/{w}"
    for i in (1, 2, 3) for w in ("kernel", "bias")
)

#: The eighteen weights of mask-tokens 1..3's hypernetwork MLPs.
HYPERNET_123 = frozenset(
    f"hypernetwork_mlp_{m}/hyper_dense{i}_{m}/{w}"
    for m in (1, 2, 3) for i in (1, 2, 3) for w in ("kernel", "bias")
)


def _full_coverage_inputs(num_masks: int = 3, seed: int = 0) -> dict:
    """Every prompt branch the wrapper can reach, in one batch.

    Foreground / background / padding point labels, a box, and a GT mask so the
    second refinement round feeds a MASK prompt back in.
    """
    rng = np.random.RandomState(seed)
    # The GT mask axis must match the wrapper's own mask count -- 3 at
    # `multimask_output=True`, 1 at False -- or `match_mask_axis` refuses.
    gt = np.zeros((BATCH, num_masks, LOW_RES, LOW_RES), dtype="float32")
    gt[:, :, 12:40, 20:52] = 1.0
    return {
        INPUT_IMAGE: rng.uniform(
            0.0, 255.0, (BATCH, IMG_SIZE, IMG_SIZE, 3)).astype("float32"),
        INPUT_POINT_COORDS: rng.uniform(
            0.0, float(IMG_SIZE), (BATCH, POINTS, 2)).astype("float32"),
        INPUT_POINT_LABELS: np.tile(
            np.array([[1, 0, -1]], dtype="int32"), (BATCH, 1)),
        INPUT_BOXES: np.tile(
            np.array([[[10.0, 20.0, 100.0, 120.0]]], dtype="float32"),
            (BATCH, 1, 1)),
        INPUT_GT_MASK: gt,
    }


def _wrapper(multimask_output: bool = True, seed: int = 7,
             **sam_overrides) -> SAMTrainingModel:
    keras.utils.set_random_seed(seed)
    model = SAMTrainingModel(
        build_reduced_sam(**sam_overrides),
        multimask_output=multimask_output,
        num_refinement_rounds=ROUNDS,
    )
    model(_full_coverage_inputs(3 if multimask_output else 1))
    seed_nonzero_weights(model)
    return model


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


def _dead_set(model: keras.Model, inputs) -> set:
    report = gradient_report(model, inputs)
    return {p for p, v in report.items() if v is None or v == 0.0}


def _matches(paths, suffixes) -> bool:
    return (len(paths) == len(suffixes)
            and all(any(p.endswith(s) for s in suffixes) for p in paths)
            and all(any(p.endswith(s) for p in paths) for s in suffixes))


class TestSAM1GradientFlow:

    def test_no_layer_is_stochastic(self):
        """SAM 1 has no stochastic depth and no dropout at all."""
        model = _wrapper()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], (
            f"a non-zero stochastic rate is live: {stochastic}. A gradient "
            f"report taken under one reports the DRAW, not the model"
        )

    def test_gradients_reach_every_weight_but_the_unused_mask_token(self):
        model = _wrapper(multimask_output=True)
        x = _full_coverage_inputs(3)
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, expect_zero=sorted(HYPERNET_0))

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)

    def test_both_settings_together_cover_every_hypernetwork(self):
        """The union claim -- see the module docstring.

        ``expect_zero`` is two-sided, so each arm above already proves its own
        waived weights ARE dead. This adds the half neither arm can state: the
        two shipped settings between them leave no hypernetwork untrained, and
        they share no dead weight.
        """
        multimask = _wrapper(multimask_output=True)
        x_multi = _full_coverage_inputs(3)
        _one_adam_step(multimask, x_multi)
        dead_multimask = _dead_set(multimask, x_multi)

        single = _wrapper(multimask_output=False)
        x_single = _full_coverage_inputs(1)
        _one_adam_step(single, x_single)
        dead_single = _dead_set(single, x_single)

        assert _matches(dead_multimask, HYPERNET_0), (
            f"multimask arm: expected exactly mask-token 0's hypernetwork, got "
            f"{sorted(dead_multimask)}"
        )
        assert _matches(dead_single, HYPERNET_123), (
            f"single-mask arm: expected exactly mask-tokens 1..3, got "
            f"{sorted(dead_single)}"
        )
        def hypernet(path):
            return next(seg for seg in path.split("/")
                        if seg.startswith("hypernetwork_mlp_"))

        shared = ({hypernet(p) for p in dead_multimask}
                  & {hypernet(p) for p in dead_single})
        assert not shared, (
            f"the two settings share a dead hypernetwork: {shared}; the union "
            f"claim is then false"
        )

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _wrapper()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _full_coverage_inputs(3))


class TestSAM1KnobSensitivity:

    def test_encoder_depth_changes_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _wrapper(depth=d)) for d in (2, 3, 4)
        }
        assert_structural_knob_changes_weights(builders, knob="depth")

    def test_use_rel_pos_changes_the_attention_parameterisation(self):
        """A knob that reaches ONLY the ViT blocks' relative-position tables.

        ``depth`` above would still pass if ``use_rel_pos`` were ignored. This
        one would not: at ``False`` the ``rel_pos_h/w`` weights are not created.
        """
        builders = {
            flag: (lambda flag=flag: _wrapper(use_rel_pos=flag))
            for flag in (False, True)
        }
        assert_structural_knob_changes_weights(builders, knob="use_rel_pos")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _wrapper()), "b": (lambda: _wrapper())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="depth")


class TestSAM1SmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _wrapper(multimask_output=True)
        x = _full_coverage_inputs(3)
        # `low_res_logits` carries `num_masks * num_refinement_rounds` masks,
        # concatenated ROUND-major -- 3 * 2 here, not 3.
        packed_masks = 3 * ROUNDS

        def contract(out):
            assert isinstance(out, dict), (
                f"SAMTrainingModel returns a dict, got {type(out)}"
            )
            assert set(out) >= {LOW_RES_LOGITS, IOU_PREDICTIONS}, (
                f"missing an output key: {sorted(set(out))}"
            )
            logits = out[LOW_RES_LOGITS]
            iou = out[IOU_PREDICTIONS]
            assert tuple(logits.shape) == (
                BATCH, packed_masks, LOW_RES, LOW_RES), (
                f"{LOW_RES_LOGITS}: expected "
                f"{(BATCH, packed_masks, LOW_RES, LOW_RES)}, got "
                f"{tuple(logits.shape)}"
            )
            assert tuple(iou.shape) == (BATCH, packed_masks), (
                f"{IOU_PREDICTIONS}: expected {(BATCH, packed_masks)}, got "
                f"{tuple(iou.shape)}"
            )
            assert_finite(logits)
            assert_finite(iou)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
