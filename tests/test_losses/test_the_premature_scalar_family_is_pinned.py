"""An EXECUTABLE follow-up record for the premature-scalar loss family.

Plan ``plan-2026-08-30T203107-30455f66`` fixed four classes
(``SigLIPContrastiveLoss``, ``AdaptiveSigLIPLoss``, ``HybridContrastiveLoss``,
``HuberLoss``) whose ``call()`` returned a SCALAR instead of one value per
sample. It did NOT fix the rest of the family. This file is that remainder,
written as a test rather than as a prose TODO: every member below is MEASURED
to still carry the defect, so the day someone fixes one, this file goes RED and
whoever fixed it updates the list.

**The defect predicate, measured not grepped.** ``keras.losses.Loss.__call__``
multiplies ``call()``'s output by ``sample_weight`` and only THEN reduces. If
``call()`` returned a scalar, that scalar broadcasts, and the result is exactly
``unweighted * mean(sample_weight)`` -- every row is charged the batch
aggregate and WHICH rows were weighted is discarded. So::

    loss(y, p, sample_weight=[1, 1, 1, 0]) == loss(y, p) * 0.75

holds if and only if the class is a member. ``reduction=`` is dead for the same
reason. A correctly shaped class fails that equality (see the anti-vacuity arm).

**Membership was derived by EXECUTION, and it grew.** The findings that opened
the plan listed 8 modules; measuring found 14 classes across 12 modules. Four of
them -- ``DINOLoss``, ``KoLeoLoss``, ``ScaledMseLoss``, ``SAMIoULoss`` -- were
NOT on that list and were found only by running the predicate.

**An empty ``STILL_BROKEN`` is the SUCCESS state of this file, not a dead file.**
Plan ``plan-2026-08-31T045723-c0d5ffa9`` is fixing the members one at a time; each
fix MOVES its row from ``STILL_BROKEN`` to ``KNOWN_GOOD`` in the same commit,
where the same predicate is then run in the negative direction. Deleting a row is
never the right response to a red test here. The ``KNOWN_GOOD`` arm is what keeps
the file non-vacuous once ``STILL_BROKEN`` is empty: injecting a scalar return
into any fixed class must turn it red.

**NOT MEASURED (named, never silently dropped).** These are neither pinned nor
cleared by this file:

*   ``hrm_loss.StableMaxCrossEntropy`` -- measured, and it is NOT a member: its
    ``call()`` returns ``(batch, seq_len)``. A ``(batch,)`` ``sample_weight``
    then RAISES ``InvalidArgumentError: Incompatible shapes: [4,5] vs. [4]`` --
    but so does stock ``keras.losses.SparseCategoricalCrossentropy`` on the
    identical inputs, so this is the Keras convention for a TOKEN-LEVEL loss and
    not a defect at all. The earlier wording here called it "a different defect
    class"; that was refuted by measurement on 2026-08-31 and the class is
    documented rather than changed. See ``decisions.md`` D-002 of
    ``plan-2026-08-31T045723-c0d5ffa9``.
*   ``yolo12_multitask_loss`` (4 classes) and ``nano_vlm_loss.NanoVLMLoss`` --
    require structured multi-head ``y_pred`` dicts that cannot be built cheaply
    here.
*   The remaining ~60 ``keras.losses.Loss`` subclasses in
    ``src/dl_techniques/losses/`` were NOT swept. A static scan of their
    ``call()`` return expressions shows further candidates (for example
    ``multi_labels_loss`` and ``utilization_loss``).
    ``brier_spiegelhalters_ztest_loss`` was on that list and its three loss
    classes were fixed on 2026-09-02; they are in ``KNOWN_GOOD`` below.
    **This file is a floor, not a census of the package.**
"""

import keras
import numpy as np
import pytest

PLAN = "plan-2026-08-30T203107-30455f66"

BATCH = 4
SAMPLE_WEIGHT = np.array([1.0, 1.0, 1.0, 0.0], dtype="float32")
MEAN_W = 0.75


def _t(x):
    return keras.ops.convert_to_tensor(np.asarray(x, dtype="float32"))


def _i(x):
    return keras.ops.convert_to_tensor(np.asarray(x, dtype="int32"))


def _rng():
    """One fixed generator per builder call: inputs are identical every run."""
    return np.random.default_rng(1234)


# ---------------------------------------------------------------------
# input builders -- one per class, returning (y_true, y_pred)
# ---------------------------------------------------------------------


def _regression_pair():
    r = _rng()
    return _t(r.normal(size=(BATCH, 6)) * 2.0), _t(r.normal(size=(BATCH, 6)) * 2.0)


def _positive_pair():
    r = _rng()
    return (
        _t(np.abs(r.normal(size=(BATCH, 6))) + 1.0),
        _t(np.abs(r.normal(size=(BATCH, 6))) + 1.0),
    )


def _image_pair():
    r = _rng()
    return _t(r.normal(size=(BATCH, 4, 4, 1))), _t(r.normal(size=(BATCH, 4, 4, 1)))


def _quantile_pair():
    r = _rng()
    return _t(r.normal(size=(BATCH, 5))), _t(r.normal(size=(BATCH, 5, 3)))


def _token_pair():
    r = _rng()
    return _i(r.integers(0, 10, size=(BATCH, 5))), _t(r.normal(size=(BATCH, 5, 10)))


def _hrm_pair():
    r = _rng()
    y_true = {
        "labels": _i(r.integers(0, 10, size=(BATCH, 5))),
        "halted": _t(np.zeros((BATCH,))),
    }
    y_pred = {
        "logits": _t(r.normal(size=(BATCH, 5, 10))),
        "q_halt_logits": _t(r.normal(size=(BATCH,))),
        "q_continue_logits": _t(r.normal(size=(BATCH,))),
    }
    return y_true, y_pred


def _embedding_pair():
    r = _rng()
    return _t(r.normal(size=(BATCH, 5))), _t(r.normal(size=(BATCH, 5)))


def _iou_pair():
    r = _rng()
    return _t(r.random((BATCH, 3))), _t(r.random((BATCH, 3, 2)))


def _logit_pair():
    r = _rng()
    return _t(np.eye(BATCH, 5)), _t(r.normal(size=(BATCH, 5)))


def _calibration_pair():
    """Binary outcomes with an OVER-CONFIDENT, WRONG half.

    The generic pairs above do not work for the calibration losses: their
    default penalty is ``relu(Z**2 - 1) / N``, which is exactly ZERO whenever
    the batch is within chance of calibrated, and ``0 == 0 * 0.75`` would make
    the predicate report every one of them a member. Measured on this fixture:
    ``Z_sh = 2.67``, ``Z_sh**2 = 7.11``, so the chance gate is open.
    """
    return (
        _t([[1.0], [0.0], [1.0], [0.0]]),
        _t([[0.9], [0.1], [0.1], [0.9]]),
    )


def _siglip_pair():
    r = _rng()
    a = r.normal(size=(BATCH, 8))
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b = r.normal(size=(BATCH, 8))
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    m = (a @ b.T).astype("float32")
    return _t(np.zeros((BATCH, 1))), {
        "logits_per_image": _t(m),
        "logits_per_text": _t(m.T),
    }


# ---------------------------------------------------------------------
# the pinned membership, derived by execution 2026-08-31
# ---------------------------------------------------------------------

# (module, class, ctor kwargs, input builder)
# EMPTY IS THE SUCCESS STATE. All 14 measured members have been fixed and every
# one of them is in KNOWN_GOOD below -- none was deleted. The two arms
# parametrized on this list therefore contribute no test nodes; that is expected
# and is NOT the file going vacuous. The KNOWN_GOOD arm is what keeps the
# predicate honest, and `test_the_pinned_population_has_not_shrunk_silently`
# guards the TOTAL, so a member cannot leave measurement in either direction.
# A new member found later is APPENDED here with its measurement.
STILL_BROKEN = []

# Classes that MUST be rejected by the same predicate. Without this arm the
# predicate could be vacuously true for everything.
KNOWN_GOOD = [
    # fixed by this plan, steps 4 and 5
    ("huber_loss", "HuberLoss", {}, _regression_pair),
    ("siglip_contrastive_loss", "SigLIPContrastiveLoss", {}, _siglip_pair),
    ("siglip_contrastive_loss", "AdaptiveSigLIPLoss", {}, _siglip_pair),
    ("siglip_contrastive_loss", "HybridContrastiveLoss", {}, _siglip_pair),
    # correct all along -- the siblings whose shape discipline was copied
    ("goodhart_loss", "GoodhartAwareLoss", {}, _logit_pair),
    ("focal_uncertainty_loss", "FocalUncertaintyLoss", {}, _logit_pair),
    # fixed by plan-2026-08-31T045723-c0d5ffa9 step 2 (Tranche A batch 1). Each was
    # proven value-unchanged against its pre-edit module on a RAGGED fixture at
    # atol=1e-6 before being moved here.
    ("feature_alignment_loss", "FeatureAlignmentLoss", {}, _regression_pair),
    ("quantile_loss", "MQLoss", {"quantile": 0.7}, _regression_pair),
    ("quantile_loss", "QuantileLoss", {"quantiles": [0.1, 0.5, 0.9]}, _quantile_pair),
    (
        "quantile_loss",
        "QuantileLoss",
        {"quantiles": [0.1, 0.5, 0.9], "normalize": True},
        _quantile_pair,
    ),
    ("affine_invariant_loss", "AffineInvariantLoss", {}, _image_pair),
    ("smape_loss", "SMAPELoss", {}, _positive_pair),
    # fixed by plan-2026-08-31T045723-c0d5ffa9 step 3 (Tranche A batch 2), same
    # two-part proof. `ScaledMseLoss`'s invariant (a) is exact in float64; in
    # float32 at the ragged fixture's 500x row scale the two summation orders
    # differ by exactly ONE ulp (200654.328125 vs 200654.34375, and float64 says
    # 200654.34474678 -- the new order is the closer of the two).
    ("scaled_mse_loss", "ScaledMseLoss", {}, _image_pair),
    ("sam_mask_loss", "SAMIoULoss", {}, _iou_pair),
    # fixed by plan-2026-08-31T045723-c0d5ffa9 step 4 (Tranche A batch 3). These
    # two are BATCH-COUPLED: a row's value is computed against the other rows, so
    # `sample_weight=0` removes a row's loss CONTRIBUTION but not its INFLUENCE
    # (as a neighbour for KoLeo, as a participant in the centering EMA for DINO).
    # The per-row attribution is still honest and both docstrings now say so.
    # `_charges_every_row_the_batch_aggregate` builds a FRESH instance per call,
    # which is what keeps DINOLoss's stateful centering out of this measurement.
    ("dino_loss", "DINOLoss", {"out_dim": 5}, _embedding_pair),
    ("dino_loss", "KoLeoLoss", {}, _embedding_pair),
    # fixed by plan-2026-08-31T045723-c0d5ffa9 step 5 (Tranche B). These average
    # over ALL VALID TOKENS IN THE BATCH, so the naive per-sample repair -- each
    # sequence's OWN mean -- is a DIFFERENT NUMBER at unequal valid-token counts.
    # Measured at counts [20, 3, 1, 1]: the per-sequence mean gives 5.5544796
    # against this loss's 3.5734539, 55.4% off, which would silently re-weight
    # short sequences UP. They instead return
    # `row_token_sum_i / total_valid_tokens_in_batch * batch`, whose mean under
    # `sum_over_batch_size` is the original batch-wide token mean (measured
    # 3.57345390 -> 3.57345438, 4.8e-07, one float32 summation order apart).
    # NOTE the fixture below is EQUAL-LENGTH and therefore blind to that choice;
    # the ragged proof lives in the commit message, not here.
    ("masked_causal_lm_loss", "MaskedCausalLMLoss", {}, _token_pair),
    ("masked_causal_lm_loss", "PrefixMaskedCausalLMLoss", {}, _token_pair),
    # fixed by plan-2026-08-31T045723-c0d5ffa9 step 6, same T-2 token-pool form.
    # Its ragged divergence was MEASURED under focal weighting rather than
    # inherited from the two above, and it is WORSE than theirs: at counts
    # [20, 3, 1, 1] the per-sequence-mean candidate reads 5.4263663 against this
    # loss's 3.3156378 (63.7% off) at the default gamma=2.0, and 67.0% off at
    # gamma=3.0/alpha=0.25 -- the focal modulator re-weights within a sequence
    # too. Value unchanged at 2.4e-07, and exactly 0.0 at both of those configs.
    ("focal_causal_lm_loss", "FocalCausalLMLoss", {}, _token_pair),
    # fixed by plan-2026-08-31T045723-c0d5ffa9 step 7. Only the NUMERATOR is
    # decomposed: `mae_naive` stays a BATCH-GLOBAL scalar, because the batch-wise
    # scaling factor is this implementation's documented approximation of
    # canonical MASE and a per-row denominator is a different metric -- measured
    # 0.3003199 against 0.1984148 (51.4% off) at seasonal_periods=1 and 0.1537864
    # against 0.1281814 (20.0% off) at seasonal_periods=3, on a 4x12 batch whose
    # rows span scales 0.01/1/30/500. Value unchanged at 1.5e-08.
    ("mase_loss", "MASELoss", {}, _positive_pair),
    # fixed by plan-2026-08-31T045723-c0d5ffa9 step 9, the last member. Three
    # terms, each decomposed so its OWN mean is the scalar it used to
    # contribute -- proven per TERM, because three terms whose errors cancel is a
    # passing total over a broken decomposition. LM via the token-pool form; the
    # two BinaryCrossentropy Q-terms via a `(batch, 1)` reshape, which is
    # load-bearing: BCE means over the LAST axis, so `(batch,)` inputs collapse
    # the BATCH axis to a scalar even under reduction="none" (measured: shape ()
    # for (4,) inputs, (4,) for (4, 1)). Invariant (a) is against the
    # POST-2f3fafa09 behaviour, since that commit corrected the LM term's own
    # double reduction and there is no earlier value worth preserving.
    ("hrm_loss", "HRMLoss", {}, _hrm_pair),
    # fixed by plan-2026-09-02 (the brier/spiegelhalter review). All three were
    # named as unswept candidates in this file's own docstring and MEASURED to
    # be members before the fix. The Z statistic is batch-global, but it
    # decomposes exactly the same way MASELoss and HRMLoss above do: with
    # `c_i = (o_i - p_i)(1 - 2p_i)`, `num = sum(c_i)` and `den = sum(v_i w_i^2)`,
    # `mean_i(N * c_i * num/den) == num^2/den == Z^2` is an algebraic identity,
    # so value AND gradient are unchanged while a zero-weighted row drops out.
    # `_calibration_pair` is required: at the generic fixtures the default
    # chance-corrected penalty is exactly 0.0 and the predicate would report a
    # false membership off `0 == 0 * 0.75`.
    ("brier_spiegelhalters_ztest_loss", "BrierScoreLoss", {}, _calibration_pair),
    ("brier_spiegelhalters_ztest_loss", "SpiegelhalterZLoss", {}, _calibration_pair),
    (
        "brier_spiegelhalters_ztest_loss",
        "CombinedCalibrationLoss",
        {},
        _calibration_pair,
    ),
]


def _ids(cases):
    seen = {}
    out = []
    for m, c, _, _ in cases:
        key = f"{m}.{c}"
        seen[key] = seen.get(key, 0) + 1
        out.append(key if seen[key] == 1 else f"{key}[{seen[key]}]")
    return out


def _make(module, cls, kwargs):
    """A FRESH instance per measurement: several of these mutate state on call."""
    import importlib

    return getattr(importlib.import_module(f"dl_techniques.losses.{module}"), cls)(
        **kwargs
    )


def _charges_every_row_the_batch_aggregate(module, cls, kwargs, builder):
    """Return (is_member, unweighted, weighted). Fresh instance per call."""
    y_true, y_pred = builder()
    w = keras.ops.convert_to_tensor(SAMPLE_WEIGHT)

    unweighted = float(
        keras.ops.convert_to_numpy(_make(module, cls, kwargs)(y_true, y_pred))
    )
    weighted = float(
        keras.ops.convert_to_numpy(
            _make(module, cls, kwargs)(y_true, y_pred, sample_weight=w)
        )
    )
    scaled = unweighted * MEAN_W
    is_member = abs(weighted - scaled) <= 1e-6 * max(1.0, abs(scaled))
    return is_member, unweighted, weighted


# ---------------------------------------------------------------------
# the pin
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "module,cls,kwargs,builder", STILL_BROKEN, ids=_ids(STILL_BROKEN)
)
def test_still_carries_the_premature_scalar_defect(module, cls, kwargs, builder):
    """Pins a KNOWN, UNFIXED defect. A failure here is good news, not a bug.

    If this goes red, ``{module}.{cls}`` was fixed -- remove it from
    ``STILL_BROKEN`` and say so in the commit.
    """
    is_member, unweighted, weighted = _charges_every_row_the_batch_aggregate(
        module, cls, kwargs, builder
    )
    assert is_member, (
        f"{module}.{cls} no longer charges every row the batch aggregate "
        f"(unweighted={unweighted!r}, weighted={weighted!r}, "
        f"unweighted*0.75={unweighted * MEAN_W!r}). It was pinned as an UNFIXED "
        f"member of the premature-scalar family by {PLAN}. If you fixed it, "
        f"delete it from STILL_BROKEN in this file."
    )


@pytest.mark.parametrize("module,cls,kwargs,builder", STILL_BROKEN, ids=_ids(STILL_BROKEN))
def test_still_returns_a_scalar_from_call(module, cls, kwargs, builder):
    """The mechanism behind the predicate, asserted directly."""
    y_true, y_pred = builder()
    out = _make(module, cls, kwargs).call(y_true, y_pred)
    assert tuple(keras.ops.shape(out)) == (), (
        f"{module}.{cls}.call() no longer returns a scalar -- update STILL_BROKEN"
    )


@pytest.mark.parametrize("module,cls,kwargs,builder", KNOWN_GOOD, ids=_ids(KNOWN_GOOD))
def test_the_predicate_rejects_a_correctly_shaped_class(module, cls, kwargs, builder):
    """ANTI-VACUITY. The predicate must distinguish fixed from broken."""
    is_member, unweighted, weighted = _charges_every_row_the_batch_aggregate(
        module, cls, kwargs, builder
    )
    assert not is_member, (
        f"{module}.{cls} is supposed to return per-sample values, but the "
        f"defect predicate accepted it (unweighted={unweighted!r}, "
        f"weighted={weighted!r}). Either it regressed, or this predicate is "
        f"vacuous -- both are serious."
    )


def test_the_pinned_population_has_not_shrunk_silently():
    """A census that quietly loses members is the failure mode this file exists for.

    The floor is on the TOTAL measured population, not on ``STILL_BROKEN`` alone.
    A floor on the broken list becomes unsatisfiable the moment a fix lands, which
    would force whoever fixed it to lower the number -- turning the guard into a
    tally of the work instead of a constraint on it. Expressed this way, the only
    thing that can breach it is a member DISAPPEARING from measurement in either
    direction.
    """
    total = len(STILL_BROKEN) + len(KNOWN_GOOD)
    assert total >= 20, (
        f"{PLAN} measured 14 members on 2026-08-31 and 6 known-good controls; "
        f"the total measured population is now {total} "
        f"(STILL_BROKEN={len(STILL_BROKEN)}, KNOWN_GOOD={len(KNOWN_GOOD)}). A "
        f"member is removed from STILL_BROKEN only by being FIXED, and a fixed "
        f"member MOVES to KNOWN_GOOD in the same commit -- it is never deleted."
    )
    assert len(KNOWN_GOOD) >= 6, (
        "The anti-vacuity arm is what proves the defect predicate still "
        "discriminates. It must never be emptied, least of all as STILL_BROKEN "
        "empties."
    )


def test_stable_max_cross_entropy_is_token_shaped_on_purpose():
    """NOT a premature scalar, and after measurement NOT a defect either.

    ``StableMaxCrossEntropy.call()`` returns ``(batch, seq_len)`` and never
    reduces the token axis, so a ``(batch,)`` ``sample_weight`` RAISES. This file
    previously recorded that as "a different defect class". Measuring stock
    ``keras.losses.SparseCategoricalCrossentropy`` on the SAME inputs refutes
    that: it is token-shaped too and raises the same ``Incompatible shapes``
    error. The convention is Keras', and the rank-matching weight shapes work.

    The comparison arm is what makes this a ruling rather than an assertion: if
    someone reduces this class's token axis, the arm goes red for the right
    reason -- it now differs from the stock loss it is interchangeable with in
    ``HRMLoss``.
    """
    from dl_techniques.losses.hrm_loss import StableMaxCrossEntropy

    y_true, y_pred = _token_pair()
    stock = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    assert tuple(keras.ops.shape(StableMaxCrossEntropy().call(y_true, y_pred))) == (
        BATCH,
        5,
    )
    assert tuple(keras.ops.shape(stock.call(y_true, y_pred))) == (BATCH, 5), (
        "stock SparseCategoricalCrossentropy is no longer token-shaped here, so "
        "the comparison this ruling rests on has changed -- re-derive it"
    )

    w = keras.ops.convert_to_tensor(SAMPLE_WEIGHT)
    with pytest.raises(Exception):
        StableMaxCrossEntropy()(y_true, y_pred, sample_weight=w)
    with pytest.raises(Exception):
        stock(y_true, y_pred, sample_weight=w)

    # A rank-matching weight is the documented way to select rows, and it does
    # NOT collapse to `unweighted * mean(sample_weight)`.
    w2 = keras.ops.reshape(w, (BATCH, 1))
    unweighted = float(
        keras.ops.convert_to_numpy(StableMaxCrossEntropy()(y_true, y_pred))
    )
    weighted = float(
        keras.ops.convert_to_numpy(
            StableMaxCrossEntropy()(y_true, y_pred, sample_weight=w2)
        )
    )
    assert abs(weighted - unweighted * MEAN_W) > 1e-6, (
        f"a (batch, 1) sample_weight was applied as a broadcast scalar "
        f"(unweighted={unweighted!r}, weighted={weighted!r})"
    )
