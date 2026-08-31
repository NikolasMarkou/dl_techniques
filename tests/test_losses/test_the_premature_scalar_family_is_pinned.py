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
    ``call()`` returns ``(batch, seq_len)``. But a ``(batch,)`` ``sample_weight``
    then RAISES ``InvalidArgumentError: Incompatible shapes: [4,5] vs. [4]``.
    That is a different defect class (an unreduced token axis), pinned in its own
    test below so it cannot change unnoticed.
*   ``yolo12_multitask_loss`` (4 classes) and ``nano_vlm_loss.NanoVLMLoss`` --
    require structured multi-head ``y_pred`` dicts that cannot be built cheaply
    here.
*   The remaining ~60 ``keras.losses.Loss`` subclasses in
    ``src/dl_techniques/losses/`` were NOT swept. A static scan of their
    ``call()`` return expressions shows further candidates (for example
    ``multi_labels_loss``, ``utilization_loss``, ``brier_spiegelhalters_ztest_loss``).
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
STILL_BROKEN = [
    ("mase_loss", "MASELoss", {}, _positive_pair),
    ("masked_causal_lm_loss", "MaskedCausalLMLoss", {}, _token_pair),
    ("masked_causal_lm_loss", "PrefixMaskedCausalLMLoss", {}, _token_pair),
    ("focal_causal_lm_loss", "FocalCausalLMLoss", {}, _token_pair),
    ("hrm_loss", "HRMLoss", {}, _hrm_pair),
    # --- found by execution, absent from the findings' list ---
    ("dino_loss", "DINOLoss", {"out_dim": 5}, _embedding_pair),
    ("dino_loss", "KoLeoLoss", {}, _embedding_pair),
    ("scaled_mse_loss", "ScaledMseLoss", {}, _image_pair),
    ("sam_mask_loss", "SAMIoULoss", {}, _iou_pair),
]

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


def test_stable_max_cross_entropy_is_a_different_defect_not_this_one():
    """NOT a premature scalar: it returns ``(batch, seq_len)`` and its token axis
    is never reduced, so a ``(batch,)`` ``sample_weight`` RAISES instead of being
    silently misapplied. Pinned so the distinction cannot rot.
    """
    from dl_techniques.losses.hrm_loss import StableMaxCrossEntropy

    y_true, y_pred = _token_pair()
    out = StableMaxCrossEntropy().call(y_true, y_pred)
    assert tuple(keras.ops.shape(out)) == (BATCH, 5)

    with pytest.raises(Exception):
        StableMaxCrossEntropy()(
            y_true, y_pred, sample_weight=keras.ops.convert_to_tensor(SAMPLE_WEIGHT)
        )
