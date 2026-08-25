"""Guards for the two ColBERT training objectives.

The two losses in :mod:`dl_techniques.losses.colbert_loss` are structurally similar
enough that a copy-paste of one into the other type-checks, runs, and produces a
plausible number. Every test below is therefore written against a numpy oracle derived
from the **reference formula** quoted in
``plans/.../findings/colbert-architecture-reference.md`` §1 and §2 (itself ``curl``ed
from ``colbert/training/training.py``), never transcribed from the implementation, and
the suite carries an explicit v1-vs-v2 disagreement guard.

RED-proof record (2026-08-25, plan-2026-08-25T121346-c71fc3ad iter-1/step-4). Each
injection was applied to ``src/dl_techniques/losses/colbert_loss.py`` from a ``cp``
backup, the suite run, and the file restored and verified with ``diff -q``:

(a) Dropped the ``* alpha`` multiply on the teacher scores. 4 failed / 21 passed.
    RED by name: ``test_the_distillation_alpha_knob_is_live[0.25]`` and ``[4.0]`` at
    ``AssertionError: distillation_alpha is a DEAD KNOB: alpha=0.25 produced
    0.7639342546463013, identical to the alpha=1.0 baseline 0.7639342546463013``.
    Collaterally RED: ``test_the_v2_loss_reproduces_the_log_target_kl_kernel[2.5]`` and
    ``[0.3]`` (the ``[1.0]`` arm is invariant to this injection, as it must be).

(b) Replaced the hand-written ``log_target=True`` kernel with
    ``keras.losses.kl_divergence(target_log_probs, student_log_probs)`` -- Keras' own
    probabilities-in KL semantics. 9 failed / 16 passed.
    RED by name: ``test_the_v2_loss_reproduces_the_log_target_kl_kernel`` at
    ``AssertionError: v2 does not reproduce sum(exp(t)*(t-s))/batch ... got 0.0,
    reference oracle 0.7639342966070863``. Note the injected value is EXACTLY 0.0: Keras
    clips both operands into ``[epsilon, 1]``, and every log-probability here is negative,
    so all of them clip to the same epsilon and the divergence vanishes. That is the
    "plausible, silent, wrong" failure the D-015 anchor warns about, measured.

(c) Replaced ``ColBERTDistillationLoss.call``'s body with v1's body (softmax CE at
    index 0 of the student scores, teacher ignored). 9 failed / 16 passed.
    RED by name: ``test_the_two_losses_disagree_on_a_shared_input`` at
    ``AssertionError: v2 collapsed onto v1 ... got v1=1.2381008863449097,
    v2=1.2381008863449097``.
    This injection is also why that test carries no per-loss oracle pre-check: a first
    draft did, and the pre-check -- an UNNAMED sanity line -- fired instead of the guard
    the test exists for. Attribution to the wrong assertion is the plan's own documented
    stop trigger, so the pre-checks were removed rather than the finding absorbed.
"""

from typing import Tuple

import keras
import numpy as np
import pytest

from dl_techniques.losses import (
    ColBERTDistillationLoss,
    ColBERTPairwiseSoftmaxLoss,
)

# ---------------------------------------------------------------------
# Independent numpy oracles, written from the reference formulas.
#
# v1 (colbert/training/training.py):
#     scores.view(-1, nway); labels = zeros; nn.CrossEntropyLoss()(scores, labels)
# v2 (same file):
#     target = log_softmax(teacher * distillation_alpha, dim=-1)
#     log_scores = log_softmax(student, dim=-1)
#     KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target)
# with the documented PyTorch kernel for log_target=True being
#     sum(exp(target) * (target - input))
# and 'batchmean' dividing that total by the batch size.
# ---------------------------------------------------------------------


def _np_log_softmax(x: np.ndarray) -> np.ndarray:
    """log_softmax along the last axis, shift-stabilized."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _oracle_v1(scores: np.ndarray, nway: int) -> float:
    """Softmax cross-entropy against label 0 of each nway group, mean over the batch."""
    grouped = np.asarray(scores, dtype=np.float64).reshape(-1, nway)
    # CrossEntropyLoss(s, 0) == logsumexp_k(s_k) - s_0
    max_k = np.max(grouped, axis=-1)
    logsumexp = max_k + np.log(np.sum(np.exp(grouped - max_k[:, None]), axis=-1))
    per_sample = logsumexp - grouped[:, 0]
    return float(np.mean(per_sample))


def _oracle_v2(
    teacher: np.ndarray,
    student: np.ndarray,
    nway: int,
    distillation_alpha: float,
) -> float:
    """batchmean KL with both operands as log-probabilities."""
    t = np.asarray(teacher, dtype=np.float64).reshape(-1, nway) * distillation_alpha
    s = np.asarray(student, dtype=np.float64).reshape(-1, nway)
    target = _np_log_softmax(t)
    inp = _np_log_softmax(s)
    total = np.sum(np.exp(target) * (target - inp))
    batch = t.shape[0]
    return float(total / batch)


# ---------------------------------------------------------------------
# Shared fixture data. batch = 2, nway = 3.
# ---------------------------------------------------------------------

_STUDENT = np.array([[3.0, 1.0, 0.0], [0.5, 2.5, 1.0]], dtype=np.float32)
_TEACHER = np.array([[2.0, 0.0, 1.0], [1.0, 0.0, 3.0]], dtype=np.float32)
_NWAY = 3


def _scalar(value) -> float:
    return float(keras.ops.convert_to_numpy(value))


# ---------------------------------------------------------------------
# 1. v1 reproduces its own reference formula
# ---------------------------------------------------------------------


@pytest.mark.parametrize("nway", [2, 3, 4])
def test_the_v1_loss_reproduces_the_reference_softmax_cross_entropy(nway: int) -> None:
    rng = np.random.default_rng(20260825 + nway)
    batch = 5
    scores = rng.normal(0.0, 2.0, size=(batch * nway,)).astype(np.float32)

    loss_fn = ColBERTPairwiseSoftmaxLoss(nway=nway)
    got = _scalar(loss_fn(keras.ops.zeros_like(scores), scores))
    expected = _oracle_v1(scores, nway)

    assert got == pytest.approx(expected, abs=1e-6, rel=0.0), (
        f"v1 does not reproduce logsumexp_k(s_k) - s_0 averaged over the batch: "
        f"got {got!r}, reference oracle {expected!r}"
    )


def test_the_v1_loss_accepts_an_already_grouped_score_matrix() -> None:
    loss_fn = ColBERTPairwiseSoftmaxLoss(nway=_NWAY)
    flat = _STUDENT.reshape(-1)

    grouped_loss = _scalar(loss_fn(keras.ops.zeros_like(_STUDENT), _STUDENT))
    flat_loss = _scalar(loss_fn(keras.ops.zeros_like(flat), flat))

    assert grouped_loss == pytest.approx(flat_loss, abs=1e-6, rel=0.0), (
        "the (batch, nway) and the flat (batch*nway,) input forms must give the "
        f"identical loss; got {grouped_loss!r} vs {flat_loss!r}"
    )


def test_the_v1_loss_ignores_y_true_by_design() -> None:
    """The positive is POSITIONAL (index 0). y_true is documented as unused."""
    loss_fn = ColBERTPairwiseSoftmaxLoss(nway=_NWAY)
    flat = _STUDENT.reshape(-1)

    with_zeros = _scalar(loss_fn(keras.ops.zeros_like(flat), flat))
    with_garbage = _scalar(loss_fn(keras.ops.ones_like(flat) * 17.0, flat))

    assert with_zeros == pytest.approx(with_garbage, abs=1e-6, rel=0.0), (
        "ColBERTPairwiseSoftmaxLoss is documented to ignore y_true entirely (the "
        "positive is index 0 by position); a y_true-dependent result contradicts the "
        f"docstring: {with_zeros!r} vs {with_garbage!r}"
    )


# ---------------------------------------------------------------------
# 2. v2 reproduces sum(exp(t)*(t-s))/batch
# ---------------------------------------------------------------------


@pytest.mark.parametrize("distillation_alpha", [1.0, 2.5, 0.3])
def test_the_v2_loss_reproduces_the_log_target_kl_kernel(
    distillation_alpha: float,
) -> None:
    loss_fn = ColBERTDistillationLoss(
        nway=_NWAY, distillation_alpha=distillation_alpha
    )
    got = _scalar(loss_fn(_TEACHER, _STUDENT))
    expected = _oracle_v2(_TEACHER, _STUDENT, _NWAY, distillation_alpha)

    assert got == pytest.approx(expected, abs=1e-6, rel=0.0), (
        f"v2 does not reproduce sum(exp(t)*(t-s))/batch with t = log_softmax(teacher * "
        f"{distillation_alpha}) and s = log_softmax(student): got {got!r}, reference "
        f"oracle {expected!r}"
    )


def test_the_v2_batchmean_divides_by_the_batch_not_by_the_element_count() -> None:
    """batchmean != mean over elements. Doubling nway must not halve the loss."""
    nway = 6
    rng = np.random.default_rng(4242)
    teacher = rng.normal(0.0, 1.5, size=(4, nway)).astype(np.float32)
    student = rng.normal(0.0, 1.5, size=(4, nway)).astype(np.float32)

    got = _scalar(ColBERTDistillationLoss(nway=nway)(teacher, student))
    batchmean = _oracle_v2(teacher, student, nway, 1.0)
    elementmean = batchmean / nway

    assert got == pytest.approx(batchmean, abs=1e-6, rel=0.0), (
        f"v2 must implement batchmean (divide by batch={teacher.shape[0]}); got "
        f"{got!r}, batchmean {batchmean!r}"
    )
    assert abs(got - elementmean) > 1e-3, (
        f"v2 collapsed onto an element-mean reduction (divide by batch*nway), an "
        f"{nway}x silent under-scaling: got {got!r}, element-mean {elementmean!r}"
    )


# ---------------------------------------------------------------------
# 3. The two losses disagree -- a copy-paste of one into the other fails here
# ---------------------------------------------------------------------


def test_the_two_losses_disagree_on_a_shared_input() -> None:
    """Neither loss may silently be the other.

    Two arms, both hand-derived from the oracles above on the fixture batch:

    * General input. Hand-derived from the oracles: v1 = 1.23805, v2 (alpha=1) =
      0.76412, so the separation is 0.47393 (float32-measured: 1.238101 / 0.763934 /
      0.474167). The asserted floor is 0.25 -- comfortably below the derived value, and
      far above any float32 noise.
    * Teacher == student. Then log_softmax(t) == log_softmax(s) and the KL is EXACTLY
      zero, while v1's cross-entropy on the same scores is 1.23805. This arm is the
      sharp one: any implementation of v2 that ignores the teacher (i.e. is really v1)
      returns ~1.24 where 0 is required.

    This test deliberately carries NO "each loss agrees with its own oracle" pre-check.
    Such a pre-check fires FIRST under the v2-collapsed-onto-v1 injection, so the named
    assertions below would never be reached and the RED proof would attribute to an
    unnamed sanity line. Per-loss oracle agreement is the job of
    ``test_the_v1_loss_reproduces_the_reference_softmax_cross_entropy`` and
    ``test_the_v2_loss_reproduces_the_log_target_kl_kernel``.
    """
    flat_student = _STUDENT.reshape(-1)
    v1 = _scalar(ColBERTPairwiseSoftmaxLoss(nway=_NWAY)(
        keras.ops.zeros_like(flat_student), flat_student
    ))
    v2 = _scalar(ColBERTDistillationLoss(nway=_NWAY)(_TEACHER, _STUDENT))

    assert abs(v1 - v2) >= 0.25, (
        f"v2 collapsed onto v1: the two objectives must not agree on a shared input "
        f"(hand-derived separation on this batch is 0.47393, floor 0.25); got "
        f"v1={v1!r}, v2={v2!r}"
    )

    # Sharp arm: identical teacher and student => KL is exactly zero.
    v2_self = _scalar(ColBERTDistillationLoss(nway=_NWAY)(_STUDENT, _STUDENT))
    assert v2_self == pytest.approx(0.0, abs=1e-6, rel=0.0), (
        "v2 collapsed onto v1: with teacher == student the log_target KL is exactly "
        f"zero, but a v1-shaped body returns the cross-entropy {v1!r}; got {v2_self!r}"
    )


# ---------------------------------------------------------------------
# 4. distillation_alpha is a live knob
# ---------------------------------------------------------------------


@pytest.mark.parametrize("distillation_alpha", [0.25, 4.0])
def test_the_distillation_alpha_knob_is_live(distillation_alpha: float) -> None:
    baseline = _scalar(
        ColBERTDistillationLoss(nway=_NWAY, distillation_alpha=1.0)(_TEACHER, _STUDENT)
    )
    scaled = _scalar(
        ColBERTDistillationLoss(nway=_NWAY, distillation_alpha=distillation_alpha)(
            _TEACHER, _STUDENT
        )
    )

    assert abs(scaled - baseline) > 1e-3, (
        f"distillation_alpha is a DEAD KNOB: alpha={distillation_alpha} produced "
        f"{scaled!r}, identical to the alpha=1.0 baseline {baseline!r}. The reference "
        f"scales the teacher scores BEFORE the log_softmax."
    )
    # And it moves in the direction the oracle says it should.
    assert scaled == pytest.approx(
        _oracle_v2(_TEACHER, _STUDENT, _NWAY, distillation_alpha), abs=1e-6, rel=0.0
    )


# ---------------------------------------------------------------------
# 5. Equal teacher scores across the nway group
# ---------------------------------------------------------------------


def test_equal_teacher_scores_give_a_finite_loss() -> None:
    """log_softmax of a constant row is well defined (uniform); nothing may be NaN.

    Note the refinement over the plan's phrasing: the loss is near zero only when the
    STUDENT is also uniform, since KL(uniform || student) is a genuine divergence.
    Both directions are asserted -- exactly zero when both rows are constant, finite and
    strictly positive when only the teacher is.
    """
    flat_teacher = np.full((2, _NWAY), 7.0, dtype=np.float32)

    both_uniform = _scalar(
        ColBERTDistillationLoss(nway=_NWAY)(flat_teacher, np.zeros_like(flat_teacher))
    )
    assert np.isfinite(both_uniform), f"loss is not finite: {both_uniform!r}"
    assert both_uniform == pytest.approx(0.0, abs=1e-6, rel=0.0), (
        "with a constant teacher row AND a constant student row both distributions are "
        f"uniform and the KL must be exactly 0; got {both_uniform!r}"
    )

    against_peaked = _scalar(
        ColBERTDistillationLoss(nway=_NWAY)(flat_teacher, _STUDENT)
    )
    assert np.isfinite(against_peaked), (
        f"a constant teacher row must not produce NaN/inf; got {against_peaked!r}"
    )
    assert against_peaked > 0.0
    assert against_peaked == pytest.approx(
        _oracle_v2(flat_teacher, _STUDENT, _NWAY, 1.0), abs=1e-6, rel=0.0
    )


def test_a_huge_constant_teacher_row_does_not_overflow() -> None:
    """The shift-stabilized log_softmax must survive an extreme constant row."""
    teacher = np.full((2, _NWAY), 1.0e4, dtype=np.float32)
    got = _scalar(ColBERTDistillationLoss(nway=_NWAY, distillation_alpha=10.0)(
        teacher, _STUDENT
    ))
    assert np.isfinite(got), f"log_softmax overflowed on a 1e4 teacher row: {got!r}"


# ---------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("bad_nway", [1, 0, -3])
def test_nway_below_two_raises(bad_nway: int) -> None:
    with pytest.raises(ValueError, match="nway must be >= 2"):
        ColBERTPairwiseSoftmaxLoss(nway=bad_nway)
    with pytest.raises(ValueError, match="nway must be >= 2"):
        ColBERTDistillationLoss(nway=bad_nway)


@pytest.mark.parametrize("bad_alpha", [0.0, -1.0])
def test_a_non_positive_distillation_alpha_raises(bad_alpha: float) -> None:
    with pytest.raises(ValueError, match="distillation_alpha must be > 0"):
        ColBERTDistillationLoss(nway=_NWAY, distillation_alpha=bad_alpha)


def test_a_score_vector_not_divisible_by_nway_raises() -> None:
    scores = np.arange(7, dtype=np.float32)  # 7 is not divisible by 3

    with pytest.raises(ValueError, match="not divisible by nway"):
        ColBERTPairwiseSoftmaxLoss(nway=_NWAY)(keras.ops.zeros_like(scores), scores)

    with pytest.raises(ValueError, match="not divisible by nway"):
        ColBERTDistillationLoss(nway=_NWAY)(scores, scores)


def test_a_grouped_matrix_with_the_wrong_trailing_axis_raises() -> None:
    scores = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="must be a flat"):
        ColBERTPairwiseSoftmaxLoss(nway=_NWAY)(keras.ops.zeros_like(scores), scores)


# ---------------------------------------------------------------------
# 7. get_config round trip
# ---------------------------------------------------------------------


def test_the_v1_loss_round_trips_through_the_keras_registry() -> None:
    original = ColBERTPairwiseSoftmaxLoss(nway=7, name="colbert_v1_probe")
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(original)
    )

    assert isinstance(restored, ColBERTPairwiseSoftmaxLoss)
    assert restored.nway == 7
    assert restored.name == "colbert_v1_probe"

    scores = np.random.default_rng(7).normal(0.0, 2.0, size=(2, 7)).astype(np.float32)
    assert _scalar(restored(keras.ops.zeros_like(scores), scores)) == pytest.approx(
        _scalar(original(keras.ops.zeros_like(scores), scores)), abs=1e-6, rel=0.0
    )


def test_the_v2_loss_round_trips_through_the_keras_registry() -> None:
    original = ColBERTDistillationLoss(
        nway=_NWAY, distillation_alpha=3.25, name="colbert_v2_probe"
    )
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(original)
    )

    assert isinstance(restored, ColBERTDistillationLoss)
    assert restored.nway == _NWAY
    assert restored.distillation_alpha == pytest.approx(3.25)
    assert restored.name == "colbert_v2_probe"

    assert _scalar(restored(_TEACHER, _STUDENT)) == pytest.approx(
        _scalar(original(_TEACHER, _STUDENT)), abs=1e-6, rel=0.0
    )


# ---------------------------------------------------------------------
# 8. Public surface
# ---------------------------------------------------------------------


def test_both_losses_are_importable_from_the_losses_package() -> None:
    import dl_techniques.losses as losses_pkg

    for name in ("ColBERTPairwiseSoftmaxLoss", "ColBERTDistillationLoss"):
        assert hasattr(losses_pkg, name), f"{name} is not exported from dl_techniques.losses"
        assert name in losses_pkg.__all__, f"{name} is missing from losses.__all__"


# ---------------------------------------------------------------------
# 9. mixed_float16 arm
# ---------------------------------------------------------------------


def _fp16_pair() -> Tuple[np.ndarray, np.ndarray]:
    return _TEACHER.astype(np.float16), _STUDENT.astype(np.float16)


def test_both_losses_are_finite_under_mixed_float16() -> None:
    previous = keras.mixed_precision.global_policy()
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
        teacher, student = _fp16_pair()

        v1 = _scalar(ColBERTPairwiseSoftmaxLoss(nway=_NWAY)(
            keras.ops.zeros_like(student), student
        ))
        v2 = _scalar(ColBERTDistillationLoss(nway=_NWAY)(teacher, student))

        assert np.isfinite(v1), f"v1 is not finite under mixed_float16: {v1!r}"
        assert np.isfinite(v2), f"v2 is not finite under mixed_float16: {v2!r}"
        assert v1 == pytest.approx(_oracle_v1(_STUDENT, _NWAY), abs=1e-3, rel=0.0)
        assert v2 == pytest.approx(
            _oracle_v2(_TEACHER, _STUDENT, _NWAY, 1.0), abs=1e-3, rel=0.0
        )
    finally:
        keras.mixed_precision.set_global_policy(previous)
