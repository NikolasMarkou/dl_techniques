"""Shared internals for the three DINO factory functions.

This module holds the pieces that more than one DINO module needs verbatim:

* `reject_input_shape` — `create_dino_v1`, `create_dino_v2` and `create_dino_v3`
  converged on a single parameter scheme (see
  `src/dl_techniques/models/dino/README.md` § "Factory signatures"), and refusing the
  removed `input_shape` spelling is identical in all three.
* `sync_teacher_to_student` — the DINO teacher must START as a copy of the student
  (reference `main_dino.py` does `teacher.load_state_dict(student.state_dict())`
  before step 0). Both `create_dino_teacher_student_pair` and
  `DINOTrainingModel.__init__` need it, and a second hand-written copy loop is
  exactly the duplication that lets the two drift.

It is imported by `src/dl_techniques/models/dino/dino_v1.py`,
`src/dl_techniques/models/dino/dino_v2.py`,
`src/dl_techniques/models/dino/dino_v3.py` and
`src/dl_techniques/models/dino/dino_training.py`.

Do NOT grow this into a shared ViT trunk or a shared `MODEL_VARIANTS` table — that
unification is a deliberate, recorded non-goal (plan decision D-003: the three model
files have no test suite dense enough to prove a behaviour-preserving merge).
"""

from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------

__all__ = ["reject_input_shape", "sync_teacher_to_student"]

# ---------------------------------------------------------------------


def reject_input_shape(kwargs: Dict[str, Any], factory_name: str) -> None:
    """Raise if a caller passed the removed ``input_shape`` factory argument.

    Interface contract:
        Parameters:
            kwargs: The factory's own ``**kwargs`` dict. Inspected, never mutated.
            factory_name: The calling factory's name, quoted back in the message so
                the error names the call site the user actually wrote.
        Returns:
            ``None`` when ``input_shape`` is absent.
        Failure mode:
            ``TypeError`` when ``input_shape`` is present. It is a ``TypeError`` and
            not a ``ValueError`` because, from the caller's point of view, this is an
            unexpected keyword argument — the same class of failure Python raises for
            a name a function does not accept.

    Why this refusal exists rather than a silent pass-through: the ``DINOv1`` and
    ``DINOv2`` CONSTRUCTORS still accept ``input_shape`` as a lower-level escape
    hatch, so an ``input_shape`` left in a factory call would flow through
    ``**kwargs`` and reach the constructor, where it can DISAGREE with
    ``image_size``. That disagreement is the measured silent defect recorded as
    plan decision D-013: construction succeeds with the wrong patch count and the
    model only fails (or worse, does not fail) much later.
    """
    if "input_shape" in kwargs:
        raise TypeError(
            f"{factory_name}() no longer accepts 'input_shape'; use 'image_size' "
            f"instead. The input shape is derived as (*image_size, in_channels). "
            f"Passing both spellings allowed them to disagree, which built a model "
            f"with a patch grid that did not match its input."
        )


# ---------------------------------------------------------------------


def _path_suffix(weight: Any) -> str:
    """Return a weight's path with its owning model's root name removed.

    Interface contract:
        Parameters:
            weight: A `keras.Variable` (anything exposing `.path`).
        Returns:
            The path after the first `/`, or the whole path when there is no
            `/`. Read only, no side effects.
        Failure mode:
            `AttributeError` if `weight` has no `.path` -- not defended
            against, because the only callers iterate `model.weights`.

    Why the root is stripped rather than compared: two models built by the
    same factory in the same process get DIFFERENT root names (`sequential`
    and `sequential_1`), so a full-path comparison would reject every
    legitimate pair. MEASURED on the in-tree factory pair
    (`create_dino_teacher_student_pair("tiny", ...)`): 157/157 weights, 0
    suffix mismatches -- the paths are in fact identical there, and the strip
    only matters for a hand-built pair.
    """
    return weight.path.split("/", 1)[-1]


# ---------------------------------------------------------------------


# DECISION plan-2026-08-01T105809-dc0c402e/D-034
# Copy the STUDENT into the TEACHER, never the other way round, and do it
# UNCONDITIONALLY at pair construction. Do NOT "improve" this into an opt-in
# flag (`sync_teacher=False`), and do NOT delete it because the two networks
# "are the same architecture anyway".
#   * DINO's teacher is defined as an exponential moving average OF THE
#     STUDENT'S OWN TRAJECTORY, starting from the student's own
#     initialization. Reference `main_dino.py` runs
#     `teacher_without_ddp.load_state_dict(student.module.state_dict())`
#     BEFORE step 0.
#   * MEASURED without this copy, at `create_dino_training_model("tiny",
#     image_size=32, patch_size=16, n_local_views=2, dino_out_dim=64)`:
#     55 of 157 weight tensors differ, and the two networks' outputs on the
#     same input differ by max|d| = 0.3002. The other 102 tensors agree only
#     because they are zero-initialized biases and unit-initialized norm
#     scales -- every tensor carrying information differed.
#   * The corruption does not wash out quickly. At `ema_decay_start=0.996`,
#     `0.996**295 = 30.66%` of the UNRELATED initial teacher is still present
#     after the smoke run's first epoch, so the first 1-3 epochs distil the
#     student against a network that has nothing to do with it.
#   * A flag would invite the defect straight back, and there is no caller for
#     whom an unrelated random teacher is correct.
def sync_teacher_to_student(teacher: Any, student: Any) -> None:
    """Assign every student weight into its teacher counterpart, in place.

    Interface contract:
        Parameters:
            teacher: The EMA target network (a `keras.Model`). MUTATED. Typed
                `Any` rather than `keras.Model` so this module stays free of a
                Keras import — it is imported by all four DINO modules and by
                the package-surface test, which enumerates its public names.
            student: The trainable network (a `keras.Model`). Read only.
        Returns:
            ``None``. On return every teacher weight equals its student
            counterpart exactly.
        Failure mode:
            ``ValueError`` when the copy could not be performed, would be
            performed against the wrong counterpart, or did not take effect —
            an empty weight list (an unbuilt model, where the copy would
            silently no-op), a weight-count mismatch (the ``zip`` would
            silently copy a prefix), a per-weight path-suffix or shape
            disagreement (the ``zip`` would copy into the wrong slot), or a
            surviving difference after the assignment.

    **The guards are not belt-and-braces.** The failure this function exists to
    fix is a silent one, and the two ways a "fix" for it reproduces the defect
    are (1) running against a model whose weights do not exist yet — `zip()`
    over two empty lists completes happily and copies nothing — and (2) running
    against a correctly-sized but MIS-ORDERED pair, where every tensor is
    copied into a same-shaped neighbour's slot.

    The per-weight check is in the COPY loop, deliberately, not in the trailing
    equality sweep. That sweep re-uses the same positional `zip` the copy used,
    so it can only ever confirm the pairing it was given: a mis-paired copy
    makes the teacher equal to the student *under that pairing* and the sweep
    reports success. It is retained as a did-the-assignment-land check, which
    is all it can honestly be. REPRODUCED before this guard existed, on two
    `keras.Sequential([Dense(4), Dense(4)])` models whose layers were named in
    opposite order: the teacher's `b` received the student's `a` values and the
    function returned without raising.

    This is a CONSTRUCTION-time operation only, and it is unconditional —
    `DINOTrainingModel.__init__` calls it on EVERY construction, so any
    teacher values held by the object passed in are discarded. Safe and unsafe
    uses, measured:

    * **SAFE — `keras.models.load_model` on a saved `DINOTrainingModel`.** The
      restore happens AFTER `from_config` returns and is authoritative; a
      deliberately-perturbed teacher survived a round-trip bit-identically
      (drift 0.0, teacher-student gap preserved). Pinned by
      `test_reload_keeps_a_trained_teacher`.
    * **UNSAFE — resuming by rebuilding the two backbones separately** (two
      `create_dino_v1(...)` calls, `load_weights` into each, then wrapping them
      in `DINOTrainingModel`). The sync overwrites the restored teacher with
      the restored student and the EMA history is gone. Nothing can detect
      this: a trained teacher is structurally indistinguishable from a fresh
      one. Resume from the `.keras` file instead.
    """
    if not teacher.weights or not student.weights:
        raise ValueError(
            f"sync_teacher_to_student needs two BUILT models: the teacher has "
            f"{len(teacher.weights)} weight(s) and the student has "
            f"{len(student.weights)}. Copying between unbuilt models is a "
            f"silent no-op -- the teacher would keep its own unrelated random "
            f"initialization and the EMA would never be an EMA of the student."
        )
    if len(teacher.weights) != len(student.weights):
        raise ValueError(
            f"sync_teacher_to_student requires structurally identical models, "
            f"but the teacher has {len(teacher.weights)} weights and the "
            f"student has {len(student.weights)}; zip() would silently copy "
            f"only the shorter prefix."
        )

    for index, (teacher_weight, student_weight) in enumerate(
            zip(teacher.weights, student.weights)):
        if (_path_suffix(teacher_weight) != _path_suffix(student_weight)
                or teacher_weight.shape != student_weight.shape):
            raise ValueError(
                f"sync_teacher_to_student refuses to copy weight {index}: the "
                f"teacher's '{teacher_weight.path}' {teacher_weight.shape} is "
                f"not the counterpart of the student's "
                f"'{student_weight.path}' {student_weight.shape}. The two "
                f"weight lists are paired POSITIONALLY, so a pair whose "
                f"layers were created in a different order copies each "
                f"tensor into the wrong slot -- and because the two networks "
                f"are full of same-shaped tensors (every LayerNorm gamma and "
                f"beta at a given width is interchangeable by shape alone), "
                f"the result is a teacher assembled from the right values in "
                f"the wrong places. Build both networks with the same factory."
            )
        teacher_weight.assign(
            np.asarray(student_weight).astype(
                np.asarray(teacher_weight).dtype))

    worst = 0.0
    for teacher_weight, student_weight in zip(teacher.weights, student.weights):
        worst = max(worst, float(np.max(np.abs(
            np.asarray(teacher_weight) - np.asarray(student_weight)))))
    if worst != 0.0:
        raise ValueError(
            f"sync_teacher_to_student ran but the teacher still differs from "
            f"the student by max|delta| = {worst:.6e}. The copy did not take "
            f"effect, which is the very defect it exists to remove."
        )


# ---------------------------------------------------------------------
