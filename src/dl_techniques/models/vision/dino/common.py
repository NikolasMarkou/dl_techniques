"""Shared internals used by all three DINO factory functions.

This module holds two functions: `reject_input_shape`, which rejects the
removed `input_shape` factory argument identically for `create_dino_v1`,
`create_dino_v2` and `create_dino_v3`; and `sync_teacher_to_student`, which
copies every student weight into its teacher counterpart, used by both
`create_dino_teacher_student_pair` and `DINOTrainingModel.__init__` so the
DINO teacher always starts as an exact copy of the student, matching the
reference `main_dino.py`'s `teacher.load_state_dict(student.state_dict())`
before step 0.

This module holds no shared ViT trunk and no shared `MODEL_VARIANTS` table:
the three DINO model files have no test suite dense enough to prove a
behaviour-preserving merge (plan decision D-003).
"""

from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------

__all__ = ["reject_input_shape", "sync_teacher_to_student"]

# ---------------------------------------------------------------------


def reject_input_shape(kwargs: Dict[str, Any], factory_name: str) -> None:
    """Raise if a caller passed the removed ``input_shape`` factory argument.

    The `DINOv1` and `DINOv2` constructors still accept `input_shape` as a
    lower-level escape hatch, so a leftover `input_shape` in a factory call
    would flow through `**kwargs` into the constructor and can disagree with
    `image_size` there. Plan decision D-013 records the resulting silent
    defect: construction succeeds with the wrong patch count and the model
    only fails, or does not fail, much later.

    :param kwargs: The factory's own ``**kwargs`` dict. Inspected, never mutated.
    :type kwargs: Dict[str, Any]
    :param factory_name: The calling factory's name, quoted back in the
        error message so it names the call site the user actually wrote.
    :type factory_name: str
    :raises TypeError: If ``input_shape`` is present, since from the
        caller's point of view this is an unexpected keyword argument.
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

    Two models built by the same factory in the same process get different
    root names (``sequential`` and ``sequential_1``), so a full-path
    comparison would reject every legitimate pair; stripping the root lets
    the comparison work.

    :param weight: A ``keras.Variable`` (anything exposing ``.path``).
    :type weight: Any
    :return: The path after the first ``/``, or the whole path when there is
        no ``/``.
    :rtype: str
    """
    return weight.path.split("/", 1)[-1]


# ---------------------------------------------------------------------


# DECISION plan-2026-08-01T105809-dc0c402e/D-034: copy student into teacher unconditionally, on every construction, never an opt-in flag.
# Without it 55/157 weight tensors differed and outputs diverged by max|d|=0.3002; a flag would let a caller reintroduce that. See decisions.md.
def sync_teacher_to_student(teacher: Any, student: Any) -> None:
    """Assign every student weight into its teacher counterpart, in place.

    Called unconditionally by every `DINOTrainingModel` construction, so any
    teacher values already held by the object passed in are discarded. Safe
    after `keras.models.load_model` on a saved `DINOTrainingModel`, since the
    weight restore runs after `from_config` and overwrites this copy. Unsafe
    when resuming by rebuilding the two backbones separately and wrapping
    them afterward: this call would overwrite the restored teacher with the
    restored student and discard its EMA history, and a trained teacher is
    structurally indistinguishable from a fresh one, so nothing would detect
    it. Resume from the ``.keras`` file instead.

    The per-weight path-suffix and shape check happens during the copy loop,
    not only in the trailing equality sweep: that sweep reuses the same
    positional pairing the copy used, so it can only confirm the pairing it
    was given, and a mis-paired copy that lands every tensor in a
    same-shaped neighbour's slot would otherwise pass it.

    :param teacher: The EMA target network (a ``keras.Model``), mutated in
        place. Typed ``Any`` rather than ``keras.Model`` so this module stays
        free of a Keras import.
    :type teacher: Any
    :param student: The trainable network (a ``keras.Model``), read only.
    :type student: Any
    :raises ValueError: If either model is unbuilt, if the two have a
        different weight count, if a weight pair disagrees in path suffix or
        shape, or if a difference survives the assignment.
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
