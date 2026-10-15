"""Every VLM head's ``compute_output_shape`` and symbolic call mirror its eager call.

Guard for plan-2026-10-14T062616-81938e81 step 11.2 (the user's heads review,
``research/reviews/20261015_heads_review.md`` item VLM-4). Three claims, each
measured at ``8b4182f9d``:

* ``MultiTaskVLMHead.compute_output_shape`` takes no ``task_name``. A
  symbolic ``head(inputs, task_name='itm')`` returned the dict of EVERY
  task's outputs (``['grd', 'itm']``), while the eager call returned the one
  head's dict. Reproduced.
* ``VQAHead.compute_output_shape`` read the batch axis of ``vision_features``
  only: with ``(None, 5, D)`` vision and ``(4, 7, D)`` question features it
  reported ``(None, C)``, though ``call`` concatenates the two pooled vectors
  and its batch is 4. Reproduced (lost static batch only).
* ``VisualGroundingHead`` reports ``'grounded_features'`` as
  ``(B, hidden_dim)``; the review read this as the wrong width. Refuted: the
  fusion is built with ``dim=hidden_dim`` (``fusion_config`` cannot override
  it), and all six usable fusion strategies return exactly that width. The
  parametrized parity test below pins it for every strategy.

The parity test compares three things per head: the eager call's shapes,
``compute_output_shape`` on the concrete shapes, and the symbolic call on
``keras.Input`` with an unknown batch.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vlm import (
    VLMTaskConfig,
    VLMTaskType,
    VQAHead,
    create_multi_task_vlm_head,
    create_vlm_head,
)

DIM = 16
B, N_VIS, S = 2, 6, 5
_FUSIONS = ("concatenation", "addition", "multiplication", "gated", "bilinear",
            "tensor_fusion")


def _config(task: VLMTaskType, name: str = "h") -> VLMTaskConfig:
    extra = {
        VLMTaskType.IMAGE_CAPTIONING: {"vocab_size": 11},
        VLMTaskType.VISUAL_QUESTION_ANSWERING: {"num_classes": 7},
    }.get(task, {})
    return VLMTaskConfig(name=name, task_type=task, hidden_size=DIM, **extra)


#: (id, task type, head kwargs, text key, whether to add masks).
CASES = [
    ("captioning", VLMTaskType.IMAGE_CAPTIONING, {"num_layers": 1, "num_heads": 4},
     "text_features"),
    *[(f"vqa-{s}", VLMTaskType.VISUAL_QUESTION_ANSWERING, {"pooling_strategy": s},
       "question_features") for s in ("mean", "max", "attention")],
    *[(f"grounding-{f}", VLMTaskType.VISUAL_GROUNDING, {"fusion_strategy": f},
       "text_features") for f in _FUSIONS],
    ("matching", VLMTaskType.IMAGE_TEXT_MATCHING, {}, "text_features"),
]


def _payload(text_key: str, masks: bool) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(4)
    out = {"vision_features": rng.normal(size=(B, N_VIS, DIM)).astype("float32"),
           text_key: rng.normal(size=(B, S, DIM)).astype("float32")}
    if masks:
        out["vision_mask"] = np.ones((B, N_VIS), "float32")
        out[text_key.replace("_features", "_mask")] = np.ones((B, S), "float32")
    return out


def _shapes(out) -> Any:
    if isinstance(out, dict):
        return {k: _shapes(v) for k, v in out.items()}
    return tuple(out.shape)


def _with_batch(shapes, batch) -> Any:
    if isinstance(shapes, dict):
        return {k: _with_batch(v, batch) for k, v in shapes.items()}
    # The matching head's similarity matrix is (batch, batch).
    return tuple(batch if d == B else d for d in shapes)


@pytest.mark.parametrize("masks", [False, True], ids=["no-mask", "masks"])
@pytest.mark.parametrize("label,task,kwargs,text_key", CASES, ids=[c[0] for c in CASES])
def test_eager_shape_and_symbolic_shapes_agree(label, task, kwargs, text_key, masks) -> None:
    payload = _payload(text_key, masks)

    eager = _shapes(create_vlm_head(_config(task), **kwargs)(payload))
    reported = create_vlm_head(_config(task), **kwargs).compute_output_shape(
        {k: v.shape for k, v in payload.items()})
    assert {k: tuple(v) for k, v in reported.items()} == eager, label

    inputs = {k: keras.Input(shape=v.shape[1:], name=k) for k, v in payload.items()}
    symbolic = _shapes(create_vlm_head(_config(task), **kwargs)(inputs))
    assert symbolic == _with_batch(eager, None), label


def test_the_vqa_batch_is_read_from_either_input() -> None:
    head = VQAHead(task_config=_config(VLMTaskType.VISUAL_QUESTION_ANSWERING),
                   pooling_strategy="mean")
    shapes = head.compute_output_shape({"vision_features": (None, 5, DIM),
                                        "question_features": (4, 7, DIM)})
    assert tuple(shapes["answer_logits"]) == (4, 7)


# ---------------------------------------------------------------------
# Multi-task: the symbolic call and compute_output_shape honour task_name
# ---------------------------------------------------------------------

def _multi():
    return create_multi_task_vlm_head([
        _config(VLMTaskType.IMAGE_TEXT_MATCHING, "itm"),
        _config(VLMTaskType.VISUAL_GROUNDING, "grd"),
    ])


@pytest.mark.parametrize("task_name", [None, "itm", "grd"])
def test_the_multi_task_shapes_follow_task_name(task_name) -> None:
    payload = _payload("text_features", masks=False)
    eager = _shapes(_multi()(payload, task_name=task_name))

    reported = _multi().compute_output_shape(
        {k: v.shape for k, v in payload.items()}, task_name=task_name)
    assert _shapes_of_reported(reported) == eager

    inputs = {k: keras.Input(shape=v.shape[1:], name=k) for k, v in payload.items()}
    symbolic = _shapes(_multi()(inputs, task_name=task_name))
    assert symbolic == _with_batch(eager, None)


def _shapes_of_reported(reported) -> Any:
    if isinstance(reported, dict):
        return {k: _shapes_of_reported(v) for k, v in reported.items()}
    return tuple(reported)


def test_an_unknown_task_name_is_refused_symbolically_too() -> None:
    payload = _payload("text_features", masks=False)
    inputs = {k: keras.Input(shape=v.shape[1:], name=k) for k, v in payload.items()}
    with pytest.raises(ValueError, match="bogus"):
        _multi()(inputs, task_name="bogus")
