"""Every NLP head's ``compute_output_shape`` and symbolic call mirror its eager call.

Guard for plan-2026-10-14T062616-81938e81 step 11.3 (the user's heads review,
``research/reviews/20261015_heads_review.md``, test gap "no
``compute_output_shape``-vs-``call`` contract"). Measured at ``846589b7f``:

* ``TextSimilarityHead`` read any two-element shape as a pair. A pooled
  single input of shape ``(2, 8)`` is a two-element tuple, so
  ``compute_output_shape((2, 8))`` reported ``'similarity_score'`` and
  ``build`` took ``2`` as the input shape. Reproduced.
* ``MultiTaskNLPHead.compute_output_shape`` took no ``task_name``: a
  symbolic ``head(x, task_name='clf')`` returned every task's outputs while
  the eager call returned one head's (the VLM head had the same defect,
  D-060). Its ``call`` also took ``task_name`` in the second positional slot,
  ahead of ``training``. Reproduced; ``task_name`` is keyword-only now.

The parity test compares three things per head: the eager call's shapes,
``compute_output_shape`` on the concrete shapes, and the symbolic call on
``keras.Input`` with an unknown batch, with and without a mask.
"""

import inspect
from typing import Any

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    NLPTaskType,
    create_multi_task_nlp_head,
    create_nlp_head,
)

D = 8
B, S, C = 2, 5, 3
_RNG = np.random.default_rng(5)
SEQ = _RNG.normal(size=(B, S, D)).astype("float32")
CHOICES = _RNG.normal(size=(B, C, S, D)).astype("float32")


def _cfg(task, **kw):
    return NLPTaskConfig(name=task.value, task_type=task, **kw)


#: id -> (config, head kwargs, hidden_states).
CASES = {
    "classification": (_cfg(NLPTaskType.TEXT_CLASSIFICATION, num_classes=3), {}, SEQ),
    "classification-pooled": (
        _cfg(NLPTaskType.TEXT_CLASSIFICATION, num_classes=3), {}, SEQ.mean(axis=1)),
    "token": (_cfg(NLPTaskType.TOKEN_CLASSIFICATION, num_classes=4), {}, SEQ),
    "token-crf": (_cfg(NLPTaskType.TOKEN_CLASSIFICATION, num_classes=4, use_crf=True), {}, SEQ),
    "qa": (_cfg(NLPTaskType.QUESTION_ANSWERING), {"use_task_attention": True}, SEQ),
    "similarity": (_cfg(NLPTaskType.TEXT_SIMILARITY, hidden_size=12), {}, SEQ),
    "similarity-pooled": (_cfg(NLPTaskType.TEXT_SIMILARITY), {}, SEQ.mean(axis=1)),
    "generation": (_cfg(NLPTaskType.TEXT_GENERATION, vocabulary_size=11), {"use_ffn": True}, SEQ),
    "multiple-choice": (_cfg(NLPTaskType.MULTIPLE_CHOICE, hidden_size=16), {}, CHOICES),
    "multiple-choice-pooled": (_cfg(NLPTaskType.MULTIPLE_CHOICE), {}, CHOICES.mean(axis=2)),
}

#: The cases a mask applies to (a sequence input that the head pools or scores).
MASKABLE = {"classification", "token", "token-crf", "qa", "similarity", "generation",
            "multiple-choice"}


def _payload(case, masked):
    _, _, x = CASES[case]
    if not masked:
        return x
    return {"hidden_states": x, "attention_mask": np.ones(x.shape[:-1], "float32")}


def _shapes(out) -> Any:
    if isinstance(out, dict):
        return {k: _shapes(v) for k, v in out.items()}
    return tuple(out.shape)


def _shape_of(x):
    return {k: v.shape for k, v in x.items()} if isinstance(x, dict) else x.shape


def _symbolic(x):
    if isinstance(x, dict):
        return {k: keras.Input(shape=v.shape[1:], name=k) for k, v in x.items()}
    return keras.Input(shape=x.shape[1:])


def _unbatched(shapes):
    if isinstance(shapes, dict):
        return {k: _unbatched(v) for k, v in shapes.items()}
    return (None,) + tuple(shapes[1:])


_PARAMS = [(c, m) for c in sorted(CASES) for m in (False, True) if not m or c in MASKABLE]


@pytest.mark.parametrize("case,masked", _PARAMS,
                         ids=[f"{c}-{'mask' if m else 'nomask'}" for c, m in _PARAMS])
def test_eager_reported_and_symbolic_shapes_agree(case, masked) -> None:
    config, kwargs, _ = CASES[case]
    x = _payload(case, masked)

    eager = _shapes(create_nlp_head(config, D, **kwargs)(x))
    reported = create_nlp_head(config, D, **kwargs).compute_output_shape(_shape_of(x))
    assert _shapes_of_reported(reported) == eager

    symbolic = _shapes(create_nlp_head(config, D, **kwargs)(_symbolic(x)))
    assert symbolic == _unbatched(eager)

    inputs = _symbolic(x)
    model = keras.Model(inputs, create_nlp_head(config, D, **kwargs)(inputs))
    predicted = model.predict(x, verbose=0)
    assert _shapes(_as_arrays(predicted)) == eager


def _as_arrays(out):
    if isinstance(out, dict):
        return {k: _as_arrays(v) for k, v in out.items()}
    return np.asarray(out)


def _shapes_of_reported(reported) -> Any:
    if isinstance(reported, dict):
        return {k: _shapes_of_reported(v) for k, v in reported.items()}
    return tuple(reported)


def test_a_pooled_single_input_is_not_read_as_a_pair() -> None:
    head = create_nlp_head(_cfg(NLPTaskType.TEXT_SIMILARITY), D)
    assert set(head.compute_output_shape((B, D))) == {"embeddings"}


@pytest.mark.parametrize("container", [tuple, list])
def test_the_pair_shapes_agree(container) -> None:
    config = _cfg(NLPTaskType.TEXT_SIMILARITY, hidden_size=12)
    pair = container([SEQ, SEQ[::-1].copy()])
    eager = _shapes(create_nlp_head(config, D)(pair))
    reported = create_nlp_head(config, D).compute_output_shape(container([SEQ.shape, SEQ.shape]))
    assert _shapes_of_reported(reported) == eager == {
        "similarity_score": (B,), "embeddings_1": (B, 12), "embeddings_2": (B, 12)}


# ---------------------------------------------------------------------
# Multi-task: the symbolic call and compute_output_shape honour task_name
# ---------------------------------------------------------------------

def _multi(**kwargs):
    return create_multi_task_nlp_head([
        _cfg(NLPTaskType.TEXT_CLASSIFICATION, num_classes=3, hidden_size=16),
        _cfg(NLPTaskType.TOKEN_CLASSIFICATION, num_classes=4),
    ], D, **kwargs)


@pytest.mark.parametrize("projections", [False, True])
@pytest.mark.parametrize("task_name", [None, "text_classification", "token_classification"])
def test_the_multi_task_shapes_follow_task_name(task_name, projections) -> None:
    eager = _shapes(_multi(use_task_specific_projections=projections)(
        SEQ, task_name=task_name))
    reported = _multi(use_task_specific_projections=projections).compute_output_shape(
        SEQ.shape, task_name=task_name)
    assert _shapes_of_reported(reported) == eager
    symbolic = _shapes(_multi(use_task_specific_projections=projections)(
        keras.Input(shape=(S, D)), task_name=task_name))
    assert symbolic == _unbatched(eager)


def test_an_unknown_task_name_is_refused_symbolically_too() -> None:
    with pytest.raises(ValueError, match="bogus"):
        _multi()(keras.Input(shape=(S, D)), task_name="bogus")


def test_the_second_positional_slot_is_the_training_flag() -> None:
    """A positional flag used to land in ``task_name`` ("Unknown task: False")."""
    head = _multi()
    head(SEQ)
    assert sorted(head.call(SEQ, False)) == ["text_classification", "token_classification"]
    parameter = inspect.signature(head.call).parameters["task_name"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
