"""``NLPTaskConfig`` holds only fields a head reads, and refuses a config its head cannot build.

Guard for plan-2026-10-14T062616-81938e81 step 11.3 (the user's heads review,
``research/reviews/20261015_heads_review.md`` items NLP-6 and NLP-7, and the
serialization gap "``get_config`` saves 10/14 fields"). Measured at
``846589b7f``:

* NLP-6: ``create_nlp_head({'task_type': 'sentiment_analysis', ...})`` died
  with ``AttributeError: 'str' object has no attribute 'name'``, because the
  dataclass stored the string as it was. ``create_multi_task_nlp_head`` with a
  dict of dicts died with ``'dict' object has no attribute 'task_type'``.
  ``nlp/model_head.py`` parsed the same string. Reproduced on both paths.
* NLP-7: ``__post_init__`` required ``num_classes`` for five of the fourteen
  task types routed to a classification head. The other nine
  (``EMOTION_DETECTION``, ``INTENT_CLASSIFICATION``, ``TOPIC_CLASSIFICATION``,
  ``SPAM_DETECTION``, ``NATURAL_LANGUAGE_INFERENCE``, ``TEXT_REGRESSION``,
  ``READABILITY_SCORING``, ``QUALITY_SCORING``, ``SEQUENCE_LABELING``) built a
  config and then died in the head (``num_classes must be specified``).
  Reproduced. The requirement is now read from the one dispatch table, so a
  task type cannot be routed to a head without the field that head needs.
* Seven fields were read by no head (``num_labels``, an alias of
  ``num_classes``; ``max_length``, ``loss_weight``, ``label_smoothing``,
  ``use_attention_pooling``, ``beam_size``, ``temperature``). They are
  deleted (D-029); passing one raises ``TypeError``. A set count is validated
  for every task type (``0`` built an ``(N, 0)`` head), and ``dropout_rate``
  must lie in ``[0, 1)``.
* ``MACHINE_TRANSLATION`` was given a 32000-word vocabulary though no head
  serves it; a task type with no head is given nothing now.
"""

import dataclasses

import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    NLPTaskType,
    create_multi_task_nlp_head,
    create_nlp_head,
)
from dl_techniques.layers.heads.nlp.factory import get_head_class

D = 8
_SEQ = np.random.default_rng(2).normal(size=(2, 5, D)).astype("float32")
_CHOICES = np.random.default_rng(2).normal(size=(2, 3, 5, D)).astype("float32")

LIVE_FIELDS = {"name", "task_type", "num_classes", "dropout_rate", "hidden_size",
               "use_crf", "vocabulary_size"}
DEAD_FIELDS = ("num_labels", "max_length", "loss_weight", "label_smoothing",
               "use_attention_pooling", "beam_size", "temperature")
_NEEDS_CLASSES = ("TextClassificationHead", "TokenClassificationHead")


def _routes():
    """Task type -> head class name, read through ``get_head_class``."""
    routes = {}
    for task in NLPTaskType:
        try:
            routes[task] = get_head_class(task).__name__
        except ValueError:
            continue
    return routes


NLP_HEAD_DISPATCH = _routes()


def test_the_dataclass_holds_exactly_the_live_fields() -> None:
    assert {f.name for f in dataclasses.fields(NLPTaskConfig)} == LIVE_FIELDS


@pytest.mark.parametrize("field", DEAD_FIELDS)
def test_a_deleted_field_is_refused(field) -> None:
    with pytest.raises(TypeError, match=field):
        NLPTaskConfig(name="t", task_type=NLPTaskType.TEXT_CLASSIFICATION,
                      num_classes=2, **{field: 1})


# ---------------------------------------------------------------------
# NLP-6: a string task type works on every path
# ---------------------------------------------------------------------

def test_the_dataclass_parses_a_string_task_type() -> None:
    cfg = NLPTaskConfig(name="s", task_type=" Sentiment_Analysis ", num_classes=2)
    assert cfg.task_type is NLPTaskType.SENTIMENT_ANALYSIS


def test_create_nlp_head_takes_a_dict_with_a_string_task_type() -> None:
    head = create_nlp_head({"name": "s", "task_type": "sentiment_analysis",
                            "num_classes": 2}, input_dim=D)
    assert head.task_config.task_type is NLPTaskType.SENTIMENT_ANALYSIS
    assert tuple(head(_SEQ)["logits"].shape) == (2, 2)


def test_the_multi_task_factory_takes_dicts_too() -> None:
    head = create_multi_task_nlp_head(
        {"s": {"name": "s", "task_type": "sentiment_analysis", "num_classes": 2},
         "n": {"name": "n", "task_type": "token_classification", "num_classes": 3}},
        input_dim=D)
    assert sorted(head(_SEQ)) == ["n", "s"]


def test_a_task_type_that_is_neither_enum_nor_string_is_refused() -> None:
    with pytest.raises(TypeError, match="task_type"):
        NLPTaskConfig(name="s", task_type=3, num_classes=2)


def test_an_unknown_string_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match="bogus"):
        NLPTaskConfig(name="s", task_type="bogus", num_classes=2)


# ---------------------------------------------------------------------
# NLP-7: every routed task type is validated for the head it is routed to
# ---------------------------------------------------------------------

_CLASSIFIED = sorted((t for t, h in NLP_HEAD_DISPATCH.items() if h in _NEEDS_CLASSES),
                     key=lambda t: t.name)


@pytest.mark.parametrize("task", _CLASSIFIED, ids=lambda t: t.name)
def test_a_classification_task_without_a_class_count_is_refused(task) -> None:
    with pytest.raises(ValueError, match=rf"{task.value}.*num_classes"):
        NLPTaskConfig(name="t", task_type=task)


@pytest.mark.parametrize("task", sorted(NLP_HEAD_DISPATCH, key=lambda t: t.name),
                         ids=lambda t: t.name)
def test_every_routed_task_with_its_required_fields_builds_and_runs(task) -> None:
    extra = {"num_classes": 3} if NLP_HEAD_DISPATCH[task] in _NEEDS_CLASSES else {}
    head = create_nlp_head(NLPTaskConfig(name="t", task_type=task, **extra), input_dim=D)
    out = head(_CHOICES if NLP_HEAD_DISPATCH[task] == "MultipleChoiceHead" else _SEQ)
    assert out


def test_the_routed_set_is_the_review_census() -> None:
    assert len(NLP_HEAD_DISPATCH) == 24
    assert len(_CLASSIFIED) == 14


@pytest.mark.parametrize("task", sorted(NLPTaskType, key=lambda t: t.name), ids=lambda t: t.name)
def test_only_a_generation_head_gets_a_default_vocabulary(task) -> None:
    extra = {"num_classes": 3} if NLP_HEAD_DISPATCH.get(task) in _NEEDS_CLASSES else {}
    cfg = NLPTaskConfig(name="t", task_type=task, **extra)
    if NLP_HEAD_DISPATCH.get(task) == "TextGenerationHead":
        assert cfg.vocabulary_size == 32000
    else:
        assert cfg.vocabulary_size is None


# ---------------------------------------------------------------------
# values
# ---------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("num_classes", 0), ("num_classes", -1), ("num_classes", True), ("num_classes", 2.0),
    ("hidden_size", 0), ("hidden_size", 1.5), ("vocabulary_size", 0),
    ("vocabulary_size", "big"),
])
def test_a_set_count_must_be_a_positive_integer(field, value) -> None:
    with pytest.raises(ValueError, match=rf"NLPTaskConfig.*{field}"):
        NLPTaskConfig(name="t", task_type=NLPTaskType.QUESTION_ANSWERING, **{field: value})


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.5, "0.1", None])
def test_the_dropout_rate_must_lie_in_zero_one(value) -> None:
    with pytest.raises(ValueError, match="NLPTaskConfig.*dropout_rate"):
        NLPTaskConfig(name="t", task_type=NLPTaskType.QUESTION_ANSWERING, dropout_rate=value)


@pytest.mark.parametrize("value", ["", None, 3])
def test_the_name_must_be_a_non_empty_string(value) -> None:
    with pytest.raises(ValueError, match="NLPTaskConfig.*name"):
        NLPTaskConfig(name=value, task_type=NLPTaskType.QUESTION_ANSWERING)


def test_a_numpy_count_is_stored_as_a_plain_int() -> None:
    cfg = NLPTaskConfig(name="t", task_type=NLPTaskType.TEXT_CLASSIFICATION,
                        num_classes=np.int64(3))
    assert type(cfg.num_classes) is int and cfg.num_classes == 3


def test_to_dict_round_trips_to_an_equal_config() -> None:
    cfg = NLPTaskConfig(name="t", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
                        num_classes=5, use_crf=True, hidden_size=16, dropout_rate=0.2)
    as_dict = cfg.to_dict()
    assert as_dict["task_type"] == "named_entity_recognition"
    assert set(as_dict) == LIVE_FIELDS
    assert NLPTaskConfig(**as_dict) == cfg
