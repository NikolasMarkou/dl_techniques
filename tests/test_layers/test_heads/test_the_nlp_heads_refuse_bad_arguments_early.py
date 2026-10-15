"""The NLP heads refuse a bad argument at construction, by head and argument name.

Guard for plan-2026-10-14T062616-81938e81 step 11.3 (the user's heads review,
``research/reviews/20261015_heads_review.md``, NLP validation gaps). Measured
at ``846589b7f``:

* A bad ``pooling_type`` built no pooler and failed at the first ``call``
  (``Unknown pooling type``) or not at all, for a head that never pools.
  Reproduced; refused in ``__init__`` now, for every head.
* ``TextClassificationHead``, ``TextSimilarityHead`` and
  ``MultipleChoiceHead`` with ``use_pooling=False`` took a sequence and
  failed inside ``call`` with ``TypeError: 'NoneType' object is not
  callable``. Reproduced; the sequence is refused by name, and an already
  pooled input still works.
* ``TokenClassificationHead``, ``QuestionAnsweringHead`` and
  ``TextGenerationHead`` overwrote a caller's ``use_pooling=True`` with
  ``False`` without a word. Reproduced; refused now.
* ``MultiTaskNLPHead`` accepted an empty task set (a layer returning ``{}``),
  and ``create_multi_task_nlp_head`` keyed a list by name, so two configs with
  one name built one head (the VLM factory had the same defect, D-060).
  Reproduced.
* ``input_dim`` and ``intermediate_size`` were unchecked; ``input_dim=0``
  built and failed in the norm. Reproduced.
* ``TextSimilarityHead`` took any ``similarity_function``; a pair then failed
  with ``UnboundLocalError`` on ``similarity``. Found at this step, not by the
  review; reproduced.
"""

import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import (
    MultiTaskNLPHead,
    NLPTaskConfig,
    NLPTaskType,
    create_multi_task_nlp_head,
    create_nlp_head,
)

D = 8
_SEQ = np.random.default_rng(1).normal(size=(2, 5, D)).astype("float32")

CONFIGS = {
    "TextClassificationHead": dict(task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3),
    "TokenClassificationHead": dict(task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=3),
    "QuestionAnsweringHead": dict(task_type=NLPTaskType.QUESTION_ANSWERING),
    "TextSimilarityHead": dict(task_type=NLPTaskType.TEXT_SIMILARITY),
    "TextGenerationHead": dict(task_type=NLPTaskType.TEXT_GENERATION, vocabulary_size=11),
    "MultipleChoiceHead": dict(task_type=NLPTaskType.MULTIPLE_CHOICE),
}
_POOLING_HEADS = ("TextClassificationHead", "TextSimilarityHead", "MultipleChoiceHead")
_SEQUENCE_HEADS = ("TokenClassificationHead", "QuestionAnsweringHead", "TextGenerationHead")


def _cfg(name):
    return NLPTaskConfig(name="t", **CONFIGS[name])


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("pooling", ["bogus", "CLS", "", None, 3])
def test_a_bad_pooling_type_is_refused_at_construction(name, pooling) -> None:
    with pytest.raises(ValueError, match=rf"{name}.*pooling_type"):
        create_nlp_head(_cfg(name), D, pooling_type=pooling)


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("pooling", ["cls", "mean", "max", "last", "attention"])
def test_every_known_pooling_type_is_accepted(name, pooling) -> None:
    create_nlp_head(_cfg(name), D, pooling_type=pooling)


@pytest.mark.parametrize("name", _SEQUENCE_HEADS)
def test_a_head_that_never_pools_refuses_use_pooling(name) -> None:
    with pytest.raises(ValueError, match=rf"{name}.*use_pooling"):
        create_nlp_head(_cfg(name), D, use_pooling=True)


@pytest.mark.parametrize("name", ("TextClassificationHead", "TextSimilarityHead"))
def test_a_sequence_without_a_pooler_is_refused_by_name(name) -> None:
    head = create_nlp_head(_cfg(name), D, use_pooling=False)
    with pytest.raises(ValueError, match=rf"{name}.*use_pooling"):
        head(_SEQ)
    assert head(_SEQ.mean(axis=1))  # an already pooled input needs no pooler


def test_multiple_choice_without_a_pooler_refuses_unpooled_choices() -> None:
    head = create_nlp_head(_cfg("MultipleChoiceHead"), D, use_pooling=False)
    choices = np.zeros((2, 3, 5, D), "float32")
    with pytest.raises(ValueError, match="MultipleChoiceHead.*use_pooling"):
        head(choices)
    assert head(choices.mean(axis=2))


@pytest.mark.parametrize("value", [0, -1, 2.5, True, None])
def test_input_dim_must_be_a_positive_integer(value) -> None:
    with pytest.raises(ValueError, match="TextClassificationHead.*input_dim"):
        create_nlp_head(_cfg("TextClassificationHead"), value)


@pytest.mark.parametrize("value", [0, 2.5, True])
def test_intermediate_size_must_be_a_positive_integer(value) -> None:
    with pytest.raises(ValueError, match="TextClassificationHead.*intermediate_size"):
        create_nlp_head(_cfg("TextClassificationHead"), D, intermediate_size=value)


def test_an_empty_task_set_is_refused() -> None:
    with pytest.raises(ValueError, match="MultiTaskNLPHead.*task"):
        MultiTaskNLPHead(task_configs={}, shared_input_dim=D)
    with pytest.raises(ValueError, match="MultiTaskNLPHead.*task"):
        create_multi_task_nlp_head([], D)


def test_a_list_with_a_repeated_name_is_refused_naming_it() -> None:
    a = NLPTaskConfig(name="dup", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=2)
    b = NLPTaskConfig(name="dup", task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=3)
    with pytest.raises(ValueError, match="dup"):
        create_multi_task_nlp_head([a, b], D)


@pytest.mark.parametrize("function", ["bogus", "Cosine", None])
def test_a_bad_similarity_function_is_refused_at_construction(function) -> None:
    """Measured at ``846589b7f``: it built, and a pair then hit an unbound name."""
    with pytest.raises(ValueError, match="TextSimilarityHead.*similarity_function"):
        create_nlp_head(_cfg("TextSimilarityHead"), D, similarity_function=function)
