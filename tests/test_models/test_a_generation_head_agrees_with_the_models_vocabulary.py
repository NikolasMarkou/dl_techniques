"""A generation-type head's vocabulary is the model's own, in every class that owns a head.

Subject: the fourteen language classes with ``include_top`` and the
``vocab_size`` they hand to ``build_model_task_head``
(plan-2026-10-11T170958-dc4d2202 step 8c, D-009). A generation head emits logits
over ``task_config.vocabulary_size``, which defaulted to 32000 whatever the
model: BERT with vocabulary 64 and ``TEXT_SUMMARIZATION`` returned
``(batch, seq_len, 32000)``. The check lives in the helper, so each class only
has to pass its vocabulary; this file is the census that every one does.

``test_an_unset_vocabulary_is_the_models``
    A dict task config without ``vocabulary_size`` yields a head (and a
    ``get_config`` that round-trips) with the model's.
``test_a_different_vocabulary_is_refused_at_construction``
    An explicit other value raises ``ValueError`` before any weight exists; the
    matching value is accepted, and a classification head is unaffected.
"""

import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

from .tiny_language_models import TINY, VOCAB

SUBJECTS = tuple(sorted(TINY))
_GENERATION = {"name": "g", "task_type": "text_summarization"}


def _build(model_name, **task):
    tiny = TINY[model_name]
    return tiny.cls()(**tiny.kwargs, include_top=True, task_config=task)


@pytest.fixture(params=SUBJECTS)
def subject(request):
    return request.param


def test_the_census_covers_the_fourteen_head_owners():
    assert len(SUBJECTS) == 14


def test_an_unset_vocabulary_is_the_models(subject):
    model = _build(subject, **_GENERATION)
    assert model.task_head.task_config.vocabulary_size == VOCAB
    config = model.get_config()
    assert config["task_config"]["vocabulary_size"] == VOCAB
    rebuilt = type(model).from_config(config)
    assert rebuilt.task_head.task_config.vocabulary_size == VOCAB


def test_a_different_vocabulary_is_refused_at_construction(subject):
    other = VOCAB + 1
    with pytest.raises(ValueError, match=f"vocabulary_size={other}"):
        _build(subject, **_GENERATION, vocabulary_size=other)
    ok = _build(subject, **_GENERATION, vocabulary_size=VOCAB)
    assert ok.task_head.task_config.vocabulary_size == VOCAB
    clf = _build(subject, name="c", task_type="text_classification", num_classes=3)
    assert clf.task_head is not None


def test_a_dataclass_that_kept_the_default_is_refused(subject):
    config = NLPTaskConfig(name="g", task_type=NLPTaskType.MASKED_LANGUAGE_MODELING)
    tiny = TINY[subject]
    with pytest.raises(ValueError, match="vocabulary_size=32000"):
        tiny.cls()(**tiny.kwargs, include_top=True, task_config=config)
