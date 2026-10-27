"""One table routes an NLP task type to its head, and the output types describe that head.

Guard for plan-2026-10-14T062616-81938e81 step 11.3 (the user's heads review,
``research/reviews/20261015_heads_review.md``, NLP validation gaps, and the
11.1/11.2 dispatch pattern, D-052 and D-059). Measured at ``846589b7f``:

* Three hand-kept lists said which task types an NLP head serves:
  ``get_head_class``'s mapping (24 rows), ``NLPTaskConfig.__post_init__``'s
  classification and generation lists (5 and 5 rows, which is how NLP-7 and
  the ``MACHINE_TRANSLATION`` vocabulary fill happened) and
  ``model_head._GENERATION_HEAD_TASKS`` (4 rows). Now ``NLP_HEAD_DISPATCH``
  in ``nlp/task_types.py`` is the one table; ``get_head_class``, the config
  checks and ``model_head`` read it. It names head classes rather than holding
  them, because ``nlp/task_types.py`` cannot import ``nlp/factory.py``.
* ``NLPTaskType.get_output_types`` documented keys no head returned for 6 of
  its 11 entries (token ``labels``, QA ``answer_spans``, similarity
  ``similarity_score`` next to ``embeddings``, generation ``generated_ids``,
  span ``span_labels``, regression ``value``/``confidence``), described
  ``RANKING``, which has no head, and fell back to ``{'output': ...}`` for 26
  more. Now it names exactly the keys each dispatched head returns and raises
  for the 13 task types with no head.
* ``get_head_class`` raised ``AttributeError`` for a string task type.
* ``NLPHeadConfiguration.get_default_config`` returned ``dropout_rate``,
  ``use_crf`` and ``vocabulary_size`` (task-config fields, not head
  arguments: ``create_nlp_head(cfg, D, **defaults)`` raised ``TypeError``) and
  had no caller. Deleted.
* ``BaseNLPHead`` was not exported, unlike ``BaseVisionHead`` and
  ``BaseVLMHead``. Exported now.
"""

import numpy as np
import pytest

import dl_techniques.layers.heads as heads
import dl_techniques.layers.heads.nlp as nlp
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType, create_nlp_head
from dl_techniques.layers.heads.nlp import dispatch as nlp_dispatch
from dl_techniques.layers.heads.nlp import model_head
from dl_techniques.layers.heads.nlp.base import BaseNLPHead
from dl_techniques.layers.heads.nlp.task_types import NLP_HEAD_DISPATCH

D = 8
_SEQ = np.random.default_rng(9).normal(size=(2, 5, D)).astype("float32")
_CHOICES = np.random.default_rng(9).normal(size=(2, 3, 5, D)).astype("float32")
_NEEDS_CLASSES = {"TextClassificationHead", "TokenClassificationHead"}


@pytest.mark.parametrize("task", sorted(NLPTaskType, key=lambda t: t.name), ids=lambda t: t.name)
def test_get_head_class_reads_the_table(task) -> None:
    if task not in NLP_HEAD_DISPATCH:
        with pytest.raises(ValueError, match=task.name):
            nlp_dispatch.get_head_class(task)
        return
    head_class = nlp_dispatch.get_head_class(task)
    assert head_class.__name__ == NLP_HEAD_DISPATCH[task]
    assert nlp_dispatch.get_head_class(task.value) is head_class


def test_the_table_names_six_real_head_classes() -> None:
    names = set(NLP_HEAD_DISPATCH.values())
    assert names == {"TextClassificationHead", "TokenClassificationHead",
                     "QuestionAnsweringHead", "TextSimilarityHead",
                     "TextGenerationHead", "MultipleChoiceHead"}
    for name in names:
        assert issubclass(getattr(nlp, name), BaseNLPHead)
    assert len(NLP_HEAD_DISPATCH) == 24


def test_the_model_head_generation_set_is_read_from_the_table() -> None:
    assert model_head._GENERATION_HEAD_TASKS == frozenset(
        t for t, h in NLP_HEAD_DISPATCH.items() if h == "TextGenerationHead")


def _live_keys(task):
    extra = {"num_classes": 3} if NLP_HEAD_DISPATCH[task] in _NEEDS_CLASSES else {}
    head = create_nlp_head(NLPTaskConfig(name="t", task_type=task, **extra), input_dim=D)
    out = head(_CHOICES if NLP_HEAD_DISPATCH[task] == "MultipleChoiceHead" else _SEQ)
    return set(out)


@pytest.mark.parametrize("task", sorted(NLP_HEAD_DISPATCH, key=lambda t: t.name),
                         ids=lambda t: t.name)
def test_the_documented_keys_are_the_returned_keys(task) -> None:
    assert set(NLPTaskType.get_output_types(task)) == _live_keys(task)


@pytest.mark.parametrize("task", sorted((t for t in NLPTaskType if t not in NLP_HEAD_DISPATCH),
                                        key=lambda t: t.name), ids=lambda t: t.name)
def test_a_task_with_no_head_has_no_output_types(task) -> None:
    with pytest.raises(ValueError, match=task.value):
        NLPTaskType.get_output_types(task)


def test_the_dead_default_config_helper_is_gone() -> None:
    assert not hasattr(nlp_dispatch, "NLPHeadConfiguration")


def test_base_nlp_head_is_exported_like_the_other_bases() -> None:
    assert nlp.BaseNLPHead is BaseNLPHead
    assert heads.BaseNLPHead is BaseNLPHead
    assert "BaseNLPHead" in nlp.__all__ and "BaseNLPHead" in heads.__all__
    assert "BaseVisionHead" in heads.__all__ and "BaseVLMHead" in heads.__all__
