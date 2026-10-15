"""An NLP head keeps its own copy of every task config it was built from.

Guard for plan-2026-10-14T062616-81938e81 step 11.3 (the user's heads review,
``research/reviews/20261015_heads_review.md`` item NLP-5,
``MultiTaskNLPHead``'s ``self.task_configs = task_configs``). Measured at
``846589b7f``:

* The outer dict: refuted, as VIS-7 was for the vision head (D-052). Keras'
  attribute tracking stores a dict assigned to a layer as a ``TrackedDict``
  copy, so adding a task to the caller's dict after construction changed
  neither ``task_heads`` nor ``get_config``. The first test pins it; it passed
  at ``846589b7f``.
* The configs inside it: reproduced. The ``NLPTaskConfig`` instances were the
  caller's own objects, held by the wrapper and by every sub-head. Setting
  ``num_classes=7`` on the caller's config after construction left the built
  classifier at 3 units while ``get_config`` reported 7, so a rebuilt head
  had a different architecture than the saved weights. A single head had the
  same hole. Each head now stores a copy.
"""

import numpy as np

from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    NLPTaskType,
    create_multi_task_nlp_head,
    create_nlp_head,
)

D = 8
_SEQ = np.zeros((2, 5, D), "float32")


def _clf(name="clf", n=3):
    return NLPTaskConfig(name=name, task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=n)


def test_adding_a_task_to_the_callers_dict_changes_nothing() -> None:
    configs = {"clf": _clf()}
    head = create_multi_task_nlp_head(configs, D)
    configs["other"] = _clf("other", 2)
    assert sorted(head.task_heads) == ["clf"]
    assert sorted(head.get_config()["task_configs"]) == ["clf"]


def test_editing_the_callers_config_changes_neither_multi_task_head_nor_config() -> None:
    configs = {"clf": _clf()}
    head = create_multi_task_nlp_head(configs, D)
    configs["clf"].num_classes = 7
    assert head.get_config()["task_configs"]["clf"]["num_classes"] == 3
    assert head.task_heads["clf"].task_config.num_classes == 3
    assert tuple(head(_SEQ, task_name="clf")["logits"].shape) == (2, 3)


def test_editing_the_callers_config_changes_neither_single_head_nor_config() -> None:
    config = _clf()
    head = create_nlp_head(config, D)
    config.num_classes = 7
    assert head.get_config()["task_config"]["num_classes"] == 3
    assert tuple(head(_SEQ)["logits"].shape) == (2, 3)
