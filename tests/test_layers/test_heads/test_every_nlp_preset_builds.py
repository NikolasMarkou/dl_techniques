"""Every ``CommonNLPTaskConfigurations`` preset names only task types that have a head.

Guard for plan-2026-10-14T062616-81938e81 step 11.4(e) (D-068, a Step 11.3
leftover). Measured at ``56b696fe2``: ``TOKEN_TASKS`` held
``DEPENDENCY_PARSING`` and ``INFORMATION_EXTRACTION`` held
``RELATION_EXTRACTION`` and ``EVENT_EXTRACTION``. None of the three is in
``NLP_HEAD_DISPATCH``, so building a multi-task head from either preset
raised ``ValueError`` before anything ran.

Each preset is found by type, not by a hand-kept list, so a preset added
later is checked too. A preset builds when a multi-task head made from it
constructs and every task head runs on its input form.
"""

import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, create_multi_task_nlp_head
from dl_techniques.layers.heads.nlp.task_types import (
    NLP_HEAD_DISPATCH, CommonNLPTaskConfigurations, NLPTaskConfiguration)

D = 8
_SEQ = np.random.default_rng(5).normal(size=(2, 5, D)).astype("float32")
_CHOICES = np.random.default_rng(5).normal(size=(2, 3, 5, D)).astype("float32")
_NEEDS_CLASSES = {"TextClassificationHead", "TokenClassificationHead"}

PRESETS = sorted(
    name for name, value in vars(CommonNLPTaskConfigurations).items()
    if isinstance(value, NLPTaskConfiguration))


def test_the_presets_are_found() -> None:
    assert len(PRESETS) >= 10, PRESETS


@pytest.mark.parametrize("preset", PRESETS)
def test_the_preset_names_only_dispatched_task_types(preset) -> None:
    tasks = getattr(CommonNLPTaskConfigurations, preset).tasks
    headless = sorted(t.name for t in tasks if t not in NLP_HEAD_DISPATCH)
    assert not headless, f"{preset} names task types with no head: {headless}"


@pytest.mark.parametrize("preset", PRESETS)
def test_a_multi_task_head_builds_from_the_preset(preset) -> None:
    tasks = sorted(getattr(CommonNLPTaskConfigurations, preset).tasks, key=lambda t: t.name)
    configs = [
        NLPTaskConfig(name=t.value, task_type=t,
                      **({"num_classes": 3} if NLP_HEAD_DISPATCH[t] in _NEEDS_CLASSES else {}))
        for t in tasks]
    head = create_multi_task_nlp_head(configs, input_dim=D)
    for task in tasks:
        inputs = _CHOICES if NLP_HEAD_DISPATCH[task] == "MultipleChoiceHead" else _SEQ
        out = head(inputs, task_name=task.value, training=False)
        assert out, (preset, task.name)
