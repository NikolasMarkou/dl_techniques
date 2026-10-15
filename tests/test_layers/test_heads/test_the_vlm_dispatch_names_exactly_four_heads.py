"""One VLM dispatch table names the four heads, and the docs name their keys.

Guard for plan-2026-10-14T062616-81938e81 step 11.2 (the step 11.1 pattern,
decision D-052, applied to the VLM heads). Measured at ``8b4182f9d``:
``get_head_class`` built its table on each call and mapped two task types,
``DENSE_CAPTIONING`` and ``VISUAL_DIALOGUE``, to ``BaseVLMHead`` only to
reject them a line later; a string task type died with ``AttributeError:
'str' object has no attribute 'name'``.

The VLM domain has no ``get_output_types``; its output contract is the table
in ``vlm/README.md`` § 2 and each head's ``compute_output_shape``. The last
test pins the README table against what each head returns
(``test_every_vlm_head_reports_the_shapes_it_returns.py`` pins
``compute_output_shape``).
"""

import pathlib
import re

import numpy as np
import pytest

from dl_techniques.layers.heads.vlm import (
    BaseVLMHead,
    ImageCaptioningHead,
    ImageTextMatchingHead,
    VisualGroundingHead,
    VLMTaskConfig,
    VLMTaskType,
    VQAHead,
    create_vlm_head,
)
from dl_techniques.layers.heads.vlm import factory as vlm_factory

FOUR = {
    VLMTaskType.IMAGE_CAPTIONING: ImageCaptioningHead,
    VLMTaskType.VISUAL_QUESTION_ANSWERING: VQAHead,
    VLMTaskType.VISUAL_GROUNDING: VisualGroundingHead,
    VLMTaskType.IMAGE_TEXT_MATCHING: ImageTextMatchingHead,
}

README = (pathlib.Path(__file__).resolve().parents[3]
          / "src/dl_techniques/layers/heads/vlm/README.md")


def test_the_dispatch_table_holds_the_four_heads_and_nothing_else() -> None:
    assert vlm_factory.VLM_HEAD_DISPATCH == FOUR
    assert BaseVLMHead not in vlm_factory.VLM_HEAD_DISPATCH.values()


@pytest.mark.parametrize("task", list(FOUR), ids=lambda t: t.value)
def test_get_head_class_reads_the_table(task) -> None:
    assert vlm_factory.get_head_class(task) is FOUR[task]
    assert vlm_factory.get_head_class(task.value) is FOUR[task]


@pytest.mark.parametrize("task", [t for t in VLMTaskType if t not in FOUR],
                         ids=lambda t: t.value)
def test_every_other_task_type_raises_listing_the_four(task) -> None:
    with pytest.raises(ValueError, match=task.name) as info:
        vlm_factory.get_head_class(task)
    for implemented in FOUR:
        assert implemented.name in str(info.value)


def _readme_output_keys() -> dict:
    """The output dict keys per task row of the README § 2 output table."""
    rows = {}
    for line in README.read_text().splitlines():
        match = re.match(r"\|\s*\*\*(.+?)\*\*\s*\|\s*`\{(.+)\}`\s*\|", line)
        if match:
            rows[match.group(1)] = set(re.findall(r"'(\w+)'", match.group(2)))
    return rows


_README_ROW = {
    VLMTaskType.IMAGE_CAPTIONING: "Image Captioning",
    VLMTaskType.VISUAL_QUESTION_ANSWERING: "VQA",
    VLMTaskType.VISUAL_GROUNDING: "Visual Grounding",
    VLMTaskType.IMAGE_TEXT_MATCHING: "Matching",
}


@pytest.mark.parametrize("task", list(FOUR), ids=lambda t: t.value)
def test_the_readme_output_table_names_exactly_the_returned_keys(task) -> None:
    documented = _readme_output_keys()
    assert set(documented) == set(_README_ROW.values()), sorted(documented)
    extra = {VLMTaskType.IMAGE_CAPTIONING: {"vocab_size": 9},
             VLMTaskType.VISUAL_QUESTION_ANSWERING: {"num_classes": 3}}.get(task, {})
    kwargs = {"num_layers": 1, "num_heads": 4} if task is VLMTaskType.IMAGE_CAPTIONING else {}
    head = create_vlm_head(VLMTaskConfig(name="d", task_type=task, hidden_size=16,
                                         **extra), **kwargs)
    x = np.random.default_rng(1).normal(size=(2, 5, 16)).astype("float32")
    text_key = ("question_features" if task is VLMTaskType.VISUAL_QUESTION_ANSWERING
                else "text_features")
    out = head({"vision_features": x, text_key: x})
    assert set(out) == documented[_README_ROW[task]]
