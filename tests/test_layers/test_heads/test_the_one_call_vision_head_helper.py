"""Every vision model builds its head through ``build_model_vision_head``.

Guard for plan-2026-10-14T062616-81938e81 step 11.1 (review item P1 "one-call
vision helper", decision D-053). At ``9343b0f87`` 19 construction sites in
``models/vision`` and FastVLM spelled out the same four steps
(``HeadConfiguration.get_default_config`` -> ``dict.update`` ->
``merge_head_config_overrides`` -> ``create_vision_head``); three of them
(AlexNet, CBAMNet, BEiT) skipped the merge step. One helper now owns the
order. Every consumer's parameter count and fixed-seed output were measured
unchanged across the migration (the step's census table).

Two claims:

* the helper's contract: preset, then the model's keys, then the caller's
  overrides, the reserved keys refused, the model's dict never mutated, and a
  task type with no head refused;
* the census: no file under ``src/dl_techniques/models`` calls
  ``HeadConfiguration``, ``merge_head_config_overrides`` or the four-step
  sequence any more, and every file that builds a head from a preset does it
  through the helper.
"""

import pathlib
import re

import pytest

from dl_techniques.layers.heads.vision import (
    ClassificationHead,
    HeadConfiguration,
    VisionTaskType,
)
from dl_techniques.layers.heads.vision.factory import build_model_vision_head

MODELS_ROOT = (pathlib.Path(__file__).resolve().parents[3]
               / "src" / "dl_techniques" / "models")


def test_the_model_keys_win_over_the_preset_and_the_overrides_win_last() -> None:
    model_keys = {"num_classes": 4, "block_dims": (), "name": "classification_head",
                  "classifier_dropout_rate": 0.1}
    head = build_model_vision_head(
        VisionTaskType.CLASSIFICATION, model_keys,
        {"classifier_dropout_rate": 0.3})
    preset = HeadConfiguration.get_default_config(VisionTaskType.CLASSIFICATION)
    assert isinstance(head, ClassificationHead)
    assert (head.num_classes, head.block_dims, head.name) == (
        4, (), "classification_head")
    assert head.classifier_dropout_rate == 0.3
    # A preset key neither the model nor the caller set comes from the preset.
    assert head.hidden_dim == preset["hidden_dim"]
    assert model_keys["classifier_dropout_rate"] == 0.1  # not mutated


def test_a_string_task_type_and_no_overrides_are_accepted() -> None:
    head = build_model_vision_head("classification", {"num_classes": 4})
    assert isinstance(head, ClassificationHead)


@pytest.mark.parametrize("reserved", ["name", "num_classes", "task_type"])
def test_an_override_of_a_model_owned_key_is_refused(reserved) -> None:
    with pytest.raises(ValueError, match=reserved):
        build_model_vision_head(VisionTaskType.CLASSIFICATION,
                                {"num_classes": 4}, {reserved: 1})


def test_a_task_type_with_no_head_is_refused() -> None:
    with pytest.raises(ValueError, match="pose_estimation"):
        build_model_vision_head("pose_estimation", {})


def _model_sources():
    return sorted(MODELS_ROOT.rglob("*.py"))


def test_no_model_spells_out_the_four_step_sequence() -> None:
    offenders = []
    for path in _model_sources():
        text = path.read_text()
        for needle in ("HeadConfiguration", "merge_head_config_overrides",
                       "**head_config"):
            if needle in text:
                offenders.append(f"{path.relative_to(MODELS_ROOT)}: {needle}")
    assert not offenders, offenders


def test_the_helper_census_is_twenty_two_sites() -> None:
    """The count is the census at the migration (19), plus FFTNet's detection
    and segmentation heads (iteration 2 step 6.6, D-210: they used to call
    ``create_vision_head`` with four override keys read by hand and the rest
    dropped in silence), plus FFTNet's classification head (step 10.1,
    D-229: a hand-built ``Sequential`` until the class took its heads); a new
    consumer raises it, and must be added here on purpose rather than by a
    fifth copy of the sequence."""
    calls = {str(p.relative_to(MODELS_ROOT)): len(re.findall(
        r"\bbuild_model_vision_head\(", p.read_text())) for p in _model_sources()}
    calls = {k: v for k, v in calls.items() if v}
    assert sum(calls.values()) == 22, calls
    assert calls["vision/resnet/model.py"] == 2
    assert calls["vision/fftnet/model.py"] == 3
