"""The vision head builders do not refuse a config over their own defaults.

Measured on 2026-11-02 before the fixes (audit of ``layers/heads``):

* ``build_model_vision_head('depth_estimation', {'use_ffn': False})`` raised:
  the preset's ``dropout_rate=0.1`` outlived the stage it configures. Normals
  and flow failed the same way.
* ``create_multi_task_head(['denoising'])`` raised: the default
  ``use_task_specific_attention=True`` was forced onto ``EnhancementHead``.
* ``MultiTaskHead`` raised ``KeyError`` on a config with no ``task_type`` and
  ``AttributeError`` on a list.
"""

import numpy as np
import pytest

from dl_techniques.layers.heads.vision import MultiTaskHead, create_multi_task_head
from dl_techniques.layers.heads.vision.config import build_model_vision_head

DENSE_TASKS = ["depth_estimation", "surface_normals", "optical_flow"]


@pytest.mark.parametrize("task", DENSE_TASKS)
def test_turning_the_stage_off_takes_the_preset_dropout_with_it(task) -> None:
    head = build_model_vision_head(task, {"use_ffn": False})
    assert head.use_ffn is False
    assert head.dropout_rate == 0.0


@pytest.mark.parametrize("task", DENSE_TASKS)
def test_overrides_can_turn_the_stage_off_too(task) -> None:
    head = build_model_vision_head(task, {}, {"use_ffn": False})
    assert head.dropout_rate == 0.0


@pytest.mark.parametrize("task", DENSE_TASKS)
def test_a_dropout_the_caller_set_is_still_refused(task) -> None:
    with pytest.raises(ValueError, match="dropout_rate"):
        build_model_vision_head(task, {"use_ffn": False, "dropout_rate": 0.2})


@pytest.mark.parametrize("task", DENSE_TASKS)
def test_the_preset_dropout_stays_while_a_stage_runs(task) -> None:
    assert build_model_vision_head(task, {}).dropout_rate == 0.1


def test_a_head_that_reads_dropout_elsewhere_keeps_the_preset_value() -> None:
    head = build_model_vision_head("detection", {"use_ffn": False, "use_attention": False})
    assert head.dropout_rate == 0.1


@pytest.mark.parametrize("task", ["denoising", "super_resolution"])
def test_a_multi_task_head_serves_enhancement_under_its_defaults(task) -> None:
    head = create_multi_task_head([task], shared_backbone_dim=16)
    outputs = head(np.zeros((1, 4, 4, 8), "float32"))
    assert "enhanced" in outputs[task]
    assert head.use_task_specific_attention is True


def test_the_attention_default_still_reaches_the_other_heads() -> None:
    head = create_multi_task_head(
        ["detection", "denoising"], detection={"num_classes": 2},
        shared_backbone_dim=16)
    assert head.task_heads["detection"].use_attention is True
    assert head.task_heads["denoising"].use_attention is False


def test_a_per_task_attention_request_on_enhancement_is_still_refused() -> None:
    with pytest.raises(ValueError, match="use_attention"):
        MultiTaskHead({"d": {"task_type": "denoising", "use_attention": True}},
                      shared_backbone_dim=16)


def test_a_config_with_no_task_type_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="'a'.*task_type"):
        MultiTaskHead({"a": {"num_classes": 2}})


@pytest.mark.parametrize("configs", [[{"task_type": "detection"}], ("detection",)])
def test_a_non_dict_task_configs_is_refused(configs) -> None:
    with pytest.raises(ValueError, match="create_multi_task_head"):
        MultiTaskHead(configs)
