"""``MultiTaskHead``, its presets and ``HeadConfiguration`` cover exactly the
task types ``create_vision_head`` dispatches, and refuse the rest by name.

Guard for plan-2026-10-14T062616-81938e81 step 11.1 (review items VIS-6,
VIS-7 and the multi-task/preset/``HeadConfiguration`` items). Measured at
``9343b0f87``:

* ``MultiTaskHead`` kept its own five-branch dispatch, so surface normals,
  optical flow, keypoints, denoising and super-resolution, which
  ``create_vision_head`` builds, raised ``Unsupported task type``.
* Five ``CommonTaskConfigurations`` presets named a task with no head at all
  (``PANOPTIC_SEGMENTATION_ONLY``, ``GEOMETRIC_UNDERSTANDING``,
  ``IMAGE_ENHANCEMENT``, ``POSE_AND_STRUCTURE``, ``ALL_TASKS``), and seven more
  named a task ``MultiTaskHead`` could not build.
* VIS-6: a dict input with neither the task's key nor ``'shared'`` handed
  ``None`` to the head: ``TypeError: 'NoneType' object is not subscriptable``.
* VIS-7 (refuted, pinned): the review read ``self.task_configs =
  task_configs`` as an alias of the caller's outer dict. Measured at
  ``9343b0f87``: Keras' attribute tracking stores a dict assigned to a layer
  as a ``TrackedDict`` copy, nested dicts included, so a task the caller adds
  or edits afterwards reaches neither the heads nor ``get_config``. The two
  aliasing tests below passed at ``9343b0f87`` and pin that.
* ``HeadConfiguration`` returned the seven shared keys for any of the 27 task
  types with no head, and a keypoint preset with no ``num_classes`` that its
  own head refused (``TypeError``).
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vision import (
    CommonTaskConfigurations,
    HeadConfiguration,
    MultiTaskHead,
    TaskConfiguration,
    VisionTaskType,
    create_multi_task_head,
    create_vision_head,
)

HIDDEN_DIM = 16
SPATIAL = (2, 6, 10, HIDDEN_DIM)

#: The minimal per-task options for each dispatched task type.
DISPATCHED: Dict[VisionTaskType, Dict[str, Any]] = {
    VisionTaskType.DETECTION: {"num_classes": 3},
    VisionTaskType.SEGMENTATION: {"num_classes": 3, "use_skip_connections": False},
    VisionTaskType.INSTANCE_SEGMENTATION: {"num_classes": 3, "num_instances": 4},
    VisionTaskType.CLASSIFICATION: {"num_classes": 3},
    VisionTaskType.DEPTH_ESTIMATION: {},
    VisionTaskType.SURFACE_NORMALS: {},
    VisionTaskType.OPTICAL_FLOW: {},
    VisionTaskType.KEYPOINT_DETECTION: {"num_classes": 65},
    VisionTaskType.DENOISING: {},
    VisionTaskType.SUPER_RESOLUTION: {},
}
UNDISPATCHED = [t for t in VisionTaskType if t not in DISPATCHED]


def _inputs(shape=SPATIAL) -> np.ndarray:
    return np.random.default_rng(13).normal(size=shape).astype("float32")


def test_the_dispatched_set_is_ten_and_the_rest_raise() -> None:
    assert len(DISPATCHED) == 10 and len(UNDISPATCHED) == 27
    for task in UNDISPATCHED:
        with pytest.raises(ValueError, match=task.name):
            create_vision_head(task, hidden_dim=HIDDEN_DIM)


@pytest.mark.parametrize("task", list(DISPATCHED), ids=lambda t: t.value)
def test_a_multi_task_head_builds_the_class_create_vision_head_builds(task) -> None:
    options = {"hidden_dim": HIDDEN_DIM, **DISPATCHED[task]}
    head = MultiTaskHead({"t": {"task_type": task.value, **options}},
                         use_task_specific_attention=False)
    single = create_vision_head(task, **options)
    assert type(head.task_heads["t"]) is type(single)
    assert head.task_heads["t"].get_config() == {**single.get_config(),
                                                 "name": "t_head"}
    head(_inputs())


@pytest.mark.parametrize("task", UNDISPATCHED[:3], ids=lambda t: t.value)
def test_a_task_with_no_head_is_refused_naming_the_entry_and_the_task(task) -> None:
    with pytest.raises(ValueError, match=rf"'pose'.*{task.name}|{task.name}.*'pose'"):
        MultiTaskHead({"pose": {"task_type": task, "hidden_dim": HIDDEN_DIM}},
                      use_task_specific_attention=False)


# ---------------------------------------------------------------------
# VIS-6 / VIS-7
# ---------------------------------------------------------------------

def _det_head() -> MultiTaskHead:
    return MultiTaskHead(
        {"det": {"task_type": "detection", "num_classes": 3,
                 "hidden_dim": HIDDEN_DIM}},
        use_task_specific_attention=False)


def test_a_dict_input_with_no_entry_for_a_task_is_refused_by_name() -> None:
    head = _det_head()
    with pytest.raises(ValueError, match="'det'"):
        head({"other": _inputs()})
    with pytest.raises(ValueError, match="'det'"):
        head.compute_output_shape({"other": SPATIAL})
    with pytest.raises(ValueError, match="'det'"):
        _det_head().build({"other": SPATIAL})


def test_the_shared_entry_still_serves_a_task_with_no_own_key() -> None:
    out = _det_head()({"shared": _inputs()})
    assert out["det"]["classifications"].shape == (2, 6, 10, 27)


def test_the_callers_outer_dict_is_not_aliased() -> None:
    caller = {"det": {"task_type": "detection", "num_classes": 3,
                      "hidden_dim": HIDDEN_DIM}}
    head = MultiTaskHead(caller, use_task_specific_attention=False)
    caller["seg"] = {"task_type": "segmentation", "num_classes": 3}
    caller["det"]["num_classes"] = 99
    config = head.get_config()["task_configs"]
    assert sorted(config) == ["det"]
    assert config["det"]["num_classes"] == 3


def test_create_multi_task_head_does_not_alias_a_dict_configuration() -> None:
    caller = {"det": {"task_type": "detection", "num_classes": 3,
                      "hidden_dim": HIDDEN_DIM}}
    head = create_multi_task_head(caller, use_task_specific_attention=False)
    caller.clear()
    assert sorted(head.get_config()["task_configs"]) == ["det"]


# ---------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------

def _presets() -> Dict[str, TaskConfiguration]:
    return {name: value for name, value in vars(CommonTaskConfigurations).items()
            if isinstance(value, TaskConfiguration)}


def test_every_listed_preset_is_a_named_attribute() -> None:
    named = {id(v) for v in _presets().values()}
    listed = list(CommonTaskConfigurations.get_all_configurations())
    for group in CommonTaskConfigurations.get_configurations_by_complexity().values():
        listed.extend(group)
    assert {id(v) for v in listed} <= named
    assert {id(v) for v in CommonTaskConfigurations.get_all_configurations()} == named


@pytest.mark.parametrize("name", sorted(_presets()))
def test_every_preset_builds_a_multi_task_head(name: str) -> None:
    preset = _presets()[name]
    options = {task.value: {"hidden_dim": HIDDEN_DIM, **DISPATCHED.get(task, {})}
               for task in preset.get_enabled_tasks()}
    keras.utils.set_random_seed(3)
    head = create_multi_task_head(preset, use_task_specific_attention=False,
                                  **options)
    assert sorted(head.task_heads) == sorted(t.value for t in preset.tasks)
    head(_inputs())


# ---------------------------------------------------------------------
# HeadConfiguration
# ---------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["get_default_config", "get_efficient_config"])
@pytest.mark.parametrize("task", list(DISPATCHED), ids=lambda t: t.value)
def test_every_dispatched_task_has_a_preset_that_builds_its_head(preset, task) -> None:
    config = getattr(HeadConfiguration, preset)(task)
    config["hidden_dim"] = HIDDEN_DIM
    keras.utils.set_random_seed(3)
    head = create_vision_head(task, **config)
    head(_inputs())


@pytest.mark.parametrize("preset", ["get_default_config", "get_efficient_config"])
@pytest.mark.parametrize("task", UNDISPATCHED, ids=lambda t: t.value)
def test_a_task_with_no_head_has_no_preset(preset, task) -> None:
    with pytest.raises(ValueError, match=task.name):
        getattr(HeadConfiguration, preset)(task)
