"""``VisionTaskType`` is the vision task enum's only name.

Guard for plan-2026-10-14T062616-81938e81 step 7. Until then
``layers/heads/vision/task_types.py`` kept ``TaskType = VisionTaskType`` as a
back-compat alias, re-exported by ``heads.vision``, ``heads`` and
``heads.task_types``. The user ruled out legacy names, the alias had no in-tree
importer, and it was deleted with its anchor (``plan_2026-06-08_8b32ca51/D-003``,
retired per D-009). The NLP and VLM enums never had one.
"""

import importlib

import pytest

_MODULES = (
    "dl_techniques.layers.heads",
    "dl_techniques.layers.heads.task_types",
    "dl_techniques.layers.heads.vision",
    "dl_techniques.layers.heads.vision.task_types",
)


@pytest.mark.parametrize("module_name", _MODULES)
def test_no_module_binds_or_exports_the_old_name(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert not hasattr(module, "TaskType"), f"{module_name} still binds TaskType"
    assert "TaskType" not in getattr(module, "__all__", ()), (
        f"{module_name}.__all__ still exports TaskType"
    )


@pytest.mark.parametrize("module_name", _MODULES[:3])
def test_every_aggregator_still_exports_the_enum(module_name: str) -> None:
    """Anti-vacuity: the enum itself is still reachable where it was."""
    from dl_techniques.layers.heads.vision.task_types import VisionTaskType

    module = importlib.import_module(module_name)
    assert "VisionTaskType" in module.__all__
    assert module.VisionTaskType is VisionTaskType
