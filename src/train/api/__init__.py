"""Training API: strict configs, a key=value command line, one function per task.

The package surface is lazy (PEP 562): ``import train.api`` loads nothing,
and a name is imported from its module on first access. That keeps
``python -m train help`` free of TensorFlow, Keras and ``yaml``.

A name enters ``__all__`` when its module exists (D-007).

``export`` is a function in the module ``train.api.exporting`` (D-068): a submodule named
``export`` would replace the function on the package the moment it was first imported, so
``train.api.export`` would be the module after the command line ran. The submodule has a
different name, and ``train.api.export`` is the function in every import order.
"""

import importlib
from typing import Dict, List

# DECISION plan-2026-10-30T132432-a865fa08/D-068: the row ``export`` points at the module
# ``exporting``. Do NOT name that module ``export`` (importing it would bind the module over the
# function on this package, so ``train.api.export`` would depend on import order, as D-064
# accepted wrongly) and do NOT import it eagerly here (a fresh ``import train.api`` loads no
# submodule, pinned by test_registry).
_LAZY: Dict[str, str] = {
    "DLTrainError": ".errors",
    "ConfigError": ".errors",
    "ConfigValidationError": ".errors",
    "UnknownModelError": ".errors",
    "UnresolvedAutoError": ".errors",
    "StrictConfig": ".config",
    "AutoContext": ".args",
    "OptimArgs": ".args",
    "TransformArgs": ".args",
    "MetricArgs": ".args",
    "CheckpointArgs": ".args",
    "LoggerArgs": ".args",
    "AnalysisArgs": ".args",
    "DebugArgs": ".args",
    "ModelSpec": ".registry",
    "TaskSpec": ".registry",
    "register_model": ".registry",
    "register_task": ".registry",
    "parse_model_name": ".registry",
    "list_models": ".registry",
    "list_tasks": ".registry",
    "resolve_out": ".run_dir",
    "open_run": ".run_dir",
    "write_requested": ".run_dir",
    "write_resolved": ".run_dir",
    "main": ".cli",
    "BuiltinImageData": ".data_args",
    "FolderImageData": ".data_args",
    "CsvImageData": ".data_args",
    "load_data_config": ".data_args",
    "load_model": ".predictor",
    "export": ".exporting",
    "TaskTrainer": ".trainer",
    "TrainResult": ".trainer",
    "ImageClassificationConfig": ".tasks.image_classification",
    "train_image_classification": ".tasks.image_classification",
    "BuiltinSegData": ".segmentation_args",
    "FolderSegData": ".segmentation_args",
    "CsvSegData": ".segmentation_args",
    "load_seg_data_config": ".segmentation_args",
    "ImageSegmentationConfig": ".tasks.image_segmentation",
    "train_image_segmentation": ".tasks.image_segmentation",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    """Import ``name`` from its module on first access and cache it."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
