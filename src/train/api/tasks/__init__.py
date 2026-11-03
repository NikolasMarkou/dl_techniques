"""The training tasks of ``python -m train``, one module each.

Interface contract
------------------
``register_builtin_tasks()``
    Register every task of this package into
    :data:`train.api.registry.TASKS`. Idempotent: a task already registered is
    left alone. Imports no task code: a ``TaskSpec`` holds ``"module:attr"``
    text (D-012), so ``python -m train help`` stays free of TensorFlow.

Importing this package registers nothing, as ``train.api.registry`` itself
registers nothing at import: ``test_registry`` swaps both dicts for empty ones.
"""

from .. import registry

#: One row per task module, in listing order.
BUILTIN_TASKS = (
    registry.TaskSpec(
        name="image_classification",
        config_class="train.api.tasks.image_classification:ImageClassificationConfig",
        function="train.api.tasks.image_classification:train_image_classification",
        summary="train an image classifier on a built-in dataset",
    ),
    registry.TaskSpec(
        name="image_segmentation",
        config_class="train.api.tasks.image_segmentation:ImageSegmentationConfig",
        function="train.api.tasks.image_segmentation:train_image_segmentation",
        summary="train a semantic segmenter on Oxford-IIIT Pet or on image and mask files",
    ),
)


def register_builtin_tasks() -> None:
    """Register the built-in tasks; see the module contract."""
    for spec in BUILTIN_TASKS:
        if spec.name not in registry.TASKS:
            registry.register_task(spec)
