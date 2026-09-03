"""Residual network (ResNet) public API.

Re-exports the model class and the factory function. Internal callers may still
import from ``dl_techniques.models.vision.resnet.model`` directly.

Not re-exported here: ``get_model_output_info`` and
``create_inference_model_from_training_model``. Both are model-agnostic — they
inspect any Keras model's outputs and slice a training model down to its
primary head — and live in ``dl_techniques/utils/deep_supervision.py`` because
nothing about them is ResNet-specific, even though ``ResNet`` has an
``enable_deep_supervision`` parameter that makes this package a plausible place
to look for them.

If you need them, take them from their canonical home::

    from dl_techniques.utils.deep_supervision import (
        create_inference_model_from_training_model,
        get_model_output_info,
    )

Do not re-add a pass-through here. See ``dl_techniques/models/vision/vit/__init__.py``
for the same removal on the ViT package.
"""

from .model import ResNet, create_resnet

__all__ = [
    'ResNet',
    'create_resnet'
]
