"""Residual network (ResNet) public API.

Re-exports the model class and the factory function. Internal callers may still
import from ``dl_techniques.models.resnet.model`` directly.

Deliberately NOT re-exported: ``get_model_output_info`` and
``create_inference_model_from_training_model``. ``ResNet`` genuinely does have an
``enable_deep_supervision`` parameter, so unlike ``ViT`` this package is a
plausible place to look for those helpers -- but they are model-agnostic: they
inspect any Keras model's outputs and slice a training model down to its primary
head, and they live in ``dl_techniques/utils/deep_supervision.py`` because
nothing about them is ResNet-specific. ``model.py`` never defined either one; it
imported them at the file tail purely so this file could pass them through, which
gave one function two import paths for no gain and left ``models/resnet``
advertising a surface it does not own.

If you need them, take them from their canonical home::

    from dl_techniques.utils.deep_supervision import (
        create_inference_model_from_training_model,
        get_model_output_info,
    )

Do not re-add a pass-through here. See ``dl_techniques/models/vit/__init__.py``
for the same removal on the ViT package.
"""

from .model import ResNet, create_resnet

__all__ = [
    'ResNet',
    'create_resnet'
]
