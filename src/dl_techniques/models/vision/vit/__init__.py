"""Vision Transformer (ViT) public API.

Re-exports the model class and the factory function. Internal callers (e.g.
``depth_anything``, ``lewm``) may still import from
``dl_techniques.models.vit.model`` directly.

Deliberately NOT re-exported: ``create_inference_model_from_training_model``.
This package used to pass it through "for Resnet-template parity", but the
parity was false -- ``ViT`` has no ``enable_deep_supervision`` parameter and no
deep-supervision code path at all (``grep -c deep_supervision`` over
``vit/model.py`` returns 0), whereas ``ResNet`` genuinely has one. ``model.py``
never defined the helper either; it imported it from
``dl_techniques.utils.deep_supervision`` purely so this file could re-export it,
which is why removing that unused import broke ``import dl_techniques.models.vit``
and, through it, collection of ``tests/test_train/test_vit`` and
``tests/test_train/test_dino``. Nothing imported the helper from this package.

If you need it, take it from its canonical home:
``from dl_techniques.utils.deep_supervision import create_inference_model_from_training_model``
(``dl_techniques/utils/deep_supervision.py``). Do not re-add a pass-through here
unless ViT actually grows deep-supervision outputs.
"""

from .model import (
    ViT,
    create_vit,
)

__all__ = [
    "ViT",
    "create_vit",
]
