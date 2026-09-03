"""Vision Transformer (ViT) public API.

Re-exports the model class and the factory function. Internal callers such as
``depth_anything`` and ``lewm`` can still import from
``dl_techniques.models.vision.vit.model`` directly.

This package does not re-export ``create_inference_model_from_training_model``.
ViT has no deep-supervision code path, so the helper does not apply here. Import
it from ``dl_techniques.utils.deep_supervision`` if a future ViT variant needs it.
"""

from .model import (
    ViT,
    create_vit,
)

__all__ = [
    "ViT",
    "create_vit",
]
