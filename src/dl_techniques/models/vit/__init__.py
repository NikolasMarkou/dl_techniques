"""Vision Transformer (ViT) public API.

Resnet-template parity: re-exports the model class, the factory function,
and the deep-supervision inference-model helper. Internal callers (e.g.
``depth_anything``, ``lewm``) may still import from
``dl_techniques.models.vit.model`` directly.
"""

from .model import (
    ViT,
    create_vit,
    create_inference_model_from_training_model,
)

__all__ = [
    "ViT",
    "create_vit",
    "create_inference_model_from_training_model",
]
