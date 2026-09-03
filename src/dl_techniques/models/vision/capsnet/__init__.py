"""Capsule Networks — public API re-exports.

`model.py` holds the original CapsNet (dynamic routing, optional reconstruction
decoder). `model_v2.py` holds CapsNetV2, which swaps the convolutional stem for a
ResNet backbone. Neither has named scale variants: capsule counts and dimensions
are set directly rather than through a tiny/small/base ladder, so this package
has no `MODEL_VARIANTS` or `from_variant`.
"""
from .model import CapsNet, create_capsnet
from .model_v2 import CapsNetV2, create_capsnet_v2, create_capsnet_v2_pretrained

__all__ = [
    "CapsNet",
    "create_capsnet",
    "CapsNetV2",
    "create_capsnet_v2",
    "create_capsnet_v2_pretrained",
]
