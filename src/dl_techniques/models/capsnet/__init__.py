"""Capsule Networks — public API re-exports.

`model.py` is the original CapsNet (dynamic routing, optional reconstruction
decoder); `model_v2.py` is CapsNetV2, which swaps the convolutional stem for a
ResNet backbone. Neither has named scale variants — CapsNet is parameterized by
capsule counts and dimensions rather than by a tiny/small/base ladder — so this
package has no `MODEL_VARIANTS` / `from_variant` by design.
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
