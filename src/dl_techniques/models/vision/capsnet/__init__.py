"""Capsule Networks — public API re-exports.

`model.py` holds the original CapsNet (dynamic routing, optional reconstruction
decoder). `model_v2.py` holds CapsNetV2, which routes once through an attention block and
takes either the original paper's two-convolution trunk (`stem_type="legacy"`, held as
`conv_blocks`) or a ResNet backbone (`stem_type="resnet18"` and the other ResNet names,
held as `stem`). Neither has named scale variants: capsule counts and dimensions
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
