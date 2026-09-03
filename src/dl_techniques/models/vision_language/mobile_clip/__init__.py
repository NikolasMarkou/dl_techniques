"""MobileCLIP: two generations of Apple's on-device dual encoder.

The package ships two models that share a text tower and differ on the
image side. `MobileClipModel` (`.mobile_clip_v1`) substitutes MobileNetV2 or
MobileNetV3 for the image branch, since the real MCi backbones have no
`keras.applications` equivalent. `MobileClipV2Model` (`.mobile_clip_v2`,
MobileCLIP2) is the faithful port: a real FastViT MCi tower, transcribed
from timm. Neither model carries pretrained weights, so neither makes an
accuracy claim; see `README.md` before quoting either against a published
number.

Both models keep their own `MODEL_VARIANTS` table (`MobileClipModel.MODEL_VARIANTS`,
`MobileClipV2Model.MODEL_VARIANTS`) with disjoint keys, and neither is
re-exported here. The two use the same variant-string ``mci0``/``mci1``/``mci2``
for different things: real FastViT MCi rows in v2, MobileNet stand-in keys in
v1. `from_variant`'s `text_config=` and `image_config=` overrides replace a
row's sub-dict wholesale rather than merging into it, so changing one field
means merging the preset dict yourself first.
"""

from .mobile_clip_v1 import MobileClipModel, create_mobile_clip_model
from .mobile_clip_v2 import MobileClipV2Model, create_mobile_clip_v2

__all__ = [
    "MobileClipModel",
    "MobileClipV2Model",
    "create_mobile_clip_model",
    "create_mobile_clip_v2",
]
