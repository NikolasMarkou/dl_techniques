"""MobileCLIP2 public API — the FAITHFUL FastViT (MCi) port.

MobileCLIP2 (arXiv:2508.20691) pairs a convolutional/attention hybrid FastViT
image tower with a standard CLIP text transformer. This package is the faithful
port of that image tower, built from the ``layers/fastvit/`` primitives.

.. warning::
   This is NOT ``models/mobile_clip/``. That package is a deliberately
   non-faithful v1 that substitutes ``keras.applications`` backbones for the MCi
   tower (its own in-file decision D-001). It is shipped and tested and is left
   untouched; the two packages coexist on purpose.

Three facts that are easy to get wrong:

* **The image tower's terminal ``Dense`` IS the CLIP image projection.**
  MobileCLIP's open_clip configs use ``timm_pool="avg"`` with
  ``timm_proj=null``, so the trunk's own classifier linear projects into the
  joint embedding space. Do not stack a second projection on top of
  :class:`FastVitImageEncoder`.
* **The stochastic-depth ramp is GLOBAL**, computed once over every block of
  every stage and then sliced stagewise — not recomputed per stage.
* **``MODEL_VARIANTS`` deliberately holds two families.** The four
  ``mobileclip2_s*`` rows set ``use_causal_mask=False`` (their JSON configs say
  ``"no_causal_mask": true``); the two earlier ``mobileclip_s3``/``mobileclip_s4``
  rows are causal. Same image backbones, different text-tower attention.

The text tower is :class:`MobileClipTextEncoder`, imported from the v1 package
rather than re-implemented — see the comment at that import in :mod:`.model`.
"""

from .image_encoder import (
    MCI_VARIANTS,
    FastVitImageEncoder,
    create_fastvit_image_encoder,
)
from .model import (
    MODEL_VARIANTS,
    MobileClipV2Model,
    create_mobile_clip_v2,
)

__all__ = [
    "MCI_VARIANTS",
    "MODEL_VARIANTS",
    "FastVitImageEncoder",
    "MobileClipV2Model",
    "create_fastvit_image_encoder",
    "create_mobile_clip_v2",
]
