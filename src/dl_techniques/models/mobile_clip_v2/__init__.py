"""MobileCLIP2 public API — the FAITHFUL FastViT (MCi) port.

MobileCLIP2 (arXiv:2508.20691) pairs a convolutional/attention hybrid FastViT
image tower with a standard CLIP text transformer. This package is the faithful
port of that image tower, built from the ``layers/fastvit/`` primitives.

.. warning::
   This is NOT ``models/mobile_clip/``. That package is a deliberately
   non-faithful v1 that substitutes ``keras.applications`` backbones for the MCi
   tower (its own in-file decision D-001). It is shipped and tested and is left
   untouched; the two packages coexist on purpose.

Two facts about the image tower that are easy to get wrong:

* **The head ``Dense`` IS the CLIP image projection.** MobileCLIP's open_clip
  configs use ``timm_pool="avg"`` with ``timm_proj=null``, so the trunk's own
  classifier linear projects into the joint embedding space. Do not stack a
  second projection on top of :class:`FastVitImageEncoder`.
* **The stochastic-depth ramp is GLOBAL**, computed once over every block of
  every stage and then sliced stagewise — not recomputed per stage.

This ``__init__`` currently exports the image tower only; the dual-encoder model
is added in the next step.
"""

from .image_encoder import (
    MCI_VARIANTS,
    FastVitImageEncoder,
    create_fastvit_image_encoder,
)

__all__ = [
    "MCI_VARIANTS",
    "FastVitImageEncoder",
    "create_fastvit_image_encoder",
]
