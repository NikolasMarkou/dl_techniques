"""MobileCLIP — both generations of Apple's on-device dual encoder.

This package ships TWO models that share a text tower and differ on the image
side. They are not interchangeable and neither deprecates the other:

* :class:`MobileClipModel` (:mod:`.mobile_clip_v1`) — MobileCLIP. **Deliberately
  non-faithful on the image branch**: the real MCi backbones do not exist in
  ``keras.applications``, so :mod:`.components` substitutes MobileNetV2 /
  MobileNetV3 under its own in-file decision D-001. Shipped, tested, and left
  that way on purpose.
* :class:`MobileClipV2Model` (:mod:`.mobile_clip_v2`) — MobileCLIP2
  (arXiv:2508.20691). **The faithful port**: a real FastViT MCi tower from
  :mod:`dl_techniques.models.fastvit`, transcribed from timm. Architecture only
  — no pretrained weights are ported and it makes no accuracy claim. See
  ``README.md`` §15 before quoting it against any published number.

Both use :class:`~.components.MobileClipTextEncoder`, which is faithful for
both: v1's substitution is confined to the image branch.

.. warning::
   **``MODEL_VARIANTS`` is deliberately NOT re-exported here — the two tables
   are different objects with disjoint keys and incompatible meanings.**

   * v1's is the class attribute ``MobileClipModel.MODEL_VARIANTS``, keyed
     ``b``/``s0``/``s1``/``s2``.
   * v2's is the module-level ``mobile_clip_v2.MODEL_VARIANTS``, keyed
     ``mobileclip2_s0``/``s2``/``s3``/``s4`` and ``mobileclip_s3``/``s4``.

   The same trap applies to the backbone strings ``mci0``/``mci1``/``mci2``:
   in v2 they name real MCi rows (``models.fastvit.MCI_VARIANTS``); in v1 they
   are keys of ``components._BACKBONE_ALIASES`` that resolve to
   ``keras.applications`` MobileNet stand-ins. Same string, opposite meaning.

Import each table from its own module, explicitly.
"""

from .mobile_clip_v1 import MobileClipModel, create_mobile_clip_model
from .mobile_clip_v2 import MobileClipV2Model, create_mobile_clip_v2

__all__ = [
    "MobileClipModel",
    "MobileClipV2Model",
    "create_mobile_clip_model",
    "create_mobile_clip_v2",
]
