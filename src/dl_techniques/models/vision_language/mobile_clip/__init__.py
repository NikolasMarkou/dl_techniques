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

Both models share one shape: a nested ``MODEL_VARIANTS`` **class attribute**
whose rows are ``{'embed_dim': int, 'image_config': {...}, 'text_config': {...}}``,
a constructor taking those two sub-dicts, ``output_dict``, and ``from_variant``.
Reach either table the same way — ``MobileClipModel.MODEL_VARIANTS`` /
``MobileClipV2Model.MODEL_VARIANTS``. Neither is re-exported at package level,
because the two are different objects with disjoint keys (``b``/``s0``/``s1``/
``s2`` versus ``mobileclip2_s0``/``s2``/``s3``/``s4`` and ``mobileclip_s3``/``s4``).

.. warning::
   **Three names mean different things depending on where you read them.**

   * ``mci0``/``mci1``/``mci2`` name real MCi rows in v2
     (``models.fastvit.MCI_VARIANTS``); in v1 they are keys of
     ``components._BACKBONE_ALIASES`` resolving to ``keras.applications``
     MobileNet stand-ins. Same string, opposite meaning.
   * ``text_config['embed_dim']`` is the TEXT WIDTH. The joint image-text space
     is the row's own top-level ``embed_dim``.
   * v2's ``image_config['variant']`` (``'mci0'``) is FastViT's kwarg name and is
     a DIFFERENT "variant" from the row key (``'mobileclip2_s0'``).

.. note::
   ``from_variant`` overrides at the TOP level, so passing ``text_config=``
   replaces the row's sub-dict wholesale. To change one field, merge::

       row = MobileClipV2Model.MODEL_VARIANTS['mobileclip2_s0']
       model = MobileClipV2Model.from_variant(
           'mobileclip2_s0',
           text_config={**row['text_config'], 'num_layers': 2},
       )
"""

from .mobile_clip_v1 import MobileClipModel, create_mobile_clip_model
from .mobile_clip_v2 import MobileClipV2Model, create_mobile_clip_v2

__all__ = [
    "MobileClipModel",
    "MobileClipV2Model",
    "create_mobile_clip_model",
    "create_mobile_clip_v2",
]
