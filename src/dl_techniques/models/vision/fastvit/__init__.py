"""FastViT (MCi) image backbone, a channels-last port of timm's ``FastVit``.

A Keras 3 port of timm's ``FastVit`` class restricted to the five ``MCi``
configurations (``mci0``..``mci4``), assembled from the block primitives in
``layers/fastvit/``. Architecture only: no pretrained weights are ported and
this package makes no accuracy claim.

The tower works as a standalone image encoder and also as the vision branch
of MobileCLIP2 — :mod:`dl_techniques.models.vision_language.mobile_clip.mobile_clip_v2`
imports :class:`FastVitImageEncoder` from here.

Three things to know before using it:

* The head ``Dense`` is a projection, not a classifier. In the MobileCLIP
  configs (``timm_pool="avg"``, ``timm_proj=null``) the trunk's own terminal
  linear is the CLIP image projection into the joint embedding space. Do not
  stack a second projection on top of :class:`FastVitImageEncoder` when using
  it inside a CLIP model.
* The stochastic-depth ramp is global: computed once over every block of
  every stage, then sliced stagewise, not recomputed per stage. A per-stage
  ramp gives an identically-shaped, identically-parameterized, subtly wrong
  model.
* ``FastVitRepMixerBlock`` (in ``layers/fastvit/``) is not the pre-existing
  ``layers/repmixer_block.py::RepMixerBlock``. They are different
  architectures sharing a name; the latter is used by
  ``models/vision_language/fastvlm/`` and is untouched by this package.

See ``README.md`` for the variant table and the recorded deviations from the
reference implementation.
"""

from .model import (
    MCI_VARIANTS,
    FastVitImageEncoder,
    create_fastvit_image_encoder,
)

__all__ = [
    "MCI_VARIANTS",
    "FastVitImageEncoder",
    "create_fastvit_image_encoder",
]
