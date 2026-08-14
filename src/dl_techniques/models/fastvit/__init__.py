"""FastViT (MCi) image backbone — the faithful timm ``FastVit`` transcription.

A channels-last Keras 3 port of timm's ``FastVit`` class restricted to the five
``MCi`` configurations (``mci0``..``mci4``), assembled from the block primitives
in ``layers/fastvit/``. Architecture only: no pretrained weights are ported and
this package makes no accuracy claim.

The tower stands alone as an image encoder, and is *also* the vision branch of
MobileCLIP2 — :mod:`dl_techniques.models.mobile_clip.mobile_clip_v2` imports
:class:`FastVitImageEncoder` from here.

Three facts that are easy to get wrong:

* **The head ``Dense`` is a projection, not a classifier.** In the MobileCLIP
  configs (``timm_pool="avg"``, ``timm_proj=null``) the trunk's own terminal
  linear *is* the CLIP image projection into the joint embedding space. Do not
  stack a second projection on top of :class:`FastVitImageEncoder` when using it
  inside a CLIP model.
* **The stochastic-depth ramp is GLOBAL**, computed once over every block of
  every stage and then sliced stagewise — not recomputed per stage. A per-stage
  ramp yields an identically-shaped, identically-parameterized, subtly-wrong
  model.
* **``FastVitRepMixerBlock`` (in ``layers/fastvit/``) is NOT the pre-existing
  ``layers/repmixer_block.py::RepMixerBlock``.** They are different
  architectures sharing a name; the latter is consumed by ``models/fastvlm/``
  and is untouched by this package.

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
