"""Segment Anything, three generations as nested subpackages.

``sam1/``, ``sam2/`` and ``sam3/`` are each a full architecture with its own
``MODEL_VARIANTS`` and ``from_variant``: SAM 1 is an image encoder, prompt
encoder and mask decoder; SAM 2 adds a memory bank and memory attention for
video; SAM 3 adds text-conditioned open-vocabulary segmentation. Nothing is
shared between them beyond the repo's layer library, and none deprecates the
others.

This package exports nothing; import from the subpackage directly::

    from dl_techniques.models.vision_language.sam.sam2.model import SAM2, create_sam2

Every family and subfamily container under ``models/`` carries a docstring
and no public surface, so a caller always imports from the leaf package
(see ``models/CLAUDE.md``). The three SAM generations also define
same-named components (encoders, necks, decoders) that would collide with
each other in a shared namespace.
"""
