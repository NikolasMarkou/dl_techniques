"""Segment Anything — three generations as nested subpackages.

`sam1/`, `sam2/` and `sam3/` are each a full architecture with its own
`MODEL_VARIANTS` and `from_variant`: SAM 1 is image encoder + prompt encoder +
mask decoder, SAM 2 adds the memory bank and memory attention for video, SAM 3
adds text-conditioned open-vocabulary segmentation. Nothing is shared between
them beyond the repo's layer library, and none deprecates the others.

**This package deliberately exports nothing — import from the subpackage.**

    from dl_techniques.models.vision_language.sam.sam2.model import SAM2, create_sam2

This module is deliberately free of re-exports, like every other container under
``models/``: family and subfamily directories carry a docstring and no public
surface, so a caller imports from the leaf package. Its sibling subfamilies
``vision/image_restoration/``, ``vision/keypoints/`` and
``vision/super_resolution/`` are all shaped the same way. Re-exporting here
would buy one saved import line and cost an eager import of every package in
the subfamily; the reasoning is recorded in
``plan-2026-08-24T205033-8fd4f20d/D-002`` and summarised in
``models/CLAUDE.md``. A second, SAM-specific reason survives independently: the
three generations define same-named components (encoders, necks, decoders) that
would collide with each other in a shared namespace.

Historical note, so nobody re-derives it wrongly. Until the 2026-08-24
restructure the subpackages were spelled `SAM1/`, `SAM2/`, `SAM3/`, and the
no-re-export rule here was justified by a *name collision*: binding the model
class `SAM2` in this package's namespace shadowed the identically-spelled
subpackage `SAM2`, so `...sam.SAM2.model` failed at collection time with
`ImportError: cannot import name 'model' from 'SAM2' (unknown location)` and
broke `tests/test_models/test_sam2/`. That hazard was real, and it is now
**retired**: the subpackages are lowercase, Python is case-sensitive, and
MEASURED on 2026-08-25 the class `SAM2` and the subpackage `sam2` bind side by
side in this namespace with `...sam.sam2.model` still importing cleanly. The
collision argument no longer applies — D-002 above is what keeps this package
empty. Do not resurrect the collision rationale, and do not treat its
disappearance as licence to add re-exports.
"""
