"""Segment Anything — three generations as nested subpackages.

`SAM1/`, `SAM2/` and `SAM3/` are each a full architecture with its own
`MODEL_VARIANTS` and `from_variant`: SAM 1 is image encoder + prompt encoder +
mask decoder, SAM 2 adds the memory bank and memory attention for video, SAM 3
adds text-conditioned open-vocabulary segmentation. Nothing is shared between
them beyond the repo's layer library, and none deprecates the others.

**This package deliberately exports nothing — import from the subpackage.**

    from dl_techniques.models.SAM.SAM2.model import SAM2, create_sam2

The house convention is a curated `__all__` (see `models/CLAUDE.md` § House
Model Module Shape, Axis 4), and this package is an explicit, measured exception.
SAM 2's model CLASS is named `SAM2` and its SUBPACKAGE is also named `SAM2`, so
re-exporting the class here binds the name `SAM2` in this package's namespace and
SHADOWS the subpackage. `dl_techniques.models.SAM.SAM2.model` then fails with

    ImportError: cannot import name 'model' from 'SAM2' (unknown location)

which is what happened when the re-exports were first added, and it broke
`tests/test_models/test_sam2/` at collection time. Aliasing the class to dodge
the collision would give the package two names for one model, which is worse
than importing from the submodule. The three generations also define same-named
components (encoders, necks, decoders) that would collide with each other here.
"""
