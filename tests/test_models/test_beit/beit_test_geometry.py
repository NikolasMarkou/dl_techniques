"""The one small-geometry test fixture every ``test_beit`` behaviour file builds on.

Extracted from ``test_model.py`` when that 1173-line module was decomposed into one
behaviour-named file per concern (plan-2026-08-24T074054-247151fd, step 8). The
constants and factories below lived at the top of ``test_model.py`` and were shared by
all ten test classes in it; once those classes moved to seven files, keeping them
inline would have meant copying the same geometry into seven places, where the copies
drift silently -- a test suite whose files disagree about what ``tiny`` means measures
nothing in particular.

The geometry is deliberately the smallest one that still exercises every code path:
a 32x32 image at ``patch_size=16`` gives a 2x2 patch grid, 4 patches, 5 tokens with
the class token. Anything smaller stops being a grid.
"""

# DECISION plan-2026-08-24T074054-247151fd/D-013
# WHAT NOT TO DO: do not "follow the resnet convention" by inlining these constants and
# factories back into each of the seven test files. tests/test_models/test_resnet/ does
# keep every file self-contained, and that reads like the rule to copy -- but resnet's
# files each pin a DIFFERENT geometry (test_deep_supervision.py:38-40 FILTERS/INPUT_SHAPE
# vs test_basic_blocks_work_at_any_stage0_width.py's width sweep), so there is nothing
# there to share. Here all seven files must agree that `tiny` means a 32x32 image at
# patch 16 -> a 2x2 grid -> 5 tokens, because SEQ_LEN and NUM_PATCHES are asserted
# against each other across file boundaries (test_model.py's build assertions vs
# test_masked_image_modeling_head.py's token-slice orientation test). Seven copies of a
# cross-checked invariant is the duplication smell the repo's own de-dup lesson names.
# The precedent for a plain sibling helper module is tests/test_models/test_fastvit/
# reference_oracle.py and tests/test_models/test_sam/dead_component_oracle.py.
# See decisions.md D-013.

import numpy as np

from dl_techniques.models.vision.beit import (
    BeitForMaskedImageModeling,
    BeitModel,
    create_beit_classifier,
    create_beit_mim,
)

# A 32x32 image at patch 16 -> a 2x2 patch grid -> 4 patches + 1 cls = 5 tokens.
IMG = (32, 32, 3)
PATCH = 16
GRID = (2, 2)
NUM_PATCHES = 4
SEQ_LEN = NUM_PATCHES + 1
EPS = 1e-12

VOCAB = 64  # a toy codebook; the real default is DEFAULT_VOCAB_SIZE == 8192


def _tiny(**overrides) -> BeitModel:
    """A `tiny` backbone at the small test geometry.

    :param overrides: any :class:`BeitModel` keyword, overriding the pinned
        ``input_shape=IMG``, ``patch_size=PATCH``, ``scale='tiny'`` defaults.
    :returns: an UNBUILT :class:`BeitModel`.
    """
    config = dict(input_shape=IMG, patch_size=PATCH, scale='tiny')
    config.update(overrides)
    return BeitModel(**config)


def _images(batch: int = 2, seed: int = 0) -> np.ndarray:
    """Deterministic standard-normal images at the pinned geometry.

    :param batch: leading batch dimension.
    :param seed: seed for :func:`numpy.random.default_rng`; the same seed always
        returns the same array.
    :returns: ``float32`` array of shape ``(batch,) + IMG``.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch,) + IMG).astype('float32')


def _mask(batch: int = 2, num_patches: int = NUM_PATCHES) -> np.ndarray:
    """A boolean MIM mask that is neither all-True nor all-False, and differs per row.

    :param batch: leading batch dimension.
    :param num_patches: patch count; defaults to the pinned :data:`NUM_PATCHES`.
    :returns: ``bool`` array of shape ``(batch, num_patches)``; patch 0 is masked in
        every row and the last patch is masked in row 0 only.
    """
    m = np.zeros((batch, num_patches), dtype=bool)
    m[:, 0] = True
    m[0, -1] = True
    return m


def _mim(variant: str = 'tiny', **overrides) -> BeitForMaskedImageModeling:
    """A masked-image-modeling head at the small geometry over a toy codebook.

    :param variant: a :data:`MODEL_VARIANTS` key.
    :param overrides: any :func:`create_beit_mim` keyword; ``vocab_size`` defaults to
        the toy :data:`VOCAB`, not the 8192-entry DALL-E codebook.
    :returns: an UNBUILT :class:`BeitForMaskedImageModeling`.
    """
    return create_beit_mim(variant, IMG, PATCH, vocab_size=VOCAB, **overrides)


def _classifier(variant: str = 'tiny', num_classes: int = 7, **overrides):
    """A classification head at the small geometry.

    :param variant: a :data:`MODEL_VARIANTS` key.
    :param num_classes: head width; 7 is an arbitrary non-round number chosen so a
        shape assertion cannot pass by coincidence with a batch or token count.
    :param overrides: any :func:`create_beit_classifier` keyword.
    :returns: an UNBUILT :class:`BeitForImageClassification`.
    """
    return create_beit_classifier(
        variant, IMG, PATCH, num_classes=num_classes, **overrides
    )
