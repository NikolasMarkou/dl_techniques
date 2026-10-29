"""Shared denoiser patch augmentations.

Extracted (plan plan_2026-06-02_cc4d4e14, C2) from the bfcnn / bfunet /
cliffordnet denoiser trainers, where ``augment_patch`` was duplicated
verbatim across 6 sites. The synchronized-pair variant ``augment_pair`` served
the cliffordnet conditional/confidence denoisers, which were deleted, and went
with them.

``augment_patch`` applies independent random horizontal/vertical flips and a
random 90-degree rotation to a single image patch. It is
``train.common.image_augment.random_dihedral`` with its seed drawn from the global TF
generator (the denoiser pipelines carry no per-element seed); the THERA sample builder uses
it too.
"""

import tensorflow as tf

from train.common.image_augment import random_dihedral, unseeded

_dihedral = unseeded(random_dihedral())


def augment_patch(patch: tf.Tensor) -> tf.Tensor:
    """Apply random flips and 90-degree rotations."""
    return _dihedral(patch)
