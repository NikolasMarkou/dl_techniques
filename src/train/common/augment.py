"""Shared denoiser patch augmentations.

Extracted (plan plan_2026-06-02_cc4d4e14, C2) from the bfcnn / bfunet /
cliffordnet denoiser trainers, where ``augment_patch`` was duplicated
verbatim across 6 sites. The synchronized-pair variant ``augment_pair`` served
the cliffordnet conditional/confidence denoisers, which were deleted, and went
with them.

``augment_patch`` applies independent random horizontal/vertical flips and a
random 90-degree rotation to a single image patch.
"""

import tensorflow as tf


def augment_patch(patch: tf.Tensor) -> tf.Tensor:
    """Apply random flips and 90-degree rotations."""
    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    return tf.image.rot90(patch, k)
