"""Shared denoiser patch augmentations.

Extracted (plan plan_2026-06-02_cc4d4e14, C2) from the bfcnn / bfunet /
cliffordnet denoiser trainers, where ``augment_patch`` was duplicated
verbatim across 6 sites and the synchronized-pair variant ``augment_pair``
across the 2 cliffordnet conditional/confidence denoisers.

``augment_patch`` applies independent random horizontal/vertical flips and a
random 90-degree rotation to a single image patch.

``augment_pair`` applies a SINGLE shared flip/rotation transform to a
(target, conditioning) pair so the two stay spatially aligned: it concatenates
them on the channel axis, augments once, then splits back. This relies on the
``target`` tensor having a STATIC last (channel) dimension, since the split
index is ``target.shape[-1]``. That precondition holds for the denoising
pipelines this is used in (fixed-channel images); do not call it on tensors
with an unknown channel dim.
"""

import tensorflow as tf
from typing import Tuple


def augment_patch(patch: tf.Tensor) -> tf.Tensor:
    """Apply random flips and 90-degree rotations."""
    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    return tf.image.rot90(patch, k)


def augment_pair(
    target: tf.Tensor, cond: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply synchronized augmentations to target/conditioning pair.

    Precondition: ``target`` must have a static last (channel) dimension,
    used as the channel-split index after the shared transform.
    """
    combined = tf.concat([target, cond], axis=-1)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k)
    t_ch = target.shape[-1]
    return combined[..., :t_ch], combined[..., t_ch:]
