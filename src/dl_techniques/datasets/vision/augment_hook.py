"""The train-augmentation hook of the vision loaders, and its per-element seeds.

A hook is a ``(image, label) -> (image, label)`` map that sees the ``[0, 1]`` float image
AFTER the loader's geometry (resize or crop) and BEFORE normalization. Augmenting normalized
data and then clipping to ``[0, 1]`` saturated most pixels in ``train_vit`` once (its
D-006/D-007), so every loader hands the hook ``[0, 1]`` pixels, whatever it stores them as.

A *seeded* hook (:func:`seeded`) is ``(seed, image, label) -> (image, label)``: ``seed`` is an
``int32`` pair, one per element, drawn from a stream that is fixed by the loader's seed, is
attached BEFORE the parallel map (so an element's augmentation does not depend on which worker
finishes first) and is re-randomized every epoch. Two builds of the same seed therefore see the
same augmentation, and no epoch repeats another's.

This module holds only the protocol; the operations live in ``train.common.image_augment``.
"""

from typing import Any, Callable, Optional, Tuple

import tensorflow as tf

Augment = Callable[..., Tuple[tf.Tensor, tf.Tensor]]


def seeded(hook: Callable[..., Any]) -> Callable[..., Any]:
    """Mark ``hook`` as taking ``(seed, image, label)`` instead of ``(image, label)``.

    :param hook: The map to mark.
    :return: ``hook`` itself, with ``seeded = True``.
    """
    hook.seeded = True  # type: ignore[attr-defined]
    return hook


def is_seeded(hook: Optional[Callable[..., Any]]) -> bool:
    """Whether ``hook`` was marked with :func:`seeded`."""
    return bool(getattr(hook, "seeded", False))


def seed_stream(seed: int) -> "tf.data.Dataset":
    """The endless dataset of ``int32[2]`` seeds, fixed by ``seed``, new every epoch."""
    return tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True).batch(2)


def apply_hook(
        hook: Augment, image: tf.Tensor, label: tf.Tensor, seed: Optional[tf.Tensor],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Run ``hook`` on one example; a seeded hook gets ``seed``, a plain one does not."""
    if is_seeded(hook):
        return hook(seed, image, label)
    return hook(image, label)


def with_hook(
        dataset: "tf.data.Dataset", hook: Optional[Augment], seed: int,
        num_parallel_calls: Any = tf.data.AUTOTUNE,
) -> "tf.data.Dataset":
    """Map ``hook`` over an unbatched ``(image, label)`` dataset of ``[0, 1]`` images.

    :param dataset: Unbatched ``(image, label)`` examples.
    :param hook: A plain or seeded hook, or ``None`` (the dataset is returned as it is).
    :param seed: Seed of the per-element stream (used by a seeded hook only).
    :param num_parallel_calls: Parallelism of the map.
    :return: The augmented ``(image, label)`` dataset.
    """
    if hook is None:
        return dataset
    if not is_seeded(hook):
        return dataset.map(hook, num_parallel_calls=num_parallel_calls)
    zipped = tf.data.Dataset.zip((dataset, seed_stream(seed)))
    return zipped.map(lambda item, element_seed: hook(element_seed, *item),
                      num_parallel_calls=num_parallel_calls)
