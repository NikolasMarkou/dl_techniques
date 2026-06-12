# DECISION plan_2026-06-11_f662207d/D-011
"""Pure ``tf.data`` arbitrary-scale super-resolution pipeline (THERA port).

This module ports THERA's PyTorch ``ArbitraryScaleWrapper`` (reference
``data.py``) to a pure ``tf.data`` graph pipeline. There is **no** PyTorch and
**no** PIL dependency: image decode, bicubic/nearest resampling, random
cropping, flips and rotations are all expressed with ``tf.image`` / ``tf.*``
ops so the per-sample map runs inside the dataset graph.

Per-sample logic (mirrors the reference, ``source_size`` = LR side = 48,
``target_samples`` = query points per side = 48):

#. Decode an image to ``float32`` ``[0, 1]``, 3-channel.
#. Draw ``scale ~ U(scale_range)`` and, with probability ``augment_scale_prob``,
   ``augment_scale ~ U(augment_scale_range)`` (else ``1.0``).
#. Random-crop a ``crop x crop`` window, ``crop = round(source_size * scale *
   augment_scale)``. If the image is smaller than ``crop`` on either side it is
   first resized up (aspect-preserving, short side -> ``crop``) so the crop is
   always valid (robustness guard for tiny corpora).
#. ``target_size = round(crop / augment_scale)``; if ``augment_scale != 1`` the
   crop is bicubic-resized to ``target_size`` to form the HR ``target``.
#. Random h/v flip + a random ``k * 90`` rotation.
#. ``source`` = bicubic downscale of ``target`` to ``source_size`` (the LR
   input, fixed ``48 x 48``). ``source_nearest`` = nearest upsample of
   ``source`` back to ``target_size`` (the residual base).
#. Build a pixel-center coordinate grid for ``target_size`` (matching the
   step-2 ``make_grid`` convention exactly), pick ``target_samples ** 2`` random
   query points, and gather ``target_coords`` / ``target`` pixels /
   ``source_nearest`` pixels at those points, reshaped to
   ``(target_samples, target_samples, C)``.

The dataset returns RAW ``[0, 1]`` images. Per-image standardization and the
``+ source_nearest`` residual add are the trainer's responsibility (step 11);
doing them here would double-apply.

# DECISION plan_2026-06-11_f662207d/D-011 (see decisions.md):
- Do NOT reach for PyTorch / PIL — TF 2.18 ``tf.image`` covers decode + bicubic
  (``method='bicubic'``, ``antialias=True`` for the downscale, E6 resolved) +
  nearest resampling, all graph-safe.
- Do NOT import the numpy ``make_grid`` from ``layers/grid_sample.py`` here: it
  needs a STATIC int side, but ``target_size`` is a dynamic tensor inside the
  map. ``_make_grid_tf`` rebuilds the identical pixel-center convention
  (``linspace(-0.5 + 1/(2n), 0.5 - 1/(2n), n)``, ``meshgrid(indexing='ij')``,
  channel order ``[h, w]``) with ``tf.linspace`` so it matches step-2 exactly.
- Do NOT standardize images here (trainer owns the data statistics).
"""

from typing import List, Optional, Sequence, Tuple, Union

import tensorflow as tf

from dl_techniques.utils.logger import logger

# Reuse the recursive image-discovery helper instead of re-implementing rglob.
from train.common.datasets import collect_image_paths

# ---------------------------------------------------------------------


def _make_grid_tf(n: tf.Tensor) -> tf.Tensor:
    """Build a pixel-center normalized coordinate grid for a dynamic side ``n``.

    Reproduces the step-2 ``make_grid`` convention EXACTLY but for a dynamic
    (tensor) side length, so it can run inside the ``tf.data`` map graph: for an
    axis of length ``n`` the sample positions are the centers of ``n`` equal
    cells tiling ``[-0.5, 0.5]``, i.e. ``linspace(-0.5 + 1/(2n), 0.5 - 1/(2n),
    n)``. Output last axis holds ``[h_coord, w_coord]`` (``meshgrid`` with
    ``indexing='ij'``).

    Args:
        n: Scalar int32 tensor, the grid side length.

    Returns:
        A ``(n, n, 2)`` ``float32`` tensor of pixel-center coordinates.
    """
    n_f = tf.cast(n, tf.float32)
    offset = 1.0 / (2.0 * n_f)
    space = tf.linspace(-0.5 + offset, 0.5 - offset, n)  # (n,) float32
    grid_h, grid_w = tf.meshgrid(space, space, indexing="ij")  # (n, n) each
    return tf.stack([grid_h, grid_w], axis=-1)  # (n, n, 2), order [h, w]


# ---------------------------------------------------------------------


def _decode_image(path: tf.Tensor) -> tf.Tensor:
    """Read and decode an image file to ``float32`` ``[0, 1]``, 3-channel.

    Args:
        path: Scalar string tensor, the image file path.

    Returns:
        A ``(H, W, 3)`` ``float32`` tensor with values in ``[0, 1]``.
    """
    raw = tf.io.read_file(path)
    # decode_image handles JPEG/PNG/BMP/GIF; expand_animations=False keeps a
    # static 3-rank shape (GIFs would otherwise add a frame axis). channels=3
    # forces RGB and a static channel dim required for downstream ops.
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # uint8 -> [0, 1]
    img.set_shape([None, None, 3])
    return img


def _ensure_min_size(img: tf.Tensor, crop: tf.Tensor) -> tf.Tensor:
    """Resize ``img`` up if smaller than ``crop`` on either side (aspect-safe).

    Guarantees a subsequent ``tf.image.random_crop(img, [crop, crop, 3])`` is
    valid even for tiny corpus images: if the shorter side is below ``crop`` the
    whole image is bicubic-resized so its SHORT side equals ``crop`` (preserving
    aspect ratio), leaving the long side ``>= crop``.

    Args:
        img: ``(H, W, 3)`` ``float32`` image.
        crop: Scalar int32 target crop side.

    Returns:
        A ``(H', W', 3)`` ``float32`` image with ``min(H', W') >= crop``.
    """
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    short = tf.minimum(h, w)

    def _resize_up() -> tf.Tensor:
        scale = tf.cast(crop, tf.float32) / tf.cast(short, tf.float32)
        new_h = tf.cast(tf.math.ceil(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.math.ceil(tf.cast(w, tf.float32) * scale), tf.int32)
        # Guard against rounding leaving a side one pixel short of `crop`.
        new_h = tf.maximum(new_h, crop)
        new_w = tf.maximum(new_w, crop)
        out = tf.image.resize(img, (new_h, new_w), method="bicubic")
        return tf.clip_by_value(out, 0.0, 1.0)

    return tf.cond(short < crop, _resize_up, lambda: img)


def _per_sample(
    path: tf.Tensor,
    *,
    source_size: int,
    target_samples: int,
    scale_range: Tuple[float, float],
    augment_scale_range: Tuple[float, float],
    augment_scale_prob: float,
) -> dict:
    """Build one arbitrary-scale training example from an image path.

    See the module docstring for the full per-sample recipe. All ops are
    graph-safe (run inside the ``tf.data`` map).

    Args:
        path: Scalar string tensor, the image file path.
        source_size: LR input side (fixed output ``source`` side), e.g. 48.
        target_samples: Number of query points per side; outputs are
            ``(target_samples, target_samples, C)``.
        scale_range: ``(min, max)`` for the base SR scale.
        augment_scale_range: ``(min, max)`` for the extra scale augmentation.
        augment_scale_prob: Probability of applying scale augmentation.

    Returns:
        Dict with keys ``source``, ``target_coords``, ``target``,
        ``source_nearest``, ``scale`` (static shapes set via ``ensure_shape``).
    """
    img = _decode_image(path)

    # --- draw scales ---
    scale = tf.random.uniform([], scale_range[0], scale_range[1])
    do_aug = tf.random.uniform([]) < augment_scale_prob
    augment_scale = tf.where(
        do_aug,
        tf.random.uniform([], augment_scale_range[0], augment_scale_range[1]),
        tf.constant(1.0),
    )

    # --- crop window ---
    crop = tf.cast(
        tf.round(tf.cast(source_size, tf.float32) * scale * augment_scale),
        tf.int32,
    )
    img = _ensure_min_size(img, crop)
    target = tf.image.random_crop(img, [crop, crop, 3])  # (crop, crop, 3)

    # --- target size (undo the augment scale) ---
    target_size = tf.cast(
        tf.round(tf.cast(crop, tf.float32) / augment_scale), tf.int32
    )
    # When augment_scale != 1, bicubic-resize the crop to target_size; else the
    # crop already IS target_size (target_size == crop). tf.cond keeps this in
    # graph for the dynamic branch.
    target = tf.cond(
        do_aug,
        lambda: tf.clip_by_value(
            tf.image.resize(target, (target_size, target_size), method="bicubic"),
            0.0,
            1.0,
        ),
        lambda: target,
    )

    # --- augmentations: random h/v flip + k*90 rotation ---
    target = tf.image.random_flip_left_right(target)
    target = tf.image.random_flip_up_down(target)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    target = tf.image.rot90(target, k=k)

    # --- LR source (bicubic downscale, antialias for clean decimation: E6) ---
    source = tf.image.resize(
        target, (source_size, source_size), method="bicubic", antialias=True
    )
    source = tf.clip_by_value(source, 0.0, 1.0)  # bicubic can overshoot

    # --- nearest upsample of source back to target_size (residual base) ---
    source_up = tf.image.resize(
        source, (target_size, target_size), method="nearest"
    )

    # --- pixel-center query grid (dynamic side) ---
    coords = _make_grid_tf(target_size)  # (target_size, target_size, 2)

    # --- flatten + sample target_samples^2 random query points ---
    n = target_size * target_size
    n_samples = target_samples * target_samples
    # Guard: with source_size=48 and scale_range min ~1.2, target_size >= ~57,
    # so n >= 57^2 > 48^2. Assert keeps the invariant explicit / fail-loud.
    tf.debugging.assert_greater_equal(
        n,
        tf.constant(n_samples, tf.int32),
        message="target_size^2 < target_samples^2: scale_range too small",
    )

    coords_flat = tf.reshape(coords, (-1, 2))  # (n, 2)
    target_flat = tf.reshape(target, (-1, 3))  # (n, 3)
    source_up_flat = tf.reshape(source_up, (-1, 3))  # (n, 3)

    idc = tf.random.shuffle(tf.range(n))[:n_samples]  # (n_samples,)

    coords_s = tf.gather(coords_flat, idc)  # (n_samples, 2)
    target_s = tf.gather(target_flat, idc)  # (n_samples, 3)
    source_up_s = tf.gather(source_up_flat, idc)  # (n_samples, 3)

    # --- reshape to (target_samples, target_samples, C) ---
    target_coords = tf.reshape(coords_s, (target_samples, target_samples, 2))
    target_out = tf.reshape(target_s, (target_samples, target_samples, 3))
    source_nearest = tf.reshape(source_up_s, (target_samples, target_samples, 3))

    effective_scale = tf.cast(target_size, tf.float32) / float(source_size)

    # --- pin static shapes so .batch() can build a known signature ---
    source = tf.ensure_shape(source, [source_size, source_size, 3])
    target_coords = tf.ensure_shape(
        target_coords, [target_samples, target_samples, 2]
    )
    target_out = tf.ensure_shape(
        target_out, [target_samples, target_samples, 3]
    )
    source_nearest = tf.ensure_shape(
        source_nearest, [target_samples, target_samples, 3]
    )
    effective_scale = tf.ensure_shape(effective_scale, [])

    return {
        "source": source,
        "target_coords": target_coords,
        "target": target_out,
        "source_nearest": source_nearest,
        "scale": effective_scale,
    }


# ---------------------------------------------------------------------


def build_arbitrary_scale_dataset(
    image_dir_or_paths: Union[str, Sequence[str]],
    *,
    source_size: int = 48,
    target_samples: int = 48,
    scale_range: Tuple[float, float] = (1.2, 4.0),
    augment_scale_range: Tuple[float, float] = (1.0, 2.0),
    augment_scale_prob: float = 0.5,
    batch_size: int = 16,
    shuffle: bool = True,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    seed: Optional[int] = None,
    repeat: bool = True,
) -> tf.data.Dataset:
    """Build the THERA arbitrary-scale SR ``tf.data`` pipeline.

    Each element is a dict of:

    - ``source``: ``(B, source_size, source_size, 3)`` raw ``[0, 1]`` LR image.
    - ``target_coords``: ``(B, target_samples, target_samples, 2)`` query coords
      in ``[-0.5, 0.5]`` (pixel-center convention, channel order ``[h, w]``).
    - ``target``: ``(B, target_samples, target_samples, 3)`` HR pixel values at
      those coords (raw ``[0, 1]``).
    - ``source_nearest``: ``(B, target_samples, target_samples, 3)`` nearest
      upsample of ``source`` sampled at those coords (the residual base).
    - ``scale``: ``(B,)`` effective scale (``target_size / source_size``).

    Images are returned RAW (no standardization); the trainer standardizes
    ``source`` and adds ``source_nearest`` to the predicted residual.

    Args:
        image_dir_or_paths: A directory to scan recursively for images, a list
            of directories, or an explicit list of image file paths.
        source_size: LR input side and fixed ``source`` output side.
        target_samples: Query points per side (output spatial size).
        scale_range: ``(min, max)`` base SR scale (uniform).
        augment_scale_range: ``(min, max)`` extra scale augmentation (uniform).
        augment_scale_prob: Probability of applying scale augmentation.
        batch_size: Per-batch element count (``drop_remainder=True``).
        shuffle: Shuffle the file list and a per-element shuffle buffer.
        num_parallel_calls: Parallelism for the per-sample map.
        seed: Optional seed for file-list and buffer shuffles (reproducibility).
        repeat: Repeat the dataset indefinitely (for ``model.fit`` step loops).

    Returns:
        A batched, prefetched ``tf.data.Dataset``.

    Raises:
        ValueError: If no image files are discovered.
    """
    paths = _resolve_paths(image_dir_or_paths)
    if not paths:
        raise ValueError(
            f"No image files found for input: {image_dir_or_paths!r}"
        )
    logger.info(
        f"build_arbitrary_scale_dataset: {len(paths)} images, "
        f"source_size={source_size}, target_samples={target_samples}, "
        f"scale_range={scale_range}, batch_size={batch_size}, repeat={repeat}"
    )

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def _map_fn(path: tf.Tensor) -> dict:
        return _per_sample(
            path,
            source_size=source_size,
            target_samples=target_samples,
            scale_range=scale_range,
            augment_scale_range=augment_scale_range,
            augment_scale_prob=augment_scale_prob,
        )

    if shuffle:
        # Per-element buffer shuffle on top of the file-list shuffle.
        ds = ds.shuffle(
            buffer_size=max(len(paths), 8 * batch_size),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(_map_fn, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _resolve_paths(
    image_dir_or_paths: Union[str, Sequence[str]],
) -> List[str]:
    """Normalize the input into a flat list of image file paths.

    Args:
        image_dir_or_paths: A directory string, a list of directory strings, or
            an explicit list of image file paths.

    Returns:
        A list of image file paths. Directories are scanned recursively via
        :func:`train.common.datasets.collect_image_paths`.
    """
    if isinstance(image_dir_or_paths, str):
        return collect_image_paths([image_dir_or_paths])

    items = list(image_dir_or_paths)
    if not items:
        return []
    # Heuristic: if every entry is an existing directory, scan recursively;
    # otherwise treat the entries as explicit file paths.
    if all(tf.io.gfile.isdir(p) for p in items):
        return collect_image_paths(items)
    return items
