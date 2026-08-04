"""
Per-instance data sources for the SAM trainer, and the shared ``tf.data``
assembly that turns them into what ``SAMTrainingModel`` consumes.

Two sources, one interface
--------------------------
* :func:`synthetic_instance_samples` -- drawn shapes, no COCO on disk, no
  ``pycocotools``. This exists so the end-to-end training path is provable
  anywhere, and so a COCO I/O problem can never masquerade as a model problem
  (the CC3M HDD wall is this repository's precedent: 4 s/step, GPU-starved,
  diagnosed as a model issue for a while).
* the COCO 2017 instance source (added beside this one) -- real
  ``annToMask`` instance masks.

Both emit the same record:

    ``image``   ``(H, W, 3)`` float32 in ``[0, 255]`` (``SAM.preprocess``'s
                own input domain -- it normalizes, so do NOT pre-scale to [0,1])
    ``mask``    ``(mh, mw)`` float32 binary, at ``low_res_logits`` resolution
    ``box``     ``(4,)`` float32 xyxy in the PADDED IMAGE frame

and :func:`build_sam_dataset` maps them to the wrapper's input dict plus the
matching dict ``y_true``.

Why not reuse ``dl_techniques.datasets.synthetic_shapes``
---------------------------------------------------------
It was evaluated first and does not fit, for a reason that is structural rather
than cosmetic: its ``draw_*`` generators return ``(image, keypoints)`` with all
primitives composited onto ONE single-channel canvas and no per-object record,
so an instance mask cannot be recovered from their output. Extending them would
mean changing their return contract, which SuperPoint's stage-1 pipeline
consumes. What IS reused is the actual primitive underneath both -- OpenCV's
``fillPoly`` / ``ellipse`` -- and the convention that a generator takes an
explicit ``np.random.Generator`` so a caller can seed it.

``tf.data`` code here is exempt from the ``keras.ops``-only invariant (I-1):
this is a data pipeline, not a forward path.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from dl_techniques.models.sam.training_model import (
    INPUT_GT_MASK,
    INPUT_IMAGE,
    INPUT_POINT_COORDS,
    INPUT_POINT_LABELS,
    IOU_SUPERVISION,
    LOW_RES_LOGITS,
)
from dl_techniques.utils.logger import logger

#: ``low_res_logits`` is `4 * image_embedding_size`, i.e. `image_size / patch *
#: 4`. At SAM's fixed `patch_size=16` that is exactly `image_size / 4`.
MASK_DIVISOR = 4
#: Record keys the two sources agree on.
RECORD_IMAGE = "image"
RECORD_MASK = "mask"
RECORD_BOX = "box"
#: Point labels, mirroring ``PromptEncoder``'s convention.
FOREGROUND_LABEL = 1
PADDING_LABEL = -1
#: Logit written into pixels outside the mask when a point is drawn from it.
OUTSIDE_MASK_LOGIT = -1e9
#: A mask with fewer foreground pixels than this after downsampling is dropped:
#: no point can be sampled from an empty mask, and a silently kept one would be
#: supervised as "predict nothing" against a prompt pointing at a random pixel.
MIN_MASK_PIXELS = 1


# ---------------------------------------------------------------------------
# Synthetic source
# ---------------------------------------------------------------------------
def _draw_polygon_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    """Rasterize one filled convex polygon onto its OWN blank canvas."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    num_vertices = int(rng.integers(3, 8))
    centre_x = float(rng.uniform(0.25, 0.75) * size)
    centre_y = float(rng.uniform(0.25, 0.75) * size)
    radius = float(rng.uniform(0.12, 0.30) * size)
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=num_vertices))
    vertices = np.stack(
        [
            centre_x + radius * rng.uniform(0.7, 1.0, num_vertices) * np.cos(angles),
            centre_y + radius * rng.uniform(0.7, 1.0, num_vertices) * np.sin(angles),
        ],
        axis=-1,
    ).astype(np.int32)
    cv2.fillPoly(canvas, [vertices.reshape(-1, 1, 2)], 1)
    return canvas


def _draw_ellipse_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    """Rasterize one filled ellipse onto its OWN blank canvas."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    centre = (
        int(rng.integers(int(0.25 * size), int(0.75 * size))),
        int(rng.integers(int(0.25 * size), int(0.75 * size))),
    )
    axes = (
        int(rng.integers(max(2, size // 12), max(3, size // 4))),
        int(rng.integers(max(2, size // 12), max(3, size // 4))),
    )
    cv2.ellipse(
        canvas, centre, axes, float(rng.uniform(0.0, 360.0)), 0, 360, 1, -1
    )
    return canvas


#: Instance renderers. Each draws ONE object on its own canvas, which is what
#: makes a per-instance mask exist at all.
INSTANCE_RENDERERS = (_draw_polygon_mask, _draw_ellipse_mask)


def _box_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Tight xyxy box of a binary mask, in the mask's own pixel frame.

    Args:
        mask: ``(h, w)`` binary.

    Returns:
        ``(4,)`` float32 ``[x1, y1, x2, y2]``.

    Raises:
        ValueError: if the mask is empty -- an empty mask has no box, and
            returning zeros would be a plausible-looking wrong answer.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("_box_from_mask received an EMPTY mask; it has no box.")
    return np.asarray(
        [cols[0], rows[0], cols[-1] + 1, rows[-1] + 1], dtype="float32"
    )


def synthetic_instance_samples(
    num_samples: int,
    image_size: int,
    mask_size: Optional[int] = None,
    max_instances: int = 3,
    seed: int = 0,
) -> Iterator[Dict[str, np.ndarray]]:
    """
    Yield ``(image, per-instance mask, box)`` records drawn from shapes.

    Every yielded record is ONE instance: an image may contribute several
    records, each with its own mask and box, which is exactly the per-instance
    contract SAM trains on.

    Args:
        num_samples: How many INSTANCE records to yield.
        image_size: Side of the square image, in pixels.
        mask_size: Side of the square mask, i.e. ``low_res_logits``' resolution.
            Defaults to ``image_size // MASK_DIVISOR``.
        max_instances: Upper bound on objects drawn per image.
        seed: Seed for the ``np.random.Generator`` driving every draw.

    Yields:
        ``{"image": (H, W, 3) float32 [0, 255], "mask": (mh, mw) float32
        binary, "box": (4,) float32 xyxy in image pixels}``.

    Raises:
        ValueError: if ``mask_size`` does not divide ``image_size``, because the
            prompt coordinates are derived from that exact ratio.
    """
    mask_size = mask_size or image_size // MASK_DIVISOR
    if mask_size <= 0 or image_size % mask_size != 0:
        raise ValueError(
            f"mask_size={mask_size} must be a positive divisor of "
            f"image_size={image_size}; the mask grid is `low_res_logits`' "
            f"resolution and the prompt coordinates are scaled by their ratio."
        )
    rng = np.random.default_rng(seed)
    emitted = 0
    dropped_empty = 0
    while emitted < num_samples:
        num_instances = int(rng.integers(1, max_instances + 1))
        canvas = rng.uniform(0.05, 0.25, (image_size, image_size, 3)).astype(
            "float32"
        )
        instances: List[np.ndarray] = []
        for _ in range(num_instances):
            renderer = INSTANCE_RENDERERS[
                int(rng.integers(0, len(INSTANCE_RENDERERS)))
            ]
            full = renderer(image_size, rng)
            if full.sum() == 0:
                continue
            colour = rng.uniform(0.55, 1.0, 3).astype("float32")
            canvas[full > 0] = colour
            instances.append(full)

        image = (np.clip(canvas, 0.0, 1.0) * 255.0).astype("float32")
        for full in instances:
            if emitted >= num_samples:
                break
            low = cv2.resize(
                full.astype("float32"),
                (mask_size, mask_size),
                interpolation=cv2.INTER_AREA,
            )
            low = (low > 0.5).astype("float32")
            # An instance that vanishes at `low_res_logits` resolution is
            # DROPPED and COUNTED, never silently kept: a kept empty mask would
            # be supervised as "predict nothing" against a prompt pointing at an
            # arbitrary pixel, and no shape assertion could see it.
            if low.sum() < MIN_MASK_PIXELS:
                dropped_empty += 1
                continue
            yield {
                RECORD_IMAGE: image,
                RECORD_MASK: low,
                RECORD_BOX: _box_from_mask(full),
            }
            emitted += 1
    if dropped_empty:
        logger.info(
            "synthetic_instance_samples: dropped %d instance(s) that were "
            "empty after downsampling to %dx%d (emitted %d)",
            dropped_empty,
            mask_size,
            mask_size,
            emitted,
        )


# ---------------------------------------------------------------------------
# Prompt sampling (a `tf.data` map fn -- the model never does this)
# ---------------------------------------------------------------------------
def sample_point_in_mask(
    mask: tf.Tensor, image_size: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Draw one foreground point uniformly from a mask's interior.

    Args:
        mask: ``(mh, mw)`` binary float tensor at ``low_res_logits`` resolution.
        image_size: Side of the padded image frame the coordinate must land in.

    Returns:
        ``(coords, labels)`` of shapes ``(1, 2)`` float32 xy and ``(1,)`` int32.
        An EMPTY mask yields :data:`PADDING_LABEL` rather than a coordinate
        dressed up as a real prompt.
    """
    shape = tf.shape(mask)
    height, width = shape[0], shape[1]
    flat = tf.reshape(mask, (1, -1))
    logits = tf.where(
        flat > 0.5, tf.zeros_like(flat), tf.fill(tf.shape(flat), OUTSIDE_MASK_LOGIT)
    )
    index = tf.cast(tf.random.categorical(logits, 1)[0, 0], tf.int32)
    row = tf.cast(index // width, tf.float32)
    col = tf.cast(index % width, tf.float32)

    # Map the mask cell to its CENTRE in image pixels, then subtract 0.5:
    # `PromptEncoder._embed_points` adds its own +0.5 pixel-centre offset, so
    # the encoder lands at exactly the cell centre. Same convention as
    # `SAMTrainingModel._sample_from_region`; the two MUST agree or the
    # refinement points and the initial point live in different frames.
    scale_y = tf.cast(image_size, tf.float32) / tf.cast(height, tf.float32)
    scale_x = tf.cast(image_size, tf.float32) / tf.cast(width, tf.float32)
    coords = tf.stack([(col + 0.5) * scale_x - 0.5, (row + 0.5) * scale_y - 0.5])
    non_empty = tf.reduce_sum(flat) > 0.0
    label = tf.where(
        non_empty,
        tf.constant(FOREGROUND_LABEL, tf.int32),
        tf.constant(PADDING_LABEL, tf.int32),
    )
    return tf.reshape(coords, (1, 2)), tf.reshape(label, (1,))


def to_training_record(
    record: Dict[str, tf.Tensor], image_size: int
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """
    Turn one source record into ``(inputs, y_true)`` for ``SAMTrainingModel``.

    Args:
        record: A source record (``image``, ``mask``, ``box``).
        image_size: Side of the padded image frame.

    Returns:
        ``(inputs, y_true)``. ``y_true`` carries a SINGLE-instance GT stack
        whatever the model's round count is -- ``SAMMaskLoss`` repeats it across
        the concatenated mask axis, so this pipeline never learns about
        ``num_refinement_rounds``.
    """
    mask = record[RECORD_MASK]
    coords, labels = sample_point_in_mask(mask, image_size)
    gt_stack = tf.expand_dims(mask, axis=0)
    inputs = {
        INPUT_IMAGE: record[RECORD_IMAGE],
        INPUT_POINT_COORDS: coords,
        INPUT_POINT_LABELS: labels,
        INPUT_GT_MASK: gt_stack,
    }
    targets = {
        LOW_RES_LOGITS: gt_stack,
        IOU_SUPERVISION: tf.zeros((1, 2), dtype=tf.float32),
    }
    return inputs, targets


def build_sam_dataset(
    num_samples: int,
    image_size: int,
    batch_size: int,
    mask_size: Optional[int] = None,
    max_instances: int = 3,
    seed: int = 0,
    shuffle_buffer: int = 0,
) -> tf.data.Dataset:
    """
    Assemble a ``fit()``-consumable dataset from the synthetic source.

    Args:
        num_samples: Number of instance records in one epoch.
        image_size: Side of the square image.
        batch_size: Batch size.
        mask_size: ``low_res_logits`` resolution; defaults to
            ``image_size // MASK_DIVISOR``.
        max_instances: Objects drawn per image.
        seed: Seed for the source generator.
        shuffle_buffer: If > 0, shuffle with this buffer before batching.

    Returns:
        A batched ``tf.data.Dataset`` of ``(inputs, y_true)`` dicts.
    """
    mask_size = mask_size or image_size // MASK_DIVISOR
    signature = {
        RECORD_IMAGE: tf.TensorSpec((image_size, image_size, 3), tf.float32),
        RECORD_MASK: tf.TensorSpec((mask_size, mask_size), tf.float32),
        RECORD_BOX: tf.TensorSpec((4,), tf.float32),
    }
    dataset = tf.data.Dataset.from_generator(
        lambda: synthetic_instance_samples(
            num_samples=num_samples,
            image_size=image_size,
            mask_size=mask_size,
            max_instances=max_instances,
            seed=seed,
        ),
        output_signature=signature,
    )
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, seed=seed)
    dataset = dataset.map(
        lambda record: to_training_record(record, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
