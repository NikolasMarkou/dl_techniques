"""Synthetic moving-instance video for the SAM 2 trainer.

Why synthetic, and why a NEW source
-----------------------------------
Neither in-repo video dataset emits masks: ``datasets/bdd100k_video.py`` and
``datasets/synthetic_drone_video.py`` are both VideoJEPA-schema pipelines that
yield ``{"pixels": (B, T, H, W, 3)}`` and nothing else. No annotated
video-object-segmentation dataset exists on this machine either -- only static
COCO 2017. Synthetic shapes are therefore the ONLY route to per-frame ground
truth, which is what makes this file exist rather than an adapter.

What one clip is
----------------
ONE tracked instance, drawn by ``train.sam.data``'s renderers, translated with
a per-clip integer velocity across a noise background, plus an opaque
full-height distractor bar that sweeps past it. On a chosen consecutive window
of frames the bar covers the instance ENTIRELY, so those frames' ground truth
is genuinely all zeros. That is the occlusion signal -- there is no separate
label channel, exactly as upstream derives ``target_obj`` from
``target_masks``.

Three properties this module guarantees by construction, not by hope:

* **the object really moves** -- the velocity is drawn non-zero and the patch is
  pasted at an exact integer offset per frame, so consecutive mask centroids
  differ;
* **the number of all-zero ground-truth frames is EXACTLY**
  ``occlusion_frames`` -- every clip is verified after rendering against the
  window that was asked for, and a clip that disagrees (a thin sliver lost to
  the downsample to mask resolution) is DROPPED and COUNTED, never emitted;
* **frame 0 is never occluded** -- the frame-0 prompt is sampled from frame 0's
  mask, and a window that reaches frame 0 is refused by
  :func:`_occlusion_window` rather than producing a prompt drawn from nothing.

Presence is never a field
-------------------------
``object_score_logits``' target is computed as ``any(mask[t] > 0)`` inside
:func:`to_video_training_record`. There is no sampled visibility flag anywhere
in this file, so a flag cannot disagree with the mask it is supposed to
describe.

Target keys have ONE home
-------------------------
The three target keys come from ``models/sam2/training_model.py``'s exported
constants, and ``tests/test_train/test_sam2/test_data.py`` asserts the emitted
target key set equals that module's ``OUTPUT_KEYS``. That assertion was RED by
design between step 2 and step 4, while the wrapper had two output keys and the
plan asked for three -- which is the point: a pipeline target the model does not
produce, or a model output nothing supervises, cannot pass unnoticed.

Nothing is written to disk. Every clip is generated in memory, per epoch.

``numpy``/``cv2`` in the generator and ``tf`` in the map functions are exempt
from the ``keras.ops``-only invariant, exactly as ``train/sam/data.py`` is: this
is a host-side data pipeline, not a forward path.
"""

from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from dl_techniques.models.sam2.training_model import (
    INPUT_BOXES,
    INPUT_GT_MASKS,
    INPUT_IMAGE,
    INPUT_POINT_COORDS,
    INPUT_POINT_LABELS,
    SAM2_IOU_SUPERVISION,
    SAM2_LOW_RES_LOGITS,
    SAM2_OBJECT_SCORE_LOGITS,
)
from dl_techniques.utils.logger import logger

# DECISION plan-2026-08-04T044628-4c240b4c/D-057
# Reused wholesale from the SAM 1 pipeline. `INSTANCE_RENDERERS` draws ONE
# object on its own canvas, which is the only reason a per-instance mask exists
# at all, and the three prompt helpers are frame-agnostic (they take a single
# `(mh, mw)` mask), so frame 0 of a clip is just another mask to them.
# `_box_from_mask` is PRIVATE and imported anyway: publishing it would edit
# `train/sam/data.py`, which SAM 1's 357-test gate protects. See decisions.md
# D-057 for the trade-off.
from train.sam.data import (
    INSTANCE_RENDERERS,
    MASK_DIVISOR,
    MIN_MASK_PIXELS,
    RECORD_BOX,
    RECORD_IMAGE,
    RECORD_MASK,
    _box_from_mask,
    jitter_box,
    sample_point_in_mask,
    sample_point_outside_mask,
)

#: Slack added to the bar's coverage margin, in pixels. Any value >= 1 makes
#: the coverage window exact; see :func:`_bar_geometry`.
BAR_MARGIN_SLACK = 1
#: How many times one clip is re-rolled before the source gives up. A clip is
#: re-rolled when the drawn shape has no room to move, or when the rendered
#: all-zero-frame set disagrees with the requested occlusion window.
MAX_CLIP_ATTEMPTS = 64


# ---------------------------------------------------------------------------
# occlusion window
# ---------------------------------------------------------------------------
def _occlusion_window(
        num_frames: int, occlusion_frames: int, start: Optional[int]
) -> np.ndarray:
    """Build the per-frame "ground truth is empty" flags of one clip.

    :param num_frames: Clip length ``T``.
    :type num_frames: int
    :param occlusion_frames: How many CONSECUTIVE frames are fully occluded.
        ``0`` produces a clip the object is visible in throughout.
    :type occlusion_frames: int
    :param start: First occluded frame index. ``None`` is only valid when
        ``occlusion_frames`` is ``0``.
    :type start: Optional[int]
    :return: ``(T,)`` bool, ``True`` where the ground truth must be all zeros.
    :rtype: numpy.ndarray
    :raises ValueError: If the window does not fit in the clip, or if it
        reaches FRAME 0. A frame-0 occlusion leaves no region for the frame-0
        prompt to be sampled from, and the prompt would be drawn from an empty
        mask -- which ``sample_point_in_mask`` answers with a padding label
        rather than a raise, so nothing downstream would report it.
    """
    if occlusion_frames < 0 or occlusion_frames > num_frames - 1:
        raise ValueError(
            f"occlusion_frames must be in [0, num_frames - 1] = "
            f"[0, {num_frames - 1}]; got {occlusion_frames}. Frame 0 carries "
            f"the prompt and can never be occluded, so a whole-clip occlusion "
            f"is not expressible."
        )
    absent = np.zeros((num_frames,), dtype=bool)
    if occlusion_frames == 0:
        return absent
    if start is None:
        raise ValueError(
            "an occlusion window of length "
            f"{occlusion_frames} needs an explicit start frame."
        )
    if start + occlusion_frames > num_frames:
        raise ValueError(
            f"occlusion window [{start}, {start + occlusion_frames}) does not "
            f"fit in a {num_frames}-frame clip."
        )
    absent[start:start + occlusion_frames] = True
    if bool(absent[0]):
        raise ValueError(
            f"the occlusion window [{start}, {start + occlusion_frames}) "
            f"reaches FRAME 0. The frame-0 prompt is sampled from frame 0's "
            f"mask; an occluded frame 0 would have it sampled from an empty "
            f"mask, which yields a padding label and trains the model on a "
            f"clip with no prompt at all."
        )
    return absent


def _bar_geometry(occlusion_frames: int, patch_width: int) -> Tuple[int, int]:
    """Margin and sweep speed that make the coverage window EXACT.

    The bar spans ``[object_x - margin + offset(t), ... + bar_width)`` with
    ``bar_width = patch_width + 2 * margin`` and
    ``offset(t) = speed * (2t - 2*start - occlusion_frames + 1)``, an integer
    that is ``0`` at the window's centre and grows by ``2 * speed`` per frame.
    Coverage is therefore exactly ``|offset(t)| <= margin``, and disjointness
    exactly ``|offset(t)| >= patch_width + margin``.

    Inside the window ``|offset|`` peaks at ``speed * (occlusion_frames - 1)``;
    one frame outside it is already ``speed * (occlusion_frames + 1)``. Taking
    ``speed = patch_width`` and
    ``margin = speed * (occlusion_frames - 1) + BAR_MARGIN_SLACK`` makes every
    in-window frame FULLY covered and every out-of-window frame FULLY disjoint,
    with no rounding and no partial erosion anywhere:

    * in-window: ``speed * (occlusion_frames - 1) <= margin`` by construction;
    * out-of-window: ``speed * (occlusion_frames + 1) >= patch_width + margin``
      reduces to ``patch_width >= BAR_MARGIN_SLACK``.

    .. note::

       DECISION plan-2026-08-04T044628-4c240b4c/D-059. Do NOT lower ``speed``
       to a small constant for a smoother-looking sweep. That was the first
       design (2 px per half-frame) and it MEASURED a 3-column partially-eroded
       sliver on the frame adjacent to the window at a 20 px instance. Such a
       sliver survives at image resolution and vanishes under ``INTER_AREA`` at
       mask resolution, which pushes the all-zero-frame count above
       ``occlusion_frames`` with no other symptom -- and that count is exactly
       what the step-3 loss gate is calibrated against. See decisions.md D-059.

    :param occlusion_frames: Length of the occlusion window.
    :type occlusion_frames: int
    :param patch_width: Width of the tracked instance's bounding box, in
        pixels.
    :type patch_width: int
    :return: ``(margin, speed)`` in pixels.
    :rtype: Tuple[int, int]
    """
    speed = max(1, int(patch_width))
    margin = max(0, speed * (occlusion_frames - 1) + BAR_MARGIN_SLACK)
    return margin, speed


# ---------------------------------------------------------------------------
# clip rendering
# ---------------------------------------------------------------------------
def _draw_clip(
        rng: np.random.Generator,
        num_frames: int,
        image_size: int,
        mask_size: int,
        absent: np.ndarray,
        occlusion_start: int,
        occlusion_frames: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Render one clip, or return ``None`` if this roll is unusable.

    :param rng: The clip's random source.
    :type rng: numpy.random.Generator
    :param num_frames: Clip length ``T``.
    :type num_frames: int
    :param image_size: Side of the square frame, in pixels.
    :type image_size: int
    :param mask_size: Side of the square ground-truth grid.
    :type mask_size: int
    :param absent: ``(T,)`` bool from :func:`_occlusion_window`.
    :type absent: numpy.ndarray
    :param occlusion_start: First occluded frame (ignored when
        ``occlusion_frames`` is ``0``).
    :type occlusion_start: int
    :param occlusion_frames: Length of the occlusion window.
    :type occlusion_frames: int
    :return: A record, or ``None`` when the drawn shape has no room to move or
        the rendered empty-frame set disagrees with ``absent``.
    :rtype: Optional[Dict[str, numpy.ndarray]]
    """
    renderer = INSTANCE_RENDERERS[int(rng.integers(0, len(INSTANCE_RENDERERS)))]
    stencil = renderer(image_size, rng)
    if stencil.sum() == 0:
        return None

    x1, y1, x2, y2 = (int(v) for v in _box_from_mask(stencil))
    patch = stencil[y1:y2, x1:x2]
    patch_h, patch_w = patch.shape
    span_x = image_size - patch_w
    span_y = image_size - patch_h

    # The object MUST move, and it must move by at least ONE ground-truth cell
    # per frame. A sub-cell velocity is real motion at image resolution and can
    # quantize to a stationary mask, at which point every "the mask changed"
    # assertion passes on a clip that carries no visible motion at all.
    stride = image_size // mask_size
    steps = max(1, num_frames - 1)
    limit_x = span_x // steps
    limit_y = span_y // steps
    velocity_x = int(rng.integers(-limit_x, limit_x + 1))
    velocity_y = int(rng.integers(-limit_y, limit_y + 1))
    if max(abs(velocity_x), abs(velocity_y)) < stride:
        if limit_x >= stride:
            velocity_x = int(rng.choice([-1, 1])) * int(
                rng.integers(stride, limit_x + 1)
            )
        elif limit_y >= stride:
            velocity_y = int(rng.choice([-1, 1])) * int(
                rng.integers(stride, limit_y + 1)
            )
        else:
            return None

    def _origin(span: int, velocity: int) -> int:
        low = max(0, -velocity * (num_frames - 1))
        high = min(span, span - velocity * (num_frames - 1))
        return int(rng.integers(low, high + 1))

    origin_x = _origin(span_x, velocity_x)
    origin_y = _origin(span_y, velocity_y)

    background = rng.uniform(0.05, 0.25, (image_size, image_size, 3)).astype(
        "float32"
    )
    object_colour = rng.uniform(0.55, 1.0, 3).astype("float32")
    bar_colour = rng.uniform(0.55, 1.0, 3).astype("float32")
    margin, speed = _bar_geometry(occlusion_frames, patch_w)
    bar_width = patch_w + 2 * margin

    images = np.zeros((num_frames, image_size, image_size, 3), dtype="float32")
    low_res = np.zeros((num_frames, mask_size, mask_size), dtype="float32")
    frame_zero_full: Optional[np.ndarray] = None

    for t in range(num_frames):
        pos_x = origin_x + velocity_x * t
        pos_y = origin_y + velocity_y * t
        full = np.zeros((image_size, image_size), dtype="uint8")
        full[pos_y:pos_y + patch_h, pos_x:pos_x + patch_w] = patch

        canvas = background.copy()
        canvas[full > 0] = object_colour

        # The bar is OPAQUE and full-height: it is composited after the object
        # and the mask is erased under it. A transparent distractor -- one that
        # paints pixels but leaves the ground truth alone -- would give clips
        # that look occluded and are not, which is exactly the defect the
        # exact-count check below exists to catch.
        offset = speed * (2 * t - 2 * occlusion_start - occlusion_frames + 1)
        bar_x = pos_x - margin + offset
        col_start = max(0, bar_x)
        col_stop = min(image_size, bar_x + bar_width)
        if col_stop > col_start:
            canvas[:, col_start:col_stop] = bar_colour
            full[:, col_start:col_stop] = 0

        images[t] = np.clip(canvas, 0.0, 1.0) * 255.0
        resized = cv2.resize(
            full.astype("float32"),
            (mask_size, mask_size),
            interpolation=cv2.INTER_AREA,
        )
        low_res[t] = (resized > 0.5).astype("float32")
        if t == 0:
            frame_zero_full = full

    # The count is VERIFIED, not assumed. A frame that survives at image
    # resolution can still vanish at mask resolution, which would make the
    # all-zero-frame count exceed `occlusion_frames` with no other symptom.
    rendered_absent = low_res.reshape(num_frames, -1).sum(axis=1) < MIN_MASK_PIXELS
    if not np.array_equal(rendered_absent, absent):
        return None

    assert frame_zero_full is not None  # frame 0 always runs
    return {
        RECORD_IMAGE: images,
        RECORD_MASK: low_res,
        RECORD_BOX: _box_from_mask(frame_zero_full),
    }


def synthetic_moving_instance_video(
        num_clips: int,
        num_frames: int,
        image_size: int,
        mask_size: Optional[int] = None,
        occlusion_frames: int = 1,
        occlusion_start: Optional[int] = None,
        seed: int = 0,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield moving-instance video clips with counted, genuine occlusions.

    :param num_clips: How many clips to yield.
    :type num_clips: int
    :param num_frames: Clip length ``T``.
    :type num_frames: int
    :param image_size: Side of the square frame, in pixels.
    :type image_size: int
    :param mask_size: Side of the square ground-truth grid, i.e.
        ``low_res_logits``' resolution. Defaults to
        ``image_size // MASK_DIVISOR``, which is also SAM 2's own
        ``feature_grid * 4``.
    :type mask_size: Optional[int]
    :param occlusion_frames: Number of CONSECUTIVE fully-occluded frames per
        clip. Every emitted clip has exactly this many all-zero ground-truth
        frames -- it is a count, not a target.
    :type occlusion_frames: int
    :param occlusion_start: First occluded frame. ``None`` draws it uniformly
        from ``[1, T - occlusion_frames]``; pinning it makes a clip's occlusion
        layout reproducible independently of the shape draw.
    :type occlusion_start: Optional[int]
    :param seed: Seed of the ``numpy`` generator driving every draw.
    :type seed: int
    :return: An iterator of ``{"image": (T, S, S, 3) float32 in [0, 255],
        "mask": (T, mh, mw) float32 binary, "box": (4,) float32 xyxy in FRAME
        0's image pixels}``.
    :rtype: Iterator[Dict[str, numpy.ndarray]]
    :raises ValueError: If ``mask_size`` does not divide ``image_size`` (the
        prompt coordinates are scaled by exactly that ratio), or from
        :func:`_occlusion_window` if the window does not fit or reaches
        frame 0.
    :raises RuntimeError: If :data:`MAX_CLIP_ATTEMPTS` consecutive rolls fail
        to produce a clip matching the requested window. Reported rather than
        returned as a silently short -- or silently wrong -- epoch.
    """
    mask_size = mask_size or image_size // MASK_DIVISOR
    if mask_size <= 0 or image_size % mask_size != 0:
        raise ValueError(
            f"mask_size={mask_size} must be a positive divisor of "
            f"image_size={image_size}; the mask grid is `low_res_logits`' "
            f"resolution and the prompt coordinates are scaled by their ratio."
        )
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1; got {num_frames}.")
    # Validate the window ONCE, up front, through the same function that builds
    # it per clip -- so the bound and the fit have exactly one home and a
    # per-clip `start` draw can never be handed an empty range.
    _occlusion_window(
        num_frames, occlusion_frames, 1 if occlusion_frames else None
    )
    if occlusion_start is not None:
        _occlusion_window(num_frames, occlusion_frames, occlusion_start)

    rng = np.random.default_rng(seed)
    dropped = 0
    for _ in range(num_clips):
        record: Optional[Dict[str, np.ndarray]] = None
        for _attempt in range(MAX_CLIP_ATTEMPTS):
            if occlusion_frames == 0:
                start = 0
            elif occlusion_start is None:
                start = int(rng.integers(1, num_frames - occlusion_frames + 1))
            else:
                start = int(occlusion_start)
            absent = _occlusion_window(num_frames, occlusion_frames, start)
            record = _draw_clip(
                rng,
                num_frames,
                image_size,
                mask_size,
                absent,
                start,
                occlusion_frames,
            )
            if record is not None:
                break
            dropped += 1
        if record is None:
            raise RuntimeError(
                f"synthetic_moving_instance_video failed to render a clip "
                f"whose all-zero ground-truth frames match the requested "
                f"{occlusion_frames}-frame occlusion window in "
                f"{MAX_CLIP_ATTEMPTS} attempts at image_size={image_size}, "
                f"mask_size={mask_size}, num_frames={num_frames}. The clip "
                f"count is a HARD property of this source, so an approximate "
                f"epoch is refused rather than emitted."
            )
        yield record
    if dropped:
        logger.info(
            "synthetic_moving_instance_video: dropped %d rolled clip(s) whose "
            "shape had no room to move or whose empty-frame set did not match "
            "the requested %d-frame occlusion window (emitted %d)",
            dropped,
            occlusion_frames,
            num_clips,
        )


# ---------------------------------------------------------------------------
# record -> (inputs, targets)
# ---------------------------------------------------------------------------
def to_video_training_record(
        record: Dict[str, tf.Tensor],
        image_size: int,
        num_background_points: int = 0,
        include_box: bool = False,
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Turn one clip into ``(inputs, targets)`` for ``SAM2TrainingModel``.

    :param record: A clip record from :func:`synthetic_moving_instance_video`.
    :type record: Dict[str, tf.Tensor]
    :param image_size: Side of the square frame the prompt must land in.
    :type image_size: int
    :param num_background_points: Background points added beside the single
        foreground one. All of them are drawn from FRAME 0.
    :type num_background_points: int
    :param include_box: Whether to add a jittered frame-0 box prompt.
    :type include_box: bool
    :return: ``(inputs, targets)``. ``targets`` is keyed by the wrapper's own
        exported output-key constants.
    :rtype: Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]
    """
    mask = record[RECORD_MASK]
    # FRAME 0, and only frame 0. Frames 1..T-1 are conditioned by the memory
    # bank; prompting them too would teach the model that the object never
    # leaves its frame-0 location, and prompting frame t instead would put the
    # prompt somewhere the conditioning frame's object is not.
    frame_zero = mask[0]
    coords, labels = sample_point_in_mask(frame_zero, image_size)
    for _ in range(num_background_points):
        background_coords, background_labels = sample_point_outside_mask(
            frame_zero, image_size
        )
        coords = tf.concat([coords, background_coords], axis=0)
        labels = tf.concat([labels, background_labels], axis=0)

    # Presence is DERIVED here and stored nowhere. Upstream's `target_obj` is
    # `any(target_masks > 0)`; a sampled visibility field would be a second
    # source of truth that can disagree with the mask it describes, with no
    # shape or finiteness symptom when it does.
    present = tf.cast(
        tf.reduce_max(mask, axis=[1, 2]) > 0.0, tf.float32
    )

    inputs = {
        INPUT_IMAGE: record[RECORD_IMAGE],
        INPUT_POINT_COORDS: coords,
        INPUT_POINT_LABELS: labels,
        INPUT_GT_MASKS: mask,
    }
    if include_box:
        inputs[INPUT_BOXES] = tf.expand_dims(
            jitter_box(record[RECORD_BOX], image_size), axis=0
        )
    # DECISION plan-2026-08-04T044628-4c240b4c/D-058
    # THREE target keys, each sourced from `models/sam2/training_model.py`'s
    # exported constants. Do NOT spell any of these strings locally: a key
    # spelled in two places is exactly the drift H-5 exists to prevent, and
    # `compile(loss={...})` reports the mismatch as a key error with no cause
    # attached. The coupling is executable:
    # `test_the_target_keys_are_exactly_the_wrappers_output_keys` asserts this
    # dict's keys equal `training_model.OUTPUT_KEYS`. That canary was RED BY
    # DESIGN from step 2 until step 4 added the `iou_supervision` output; it is
    # what made adding the output without the target impossible.
    # See decisions.md D-058.
    #
    # The `iou_supervision` TARGET is structurally unused, and that is a
    # property of the quantity rather than an omission: the achieved IoU is a
    # function of the prediction and the ground truth together, so it exists
    # only inside the model, which packs it into `y_pred` alongside the
    # predicted IoU. `SAMIoULoss` reads both from there and ignores `y_true`
    # entirely. A `tf.data` pipeline cannot produce it; the zeros are a
    # placeholder Keras requires, not a supervision signal.
    targets = {
        SAM2_LOW_RES_LOGITS: mask,
        SAM2_OBJECT_SCORE_LOGITS: tf.expand_dims(present, axis=-1),
        SAM2_IOU_SUPERVISION: tf.zeros((tf.shape(mask)[0], 2), tf.float32),
    }
    return inputs, targets


def build_sam2_video_dataset(
        num_clips: int,
        num_frames: int,
        image_size: int,
        batch_size: int,
        mask_size: Optional[int] = None,
        occlusion_frames: int = 1,
        occlusion_start: Optional[int] = None,
        seed: int = 0,
        shuffle_buffer: int = 0,
        num_background_points: int = 0,
        include_box: bool = False,
) -> tf.data.Dataset:
    """Assemble a ``fit()``-consumable clip dataset.

    :param num_clips: Clips in one epoch.
    :type num_clips: int
    :param num_frames: Clip length ``T``. Must equal the wrapper's
        ``num_frames``: it is the bound of an unrolled Python loop, so a
        mismatch is a shape error, not a slow path.
    :type num_frames: int
    :param image_size: Side of the square frame.
    :type image_size: int
    :param batch_size: Batch size, in clips.
    :type batch_size: int
    :param mask_size: Ground-truth grid side; defaults to
        ``image_size // MASK_DIVISOR``.
    :type mask_size: Optional[int]
    :param occlusion_frames: Fully-occluded frames per clip.
    :type occlusion_frames: int
    :param occlusion_start: First occluded frame, or ``None`` to draw it.
    :type occlusion_start: Optional[int]
    :param seed: Seed for the source generator and the shuffle.
    :type seed: int
    :param shuffle_buffer: If > 0, shuffle with this buffer before batching.
    :type shuffle_buffer: int
    :param num_background_points: Background points added to the frame-0
        prompt.
    :type num_background_points: int
    :param include_box: Whether to add a jittered frame-0 box prompt.
    :type include_box: bool
    :return: A batched dataset of ``(inputs, targets)`` dicts.
    :rtype: tensorflow.data.Dataset
    """
    mask_size = mask_size or image_size // MASK_DIVISOR
    signature = {
        RECORD_IMAGE: tf.TensorSpec(
            (num_frames, image_size, image_size, 3), tf.float32
        ),
        RECORD_MASK: tf.TensorSpec((num_frames, mask_size, mask_size), tf.float32),
        RECORD_BOX: tf.TensorSpec((4,), tf.float32),
    }
    dataset = tf.data.Dataset.from_generator(
        lambda: synthetic_moving_instance_video(
            num_clips=num_clips,
            num_frames=num_frames,
            image_size=image_size,
            mask_size=mask_size,
            occlusion_frames=occlusion_frames,
            occlusion_start=occlusion_start,
            seed=seed,
        ),
        output_signature=signature,
    )
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, seed=seed)
    dataset = dataset.map(
        lambda record: to_video_training_record(
            record,
            image_size,
            num_background_points=num_background_points,
            include_box=include_box,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
