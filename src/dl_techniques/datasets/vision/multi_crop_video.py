"""Video-shaped multi-crop ``tf.data`` transform: 1 global + N local views.

Produces the ``{"global_frame": (T, S, S, C), "local_frames": (V, T, S, S, C)}``
element :class:`~dl_techniques.models.vision.levjepa.training.LeVJEPATrainingModel`
consumes, from one ``(T, H, W, C)`` source clip -- the ``"pixels"`` key
:func:`~dl_techniques.datasets.synthetic_drone_video.synthetic_drone_video_dataset`
/ :func:`~dl_techniques.datasets.bdd100k_video.bdd100k_video_dataset` both yield
(see D-005, which already decided this module's location).

**Reuses, does not re-derive**, the crop/augmentation primitives from
:mod:`dl_techniques.datasets.vision.multi_crop`
(``_random_resized_crop`` / ``_gaussian_blur`` / ``_maybe``) -- the SAME area
scale / aspect-ratio / flip / colour-jitter / grayscale / Gaussian-blur recipe
that module documents, applied per view.

Same crop across every frame of one clip (D-019)
--------------------------------------------------
A per-frame independent random crop would make a "clip" flicker between
unrelated spatial windows frame to frame, defeating the point of feeding a
temporal encoder. ``_random_resized_crop`` draws its crop box and every other
augmentation decision from the ``draw`` callable it is given, in a FIXED call
order (target-area, aspect, offset-w, offset-h, ...). This module wraps that
``draw`` in a small replay cache (:func:`_freeze_draw`) that answers the
``i``-th call of one view with the value it returned the FIRST time it was
asked -- so re-running the exact same augmentation pipeline (same call order)
on each of the ``T`` frames of one view reproduces the identical crop box,
flip decision, jitter offsets and blur sigma every time, with zero duplicated
crop-box math. See the ``DECISION`` note on :func:`_augment_clip_view`.

Global/local resolution (mirrors ``multi_crop.py``'s D-002)
-------------------------------------------------------------
Both global and local views are rendered at the SAME ``crop_size`` (a smaller
AREA is cropped for local views via ``local_scale``, then resized UP) because
``LeVJEPAEncoder`` is one shared encoder with one fixed spatial input shape --
the same constraint that made ``multi_crop.py`` raise on a differing
``local_crop_size``.

What is intentionally NOT implemented (named, not silently dropped)
----------------------------------------------------------------------
Saturation/hue jitter and solarization: same reason as ``multi_crop.py`` --
they assume an ``[0, 1]``-valued input and this pipeline's value domain is not
asserted to be that (the loaders document ``[0, 1]`` today, but this module
does not depend on it). Resolution-asymmetric global/local crops (paper: full
res global, lower res local): plan.md Assumption A4 explicitly scopes this
out for Step 6 -- a reasonable video-shaped multi-crop is the bar, not exact
paper fidelity.

Raw TensorFlow ops are used deliberately: this is a ``tf.data`` transform, not
a Keras layer (same rule ``multi_crop.py`` states).
"""

from typing import Any, Callable, Dict, Tuple

import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.datasets.vision.multi_crop import (
    DrawFn,
    _gaussian_blur,
    _maybe,
    _random_resized_crop,
)

# ---------------------------------------------------------------------


def _replay_draw(seed_pair: tf.Tensor) -> Tuple[DrawFn, Callable[[], None]]:
    """Build a ``draw`` whose ``i``-th call is DETERMINISTIC given ``seed_pair``.

    Interface contract:
        Parameters:
            seed_pair: ``(2,)`` int64 tensor, a fresh per-VIEW seed drawn
                ONCE, at a point in the graph that is NOT inside any
                ``tf.cond`` branch (see the ``DECISION`` note below for why
                that placement matters).
        Returns:
            ``(draw, reset)``. ``draw(shape, minval, maxval)`` returns
            ``tf.random.stateless_uniform`` keyed on ``seed_pair`` plus the
            call's 0-based index since the last ``reset()``. Replaying the
            SAME call sequence after a ``reset()`` therefore returns
            BIT-IDENTICAL values, with no tensor caching at all -- every call
            is a fresh, independently-valid stateless draw.
        Failure mode:
            None. Two views never collide because each gets its own
            freshly-drawn ``seed_pair``.

    Not promoted elsewhere: single-consumer primitive
    (:func:`_augment_clip_view`), the earned-abstraction bar this plan's
    Complexity Budget already applies to every new abstraction.
    """
    counter = [0]

    def draw(shape: Any, minval: float, maxval: float) -> tf.Tensor:
        idx = counter[0]
        counter[0] += 1
        call_seed = seed_pair + tf.constant([idx, 0], dtype=tf.int64)
        return tf.random.stateless_uniform(
            shape, seed=call_seed, minval=minval, maxval=maxval, dtype=tf.float32
        )

    def reset() -> None:
        counter[0] = 0

    return draw, reset


# DECISION plan-2026-09-03T113223-2a714a91/D-019
# A per-frame-independent crop would make one clip's "view" flicker between
# unrelated spatial windows every frame, defeating the point of a temporal
# encoder. `_random_resized_crop` (from `multi_crop.py`, reused unmodified,
# not re-derived per D-005) draws its crop box purely from the `draw`
# callable it is given, in a FIXED call order -- so replaying the SAME call
# sequence with the SAME underlying randomness reproduces the SAME crop box,
# with zero duplicated crop-box math.
#
# WHAT NOT TO DO, and WHY (measured, not assumed): the first implementation
# tried a Python-level CACHE of each `draw()` call's tensor RESULT, replayed
# by returning the cached tensor on subsequent frames. That FAILS under
# `tf.data`'s graph tracing with `InaccessibleTensorError`: `_maybe` (reused
# from `multi_crop.py`) wraps its `transform` in `tf.cond`, so a draw made
# inside `_jitter()`/`_blur()` is defined inside THAT `tf.cond` call's own
# FuncGraph; reusing the cached tensor on frame 2 (inside a DIFFERENT
# `tf.cond` call, hence a different FuncGraph) tries to read across FuncGraph
# boundaries, which TensorFlow refuses outright, and the whole `.map()`
# construction raises before a single batch is ever produced.
#
# Reproduced with `tf.random.stateless_uniform` instead: every `draw()` call
# is a FRESH stateless op, so no tensor ever needs to cross a FuncGraph
# boundary; determinism-across-frames comes from `seed_pair` (drawn ONCE per
# view, at the top of `_augment_clip_view`, OUTSIDE any `tf.cond`) plus a
# reset-per-frame call-order counter, not from caching. See decisions.md
# D-019.
def _augment_clip_view(
    clip: tf.Tensor,
    out_size: int,
    *,
    is_global: bool,
    global_scale: Tuple[float, float],
    local_scale: Tuple[float, float],
    aspect_ratio_range: Tuple[float, float],
    flip_prob: float,
    color_jitter_prob: float,
    brightness: float,
    contrast: float,
    grayscale_prob: float,
    blur_prob: float,
    blur_sigma_range: Tuple[float, float],
    blur_radius: int,
) -> tf.Tensor:
    """Augment ONE view of a ``(T, H, W, C)`` clip, consistently across ``T``.

    :param clip: ``(T, H, W, C)`` float32 clip.
    :param out_size: Side length of the returned square view.
    :param is_global: Whether this is the (one) global view (uses
        ``global_scale``) or a local view (uses ``local_scale``).
    :return: ``(T, out_size, out_size, C)`` float32.
    """
    num_frames = clip.shape[0]

    seed_pair = tf.cast(
        tf.random.uniform((2,), minval=0, maxval=2 ** 30, dtype=tf.int32), tf.int64
    )
    draw, reset = _replay_draw(seed_pair)
    scale = global_scale if is_global else local_scale

    def _augment_one_frame(frame: tf.Tensor) -> tf.Tensor:
        reset()
        view = _random_resized_crop(
            frame, out_size=out_size, scale=scale, ratio=aspect_ratio_range, draw=draw,
        )
        view = _maybe(
            flip_prob, draw, lambda: tf.image.flip_left_right(view), view,
        )

        def _jitter() -> tf.Tensor:
            offset = draw([], -brightness, brightness)
            factor = draw([], 1.0 - contrast, 1.0 + contrast)
            mean = tf.reduce_mean(view, axis=(0, 1), keepdims=True)
            return (view - mean) * factor + mean + offset

        view = _maybe(color_jitter_prob, draw, _jitter, view)

        view = _maybe(
            grayscale_prob, draw,
            lambda: tf.tile(
                tf.reduce_mean(view, axis=-1, keepdims=True),
                (1, 1, tf.shape(view)[-1]),
            ),
            view,
        )

        def _blur() -> tf.Tensor:
            sigma = draw([], blur_sigma_range[0], blur_sigma_range[1])
            return _gaussian_blur(view, sigma, blur_radius)

        view = _maybe(blur_prob, draw, _blur, view)
        return view

    frames = [_augment_one_frame(clip[t]) for t in range(num_frames)]
    return tf.stack(frames, axis=0)


def make_multi_crop_video_map_fn(
    crop_size: int,
    *,
    num_frames: int,
    local_crops_number: int = 2,
    global_scale: Tuple[float, float] = (0.4, 1.0),
    local_scale: Tuple[float, float] = (0.05, 0.4),
    aspect_ratio_range: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    flip_prob: float = 0.5,
    color_jitter_prob: float = 0.8,
    brightness: float = 0.4,
    contrast: float = 0.4,
    grayscale_prob: float = 0.2,
    blur_prob: float = 0.5,
    blur_sigma_range: Tuple[float, float] = (0.1, 2.0),
) -> Callable[[Dict[str, tf.Tensor], Any], Tuple[Dict[str, tf.Tensor], Any]]:
    """Build the video multi-crop ``tf.data`` map function.

    The returned function maps ONE UNBATCHED element
    ``({"pixels": (T, H, W, C)}, label)`` -- the per-clip shape a
    ``synthetic_drone_video_dataset`` / ``bdd100k_video_dataset`` batch
    produces after ``.unbatch()`` -- to
    ``({"global_frame": (T, S, S, C), "local_frames": (V, T, S, S, C)}, label)``,
    ``S = crop_size``, ``V = local_crops_number``.

    :param crop_size: Side length of every returned view (global and local
        alike -- see the module docstring's "Global/local resolution" note).
    :param num_frames: ``T``, the clip's frame count. Used only to pin the
        output's static shape via ``tf.ensure_shape``.
    :param local_crops_number: Number of local views, ``>= 0``. Deliberately
        small by default (paper: 10) for a smoke-scale configuration.
    :param global_scale: ``(min, max)`` source-area fraction for the one
        global crop.
    :param local_scale: ``(min, max)`` source-area fraction for local crops;
        must not exceed ``global_scale``'s maximum.
    :param aspect_ratio_range: ``(min, max)`` width/height ratio of the crop.
    :param flip_prob: Per-view horizontal-flip probability.
    :param color_jitter_prob: Per-view brightness+contrast jitter probability.
    :param brightness: Half-width of the additive brightness offset.
    :param contrast: Half-width of the contrast factor around ``1.0``.
    :param grayscale_prob: Per-view grayscale-collapse probability.
    :param blur_prob: Per-view Gaussian-blur probability (both global and
        local views share one probability -- a simplification of
        ``multi_crop.py``'s asymmetric ``global_blur_probs``, since LeVJEPA
        has only one global view so there is no pair to be asymmetric
        between).
    :param blur_sigma_range: ``(min, max)`` Gaussian blur sigma, in pixels.
    :return: A ``tf.data``-mappable callable.
    :raises ValueError: If ``crop_size``/``num_frames`` are not positive, if
        ``local_crops_number`` is negative, if either scale range is invalid,
        or if ``local_scale``'s maximum exceeds ``global_scale``'s maximum.
    """
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if local_crops_number < 0:
        raise ValueError(
            f"local_crops_number must be >= 0, got {local_crops_number}"
        )
    for name, rng_pair in (("global_scale", global_scale), ("local_scale", local_scale)):
        if len(rng_pair) != 2 or not 0.0 < rng_pair[0] <= rng_pair[1] <= 1.0:
            raise ValueError(
                f"{name} must be an increasing (min, max) pair inside (0, 1], "
                f"got {rng_pair}"
            )
    if local_scale[1] > global_scale[1]:
        raise ValueError(
            f"local_scale {local_scale} reaches a larger area than "
            f"global_scale {global_scale}; local views must cover LESS of "
            f"the clip than the global view."
        )
    if len(aspect_ratio_range) != 2 or not 0.0 < aspect_ratio_range[0] <= aspect_ratio_range[1]:
        raise ValueError(
            f"aspect_ratio_range must be an increasing pair of positive "
            f"numbers, got {aspect_ratio_range}"
        )
    for name, value in (
        ("flip_prob", flip_prob),
        ("color_jitter_prob", color_jitter_prob),
        ("grayscale_prob", grayscale_prob),
        ("blur_prob", blur_prob),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if brightness < 0.0 or contrast < 0.0:
        raise ValueError(f"brightness ({brightness}) and contrast ({contrast}) must be >= 0")
    if len(blur_sigma_range) != 2 or not 0.0 < blur_sigma_range[0] <= blur_sigma_range[1]:
        raise ValueError(
            f"blur_sigma_range must be an increasing pair of positive "
            f"numbers, got {blur_sigma_range}"
        )

    blur_radius = max(1, int(round(2.0 * blur_sigma_range[1])))

    logger.info(
        f"multi-crop video map fn: 1 global + {local_crops_number} local views "
        f"at {crop_size}x{crop_size}x{num_frames}, global_scale={global_scale}, "
        f"local_scale={local_scale}, blur_radius={blur_radius}"
    )

    def map_fn(
        inputs: Dict[str, tf.Tensor], label: Any = None
    ) -> Tuple[Dict[str, tf.Tensor], Any]:
        """Map one unbatched ``{"pixels": (T, H, W, C)}`` element."""
        clip = tf.cast(inputs["pixels"], tf.float32)
        channels = clip.shape[-1]

        global_frame = _augment_clip_view(
            clip, crop_size, is_global=True,
            global_scale=global_scale, local_scale=local_scale,
            aspect_ratio_range=aspect_ratio_range, flip_prob=flip_prob,
            color_jitter_prob=color_jitter_prob, brightness=brightness,
            contrast=contrast, grayscale_prob=grayscale_prob,
            blur_prob=blur_prob, blur_sigma_range=blur_sigma_range,
            blur_radius=blur_radius,
        )
        global_frame = tf.ensure_shape(
            global_frame, (num_frames, crop_size, crop_size, channels)
        )

        local_views = [
            _augment_clip_view(
                clip, crop_size, is_global=False,
                global_scale=global_scale, local_scale=local_scale,
                aspect_ratio_range=aspect_ratio_range, flip_prob=flip_prob,
                color_jitter_prob=color_jitter_prob, brightness=brightness,
                contrast=contrast, grayscale_prob=grayscale_prob,
                blur_prob=blur_prob, blur_sigma_range=blur_sigma_range,
                blur_radius=blur_radius,
            )
            for _ in range(local_crops_number)
        ]
        if local_crops_number > 0:
            local_frames = tf.stack(local_views, axis=0)
        else:
            local_frames = tf.zeros(
                (0, num_frames, crop_size, crop_size, channels), dtype=tf.float32
            )
        local_frames = tf.ensure_shape(
            local_frames,
            (local_crops_number, num_frames, crop_size, crop_size, channels),
        )

        return {"global_frame": global_frame, "local_frames": local_frames}, label

    return map_fn

# ---------------------------------------------------------------------
