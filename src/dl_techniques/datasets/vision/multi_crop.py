"""DINO multi-crop ``tf.data`` transform — 2 global + N local views per sample.

This module supplies the DATA-SIDE half of DINO self-distillation. It exists so
that the multi-crop augmentation reaches the model as ONE fixed-shape tensor,
which is what lets DINO train under a stock ``model.compile(...)`` +
``model.fit(ds)`` with no ``train_step`` override anywhere.

Data contract
-------------
``make_multi_crop_map_fn`` returns a function mappable over a ``tf.data``
dataset of ``(image, label)`` pairs (the shape
``src/train/energy_transformer/common.py``'s ``build_raw_image_dataset``
produces), yielding the 2-tuple element::

    (views, label)

    views  (n_views, S, S, C)  float32   n_views = 2 + n_local_crops
    label  (as received)                 PASSED THROUGH UNCHANGED

**Views ``0`` and ``1`` are the GLOBAL crops**; views ``2 ...`` are the local
crops. That ordering, and the number of global views, are the contract stated
in ``src/dl_techniques/models/dino/dino_training.py`` — batching this element
gives exactly the ``(batch, n_views, H, W, C)`` tensor
``DINOTrainingModel.call`` documents. ``N_GLOBAL_VIEWS`` is re-declared here
rather than imported, so that a ``tf.data`` module does not pull in the whole
Keras model package; ``tests/test_datasets/test_multi_crop.py`` asserts the two
constants are EQUAL, so they cannot drift silently.

The ``label`` is passed through for pipeline compatibility only. ``DINOLoss``
IGNORES ``y_true`` — see ``src/dl_techniques/losses/dino_loss.py``.

Local crops at global RESOLUTION (plan decision D-002)
------------------------------------------------------
In the paper, local views are rendered at a *smaller pixel resolution* (96 vs
224), which is where multi-crop's compute saving comes from. Here they are
rendered at the SAME pixel resolution as the global views: a smaller AREA of
the source image is cropped and then resized UP to ``global_crop_size``.

Why: one backbone, one positional-embedding table, and ``tf.data``'s
fixed-shape batching then serve every view. A genuinely smaller local
resolution changes the patch-grid length, which requires interpolating the
positional-embedding table — deliberately not implemented (it is named as
backlog item 1 in ``src/dl_techniques/models/dino/README.md``).

**The cost is real and is not hidden**: local crops are exactly as expensive as
global ones, in both compute and activation memory. ``local_crop_size !=
global_crop_size`` therefore raises ``NotImplementedError`` naming
positional-embedding interpolation, rather than silently mis-shaping a batch.

"AREA of the source image" means the SOURCE THIS FUNCTION RECEIVES
------------------------------------------------------------------
This transform runs as ``build_raw_image_dataset``'s ``element_map_fn``, and
that pipeline resizes every record to ``(image_size, image_size)`` in its
``_decode`` step BEFORE the map fn runs (order: ``_decode`` -> ``shuffle`` ->
``_normalize`` -> ``element_map_fn`` -> ``batch``). The trainer passes
``image_size = global_crop_size``. So the "source image" cropped here is already
a ``global_crop_size``-square, aspect-distorted thumbnail — NOT the original
record — and a local crop is an UPSAMPLE of a small piece of it.

MEASURED at the smoke scale (``global_crop_size=96``, ``local_scale=(0.05,
0.4)``, ``aspect_ratio_range=(3/4, 4/3)``, 2000 draws)::

    global views: crop side 53-96 px  (mean 79)  -> upsampled 1.23x mean
    local  views: crop side 19-69 px  (mean 44)  -> upsampled 2.33x mean,
                                                    4.50x worst case

At ``patch_size=16`` a patch of the most extreme local view therefore covers
about 3.5 distinct source pixels, and since 8 of the 10 (teacher, student) loss
pairs at ``n_local_crops=4`` carry a local student view, most of the objective
at that scale is computed on heavily interpolated content. **This is a candidate
explanation for a weak representation result, alongside the usual scale
arguments (``out_dim``, epochs, dataset size).**

The remedy that does NOT require restructuring the shared pipeline is to decode
the records at a HIGHER resolution than the crop size, so the crop is taken from
a larger image and then resized DOWN. ``train_dino.py``'s ``source_image_size``
config field does exactly that (its default preserves the behaviour measured
above; see that field's documentation for why the default was not moved).

Augmentations: what is implemented, and what is NOT
---------------------------------------------------
The paper's recipe is RandomResizedCrop, horizontal flip, colour jitter
(brightness / contrast / saturation / hue), random grayscale, Gaussian blur and
solarization (the last two applied ASYMMETRICALLY across the two global views).

IMPLEMENTED here:

* **RandomResizedCrop** — per-view area scale + aspect ratio, then a bilinear
  resize to ``global_crop_size``. Single-attempt with clipping (see
  ``_random_resized_crop``), not torchvision's 10-try rejection loop.
* **Random horizontal flip.**
* **Colour jitter — BRIGHTNESS and CONTRAST only** (additive offset; scaling
  about the per-view per-channel mean). Both are range-agnostic.
* **Random grayscale**, as the unweighted mean over the channel axis.
* **Gaussian blur**, separable, with reflect padding, applied with a
  per-view-role probability so the two global views are asymmetric.

NOT implemented, deliberately, and named rather than quietly dropped:

* **Saturation and hue jitter.** ``tf.image.adjust_saturation`` /
  ``adjust_hue`` go through an RGB<->HSV conversion that assumes values in
  ``[0, 1]``. This transform runs AFTER ``build_raw_image_dataset``'s
  per-channel mean/std normalization, so its input is not in ``[0, 1]`` and
  those two ops would produce meaningless colours rather than a jitter.
* **Solarization.** Same reason: it inverts values above a threshold expressed
  in the image's value domain, which is not known here. In the paper it is the
  second half of the global-view asymmetry; here only Gaussian blur carries
  that asymmetry.
* Because the input is normalized rather than in ``[0, 1]``, "grayscale" here
  is the mean of the NORMALIZED channels, not a luma-weighted grayscale of the
  original image. It still removes colour information, which is the point of
  the augmentation, but it is not the paper's exact operator.

This is a weaker augmentation than the paper's. It is written down here so that
nobody reads the module name and infers the full recipe.

``seed`` reproduces a SERIAL map only (MEASURED)
------------------------------------------------
``seed`` seeds ONE module-level ``tf.random.Generator`` shared by every element,
i.e. it seeds a STREAM, not each element independently. Under
``ds.map(fn, num_parallel_calls=...)`` — which is what the shipped trainer uses
(``build_raw_image_dataset`` passes ``tf.data.AUTOTUNE``) — several elements read
that one generator concurrently, so which element gets which slice of the stream
varies run to run.

MEASURED at HEAD, same seed (777), same 8 source images, ``global_crop_size=96``,
``n_local_crops=4``::

    serial   .map(fn)                    identical = True    maxdiff 0.0000
    parallel .map(fn, AUTOTUNE) x2       identical = False   maxdiff 1.5312
    serial vs parallel                   identical = False   maxdiff 1.5908

**So two runs of ``train_dino.py --seed 42`` see DIFFERENT augmentation streams.**
The seed still reproduces weight initialization, shuffling order and every other
``set_seeds``-governed source; it does not reproduce this transform's randomness
in the trainer's configuration. Any A/B that assumes bit-identical data between
two runs is unsound.

Making this a real guarantee needs stateless per-element randomness
(``tf.random.stateless_uniform`` keyed on a per-element counter), and the counter
has to come from the pipeline: ``ds.enumerate()`` placed AFTER ``repeat()``, so
that the same source image draws a different augmentation each epoch. That is a
change to ``build_raw_image_dataset``'s pipeline shape, which is shared by every
``src/train/`` consumer — recorded as backlog item 3 in
``src/dl_techniques/models/dino/README.md`` rather than done here.
``tests/test_datasets/test_multi_crop.py`` pins BOTH halves: the serial guarantee
that is real, and the stream-not-per-element mechanism that scopes it.

Raw TensorFlow ops are used deliberately: this is a ``tf.data`` transform, not
a Keras layer, so ``keras.ops`` purity does not apply (the same rule and the
same wording as ``src/dl_techniques/datasets/vision/masked_patches.py``). No
model forward path is involved.
"""

import math
from typing import Any, Callable, Optional, Tuple

import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# DINO always uses exactly two global crops. This MUST equal
# `dl_techniques.models.dino.dino_training.N_GLOBAL_VIEWS`; the equality is
# asserted in tests/test_datasets/test_multi_crop.py rather than created by an
# import, so that this tf.data module stays free of the Keras model package.
N_GLOBAL_VIEWS = 2


# ---------------------------------------------------------------------
# augmentation primitives (module-private -- see the "earned abstraction" note
# at the bottom of this docstring block)
# ---------------------------------------------------------------------
#
# DECISION plan-2026-08-01T105809-dc0c402e/D-025
# These are NOT promoted into `src/train/common/augment.py`. That module holds
# `augment_patch` / `augment_pair`, shared by the bfunet / cliffordnet denoiser
# family, whose augmentation is flips plus a random 90-degree rotation on a
# fixed-size patch. It has no plausible use for an area-scale random resized
# crop, for per-view-role blur probabilities, or for jitter expressed in a
# normalized value domain -- and `augment_patch` itself is NOT reused here,
# because its vertical flip and 90-degree rotation are not part of DINO's
# recipe. With no second concrete call site, promoting these would be a
# speculative abstraction (`references/complexity-control.md`).


def _random_resized_crop(
    image: tf.Tensor,
    out_size: int,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    rng: tf.random.Generator,
) -> tf.Tensor:
    """Crop a random area/aspect region and resize it to ``out_size``.

    Interface contract:
        Parameters:
            image: ``(H, W, C)`` float32 tensor.
            out_size: Side length of the returned square view.
            scale: ``(min, max)`` fraction of the source AREA to crop.
            ratio: ``(min, max)`` aspect ratio (width / height) of the crop.
            rng: The map fn's ``tf.random.Generator``; all randomness here
                comes from it, so a seeded pipeline is reproducible.
        Returns:
            ``(out_size, out_size, C)`` float32.
        Failure mode:
            None at runtime — the crop box is clipped into the image, so no
            configuration produces an out-of-bounds crop.

    Single attempt with CLIPPING, not torchvision's 10-try rejection loop: a
    ``while`` loop with a data-dependent trip count is exactly the shape that
    breaks under ``tf.data`` tracing. The consequence is that extreme
    ``scale``/``ratio`` combinations get clipped rather than resampled, so the
    achieved area distribution is slightly compressed against the top of
    ``scale``. That biases the crop towards the requested area range; it never
    inverts the ordering between two disjoint ranges, which is the property the
    global-vs-local distinction actually depends on.
    """
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    area = height * width

    target_area = area * rng.uniform([], scale[0], scale[1], dtype=tf.float32)
    log_ratio = rng.uniform(
        [], math.log(ratio[0]), math.log(ratio[1]), dtype=tf.float32)
    aspect = tf.exp(log_ratio)

    crop_w = tf.sqrt(target_area * aspect)
    crop_h = tf.sqrt(target_area / aspect)
    crop_w = tf.clip_by_value(tf.round(crop_w), 1.0, width)
    crop_h = tf.clip_by_value(tf.round(crop_h), 1.0, height)

    offset_w = rng.uniform([], 0.0, 1.0, dtype=tf.float32) * (width - crop_w)
    offset_h = rng.uniform([], 0.0, 1.0, dtype=tf.float32) * (height - crop_h)

    crop = tf.image.crop_to_bounding_box(
        image,
        offset_height=tf.cast(offset_h, tf.int32),
        offset_width=tf.cast(offset_w, tf.int32),
        target_height=tf.cast(crop_h, tf.int32),
        target_width=tf.cast(crop_w, tf.int32),
    )
    return tf.image.resize(crop, (out_size, out_size), method="bilinear")


def _gaussian_blur(
    image: tf.Tensor, sigma: tf.Tensor, radius: int
) -> tf.Tensor:
    """Separable Gaussian blur with reflect padding.

    Interface contract:
        Parameters:
            image: ``(H, W, C)`` float32, with a STATIC channel dimension.
            sigma: Scalar float32 tensor, ``> 0``.
            radius: Static half-width of the kernel; the kernel is
                ``2 * radius + 1`` wide.
        Returns:
            ``(H, W, C)`` float32, the blurred image.
        Failure mode:
            ``ValueError`` at trace time if the channel dimension is dynamic
            (the depthwise filter needs it statically).

    Reflect padding rather than ``padding="SAME"``: zero padding in a
    mean/std-NORMALIZED value domain is not "black", it is the dataset mean, so
    a SAME-padded blur would pull every border pixel toward the mean and put a
    visible frame on every blurred view.
    """
    channels = image.shape[-1]
    if channels is None:
        raise ValueError(
            "_gaussian_blur needs a static channel dimension, got "
            f"{image.shape}"
        )

    offsets = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    weights = tf.exp(-tf.square(offsets) / (2.0 * tf.square(sigma)))
    weights = weights / tf.reduce_sum(weights)

    kernel_h = tf.tile(
        tf.reshape(weights, (2 * radius + 1, 1, 1, 1)), (1, 1, channels, 1))
    kernel_w = tf.tile(
        tf.reshape(weights, (1, 2 * radius + 1, 1, 1)), (1, 1, channels, 1))

    padded = tf.pad(
        image[tf.newaxis],
        [[0, 0], [radius, radius], [radius, radius], [0, 0]],
        mode="REFLECT",
    )
    blurred = tf.nn.depthwise_conv2d(
        padded, kernel_h, strides=[1, 1, 1, 1], padding="VALID")
    blurred = tf.nn.depthwise_conv2d(
        blurred, kernel_w, strides=[1, 1, 1, 1], padding="VALID")
    return blurred[0]


def _maybe(
    condition_prob: float,
    rng: tf.random.Generator,
    transform: Callable[[], tf.Tensor],
    identity: tf.Tensor,
) -> tf.Tensor:
    """Apply ``transform`` with probability ``condition_prob``.

    Interface contract:
        Parameters:
            condition_prob: Probability in ``[0, 1]``. ``0.0`` and ``1.0`` are
                resolved at TRACE time (no random draw, no ``tf.cond``), so a
                disabled augmentation costs nothing and cannot perturb the
                random stream — which is what makes the seeded-determinism and
                the crop-only tests comparable across configurations.
            rng: The map fn's generator.
            transform: Zero-argument callable returning the transformed tensor.
            identity: The tensor to return when the draw fails.
        Returns:
            Either ``transform()`` or ``identity``, same shape and dtype.
        Failure mode:
            None.
    """
    if condition_prob <= 0.0:
        return identity
    if condition_prob >= 1.0:
        return transform()
    draw = rng.uniform([], 0.0, 1.0, dtype=tf.float32)
    return tf.cond(draw < condition_prob, transform, lambda: identity)


# ---------------------------------------------------------------------


def make_multi_crop_map_fn(
    global_crop_size: int,
    *,
    local_crop_size: Optional[int] = None,
    n_local_crops: int = 4,
    global_scale: Tuple[float, float] = (0.4, 1.0),
    local_scale: Tuple[float, float] = (0.05, 0.4),
    aspect_ratio_range: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    flip_prob: float = 0.5,
    color_jitter_prob: float = 0.8,
    brightness: float = 0.4,
    contrast: float = 0.4,
    grayscale_prob: float = 0.2,
    global_blur_probs: Tuple[float, float] = (1.0, 0.1),
    local_blur_prob: float = 0.5,
    blur_sigma_range: Tuple[float, float] = (0.1, 2.0),
    seed: Optional[int] = None,
) -> Callable[..., Tuple[tf.Tensor, Any]]:
    """Build the ``tf.data`` map function for DINO multi-crop pretraining.

    The returned function maps one ``(image, label)`` pair to
    ``(views, label)``, where ``views`` has shape
    ``(2 + n_local_crops, global_crop_size, global_crop_size, C)`` and views 0
    and 1 are the global crops. See the module docstring for the full contract,
    for the D-002 same-resolution rule, and for the list of paper augmentations
    that are deliberately NOT implemented here.

    All configuration is validated EAGERLY, at construction time — a bad scale
    range must fail when the pipeline is built, not silently produce degenerate
    crops a thousand steps into training.

    Args:
        global_crop_size: Side length of EVERY returned view, global and local
            alike (D-002).
        local_crop_size: Must be ``None`` or equal to ``global_crop_size``. Any
            other value raises ``NotImplementedError`` — see Raises.
        n_local_crops: Number of local views per sample, ``>= 0``. Total views
            is ``N_GLOBAL_VIEWS + n_local_crops``.
        global_scale: ``(min, max)`` fraction of the source area for the two
            global crops. Paper: ``(0.4, 1.0)``.
        local_scale: ``(min, max)`` fraction of the source area for the local
            crops. Paper: ``(0.05, 0.4)``. Must not exceed ``global_scale``'s
            maximum — the point of multi-crop is that locals see LESS.
        aspect_ratio_range: ``(min, max)`` width/height ratio of the crop box.
        flip_prob: Probability of a horizontal flip, per view.
        color_jitter_prob: Probability of applying brightness+contrast jitter,
            per view.
        brightness: Half-width of the uniform additive brightness offset, in
            the input's (normalized) value units.
        contrast: Half-width of the uniform contrast factor around ``1.0``;
            the view is scaled about its own per-channel mean.
        grayscale_prob: Probability of collapsing the channels to their mean.
        global_blur_probs: ``(p_view0, p_view1)`` Gaussian-blur probabilities
            for the two global views. Asymmetric on purpose (paper: 1.0 and
            0.1); this is the only global-view asymmetry implemented, since
            solarization is not.
        local_blur_prob: Gaussian-blur probability for every local view.
        blur_sigma_range: ``(min, max)`` Gaussian sigma, in pixels.
        seed: Optional seed for the module-level ``tf.random.Generator``.
            **Reproduces a SERIAL ``.map(fn)`` ONLY.** It seeds one shared
            STREAM, not each element, so under
            ``.map(fn, num_parallel_calls=...)`` — the shipped trainer's
            configuration — the element-to-draw assignment varies run to run
            and the outputs are NOT reproducible (MEASURED: serial
            identical=True, parallel identical=False, maxdiff 1.5312). See the
            module docstring's "``seed`` reproduces a SERIAL map only" section.
            Leave ``None`` for an explicitly non-deterministic stream.

    Returns:
        A ``tf.data``-mappable callable ``(image, label) -> (views, label)``.
        It also accepts a bare ``image``, in which case the returned label is
        an ``int32`` zero scalar.

    Raises:
        NotImplementedError: If ``local_crop_size`` is given and differs from
            ``global_crop_size``. Rendering local views at a smaller pixel
            resolution changes the patch-grid length and therefore requires
            positional-embedding interpolation, which this repository does not
            implement (D-002; backlog item 1 in
            ``src/dl_techniques/models/dino/README.md``).
        ValueError: If ``global_crop_size`` is not positive; if
            ``n_local_crops`` is negative; if either scale range is not an
            increasing pair inside ``(0, 1]``; if ``local_scale``'s maximum
            exceeds ``global_scale``'s maximum; if ``aspect_ratio_range`` is
            not an increasing pair of positive numbers; if any probability is
            outside ``[0, 1]``; if ``brightness``/``contrast`` are negative; or
            if ``blur_sigma_range`` is not an increasing pair of positive
            numbers.
    """
    if global_crop_size <= 0:
        raise ValueError(
            f"global_crop_size must be positive, got {global_crop_size}")

    if local_crop_size is not None and local_crop_size != global_crop_size:
        raise NotImplementedError(
            f"local_crop_size ({local_crop_size}) differs from "
            f"global_crop_size ({global_crop_size}). Rendering local views at "
            f"a smaller pixel resolution changes the patch-grid length, which "
            f"requires positional-embedding interpolation in the backbone. "
            f"That interpolation is NOT implemented (plan decision D-002; "
            f"named as backlog item 1 in "
            f"src/dl_techniques/models/dino/README.md). Local views are "
            f"rendered at the SAME resolution as global views by cropping a "
            f"smaller AREA (see local_scale) and resizing up, so pass "
            f"local_crop_size=None or local_crop_size=global_crop_size."
        )

    if n_local_crops < 0:
        raise ValueError(
            f"n_local_crops must be >= 0, got {n_local_crops}")

    for name, rng_pair in (("global_scale", global_scale),
                           ("local_scale", local_scale)):
        if len(rng_pair) != 2 or not 0.0 < rng_pair[0] <= rng_pair[1] <= 1.0:
            raise ValueError(
                f"{name} must be an increasing (min, max) pair inside (0, 1], "
                f"got {rng_pair}"
            )
    if local_scale[1] > global_scale[1]:
        raise ValueError(
            f"local_scale {local_scale} reaches a larger area than "
            f"global_scale {global_scale}; multi-crop requires the local views "
            f"to cover LESS of the image than the global views, otherwise the "
            f"local/global distinction is decorative."
        )

    if (len(aspect_ratio_range) != 2
            or not 0.0 < aspect_ratio_range[0] <= aspect_ratio_range[1]):
        raise ValueError(
            f"aspect_ratio_range must be an increasing pair of positive "
            f"numbers, got {aspect_ratio_range}"
        )

    probabilities = {
        "flip_prob": flip_prob,
        "color_jitter_prob": color_jitter_prob,
        "grayscale_prob": grayscale_prob,
        "local_blur_prob": local_blur_prob,
        "global_blur_probs[0]": global_blur_probs[0],
        "global_blur_probs[1]": global_blur_probs[1],
    }
    for name, value in probabilities.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if len(global_blur_probs) != N_GLOBAL_VIEWS:
        raise ValueError(
            f"global_blur_probs must hold exactly {N_GLOBAL_VIEWS} values "
            f"(one per global view), got {global_blur_probs}"
        )

    if brightness < 0.0 or contrast < 0.0:
        raise ValueError(
            f"brightness ({brightness}) and contrast ({contrast}) must be "
            f">= 0"
        )
    if (len(blur_sigma_range) != 2
            or not 0.0 < blur_sigma_range[0] <= blur_sigma_range[1]):
        raise ValueError(
            f"blur_sigma_range must be an increasing pair of positive "
            f"numbers, got {blur_sigma_range}"
        )

    n_views = N_GLOBAL_VIEWS + n_local_crops
    # Kernel half-width covering +-2 sigma at the widest sigma, as an ODD
    # kernel. Static, because a depthwise filter's shape cannot depend on the
    # sampled sigma.
    blur_radius = max(1, int(round(2.0 * blur_sigma_range[1])))

    # DECISION plan-2026-08-01T105809-dc0c402e/D-035
    # This ONE shared stateful generator is DOCUMENTED-not-fixed, deliberately.
    # Do NOT "fix" the parallel-map non-determinism by (i) hashing the image
    # content into a stateless key -- that makes every epoch replay the SAME
    # augmentation for the same image, which is worse than a non-reproducible
    # stream for SSL; or (ii) calling `rng.reset_from_seed(seed)` per element --
    # every element then gets the IDENTICAL augmentation, which is not
    # augmentation at all (that arm was executed as a RED proof and it fires).
    # A correct stateless form needs a per-element counter from
    # `ds.enumerate()` placed AFTER `repeat()`, i.e. a change to
    # `build_raw_image_dataset`'s pipeline shape, which every `src/train/`
    # consumer shares. Measurement and scope are in the module docstring's
    # "``seed`` reproduces a SERIAL map only" section.
    rng = (
        tf.random.Generator.from_seed(seed)
        if seed is not None
        else tf.random.Generator.from_non_deterministic_state()
    )

    logger.info(
        f"multi-crop map fn: {n_views} views ({N_GLOBAL_VIEWS} global + "
        f"{n_local_crops} local) at {global_crop_size}x{global_crop_size}, "
        f"global_scale={global_scale}, local_scale={local_scale}, "
        f"blur_radius={blur_radius}, seed={seed}"
    )

    def _blur_prob(view_index: int) -> float:
        if view_index < N_GLOBAL_VIEWS:
            return float(global_blur_probs[view_index])
        return float(local_blur_prob)

    def _augment_view(image: tf.Tensor, view_index: int) -> tf.Tensor:
        """Crop + photometrically augment ONE view. `view_index` is static."""
        is_global = view_index < N_GLOBAL_VIEWS
        view = _random_resized_crop(
            image,
            out_size=global_crop_size,
            scale=global_scale if is_global else local_scale,
            ratio=aspect_ratio_range,
            rng=rng,
        )

        view = _maybe(
            flip_prob, rng,
            lambda: tf.image.flip_left_right(view),
            view,
        )

        def _jitter() -> tf.Tensor:
            offset = rng.uniform([], -brightness, brightness, tf.float32)
            factor = rng.uniform([], 1.0 - contrast, 1.0 + contrast, tf.float32)
            mean = tf.reduce_mean(view, axis=(0, 1), keepdims=True)
            return (view - mean) * factor + mean + offset

        view = _maybe(color_jitter_prob, rng, _jitter, view)

        view = _maybe(
            grayscale_prob, rng,
            lambda: tf.tile(
                tf.reduce_mean(view, axis=-1, keepdims=True),
                (1, 1, tf.shape(view)[-1]),
            ),
            view,
        )

        def _blur() -> tf.Tensor:
            sigma = rng.uniform(
                [], blur_sigma_range[0], blur_sigma_range[1], tf.float32)
            return _gaussian_blur(view, sigma, blur_radius)

        view = _maybe(_blur_prob(view_index), rng, _blur, view)
        return view

    def map_fn(image: tf.Tensor, *rest: Any) -> Tuple[tf.Tensor, Any]:
        """Map one raw image to the DINO multi-crop training element."""
        image = tf.cast(image, tf.float32)
        channels = image.shape[-1]

        # Python loop over a STATIC view count -- nothing here reads a tensor
        # value, so the whole body traces into a straight-line graph.
        views = [_augment_view(image, index) for index in range(n_views)]
        stacked = tf.stack(views, axis=0)

        # Static shape: tf.data needs it for a well-defined batch spec, and
        # DINOTrainingModel.build() refuses anything else.
        stacked = tf.ensure_shape(
            stacked, (n_views, global_crop_size, global_crop_size, channels))

        label = rest[0] if rest else tf.zeros((), dtype=tf.int32)
        return stacked, label

    return map_fn

# ---------------------------------------------------------------------
