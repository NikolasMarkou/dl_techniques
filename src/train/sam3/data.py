"""The synthetic text-prompted detection source for the SAM 3 trainer.

What the task is, and why it has DISTRACTORS
--------------------------------------------
Every image carries instances of SEVERAL shape categories drawn from a closed
set (:data:`CATEGORIES`). The prompt names exactly ONE of them, and the targets
are that category's instances **only**. The other categories are drawn into the
same image and are NOT targets.

That is the whole point. Without distractors a model can ignore the text channel
entirely, emit every object it sees, and score perfectly -- which would make the
first learnability claim measure "can this architecture find blobs", not "can
this architecture find the blobs a phrase names". The distractors are what make
the text channel load-bearing, and therefore what makes step 7's claim mean
anything. For the same reason the instance COLOUR is drawn at random,
independent of the category: shape is the only cue that separates a target from
a distractor, so a colour shortcut cannot be learned instead.

The zero-instance case is a FIRST-CLASS SAMPLE, not a unit-test fixture
-----------------------------------------------------------------------
Each image draws 2 or 3 of the 4 categories, so at least one category is always
absent. With probability ``zero_instance_rate`` the prompt names an ABSENT
category: the emitted target then has zero valid rows, ``keep_loss`` derives to
``0.0``, classification is fully gated off for that row and ``presence_loss``
supervises the negative. That path is therefore exercised by the REAL pipeline
on every epoch, which is what makes the presence head's signal real rather than
an artifact of a hand-built fixture.

Tokenization: a fixed phrase -> id dict, no tokenizer
-----------------------------------------------------
No tokenizer is wired to ``sam3`` and none is needed here. What a learnability
claim requires is that the same phrase always maps to the same ids and that
different phrases map to DIFFERENT ids -- i.e. that the text channel is a real
distinguishing signal. :data:`WORD_TO_ID` is a fixed, deterministically-ordered
dict; id ``0`` is reserved for padding and :func:`encode_phrase` returns the
matching ``token_padding_mask``. Supplying that mask is not optional politeness:
``Sam3Image`` never derives one and defaults to "every token is valid", so a
producer that omits it silently trains the text tower to attend to padding.
The ``small`` variant's ``vocab_size=512`` / ``context_length=32`` leave the
vocabulary far from binding (decisions.md D-019).

Boxes, masks and ``is_exhaustive``
-----------------------------------
Boxes are normalized ``cxcywh`` (H-5) computed ANALYTICALLY from the sampled
shape extent, never read back off the raster. The test suite recomputes each box
from the emitted RASTERIZED mask and compares -- an oracle derived from the
drawn geometry rather than from this module's own expression, so a rasterization
or placement bug is visible instead of cancelling out. Masks are binary at the
model's own mask grid. ``is_exhaustive`` is ``1``: this generator draws every
instance in the image itself, so nothing of the prompted category exists that is
not in the target list -- the annotation is complete BY CONSTRUCTION, which is
exactly what that flag claims (decisions.md D-010).

Why COCO is not here
---------------------
COCO 2017 is on disk and reachable, and it is DEFERRED with a measured reason:
18 s/epoch against 1 s for a synthetic source, ~6 % GPU utilization, because
``tf.data.Dataset.from_generator`` rebuilds the ``pycocotools`` index every
epoch. A COCO I/O problem must never be able to masquerade as a model problem --
that is the entire reason a FIRST learnability claim runs on synthetic data.

Nothing is written to disk
---------------------------
Every image, mask and token sequence is generated in memory, per epoch. There is
no cache file, no TFRecord, no download. ``tests/test_train/test_sam3/
test_data.py`` asserts it by iterating a dataset inside an empty temporary
working directory and checking that the directory is still empty.

``import tensorflow`` is SANCTIONED in this file
-------------------------------------------------
This is a ``tf.data`` pipeline, not a forward path, so it is exempt from the
``keras.ops``-only invariant exactly as ``src/train/sam/data.py:36`` records for
its own module. The packing itself is delegated to
``models/sam3/training_model.py``'s ``pack_targets``, so no channel index is
spelled here either.
"""

from typing import Any, Dict, Iterator, List, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf

from dl_techniques.models.sam3.training_model import pack_targets
from dl_techniques.utils.logger import logger

#: The closed category set. The four are separated by two coarse statistics that
#: survive downsampling: mask-fill ratio (mask area / box area) and box aspect
#: ratio. MEASURED over 300 draws each, per-instance RANGES not just means:
#:
#: ===========  =========================  ========================
#: category     fill / aspect at 224 px    fill / aspect at 64x64
#: ===========  =========================  ========================
#: ``triangle`` ``[.504,.524]`` / ``1.0``  ``[.502,.679]`` / ``1.0``
#: ``circle``   ``[.753,.797]`` / ``1.0``  ``[.642,.880]`` / ``1.0``
#: ``square``   ``1.000`` / ``[.97,1.04]`` ``[.960,1.00]`` / ``1.0``
#: ``bar``      ``1.000`` / ``[2.29,2.53]`` ``[.972,1.00]`` / ``[2.0,3.0]``
#: ===========  =========================  ========================
#:
#: **At 224 px -- the resolution the model actually classifies from -- the four
#: ranges are DISJOINT with no overlap at all.** At the 64x64 mask grid the
#: ``circle`` and ``triangle`` fill ranges overlap slightly at their extremes
#: (0.642 vs 0.679), which is a property of the SUPERVISION target, not of the
#: cue: a 6-cell instance simply cannot carry a fill ratio to better than a few
#: percent. Stated here rather than hidden behind the means, and
#: ``test_data.py::test_the_categories_are_separable_at_both_resolutions``
#: measures both regimes rather than asserting either.
CATEGORIES: Tuple[str, ...] = ("circle", "square", "triangle", "bar")
#: The prompt phrase per category. Lengths deliberately DIFFER (2 words vs 3) so
#: the padding mask actually varies across the closed set instead of being a
#: constant that a wrong implementation could reproduce by accident.
CATEGORY_PHRASES: Dict[str, str] = {
    "circle": "a circle",
    "square": "a square",
    "triangle": "a triangle",
    "bar": "a wide bar",
}
#: Reserved ids. ``0`` is padding, as the packed-token convention requires.
PAD_ID: int = 0
START_ID: int = 1
END_ID: int = 2
_FIRST_WORD_ID: int = 3
#: The fixed vocabulary. Built from the SORTED unique word set, so it is stable
#: across processes (a dict-iteration order or a hash seed cannot move it).
WORD_TO_ID: Dict[str, int] = {
    word: index + _FIRST_WORD_ID
    for index, word in enumerate(
        sorted({w for phrase in CATEGORY_PHRASES.values()
                for w in phrase.split()}))
}

#: Shape geometry. A `bar` is a wide rectangle; its aspect is what separates it
#: from `square`, whose fill ratio is identical.
_BAR_ASPECT: float = 2.5
_SQUARE_EXTENT: Tuple[float, float] = (0.12, 0.26)
_BAR_HEIGHT: Tuple[float, float] = (0.08, 0.12)
#: Shapes stay this far from the border, so an instance is never clipped and its
#: ANALYTIC box always equals its rasterized extent. A clipped instance would
#: make the box-vs-mask oracle disagree for a reason that is not a defect.
_BORDER_MARGIN: float = 0.02
#: Attempts at a non-overlapping placement before an instance is abandoned.
_PLACEMENT_ATTEMPTS: int = 12
#: Minimum background gap between two instances, as a fraction of the image
#: side. Instances that merely fail to OVERLAP can still TOUCH after rounding to
#: integer pixels, and two touching instances are one connected component: the
#: image-side oracle in ``test_data.py`` then counts them as a single, wrongly
#: classified object. MEASURED: at a zero gap that oracle undercounted on 4 of 4
#: seeds. The gap also removes a real ambiguity from the task itself.
_PLACEMENT_GAP: float = 0.02

#: Record keys the generator emits.
RECORD_IMAGE = "image"
RECORD_TOKENS = "token_ids"
RECORD_TOKEN_MASK = "token_padding_mask"
RECORD_BOXES = "target_boxes"
RECORD_VALID = "target_valid"
RECORD_MASKS = "target_masks"
RECORD_EXHAUSTIVE = "is_exhaustive"
RECORD_CATEGORY = "prompt_category"


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def encode_phrase(category: str,
                  context_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Encode one category's phrase into fixed-length ids plus a padding mask.

    Interface contract: the return is ``(token_ids (L,) int32,
    token_padding_mask (L,) bool)`` with ``L == context_length``. The mask is
    ``True`` AT PADDING -- the key-padding polarity ``Sam3Image.call`` documents,
    which is the OPPOSITE of a keep mask. The map is injective over
    :data:`CATEGORIES` and is pinned as such by test.

    Args:
        category: A member of :data:`CATEGORIES`.
        context_length: The text tower's ``context_length``.

    Returns:
        ``(token_ids, token_padding_mask)``.

    Raises:
        KeyError: If ``category`` is not in :data:`CATEGORY_PHRASES`.
        ValueError: If the phrase does not fit in ``context_length``. Truncating
            silently would let two categories collide on the same ids, which is
            the ONE property this scheme has to guarantee.
    """
    words = CATEGORY_PHRASES[category].split()
    ids = [START_ID] + [WORD_TO_ID[word] for word in words] + [END_ID]
    if len(ids) > context_length:
        raise ValueError(
            f"encode_phrase: '{category}' needs {len(ids)} tokens but "
            f"context_length is {context_length}. Truncation is refused -- it "
            f"could make two categories share an encoding.")
    ids = ids + [PAD_ID] * (context_length - len(ids))
    tokens = np.asarray(ids, dtype="int32")
    return tokens, tokens == PAD_ID


# ---------------------------------------------------------------------------
# Shape drawing
# ---------------------------------------------------------------------------
def _sample_extent(category: str, image_size: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Sample one axis-aligned ``[x1, y1, x2, y2]`` extent, in float pixels."""
    if category == "bar":
        height = float(rng.uniform(*_BAR_HEIGHT)) * image_size
        width = _BAR_ASPECT * height
    else:
        height = float(rng.uniform(*_SQUARE_EXTENT)) * image_size
        width = height
    low = _BORDER_MARGIN * image_size
    x1 = float(rng.uniform(low, image_size - low - width))
    y1 = float(rng.uniform(low, image_size - low - height))
    return np.asarray([x1, y1, x1 + width, y1 + height], dtype="float64")


def _overlaps(extent: np.ndarray, taken: Sequence[np.ndarray],
              gap: float) -> bool:
    """Whether ``extent`` comes within ``gap`` pixels of an already-placed one.

    A strict non-intersection test is NOT enough -- see :data:`_PLACEMENT_GAP`.
    """
    grown = extent + np.asarray([-gap, -gap, gap, gap])
    return any(
        grown[0] < other[2] and other[0] < grown[2]
        and grown[1] < other[3] and other[1] < grown[3]
        for other in taken)


def _rasterize(category: str, extent: np.ndarray,
               image_size: int) -> np.ndarray:
    """Rasterize ONE instance onto its OWN blank canvas.

    Per-instance canvases are what makes an instance mask exist at all: the
    composited image may occlude, the mask never does.

    Args:
        category: A member of :data:`CATEGORIES`.
        extent: ``[x1, y1, x2, y2]`` in float pixels.
        image_size: Side of the square canvas.

    Returns:
        ``(image_size, image_size)`` uint8, 1 inside the shape.

    Raises:
        ValueError: On an unknown category -- a silently blank canvas would be
            supervised as an empty instance.
    """
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    x1, y1, x2, y2 = (int(round(float(v))) for v in extent)
    if category == "circle":
        cv2.ellipse(canvas, ((x1 + x2) // 2, (y1 + y2) // 2),
                    ((x2 - x1) // 2, (y2 - y1) // 2), 0.0, 0, 360, 1, -1)
    elif category in ("square", "bar"):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 1, -1)
    elif category == "triangle":
        vertices = np.asarray(
            [[(x1 + x2) // 2, y1], [x1, y2], [x2, y2]], dtype=np.int32)
        cv2.fillPoly(canvas, [vertices.reshape(-1, 1, 2)], 1)
    else:
        raise ValueError(f"_rasterize: unknown category {category!r}")
    return canvas


def _downsample(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Binarize a full-resolution instance mask onto the model's mask grid."""
    low = cv2.resize(mask.astype("float32"), (cols, rows),
                     interpolation=cv2.INTER_AREA)
    return (low > 0.5).astype("float32")


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------
def synthetic_prompt_samples(
        num_samples: int,
        image_size: int,
        mask_grid: Tuple[int, int],
        context_length: int,
        max_instances: int = 8,
        zero_instance_rate: float = 0.25,
        max_per_category: int = 3,
        seed: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Yield text-prompted detection records drawn from shapes.

    Interface contract: yields ``num_samples`` dicts, each one IMAGE with one
    prompt. Keys: ``image`` ``(S, S, 3)`` float32 in ``[0, 255]``, ``token_ids``
    / ``token_padding_mask`` ``(context_length,)``, ``target_boxes``
    ``(max_instances, 4)`` float32 normalized ``cxcywh``, ``target_valid``
    ``(max_instances,)``, ``target_masks`` ``(max_instances, mh, mw)`` float32
    binary, ``is_exhaustive`` float32 scalar and ``prompt_category`` (a Python
    string, for tests and per-category evaluation -- :func:`build_sam3_dataset`
    drops it). Nothing is written to disk.

    Args:
        num_samples: Records in one epoch.
        image_size: Side of the square image.
        mask_grid: ``(rows, cols)`` of the model's mask output.
        context_length: The text tower's ``context_length``.
        max_instances: ``N_max``, the padded GT slot count.
        zero_instance_rate: Probability that the prompt names a category with
            NO instances in the image. Sampled per record.
        max_per_category: Upper bound on instances drawn per present category.
        seed: Seed for the single ``np.random.Generator`` driving every draw.

    Yields:
        One record dict per image.

    Raises:
        ValueError: If ``max_instances`` cannot hold ``max_per_category``
            instances, or if ``zero_instance_rate`` is outside ``[0, 1]``.
    """
    if max_instances < max_per_category:
        raise ValueError(
            f"max_instances={max_instances} < max_per_category="
            f"{max_per_category}: targets would be silently truncated.")
    if not 0.0 <= zero_instance_rate <= 1.0:
        raise ValueError(
            f"zero_instance_rate={zero_instance_rate} must be in [0, 1].")

    rng = np.random.default_rng(seed)
    rows, cols = int(mask_grid[0]), int(mask_grid[1])
    tokens = {name: encode_phrase(name, context_length) for name in CATEGORIES}
    abandoned = 0

    for _ in range(num_samples):
        # 2 or 3 of 4 categories are drawn, so at least ONE is always absent and
        # the zero-instance prompt is always constructible; and a POSITIVE
        # prompt always has at least one distractor category beside it.
        count = int(rng.integers(2, len(CATEGORIES)))
        present = rng.permutation(len(CATEGORIES))[:count]

        canvas = rng.uniform(0.05, 0.25, (image_size, image_size, 3))
        placed: List[Tuple[str, np.ndarray, np.ndarray]] = []
        taken: List[np.ndarray] = []
        for index in present:
            category = CATEGORIES[int(index)]
            for _ in range(int(rng.integers(1, max_per_category + 1))):
                extent = None
                for _ in range(_PLACEMENT_ATTEMPTS):
                    candidate = _sample_extent(category, image_size, rng)
                    if not _overlaps(candidate, taken,
                                     _PLACEMENT_GAP * image_size):
                        extent = candidate
                        break
                if extent is None:
                    abandoned += 1
                    continue
                taken.append(extent)
                full = _rasterize(category, extent, image_size)
                # The colour is INDEPENDENT of the category on purpose: shape is
                # the only cue separating a target from a distractor.
                canvas[full > 0] = rng.uniform(0.55, 1.0, 3)
                placed.append((category, extent, full))

        drawn = {category for category, _, _ in placed}
        absent = [name for name in CATEGORIES if name not in drawn]
        if not drawn or float(rng.random()) < zero_instance_rate:
            prompt = absent[int(rng.integers(len(absent)))]
        else:
            ordered = sorted(drawn)
            prompt = ordered[int(rng.integers(len(ordered)))]

        boxes = np.zeros((max_instances, 4), dtype="float32")
        valid = np.zeros((max_instances,), dtype="float32")
        masks = np.zeros((max_instances, rows, cols), dtype="float32")
        slot = 0
        for category, extent, full in placed:
            if category != prompt or slot >= max_instances:
                continue
            low = _downsample(full, rows, cols)
            if low.sum() <= 0.0:
                # An instance that vanishes at the mask grid is DROPPED, never
                # kept: a zero mask paired with a real box would break the
                # box-vs-mask agreement for a reason that is not a defect.
                abandoned += 1
                continue
            # The box is ANALYTIC -- from the sampled extent, not read back off
            # the raster. The oracle in test_data.py derives it from the mask.
            boxes[slot] = [
                (extent[0] + extent[2]) * 0.5 / image_size,
                (extent[1] + extent[3]) * 0.5 / image_size,
                (extent[2] - extent[0]) / image_size,
                (extent[3] - extent[1]) / image_size,
            ]
            valid[slot] = 1.0
            masks[slot] = low
            slot += 1

        token_ids, padding_mask = tokens[prompt]
        yield {
            RECORD_IMAGE: (np.clip(canvas, 0.0, 1.0) * 255.0).astype("float32"),
            RECORD_TOKENS: token_ids,
            RECORD_TOKEN_MASK: padding_mask,
            RECORD_BOXES: boxes,
            RECORD_VALID: valid,
            RECORD_MASKS: masks,
            # 1.0 because this generator DREW every instance in the image: no
            # instance of the prompted category exists outside the target list,
            # so the annotation is exhaustive by construction (D-010).
            RECORD_EXHAUSTIVE: np.float32(1.0),
            RECORD_CATEGORY: prompt,
        }

    if abandoned:
        logger.info(
            "synthetic_prompt_samples: abandoned %d instance(s) (no "
            "non-overlapping placement, or empty at the %dx%d mask grid)",
            abandoned, rows, cols)


# ---------------------------------------------------------------------------
# The `tf.data` assembly
# ---------------------------------------------------------------------------
def build_sam3_dataset(
        model: Any,
        num_samples: int,
        batch_size: int,
        max_instances: int = 8,
        zero_instance_rate: float = 0.25,
        max_per_category: int = 3,
        seed: int = 0,
        shuffle_buffer: int = 0,
) -> tf.data.Dataset:
    """Assemble a ``fit()``-consumable ``(inputs, packed_target)`` dataset.

    Interface contract: every geometry is DERIVED from ``model`` -- image side,
    ``context_length``, mask grid, ``include_masks`` and the packed width all
    come off the wrapper, and the packing itself is
    ``training_model.pack_targets``. This function therefore restates no channel
    index and no width; the derivation is checked against
    ``model.packed_target_spec(max_instances)`` before the dataset is returned,
    so a three-way width drift raises here instead of slicing garbage.

    Args:
        model: A built or unbuilt
            :class:`~dl_techniques.models.sam3.training_model.Sam3TrainingModel`.
        num_samples: Records in one epoch.
        batch_size: Batch size, in images.
        max_instances: ``N_max``, the padded GT slot count.
        zero_instance_rate: Probability the prompt names an absent category.
        max_per_category: Upper bound on instances per present category.
        seed: Seed for the source generator and the shuffle.
        shuffle_buffer: If > 0, shuffle with this buffer before batching.

    Returns:
        A batched ``tf.data.Dataset`` of ``(inputs, packed_target)``. The batch
        axis is STATIC.

    Raises:
        ValueError: If the vocabulary overflows the text tower's ``vocab_size``,
            or if the packed target shape disagrees with
            ``model.packed_target_spec``.
    """
    image_size = int(model.sam3.backbone.img_size)
    context_length = int(model.sam3.text_encoder.context_length)
    vocab_size = int(model.sam3.text_encoder.vocab_size)
    rows, cols = (int(model.mask_grid[0]), int(model.mask_grid[1]))
    include_masks = bool(model.include_masks)

    highest = max([END_ID] + list(WORD_TO_ID.values()))
    if highest >= vocab_size:
        raise ValueError(
            f"build_sam3_dataset: the fixed vocabulary reaches id {highest} "
            f"but the text tower has vocab_size={vocab_size}. The embedding "
            f"lookup would be out of range.")

    signature = {
        RECORD_IMAGE: tf.TensorSpec((image_size, image_size, 3), tf.float32),
        RECORD_TOKENS: tf.TensorSpec((context_length,), tf.int32),
        RECORD_TOKEN_MASK: tf.TensorSpec((context_length,), tf.bool),
        RECORD_BOXES: tf.TensorSpec((max_instances, 4), tf.float32),
        RECORD_VALID: tf.TensorSpec((max_instances,), tf.float32),
        RECORD_MASKS: tf.TensorSpec((max_instances, rows, cols), tf.float32),
        RECORD_EXHAUSTIVE: tf.TensorSpec((), tf.float32),
    }

    def _source() -> Iterator[Dict[str, Any]]:
        for record in synthetic_prompt_samples(
                num_samples=num_samples, image_size=image_size,
                mask_grid=(rows, cols), context_length=context_length,
                max_instances=max_instances,
                zero_instance_rate=zero_instance_rate,
                max_per_category=max_per_category, seed=seed):
            # `prompt_category` is a Python string for tests and per-category
            # evaluation; it is not part of the model's input contract.
            yield {key: record[key] for key in signature}

    def _to_training_record(
            record: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], Any]:
        inputs = {
            "image": record[RECORD_IMAGE],
            "token_ids": record[RECORD_TOKENS],
            # Supplied EXPLICITLY: `Sam3Image` derives no padding mask and
            # defaults to "every token is valid", which would train the text
            # tower to attend to padding ids.
            "token_padding_mask": record[RECORD_TOKEN_MASK],
        }
        packed = pack_targets(
            record[RECORD_BOXES], record[RECORD_VALID],
            target_masks=record[RECORD_MASKS] if include_masks else None,
            is_exhaustive=record[RECORD_EXHAUSTIVE],
            include_masks=include_masks)
        return inputs, packed

    dataset = tf.data.Dataset.from_generator(
        _source, output_signature=signature)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, seed=seed)

    # DECISION plan-2026-08-05T124709-6c4fac48/D-023
    # `drop_remainder=True` is a HARD REQUIREMENT, not a throughput choice. Do
    # NOT relax it to keep a partial last batch: MEASURED at this step, a batch
    # axis of `None` reaches `Sam3DualViTDetNeck.call()`, where the positional
    # encoding traces as `(32, 32, None)` -- its CHANNEL axis unknown -- and the
    # neck's own shape check raises `ValueError: positional encoding shape
    # (32, 32, None) must match the feature's (32, 32, 8)` at the FIRST `fit()`
    # step. That is the same failure CLASS as D-068 in the SAM 2 trainer but a
    # different site, and it fires even when the sample count divides the batch
    # size exactly, because the drop flag is what makes the axis static at all.
    # A `fit()` over numpy arrays never sees it (Keras traces those with a
    # static batch), so it is invisible to every hand-built fixture.
    # The batch is also applied BEFORE the map so `pack_targets` -- a
    # `keras.ops` function written for a batch axis, which reads
    # `int(masks.shape[1])` off a static shape -- runs once per batch rather
    # than once per sample. See decisions.md D-023.
    remainder = num_samples % batch_size
    if remainder:
        logger.warning(
            "build_sam3_dataset: %d of %d sample(s) are dropped -- the batch "
            "axis must be STATIC, so the last partial batch cannot be emitted. "
            "Choose a num_samples divisible by batch_size (%d).",
            remainder, num_samples, batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(_to_training_record,
                          num_parallel_calls=tf.data.AUTOTUNE)

    expected = tuple(model.packed_target_spec(max_instances))
    produced = tuple(dataset.element_spec[1].shape[1:])
    if produced != expected:
        raise ValueError(
            f"build_sam3_dataset: packed target shape {produced} disagrees "
            f"with model.packed_target_spec({max_instances}) = {expected}. "
            f"The pipeline, the model and the loss must land on one width.")
    return dataset.prefetch(tf.data.AUTOTUNE)
