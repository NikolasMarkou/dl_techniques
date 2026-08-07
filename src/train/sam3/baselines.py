"""The non-model baseline family SAM 3's ``box_iou`` must be quoted against.

Why this is repository code and not a probe script
--------------------------------------------------
``plans/SYSTEM.md:220`` makes comparison against the family
``{5x5 fixed grid} u {k-means prior, k in 8/16/32}`` a HARD constraint on any
accuracy claim made on this synthetic generator. Until this module existed the
family lived ONLY as scratch inside two gitignored plan directories
(``probe_chance_floor.py`` and ``probe_pass2_arms.py`` of
``plan-2026-08-06T055747-1e650383``), which means the bar vanished with the
plan directory and the comparator's free choices shipped with nothing. Two
independent fits of the SAME k-means prior in that plan already disagreed
enough to FLIP the sign of the seed-1 comparison (0.4150 vs 0.3989), so the
fit's free choices are pinned here as module constants and reported beside
every number.

What it measures, and what it does not
--------------------------------------
Every arm of the NAMED FAMILY (``{5x5 fixed grid} u {k-means prior}``) is a
predictor that reads NO image: one fixed set of ``Q`` boxes emitted unchanged
for every image of the scoring split. That is the point -- SAM 3's boxes were
MEASURED to be image-independent (``val_box_std_across_images`` 6.9e-06), so
the honest comparator is the best image-independent predictor available, not an
untrained network. Nothing here is a chance level on any dataset other than
this generator at ``small`` / ``Q=32``; the grid's score is a property of the
target-size distribution it happens to sit near.

The family is NOT the whole bar -- see the connected-components detector
-----------------------------------------------------------------------
:func:`connected_components_predictor` is deliberately NOT a member of that
family and is deliberately NOT a "prior": it is a zero-parameter, zero-training
PER-IMAGE PREDICTOR that thresholds the canvas and emits its bright blobs'
bounding boxes. It exists because a family with no image-reading member cannot
distinguish "the model learned detection" from "this generator's box task is
solvable by any rule that looks at the pixels" -- and on this generator the
detector reads ~0.94 box IoU, ABOVE the ``step9_qsel`` checkpoint -- deep
supervision + query selection, WITHOUT prompt conditioning -- which reads
0.8450 / 0.8296 / 0.8191 on 3 of 3 seeds. Name the CHECKPOINT and not the
flag: ``step10_pcq`` is also a query-selection arm and reads a different
0.8494 / 0.8223 / 0.8299. Adversarial review of
``plan-2026-08-06T185813-fd80240f`` raised exactly this (CRITICAL 1) and the
measurement is the answer to it: any accuracy claim on this generator is a
WIRING / LEARNABILITY result, not a capability result. :func:`family_max`
excludes the detector on purpose -- ``plans/SYSTEM.md:220`` names the family --
so the two bars stay separately quotable.

The metric a category-blind predictor cannot win
------------------------------------------------
The detector above is the reason ``box_iou`` alone settles nothing on this
generator, and :func:`distractor_gap` is the answer to it: the SAME predicted
boxes are scored twice, against the prompted category's ground truth and
against the pooled ground truth of every OTHER category in the SAME image. A
ground-truth oracle reads 1.0. A predictor that boxes every bright blob matches
both sets about equally, so its gap is NOISE AROUND ZERO whose SIGN is
seed-dependent, not a small positive number: MEASURED on seeds 1/2/3, the
connected-components detector reads +0.004954 / -0.004813 / -0.000371 while
reading ~0.94 raw box_iou. That magnitude is NOT a general ceiling for blind
arms -- the wider category-blind band, including the FIXED-GRID and
KMEANS-PRIOR family priors, reaches an absolute gap of 0.0358 (KMEANS k=8,
seed 3), so 0.02 is not a category-blind ceiling and a small gap of either
sign is not on its own evidence of category selectivity. Unlike every arm
above, this one needs the eval-only all-category export of ``train.sam3.data``
(``include_all_instances``), and it is NOT a member of the family: it never
enters :func:`evaluate_family`'s results, so :func:`family_max` cannot see it.

The scoring path is ``evaluate_sam3``'s, expression for expression
--------------------------------------------------------------------
:func:`score_prior` reuses ``loss.matcher``, ``box_cxcywh_to_xyxy``,
``iou_and_generalized_iou`` and the ``ops.where(is_matched > 0, iou, 0)``
masked sum divided by ``sum(is_matched)`` -- imported from the same modules
``src/train/sam3/train_sam3.py:703-899`` imports them from, never copied. Only
the two lines that ``evaluate_sam3`` spends running the MODEL are absent, for
the obvious reason that there is no model here.

Seed discipline, printed rather than asserted
---------------------------------------------
The prior is fitted on ``build_sam3_dataset(seed=seed)`` (the TRAIN split) and
scored on ``build_sam3_dataset(seed=seed + VAL_SEED_OFFSET)`` (the val split,
the same offset ``create_datasets`` uses). Both seeds are printed side by side
at every fit, so a leak is visible in the OUTPUT rather than promised in a
comment.

Instrument liveness -- not optional
-----------------------------------
A broken IoU instrument reads plausible. Two arms with values known IN ADVANCE
guard against that and are part of the CLI's required output:

* ``ORACLE`` -- each image's own padded GT block tiled into the ``Q`` slots.
  Under a matcher free to pick any query per target it must read EXACTLY 1.0.
* ``DEGENERATE`` -- one centered box repeated ``Q`` times. It reads ~0.025.

If either moves, no other number this module prints may be believed.

Calibration
-----------
Executed before this docstring was written, on GPU 1::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg \\
        .venv/bin/python -m train.sam3.baselines --seeds 1 2 3

reproduces the published k=32 figures 0.4150 / 0.3890 / 0.4111 and the
cross-script fixed-grid control 0.3570 / 0.3314 / 0.3425 on seeds 1 / 2 / 3.
The reproduced values are recorded in the plan's ``baselines_calibration.out``.

Output convention: library functions log through
``dl_techniques.utils.logger`` and never print. The CLI's table is written with
an EXPLICIT stdout writer (:func:`_emit`), because a table piped to ``tee`` is
the artifact and log formatting would corrupt it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import keras
import numpy as np
from keras import ops

from dl_techniques.losses.sam3_detection_loss import (
    box_cxcywh_to_xyxy,
    iou_and_generalized_iou,
    unpack_targets,
)
from dl_techniques.utils.logger import logger
from train.common import set_seeds
from train.sam3.data import (
    RECORD_ALL_BOXES,
    RECORD_ALL_CATEGORY_IDS,
    RECORD_ALL_VALID,
    RECORD_PROMPT_ID,
    build_sam3_dataset,
    distractor_targets,
)
from train.sam3.train_sam3 import (
    Sam3TrainingConfig, create_training_model, matched_box_iou)

#: The offset between a run's TRAIN seed and its VAL seed. Mirrors
#: ``create_datasets`` (`train_sam3.py`), which is what makes the split scored
#: here the split the runs are scored on.
VAL_SEED_OFFSET: int = 10_000

#: `KMeans` free choices, PINNED. The prior plan measured that leaving these
#: unpinned moves the fitted prior by up to 0.016 IoU and can FLIP a per-seed
#: sign, so they are constants with names rather than call-site defaults, and
#: :func:`fit_kmeans_prior` logs both beside every fit.
KMEANS_N_INIT: int = 10

#: `k` values of the named family. `plans/SYSTEM.md:220`.
FAMILY_KS: Tuple[int, ...] = (8, 16, 32)

#: The hand-written grid's geometry. `0.2` here is a HAND-CHOSEN constant with
#: no provenance in any dataset -- deliberately so, since the grid's whole
#: point is to read nothing. It is NOT the measured anchor size used elsewhere
#: in this package; the two are different constants with different origins.
GRID_SIDE: int = 5
GRID_BOX_SIZE: float = 0.2

#: Arm keys. Named constants rather than positions: the two LIVENESS arms are
#: looked up by name in :func:`main` and in the tests, and a dict-ordering
#: assumption is exactly the kind of silent drift this module exists to catch.
ORACLE_ARM: str = "ORACLE (per-image GT tiled to Q)"
DEGENERATE_ARM: str = "DEGENERATE (one centered box)"
GRID_ARM: str = f"FIXED-GRID {GRID_SIDE}x{GRID_SIDE} wh{GRID_BOX_SIZE}"

#: The image-reading predictor. Named so that :func:`family_max`'s
#: ``startswith(("FIXED-GRID", "KMEANS-PRIOR"))`` filter CANNOT pick it up: it
#: is not a member of the `SYSTEM.md:220` family and must never silently become
#: one. It is reported in its own CLI section.
CONNECTED_COMPONENTS_ARM: str = "CONNECTED-COMPONENTS (reads pixels, 0 params)"

#: Intensity threshold, on the generator's own ``[0, 255]`` image scale, above
#: which a pixel is foreground. NOT tuned: `data.py` draws shapes at
#: ``uniform(0.55, 1.0)`` (140..255) on a canvas at ``uniform(0.05, 0.25)``
#: (13..64), so any cut inside (64, 140) separates them and 100 is its middle.
CC_THRESHOLD: float = 100.0

#: The two input keys that carry the TEXT PROMPT and nothing else -- `data.py`
#: emits exactly these beside ``image`` and the targets. Swapping BOTH together
#: is not optional: ``token_padding_mask`` is per-prompt (the four category
#: phrases have different word counts), so swapping the ids alone would feed a
#: prompt under another prompt's mask, which is a third thing and not a swap.
PROMPT_KEYS: Tuple[str, str] = ("token_ids", "token_padding_mask")

#: Batch-axis rotations used to hand every image ANOTHER image's prompt. Three
#: of them, because ONE rotation can land on an image of the same category by
#: chance; :func:`prompt_swap_retention` reports the fraction of images whose
#: prompt tokens actually changed rather than assuming they all did.
PROMPT_SWAP_SHIFTS: Tuple[int, ...] = (1, 2, 3)

#: The label the distractor-gap diagnostic is printed and filed under. Named
#: here for the same reason :data:`CONNECTED_COMPONENTS_ARM` is: it must be
#: impossible for :func:`family_max`'s ``startswith(("FIXED-GRID",
#: "KMEANS-PRIOR"))`` allowlist to pick it up. It is not a member of the
#: ``SYSTEM.md:220`` family, it never enters :func:`evaluate_family`'s results,
#: and it is reported in its own CLI section.
DISTRACTOR_GAP_ARM: str = "DISTRACTOR-GAP (same boxes, other categories' GT)"


def kmeans_arm(k: int) -> str:
    """The arm key of the k-means prior at ``k``.

    Args:
        k: Number of cluster centers.

    Returns:
        The key :func:`evaluate_family` files that arm's result under.
    """
    return f"KMEANS-PRIOR k={k} (train-fit)"

#: The step-8 / step-71 run geometry, field for field. Every published figure
#: this module must reproduce was measured on exactly this split.
SPLIT: Dict[str, Any] = {
    "num_train_samples": 1024,
    "num_val_samples": 64,
    "max_instances": 8,
    "max_per_category": 3,
    "zero_instance_rate": 0.25,
    "variant": "small",
    "include_masks": False,
    "batch_size": 4,
}

#: `(target_boxes, target_valid, num_queries) -> (B, Q, 4)`. The escape hatch
#: that lets an IMAGE-DEPENDENT liveness arm (the GT oracle) run through the
#: same scorer as the image-independent priors.
Predictor = Callable[[np.ndarray, np.ndarray, int], np.ndarray]

#: `(images, target_boxes, target_valid, num_queries) -> (B, Q, 4)`. Same shape
#: contract as :data:`Predictor` with the batch's IMAGES prepended. A callable
#: is dispatched onto this signature by :func:`_score` iff it carries a truthy
#: ``reads_image`` attribute -- an explicit opt-in marker rather than an arity
#: guess, so a predictor whose signature is mistyped fails loudly instead of
#: being scored blind to the pixels.
ImagePredictor = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def _emit(line: str = "") -> None:
    """Write one CLI table line to stdout.

    Explicit rather than ``print``: this module's table is a piped artifact and
    must not be interleaved with the logger's formatting.

    Args:
        line: The text to write. A newline is appended.

    Returns:
        None.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def tile_to_queries(prior: np.ndarray, num_queries: int) -> np.ndarray:
    """Repeat a prior's boxes until it fills ``num_queries`` slots.

    Interface contract: ``prior`` is ``(P, 4)`` with ``P >= 1``; the result is
    ``(num_queries, 4)`` float32, whose row ``i`` is ``prior[i % P]``. Raises
    on an empty prior rather than emitting a zero-length tile.

    Args:
        prior: ``(P, 4)`` cxcywh boxes in [0, 1].
        num_queries: The model's ``Q``.

    Returns:
        ``(num_queries, 4)`` float32.

    Raises:
        ValueError: If ``prior`` is empty or is not ``(P, 4)``.
    """
    boxes = np.asarray(prior, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(
            f"tile_to_queries: prior must be (P, 4) cxcywh; got "
            f"{boxes.shape}")
    if boxes.shape[0] == 0:
        raise ValueError("tile_to_queries: prior is empty; nothing to tile.")
    reps = int(np.ceil(num_queries / boxes.shape[0]))
    return np.tile(boxes, (reps, 1))[:num_queries]


def fixed_grid_prior(side: int = GRID_SIDE,
                     box_size: float = GRID_BOX_SIZE) -> np.ndarray:
    """The hand-written ``side x side`` grid of boxes.

    Interface contract: reads no image, no dataset and no training run --
    ``fixed_grid_prior()`` is a pure function of its two arguments. Centers are
    ``(i + 0.5) / side`` on both axes, in ROW-MAJOR order over ``(cy, cx)``, so
    row ``r * side + c`` is ``((c + 0.5) / side, (r + 0.5) / side, box_size,
    box_size)``. This ordering is the one the published 0.3570 / 0.3314 /
    0.3425 figures were produced under; the score is order-invariant (the
    matcher is free over queries) but the ordering is pinned so the value
    oracle in the tests can hand-compute a row.

    Args:
        side: Grid resolution per axis.
        box_size: Width and height of every box, in normalized units.

    Returns:
        ``(side * side, 4)`` float32 cxcywh in [0, 1].

    Raises:
        ValueError: If ``side < 1`` or ``box_size`` is outside ``(0, 1]``.
    """
    if side < 1:
        raise ValueError(f"fixed_grid_prior: side must be >= 1; got {side}")
    if not 0.0 < box_size <= 1.0:
        raise ValueError(
            f"fixed_grid_prior: box_size must be in (0, 1]; got {box_size}")
    centers = (np.arange(side, dtype=np.float32) + 0.5) / float(side)
    grid: List[List[float]] = []
    for cy in centers:
        for cx in centers:
            grid.append([float(cx), float(cy), box_size, box_size])
    return np.asarray(grid, dtype=np.float32)


def degenerate_prior() -> np.ndarray:
    """One centered box -- the crudest possible non-reader.

    Interface contract: a ``(1, 4)`` prior whose scored value is known in
    advance (~0.025 on this split). It exists as a LIVENESS arm: an instrument
    that reads ~0.35 for this prior is broken.

    Returns:
        ``(1, 4)`` float32 cxcywh.
    """
    return np.asarray([[0.5, 0.5, GRID_BOX_SIZE, GRID_BOX_SIZE]],
                      dtype=np.float32)


def gt_oracle_predictor(target_boxes: np.ndarray,
                        target_valid: np.ndarray,
                        num_queries: int) -> np.ndarray:
    """Each image's OWN padded GT block, tiled into the ``Q`` query slots.

    Interface contract: this is the ONLY image-DEPENDENT arm in this module,
    and it exists purely as a liveness guard -- under a matcher free to pick
    any query per target its box IoU is EXACTLY 1.0, a non-trivial value known
    before the instrument runs. The construction is the PADDED one (the
    ``(B, N_max, 4)`` block including padding rows), which is what the
    published oracle figures used; a valid-only construction scores the same
    1.0 but a different ``box_std_across_images``.

    Args:
        target_boxes: ``(B, N_max, 4)`` cxcywh ground truth.
        target_valid: ``(B, N_max)`` validity flags. Unused -- the padded
            construction tiles the whole block; kept for signature uniformity
            with :data:`Predictor`.
        num_queries: The model's ``Q``.

    Returns:
        ``(B, num_queries, 4)`` float32.
    """
    del target_valid
    reps = int(np.ceil(num_queries / target_boxes.shape[1]))
    return np.tile(target_boxes, (1, reps, 1))[:, :num_queries, :].astype(
        np.float32)


def connected_components_boxes(image: np.ndarray,
                               num_queries: int,
                               threshold: float = CC_THRESHOLD) -> np.ndarray:
    """Bounding boxes of ONE image's brightest connected components.

    Interface contract: a pure function of ``(image, num_queries, threshold)``.
    It reads NO label, NO dataset and NO network -- it is a hand-written
    detector, not a prior, and it is the only thing in this module that looks
    at pixels. It is also category-BLIND: the generator's prompt names one of
    four shape categories and this function cannot tell them apart, so it emits
    every blob it finds. Best-of-Q Hungarian matching is what makes that
    unpenalised, which is itself part of the finding.

    Components are ranked by PIXEL COUNT, descending, and the top
    ``num_queries`` are emitted; if fewer are found the list is tiled (the same
    treatment :func:`tile_to_queries` gives a short prior). An image with no
    foreground at all falls back to one centered box, so the returned shape is
    always ``(num_queries, 4)``.

    Args:
        image: ``(S, S, 3)`` float32 on the generator's ``[0, 255]`` scale.
        num_queries: The model's ``Q``.
        threshold: Foreground cut on that same scale. See :data:`CC_THRESHOLD`.

    Returns:
        ``(num_queries, 4)`` float32 cxcywh in [0, 1].

    Raises:
        ValueError: If ``image`` is not ``(S, S, C)``.
    """
    # `scipy.ndimage`, not `cv2`: scipy is a DECLARED dependency of this repo
    # (`pyproject.toml`), OpenCV is not -- it happens to be installed in this
    # environment, which is not the same thing as being available to a user.
    from scipy import ndimage

    pixels = np.asarray(image, dtype=np.float32)
    if pixels.ndim != 3:
        raise ValueError(
            f"connected_components_boxes: image must be (S, S, C); got "
            f"{pixels.shape}")
    height, width = pixels.shape[0], pixels.shape[1]
    labels, count = ndimage.label(pixels.max(axis=-1) > threshold)
    if count < 1:
        return tile_to_queries(degenerate_prior(), num_queries)
    # `find_objects` returns the per-label bounding SLICES -- the same quantity
    # `cv2.connectedComponentsWithStats` returns as `CC_STAT_*`.
    extents = ndimage.find_objects(labels)
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    boxes: List[List[float]] = []
    for index in np.argsort(-areas)[:num_queries]:
        rows, cols = extents[int(index)][0], extents[int(index)][1]
        boxes.append([
            0.5 * (cols.start + cols.stop) / float(width),
            0.5 * (rows.start + rows.stop) / float(height),
            (cols.stop - cols.start) / float(width),
            (rows.stop - rows.start) / float(height),
        ])
    return tile_to_queries(np.asarray(boxes, dtype=np.float32), num_queries)


def connected_components_predictor(images: np.ndarray,
                                   target_boxes: np.ndarray,
                                   target_valid: np.ndarray,
                                   num_queries: int) -> np.ndarray:
    """:data:`ImagePredictor` wrapper around :func:`connected_components_boxes`.

    Interface contract: reads ONLY ``images``. ``target_boxes`` and
    ``target_valid`` are accepted for signature uniformity and are deleted on
    the first line -- this arm cannot leak the labels it is scored against, and
    that is enforced by the code rather than promised in a comment.

    Args:
        images: ``(B, S, S, 3)`` float32 on the ``[0, 255]`` scale.
        target_boxes: Unused. Deleted.
        target_valid: Unused. Deleted.
        num_queries: The model's ``Q``.

    Returns:
        ``(B, num_queries, 4)`` float32 cxcywh.
    """
    del target_boxes, target_valid
    return np.stack([connected_components_boxes(image, num_queries)
                     for image in np.asarray(images)]).astype(np.float32)


#: The opt-in marker :func:`_score` dispatches on. Set here rather than
#: inferred, so a predictor that forgets it is scored WITHOUT the image and
#: reads ~0.03, not silently ~0.94.
connected_components_predictor.reads_image = True


def build_context(seed: int, split: Optional[Dict[str, Any]] = None,
                  ) -> Tuple[Any, Any]:
    """Build the model and loss whose ``matcher`` every arm is scored through.

    Interface contract: returns ``(model, loss)`` where ``loss is model.loss``,
    the COMPILED loss, so the Hungarian assignment here is the one the runs
    used. Calls :func:`train.common.set_seeds` first, exactly as the scratch
    probes did, so the model's random initialization is reproducible -- it does
    not affect any score (no arm reads the model's weights) but it does affect
    nothing silently either way.

    Args:
        seed: The run seed. Also the TRAIN split's seed.
        split: Overrides for :data:`SPLIT`. ``None`` uses it unchanged.

    Returns:
        ``(model, loss)``.
    """
    resolved = dict(SPLIT)
    resolved.update(split or {})
    set_seeds(seed)
    config = Sam3TrainingConfig(seed=seed, **resolved)
    model = create_training_model(config)
    return model, model.loss


# DECISION plan-2026-08-07-3b8002c3/D-001
# The val-seed offset is a PARAMETER whose default equals the previously
# hardcoded `VAL_SEED_OFFSET` constant, so no published number moves unless a
# caller explicitly passes something else. Do NOT re-hardcode `VAL_SEED_OFFSET`
# at the `seed=` expression below, and do NOT fork a second split-building
# function: `baselines.py` OWNS the comparator family and every addition to it
# is a defaulted optional parameter (I-4).
# Do NOT give `pool_train_gt`, or any train-side caller, this parameter. The
# TRAIN split is the split the k-means priors are FITTED on; offsetting it
# would silently refit every published prior on different data while the
# scoring numbers still looked comparable. The offset applies to the SCORING
# split only, which is why it is consumed solely by the `train=False` branch.
# See decisions.md D-001.
def build_split_dataset(model: Any, seed: int, train: bool,
                        split: Optional[Dict[str, Any]] = None,
                        include_all_instances: bool = False,
                        val_seed_offset: int = VAL_SEED_OFFSET) -> Any:
    """Build the TRAIN or VAL split for ``seed``.

    Interface contract: the VAL split's seed is ``seed + val_seed_offset``,
    derived HERE and nowhere else, so the fit split and the scoring split
    cannot drift apart. Never shuffles: the k-means fit is order-sensitive
    through ``random_state``, and the published figures were produced on the
    unshuffled pool.

    Args:
        model: A built ``Sam3TrainingModel`` -- every geometry derives from it.
        seed: The run seed (the TRAIN seed).
        train: ``True`` for the train split, ``False`` for the val split.
        split: Overrides for :data:`SPLIT`.
        val_seed_offset: The offset added to ``seed`` to derive the VAL split's
            seed. The DEFAULT, :data:`VAL_SEED_OFFSET`, reproduces every
            published number byte-identically -- it is the same constant the
            expression carried literally before this became a parameter. Any
            NON-default value builds an INDEPENDENT split, whose numbers are
            therefore not comparable to a published figure unless the
            comparison is stated as cross-split. Ignored entirely when
            ``train`` is ``True``.
        include_all_instances: Forwarded verbatim to
            :func:`train.sam3.data.build_sam3_dataset`. ``True`` makes the
            elements 3-tuples carrying the eval-only all-category geometry
            BESIDE the prompted targets of the SAME record -- which is what
            :func:`distractor_gap` consumes. The default leaves every existing
            caller byte-identical.

    Returns:
        A batched ``tf.data.Dataset`` of ``(inputs, packed_target)``, or of
        ``(inputs, packed_target, all_instances)`` when
        ``include_all_instances``.
    """
    resolved = dict(SPLIT)
    resolved.update(split or {})
    return build_sam3_dataset(
        model=model,
        num_samples=(resolved["num_train_samples"] if train
                     else resolved["num_val_samples"]),
        batch_size=resolved["batch_size"],
        max_instances=resolved["max_instances"],
        zero_instance_rate=resolved["zero_instance_rate"],
        max_per_category=resolved["max_per_category"],
        seed=seed if train else seed + val_seed_offset,
        include_all_instances=include_all_instances,
    )


def pool_train_gt(seed: int, model: Optional[Any] = None,
                  split: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Every VALID ground-truth box of the TRAIN split, as ``(N, 4)`` cxcywh.

    Interface contract: THE single home for "read the train split's GT". It
    reads labels only -- no image, no network, no gradient -- and it reads the
    TRAIN split (``seed``), never the scoring split (``seed +
    VAL_SEED_OFFSET``). Padding rows are dropped by ``target_valid``, so the
    returned pool contains only real boxes. Two consumers: the k-means fit
    here, and the proposal-head anchor-size measurement.

    Args:
        seed: The run seed. The TRAIN split's seed, verbatim.
        model: A built ``Sam3TrainingModel``. ``None`` builds one via
            :func:`build_context` (convenient standalone; wasteful in a loop).
        split: Overrides for :data:`SPLIT`.

    Returns:
        ``(N, 4)`` float32 cxcywh boxes in [0, 1]. Empty ``(0, 4)`` if the
        split yielded no valid box.
    """
    if model is None:
        model, _loss = build_context(seed, split)
    # NO `val_seed_offset` here, deliberately (D-001). This reads the TRAIN
    # split, which is what every k-means prior is FITTED on; a train-side
    # offset would silently change the fit data of every published prior.
    dataset = build_split_dataset(model, seed, train=True, split=split)
    pool: List[np.ndarray] = []
    for _inputs, y_true in dataset:
        targets = unpack_targets(ops.cast(y_true, "float32"),
                                 model.include_masks)
        boxes = np.asarray(ops.cast(targets["target_boxes"], "float32"))
        valid = np.asarray(ops.cast(targets["target_valid"], "float32"))
        pool.append(boxes[valid > 0.0])
    stacked = (np.concatenate(pool) if pool
               else np.zeros((0, 4), dtype=np.float32))
    logger.info(
        "pool_train_gt: FIT SPLIT seed=%d (train) -> pool %s; the SCORING "
        "split is seed=%d (val) and was NOT read here.",
        seed, stacked.shape, seed + VAL_SEED_OFFSET)
    return stacked


def fit_kmeans_prior(train_boxes: np.ndarray, k: int, seed: int,
                     n_init: int = KMEANS_N_INIT) -> np.ndarray:
    """Fit a ``k``-box image-independent prior on TRAIN-split GT boxes.

    Interface contract: pure in ``(train_boxes, k, seed, n_init)`` -- it reads
    no dataset and cannot reach the scoring split, because the only split it
    ever sees is the array handed to it (see :func:`pool_train_gt`, whose
    contract is that the array comes from the TRAIN split). ``random_state`` is
    ``seed`` and ``n_init`` defaults to :data:`KMEANS_N_INIT`; BOTH are logged
    beside the fit, because the prior plan measured that leaving them free
    moves the prior by up to 0.016 IoU and can flip a per-seed sign.

    Args:
        train_boxes: ``(N, 4)`` cxcywh pool from the TRAIN split.
        k: Number of cluster centers.
        seed: Passed to ``KMeans(random_state=...)``, verbatim.
        n_init: Passed to ``KMeans(n_init=...)``, verbatim.

    Returns:
        ``(k, 4)`` float32 cluster centers, cxcywh.

    Raises:
        ValueError: If ``k < 1`` or the pool has fewer than ``k`` rows.
        ImportError: If scikit-learn is not installed. The grid arm alone is
            NOT a substitute for the family (``plans/SYSTEM.md:220``), so this
            is a hard failure rather than a degraded mode.
    """
    from sklearn.cluster import KMeans  # local: keeps import cost off tests

    pool = np.asarray(train_boxes, dtype=np.float32)
    if k < 1:
        raise ValueError(f"fit_kmeans_prior: k must be >= 1; got {k}")
    if pool.ndim != 2 or pool.shape[1] != 4:
        raise ValueError(
            f"fit_kmeans_prior: train_boxes must be (N, 4); got {pool.shape}")
    if pool.shape[0] < k:
        raise ValueError(
            f"fit_kmeans_prior: pool has {pool.shape[0]} box(es), fewer than "
            f"k={k}. A prior with empty clusters is not a prior.")
    fit = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(pool)
    logger.info(
        "fit_kmeans_prior: k=%d, n_init=%d, random_state=%d, pool=%s "
        "(the two free choices this repo measured to be worth up to 0.016 IoU)",
        k, n_init, seed, pool.shape)
    return np.asarray(fit.cluster_centers_, dtype=np.float32)


def _score(prior: Union[np.ndarray, Predictor], model: Any, loss: Any,
           dataset: Any) -> Tuple[float, float]:
    """``(box_iou, matched_total)`` -- :func:`score_prior` plus the denominator.

    Interface contract: private because ``matched_total`` is a diagnostic, not
    a result. It is exposed to the CLI so the shared-denominator check ("no arm
    can win by matching fewer pairs") can be printed; ``is_matched`` depends
    only on ``target_valid``, so every arm on a split MUST share it.

    Args:
        prior: ``(P, 4)`` boxes, a :data:`Predictor`, or an
            :data:`ImagePredictor` (a callable carrying ``reads_image = True``).
        model: A built ``Sam3TrainingModel``.
        loss: The compiled ``Sam3DetectionLoss`` whose matcher is used.
        dataset: The SCORING split.

    Returns:
        ``(box_iou over matched pairs, total matched pairs)``.
    """
    num_queries = int(model.num_queries)
    reads_image = bool(getattr(prior, "reads_image", False))
    if callable(prior) and reads_image:
        predictor: Predictor = prior
    elif callable(prior):
        def predictor(images, target_boxes, target_valid, count, _p=prior):
            del images
            return _p(target_boxes, target_valid, count)
    else:
        tiled = tile_to_queries(prior, num_queries)

        def predictor(images, target_boxes, _target_valid, _count):
            del images
            return np.tile(tiled[None], (target_boxes.shape[0], 1, 1))

    box_iou_sum = 0.0
    matched_total = 0.0
    for inputs, y_true in dataset:
        targets = unpack_targets(ops.cast(y_true, "float32"),
                                 model.include_masks)
        tgt_boxes = np.asarray(ops.cast(targets["target_boxes"], "float32"))
        tgt_valid = np.asarray(ops.cast(targets["target_valid"], "float32"))
        # The images are materialized ONLY for an `ImagePredictor`; every
        # image-independent arm never sees them, so no arm can quietly start
        # reading pixels without flipping `reads_image`.
        images = (np.asarray(inputs["image"], dtype=np.float32)
                  if reads_image else np.zeros((tgt_boxes.shape[0], 0)))
        pred_boxes = ops.convert_to_tensor(
            np.asarray(predictor(images, tgt_boxes, tgt_valid, num_queries),
                       dtype=np.float32))
        # Zeroed logits: the family is a BOX prior and makes no class claim, so
        # every query is equally (un)confident and the matcher's class cost is
        # constant. `evaluate_sam3` feeds the model's real logits here.
        pred_logits = ops.zeros(
            (pred_boxes.shape[0], num_queries, 1), dtype="float32")
        total, pairs = matched_box_iou(pred_boxes, pred_logits, targets, loss)
        box_iou_sum += total
        matched_total += pairs
    return (box_iou_sum / max(matched_total, 1.0), matched_total)


def score_prior(prior: Union[np.ndarray, Predictor], model: Any, loss: Any,
                dataset: Any) -> float:
    """Box IoU of a prior on the scoring split, through ``evaluate_sam3``'s path.

    Interface contract: the returned number is comparable, digit for digit,
    with ``evaluate_sam3``'s ``box_iou`` -- same ``loss.matcher``, same
    ``box_cxcywh_to_xyxy``, same ``iou_and_generalized_iou``, same
    ``ops.where(is_matched > 0, iou, 0)`` masked sum divided by
    ``sum(is_matched)``, all imported from the modules ``evaluate_sam3``
    imports them from. Averaged over MATCHED PAIRS ONLY; an unmatched query has
    no ground truth to be scored against.

    Args:
        prior: Either ``(P, 4)`` cxcywh boxes -- tiled to ``Q`` and emitted
            unchanged for EVERY image, i.e. image-independent by construction
            -- or a :data:`Predictor` callable (the GT-oracle liveness arm), or
            an :data:`ImagePredictor` (the connected-components detector).
        model: A built ``Sam3TrainingModel``; supplies ``num_queries`` and
            ``include_masks``.
        loss: The compiled ``Sam3DetectionLoss``.
        dataset: The SCORING split -- ``seed + VAL_SEED_OFFSET``.

    Returns:
        Box IoU over matched pairs. ``0.0`` on an empty split.
    """
    box_iou, _matched = _score(prior, model, loss, dataset)
    return box_iou


def swap_batch_prompts(inputs: Dict[str, Any], shift: int) -> Dict[str, Any]:
    """Give every image of a batch ANOTHER image's text prompt.

    Interface contract: returns a NEW dict in which exactly
    :data:`PROMPT_KEYS` are rotated by ``shift`` along the BATCH axis and every
    other key -- ``image`` above all -- is the caller's object unchanged. The
    targets are not touched here and must not be touched by the caller either:
    the point of the diagnostic is to score the SAME image against the SAME
    ground truth under a DIFFERENT prompt.

    Args:
        inputs: One batch of the dataset's input dict.
        shift: Batch-axis rotation. ``0`` is a no-op and makes the diagnostic
            vacuous, which is why :func:`prompt_swap_retention` reports the
            fraction of prompts that actually changed.

    Returns:
        A shallow copy with the two prompt tensors rotated.
    """
    swapped = dict(inputs)
    for key in PROMPT_KEYS:
        swapped[key] = ops.roll(inputs[key], shift, axis=0)
    return swapped


def prompt_swap_retention(model: Any, loss: Any, dataset: Any,
                          shifts: Sequence[int] = PROMPT_SWAP_SHIFTS,
                          ) -> Dict[str, float]:
    """How much ``box_iou`` a checkpoint KEEPS when the prompt is wrong.

    Interface contract: this is the one MODEL-scoring arm in a module of
    non-model baselines, and it lives here on purpose -- it is a QUALIFIER on
    the same number the family bars, so it must be quotable in the same table
    and computed through the same
    :func:`~train.sam3.train_sam3.matched_box_iou` reduction. It runs the
    model at ``training=False``, never touches the targets, and returns a flat
    ``Dict[str, float]``.

    Reading the result. ``retained`` near ``1.0`` means the box metric is
    INVARIANT to the text prompt: on that generator ``box_iou`` measures "find
    any bright shape", not "find the NAMED shape", and no text-grounded
    detection claim may be made from it. That verdict is only meaningful
    alongside the other two keys: ``prompt_changed_fraction`` (a rotation can
    land on an image of the same category, so this is measured, not assumed)
    and ``rel_delta_pred_logits`` (if the swap moves NOTHING anywhere, the
    instrument is dead and ``retained == 1.0`` says nothing about the model).

    Args:
        model: A built or loaded ``Sam3TrainingModel``.
        loss: The compiled ``Sam3DetectionLoss`` whose matcher is used.
        dataset: The SCORING split, ``(inputs, packed_target)`` batches.
        shifts: Batch-axis rotations to try. The reported wrong-prompt IoU is
            the WORST (lowest) over them.

    Returns:
        ``box_iou_true``, ``box_iou_worst_wrong_prompt``, ``retained``
        (= worst / true), ``prompt_changed_fraction``, ``matched_pairs`` (the
        matched-pair DENOMINATOR, equal across every arm by assertion -- see
        below), and ``rel_delta_pred_boxes`` / ``rel_delta_pred_logits`` --
        each the max absolute change of that output under any shift, divided by
        that output's own std over the split.

    Raises:
        ValueError: if the TRUE and swapped arms matched DIFFERENT numbers of
            pairs. ``retained`` divides two independently pooled ratios, and
            the matcher keeps a pair only where its cost clears
            ``VALID_COST_THRESHOLD`` -- a cost that INCLUDES a class term
            computed from ``pred_logits``, which do move under a swap. The
            denominators are therefore not equal by construction, only equal in
            fact; this raise is what keeps the ratio from silently becoming a
            comparison of two different populations.
    """
    sums: Dict[Any, List[float]] = {key: [0.0, 0.0]
                                    for key in ("TRUE",) + tuple(shifts)}
    changed = seen = 0.0
    gaps = {"pred_boxes": 0.0, "pred_logits": 0.0}
    pooled: Dict[str, List[np.ndarray]] = {key: [] for key in gaps}
    for inputs, y_true in dataset:
        targets = unpack_targets(ops.cast(y_true, "float32"),
                                 model.include_masks)
        true_out = model.sam3(inputs, training=False)
        total, pairs = matched_box_iou(
            ops.cast(true_out["pred_boxes"], "float32"),
            ops.cast(true_out["pred_logits"], "float32"), targets, loss)
        sums["TRUE"][0] += total
        sums["TRUE"][1] += pairs
        for key in gaps:
            pooled[key].append(np.asarray(
                ops.convert_to_numpy(true_out[key]), dtype=np.float64))
        original = np.asarray(ops.convert_to_numpy(inputs["token_ids"]))
        for shift in shifts:
            swapped = swap_batch_prompts(inputs, shift)
            out = model.sam3(swapped, training=False)
            total, pairs = matched_box_iou(
                ops.cast(out["pred_boxes"], "float32"),
                ops.cast(out["pred_logits"], "float32"), targets, loss)
            sums[shift][0] += total
            sums[shift][1] += pairs
            for key in gaps:
                delta = np.abs(np.asarray(ops.convert_to_numpy(out[key]),
                                          dtype=np.float64) - pooled[key][-1])
                gaps[key] = max(gaps[key], float(delta.max()))
            rolled = np.asarray(ops.convert_to_numpy(swapped["token_ids"]))
            changed += float(np.sum(np.any(original != rolled, axis=-1)))
            seen += float(original.shape[0])

    denominators = {key: sums[key][1] for key in sums}
    if len(set(denominators.values())) != 1:
        raise ValueError(
            "the TRUE and wrong-prompt arms matched DIFFERENT numbers of "
            f"pairs {denominators}: `retained` would divide ratios taken over "
            "two different populations, so it is not a retention")
    true_iou = sums["TRUE"][0] / max(sums["TRUE"][1], 1.0)
    worst = min(sums[shift][0] / max(sums[shift][1], 1.0) for shift in shifts)
    result = {
        "box_iou_true": true_iou,
        "box_iou_worst_wrong_prompt": worst,
        "retained": worst / true_iou if true_iou else float("nan"),
        "prompt_changed_fraction": changed / max(seen, 1.0),
        "matched_pairs": sums["TRUE"][1],
    }
    for key in gaps:
        spread = float(np.concatenate(pooled[key]).std())
        result[f"rel_delta_{key}"] = (gaps[key] / spread if spread
                                      else float("nan"))
    logger.info("prompt_swap_retention: %s", result)
    return result


def distractor_gap(model: Any, loss: Any, dataset: Any) -> Dict[str, float]:
    """How much better a checkpoint's boxes fit the PROMPTED category's GT.

    Interface contract: the second MODEL-scoring arm in a module of non-model
    baselines, built exactly like :func:`prompt_swap_retention` -- one forward
    pass at ``training=False``, a flat ``Dict[str, float]``, scored through
    :func:`~train.sam3.train_sam3.matched_box_iou` so it cannot drift from any
    published ``box_iou``. It takes ONE dataset and reads BOTH target sets out
    of it (the third tuple slot), rather than zipping two datasets: two
    datasets would have to be
    proven aligned batch-for-batch, and this way there is nothing to align.

    Reading the result. ``box_iou`` alone cannot tell "the model learned to
    find the NAMED shape" from "this generator's box task is solvable by any
    rule that reads pixels": the zero-parameter connected-components detector
    scores ABOVE the trained arm. This diagnostic scores the SAME predicted
    boxes twice -- against the prompted category's GT, and against the pooled
    GT of every OTHER category in the SAME image. A category-blind predictor
    reads ~0 no matter how high its raw ``box_iou`` is; a GT oracle reads 1.0.

    The two denominators are NOT equal, and that is correct, not a defect.
    ``zero_instance_rate`` gives ~25% of images an ABSENT prompted category:
    those contribute zero prompted pairs and non-zero distractor pairs. Unlike
    :func:`prompt_swap_retention` -- whose two arms score the same targets and
    whose ratio would therefore be meaningless across different populations --
    the two arms here score DIFFERENT target sets by design, so both
    denominators are reported and neither is asserted equal to the other.

    Args:
        model: A built or loaded ``Sam3TrainingModel``.
        loss: The compiled ``Sam3DetectionLoss`` whose matcher is used.
        dataset: The SCORING split built with ``include_all_instances=True``,
            i.e. ``(inputs, packed_target, all_instances)`` batches.

    Returns:
        ``box_iou_prompted``, ``box_iou_distractor``, ``gap``
        (= prompted - distractor), ``relative_gap`` (= gap / prompted),
        ``matched_pairs_prompted`` and ``matched_pairs_distractor`` (the two
        denominators, reported because they differ), and
        ``images_with_distractor`` -- the number of images that actually had a
        non-prompted instance. An image with only ONE category drawn has no
        distractor and contributes to the prompted arm only; that count is
        reported rather than silently absorbed into the mean.

    Raises:
        ValueError: If the dataset does not yield 3-tuples (it was built
            without ``include_all_instances``), or if the alignment assertion
            of :func:`~train.sam3.data.distractor_targets` fails.
    """
    sums: Dict[str, List[float]] = {"prompted": [0.0, 0.0],
                                    "distractor": [0.0, 0.0]}
    with_distractor = 0.0
    for element in dataset:
        if len(element) != 3:
            raise ValueError(
                f"distractor_gap: the dataset yields {len(element)}-tuples; "
                "it must be built with include_all_instances=True, whose "
                "third slot carries the all-category ground truth. Scoring "
                "the prompted targets twice would report a gap of exactly "
                "0.0 for every checkpoint, which is a plausible-looking lie.")
        inputs, y_true, all_instances = element
        prompted = unpack_targets(ops.cast(y_true, "float32"),
                                  model.include_masks)
        # The alignment assertion runs BEFORE the forward pass, so a
        # misaligned batch cannot even produce a number to quote.
        distractor = distractor_targets(all_instances, prompted)
        with_distractor += float(np.sum(np.asarray(ops.convert_to_numpy(
            distractor["target_valid"])).sum(axis=-1) > 0.0))

        out = model.sam3(inputs, training=False)
        pred_boxes = ops.cast(out["pred_boxes"], "float32")
        pred_logits = ops.cast(out["pred_logits"], "float32")
        for name, targets in (("prompted", prompted),
                              ("distractor", distractor)):
            total, pairs = matched_box_iou(pred_boxes, pred_logits, targets,
                                           loss)
            sums[name][0] += total
            sums[name][1] += pairs

    prompted_iou = sums["prompted"][0] / max(sums["prompted"][1], 1.0)
    distractor_iou = sums["distractor"][0] / max(sums["distractor"][1], 1.0)
    gap = prompted_iou - distractor_iou
    result = {
        "box_iou_prompted": prompted_iou,
        "box_iou_distractor": distractor_iou,
        "gap": gap,
        "relative_gap": gap / prompted_iou if prompted_iou else float("nan"),
        "matched_pairs_prompted": sums["prompted"][1],
        "matched_pairs_distractor": sums["distractor"][1],
        "images_with_distractor": with_distractor,
    }
    logger.info("distractor_gap: %s", result)
    return result


def evaluate_family(seed: int, ks: Sequence[int] = FAMILY_KS,
                    split: Optional[Dict[str, Any]] = None,
                    val_seed_offset: int = VAL_SEED_OFFSET,
                    ) -> Dict[str, Dict[str, float]]:
    """Score every arm of the baseline family at one seed.

    Interface contract: fits on the TRAIN split of ``seed`` and scores on the
    VAL split of ``seed + val_seed_offset``; both seeds are logged AND returned
    under ``"_meta"``, along with the offset actually used, so a leak -- or a
    non-default split -- is visible in the artifact rather than only implied.
    The two liveness arms (``ORACLE``, ``DEGENERATE``) are always present --
    they are not optional and are not gated by a flag.

    Args:
        seed: The run seed.
        ks: k values to fit.
        split: Overrides for :data:`SPLIT`.
        val_seed_offset: Forwarded verbatim to :func:`build_split_dataset` for
            the SCORING split. The default reproduces every published number
            byte-identically; a non-default value scores on an INDEPENDENT
            split. The FIT split (:func:`pool_train_gt`) never sees it.

    Returns:
        ``{arm_name: {"box_iou": float, "matched": float}}`` plus a ``"_meta"``
        entry carrying ``fit_seed``, ``score_seed``, ``val_seed_offset``,
        ``pool_size``, ``k_init`` and ``num_queries``.
    """
    model, loss = build_context(seed, split)
    num_queries = int(model.num_queries)
    # `pool_train_gt` gets NO offset: it reads the TRAIN split (D-001).
    pool = pool_train_gt(seed, model=model, split=split)
    val_dataset = build_split_dataset(model, seed, train=False, split=split,
                                      val_seed_offset=val_seed_offset)

    arms: Dict[str, Union[np.ndarray, Predictor]] = {
        ORACLE_ARM: gt_oracle_predictor,
        DEGENERATE_ARM: degenerate_prior(),
        CONNECTED_COMPONENTS_ARM: connected_components_predictor,
        GRID_ARM: fixed_grid_prior(),
    }
    for k in ks:
        arms[kmeans_arm(k)] = fit_kmeans_prior(pool, k, seed)

    results: Dict[str, Dict[str, float]] = {
        "_meta": {
            "fit_seed": float(seed),
            "score_seed": float(seed + val_seed_offset),
            "val_seed_offset": float(val_seed_offset),
            "pool_size": float(pool.shape[0]),
            "kmeans_n_init": float(KMEANS_N_INIT),
            "kmeans_random_state": float(seed),
            "num_queries": float(num_queries),
        },
    }
    for name, prior in arms.items():
        box_iou, matched = _score(prior, model, loss, val_dataset)
        results[name] = {"box_iou": box_iou, "matched": matched}
        logger.info("seed %d  %-36s box_iou = %.6f   matched = %.1f",
                    seed, name, box_iou, matched)
    return results


def family_max(results: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
    """The ``max`` over the NAMED family -- the bar an accuracy claim must clear.

    Interface contract: the liveness arms are EXCLUDED. ``ORACLE`` reads the
    image (it is not a baseline, it is an instrument check) and ``DEGENERATE``
    is a floor, not a bar. The family is exactly
    ``{fixed grid} u {k-means prior}``, per ``plans/SYSTEM.md:220``.

    Args:
        results: One seed's :func:`evaluate_family` output.

    Returns:
        ``(winning arm name, its box IoU)``.

    Raises:
        ValueError: If no family arm is present.
    """
    # DECISION plan-2026-08-06T185813-fd80240f/D-009
    # This filter is an ALLOWLIST, and that is the whole mechanism keeping the
    # pre-registered SC-B bar where it was pre-registered. Do NOT add
    # `connected_components_predictor` (or `prompt_swap_retention`, or any
    # future image-reading arm) to it, and do NOT rename such an arm to start
    # with "FIXED-GRID" / "KMEANS-PRIOR": either move would retroactively raise
    # a bar that was fixed BEFORE the runs, turning a pre-registered comparison
    # into a post-hoc one. The image-reading numbers are reported in their own
    # CLI sections precisely so both bars stay separately quotable. See
    # decisions.md D-009.
    family = {name: row["box_iou"] for name, row in results.items()
              if name.startswith(("FIXED-GRID", "KMEANS-PRIOR"))}
    if not family:
        raise ValueError(
            "family_max: no {FIXED-GRID, KMEANS-PRIOR} arm in the results. "
            "The grid alone is not the family and the family cannot be empty.")
    best = max(family, key=family.get)
    return best, family[best]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        The parser for ``python -m train.sam3.baselines``.
    """
    parser = argparse.ArgumentParser(
        description=("Score the non-model baseline family SAM 3's box_iou "
                     "must be quoted against."))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                        help="Run seeds. The fit reads seed; the scoring "
                             "split is seed + %d." % VAL_SEED_OFFSET)
    parser.add_argument("--k", type=int, nargs="+", default=list(FAMILY_KS),
                        help="k values for the k-means prior.")
    # NOTE: no bare '%' anywhere in this help string -- argparse runs every
    # help= through `help % params` at --help time, so a lone '%' crashes
    # --help and ONLY --help (see tests/.../parser_help_guard.py).
    parser.add_argument("--val-seed-offset", type=int,
                        default=VAL_SEED_OFFSET,
                        help="Offset added to each run seed to derive the "
                             "SCORING (val) split's seed. The default, "
                             + str(VAL_SEED_OFFSET) + ", is the offset every "
                             "published number was produced with and "
                             "reproduces them exactly. Any other value scores "
                             "on an INDEPENDENT split, so its numbers are not "
                             "comparable to a published figure unless the "
                             "comparison is stated as cross-split. The FIT "
                             "(train) split is never offset.")
    parser.add_argument("--json", type=str, default=None,
                        help="Write the full result table to this path.")
    parser.add_argument("--prompt-swap", type=str, default=None,
                        metavar="TEMPLATE",
                        help="A checkpoint path template containing '{seed}', "
                             "e.g. 'results/step9_qsel_seed{seed}/"
                             "best_model.keras'. Scores that checkpoint's "
                             "box_iou under the TRUE prompt and under another "
                             "image's prompt, and prints the retention. Quote "
                             "it beside any accuracy claim on this generator.")
    parser.add_argument("--distractor-gap", type=str, default=None,
                        metavar="TEMPLATE",
                        help="A checkpoint path template containing '{seed}', "
                             "e.g. 'results/step9_qsel_seed{seed}/"
                             "final_model.keras'. Scores that checkpoint's "
                             "SAME predicted boxes twice -- against the "
                             "prompted category's ground truth, and against "
                             "the pooled ground truth of every OTHER category "
                             "present in the same image -- and prints the "
                             "gap. A category-blind predictor's gap here is "
                             "noise around zero whose sign is seed-dependent, "
                             "however high its raw box_iou is.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Score the family on every requested seed and print the table.

    Args:
        argv: Command line. ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``0`` when both liveness arms read their known-in-advance values on
        every seed, ``1`` otherwise -- a failed liveness check invalidates every
        other number printed, so it must be an exit code and not a warning.
    """
    args = build_parser().parse_args(argv)
    seeds = list(args.seeds)
    per_seed: Dict[int, Dict[str, Dict[str, float]]] = {}
    for seed in seeds:
        _emit(f"# seed {seed}: FIT SPLIT seed={seed} (train)  ->  "
              f"SCORING SPLIT seed={seed + args.val_seed_offset} (val, "
              f"offset {args.val_seed_offset}). "
              f"The fit never reads the scoring split.")
        per_seed[seed] = evaluate_family(seed, ks=args.k,
                                         val_seed_offset=args.val_seed_offset)
        meta = per_seed[seed]["_meta"]
        _emit(f"# seed {seed}: pool={int(meta['pool_size'])} boxes, "
              f"Q={int(meta['num_queries'])}, "
              f"KMeans(n_init={KMEANS_N_INIT}, random_state={seed})")

    arm_names = [name for name in per_seed[seeds[0]] if name != "_meta"]

    _emit("")
    _emit("| arm | " + " | ".join(f"seed {s}" for s in seeds) + " |")
    _emit("|---|" + "---|" * len(seeds))
    for name in arm_names:
        row = " | ".join(f"{per_seed[s][name]['box_iou']:.4f}" for s in seeds)
        _emit(f"| {name} | {row} |")

    _emit("")
    _emit("LIVENESS -- values known IN ADVANCE. If either fails, believe "
          "nothing above.")
    live = True
    for seed in seeds:
        oracle = per_seed[seed][ORACLE_ARM]["box_iou"]
        degenerate = per_seed[seed][DEGENERATE_ARM]["box_iou"]
        ok_oracle = abs(1.0 - oracle) < 1e-5
        ok_degenerate = degenerate < 0.05
        live = live and ok_oracle and ok_degenerate
        _emit(f"  seed {seed}: oracle = {oracle:.10f} "
              f"|1 - oracle| = {abs(1.0 - oracle):.3e} "
              f"[{'OK' if ok_oracle else 'FAIL'}];  "
              f"degenerate = {degenerate:.4f} (< 0.05) "
              f"[{'OK' if ok_degenerate else 'FAIL'}]")
    _emit(f"  LIVENESS {'PASSED' if live else 'FAILED'}")

    _emit("")
    _emit("Denominator -- `is_matched` depends only on `target_valid`, so "
          "every arm on a seed must share it:")
    for seed in seeds:
        totals = sorted({per_seed[seed][name]["matched"]
                         for name in arm_names})
        _emit(f"  seed {seed}: matched totals across arms = {totals}")

    _emit("")
    _emit("THE BAR -- max over the NAMED family "
          "{fixed 5x5 grid} U {k-means train-fit prior}:")
    for seed in seeds:
        best, value = family_max(per_seed[seed])
        _emit(f"  seed {seed}: max = {value:.4f}  ({best})")

    _emit("")
    _emit("THE IMAGE-READING PREDICTOR -- NOT a member of the family above "
          "(SYSTEM.md:220 names the family), reported separately because a "
          "family with no image-reading member cannot tell 'the model learned "
          "detection' from 'this generator's box task is solvable by any rule "
          "that looks at the pixels'. Zero parameters, zero training, "
          "category-BLIND:")
    for seed in seeds:
        cc = per_seed[seed][CONNECTED_COMPONENTS_ARM]["box_iou"]
        best, value = family_max(per_seed[seed])
        _emit(f"  seed {seed}: connected components = {cc:.6f}   "
              f"(family max {value:.4f}; the `step9_qsel` checkpoint -- deep "
              f"supervision + query selection, WITHOUT prompt conditioning -- "
              f"read 0.8450 / 0.8296 / 0.8191 on seeds 1 / 2 / 3. Name the "
              f"CHECKPOINT and not the flag: `step10_pcq` is also a "
              f"query-selection arm and reads 0.8494 / 0.8223 / 0.8299)")

    swap_by_seed: Dict[int, Dict[str, float]] = {}
    gap_by_seed: Dict[int, Dict[str, float]] = {}

    if args.prompt_swap:
        _emit("")
        _emit("PROMPT-SWAP RETENTION -- the fraction of a CHECKPOINT's box_iou "
              "that survives replacing every image's text prompt with another "
              "image's. `retained` near 1.00 means the box metric is invariant "
              "to the prompt, i.e. it measures 'find any bright shape' and not "
              "'find the NAMED shape'; no text-grounded detection claim may be "
              "made from a number with that qualifier beside it:")
        for seed in seeds:
            path = Path(args.prompt_swap.format(seed=seed))
            if not path.exists():
                _emit(f"  seed {seed}: {path} ABSENT -- skipped")
                continue
            ctx_model, ctx_loss = build_context(seed)
            swap = prompt_swap_retention(
                keras.models.load_model(path, compile=False), ctx_loss,
                build_split_dataset(ctx_model, seed, train=False,
                                    val_seed_offset=args.val_seed_offset))
            swap_by_seed[seed] = swap
            _emit(f"  seed {seed} ({path}): true = "
                  f"{swap['box_iou_true']:.6f}, worst wrong prompt = "
                  f"{swap['box_iou_worst_wrong_prompt']:.6f}, RETAINED = "
                  f"{swap['retained']:.4f}   [both arms matched the same "
                  f"{swap['matched_pairs']:.0f} pairs; prompts actually "
                  f"changed on "
                  f"{swap['prompt_changed_fraction']:.2f} of images; relative "
                  f"delta pred_boxes {swap['rel_delta_pred_boxes']:.2e}, "
                  f"pred_logits {swap['rel_delta_pred_logits']:.2e}]")

    if args.distractor_gap:
        _emit("")
        _emit(f"{DISTRACTOR_GAP_ARM} -- a CHECKPOINT's same predicted boxes "
              "scored twice: against the PROMPTED category's ground truth, "
              "and against the pooled ground truth of every OTHER category in "
              "the SAME image. A category-blind arm's gap is noise around "
              "zero with a SEED-DEPENDENT SIGN: the connected-components "
              "detector reads +0.004954 / -0.004813 / -0.000371 on seeds "
              "1/2/3 while reading ~0.94 raw box_iou, and the wider blind "
              "band (FIXED-GRID and KMEANS-PRIOR included) reaches 0.0358 "
              "absolute (KMEANS k=8, seed 3), so 0.02 is NOT a general "
              "category-blind ceiling. The two denominators "
              "DIFFER on purpose -- an image whose prompted category is "
              "absent contributes zero prompted pairs and non-zero distractor "
              "pairs -- so both are printed and neither is asserted equal:")
        for seed in seeds:
            path = Path(args.distractor_gap.format(seed=seed))
            if not path.exists():
                _emit(f"  seed {seed}: {path} ABSENT -- skipped")
                continue
            ctx_model, ctx_loss = build_context(seed)
            gap = distractor_gap(
                keras.models.load_model(path, compile=False), ctx_loss,
                build_split_dataset(ctx_model, seed, train=False,
                                    include_all_instances=True,
                                    val_seed_offset=args.val_seed_offset))
            gap_by_seed[seed] = gap
            _emit(f"  seed {seed} ({path}): prompted = "
                  f"{gap['box_iou_prompted']:.6f}, distractor = "
                  f"{gap['box_iou_distractor']:.6f}, GAP = {gap['gap']:.6f} "
                  f"({gap['relative_gap']:.4f} relative)   "
                  f"[{gap['matched_pairs_prompted']:.0f} prompted vs "
                  f"{gap['matched_pairs_distractor']:.0f} distractor pairs; "
                  f"{gap['images_with_distractor']:.0f} image(s) actually had "
                  "a distractor]")

    if args.json:
        # DECISION plan-2026-08-07T065516-6add49a9/D-022
        # `--distractor-gap` and `--prompt-swap` MUST write into the JSON, not
        # only into stdout: the plan named this file as the machine-readable
        # evidence for those numbers and for one release it carried the family
        # rows ONLY, so the numbers survived just in a log. Two constraints go
        # with that, and neither is optional. (1) Write into a COPY of
        # `per_seed[seed]`, never into `per_seed` itself -- `family_max`
        # iterates that dict, and a nested dict there is an entry a future
        # widening of the allowlist could pick up (I-6). (2) The keys are
        # `prompt_swap` / `distractor_gap`, NOT arm-shaped names, so no
        # `startswith(("FIXED-GRID", "KMEANS-PRIOR"))` reading can ever match
        # them. See decisions.md D-022.
        payload: Dict[str, Any] = {
            str(seed): dict(per_seed[seed]) for seed in seeds}
        for seed in seeds:
            if seed in swap_by_seed:
                payload[str(seed)]["prompt_swap"] = swap_by_seed[seed]
            if seed in gap_by_seed:
                payload[str(seed)]["distractor_gap"] = gap_by_seed[seed]
        payload["_family_max"] = {
            str(seed): dict(zip(("arm", "box_iou"), family_max(per_seed[seed])))
            for seed in seeds}
        Path(args.json).write_text(json.dumps(payload, indent=2))
        _emit(f"\nwrote {args.json}")

    return 0 if live else 1


if __name__ == "__main__":
    raise SystemExit(main())
