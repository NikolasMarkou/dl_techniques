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
detector reads ~0.94 box IoU, ABOVE the trained query-selection arm's
0.8450 / 0.8296 / 0.8191 on 3 of 3 seeds. Adversarial review of
``plan-2026-08-06T185813-fd80240f`` raised exactly this (CRITICAL 1) and the
measurement is the answer to it: any accuracy claim on this generator is a
WIRING / LEARNABILITY result, not a capability result. :func:`family_max`
excludes the detector on purpose -- ``plans/SYSTEM.md:220`` names the family --
so the two bars stay separately quotable.

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

import numpy as np
from keras import ops

from dl_techniques.losses.sam3_detection_loss import (
    box_cxcywh_to_xyxy,
    iou_and_generalized_iou,
    unpack_targets,
)
from dl_techniques.utils.logger import logger
from train.common import set_seeds
from train.sam3.data import build_sam3_dataset
from train.sam3.train_sam3 import Sam3TrainingConfig, create_training_model

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


def build_split_dataset(model: Any, seed: int, train: bool,
                        split: Optional[Dict[str, Any]] = None) -> Any:
    """Build the TRAIN or VAL split for ``seed``.

    Interface contract: the VAL split's seed is ``seed + VAL_SEED_OFFSET``,
    derived HERE and nowhere else, so the fit split and the scoring split
    cannot drift apart. Never shuffles: the k-means fit is order-sensitive
    through ``random_state``, and the published figures were produced on the
    unshuffled pool.

    Args:
        model: A built ``Sam3TrainingModel`` -- every geometry derives from it.
        seed: The run seed (the TRAIN seed).
        train: ``True`` for the train split, ``False`` for the val split.
        split: Overrides for :data:`SPLIT`.

    Returns:
        A batched ``tf.data.Dataset`` of ``(inputs, packed_target)``.
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
        seed=seed if train else seed + VAL_SEED_OFFSET,
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
        assignment, is_matched = loss.matcher(
            pred_logits, pred_boxes,
            targets["target_boxes"], targets["target_valid"])
        gathered = ops.take_along_axis(
            targets["target_boxes"], assignment[:, :, None], axis=1)
        # `iou_and_generalized_iou` reads **xyxy**; every box here is
        # normalized cxcywh. Feeding it cxcywh does not raise -- it silently
        # scores two rectangles that are not the boxes.
        iou, _giou = iou_and_generalized_iou(
            box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gathered))
        # `ops.where`, NOT `iou * is_matched`: the multiplicative spelling
        # propagates `nan * 0.0 = nan` from a both-degenerate padded pair.
        box_iou_sum += float(ops.sum(
            ops.where(is_matched > 0.0, iou, ops.zeros_like(iou))))
        matched_total += float(ops.sum(is_matched))
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


def evaluate_family(seed: int, ks: Sequence[int] = FAMILY_KS,
                    split: Optional[Dict[str, Any]] = None,
                    ) -> Dict[str, Dict[str, float]]:
    """Score every arm of the baseline family at one seed.

    Interface contract: fits on the TRAIN split of ``seed`` and scores on the
    VAL split of ``seed + VAL_SEED_OFFSET``; both seeds are logged AND returned
    under ``"_meta"`` so a leak is visible in the artifact. The two liveness
    arms (``ORACLE``, ``DEGENERATE``) are always present -- they are not
    optional and are not gated by a flag.

    Args:
        seed: The run seed.
        ks: k values to fit.
        split: Overrides for :data:`SPLIT`.

    Returns:
        ``{arm_name: {"box_iou": float, "matched": float}}`` plus a ``"_meta"``
        entry carrying ``fit_seed``, ``score_seed``, ``pool_size``, ``k_init``
        and ``num_queries``.
    """
    model, loss = build_context(seed, split)
    num_queries = int(model.num_queries)
    pool = pool_train_gt(seed, model=model, split=split)
    val_dataset = build_split_dataset(model, seed, train=False, split=split)

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
            "score_seed": float(seed + VAL_SEED_OFFSET),
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
    parser.add_argument("--json", type=str, default=None,
                        help="Write the full result table to this path.")
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
              f"SCORING SPLIT seed={seed + VAL_SEED_OFFSET} (val). "
              f"The fit never reads the scoring split.")
        per_seed[seed] = evaluate_family(seed, ks=args.k)
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
              f"(family max {value:.4f}; the trained query-selection arm read "
              f"0.8450 / 0.8296 / 0.8191 on seeds 1 / 2 / 3)")

    if args.json:
        payload = {str(seed): per_seed[seed] for seed in seeds}
        payload["_family_max"] = {
            str(seed): dict(zip(("arm", "box_iou"), family_max(per_seed[seed])))
            for seed in seeds}
        Path(args.json).write_text(json.dumps(payload, indent=2))
        _emit(f"\nwrote {args.json}")

    return 0 if live else 1


if __name__ == "__main__":
    raise SystemExit(main())
