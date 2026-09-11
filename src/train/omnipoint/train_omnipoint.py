"""OmniPoint metric point-cloud training script.

Pattern-5-shaped trainer (`src/train/CLAUDE.md`) for
`dl_techniques.models.vision.omnipoint.OmniPoint` on the combined KITTI-depth + MegaDepth
pipeline built in Step 8 (`train.omnipoint.data.CombinedOmniPointDataset`). Structurally closest
to `src/train/depth_anything/train_depth_anything.py` (dataclass config, `create_base_argument_
parser()` + model-specific flags on top, `create_callbacks()`, `save_config_json`), but with one
load-bearing difference documented below (D-020): OmniPoint's model output is a 5-tuple over
several heads, and Keras 3.8's `model.compile(loss=<single callable>)` broadcasts one loss
COPY per output slot rather than calling it once over the whole tuple (MEASURED against
`keras.src.trainers.compile_utils.CompileLoss.build`, see D-020) -- so `OmniPointCombinedLoss`
(which needs the FULL 5-tuple `y_true` and 4-tuple `y_pred` in one call, per
`decisions.md` D-016) cannot be wired through `compile(loss=...)` directly.

.. warning::
    **`--dataset`/`--image-size`'s dataset CHOICES do not apply here.** This script calls
    `create_base_argument_parser()` for its shared numeric flags (`--epochs`, `--batch-size`,
    `--learning-rate`, `--weight-decay`, `--lr-schedule`, `--patience`, `--gpu`,
    `--show-plots`) and to keep a consistent flag surface with every other trainer, but
    OmniPoint's data source is KITTI-depth + MegaDepth (`--kitti-root`/`--megadepth-root`
    below), never one of `create_base_argument_parser()`'s `--dataset` choices
    (`mnist`/`cifar10`/`cifar100`/`imagenet`). `--dataset` is parsed and then ignored, by
    design -- it is a CLI-surface artifact of reusing the shared parser, not a config field
    (`OmniPointTrainingConfig` carries no `dataset` field for it to silently fail to reach).

.. warning::
    **`--enable-conditioning` trains INTRINSICS conditioning only, never sparse-depth
    conditioning** (`decisions.md` D-028, refined by D-029). `OmniPointTrainingWrapper.call()`
    derives the per-sample intrinsics ray map from `data.py`'s own `gt_ray` (the same
    `pinhole_ray_map(K, ...)` value, reused rather than re-derived) and feeds it to
    `OmniPoint` every step, so the intrinsics-conditioning weights DO receive gradient
    when this flag is set. Sparse-depth conditioning remains genuinely unwired: no
    sparse-depth data exists anywhere in this pipeline (neither the KITTI nor the
    MegaDepth loader synthesizes or supplies a sparse subsample), so
    `sparse_depth`/`sparse_depth_mask` are never passed and those weights receive zero
    gradient -- a documented, intentional limitation, not a silent defect.

    **`intrinsics_present` is randomized PER-SAMPLE, not a fixed `True`** (`decisions.md`
    D-029, a completion fix to D-028's own gap): D-028 shipped with `intrinsics_present`
    hardwired to all-`True`, which made `gt_ray` -- the literal `L_ray` supervision target
    -- also the conditioning input on every step, a trivially-satisfiable copy task, and
    left `intrinsics_state/absent_embedding` permanently dead. `--intrinsics-conditioning-prob`
    (default `0.9`, matching the OmniPoint paper's own Supplementary Section B convention:
    "geometric conditions ... enabled with a probability of 90% to preserve robustness under
    RGB-only inputs") now draws a fresh per-sample Bernoulli flag every call via a
    `keras.random.SeedGenerator`-backed stateless draw, so both the present and absent
    embeddings train and `L_ray` stays a genuine RGB-only inference task for the ~10% of
    samples drawn absent each step.

Usage::

    # Smoke run (tiny model/data, proves the pipeline runs end to end):
    MPLBACKEND=Agg .venv/bin/python -m train.omnipoint.train_omnipoint --smoke --gpu 0

    # Real run:
    MPLBACKEND=Agg .venv/bin/python -m train.omnipoint.train_omnipoint \\
        --omnipoint-variant omnipoint_base --image-size 224 \\
        --epochs 50 --batch-size 8 --learning-rate 1e-4 --gpu 0
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import keras

from train.common import (
    create_callbacks,
    save_config_json,
    set_seeds,
    setup_gpu,
)
from train.common.args import create_base_argument_parser
from train.common.megadepth import discover_megadepth_pairs
from train.omnipoint.data import CombinedOmniPointDataset
from train.omnipoint.kitti_depth import discover_kitti_depth_pairs

from dl_techniques.losses.omnipoint_losses import OmniPointCombinedLoss
from dl_techniques.models.vision.omnipoint.model import MODEL_VARIANTS, create_omnipoint
from dl_techniques.optimization import learning_rate_schedule_builder, optimizer_builder
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# `--lr-schedule`'s three choices (`create_base_argument_parser()`'s own flag, shared
# across every trainer) map onto `learning_rate_schedule_builder`'s `type` strings, which
# use a different vocabulary (`cosine_decay`/`exponential_decay`/`cosine_decay_restarts`,
# no `constant`). `constant` is handled separately (a bare float passed straight to
# `optimizer_builder`, which accepts either) -- see `optimization/schedule.py`'s own
# documented reason for NOT unifying this vocabulary.
_LR_SCHEDULE_TYPE_MAP = {
    "cosine": "cosine_decay",
    "exponential": "exponential_decay",
}

# `--smoke`'s dataset/training overrides -- deliberately tiny: a 4x4 ViT patch grid
# (image_size=56, patch_size=14) and 1-2 samples per batch keep even `omnipoint_base`'s
# forward+backward pass fast, since attention cost here is dominated by the (tiny)
# sequence length, not the encoder's parameter count.
SMOKE_IMAGE_SIZE = 56
SMOKE_BATCH_SIZE = 2
SMOKE_EPOCHS = 1
SMOKE_MAX_FILES = 4
SMOKE_WORKERS = 1


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class OmniPointTrainingConfig:
    """Configuration for `OmniPoint` training on the combined KITTI + MegaDepth pipeline."""

    # Data
    kitti_root: Optional[str] = "/media/arxwn/data0_4tb/datasets/KITTI/data/depth"
    megadepth_root: Optional[str] = "/media/arxwn/data0_4tb/datasets/Megadepth"
    image_size: int = 224
    train_split: float = 0.9
    max_kitti_files: Optional[int] = None
    max_megadepth_files: Optional[int] = None
    workers: int = 8

    # Model
    omnipoint_variant: str = "omnipoint_base"
    enable_conditioning: bool = False
    intrinsics_conditioning_prob: float = 0.9

    # Loss weights (OmniPointCombinedLoss)
    lambda_ray: float = 1.0
    lambda_metric: float = 1.0
    lambda_normal: float = 1.0
    lambda_local: float = 1.0
    lambda_mask: float = 1.0

    # Training
    batch_size: int = 8
    epochs: int = 50

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_schedule_type: str = "cosine"  # one of {'cosine', 'exponential', 'constant'}
    patience: int = 15

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    show_plots: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"omnipoint_{self.omnipoint_variant}_{timestamp}"
        if self.image_size % 14 != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by OmniPoint's "
                f"patch_size (14, the ViT-L/14 convention -- see decisions.md D-009)."
            )
        if not 0.0 < self.train_split < 1.0:
            raise ValueError(f"train_split must be in (0, 1), got {self.train_split}")
        if not 0.0 <= self.intrinsics_conditioning_prob <= 1.0:
            raise ValueError(
                f"intrinsics_conditioning_prob must be in [0, 1], got "
                f"{self.intrinsics_conditioning_prob}"
            )


# ---------------------------------------------------------------------
# Training-only wrapper -- resolves the tuple-output / OmniPointCombinedLoss mismatch
# ---------------------------------------------------------------------


# DECISION plan-2026-09-11T050223-1b47bcf6/D-021
# Do NOT feed `data.py`'s full-pixel-resolution GT tensors into `OmniPointCombinedLoss`
# unchanged -- MEASURED directly (a smoke run raised `Dimensions must be equal, but are
# 56 and 4` at `compute_optimal_scale`'s `broadcast_to`): `OmniPoint`'s dense heads emit
# at the encoder's native `(H // patch_size, W // patch_size)` patch-grid resolution, per
# D-011 (`upsample_factor=1`, since `patch_size=14` is not a power of 2), while
# `CombinedOmniPointDataset` derives GT ray/distance/point/mask at the FULL input pixel
# resolution -- the two were never the same shape and D-011 already said so; Step 9 is
# where that mismatch first has to be resolved, not avoided. NEAREST-neighbor
# downsampling (never bilinear/average) is required specifically because `gt_ray` is a
# unit vector field and `gt_mask`/`valid_mask` are binary: averaging would produce
# non-unit rays and non-binary masks, silently violating both Problem Statement
# invariant 1 (`||ray||=1`) and the mask-loss's binary-target assumption. Nearest-neighbor
# picks an EXISTING full-resolution GT value verbatim per grid cell, so unit-norm and
# binary-ness survive exactly.
def _downsample_gt_to_grid(
        y_true: Tuple[keras.KerasTensor, ...],
        grid_h: int,
        grid_w: int,
) -> Tuple[keras.KerasTensor, ...]:
    """Nearest-neighbor-downsample every element of the GT 5-tuple to the heads' grid shape.

    Args:
        y_true: `(gt_ray, gt_distance, gt_point, gt_mask, valid_mask)`, each
            `(B, H, W, C)` at the dataset's full pixel resolution.
        grid_h: Target height, `OmniPoint.grid_h` (`H // patch_size`).
        grid_w: Target width, `OmniPoint.grid_w` (`W // patch_size`).

    Returns:
        The same 5-tuple, each element resized to `(B, grid_h, grid_w, C)` via
        nearest-neighbor interpolation (exact GT values, never blended).
    """
    return tuple(
        keras.ops.image.resize(
            tensor, size=(grid_h, grid_w), interpolation="nearest",
        )
        for tensor in y_true
    )


# DECISION plan-2026-09-11T050223-1b47bcf6/D-020
# Do NOT wire `OmniPointCombinedLoss` through `model.compile(loss=OmniPointCombinedLoss(...))`
# directly -- MEASURED against `keras.src.trainers.compile_utils.CompileLoss.build`: when
# `model.compile(loss=<single non-nested callable>)` and the model's own output is itself a
# nested structure (OmniPoint's 5-tuple), Keras 3.8 does `tree.map_structure(lambda x: loss,
# y_pred)`, broadcasting ONE COPY of the SAME loss instance onto EVERY output slot
# independently (`loss(y_true[i], y_pred[i])` per i), never a single call over the whole
# tuple. `OmniPointCombinedLoss.__call__` needs the FULL 5-tuple `y_true` and 4-tuple
# `y_pred` together (it computes one shared `s_star` across ray+distance+point before
# splitting into per-term losses, per D-016) -- it cannot be decomposed into 4/5
# independent per-slot losses without re-deriving `s_star` per term, which is exactly the
# "recompute the shared scale independently per loss term" anti-pattern
# `compute_optimal_scale`'s own docstring says NOT to do.
#
# This is Pre-Mortem #3's fallback (plan.md), realized as a training-only WRAPPER model
# using `add_loss` inside `call()` -- not a custom `train_step` override (the repo's hard
# invariant, `src/train/CLAUDE.md` Pattern 6 precedent: `src/train/hnet/` already
# supervises its auxiliary loss the same way, through `add_loss` inside `call()`, with a
# stock `fit()`). The REAL `OmniPoint` instance (`self.omnipoint`) is what gets
# saved/exported after training -- this wrapper exists only to make `fit()` see a scalar
# loss; it is never itself serialized. See decisions.md D-020 for the alternative
# considered (modifying `OmniPointCombinedLoss`'s own signature) and why it was rejected.
class OmniPointTrainingWrapper(keras.Model):
    """Wraps `OmniPoint` + `OmniPointCombinedLoss` for training via `add_loss`.

    `call()` accepts `(rgb, y_true)` as a single `inputs` tuple (see
    `_AddLossDatasetAdapter` below for how a batch is shaped this way), runs the real
    `OmniPoint` on `rgb`, drops its `point` output (D-016's y_pred convention has no
    `point` slot), computes `OmniPointCombinedLoss` once over the full tuples, and
    registers the batch-mean as this call's loss via `self.add_loss(...)`. `model.compile()`
    is therefore called with `loss=None` -- `fit()`/`evaluate()` read the loss from
    `self.losses` alone, exactly as `train.hnet`'s Pattern-6 trainer already does.

    Args:
        omnipoint: The real `OmniPoint` model instance (saved/exported after training).
        combined_loss: A configured `OmniPointCombinedLoss` instance.
        intrinsics_conditioning_prob: Per-sample Bernoulli probability that a given
            sample's `intrinsics_present` flag is drawn `True` (see D-029). Ignored
            when `omnipoint.enable_conditioning` is `False`.
        seed: Optional integer seed backing this wrapper's `keras.random.SeedGenerator`
            (D-029) -- matches the `ScheduledDropout`/`ClifordRNN`/etc. convention of a
            layer-held stateless RNG rather than an unseeded call, so the per-sample
            draw is reproducible across runs given the same seed.
        name: Layer name.
        **kwargs: Passthrough to the `keras.Model` base class.
    """

    def __init__(
            self,
            omnipoint: keras.Model,
            combined_loss: OmniPointCombinedLoss,
            intrinsics_conditioning_prob: float = 0.9,
            seed: Optional[int] = None,
            name: str = "omnipoint_training_wrapper",
            **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.omnipoint = omnipoint
        self.combined_loss = combined_loss
        self.intrinsics_conditioning_prob = float(intrinsics_conditioning_prob)
        # DECISION plan-2026-09-11T050223-1b47bcf6/D-029
        # Do NOT draw `intrinsics_present` from raw unseeded `numpy.random` (or a
        # Python-level `random.random()`) inside `call()` -- `call()` is traced once
        # under `tf.function` by `fit()`, so a host-side RNG call would draw ONCE at
        # trace time and bake a single frozen mask into the graph for every subsequent
        # step, never re-drawing (measured convention: `ScheduledDropout`,
        # `EnergyTransformer`, `BitLinear` and `ClifordRNN` all hold a
        # `keras.random.SeedGenerator` as layer state and pass it as `seed=` to a
        # `keras.random.*` call precisely so the draw is a graph OP that re-executes
        # every step, not a Python-side constant). This is that same pattern, applied
        # to a Bernoulli draw instead of a dropout mask.
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None,
    ):
        """Forward pass: run `OmniPoint`, compute the combined loss, `add_loss` it.

        Args:
            inputs: `(rgb, y_true)`, where `rgb` is `(B, H, W, 3)` and `y_true` is
                `OmniPointCombinedLoss`'s documented 5-tuple
                `(gt_ray, gt_distance, gt_point, gt_mask, valid_mask)`.
            training: Whether this call runs in training mode.

        Returns:
            `OmniPoint`'s own 5-tuple output (unchanged) -- returned so a caller
            inspecting `model(x)` sees the real predictions, even though `fit()` never
            reads it directly (supervision happens via `add_loss`).
        """
        rgb, y_true = inputs
        gt_ray = y_true[0]
        # DECISION plan-2026-09-11T050223-1b47bcf6/D-028
        # Do NOT leave `enable_conditioning=True` forwarding zero conditioning tensors --
        # `data.py`'s own `gt_ray` (y_true[0]) IS the intrinsics ray map: both are
        # `pinhole_ray_map(K, ...)` evaluated at the SAME full pixel resolution as `rgb`
        # (D-016), so reusing it here is the DRY choice, not a second derivation. A derived
        # K (and therefore a ray map) is ALWAYS available for every sample this pipeline
        # emits (both KITTI and MegaDepth always compute K), so `intrinsics_present` is
        # unconditionally all-True -- there is no "missing intrinsics" case in this data
        # pipeline. Sparse-depth conditioning is DELIBERATELY left unwired: no sparse-depth
        # data exists anywhere in this pipeline (neither loader synthesizes or provides a
        # sparse subsample), so `sparse_depth`/`sparse_depth_mask` are never passed --
        # passing zeros would be indistinguishable from "no signal" only by accident, and a
        # future caller adding real sparse-depth data must not mistake this branch for
        # already being wired. See decisions.md D-028 and
        # `src/train/omnipoint/README.md` for the intrinsics-only scope this implies for
        # `--enable-conditioning`.
        #
        # SUPERSEDED IN PART by D-029, directly below: `intrinsics_present` is no
        # longer the unconditional all-`True` this comment originally shipped -- see
        # D-029 for why an all-`True` flag made `L_ray` a copy task and left
        # `intrinsics_state/absent_embedding` permanently dead.
        #
        # DECISION plan-2026-09-11T050223-1b47bcf6/D-029
        # Do NOT hardwire `intrinsics_present` to all-`True` -- MEASURED (pass-2 adversarial
        # review, `findings/review-iter-1-pass2.md` concern 1): `gt_ray` (`y_true[0]`) IS the
        # literal `L_ray` supervision target (D-028), so feeding it in as the conditioning
        # input on every single step, unconditionally flagged present, turns `L_ray` into a
        # trivially-satisfiable copy task and leaves `intrinsics_state/absent_embedding`
        # exactly as dead as it was before D-028 -- D-028 traded one dead weight for another.
        # Draw a fresh PER-SAMPLE Bernoulli flag every call instead (the paper's own
        # convention, OmniPoint Supplementary Section B: "geometric conditions ... enabled
        # with a probability of 90% to preserve robustness under RGB-only inputs" --
        # `--intrinsics-conditioning-prob` defaults to 0.9). This reuses the per-sample
        # mixed-batch machinery `ConditioningInputEncoder`/`ConditioningStateEmbedding`
        # already implement and `test_conditioning.py` already exercises (D-013): passing a
        # non-None `intrinsics_present` flag makes `ConditioningInputEncoder._zero_absent_samples`
        # zero out the ray-map channels for the samples drawn absent, and
        # `ConditioningStateEmbedding` selects `absent_embedding` for exactly those same
        # samples -- both signals still agree, so nothing new needs to happen here beyond
        # generating the flag itself. See decisions.md D-029 for the direct gradient
        # measurement (before: `absent_embedding` grad == 0.0 every step; after: nonzero on
        # any batch/seed draw that includes at least one absent sample).
        intrinsics_present = keras.random.uniform(
            (keras.ops.shape(rgb)[0],), seed=self.seed_generator
        ) < self.intrinsics_conditioning_prob
        outputs = self.omnipoint(
            rgb,
            intrinsics_ray_map=gt_ray,
            intrinsics_present=intrinsics_present,
            training=training,
        )
        ray, distance, point, mask_logit, scale = outputs
        y_pred = (ray, distance, mask_logit, scale)
        y_true_grid = _downsample_gt_to_grid(
            y_true, self.omnipoint.grid_h, self.omnipoint.grid_w
        )
        per_sample_loss = self.combined_loss(y_true_grid, y_pred)
        self.add_loss(keras.ops.mean(per_sample_loss))
        return outputs


class _AddLossDatasetAdapter(keras.utils.PyDataset):
    """Repackages a `CombinedOmniPointDataset` batch as `((rgb, y_true),)`.

    A length-1 tuple return from `__getitem__` is Keras's own convention for "`x` only,
    no `y`" (`keras.src.trainers.data_adapters` unpacking rule) -- required here because
    `OmniPointTrainingWrapper.call()` supervises via `add_loss`, not a compiled
    `y_true`/`y_pred` loss (see the `OmniPointTrainingWrapper` docstring / D-020).

    Args:
        inner: The wrapped `CombinedOmniPointDataset` instance. Owns its own worker
            parallelism; this adapter does no I/O of its own and runs single-process.
    """

    def __init__(self, inner: CombinedOmniPointDataset, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        rgb, y_true = self.inner[idx]
        return (rgb, y_true),

    def on_epoch_end(self) -> None:
        self.inner.on_epoch_end()


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------


def _discover_and_split_pairs(
        config: OmniPointTrainingConfig,
) -> Tuple[
    Tuple[List[Tuple[str, str]], List[str], List[str]],
    Tuple[List[Tuple[str, str]], List[str], List[str]],
]:
    """Discover KITTI + MegaDepth pairs and split each source independently by `train_split`.

    Mirrors `train_depth_anything.py`'s own explicit-slicing convention (discover once,
    slice by a fixed fraction) rather than a shuffled/streaming split. Each of the two
    sources is split independently (not the pooled list) so both sources contribute to
    both the train and validation sets regardless of their very different sizes (92,554
    KITTI pairs vs. a much smaller local MegaDepth mirror, per D-018).

    Args:
        config: The training config (reads `kitti_root`, `megadepth_root`,
            `max_kitti_files`, `max_megadepth_files`, `train_split`).

    Returns:
        `((train_kitti, train_megadepth_rgb, train_megadepth_depth),
          (val_kitti, val_megadepth_rgb, val_megadepth_depth))`.

    Raises:
        ValueError: If discovery finds zero pairs from both sources.
    """
    kitti_pairs: List[Tuple[str, str]] = []
    if config.kitti_root is not None:
        kitti_pairs = discover_kitti_depth_pairs(
            config.kitti_root, max_files=config.max_kitti_files
        )

    mega_rgb: List[str] = []
    mega_depth: List[str] = []
    if config.megadepth_root is not None:
        mega_rgb, mega_depth = discover_megadepth_pairs(
            config.megadepth_root, max_files=config.max_megadepth_files
        )

    if not kitti_pairs and not (mega_rgb and mega_depth):
        raise ValueError(
            f"_discover_and_split_pairs: discovery found zero pairs from both "
            f"kitti_root={config.kitti_root!r} and megadepth_root={config.megadepth_root!r}."
        )

    k_split = int(len(kitti_pairs) * config.train_split)
    train_kitti, val_kitti = kitti_pairs[:k_split], kitti_pairs[k_split:]

    m_split = int(len(mega_rgb) * config.train_split)
    train_mega_rgb, val_mega_rgb = mega_rgb[:m_split], mega_rgb[m_split:]
    train_mega_depth, val_mega_depth = mega_depth[:m_split], mega_depth[m_split:]

    # A tiny (e.g. --smoke) discovery run can split a handful of pairs into an empty
    # validation set; fall back to reusing the training pairs for validation rather than
    # raising, logging loudly so this is never mistaken for a real held-out split.
    if not val_kitti and not (val_mega_rgb and val_mega_depth):
        logger.warning(
            "_discover_and_split_pairs: validation split is empty (too few discovered "
            "pairs for train_split=%s) -- reusing training pairs for validation. This "
            "is a smoke-mode fallback, never a real held-out evaluation.",
            config.train_split,
        )
        val_kitti, val_mega_rgb, val_mega_depth = train_kitti, train_mega_rgb, train_mega_depth

    logger.info(
        f"Discovered pairs: KITTI train={len(train_kitti)} val={len(val_kitti)}, "
        f"MegaDepth train={len(train_mega_rgb)} val={len(val_mega_rgb)}"
    )
    return (
        (train_kitti, train_mega_rgb, train_mega_depth),
        (val_kitti, val_mega_rgb, val_mega_depth),
    )


def build_datasets(
        config: OmniPointTrainingConfig,
) -> Tuple[_AddLossDatasetAdapter, _AddLossDatasetAdapter]:
    """Build the wrapped train/validation `CombinedOmniPointDataset` pair."""
    train_parts, val_parts = _discover_and_split_pairs(config)

    train_inner = CombinedOmniPointDataset(
        *train_parts,
        batch_size=config.batch_size,
        patch_size=config.image_size,
        is_training=True,
        workers=config.workers,
    )
    val_inner = CombinedOmniPointDataset(
        *val_parts,
        batch_size=config.batch_size,
        patch_size=config.image_size,
        is_training=False,
        workers=max(1, config.workers // 2),
    )
    return _AddLossDatasetAdapter(train_inner), _AddLossDatasetAdapter(val_inner)


# ---------------------------------------------------------------------
# MODEL + OPTIMIZER
# ---------------------------------------------------------------------


def create_model(config: OmniPointTrainingConfig) -> Tuple[keras.Model, OmniPointTrainingWrapper]:
    """Build the real `OmniPoint` model plus its training wrapper.

    Returns:
        `(omnipoint, wrapper)` -- `omnipoint` is what gets saved after training;
        `wrapper` is what gets compiled/fit.
    """
    omnipoint = create_omnipoint(
        variant=config.omnipoint_variant,
        image_shape=(config.image_size, config.image_size, 3),
        enable_conditioning=config.enable_conditioning,
    )
    combined_loss = OmniPointCombinedLoss(
        lambda_ray=config.lambda_ray,
        lambda_metric=config.lambda_metric,
        lambda_normal=config.lambda_normal,
        lambda_local=config.lambda_local,
        lambda_mask=config.lambda_mask,
    )
    wrapper = OmniPointTrainingWrapper(
        omnipoint, combined_loss,
        intrinsics_conditioning_prob=config.intrinsics_conditioning_prob,
        seed=config.seed,
    )
    return omnipoint, wrapper


def _build_lr_schedule(config: OmniPointTrainingConfig, steps_per_epoch: int):
    """Resolve `config.lr_schedule_type` into a schedule or bare float learning rate."""
    if config.lr_schedule_type == "constant":
        return config.learning_rate
    schedule_type = _LR_SCHEDULE_TYPE_MAP[config.lr_schedule_type]
    schedule_config = {
        "type": schedule_type,
        "learning_rate": config.learning_rate,
        "decay_steps": max(1, steps_per_epoch * config.epochs),
        "warmup_steps": 0,
        "alpha": 0.01,
    }
    if schedule_type == "exponential_decay":
        schedule_config["decay_rate"] = 0.96
    return learning_rate_schedule_builder(schedule_config)


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train_omnipoint(config: OmniPointTrainingConfig) -> keras.Model:
    """Train `OmniPoint` on the combined KITTI + MegaDepth pipeline.

    Returns:
        The trained (real, unwrapped) `OmniPoint` model.
    """
    set_seeds(config.seed)
    logger.info(f"Starting OmniPoint training: {config.experiment_name}")

    train_ds, val_ds = build_datasets(config)
    steps_per_epoch = len(train_ds)
    logger.info(
        f"Datasets built: {len(train_ds)} training batches, {len(val_ds)} validation "
        f"batches, batch_size={config.batch_size}, image_size={config.image_size}"
    )

    omnipoint, model = create_model(config)
    omnipoint.summary(print_fn=logger.info)

    lr_schedule = _build_lr_schedule(config, steps_per_epoch)
    optimizer = optimizer_builder(
        {
            "type": "adamw",
            "gradient_clipping_by_norm": 1.0,
            "weight_decay": config.weight_decay,
        },
        lr_schedule,
    )

    # `loss=None`: supervision happens entirely through `OmniPointTrainingWrapper.call()`'s
    # `add_loss` (see D-020) -- no custom `train_step`, matching the repo's hard invariant.
    model.compile(optimizer=optimizer)
    logger.info(f"Model compiled with {omnipoint.count_params():,} parameters (OmniPoint only)")

    common_callbacks, results_dir = create_callbacks(
        model_name=config.omnipoint_variant,
        results_dir_prefix="omnipoint",
        output_root=config.output_dir,
        monitor="val_loss",
        patience=config.patience,
        use_lr_schedule=(config.lr_schedule_type == "constant"),
        include_tensorboard=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )
    config.output_dir = str(Path(results_dir).parent)
    config.experiment_name = Path(results_dir).name
    output_dir = Path(results_dir)

    save_config_json(config, str(output_dir), "config.json")

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=common_callbacks,
        verbose=1,
    )
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds")

    if config.show_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(history.history.get("loss", []), label="train_loss")
            if "val_loss" in history.history:
                ax.plot(history.history["val_loss"], label="val_loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_title(f"OmniPoint training loss ({config.experiment_name})")
            ax.legend()
            plot_path = output_dir / "training_loss_curve.png"
            fig.savefig(str(plot_path))
            plt.close(fig)
            logger.info(f"Loss curve saved to: {plot_path}")
        except Exception as exc:  # noqa: BLE001 -- plotting must never fail a training run
            logger.warning(f"Failed to save loss curve plot: {exc}")

    try:
        inference_path = output_dir / "omnipoint_inference.keras"
        omnipoint.save(str(inference_path))
        logger.info(f"Inference model saved to: {inference_path}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to save inference model: {exc}")

    return omnipoint


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments. MUST be the first statement of `main()` -- `--help` exits 0
    with a `usage:` line and allocates nothing (no GPU setup, no dataset discovery, no
    model construction)."""
    parser = create_base_argument_parser(
        description="Train OmniPoint (universal monocular metric point-cloud model)",
        default_dataset="cifar10",
    )

    data_group = parser.add_argument_group("OmniPoint data")
    data_group.add_argument(
        "--kitti-root", type=str,
        default="/media/arxwn/data0_4tb/datasets/KITTI/data/depth",
        help="Path to the KITTI depth-benchmark root, or 'none' to skip KITTI.",
    )
    data_group.add_argument(
        "--megadepth-root", type=str,
        default="/media/arxwn/data0_4tb/datasets/Megadepth",
        help="Path to the MegaDepth dataset root, or 'none' to skip MegaDepth.",
    )
    data_group.add_argument(
        "--max-kitti-files", type=int, default=None,
        help="Cap on discovered KITTI pairs (smoke/dev runs).",
    )
    data_group.add_argument(
        "--max-megadepth-files", type=int, default=None,
        help="Cap on discovered MegaDepth pairs (smoke/dev runs).",
    )
    data_group.add_argument(
        "--train-split", type=float, default=0.9,
        help="Fraction of each source's discovered pairs used for training.",
    )
    data_group.add_argument(
        "--workers", type=int, default=8,
        help="Multiprocessing workers for the combined training dataset.",
    )

    model_group = parser.add_argument_group("OmniPoint model")
    model_group.add_argument(
        "--omnipoint-variant", type=str, default="omnipoint_base",
        choices=list(MODEL_VARIANTS.keys()),
        help="OmniPoint.MODEL_VARIANTS key.",
    )
    model_group.add_argument(
        "--enable-conditioning", action="store_true",
        help=(
            "Enable the optional geometric conditioning path (OmniPoint's "
            "enable_conditioning=True). Trains INTRINSICS conditioning only: this "
            "trainer derives a per-sample intrinsics ray map from data.py's own "
            "gt_ray (D-028) and feeds it in every step. Sparse-depth conditioning "
            "weights are NOT trained by this flag -- no sparse-depth data exists in "
            "this pipeline (see src/train/omnipoint/README.md)."
        ),
    )
    model_group.add_argument(
        "--intrinsics-conditioning-prob", type=float, default=0.9,
        help=(
            "Per-sample probability that a sample's intrinsics_present flag is drawn "
            "True each training step (D-029). Matches the OmniPoint paper's own "
            "Supplementary Section B conditioning-dropout convention (default 0.9). "
            "Ignored unless --enable-conditioning is set."
        ),
    )

    loss_group = parser.add_argument_group("OmniPointCombinedLoss weights")
    loss_group.add_argument("--lambda-ray", type=float, default=1.0)
    loss_group.add_argument("--lambda-metric", type=float, default=1.0)
    loss_group.add_argument("--lambda-normal", type=float, default=1.0)
    loss_group.add_argument("--lambda-local", type=float, default=1.0)
    loss_group.add_argument("--lambda-mask", type=float, default=1.0)

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-dir", type=str, default="results")
    output_group.add_argument("--experiment-name", type=str, default=None)
    output_group.add_argument("--seed", type=int, default=42)
    output_group.add_argument(
        "--smoke", action="store_true",
        help=(
            "Tiny integration-smoke run: small image size, tiny batch, 1 epoch, a "
            "handful of files per source. Overrides --image-size/--batch-size/--epochs/"
            "--max-kitti-files/--max-megadepth-files/--workers."
        ),
    )

    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> OmniPointTrainingConfig:
    """Build `OmniPointTrainingConfig` from parsed args, applying `--smoke` overrides last."""
    kitti_root = None if args.kitti_root.lower() == "none" else args.kitti_root
    megadepth_root = None if args.megadepth_root.lower() == "none" else args.megadepth_root

    config = OmniPointTrainingConfig(
        kitti_root=kitti_root,
        megadepth_root=megadepth_root,
        image_size=args.image_size,
        train_split=args.train_split,
        max_kitti_files=args.max_kitti_files,
        max_megadepth_files=args.max_megadepth_files,
        workers=args.workers,
        omnipoint_variant=args.omnipoint_variant,
        enable_conditioning=args.enable_conditioning,
        intrinsics_conditioning_prob=args.intrinsics_conditioning_prob,
        lambda_ray=args.lambda_ray,
        lambda_metric=args.lambda_metric,
        lambda_normal=args.lambda_normal,
        lambda_local=args.lambda_local,
        lambda_mask=args.lambda_mask,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_schedule_type=args.lr_schedule,
        patience=args.patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        show_plots=args.show_plots,
        seed=args.seed,
    )

    if args.smoke:
        config.image_size = SMOKE_IMAGE_SIZE
        config.batch_size = SMOKE_BATCH_SIZE
        config.epochs = SMOKE_EPOCHS
        config.max_kitti_files = SMOKE_MAX_FILES
        config.max_megadepth_files = SMOKE_MAX_FILES
        config.workers = SMOKE_WORKERS
        config.lr_schedule_type = "constant"
        logger.info(
            f"--smoke: overriding config to image_size={config.image_size}, "
            f"batch_size={config.batch_size}, epochs={config.epochs}, "
            f"max_kitti_files={config.max_kitti_files}, "
            f"max_megadepth_files={config.max_megadepth_files}, "
            f"workers={config.workers}, lr_schedule_type={config.lr_schedule_type}"
        )

    return config


def main(argv: Optional[List[str]] = None) -> keras.Model:
    """Entry point. `parse_arguments()` MUST run first (before any GPU/model/dataset work)."""
    args = parse_arguments(argv)

    setup_gpu(gpu_id=args.gpu)
    config = _config_from_args(args)

    logger.info(
        f"Config: model=OmniPoint-{config.omnipoint_variant} "
        f"(enable_conditioning={config.enable_conditioning}), epochs={config.epochs}, "
        f"batch={config.batch_size}, lr={config.learning_rate}, "
        f"image_size={config.image_size}"
    )

    return train_omnipoint(config)


if __name__ == "__main__":
    main()
