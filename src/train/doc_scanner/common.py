r"""Shared building blocks for the two DocScanner trainers.

The paper trains DocScanner's two modules **independently**
(arXiv:2110.14968v2 §4.3), with two different recipes, so this package has two
entry points and this module has one config, one pipeline and one ``train()``
that both drive. Nothing here branches on a script name: the single
:attr:`DocScannerTrainingConfig.stage` field selects the model, the loss, the
supervision target and the optimizer recipe, and the two entry points differ
only in the default value they bind to it.

What the paper specifies, verbatim
----------------------------------
*Localization module* (the segmenter): *"We use Adam optimizer with a batch
size of 32. The initial learning rate is set as 1x10^-4, and reduced by a
factor of 0.1 after 30 epochs. After 45 epochs, the training loss converges."*
Plus: *"we randomly replace the background of the distorted image with the
texture images from Describable Texture Dataset (DTD)."*

*Rectification module*: *"We use AdamW optimizer with a batch size of 12. The
total training iteration is set as 560k, and the learning rate reaches the
maximum 1x10^-4 after 27k iterations for learning rate warm-up."*

THREE HYPERPARAMETERS ARE THIS REPO'S CHOICE, NOT A REPRODUCTION
---------------------------------------------------------------
The paper is **silent** on all three of the following, everywhere (finding F-16
of the porting plan, established by direct raw-text extraction of the ar5iv
HTML rather than a summary). They are chosen here, defensibly, and they are
labelled as chosen -- in this docstring, on each field, and in ``README.md``.
Do not quote any of them as a reproduction of the paper.

``warmup_shape`` -- **LINEAR**, and it is not a knob at all.
    The paper gives a warmup LENGTH (27k of 560k iterations, 4.8%) and no
    curve. This port uses :class:`dl_techniques.optimization.WarmupSchedule`,
    whose ramp is linear, because it is this repo's ONE warmup implementation
    and every trainer that warms up already uses it. A second curve would be a
    new abstraction with one call site.

``weight_decay`` -- **1e-4**.
    ``AdamW`` without a stated decay is ambiguous: Keras' own default is
    ``0.004``, PyTorch's is ``0.01``, and RAFT -- the architecture Eq. 9's
    sequence loss and this whole update block are inherited from -- trains with
    ``1e-5``. 1e-4 sits between the RAFT ancestor and the framework defaults
    and is the value the sibling dense-regression ports in this repo use. It is
    applied by the optimizer and by NOTHING else: no ``kernel_regularizer``
    exists anywhere in the doc_scanner package, because AdamW's decoupled decay
    plus an L2 penalty decays the same parameter twice
    (``src/train/CLAUDE.md``).

``gradient_clipping`` -- **global norm 1.0**.
    RAFT clips at global norm 1.0, and a 12-step recurrent unroll whose loss is
    an L1 in ABSOLUTE PIXELS (up to ~288 per element early in training) is
    exactly the setting an unclipped step blows up in. 1.0 is the RAFT value,
    not a paper value.

Data
----
Two corpora, selected by :attr:`DocScannerTrainingConfig.data_source`, both
emitting the SAME contract so no downstream code branches on the source:

``synthetic``
    :mod:`dl_techniques.datasets.document_rectification.synthetic_warp`
    composes closed-form-invertible warps, so ``f_gt`` and ``g`` are both exact
    (D-036). Flat page content comes from the staged DIBCO/NoisyOffice scans;
    backgrounds come from DTD if it has been extracted, and from a procedural
    low-frequency texture if it has not (DTD ships here as a tarball). The
    background affects realism only -- never ``f_gt``, ``g`` or ``mask``.

``uvdoc``
    :mod:`dl_techniques.datasets.document_rectification.uvdoc` densifies
    UVDoc's coarse 89x61 correspondence lattice. ``load_geometry`` costs about
    1.06 s and there are only 4,032 distinct geometries behind the 20,000
    renders, so geometry is cached by ``geom_name``
    (:func:`cached_uvdoc_geometry`) and the second render of a geometry is free.
    That cache is per-process and dies with the run; ``python -m
    train.doc_scanner.stage_uvdoc_samples`` makes it durable by writing
    ``(image, f_gt, g, mask)`` sidecars under
    :data:`DEFAULT_UVDOC_CACHE_ROOT`, and when a directory for this run's
    ``image_size`` is staged there the archive is never opened at all
    (:func:`uvdoc_cache_dir`, :func:`read_uvdoc_sidecar`). The two forms emit
    the same contract; only the source of the bytes differs.

Supervision, per stage
----------------------
``segmenter``
    ``y`` is the ``(H, W, 1)`` page mask, repeated SEVEN times -- the model
    emits ``d0..d6`` for deep supervision and ``compile(loss=BCE)`` applies the
    loss to each. The total is their sum, which is U2-Net's own objective.

``rectifier``
    ``y`` is the ``(H, W, 4)`` channel stack ``[f_gt(2), g(2)]`` that
    :class:`~dl_techniques.losses.DocScannerFlowSequenceLoss` documents as its
    ``y_true``; ``y_pred`` is the ``(B, 12, H, W, 2)`` sequence the rectifier
    returns under ``training=True``. Stock ``compile(loss=...)`` / ``fit()``;
    no custom ``train_step`` anywhere (H-5). The VALIDATION pass sees a rank-4
    tensor instead, because Keras' ``test_step`` calls the model with
    ``training=False`` and that flag selects the rectifier's output RANK --
    see :class:`DocScannerRectifierObjective`, which is where the two are
    reconciled without relaxing the sequence loss's own guard.

Import direction (D-031)
------------------------
This module imports the loss from :mod:`dl_techniques.losses`. The reverse --
a module-scope ``import dl_techniques.losses`` inside the ``doc_scanner``
model package -- would close the import cycle that the loss's own anchor says
must stay open. Nothing here is imported by either package.

Public surface:
    * :class:`DocScannerTrainingConfig` -- the run knobs. Every field is
      consumed by something other than the config dump; the class is REGISTERED
      in ``tests/test_train/test_config_fields_are_live.py``.
    * :func:`stage_defaults`, :func:`add_common_arguments`,
      :func:`config_from_args` -- the CLI half.
    * :func:`require_training_data` -- the gate the entry points run BEFORE
      ``setup_gpu``.
    * :func:`create_dataset`, :func:`build_model`, :func:`train`.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.datasets.document_rectification import (
    DegenerateWarpError,
    UVDocError,
    UVDocGeometry,
    UVDocSource,
    load_geometry,
    load_image,
    load_rgb,
    render_sample,
    sample_warp,
)
from dl_techniques.losses import DocScannerFlowSequenceLoss
from dl_techniques.models.vision.image_restoration.doc_scanner.components import (
    FLOW_CHANNELS,
    REFINE_ITERATIONS,
    SPATIAL_DIVISOR,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.model import (
    create_doc_scanner_rectifier,
    create_doc_scanner_segmenter,
)
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
)
from dl_techniques.utils.logger import logger
from train.common import create_callbacks, set_seeds
from train.common.config_io import save_config_json
from train.common.run_io import save_training_history_json

__all__ = [
    "DEFAULT_BACKGROUNDS_ROOT",
    "DEFAULT_PAGES_ROOT",
    "DEFAULT_UVDOC_CACHE_ROOT",
    "DEFAULT_UVDOC_ROOT",
    "DOC_SCANNER_SOURCES",
    "DOC_SCANNER_STAGES",
    "DocScannerDataError",
    "DocScannerRectifierObjective",
    "DocScannerTrainingConfig",
    "MissingTrainingDataError",
    "SEGMENTER_HEAD_COUNT",
    "SOURCE_SYNTHETIC",
    "SOURCE_UVDOC",
    "STAGE_RECTIFIER",
    "STAGE_SEGMENTER",
    "UVDOC_CACHE_GEOMETRY_DIR",
    "UVDOC_CACHE_MANIFEST",
    "UVDOC_CACHE_RENDER_DIR",
    "UVDOC_CACHE_SCHEMA",
    "add_common_arguments",
    "build_loss",
    "build_model",
    "build_optimizer",
    "cached_uvdoc_geometry",
    "collect_background_paths",
    "collect_page_paths",
    "collect_uvdoc_sample_ids",
    "config_from_args",
    "create_dataset",
    "read_uvdoc_sidecar",
    "rectifier_target",
    "require_training_data",
    "segmenter_target",
    "stage_defaults",
    "staged_uvdoc_sample_ids",
    "synthetic_sample",
    "train",
    "uvdoc_cache_dir",
    "uvdoc_sample",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_SEGMENTER: str = "segmenter"
"""Stage 1: the U2NET-P localization module. Trained with BCE + Adam."""

STAGE_RECTIFIER: str = "rectifier"
"""Stage 2: the 12-iteration progressive rectifier. Sequence loss + AdamW."""

DOC_SCANNER_STAGES: Tuple[str, ...] = (STAGE_SEGMENTER, STAGE_RECTIFIER)
"""The two independently-trained modules (§4.3). There is no joint recipe."""

SOURCE_SYNTHETIC: str = "synthetic"
"""Warped pages generated on the fly; ``f_gt`` and ``g`` are exact."""

SOURCE_UVDOC: str = "uvdoc"
"""Real captures from the staged UVDoc corpus; ``f_gt`` is densified."""

DOC_SCANNER_SOURCES: Tuple[str, ...] = (SOURCE_SYNTHETIC, SOURCE_UVDOC)

SEGMENTER_HEAD_COUNT: int = 7
"""``d0..d6``. DocScannerSegmenter emits seven side maps for deep supervision
and every one of them is supervised against the SAME mask, so the target is
this tensor repeated seven times. Sourced from the model rather than typed:
see :func:`segmenter_target`."""

RECTIFIER_TARGET_CHANNELS: int = 2 * FLOW_CHANNELS
"""4 -- ``[f_gt(2), g(2)]``, the stack ``DocScannerFlowSequenceLoss`` reads."""

RGB_CHANNELS: int = 3

DEFAULT_PAGES_ROOT: str = (
    "/media/arxwn/data0_4tb/datasets/doc_res/binarization"
)
"""Flat page content for the synthetic generator. The staged DIBCO / hDIBCO /
NoisyOffice scans are the only real document-page corpus on this machine (809
images); they are read here as page RASTERS only -- none of their binarization
ground truth is used."""

DEFAULT_BACKGROUNDS_ROOT: str = "/media/arxwn/data0_4tb/datasets/dtd"
"""DTD, the paper's own named background source. OPTIONAL: it ships here as an
unextracted tarball, and a missing/empty root falls back to a procedural
texture rather than failing the run -- the background is composited AFTER the
geometry and cannot perturb ``f_gt``, ``g`` or ``mask``."""

DEFAULT_UVDOC_ROOT: str = (
    "/media/arxwn/data0_4tb/datasets/doc_scanner/uvdoc/UVDoc_final.zip"
)
"""The staged UVDoc archive. ``UVDocSource`` reads members straight out of the
zip, so nothing is extracted."""

DEFAULT_UVDOC_CACHE_ROOT: str = (
    "/media/arxwn/data0_4tb/datasets/doc_scanner/uvdoc/staged"
)
"""Root of the PRECOMPUTED UVDoc sidecar corpus, written by
``train.doc_scanner.stage_uvdoc_samples``. Optional: with nothing staged there
the ``uvdoc`` source reads the archive and densifies in-process, which is
correct but pays ~1.06 s per DISTINCT geometry on every fresh worker."""

UVDOC_CACHE_SCHEMA: int = 1
"""Sidecar schema version, recorded in every ``manifest.json`` and checked on
read. A reader that finds a different number REFUSES the directory rather than
silently mixing two layouts."""

UVDOC_CACHE_MANIFEST: str = "manifest.json"
"""Marker AND contract of one staged size-scoped directory. Its presence is
what makes a directory a usable cache; it records the stored ``height`` and
``width``, the ``seed`` and the archive the sidecars were derived from."""

UVDOC_CACHE_GEOMETRY_DIR: str = "geometry"
"""``<geom_name>.npz`` -- ``f_gt``, ``g``, ``mask``. Keyed by GEOMETRY, not by
render: 20,000 UVDoc renders share 4,032 geometries, so a per-render layout
would store the same 1.7 MB triple five times over and, worse, would invite a
writer that densifies once per render (~6 h instead of ~1.2 h)."""

UVDOC_CACHE_RENDER_DIR: str = "render"
"""``<sample_id>.npz`` -- the resampled render plus the ``geom_name`` that
joins it to its geometry sidecar. The join key lives IN the render file rather
than in a shared index, so staging stays purely additive: a new render is one
new file and no read-modify-write of anything global."""

UVDOC_CACHE_IMAGE_KEY: str = "image"
UVDOC_CACHE_GEOM_NAME_KEY: str = "geom_name"
UVDOC_CACHE_F_GT_KEY: str = "f_gt"
UVDOC_CACHE_G_KEY: str = "g"
UVDOC_CACHE_MASK_KEY: str = "mask"
"""The five ``.npz`` array names. Named constants rather than literals because
the writer lives in a different module from this reader and a typo in either
would be a ``KeyError`` at the first training batch, not at staging time."""

IMAGE_SUFFIXES: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
)
"""Suffixes accepted when walking a page or background root."""

EXCLUDED_DIR_NAMES: frozenset = frozenset({"_archives", "_prompts"})
"""Directories skipped when walking the page root: ``_archives`` holds the
original download tarballs and ``_prompts`` holds DocRes's precomputed sidecar
tensors, neither of which is a page."""

PAGE_MAX_SIDE: int = 1024
"""Longest side a page raster is downscaled to on load. A 288 px sample never
resolves more, and the LRU cache below holds a bounded number of them."""

PAGE_CACHE_SIZE: int = 48
"""Decoded pages kept in memory. At <= 1024 px RGB float that is under 600 MB
worst case and, in practice, far less."""

GEOMETRY_CACHE_SIZE: int = 256
"""Densified UVDoc geometries kept in memory. UVDoc fans 20,000 renders out of
4,032 geometries and ``load_geometry`` costs ~1.06 s, essentially all of it
scattered-data interpolation, so re-densifying per render would turn a 20,000
sample epoch into ~6 hours of pure scipy."""

PROCEDURAL_BACKGROUND_SIZE: int = 24
"""Side of the low-frequency random tile used when no background corpus is
available. It is deliberately tiny: ``render_sample`` resamples it up to the
sample size, which turns it into a smooth gradient rather than pixel noise."""

MAX_SAMPLE_ATTEMPTS: int = 8
"""How many times a sample producer re-draws before giving up. Both corpora
have a legitimate reject path -- a degenerate warp, an unusable densification
-- and a single rejected draw must not take the epoch down."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DocScannerDataError(RuntimeError):
    """Base class for every data-availability failure in this package."""


class MissingTrainingDataError(DocScannerDataError):
    """No usable training corpus for the requested source.

    Raised by :func:`require_training_data` BEFORE the entry point claims a
    GPU, and carrying the REAL reason (which root was looked at, what was
    found there, and what to do about it) rather than an empty-glob mystery
    later on.
    """


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DocScannerTrainingConfig:
    """Knobs for one DocScanner training run, either stage.

    Defaults are the SHARED ones; the per-stage recipe the paper specifies is
    applied by :func:`stage_defaults` and bound as the argparse defaults of the
    matching entry point, so ``--help`` shows the recipe that script will
    actually run.

    Every field is read by something other than the config dump. A field that
    only reaches ``save_config_json`` is a knob that silently does nothing and
    is deleted rather than wired (``src/train/CLAUDE.md``,
    ``tests/test_train/test_config_fields_are_live.py``, where this class is
    REGISTERED).

    :param stage: ``"segmenter"`` or ``"rectifier"``. Deliberately NOT a CLI
        flag -- it is fixed by which script you ran, and a ``--stage`` flag
        would let ``train_doc_scanner_segmenter.py --stage rectifier`` train
        the other module under the wrong name and the wrong recipe.
    :param data_source: ``"synthetic"`` or ``"uvdoc"``.
    :param pages_root: Root walked for flat page rasters (synthetic only).
    :param backgrounds_root: Root walked for background textures. Optional; an
        empty or missing root falls back to a procedural texture.
    :param uvdoc_root: ``UVDoc_final.zip`` or a staged UVDoc directory.
    :param uvdoc_cache_root: Root of the precomputed UVDoc sidecar corpus
        (:data:`DEFAULT_UVDOC_CACHE_ROOT`). When a size-scoped directory
        exists under it for this run's ``image_size``, the ``uvdoc`` source
        reads staged ``(image, f_gt, g, mask)`` from there and never opens the
        archive; when it does not, the archive path is used unchanged. Set it
        to ``""`` to force the archive path.
    :param model_variant: A key of the stage model's ``MODEL_VARIANTS``.
    :param image_size: Square sample size. Must be a positive multiple of 32:
        ``SPATIAL_DIVISOR`` (8) is the rectifier's constraint, but the
        segmenter's U2NET-P pools five times, so 32 is the binding one.
    :param batch_size: Samples per optimizer step. The paper's own values are
        32 (segmenter) and 12 (rectifier).
    :param epochs: Training epochs.
    :param steps_per_epoch: Steps per epoch; the dataset repeats forever, so
        this is what defines an epoch.
    :param validation_steps: Validation batches per epoch.
    :param learning_rate: Peak learning rate (the paper's 1e-4 for both).
    :param warmup_epochs: Linear warmup length, in epochs, for the rectifier's
        cosine schedule. 0 disables warmup, which is the segmenter's recipe.
        THE CURVE IS THIS REPO'S CHOICE; only the length is the paper's.
    :param lr_drop_epoch: Epoch at which the segmenter's step schedule drops.
        The paper's 30.
    :param lr_drop_factor: Multiplier applied at that drop. The paper's 0.1.
    :param final_lr_fraction: Cosine floor as a fraction of the peak
        (rectifier). The paper gives no floor; a small non-zero one keeps the
        last iterations learning.
    :param weight_decay: Decoupled AdamW decay (rectifier). **THIS REPO'S
        CHOICE** -- see the module docstring. Applied by the optimizer ONLY.
    :param gradient_clipping: Global-norm gradient clip. **THIS REPO'S
        CHOICE** -- see the module docstring.
    :param synthetic_samples: Size of the synthetic index space. Each index is
        a deterministic seed, so this is both "how many distinct samples exist"
        and the thing the train/validation split partitions.
    :param max_pages: Cap on page rasters used, or ``None`` for all of them.
    :param max_geometries: Cap on UVDoc renders used, or ``None`` for all.
    :param shuffle_buffer: Shuffle buffer over the sample index / id worklist.
    :param val_split: Fraction of the worklist held out for validation.
    :param seed: Seed for ``set_seeds`` AND the root of every per-sample seed.
    :param patience: Early-stopping patience.
    :param output_dir: Root under which the timestamped run directory is made.
    """

    # DECISION plan-2026-09-10T065432-05fcb6dd/D-045
    # `stage` is the ONE config field with no CLI flag, and that hole in the
    # "every field has a flag" contract is deliberate. Do NOT add `--stage`
    # for symmetry: it would let `train_doc_scanner_segmenter.py --stage
    # rectifier` train the OTHER module while `--help` still advertised the
    # segmenter's batch 32 and its epoch-30 drop, while the run directory was
    # still named `doc_scanner_segmenter_*`, and while `config.json` recorded
    # the recipe the user never got. The exemption is checked from both ends by
    # `TestTheTwoStagesAreDistinct`: the field must be unreachable from argv,
    # and the two scripts must bind DIFFERENT values to it.
    # See decisions.md D-045.
    stage: str = STAGE_SEGMENTER
    data_source: str = SOURCE_SYNTHETIC

    pages_root: str = DEFAULT_PAGES_ROOT
    backgrounds_root: str = DEFAULT_BACKGROUNDS_ROOT
    uvdoc_root: str = DEFAULT_UVDOC_ROOT
    uvdoc_cache_root: str = DEFAULT_UVDOC_CACHE_ROOT

    model_variant: str = "docscanner-l"
    image_size: int = 288

    batch_size: int = 32
    epochs: int = 45
    steps_per_epoch: int = 200
    validation_steps: int = 20

    learning_rate: float = 1e-4
    warmup_epochs: int = 0
    lr_drop_epoch: int = 30
    lr_drop_factor: float = 0.1
    final_lr_fraction: float = 0.01

    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0

    synthetic_samples: int = 20000
    max_pages: Optional[int] = None
    max_geometries: Optional[int] = None

    shuffle_buffer: int = 256
    val_split: float = 0.05
    seed: int = 42
    patience: int = 15
    output_dir: str = "results"

    def __post_init__(self) -> None:
        """Validate the knobs at construction time.

        :raises ValueError: On an unknown stage or source, a non-positive
            count, an ``image_size`` that is not a multiple of 32, or a
            fraction outside its range. NOT on an unknown ``model_variant``:
            ``from_variant`` is the one authority on the legal keys and it
            checks when :func:`build_model` runs.
        """
        if self.stage not in DOC_SCANNER_STAGES:
            raise ValueError(
                f"stage must be one of {list(DOC_SCANNER_STAGES)}, got "
                f"{self.stage!r}"
            )
        if self.data_source not in DOC_SCANNER_SOURCES:
            raise ValueError(
                f"data_source must be one of {list(DOC_SCANNER_SOURCES)}, got "
                f"{self.data_source!r}"
            )

        # 32, not SPATIAL_DIVISOR: the rectifier needs a multiple of 8, but the
        # segmenter's U2NET-P pools five times and its skip connections only
        # line up when every one of those halvings is exact.
        segmenter_divisor = 4 * SPATIAL_DIVISOR
        if self.image_size <= 0 or self.image_size % segmenter_divisor != 0:
            raise ValueError(
                f"image_size must be a positive multiple of {segmenter_divisor}"
                f", got {self.image_size}. The rectifier needs a multiple of "
                f"{SPATIAL_DIVISOR} and the segmenter's five pooling stages "
                "need a multiple of 32; catching it here means the run fails "
                "at startup rather than part-way through the first epoch."
            )

        positive = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "validation_steps": self.validation_steps,
            "lr_drop_epoch": self.lr_drop_epoch,
            "synthetic_samples": self.synthetic_samples,
            "shuffle_buffer": self.shuffle_buffer,
            "patience": self.patience,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {self.warmup_epochs}"
            )
        if not 0.0 < self.lr_drop_factor <= 1.0:
            raise ValueError(
                "lr_drop_factor is a multiplier applied at the drop and must "
                f"lie in (0, 1]; got {self.lr_drop_factor}"
            )
        if not 0.0 <= self.final_lr_fraction < 1.0:
            raise ValueError(
                "final_lr_fraction is the cosine floor as a fraction of the "
                f"peak and must lie in [0, 1); got {self.final_lr_fraction}"
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if self.gradient_clipping <= 0.0:
            raise ValueError(
                "gradient_clipping is a global norm and must be positive; got "
                f"{self.gradient_clipping}"
            )
        if not 0.0 < self.val_split < 1.0:
            raise ValueError(
                f"val_split must be in (0, 1), got {self.val_split}"
            )
        for name, value in (
                ("max_pages", self.max_pages),
                ("max_geometries", self.max_geometries),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None, got {value}")


def stage_defaults(stage: str) -> DocScannerTrainingConfig:
    """The paper's recipe for one stage, as a config.

    Interface contract -- three callers (:func:`add_common_arguments`, which
    binds these as the argparse defaults; the two entry points, through it; and
    the CLI guard, which compares every probe value against them):

    * Parameters: ``stage``, one of :data:`DOC_SCANNER_STAGES`.
    * Returns: a validated :class:`DocScannerTrainingConfig`.
    * Failure mode: ``ValueError`` on an unknown stage, from
      ``__post_init__``.

    The two rows below are the ONLY place the per-stage recipe is written
    down. An entry point that hard-coded its own batch size would be a second
    copy of the paper's numbers, and the CLI guard drives THIS function.

    :param stage: ``"segmenter"`` or ``"rectifier"``.
    :type stage: str
    :return: The stage's defaults.
    :rtype: DocScannerTrainingConfig
    """
    base = DocScannerTrainingConfig(stage=stage)
    if stage == STAGE_SEGMENTER:
        # "Adam ... batch size of 32 ... initial learning rate 1e-4, reduced by
        # a factor of 0.1 after 30 epochs. After 45 epochs, the training loss
        # converges."
        return replace(
            base,
            batch_size=32,
            epochs=45,
            learning_rate=1e-4,
            warmup_epochs=0,
            lr_drop_epoch=30,
            lr_drop_factor=0.1,
        )
    # "AdamW ... batch size of 12 ... learning rate reaches the maximum 1e-4
    # after 27k iterations for learning rate warm-up" out of 560k total. The
    # paper counts ITERATIONS; this port counts epochs, and 27k/560k = 4.8% of
    # the run, which is the ratio reproduced here (2 of 40 epochs = 5%).
    return replace(
        base,
        batch_size=12,
        epochs=40,
        learning_rate=1e-4,
        warmup_epochs=2,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_arguments(
        parser: argparse.ArgumentParser,
        stage: str,
) -> argparse.ArgumentParser:
    """Register every config flag, defaulted to ``stage``'s recipe.

    ``--gpu`` is deliberately NOT here: it acts on the process, is consumed by
    ``setup_gpu`` in ``main()`` and is not a config field. ``--stage`` is not
    here either -- see :class:`DocScannerTrainingConfig`.

    :param parser: The parser to extend.
    :type parser: argparse.ArgumentParser
    :param stage: Which stage's defaults to bind.
    :type stage: str
    :return: The same parser, for chaining.
    :rtype: argparse.ArgumentParser
    """
    defaults = stage_defaults(stage)

    parser.add_argument(
        "--data-source", type=str, default=defaults.data_source,
        choices=list(DOC_SCANNER_SOURCES),
        help="Which corpus to train on.",
    )
    parser.add_argument(
        "--pages-root", type=str, default=defaults.pages_root,
        help="Root walked for flat page rasters (synthetic source).",
    )
    parser.add_argument(
        "--backgrounds-root", type=str, default=defaults.backgrounds_root,
        help=(
            "Root walked for background textures (DTD). Optional: an empty or "
            "missing root falls back to a procedural texture."
        ),
    )
    parser.add_argument(
        "--uvdoc-root", type=str, default=defaults.uvdoc_root,
        help="UVDoc_final.zip or a staged UVDoc directory.",
    )
    parser.add_argument(
        "--uvdoc-cache-root", type=str, default=defaults.uvdoc_cache_root,
        help=(
            "Root of the precomputed UVDoc sidecar corpus written by "
            "`python -m train.doc_scanner.stage_uvdoc_samples`. Used when a "
            "directory for this --image-size exists under it; otherwise the "
            "archive is read and densified in-process. Pass an empty string "
            "to force the archive path."
        ),
    )
    parser.add_argument(
        "--model-variant", type=str, default=defaults.model_variant,
        help="DocScanner variant key.",
    )
    parser.add_argument(
        "--image-size", type=int, default=defaults.image_size,
        help="Square sample size; must be a multiple of 32.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=defaults.batch_size,
        help="Samples per optimizer step.",
    )
    parser.add_argument(
        "--epochs", type=int, default=defaults.epochs,
        help="Training epochs.",
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=defaults.steps_per_epoch,
        help="Steps per epoch (the dataset repeats forever).",
    )
    parser.add_argument(
        "--validation-steps", type=int, default=defaults.validation_steps,
        help="Validation batches per epoch.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.learning_rate,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=defaults.warmup_epochs,
        help=(
            "Linear warmup length in epochs. The CURVE is this repo's choice; "
            "the paper specifies a length only."
        ),
    )
    parser.add_argument(
        "--lr-drop-epoch", type=int, default=defaults.lr_drop_epoch,
        help="Epoch at which the segmenter's step schedule drops.",
    )
    parser.add_argument(
        "--lr-drop-factor", type=float, default=defaults.lr_drop_factor,
        help="Multiplier applied at that drop.",
    )
    parser.add_argument(
        "--final-lr-fraction", type=float, default=defaults.final_lr_fraction,
        help="Cosine floor as a fraction of the peak (rectifier).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.weight_decay,
        help=(
            "Decoupled AdamW weight decay (never also an L2 regularizer). "
            "THIS REPO'S CHOICE -- the paper states no value."
        ),
    )
    parser.add_argument(
        "--gradient-clipping", type=float, default=defaults.gradient_clipping,
        help=(
            "Global-norm gradient clip. THIS REPO'S CHOICE -- the paper states "
            "no value."
        ),
    )
    parser.add_argument(
        "--synthetic-samples", type=int, default=defaults.synthetic_samples,
        help="Size of the synthetic index space (one index = one seed).",
    )
    parser.add_argument(
        "--max-pages", type=int, default=defaults.max_pages,
        help="Cap on page rasters used; omit for all of them.",
    )
    parser.add_argument(
        "--max-geometries", type=int, default=defaults.max_geometries,
        help="Cap on UVDoc renders used; omit for all of them.",
    )
    parser.add_argument(
        "--shuffle-buffer", type=int, default=defaults.shuffle_buffer,
        help="Shuffle buffer over the sample worklist.",
    )
    parser.add_argument(
        "--val-split", type=float, default=defaults.val_split,
        help="Fraction of the worklist held out for validation.",
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.seed, help="Random seed.",
    )
    parser.add_argument(
        "--patience", type=int, default=defaults.patience,
        help="Early-stopping patience.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=defaults.output_dir,
        help="Root for the timestamped run directory.",
    )
    return parser


def config_from_args(
        args: argparse.Namespace,
        stage: str,
) -> DocScannerTrainingConfig:
    """Build a config from a parsed namespace.

    The ONE wiring site between :func:`add_common_arguments` and
    :class:`DocScannerTrainingConfig`; a flag that does not arrive here
    silently does nothing, which ``tests/test_train/test_doc_scanner/`` pins.

    :param args: A namespace produced by a parser carrying the common flags.
    :type args: argparse.Namespace
    :param stage: The stage the entry point fixes.
    :type stage: str
    :return: The config.
    :rtype: DocScannerTrainingConfig
    """
    return DocScannerTrainingConfig(
        stage=stage,
        data_source=args.data_source,
        pages_root=args.pages_root,
        backgrounds_root=args.backgrounds_root,
        uvdoc_root=args.uvdoc_root,
        uvdoc_cache_root=args.uvdoc_cache_root,
        model_variant=args.model_variant,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        lr_drop_epoch=args.lr_drop_epoch,
        lr_drop_factor=args.lr_drop_factor,
        final_lr_fraction=args.final_lr_fraction,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        synthetic_samples=args.synthetic_samples,
        max_pages=args.max_pages,
        max_geometries=args.max_geometries,
        shuffle_buffer=args.shuffle_buffer,
        val_split=args.val_split,
        seed=args.seed,
        patience=args.patience,
        output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------------
# Worklists
# ---------------------------------------------------------------------------


def _walk_images(root: Path) -> List[Path]:
    """Every image file under ``root``, sorted, skipping the excluded dirs.

    :param root: Directory to walk. A missing root yields ``[]``.
    :type root: Path
    :return: Sorted image paths.
    :rtype: List[Path]
    """
    if not root.is_dir():
        return []
    found: List[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories if name not in EXCLUDED_DIR_NAMES
        ]
        for filename in filenames:
            if filename.lower().endswith(IMAGE_SUFFIXES):
                found.append(Path(directory) / filename)
    return sorted(found)


def collect_page_paths(config: DocScannerTrainingConfig) -> List[Path]:
    """Flat page rasters for the synthetic generator.

    :param config: The run config; reads ``pages_root`` and ``max_pages``.
    :type config: DocScannerTrainingConfig
    :return: Sorted page paths, capped at ``max_pages``.
    :rtype: List[Path]
    """
    paths = _walk_images(Path(config.pages_root))
    if config.max_pages is not None:
        paths = paths[: config.max_pages]
    return paths


def collect_background_paths(config: DocScannerTrainingConfig) -> List[Path]:
    """Background textures. May legitimately be empty.

    :param config: The run config; reads ``backgrounds_root``.
    :type config: DocScannerTrainingConfig
    :return: Sorted texture paths, possibly empty.
    :rtype: List[Path]
    """
    return _walk_images(Path(config.backgrounds_root))


def uvdoc_cache_dir(
        config: DocScannerTrainingConfig,
) -> Optional[Path]:
    """The staged sidecar directory for this run, or ``None``.

    Interface contract -- three callers in this module
    (:func:`collect_uvdoc_sample_ids`, :func:`uvdoc_sample`,
    :func:`require_training_data`) plus the writer,
    ``train.doc_scanner.stage_uvdoc_samples``, which builds the SAME path from
    the same constants rather than restating the layout:

    * Parameters: ``config``; reads ``uvdoc_cache_root`` and ``image_size``.
    * Returns: ``<uvdoc_cache_root>/<S>x<S>`` when that directory holds a
      readable, schema-matching ``manifest.json``; ``None`` when the root is
      empty/unset or nothing is staged for this size. ``None`` is the ordinary
      "no cache, read the archive" answer and is NOT an error. The training
      config is square-only, so only a square staged directory is reachable
      from here; the writer can emit non-square ones and the guard suite reads
      those through :func:`read_uvdoc_sidecar` directly.
    * Failure mode: :class:`DocScannerDataError` when a manifest IS present but
      unreadable or from another schema/size. A stale cache is reported, never
      silently half-used -- reading v1 sidecars with a v2 reader is exactly the
      class of defect that shows up as a training curve, not an exception.

    :param config: The run config.
    :type config: DocScannerTrainingConfig
    :return: The staged directory, or ``None``.
    :rtype: Optional[Path]
    :raises DocScannerDataError: On an unreadable or mismatched manifest.
    """
    root = str(config.uvdoc_cache_root or "").strip()
    if not root:
        return None
    directory = Path(root) / f"{config.image_size}x{config.image_size}"
    manifest_path = directory / UVDOC_CACHE_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise DocScannerDataError(
            f"the UVDoc sidecar manifest {str(manifest_path)!r} exists but is "
            f"not readable JSON ({error}). Re-run `python -m "
            "train.doc_scanner.stage_uvdoc_samples`, or pass "
            "--uvdoc-cache-root '' to read the archive directly."
        ) from error
    if manifest.get("schema") != UVDOC_CACHE_SCHEMA:
        raise DocScannerDataError(
            f"{str(manifest_path)!r} records sidecar schema "
            f"{manifest.get('schema')!r}, but this reader is schema "
            f"{UVDOC_CACHE_SCHEMA}. Re-stage that directory or point "
            "--uvdoc-cache-root elsewhere."
        )
    if (
            int(manifest.get("height", -1)) != int(config.image_size)
            or int(manifest.get("width", -1)) != int(config.image_size)
    ):
        raise DocScannerDataError(
            f"{str(manifest_path)!r} records "
            f"{manifest.get('height')!r}x{manifest.get('width')!r} inside a "
            f"directory named for {config.image_size}x{config.image_size}. "
            "The sidecars and their location disagree; nothing here is safe "
            "to read."
        )
    return directory


def staged_uvdoc_sample_ids(directory: Path) -> List[str]:
    """Sorted render ids present in a staged sidecar directory.

    Derived by LISTING ``render/``, not by reading an index file, so a staged
    corpus is exactly the set of render sidecars that were actually written --
    an interrupted staging run leaves a smaller corpus, never a corpus whose
    index promises files that are not there.

    :param directory: A directory returned by :func:`uvdoc_cache_dir`.
    :type directory: Path
    :return: Sorted sample ids (possibly empty).
    :rtype: List[str]
    """
    renders = directory / UVDOC_CACHE_RENDER_DIR
    if not renders.is_dir():
        return []
    return sorted(path.stem for path in renders.glob("*.npz"))


def read_uvdoc_sidecar(
        directory: Path,
        sample_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one staged ``(image, f_gt, g, mask)`` quadruple.

    Interface contract -- two callers, :func:`uvdoc_sample` here and the
    writer's own round-trip guard:

    * Parameters: ``directory`` from :func:`uvdoc_cache_dir`; ``sample_id``, a
      render id.
    * Returns the SAME four arrays, in the same shapes, dtypes, units and
      channel order, that ``load_image`` + ``load_geometry`` return for that
      render -- ``image`` ``(H, W, 3)`` float32 in ``[0, 1]``, ``f_gt`` and
      ``g`` ``(H, W, 2)`` float32 absolute pixels, ``mask`` ``(H, W, 1)``
      float32 in ``{0, 1}``. The one measured difference is that ``image`` is
      stored 8-bit, so it agrees with ``load_image`` to within 1/255 rather
      than exactly (see the writer's D-050).
    * Failure mode: :class:`UVDocError` naming the missing or malformed file --
      the SAME exception type the archive path raises, so ``uvdoc_sample``'s
      one reject path covers both corpus forms.

    :param directory: The staged sidecar directory.
    :type directory: Path
    :param sample_id: Render id.
    :type sample_id: str
    :return: ``(image, f_gt, g, mask)``.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    :raises UVDocError: On a missing or malformed sidecar.
    """
    render_path = directory / UVDOC_CACHE_RENDER_DIR / f"{sample_id}.npz"
    try:
        with np.load(render_path) as payload:
            stored_image = np.asarray(payload[UVDOC_CACHE_IMAGE_KEY])
            geometry_name = str(payload[UVDOC_CACHE_GEOM_NAME_KEY])
    except (OSError, ValueError, KeyError) as error:
        raise UVDocError(
            f"unreadable UVDoc render sidecar {str(render_path)!r}: {error}"
        ) from error

    geometry_path = (
        directory / UVDOC_CACHE_GEOMETRY_DIR / f"{geometry_name}.npz"
    )
    try:
        with np.load(geometry_path) as payload:
            f_gt = np.asarray(payload[UVDOC_CACHE_F_GT_KEY])
            g = np.asarray(payload[UVDOC_CACHE_G_KEY])
            stored_mask = np.asarray(payload[UVDOC_CACHE_MASK_KEY])
    except (OSError, ValueError, KeyError) as error:
        raise UVDocError(
            f"render sidecar {str(render_path)!r} names geometry "
            f"{geometry_name!r}, but {str(geometry_path)!r} is unreadable: "
            f"{error}"
        ) from error

    image = stored_image.astype(np.float32) / np.float32(255.0)
    return image, f_gt, g, stored_mask.astype(np.float32)


def collect_uvdoc_sample_ids(config: DocScannerTrainingConfig) -> List[str]:
    """Render ids of the staged UVDoc corpus.

    Prefers the precomputed sidecar corpus when one exists for this
    ``image_size`` -- the worklist must name what the sample producer can
    actually read, and a worklist of all 20,000 archive ids against a
    300-render cache would miss it 98% of the time.

    :param config: The run config; reads ``uvdoc_cache_root``, ``uvdoc_root``,
        ``image_size`` and ``max_geometries``.
    :type config: DocScannerTrainingConfig
    :return: Sorted sample ids, capped at ``max_geometries``.
    :rtype: List[str]
    :raises MissingTrainingDataError: If the archive cannot be opened.
    """
    cache = uvdoc_cache_dir(config)
    if cache is not None:
        ids = staged_uvdoc_sample_ids(cache)
        if ids:
            if config.max_geometries is not None:
                ids = ids[: config.max_geometries]
            return ids
        logger.warning(
            "DocScanner: %s holds a manifest but no render sidecar; falling "
            "back to the archive at %r.", cache, config.uvdoc_root,
        )
    try:
        with UVDocSource(config.uvdoc_root) as source:
            ids = sorted(source.sample_ids())
    except UVDocError as error:
        raise MissingTrainingDataError(
            f"the UVDoc corpus at {config.uvdoc_root!r} is not readable: "
            f"{error}. Stage it with "
            "`python -m train.doc_scanner.prepare_doc_scanner_data` (27.5 GB) "
            "or train on --data-source synthetic, which needs no download."
        ) from error
    if config.max_geometries is not None:
        ids = ids[: config.max_geometries]
    return ids


def require_training_data(config: DocScannerTrainingConfig) -> None:
    """Fail with the REAL reason before anything expensive happens.

    Interface contract -- two callers, both entry points, and it is called
    BEFORE ``setup_gpu`` in each. Read-only and idempotent: it opens the UVDoc
    archive to list members and closes it again, and it walks the page root.
    Calling it early only moves the failure ahead of the GPU claim.

    A missing BACKGROUND corpus is not an error and does not raise: the
    background is composited after the geometry and cannot perturb ``f_gt``,
    ``g`` or ``mask``, so a run without DTD is a less realistic run, not a
    wrong one. It logs a warning instead.

    :param config: The run config.
    :type config: DocScannerTrainingConfig
    :raises MissingTrainingDataError: If the selected source has no usable
        corpus, naming the root, what was found and the alternative.
    """
    if config.data_source == SOURCE_SYNTHETIC:
        root = Path(config.pages_root)
        if not root.is_dir():
            raise MissingTrainingDataError(
                f"the synthetic generator needs flat page content, and "
                f"{config.pages_root!r} is not a directory. Point "
                "--pages-root at a tree of document scans (the staged DIBCO / "
                "NoisyOffice corpus is the default) or stage UVDoc and pass "
                "--data-source uvdoc."
            )
        pages = collect_page_paths(config)
        if not pages:
            raise MissingTrainingDataError(
                f"{config.pages_root!r} exists but holds no image file with a "
                f"suffix in {list(IMAGE_SUFFIXES)} outside "
                f"{sorted(EXCLUDED_DIR_NAMES)}. The synthetic generator warps "
                "REAL page rasters; it has no page content of its own."
            )
        if not collect_background_paths(config):
            logger.warning(
                "DocScanner: no background textures under %r -- falling back "
                "to a procedural texture. DTD is the paper's named background "
                "source and ships here as an unextracted tarball; extract it "
                "for a realistic background distribution. Geometry (f_gt, g, "
                "mask) is unaffected either way.",
                config.backgrounds_root,
            )
        logger.info(
            "DocScanner: %d page rasters under %s", len(pages), config.pages_root
        )
        return

    cache = uvdoc_cache_dir(config)
    if cache is not None:
        staged = staged_uvdoc_sample_ids(cache)
        if staged:
            logger.info(
                "DocScanner: %d staged UVDoc sidecars under %s (the 27.5 GB "
                "archive is not opened).", len(staged), cache,
            )
            return

    if not os.path.exists(config.uvdoc_root):
        raise MissingTrainingDataError(
            f"no UVDoc corpus at {config.uvdoc_root!r}. It is a 27.5 GB "
            "download from https://igl.ethz.ch/projects/uvdoc/; stage it with "
            "`python -m train.doc_scanner.prepare_doc_scanner_data`, or train "
            "on --data-source synthetic, which needs no download."
        )
    ids = collect_uvdoc_sample_ids(config)
    if not ids:
        raise MissingTrainingDataError(
            f"the UVDoc corpus at {config.uvdoc_root!r} opened but lists no "
            "render. A partially-staged archive is the usual cause; re-run the "
            "staging script, which decodes every member before marking the "
            "corpus complete."
        )
    logger.info("DocScanner: %d UVDoc renders at %s", len(ids), config.uvdoc_root)


# ---------------------------------------------------------------------------
# Sample producers (numpy; no TF, no Keras, no global RNG)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=PAGE_CACHE_SIZE)
def _cached_rgb(path: str) -> np.ndarray:
    """Decoded, downscaled RGB for one image path.

    Cached because the synthetic pipeline re-draws the same few hundred pages
    thousands of times per epoch and a PNG decode is the only I/O in it.

    :param path: Image path.
    :type path: str
    :return: ``(H, W, 3)`` uint8.
    :rtype: np.ndarray
    """
    return load_rgb(path, max_side=PAGE_MAX_SIDE)


@functools.lru_cache(maxsize=GEOMETRY_CACHE_SIZE)
def _cached_geometry(
        root: str,
        geometry_name: str,
        height: int,
        width: int,
        seed: int,
) -> UVDocGeometry:
    """Densified UVDoc geometry, cached by geometry name.

    Interface contract -- one caller, :func:`cached_uvdoc_geometry`, which
    exists to give the cache a documented public name and a documented reset.

    ``seed`` participates in the key only so a caller can force a re-densify;
    it drives the held-out cross-validation inside ``load_geometry`` and does
    not affect the emitted maps.

    :param root: UVDoc archive or directory.
    :param geometry_name: Geometry id (NOT a render id).
    :param height: Target height.
    :param width: Target width.
    :param seed: Seed for ``load_geometry``'s quality cross-validation.
    :return: The densified geometry.
    :rtype: UVDocGeometry
    :raises UVDocError: Propagated from ``load_geometry``.
    """
    with UVDocSource(root) as source:
        return load_geometry(
            source,
            geometry_name,
            np.random.default_rng(seed),
            size=(height, width),
            strict=False,
        )


def cached_uvdoc_geometry(
        config: DocScannerTrainingConfig,
        geometry_name: str,
) -> UVDocGeometry:
    """Densify one UVDoc geometry, memoized across renders.

    UVDoc fans 20,000 renders out of 4,032 distinct geometries (each render's
    ``metadata_sample/<id>.json`` names its ``geom_name``) and densifying one
    costs ~1.06 s of scattered-data interpolation. Keyed by geometry, an epoch
    over the whole corpus pays 4,032 inversions instead of 20,000 -- about
    1.2 h instead of about 6 h.

    :param config: The run config; reads ``uvdoc_root``, ``image_size`` and
        ``seed``.
    :type config: DocScannerTrainingConfig
    :param geometry_name: Geometry id.
    :type geometry_name: str
    :return: The densified geometry.
    :rtype: UVDocGeometry
    """
    return _cached_geometry(
        config.uvdoc_root,
        geometry_name,
        config.image_size,
        config.image_size,
        config.seed,
    )


def _procedural_background(
        rng: np.random.Generator,
) -> np.ndarray:
    """A smooth random texture, for when no background corpus is staged.

    :param rng: Explicit generator.
    :type rng: np.random.Generator
    :return: ``(N, N, 3)`` float32 in ``[0, 1]``.
    :rtype: np.ndarray
    """
    size = PROCEDURAL_BACKGROUND_SIZE
    tint = rng.uniform(0.15, 0.85, size=(1, 1, RGB_CHANNELS))
    noise = rng.uniform(-0.15, 0.15, size=(size, size, RGB_CHANNELS))
    return np.clip(tint + noise, 0.0, 1.0).astype(np.float32)


def segmenter_target(mask: np.ndarray) -> Tuple[np.ndarray, ...]:
    """The page mask, repeated once per deep-supervision head.

    ``DocScannerSegmenter`` returns ``[d0, ..., d6]`` and Keras' ``CompileLoss``
    applies the compiled loss to each output against the matching element of
    the target structure. All seven are supervised against the SAME mask, which
    is U2-Net's own deep-supervision objective, and the reported ``loss`` is
    their sum.

    :param mask: ``(H, W, 1)`` float32 in ``{0, 1}``.
    :type mask: np.ndarray
    :return: A tuple of :data:`SEGMENTER_HEAD_COUNT` references to it.
    :rtype: Tuple[np.ndarray, ...]
    """
    return tuple(mask for _ in range(SEGMENTER_HEAD_COUNT))


def rectifier_target(f_gt: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The ``[f_gt(2), g(2)]`` stack ``DocScannerFlowSequenceLoss`` reads.

    The channel ORDER is the loss's contract, not a convention chosen here:
    ``f_gt`` first, then ``g``, both in absolute pixels with channel 0 = x. A
    swap keeps the shape, the dtype and the range, and trains ``L_f`` against
    the forward map. ``tests/test_train/test_doc_scanner/test_pipeline.py``
    asserts the halves against the shared ``backward_map_convention``
    instrument rather than restating the convention here.

    :param f_gt: ``(H, W, 2)`` backward map.
    :type f_gt: np.ndarray
    :param g: ``(H, W, 2)`` forward map.
    :type g: np.ndarray
    :return: ``(H, W, 4)`` float32.
    :rtype: np.ndarray
    """
    return np.concatenate(
        [np.asarray(f_gt, dtype=np.float32), np.asarray(g, dtype=np.float32)],
        axis=-1,
    )


def synthetic_sample(
        config: DocScannerTrainingConfig,
        pages: Sequence[str],
        backgrounds: Sequence[str],
        index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render one synthetic ``(image, target)`` pair for a sample INDEX.

    Interface contract -- two callers, :func:`create_dataset`'s map function
    and the pipeline tests:

    * Deterministic in ``(config.seed, index)`` and in nothing else. The
      generator is an explicit :class:`numpy.random.Generator`; the legacy
      global numpy RNG is neither read nor written, so this is safe to call
      from a ``tf.data`` worker thread and safe to call from a test that has
      seeded something else.
    * Returns ``(image, target)``, both float32: ``image`` is
      ``(S, S, 3)`` in ``[0, 1]``; ``target`` is ``(S, S, 1)`` for the
      segmenter and ``(S, S, 4)`` for the rectifier.
    * Failure mode: re-draws up to :data:`MAX_SAMPLE_ATTEMPTS` times on a
      ``DegenerateWarpError`` and re-raises if every attempt is rejected.

    :param config: The run config.
    :type config: DocScannerTrainingConfig
    :param pages: Page raster paths; must be non-empty.
    :type pages: Sequence[str]
    :param backgrounds: Texture paths; may be empty.
    :type backgrounds: Sequence[str]
    :param index: The sample index, which IS the seed.
    :type index: int
    :return: ``(image, target)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: If ``pages`` is empty.
    :raises DegenerateWarpError: If every attempt drew a degenerate warp.
    """
    if not pages:
        raise ValueError(
            "synthetic_sample needs at least one page raster; the gate in "
            "require_training_data exists to make this unreachable from a CLI"
        )
    size = (config.image_size, config.image_size)

    last_error: Optional[DegenerateWarpError] = None
    for attempt in range(MAX_SAMPLE_ATTEMPTS):
        # A SPAWNED key, not `seed + index`: consecutive indices must not give
        # correlated streams, and the attempt number must give a genuinely
        # different draw rather than the same rejected warp again.
        rng = np.random.default_rng([config.seed, int(index), attempt])
        page = _cached_rgb(str(pages[rng.integers(len(pages))]))
        if backgrounds:
            background = _cached_rgb(
                str(backgrounds[rng.integers(len(backgrounds))])
            )
        else:
            background = _procedural_background(rng)
        try:
            warp = sample_warp(rng)
        except DegenerateWarpError as error:  # pragma: no cover - rare
            last_error = error
            continue
        image, f_gt, mask = render_sample(page, background, warp, size, rng)
        if config.stage == STAGE_SEGMENTER:
            return image, mask
        return image, rectifier_target(f_gt, warp.forward_map(*size))

    raise last_error  # pragma: no cover - MAX_SAMPLE_ATTEMPTS rejections


def uvdoc_sample(
        config: DocScannerTrainingConfig,
        sample_ids: Sequence[str],
        index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read one staged UVDoc ``(image, target)`` pair by worklist position.

    Interface contract -- same two callers and the same return contract as
    :func:`synthetic_sample`, deliberately, so ``create_dataset`` has one
    downstream shape for both corpora.

    Failure mode: an unreadable render or an unusable densification advances to
    the NEXT worklist position (up to :data:`MAX_SAMPLE_ATTEMPTS`) rather than
    taking the epoch down. UVDoc has a real reject path -- D-042 recorded a
    geometry whose held-out residual is 2.48 px -- and one bad render out of
    20,000 must cost one sample, not a run.

    :param config: The run config.
    :type config: DocScannerTrainingConfig
    :param sample_ids: The render worklist; must be non-empty.
    :type sample_ids: Sequence[str]
    :param index: Position in ``sample_ids``.
    :type index: int
    :return: ``(image, target)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: If ``sample_ids`` is empty.
    :raises UVDocError: If every attempted render failed to read.
    """
    if not sample_ids:
        raise ValueError("uvdoc_sample needs a non-empty worklist")
    size = (config.image_size, config.image_size)
    cache = uvdoc_cache_dir(config)

    last_error: Optional[UVDocError] = None
    for attempt in range(MAX_SAMPLE_ATTEMPTS):
        sample_id = str(sample_ids[(int(index) + attempt) % len(sample_ids)])
        try:
            if cache is not None:
                image, f_gt, g, mask = read_uvdoc_sidecar(cache, sample_id)
            else:
                with UVDocSource(config.uvdoc_root) as source:
                    geometry_name = source.geometry_for_sample(sample_id)
                    image = load_image(source, sample_id, size=size)
                geometry = cached_uvdoc_geometry(config, geometry_name)
                f_gt, g, mask = geometry.f_gt, geometry.g, geometry.mask
        except UVDocError as error:
            last_error = error
            continue
        if config.stage == STAGE_SEGMENTER:
            return image, mask
        return image, rectifier_target(f_gt, g)

    raise last_error  # pragma: no cover - MAX_SAMPLE_ATTEMPTS bad renders


# ---------------------------------------------------------------------------
# tf.data
# ---------------------------------------------------------------------------


def _target_channels(config: DocScannerTrainingConfig) -> int:
    """Channel count of the numpy target, before the segmenter's fan-out."""
    if config.stage == STAGE_SEGMENTER:
        return 1
    return RECTIFIER_TARGET_CHANNELS


def create_dataset(
        config: DocScannerTrainingConfig,
        worklist: Sequence[Any],
        is_training: bool,
) -> tf.data.Dataset:
    """Build the ``(image, target)`` pipeline for one split.

    Interface contract -- three callers (:func:`train` twice, and the pipeline
    tests):

    * ``worklist`` is a sequence of integer sample INDICES for the synthetic
      source and of UVDoc render ID STRINGS for the UVDoc source; it is what
      :func:`_split_worklist` produced, so train and validation draw from
      disjoint halves and a synthetic validation sample is a genuinely
      held-out seed rather than a re-draw.
    * The dataset repeats forever, so ``steps_per_epoch`` is what defines an
      epoch; every stage boundary carries a ``tf.ensure_shape``.
    * Failure mode: ``ValueError`` on an empty worklist.

    :param config: The run config.
    :type config: DocScannerTrainingConfig
    :param worklist: The split's sample keys.
    :type worklist: Sequence[Any]
    :param is_training: Whether to shuffle and drop a short final batch.
    :type is_training: bool
    :return: The batched, prefetched dataset.
    :rtype: tf.data.Dataset
    :raises ValueError: If ``worklist`` is empty.
    """
    if len(worklist) == 0:
        raise ValueError(
            f"no samples for the {'training' if is_training else 'validation'} "
            "split -- check --val-split and the size of the corpus"
        )

    size = config.image_size
    channels = _target_channels(config)

    if config.data_source == SOURCE_SYNTHETIC:
        pages = [str(path) for path in collect_page_paths(config)]
        backgrounds = [str(path) for path in collect_background_paths(config)]
        indices = np.asarray(worklist, dtype=np.int64)

        def produce(key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return synthetic_sample(config, pages, backgrounds, int(key))

        dataset = tf.data.Dataset.from_tensor_slices(indices)
    else:
        ids = [str(item) for item in worklist]

        def produce(key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return uvdoc_sample(config, ids, int(key))

        dataset = tf.data.Dataset.from_tensor_slices(
            np.arange(len(ids), dtype=np.int64)
        )

    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(config.shuffle_buffer, int(dataset.cardinality())),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.repeat()

    def wrapped(key: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image, target = tf.numpy_function(
            func=produce,
            inp=[key],
            Tout=[tf.float32, tf.float32],
            # The producers are pure numpy on an explicit Generator, so
            # parallel calls are safe. `stateful=False` is what lets tf.data
            # run more than one of them at a time.
            stateful=False,
        )
        return (
            tf.ensure_shape(image, [size, size, RGB_CHANNELS]),
            tf.ensure_shape(target, [size, size, channels]),
        )

    dataset = dataset.map(wrapped, num_parallel_calls=tf.data.AUTOTUNE)

    if config.stage == STAGE_SEGMENTER:
        # `segmenter_target` is the ONE producer of the deep-supervision fan-out;
        # writing the comprehension again here would be a second place the head
        # count is encoded.
        dataset = dataset.map(
            lambda image, mask: (image, segmenter_target(mask)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    return dataset.prefetch(tf.data.AUTOTUNE)


def _split_worklist(
        config: DocScannerTrainingConfig,
        worklist: Sequence[Any],
) -> Tuple[List[Any], List[Any]]:
    """Split a worklist into ``(train, validation)`` on a seeded shuffle.

    :param config: The run config; reads ``val_split`` and ``seed``.
    :type config: DocScannerTrainingConfig
    :param worklist: The full worklist.
    :type worklist: Sequence[Any]
    :return: ``(train, validation)``, disjoint.
    :rtype: Tuple[List[Any], List[Any]]
    :raises ValueError: If the split leaves either side empty.
    """
    items = list(worklist)
    order = np.random.default_rng(config.seed).permutation(len(items))
    shuffled = [items[position] for position in order]
    n_validation = int(round(len(items) * config.val_split))
    if n_validation < 1 or n_validation >= len(items):
        raise ValueError(
            f"a val_split of {config.val_split} over {len(items)} samples "
            f"leaves {n_validation} for validation and "
            f"{len(items) - n_validation} for training; both sides must be "
            "non-empty"
        )
    return shuffled[n_validation:], shuffled[:n_validation]


# ---------------------------------------------------------------------------
# Loss, optimizer, model
# ---------------------------------------------------------------------------


# DECISION plan-2026-09-10T065432-05fcb6dd/D-044
# This class exists because `training` selects the rectifier's output RANK.
# WHAT NOT TO DO: do NOT delete it and "fix" the problem in either of the two
# places it looks fixable. (1) Relaxing `DocScannerFlowSequenceLoss` to accept
# rank 4 removes the ONE guard that catches a whole run performed at
# `training=False`, i.e. one refinement iteration instead of twelve, which is
# finite, decreasing, correctly shaped and completely wrong. (2) Dropping
# `validation_data` and monitoring `loss` moves early stopping and checkpoint
# selection onto the training loss, the one signal that cannot see overfitting.
# The rank-4 arm is NOT an approximation of the objective: `iters=1` weights
# its single element `gamma ** 0 == 1.0`, so it IS Eq. 9's k = K term.
# See decisions.md D-044.
@keras.saving.register_keras_serializable(package="train.doc_scanner")
class DocScannerRectifierObjective(keras.losses.Loss):
    """Eq. 9-14 during ``fit``, its final term during ``evaluate``.

    WHY THIS EXISTS (it is not a convenience). ``DocScannerRectifier`` uses
    Keras' own ``training`` flag to select its OUTPUT RANK: ``training=True``
    returns the whole ``(B, K, H, W, 2)`` refinement sequence, anything else
    returns the last iteration alone, ``(B, H, W, 2)``. Keras' ``test_step``
    calls ``self(x, training=False)``, so the VALIDATION pass of a plain
    ``fit(validation_data=...)`` hands the compiled loss a rank-4 tensor --
    and :class:`DocScannerFlowSequenceLoss` rightly raises on it, naming that
    exact situation. MEASURED: a rectifier run trains for a full epoch and then
    dies at the first validation batch with
    ``ValueError: y_pred must be the whole refinement sequence``.

    So there are three options and this is the third:

    1. Drop ``validation_data``, monitor ``loss``. Early stopping and
       checkpoint selection would then run off the TRAINING loss, which is the
       one signal that cannot detect overfitting.
    2. Relax the sequence loss to accept rank 4. That deletes a guard whose
       whole job is to catch a ``training=False`` forward pass -- the defect
       this class exists BECAUSE of.
    3. Score the validation pass on what the model will actually emit at
       inference: the FINAL iteration. Its weight in Eq. 9 is exactly
       ``gamma^0 == 1.0``, so ``DocScannerFlowSequenceLoss(iters=1)`` applied
       to that one map IS the ``k = K`` term of the training objective,
       unchanged and unscaled.

    Consequence, stated so nobody reads the two numbers as comparable:
    ``val_loss`` is the LAST-ITERATION term while ``loss`` is the weighted sum
    of all ``K``, so ``loss`` is roughly ``sum(gamma^i) ~ 6.9x`` the size at
    equal quality. Both are minimized, both are monotone in the same thing, and
    checkpoint selection compares ``val_loss`` only against itself.

    Interface contract -- one caller, :func:`build_loss`:

    * ``y_true``: ``(B, H, W, 4)``, the ``[f_gt(2), g(2)]`` stack.
    * ``y_pred``: rank 5 (the sequence) or rank 4 (the final map).
    * Failure mode: whatever the delegate raises. A rank-5 tensor whose
      iteration axis is not ``iters`` still fails, from the delegate.

    :param iters: The refinement count the sequence arm expects.
    :type iters: int
    :param kwargs: Forwarded to ``keras.losses.Loss``.
    :type kwargs: Any
    """

    def __init__(self, iters: int = REFINE_ITERATIONS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.iters = int(iters)
        self.sequence_loss = DocScannerFlowSequenceLoss(iters=self.iters)
        # `iters=1` puts weight `gamma ** 0 == 1.0` on its single element, for
        # every gamma. This is the k = K term, not a rescaling of it.
        self.final_loss = DocScannerFlowSequenceLoss(iters=1)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Dispatch on the STATIC rank of ``y_pred``.

        The rank is known at trace time -- 5 inside ``train_step``, 4 inside
        ``test_step`` -- so this is a Python branch over two graphs, not a
        ``tf.cond``.
        """
        if len(y_pred.shape) == 4:
            return self.final_loss(y_true, keras.ops.expand_dims(y_pred, axis=1))
        return self.sequence_loss(y_true, y_pred)

    def get_config(self) -> Any:
        config = super().get_config()
        config.update({"iters": self.iters})
        return config


def build_loss(config: DocScannerTrainingConfig) -> keras.losses.Loss:
    """The stage's objective.

    :param config: The run config; reads ``stage``.
    :type config: DocScannerTrainingConfig
    :return: A stock ``BinaryCrossentropy`` for the segmenter (applied to each
        of the seven deep-supervision heads), and
        :class:`DocScannerRectifierObjective` -- the paper's Eq. 9-14 sequence
        loss, plus the rank dispatch a validation pass forces -- for the
        rectifier.
    :rtype: keras.losses.Loss
    """
    if config.stage == STAGE_SEGMENTER:
        # The segmenter's heads are already sigmoid-activated (`seg.py` applies
        # the sigmoid inside the model), so this reads PROBABILITIES.
        return keras.losses.BinaryCrossentropy(from_logits=False)
    return DocScannerRectifierObjective(iters=REFINE_ITERATIONS)


def build_optimizer(
        config: DocScannerTrainingConfig,
) -> keras.optimizers.Optimizer:
    """The stage's optimizer, built through ``optimizer_builder``.

    Interface contract -- two callers, :func:`build_model` and the pipeline
    tests, which assert the two recipes rather than reading this source.

    Both arms go through :func:`dl_techniques.optimization.optimizer_builder`,
    including the clipping, whose keys that builder RENAMES
    (``gradient_clipping_by_norm`` becomes ``global_clipnorm``). A literal
    ``"clipnorm"`` key here would be silently ignored and the run would train
    unclipped with no error (``src/train/CLAUDE.md``).

    Weight decay is applied by the optimizer and by NOTHING else: no layer in
    the ``doc_scanner`` package carries a ``kernel_regularizer``, because
    AdamW's decoupled decay plus an L2 penalty decays the same parameter twice.

    :param config: The run config.
    :type config: DocScannerTrainingConfig
    :return: The optimizer.
    :rtype: keras.optimizers.Optimizer
    """
    total_steps = max(1, config.epochs * config.steps_per_epoch)

    if config.stage == STAGE_SEGMENTER:
        # DECISION plan-2026-09-10T065432-05fcb6dd/D-043
        # The segmenter's schedule is a PiecewiseConstantDecay built here, NOT
        # `learning_rate_schedule_builder`. Do NOT "route it through the
        # builder for consistency" with an `exponential_decay` of
        # `decay_rate=lr_drop_factor`: the builder does not forward
        # `staircase`, so Keras' default `staircase=False` makes
        # ExponentialDecay decay CONTINUOUSLY. The paper's recipe holds 1e-4
        # flat and then drops (`"reduced by a factor of 0.1 after 30 epochs"`);
        # the continuous form would already be at 3.2e-5 half way to the drop
        # -- a different schedule with the same two numbers in its config, no
        # error and no shape symptom. See decisions.md D-043.
        drop_step = max(1, config.lr_drop_epoch * config.steps_per_epoch)
        schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[drop_step],
            values=[
                config.learning_rate,
                config.learning_rate * config.lr_drop_factor,
            ],
        )
        return optimizer_builder(
            {
                "type": "adam",
                "gradient_clipping_by_norm": config.gradient_clipping,
            },
            schedule,
        )

    schedule = learning_rate_schedule_builder(
        {
            "type": "cosine_decay",
            "learning_rate": config.learning_rate,
            "decay_steps": total_steps,
            "alpha": config.final_lr_fraction,
            # LINEAR ramp -- `WarmupSchedule` is this repo's one warmup and its
            # ramp is linear. THE CURVE IS THIS REPO'S CHOICE; the paper gives
            # only the length (27k of 560k iterations).
            "warmup_steps": config.warmup_epochs * config.steps_per_epoch,
        }
    )
    return optimizer_builder(
        {
            "type": "adamw",
            "weight_decay": config.weight_decay,
            "gradient_clipping_by_norm": config.gradient_clipping,
        },
        schedule,
    )


def build_model(config: DocScannerTrainingConfig) -> keras.Model:
    """Create and compile the stage's model.

    The two stages train INDEPENDENTLY (§4.3), so this builds ONE of them --
    never the composite :class:`DocScanner`, which is an inference assembly
    with no objective of its own.

    :param config: The run config.
    :type config: DocScannerTrainingConfig
    :return: The compiled model.
    :rtype: keras.Model
    """
    if config.stage == STAGE_SEGMENTER:
        model = create_doc_scanner_segmenter(variant=config.model_variant)
    else:
        model = create_doc_scanner_rectifier(variant=config.model_variant)
    model.compile(
        optimizer=build_optimizer(config),
        loss=build_loss(config),
    )
    return model


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(config: DocScannerTrainingConfig) -> Tuple[keras.Model, Any, str]:
    """Train one DocScanner stage with stock ``fit()``.

    No custom ``train_step`` (H-5): the rectifier's 12-iteration sequence is a
    model OUTPUT under ``training=True``, and the forward map ``g`` that
    ``L_line`` needs rides in the target tensor, so the whole objective is
    reachable through ``compile(loss=...)``.

    :param config: The run config.
    :type config: DocScannerTrainingConfig
    :return: ``(model, history, results_dir)``.
    :rtype: Tuple[keras.Model, Any, str]
    """
    set_seeds(config.seed)

    if config.data_source == SOURCE_SYNTHETIC:
        worklist: List[Any] = list(range(config.synthetic_samples))
    else:
        worklist = list(collect_uvdoc_sample_ids(config))
    train_items, validation_items = _split_worklist(config, worklist)
    logger.info(
        "DocScanner %s: %d %s samples (%d train / %d val)",
        config.stage, len(worklist), config.data_source,
        len(train_items), len(validation_items),
    )

    train_dataset = create_dataset(config, train_items, is_training=True)
    validation_dataset = create_dataset(
        config, validation_items, is_training=False
    )

    model = build_model(config)

    callbacks, results_dir = create_callbacks(
        model_name=config.model_variant,
        results_dir_prefix=f"doc_scanner_{config.stage}",
        output_root=config.output_dir,
        monitor="val_loss",
        # `resolve_monitor_mode` maps the `loss` token to 'min'; stated here so
        # the direction is visible at the call site rather than inferred.
        monitor_mode="min",
        patience=config.patience,
        # Both stages carry an EXTERNAL schedule (piecewise / cosine+warmup),
        # so `use_lr_schedule=True` keeps ReduceLROnPlateau out of the way
        # rather than letting two things drive one learning rate.
        use_lr_schedule=True,
        # The loss is an L1 in ABSOLUTE PIXELS over a 12-step recurrent unroll;
        # a diverged step shows up as NaN and there is nothing to be learned
        # from the epochs after it.
        include_terminate_on_nan=True,
        # The epoch analyzer inspects a single-tensor output. Neither stage has
        # one: the segmenter emits SEVEN maps and the rectifier emits a rank-5
        # iteration sequence under `training=True`. This is the documented
        # reason `src/train/CLAUDE.md` asks for at a callbacks section that
        # departs from the default.
        include_analyzer=False,
    )
    save_config_json(config, results_dir, "config.json")

    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=config.validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    save_training_history_json(history, results_dir)
    return model, history, results_dir
