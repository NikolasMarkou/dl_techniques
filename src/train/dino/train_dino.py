"""DINO self-supervised pretraining on raw images, under STOCK ``model.fit()``.

Trains a `DINOTrainingModel` (student + frozen EMA teacher, `dl_techniques.models.dino`)
against `DINOLoss` over the multi-crop element produced by
`dl_techniques.datasets.vision.multi_crop.make_multi_crop_map_fn`. Every DINO mechanism
routes through machinery Keras already ships:

===========================  ==================================================
mechanism                    how it runs
===========================  ==================================================
multi-crop views             a per-sample ``tf.data`` ``element_map_fn``
teacher EMA                  ``TeacherEMACallback`` -> ``update_teacher_ema``
teacher centering            a ``keras.Variable`` assigned inside ``DINOLoss.call``
teacher-temperature warmup   a stock ``LambdaCallback`` -> ``set_teacher_temp``
===========================  ==================================================

**No custom ``train_step``. No bespoke training loop.** (`src/train/clip/train_clip.py`'s
hand-rolled ``fit()`` is undocumented drift, not a template.)

This docstring is the OPERATING MANUAL: what to run, what never to pass, which shipped
defaults silently change what you measure. It does NOT carry the evidence -- the per-flag
reference table and the headline result are in ``src/dl_techniques/models/dino/README.md``
§ 6 (cited below as "README"), and the measurement record behind every number either file
quotes is ``research/2026_dino_ssl_measurements.md``.

-------------------------------------------------------------------------------
THE ONE RULE THAT IS NOT OPTIONAL: this trainer NEVER passes ``validation_data``
-------------------------------------------------------------------------------
`DINOLoss` maintains its centering statistic by ``.assign()``-ing a ``keras.Variable``
inside ``call()``. That is correct under stock ``fit()`` -- but ``call()`` runs on EVERY
batch, and Keras runs the loss on validation batches too, so each validation batch
performs a full, unwanted centering EMA update, silently, with a finite loss and a clean
exit. ``validation_batch_size`` defaults to ``batch_size``, so the corruption scales with
the validation batch COUNT. (The measured instance -- a 4-sample validation set pushing
the center 81% past its correct value -- is in README § 5 Rule 1.)

Consequences, both deliberate:

* ``model.fit(...)`` below takes no ``validation_data`` and no ``validation_steps``.
* There is therefore no ``val_loss``, so the callbacks monitor the TRAINING ``loss``.
  Validation is a k-NN probe that never invokes the loss -- see "Validation" below.

-------------------------------------------------------------------------------
Scale: what ``--smoke`` pins, and the two traps that ride along with it
-------------------------------------------------------------------------------
``--smoke`` pins the shape-validation scale -- ``variant=tiny``,
``global_crop_size=96``, ``n_local_crops=4``, ``batch_size=32``,
``dino_out_dim=4096``, ``epochs=2``, ``max_steps=5``, ``ema_warmup_epochs=0.0`` and
three more warmup pins; :data:`SMOKE_OVERRIDES` below is the authoritative list.
Explicit flags still win over the preset. This validates SHAPES and wiring; it is
**NOT a paper reproduction** (the paper uses ``dino_out_dim=65536``, 224px globals and
hundreds of epochs). The defaults are the paper-shaped ones and cost accordingly.

Two traps ride along with the preset (README § 6.3 has them at length). Neither is a bug;
each silently hands you a number that is not the one you think you are reading.

**TRAP 1: ``--smoke`` sets ``max_steps=5``, so a ``--smoke`` "epoch" is 5 steps of the
~295 an imagenette epoch has at ``batch_size=32``** -- ``--smoke --epochs 40`` trains for
200 steps, not ~11 800, and that curve is not a training curve. Anything meant to TRAIN
rather than validate shapes must pass ``--max-steps 100000`` (or an explicit budget).

**TRAP 2: ``--smoke`` leaves ``knn_bank_batches=16`` / ``knn_query_batches=8``** (512
bank images), while the k-NN figures in ``research/2026_dino_ssl_measurements.md`` and
`train.dino.knn_eval`'s zero-step control band were measured at **64 / 32** (2048
images). A top-1 over a 4x smaller bank is a DIFFERENT ESTIMATOR, not a noisier reading
of the same one. Pass ``--knn-bank-batches 64 --knn-query-batches 32`` to compare against
one of those figures.

**Memory, and why ``--gpu`` will not help you.** One full train step at the ``--smoke``
scale peaks at **1518.6 MiB of the 10001 MiB free on GPU 1 (RTX 4070)** (probe method:
the comment above :data:`SMOKE_OVERRIDES`; corroborated to within 1% in
``research/2026_dino_ssl_measurements.md`` § 7). Do NOT read that off ``nvidia-smi``
while this script runs -- in this trainer's import order TF is initialized before
``train.common.setup_gpu``, which then logs ``GPU setup error: Physical devices cannot be
modified after being initialized`` and leaves TF pre-allocating ~85% of the device
(~10 400 MiB polled: TF's ARENA, not the working set). The same import order makes
``--gpu N`` INERT in BOTH halves -- its ``CUDA_VISIBLE_DEVICES`` assignment lands after
TF picked a device, and its other half IS that failed ``set_memory_growth``. **Select the
device by prefixing ``CUDA_VISIBLE_DEVICES=N``**, as every ``Usage`` line below does.

-------------------------------------------------------------------------------
Validation, and the reason a decreasing loss is not enough
-------------------------------------------------------------------------------
`train.dino.knn_eval.KNNEvalCallback` is this run's ONLY validation signal (frozen
student-backbone features, a weighted k-NN top-1, and two collapse numbers; README § 6
describes it). It is INSERTED BEFORE ``CSVLogger`` -- appending it after would silently
drop every column (MEASURED; see the D-029 anchor in :func:`create_callbacks`).

**A decreasing loss does NOT rule out collapse.** Read
``results/<run>/training_log.csv``'s ``dino_collapse_flag`` / ``dino_feat_mean_cos`` /
``dino_teacher_entropy_norm`` / ``dino_knn_top1_k20`` columns before calling a run good;
`knn_eval`'s module docstring carries the STOP thresholds (chance is 0.10 on
imagenette's 10 classes).

-------------------------------------------------------------------------------
Making a run COMPARABLE: the default is reproducible, and ``--no-`` gives that up
-------------------------------------------------------------------------------
Three flags decide it (full reference with caveats: README § 6.2):

============================  =================================================
flag                          what it buys
============================  =================================================
``--random-init-repeats N``   Default 2. Runs the k-NN probe ``N`` times in
                              ``on_train_begin``, BEFORE a single optimizer step,
                              into ``<run_dir>/random_init_control.json``. Quote
                              every k-NN delta against THAT, never against the
                              0.10 chance line. ``0`` disables it.
``--seed-training-stream``    Default ON. Seeds the TRAINING stream's TFDS **file
                              interleave** with ``--seed``.
``--stateless-augmentation``  Default ON. Keys the multi-crop augmentation on a
                              per-element counter (``stateless_uniform``) instead
                              of one shared ``tf.random.Generator`` stream -- the
                              only thing that makes the **augmentation**
                              reproducible under the shipped
                              ``num_parallel_calls=AUTOTUNE``.
============================  =================================================

**THE RULE: ``--seed`` alone does NOT make two runs the same experiment. BOTH
stream flags are required TOGETHER, and neither is redundant.** MEASURED at the
data pipeline (2 processes, the real :func:`build_dataset`, sha1 of the first 3
batches): either flag ON ALONE still gave differing batches; both ON gave
bit-identical ones. Table in ``research/2026_dino_ssl_measurements.md`` § 4.

The two flags reach two different unseeded sources -- the file interleave and the
augmentation RNG -- so an A/B run without both is comparing models trained on
different data. For the pipeline that probe measured, the bit-identity of the
both-flags cell also rules out ``tf.data`` ``options.deterministic`` and the
element ``.shuffle()`` as residual contributors. It says NOTHING about cuDNN
kernel nondeterminism: the probe was CPU-only, built no model and ran no GPU
kernel.

**Both stream flags ship ON**, so a no-flag run is already reproducible against
another no-flag run at the same ``--seed``; nothing needs to be remembered. The
non-obvious direction is the off-switch: because the two close DIFFERENT holes,
``--no-seed-training-stream`` OR ``--no-stateless-augmentation`` alone is enough
to lose reproducibility. Passing BOTH restores the older stream (unpinned file
order, shared ``Generator``) and with it comparability to runs in ``results/``
recorded while the flags shipped OFF -- each flag changes what every training
batch CONTAINS, so a default run today is not comparable to those.

Usage::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.dino.train_dino --smoke

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.dino.train_dino \\
        --variant small --global-crop-size 224 --dino-out-dim 65536 --epochs 100

    # a CONTROLLED arm: real step budget, comparable estimator, pinned seed.
    # Both stream flags are ON by default, so they are NOT passed here.
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.dino.train_dino \\
        --smoke --max-steps 100000 --epochs 60 --seed 42 \\
        --knn-bank-batches 64 --knn-query-batches 32 --random-init-repeats 2

    # the OLD stream, to compare against a run recorded before the flags flipped
    ... --no-seed-training-stream --no-stateless-augmentation
"""

import gc
import json
import math
import time
import keras
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    setup_gpu,
    set_seeds,
    save_config_json,
    create_callbacks as create_common_callbacks,
)
from train.energy_transformer.common import (
    SUPPORTED_DATASETS,
    build_optimizer,
    build_raw_image_dataset,
)
from dl_techniques.utils.logger import logger
from dl_techniques.losses.dino_loss import DINOLoss
from dl_techniques.datasets.vision.multi_crop import (
    make_multi_crop_map_fn,
    make_stateless_multi_crop_map_fn,
)
from dl_techniques.models.dino.dino_training import (
    N_GLOBAL_VIEWS,
    create_dino_training_model,
)
from dl_techniques.models.depth_anything.teacher_ema import (
    TeacherEMACallback,
    cosine_ema_schedule,
    linear_ema_schedule,
)
from train.dino.knn_eval import (
    DEFAULT_KNN_TEMPERATURE,
    KNNEvalCallback,
)

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

VARIANTS: Tuple[str, ...] = ("tiny", "small", "base", "large", "giant")

# The MEASURED shape-validation scale (D-026): one full train step peaks at 1518.6 MiB of
# the 10001 MiB free on GPU 1, i.e. ~15% -- measured by a dedicated one-config-per-process
# `reset_memory_stats` / `get_memory_info(...)['peak']` probe around a single
# `train_on_batch`, NOT by polling a run of this script (see the module docstring: polling
# reads TF's ~85% pre-allocated arena instead). The three pre-committed memory reductions
# (n_local_crops 4->2, crop 96->64, batch 32->16) were measured and are NOT needed --
# do not apply them "to be safe", they cost coverage for nothing.
SMOKE_OVERRIDES: Dict[str, Any] = {
    "variant": "tiny",
    "global_crop_size": 96,
    "n_local_crops": 4,
    "batch_size": 32,
    "dino_out_dim": 4096,
    "epochs": 2,
    "max_steps": 5,
    "warmup_epochs": 0,
    "teacher_temp_warmup_epochs": 1,
    "ema_warmup_steps": 0,
    # DECISION plan-2026-08-02T132301-93deeae2/D-001
    # This entry is LOAD-BEARING, not redundant with the `ema_warmup_steps: 0` above.
    # Under `resolve_ema_warmup_steps` a zero `ema_warmup_steps` means "defer to
    # `ema_warmup_epochs`", and the shipped `ema_warmup_epochs` default is 1.0. Without
    # this pin a `--smoke` run would SILENTLY GAIN a teacher freeze it has never had --
    # changing what every smoke measurement measures, with nothing failing. Do NOT
    # "simplify" it away as a duplicate of the line above; see decisions.md D-001.
    "ema_warmup_epochs": 0.0,
}


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for DINO self-supervised pretraining.

    Every field here is settable from exactly one CLI flag (see
    :func:`parse_arguments` / :func:`config_from_args`); that correspondence is
    enforced structurally by ``tests/test_train/test_dino/test_train_dino.py``.

    Defaults are paper-shaped, not smoke-shaped. ``--smoke`` overrides them with
    :data:`SMOKE_OVERRIDES`.
    """

    # Data
    dataset: str = "imagenette"  # imagenette | cifar10
    global_crop_size: int = 224
    # D-002: local views are rendered at the GLOBAL resolution (a smaller AREA is cropped
    # and resized up). Anything other than None / global_crop_size raises
    # NotImplementedError inside make_multi_crop_map_fn, naming positional-embedding
    # interpolation. That check is NOT duplicated here -- one definition, one message.
    local_crop_size: Optional[int] = None
    n_local_crops: int = 4
    # DECISION plan-2026-08-01T105809-dc0c402e/D-036
    # Resolution at which `build_raw_image_dataset` DECODES each record, i.e. the
    # resolution the multi-crop transform crops FROM. `None` means
    # `global_crop_size`, which is what the smoke run measured in step 12.
    #
    # Why this exists: `build_raw_image_dataset._decode` resizes every record
    # BEFORE `element_map_fn` runs, so with `None` a "local crop of the source
    # image" is a crop of a `global_crop_size` thumbnail. MEASURED at the smoke
    # scale (global_crop_size=96, local_scale=(0.05, 0.4), 2000 draws): local
    # crop sides are 19-69 px (mean 44) and get upsampled to 96 -- 2.33x mean,
    # 4.50x worst case -- and 8 of the 10 loss pairs carry a local student view.
    # Setting this to, say, 224 makes the local crops come out of a 224-square
    # source and be resized DOWN instead.
    #
    # Why the DEFAULT is still `None` (behaviour-preserving) rather than a
    # larger number: the only end-to-end evidence this trainer has was collected
    # at `None`, and the step-14 confirmation run exists to test ONE hypothesis
    # (the teacher/student initialization, D-034). Changing the data pipeline in
    # the same run would make the result unattributable. Moving this default is a
    # measurement, not a cleanup.
    #
    # UPDATE -- THE MEASUREMENT HAS NOW BEEN RUN, and the default still does not move.
    # `--source-image-size 224` vs `None`, 2 seeds, each arm read against its OWN
    # zero-step random-init control, on a fully controlled stream
    # (--seed-training-stream --stateless-augmentation): effect +0.0024 on the
    # pre-registered endpoint and +0.0028 on the mean of the last 3 evaluated epochs --
    # NO DIFFERENCE, an order of magnitude inside the +/-0.02 band.
    #
    # Read that null PRECISELY, because the geometry says why it is not the general
    # claim it looks like. Re-measured through the shipped `_random_resized_crop`,
    # N=2000 draws: at a 96 px source the mean local upsample is 2.35x, worst 4.50x,
    # and 100% of local views are upsamples (this REPRODUCES the 2.33x/4.50x above);
    # at 224 px the mean is 1.006 -- marginally ABOVE 1.0, NOT below, so the earlier
    # "falls below 1.0" prediction is FALSIFIED AS STATED -- worst 1.92x, and 39% of
    # local views are STILL upsamples. So 224 MITIGATES the thumbnail defect (2.3x
    # less mean interpolation) rather than eliminating it, and the null is evidence
    # that a 2.3x interpolation reduction buys no measurable k-NN delta AT THIS SCALE,
    # NOT evidence that local-crop resolution is irrelevant.
    source_image_size: Optional[int] = None

    # DECISION plan-2026-08-01T195746-12a1f2db/D-009
    # Route the multi-crop augmentation through `tf.random.stateless_uniform`,
    # keyed on a per-element counter from `.enumerate()` after `.repeat()`,
    # instead of through one shared `tf.random.Generator` stream. Only then does
    # `--seed` reproduce the AUGMENTATION stream under the shipped
    # `num_parallel_calls=AUTOTUNE` (MEASURED at HEAD: two same-seed parallel
    # maps differ, maxdiff 1.5312).
    #
    # DECISION plan-2026-08-02T132301-93deeae2/D-004
    # DEFAULT-ON as of this plan (it shipped OFF under D-009 above, whose
    # "default OFF for comparability" rationale is SUPERSEDED, not merely
    # relaxed). A run with no flags is now REPRODUCIBLE out of the box.
    # `--no-stateless-augmentation` restores the old behaviour, and with it
    # comparability to `results/dino_smoke_step12/` and to every arm measured
    # before this plan.
    #
    # This flag is NOT sufficient alone, and neither is `seed_training_stream`
    # below: BOTH are required TOGETHER for bit-identical batches across
    # processes. MEASURED (2-process CPU-only sha1 over the first 3 batches of
    # the real `build_dataset`): either flag alone still DIFFERS; both together
    # are bit-identical. Do not "simplify" by shipping one of them.
    #
    # The honest counter-argument, stated here because `plans/` is gitignored:
    # `--stateless-augmentation` has only ever been exercised end-to-end in ONE
    # plan's confirm/long matrices. The default path now depends on a code path
    # with that much and no more exposure.
    # Numbers, configs and all 9 confounds: research/2026_dino_ssl_measurements.md
    stateless_augmentation: bool = True

    # DECISION plan-2026-08-01T195746-12a1f2db/D-011
    # Seed the TRAINING stream's TFDS file interleave, the way `build_knn_datasets`
    # already seeds the k-NN memory bank's (D-040). Without it, `--seed` fixes the
    # measuring instrument but NOT the data the model is trained on: `seed=` reaches
    # the element `.shuffle()` and the augmentation, not the file order, so two
    # same-seed runs interleave the TFDS shards differently and see a different
    # example ORDER from step 0.
    #
    # DECISION plan-2026-08-02T132301-93deeae2/D-004
    # DEFAULT-ON as of this plan (it shipped OFF under D-011 above; that "default
    # OFF for comparability" rationale is SUPERSEDED). `--no-seed-training-stream`
    # restores the old unpinned file interleave and with it comparability to
    # `results/dino_smoke_step12/` and to every arm measured before this plan.
    #
    # Still NOT sufficient alone -- see `stateless_augmentation` above: BOTH flags
    # are required TOGETHER for bit-identical batches across processes (MEASURED,
    # 2-process CPU-only sha1 over the first 3 batches of the real `build_dataset`;
    # either alone DIFFERS). research/2026_dino_ssl_measurements.md has the table.
    #
    # This is still NOT the same decision as `build_knn_datasets`' always-on seed.
    # The bank is a frozen probe -- reseeding it changes only which images the score
    # is read against. The TRAINING stream pins what every batch contains from step 0,
    # which is why the off-switch exists and why the flip is recorded as a deliberate
    # break rather than a cleanup.
    #
    # Do NOT pass `shuffle_files_seed=None` on the `--no-` path. `build_dataset` omits
    # the kwarg ENTIRELY there (absence != None at the `ReadConfig` layer); the guard in
    # tests/test_train/test_dino/test_train_dino.py asserts ABSENCE, not `None`.
    seed_training_stream: bool = True

    # Model
    variant: str = "small"
    patch_size: Optional[int] = None  # None defers to the variant (D-017)
    dino_out_dim: int = 65536  # paper scale; the smoke scale uses 4096

    # Loss
    student_temp: float = 0.1
    teacher_temp: float = 0.04  # start of the warmup
    # DECISION plan-2026-08-02T132301-93deeae2/D-003
    # Ships at 0.04, i.e. EQUAL to `teacher_temp` above. Four facts, none optional:
    #
    # (a) Only the PAIR (`ema_warmup_epochs=1.0` -- the 295-step teacher freeze -- TOGETHER
    #     with `teacher_temp_final=0.04`) has a 60-epoch IMPROVED verdict: 2 seeds, smoke
    #     scale, controlled (bit-identical) stream, imagenette, GPU 1. There is NO
    #     temp-only arm at 60 epochs. Nothing measured says 0.04 beats 0.07 on its own.
    # (b) The teacher-temp arm ALONE was CLEARED at the pre-registered epoch-0 endpoint --
    #     it is structurally inert there, because `create_teacher_temp_callback` assigns
    #     only via `on_epoch_begin`, so it returned an EXACTLY-ZERO null at both seeds --
    #     and was AMBIGUOUS on the exploratory last-3-epochs endpoint, where its effect
    #     SHRANK from +0.0518 to +0.0387 once the training stream was controlled. It was
    #     shipped into the improved arm against that matrix's own verdict label,
    #     "must be confirmed, not shipped".
    # (c) `teacher_temp_final == teacher_temp` makes the linear temp schedule CONSTANT, so
    #     `teacher_temp_warmup_epochs` below is an INERT knob at the shipped defaults.
    #     Mechanism, not belief: `create_teacher_temp_callback` builds
    #     `linear_ema_schedule(decay_start=teacher_temp, decay_end=teacher_temp_final,
    #     total_steps=teacher_temp_warmup_epochs)`, which returns
    #     `decay_start + (decay_end - decay_start) * progress`. With the two endpoints
    #     equal that delta is exactly 0.0, so `total_steps` only scales a term multiplied
    #     by zero. Pinned by `test_teacher_temp_schedule_is_constant_at_shipped_defaults`.
    # (d) This DEPARTS from the DINO paper's `0.04 -> 0.07` teacher-temperature warmup.
    #     Anyone reproducing the paper recipe must pass `--teacher-temp-final 0.07`, which
    #     also makes `teacher_temp_warmup_epochs` live again.
    #
    # Full evidence, including the confounds and the open EMA-vs-temp attribution
    # question: `research/2026_dino_ssl_measurements.md` (section 8.4). See decisions.md
    # D-003 and D-007 (the user ruling that shipped it).
    teacher_temp_final: float = 0.04  # end of the warmup (paper: 0.04 -> 0.07)
    teacher_temp_warmup_epochs: int = 30  # INERT at the shipped defaults -- see (c) above
    center_momentum: float = 0.9

    # Teacher EMA
    ema_decay_start: float = 0.996
    ema_decay_end: float = 0.9999
    # DECISION plan-2026-08-02T132301-93deeae2/D-001
    # The teacher-EMA warmup has TWO homes and ONE resolution rule
    # (:func:`resolve_ema_warmup_steps`): `ema_warmup_steps` is an ABSOLUTE-step
    # override that WINS whenever it is > 0; `ema_warmup_epochs` is the
    # DEFAULT-BEARING knob and is used whenever `ema_warmup_steps == 0`.
    #
    # Why the default is denominated in EPOCHS. The configuration this repo actually
    # measured used `--ema-warmup-steps 295`, and 295 is not a recipe constant: it is
    # `num_train // batch_size` for the imagenette train split at `batch_size=32`, i.e.
    # exactly ONE EPOCH AT THAT SCALE ONLY. At `batch_size=64` the same literal is ~2
    # epochs, and on another dataset it is arbitrary. Shipping `ema_warmup_epochs=1.0`
    # reproduces the measured behaviour at the measured scale AND transfers.
    # Evidence, caveats and the un-separated SUPERSET question:
    # `research/2026_dino_ssl_measurements.md`.
    ema_warmup_steps: int = 0
    ema_warmup_epochs: float = 1.0

    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 5e-4
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 10
    weight_decay: float = 0.04
    gradient_clipping: float = 3.0

    # Monitoring. NOTE: there is no `val_loss` -- see the module docstring's Rule.
    early_stopping_patience: int = 30

    # k-NN probe on frozen features -- the ONLY validation signal this run has, plus
    # the collapse diagnostic (`train.dino.knn_eval`). `knn_eval_every=0` disables it,
    # which also disables Pre-Mortem 3's detection: a decreasing loss then proves
    # nothing about the representation.
    knn_eval_every: int = 1  # epochs between evaluations; 0 = off
    knn_bank_batches: int = 16  # memory-bank batches drawn from the TRAIN split
    knn_query_batches: int = 8  # query batches drawn from the VALIDATION split
    knn_temperature: float = DEFAULT_KNN_TEMPERATURE
    # DECISION plan-2026-08-01T195746-12a1f2db/D-004
    # Repeats of the ZERO-OPTIMIZER-STEP k-NN control that `KNNEvalCallback
    # .on_train_begin` writes to `<run_dir>/random_init_control.json` before
    # `fit()` performs a single update. DEFAULT-ON: the measured failure mode of
    # this codebase is quoting a `dino_knn_top1_*` delta with no baseline at all
    # (D-032/D-037), and every control ever quoted came from an uncommitted
    # scratch script. `0` is the escape hatch for a pure-throughput run; the cost
    # otherwise is two feature extractions (seconds) per run.
    random_init_repeats: int = 2

    # Debug
    max_steps: Optional[int] = None  # cap steps_per_epoch (smoke runs); None = full epoch

    # Output -- I-5: repo-root `results/`, NEVER `src/results/`.
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Runtime
    seed: int = 42
    gpu: Optional[int] = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = (
                f"dino_{self.dataset}_{self.variant}_{timestamp}")

        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset {self.dataset!r}; supported: "
                f"{sorted(SUPPORTED_DATASETS)}"
            )
        if self.variant not in VARIANTS:
            raise ValueError(
                f"Unsupported variant {self.variant!r}; supported: "
                f"{sorted(VARIANTS)}"
            )
        if self.global_crop_size <= 0:
            raise ValueError(
                f"global_crop_size must be positive, got {self.global_crop_size}")
        if self.n_local_crops < 0:
            raise ValueError(
                f"n_local_crops must be >= 0, got {self.n_local_crops}")
        if self.source_image_size is not None:
            if self.source_image_size < self.global_crop_size:
                raise ValueError(
                    f"source_image_size ({self.source_image_size}) must be >= "
                    f"global_crop_size ({self.global_crop_size}); decoding the "
                    f"records SMALLER than the crop size would upsample every "
                    f"view even further, which is the opposite of what this "
                    f"field is for."
                )
        if self.patch_size is not None and self.patch_size <= 0:
            raise ValueError(
                f"patch_size must be positive when set, got {self.patch_size}")
        if self.dino_out_dim <= 0:
            raise ValueError(
                f"dino_out_dim must be positive, got {self.dino_out_dim}")
        if self.student_temp <= 0:
            raise ValueError(
                f"student_temp must be positive, got {self.student_temp}")
        if self.teacher_temp <= 0 or self.teacher_temp_final <= 0:
            raise ValueError(
                f"teacher temperatures must be positive, got "
                f"{self.teacher_temp} -> {self.teacher_temp_final}"
            )
        if self.teacher_temp_warmup_epochs < 0:
            raise ValueError(
                f"teacher_temp_warmup_epochs must be >= 0, got "
                f"{self.teacher_temp_warmup_epochs}"
            )
        if not 0 <= self.center_momentum < 1:
            raise ValueError(
                f"center_momentum must be in [0, 1), got {self.center_momentum}")
        for name in ("ema_decay_start", "ema_decay_end"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.ema_warmup_steps < 0:
            raise ValueError(
                f"ema_warmup_steps must be >= 0, got {self.ema_warmup_steps}")
        # Refused here, in the house convention, rather than clamped in
        # `resolve_ema_warmup_steps`: a silently-clamped negative warmup would train
        # with a teacher schedule nobody asked for. NaN/inf are rejected too -- both
        # survive a `< 0` test and `int(round(nan * steps))` raises far from the cause.
        if not math.isfinite(self.ema_warmup_epochs) or self.ema_warmup_epochs < 0:
            raise ValueError(
                f"ema_warmup_epochs must be finite and >= 0, got "
                f"{self.ema_warmup_epochs}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(
                f"max_steps must be positive when set, got {self.max_steps}")
        if self.knn_eval_every < 0:
            raise ValueError(
                f"knn_eval_every must be >= 0 (0 disables the k-NN probe), got "
                f"{self.knn_eval_every}"
            )
        if self.knn_bank_batches <= 0 or self.knn_query_batches <= 0:
            raise ValueError(
                f"knn_bank_batches and knn_query_batches must be positive, got "
                f"{self.knn_bank_batches} and {self.knn_query_batches}"
            )
        if self.knn_temperature <= 0:
            raise ValueError(
                f"knn_temperature must be positive, got {self.knn_temperature}")
        if self.random_init_repeats < 0:
            raise ValueError(
                f"random_init_repeats must be >= 0 (0 disables the zero-step "
                f"control), got {self.random_init_repeats}"
            )

    @property
    def n_views(self) -> int:
        """Total views per sample: two global crops plus the local ones."""
        return N_GLOBAL_VIEWS + self.n_local_crops


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def build_dataset(config: TrainingConfig) -> Tuple[Any, int]:
    """Build the multi-crop training pipeline.

    Interface contract:
        Parameters:
            config: The trainer config.
        Returns:
            ``(train_ds, steps_per_epoch)``. The dataset yields
            ``(views, label)`` with ``views`` of shape
            ``(batch, n_views, S, S, 3)``; the label is carried through and
            IGNORED by ``DINOLoss`` (it is unlabelled pretraining), which is why
            no ``y_true``-shaped tensor is manufactured here.
        Failure mode:
            ``NotImplementedError`` from ``make_multi_crop_map_fn`` when
            ``local_crop_size`` differs from ``global_crop_size``; ``ValueError``
            for an unsupported dataset.

    NO validation pipeline is built. See the module docstring's Rule.
    """
    crop_kwargs = dict(
        local_crop_size=config.local_crop_size,
        n_local_crops=config.n_local_crops,
    )
    if config.stateless_augmentation:
        map_fn = make_stateless_multi_crop_map_fn(
            global_crop_size=config.global_crop_size,
            seed=config.seed,
            **crop_kwargs,
        )
    else:
        map_fn = make_multi_crop_map_fn(
            global_crop_size=config.global_crop_size,
            seed=config.seed,
            **crop_kwargs,
        )

    # augment=False on purpose: build_raw_image_dataset's own flip / pad-crop runs BEFORE
    # normalization and would stack a second, non-DINO augmentation under the multi-crop
    # transform. The DINO recipe's augmentation lives entirely in map_fn.
    # The DECODE resolution, i.e. what the multi-crop transform crops FROM.
    # `build_raw_image_dataset` resizes every record to this size before
    # `element_map_fn` runs; `map_fn` then emits views at `global_crop_size`
    # regardless. See `TrainingConfig.source_image_size` for the measurement
    # that motivates the knob and for why its default is behaviour-preserving.
    source_size = config.source_image_size or config.global_crop_size

    # The stateless map fn takes `(index, image, label)`, so it goes in the
    # INDEXED slot -- `build_raw_image_dataset` refuses both slots at once.
    # DECISION plan-2026-08-02T132301-93deeae2/D-004
    # As of this plan the INDEXED slot is the DEFAULT branch (the `element_map_fn`
    # branch is now the `--no-stateless-augmentation` path). The dict-spread is
    # unchanged and still correct; only which branch is the default has moved.
    map_fn_kwarg = (
        {"indexed_element_map_fn": map_fn}
        if config.stateless_augmentation
        else {"element_map_fn": map_fn}
    )

    # DECISION plan-2026-08-01T195746-12a1f2db/D-011
    # Built as a dict so that the no-seed call passes NO `shuffle_files_seed` at all,
    # rather than `shuffle_files_seed=None`. See `TrainingConfig.seed_training_stream`.
    # DECISION plan-2026-08-02T132301-93deeae2/D-004
    # D-011's wording above said "the DEFAULT call"; that is no longer the default
    # call. After this plan the DEFAULT branch PASSES `shuffle_files_seed=config.seed`,
    # and the empty-dict branch is the `--no-seed-training-stream` path. The spread
    # still exists for exactly the original reason, which the flip did not change:
    # absence of the kwarg is NOT the same as `shuffle_files_seed=None` at the
    # `ReadConfig` layer, so the off-switch must omit it rather than pass `None`.
    stream_seed_kwarg = (
        {"shuffle_files_seed": config.seed} if config.seed_training_stream else {}
    )

    train_ds, num_train, _ = build_raw_image_dataset(
        config.dataset,
        source_size,
        config.batch_size,
        is_training=True,
        augment=False,
        seed=config.seed,
        **map_fn_kwarg,
        **stream_seed_kwarg,
    )

    steps_per_epoch = max(1, num_train // config.batch_size)
    if config.max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_steps)
    return train_ds, steps_per_epoch


def build_knn_datasets(config: TrainingConfig) -> Tuple[Any, Any]:
    """Build the k-NN probe's memory bank and query set: SINGLE crops, with labels.

    Interface contract:
        Parameters:
            config: The trainer config.
        Returns:
            ``(bank_ds, query_ds)`` -- both batched ``(image, label)`` pipelines at
            the global crop resolution. NO multi-crop ``element_map_fn``: the probe
            evaluates the representation of a plain image, not of an augmented view.
        Failure mode:
            ``ValueError`` for an unsupported dataset (from
            ``build_raw_image_dataset``).

    **The bank comes from the TRAIN split and the queries from the VALIDATION split**,
    so the two are disjoint BY CONSTRUCTION -- which is the property that makes the
    reported accuracy mean anything (a query that finds itself in the bank at cosine
    1.0 scores a free hit). That disjointness is additionally checked numerically
    inside `knn_top1_accuracy`; the two guards are deliberate belt-and-braces because
    the failure is silent and looks like a good result.

    Both pipelines are ``.take(n).cache()``d so the SAME samples are used every epoch;
    an eval set that reshuffles each epoch turns run-to-run noise into apparent
    training progress.

    **The bank is seeded down to the TFDS FILE order (D-040), and that is load-bearing.**
    ``.take(knn_bank_batches)`` off the train stream selects a SMALL sample --
    ``knn_bank_batches * batch_size`` images out of 9469, i.e. **512 at the SMOKE
    defaults** (16 x 32) and 2048 at the 64/32 probe settings every measurement in
    `plan-2026-08-01T195746-12a1f2db` used -- and `build_raw_image_dataset` opens the train
    split with ``shuffle_files=True``. Without ``shuffle_files_seed`` the file interleave
    is non-deterministic ACROSS PROCESSES even at a fixed ``--seed``, so two runs score
    against two different memory banks. MEASURED before this was seeded: four bank draws
    at ``seed=42`` (two per process, two processes) gave four different label sequences,
    and four zero-optimizer-step k-NN controls at the smoke config spread
    ``dino_knn_top1_k20`` over **0.2754 / 0.2900 / 0.2910 / 0.2949 (range 0.0195)** and
    ``k10`` over **0.2773 / 0.2686 / 0.2793 / 0.2607 (range 0.0186)** -- a band LARGER
    than the +0.0127 effect a step-14 A/B was trying to read out of it. The QUERY set is
    unaffected (the validation split is opened with ``shuffle_files=False``), which is why
    ``dino_feat_mean_cos`` was bit-identical across those same repeats while the k-NN
    moved. See `knn_eval`'s module docstring for how to read the number.
    """
    bank_ds, _, _ = build_raw_image_dataset(
        config.dataset,
        config.global_crop_size,
        config.batch_size,
        is_training=True,   # -> the TRAIN split
        augment=False,      # a frozen-feature probe must not see augmentation
        seed=config.seed,
        # DECISION plan-2026-08-01T105809-dc0c402e/D-040
        # Do NOT drop this and rely on `seed=` alone: `seed` reaches the element
        # `.shuffle()` and the augmentation, NOT the TFDS file interleave, so the
        # `.take()` below would select a different bank every process.
        shuffle_files_seed=config.seed,
    )
    query_ds, _, _ = build_raw_image_dataset(
        config.dataset,
        config.global_crop_size,
        config.batch_size,
        is_training=False,  # -> the VALIDATION split
        seed=config.seed,
    )
    return (
        bank_ds.take(config.knn_bank_batches).cache(),
        query_ds.take(config.knn_query_batches).cache(),
    )


# ---------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------

def create_teacher_temp_callback(
        config: TrainingConfig,
        loss: DINOLoss,
) -> keras.callbacks.Callback:
    """Warm the teacher temperature up across epochs, with a STOCK callback.

    Interface contract:
        Parameters:
            config: Supplies ``teacher_temp`` (start), ``teacher_temp_final``
                (end) and ``teacher_temp_warmup_epochs`` (horizon).
            loss: The compiled ``DINOLoss`` whose ``set_teacher_temp`` is driven.
        Returns:
            A ``keras.callbacks.LambdaCallback`` that assigns the scheduled
            temperature at each ``on_epoch_begin``.
        Failure mode:
            ``ValueError`` from ``set_teacher_temp`` if a schedule ever produces
            a non-positive temperature (it cannot, both endpoints are validated
            positive by ``TrainingConfig.__post_init__``).

    Two things here are load-bearing and neither is cosmetic:

    * **The temperature must be a ``keras.Variable``, not a Python float.** A float is
      constant-folded into the traced training step: MEASURED, a 100x change to the plain
      attribute moved the reported loss by 7e-07, while the same change through
      ``set_teacher_temp`` moved it 9.95 -> 12.62. ``DINOLoss.teacher_temp`` is therefore
      read-only and a plain assignment RAISES.
    * **This is a stock ``LambdaCallback`` driving an EXISTING schedule function**
      (``linear_ema_schedule`` from ``dl_techniques.models.depth_anything.teacher_ema``).
      The repo already carries five schedule-callback classes with duplicated linear/cosine
      math and no shared base; do NOT add a sixth for this.
    """
    horizon = max(1, int(config.teacher_temp_warmup_epochs))
    schedule: Callable[[int], float] = linear_ema_schedule(
        decay_start=config.teacher_temp,
        decay_end=config.teacher_temp_final,
        total_steps=horizon,
    )

    def _set(epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        loss.set_teacher_temp(schedule(epoch))

    return keras.callbacks.LambdaCallback(on_epoch_begin=_set)


# DECISION plan-2026-08-02T132301-93deeae2/D-002
# This function has exactly ONE production call site (`create_callbacks`, below) and is
# therefore an UN-EARNED abstraction on its face. Do NOT inline it back "for simplicity":
# inlining puts the resolution behind a `DINOLoss` plus the k-NN bank/query datasets, so
# the only gate that stands in for a ~5 h GPU re-measurement -- that the shipped defaults
# resolve to the SAME warmup the measured `--ema-warmup-steps 295` invocation produced --
# would need data and a GPU to run. See decisions.md D-002.
def resolve_ema_warmup_steps(config: TrainingConfig, steps_per_epoch: int) -> int:
    """Resolve the teacher-EMA warmup to absolute optimizer steps.

    Precedence: ``config.ema_warmup_steps`` WINS whenever it is ``> 0``; otherwise the
    value is ``round(config.ema_warmup_epochs * steps_per_epoch)``.

    The EPOCH form is the default-bearing one because the measured value (295) is
    ``num_train // batch_size`` at one dataset and one batch size -- one epoch at that
    scale, and an arbitrary number at any other. The STEP form is retained as the exact,
    scale-free override every prior run recorded in ``results/*/config.json`` used.

    Interface contract:
        Parameters:
            config: The trainer config. Both fields are validated by ``__post_init__``
                (``ema_warmup_steps >= 0``; ``ema_warmup_epochs`` finite and ``>= 0``),
                so this function does no re-validation and cannot silently clamp.
            steps_per_epoch: Optimizer steps in one epoch, as computed by
                :func:`build_dataset` (already ``max_steps``-capped).
        Returns:
            Absolute step count, ``>= 0``. ``0`` means no freeze.
        Failure mode:
            None. A hand-built config bypassing ``__post_init__`` is out of contract.
    """
    if config.ema_warmup_steps > 0:
        return config.ema_warmup_steps
    return int(round(config.ema_warmup_epochs * steps_per_epoch))


def create_callbacks(
        config: TrainingConfig,
        loss: DINOLoss,
        run_dir: str,
        steps_per_epoch: int,
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Assemble the run's callbacks: the shared set, the teacher EMA, the temperature.

    Interface contract:
        Parameters:
            config: The trainer config.
            loss: The compiled ``DINOLoss`` (the temperature callback writes to it).
            run_dir: Exact directory the artifacts go into.
            steps_per_epoch: Drives the EMA decay schedule's horizon.
        Returns:
            ``(callbacks, results_dir)`` -- the ``create_callbacks`` contract from
            ``train.common``.
        Failure mode:
            Propagates whatever ``train.common.create_callbacks`` raises.

    ``monitor="loss"``, NOT ``"val_loss"``: this run has no validation data (module
    docstring). Monitoring a metric that is never produced makes ``ModelCheckpoint``
    silently never save and ``EarlyStopping`` silently never fire.
    """
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="dino",
        run_dir=run_dir,
        monitor="loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )

    # The teacher EMA. If DINOTrainingModel ever loses `update_teacher_ema`, this callback
    # logs ONE warning and SELF-DISABLES -- the run then completes, the loss curve looks
    # plausible, and the teacher is never updated. Grep the run log for
    # "TeacherEMACallback: model has no update_teacher_ema"; its ABSENCE is the assertion.
    callbacks.append(TeacherEMACallback(
        schedule=cosine_ema_schedule(
            decay_start=config.ema_decay_start,
            decay_end=config.ema_decay_end,
            total_steps=max(1, steps_per_epoch * config.epochs),
        ),
        warmup_steps=resolve_ema_warmup_steps(config, steps_per_epoch),
        log_every=max(1, steps_per_epoch),
    ))

    callbacks.append(create_teacher_temp_callback(config, loss))

    if config.knn_eval_every > 0:
        bank_ds, query_ds = build_knn_datasets(config)
        knn_callback = KNNEvalCallback(
            bank_ds,
            query_ds,
            bank_batches=config.knn_bank_batches,
            query_batches=config.knn_query_batches,
            temperature=config.knn_temperature,
            every_n_epochs=config.knn_eval_every,
            dino_loss=loss,
            # The ZERO-OPTIMIZER-STEP control, beside `config.json` in the SAME run
            # directory -- `run_dir`, not `results_dir`, because that is where
            # `train_dino` writes `config.json` and the reader needs the seed from
            # it to interpret the control. It cannot go into `training_log.csv`:
            # `CSVLogger` appends a row per `on_epoch_end` only (D-004).
            control_json_path=Path(run_dir) / "random_init_control.json",
            random_init_repeats=config.random_init_repeats,
        )
        # DECISION plan-2026-08-01T105809-dc0c402e/D-029
        # INSERT before `CSVLogger`, never `append`. `CSVLogger` freezes its
        # fieldnames from `sorted(logs.keys())` on the FIRST epoch it sees and then
        # writes the row -- so a callback that runs after it writes into an
        # already-serialized dict. MEASURED on keras 3.8.0, all three epochs:
        # appending after CSVLogger gave the header ['epoch', 'loss', 'val_loss'] and
        # the k-NN columns NEVER appeared. Same header when the key was first written
        # on epoch 1 instead of epoch 0 (which is why KNNEvalCallback writes every key
        # on every epoch, `nan` on skipped ones). Both failures are silent: the run
        # completes, the CSV looks well-formed, and the only validation signal this
        # trainer has is gone. Pinned by
        # tests/test_train/test_dino/test_knn_eval.py::TestCallbackOrdering.
        csv_index = next(
            (index for index, callback in enumerate(callbacks)
             if isinstance(callback, keras.callbacks.CSVLogger)),
            len(callbacks),
        )
        callbacks.insert(csv_index, knn_callback)

    return callbacks, results_dir


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def build_model_and_loss(
        config: TrainingConfig,
) -> Tuple[keras.Model, DINOLoss]:
    """Construct the training model and its loss, built and ready to compile.

    Interface contract:
        Parameters:
            config: The trainer config.
        Returns:
            ``(model, loss)``. The model is BUILT against the multi-crop input
            shape, so ``summary()`` and ``count_params()`` work before ``fit()``.
        Failure mode:
            ``ValueError`` from the DINO factories (e.g. a crop size not
            divisible by the patch size).
    """
    model = create_dino_training_model(
        config.variant,
        image_size=config.global_crop_size,
        patch_size=config.patch_size,
        n_local_views=config.n_local_crops,
        dino_out_dim=config.dino_out_dim,
    )
    model.build(
        (None, config.n_views, config.global_crop_size,
         config.global_crop_size, 3)
    )

    loss = DINOLoss(
        out_dim=config.dino_out_dim,
        student_temp=config.student_temp,
        teacher_temp=config.teacher_temp,
        center_momentum=config.center_momentum,
    )
    return model, loss


def train_dino(config: TrainingConfig) -> Dict[str, Any]:
    """Orchestrate DINO self-supervised pretraining.

    Returns:
        Dict with ``model``, ``loss``, ``first_loss``, ``final_loss``,
        ``run_dir``, ``history``.
    """
    setup_gpu(config.gpu)
    set_seeds(config.seed)

    logger.info(
        f"Experiment: {config.experiment_name} | variant={config.variant} "
        f"dataset={config.dataset} crop={config.global_crop_size} "
        f"views={config.n_views} out_dim={config.dino_out_dim}"
    )

    run_dir = Path(config.output_dir) / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(run_dir), "config.json")

    # ---- Data ----
    train_ds, steps_per_epoch = build_dataset(config)
    logger.info(f"Steps per epoch: {steps_per_epoch} (NO validation pipeline)")

    # ---- Model + loss ----
    model, loss = build_model_and_loss(config)
    model.summary(print_fn=logger.info)

    # ---- Optimization ----
    optimizer = build_optimizer(config, steps_per_epoch)

    # STOCK compile. The centering EMA lives inside DINOLoss.call(); the teacher EMA and
    # the temperature warmup are callbacks. Nothing overrides train_step.
    model.compile(optimizer=optimizer, loss=loss)

    # ---- Callbacks ----
    callbacks, results_dir = create_callbacks(
        config, loss, str(run_dir), steps_per_epoch)

    # ---- Train ----
    # DECISION plan-2026-08-01T105809-dc0c402e/D-028
    # Do NOT add `validation_data=` / `validation_steps=` here, and do NOT "fix" the
    # callbacks' monitor back to "val_loss". The omission is the mechanism, not an
    # oversight: `DINOLoss` advances its centering EMA inside `call()`, Keras runs the
    # loss over validation batches too, and `validation_batch_size` defaults to
    # `batch_size` -- so a validation set silently MULTIPLIES the per-epoch centering
    # updates. MEASURED: a 4-sample validation set at batch_size=2 took an epoch from 2
    # updates to 4 and pushed the center 81% past its correct value, with a finite loss
    # and a clean exit. There is no error to notice. Validation for SSL pretraining is a
    # k-NN probe on frozen features, which never invokes the loss.
    # Guarded by tests/test_train/test_dino/test_train_dino.py::TestNoValidationData.
    start = time.time()
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(f"Training completed in {(time.time() - start) / 3600.0:.3f} hours")

    model.save(str(run_dir / "final_model.keras"))

    loss_curve = history.history.get("loss", []) or [float("nan")]
    try:
        history_dict = {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        }
        with open(run_dir / "training_history.json", "w") as handle:
            json.dump(history_dict, handle, indent=2)
    except Exception as exc:  # pragma: no cover - best-effort artifact
        logger.warning(f"Failed to save training history: {exc}")

    gc.collect()
    return {
        "model": model,
        "loss": loss,
        "first_loss": float(loss_curve[0]),
        "final_loss": float(loss_curve[-1]),
        "run_dir": str(results_dir),
        "history": history,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the CLI. ``argv=None`` reads ``sys.argv`` (tests pass an explicit list)."""
    parser = argparse.ArgumentParser(
        description=(
            "DINO self-supervised pretraining under stock model.fit(). "
            "This trainer NEVER passes validation_data: DINOLoss updates its centering "
            "EMA inside call(), Keras runs the loss on validation batches too, and "
            "validation_batch_size defaults to batch_size -- so a validation set silently "
            "multiplies the per-epoch centering updates (MEASURED: 4 instead of 2, pushing "
            "the center 81 percent past its correct value). Validation for SSL pretraining "
            "is a "
            "k-NN probe on frozen features, which never invokes the loss. Because there is "
            "no val_loss, the callbacks monitor the TRAINING loss."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="imagenette",
                        choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--global-crop-size", type=int, default=224,
                        help="Side length of EVERY view, global and local alike")
    parser.add_argument("--local-crop-size", type=int, default=None,
                        help="Must be None or equal to --global-crop-size; anything else "
                             "raises NotImplementedError naming positional-embedding "
                             "interpolation (local views crop a smaller AREA and resize up)")
    parser.add_argument("--n-local-crops", type=int, default=4)
    parser.add_argument("--source-image-size", type=int, default=None,
                        help="Resolution at which records are DECODED, i.e. what the "
                             "multi-crop transform crops from. None = --global-crop-size, "
                             "which means local crops are taken from an already-"
                             "downsampled thumbnail (MEASURED at global-crop-size=96: "
                             "local crop sides 19-69 px, upsampled 2.33x mean / 4.50x "
                             "worst case, 100%% of local views). Set it larger (e.g. 224) "
                             "to crop from a bigger source and resize DOWN: at 224 the "
                             "mean upsample falls to 1.006 and 39%% of local views are "
                             "still upsampled, i.e. MITIGATED not eliminated. MEASURED "
                             "end-to-end (224 vs None, 2 seeds, controlled stream): NO "
                             "DIFFERENCE in k-NN top-1 (+0.0024). Must be >= "
                             "--global-crop-size")
    parser.add_argument("--stateless-augmentation",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Draw the multi-crop augmentation from "
                             "tf.random.stateless_uniform keyed on a per-element "
                             "counter instead of one shared tf.random.Generator "
                             "stream. This is what makes --seed reproduce the "
                             "AUGMENTATION stream under the shipped AUTOTUNE map "
                             "(MEASURED at HEAD: two same-seed parallel maps "
                             "differ, maxdiff 1.5312). DEFAULT ON, so a run with "
                             "no flags is reproducible. REQUIRED TOGETHER WITH "
                             "--seed-training-stream: MEASURED across two "
                             "processes, this flag ALONE still gives different "
                             "batches; both together are bit-identical. "
                             "--no-stateless-augmentation restores the old shared-"
                             "Generator behaviour and with it comparability to any "
                             "run in results/ measured before this default moved. "
                             "Caveat: this path has only ever been exercised end-"
                             "to-end in one plan's confirm/long matrices. See "
                             "research/2026_dino_ssl_measurements.md")
    parser.add_argument("--seed-training-stream",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Seed the TRAINING stream's TFDS file interleave with "
                             "--seed, the way the k-NN memory bank already is "
                             "(D-040). Without it --seed fixes the measuring "
                             "instrument but not the data: `seed` reaches the "
                             "element shuffle and the augmentation, NOT the file "
                             "order, so two same-seed runs see a different example "
                             "order from step 0. DEFAULT ON. NOT SUFFICIENT ALONE: "
                             "MEASURED across two processes, this flag by itself "
                             "still gives different batches (the augmentation RNG "
                             "is the residual source) -- it needs "
                             "--stateless-augmentation too. "
                             "--no-seed-training-stream restores the unpinned file "
                             "order and with it comparability to any run in "
                             "results/ measured before this default moved. See "
                             "research/2026_dino_ssl_measurements.md")

    # Model
    parser.add_argument("--variant", type=str, default="small", choices=list(VARIANTS))
    parser.add_argument("--patch-size", type=int, default=None,
                        help="None defers to the variant's own patch size")
    parser.add_argument("--dino-out-dim", type=int, default=65536,
                        help="Projection-head width. Paper: 65536. Smoke scale: 4096")

    # Loss
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--teacher-temp", type=float, default=0.04,
                        help="Teacher temperature at epoch 0 (start of the warmup)")
    parser.add_argument(
        "--teacher-temp-final", type=float, default=0.04,
        help="Teacher temperature after the warmup. Default 0.04 EQUALS "
             "--teacher-temp, i.e. no warmup at all. (a) Only the PAIR "
             "(--ema-warmup-epochs 1.0, the 295-step freeze, TOGETHER with this 0.04) "
             "has a 60-epoch IMPROVED verdict, at 2 seeds, smoke scale, controlled "
             "stream (plan-2026-08-01T195746-12a1f2db); no temp-only arm was ever run "
             "at 60 epochs. (b) The teacher-temp arm ALONE was CLEARED at the "
             "pre-registered epoch-0 endpoint -- structurally inert there, since the "
             "schedule callback assigns only on_epoch_begin, and it returned an "
             "exactly-zero null at both seeds -- AMBIGUOUS on the exploratory "
             "last-3-epochs endpoint, and its exploratory effect SHRANK from +0.0518 to "
             "+0.0387 once the training stream was controlled; it shipped into the "
             "improved arm against that matrix's own 'must be confirmed, not shipped' "
             "label. (c) Setting this equal to --teacher-temp makes the linear schedule "
             "CONSTANT, which makes --teacher-temp-warmup-epochs an INERT knob. "
             "(d) This DEPARTS from the DINO paper's 0.04 -> 0.07 warmup: to reproduce "
             "the paper recipe pass --teacher-temp-final 0.07. Full evidence: "
             "research/2026_dino_ssl_measurements.md")
    parser.add_argument(
        "--teacher-temp-warmup-epochs", type=int, default=30,
        help="Epoch horizon of the linear --teacher-temp -> --teacher-temp-final ramp. "
             "INERT AT THE SHIPPED DEFAULTS: the two endpoints are both 0.04, so the "
             "schedule is constant and this value changes nothing. It becomes live again "
             "only when --teacher-temp-final differs from --teacher-temp (e.g. the paper "
             "recipe, --teacher-temp-final 0.07)")
    parser.add_argument("--center-momentum", type=float, default=0.9)

    # Teacher EMA
    parser.add_argument("--ema-decay-start", type=float, default=0.996)
    parser.add_argument("--ema-decay-end", type=float, default=0.9999)
    parser.add_argument(
        "--ema-warmup-epochs", type=float, default=1.0,
        help="Freeze the teacher-weight EMA for the first N EPOCHS (teacher stays "
             "at its student-synced init). This is the DEFAULT-BEARING knob; "
             "default 1.0 = freeze the teacher for the first epoch. MEASURED "
             "(plan-2026-08-01T195746-12a1f2db) at N=295 absolute steps, which is "
             "one epoch at imagenette/batch-32: the epoch-0 k-NN rises +0.0498 vs "
             "the old no-freeze default on a controlled stream whose null is "
             "exactly 0.000, removing the dip at seed 42 and halving it at seed "
             "1337. NOTE it is a SUPERSET: it also shifts the cosine EMA ramp that "
             "many steps later, because the post-warmup index restarts at 0 while "
             "total_steps is unchanged -- freeze and re-basing have NEVER been "
             "separated (see the Open Questions in "
             "research/2026_dino_ssl_measurements.md). It is denominated in epochs "
             "because 295 is num_train // batch_size at ONE scale, not a portable "
             "constant. 0 = no freeze. --ema-warmup-steps N overrides this in "
             "absolute steps.")
    parser.add_argument(
        "--ema-warmup-steps", type=int, default=0,
        help="ABSOLUTE-step override for the teacher-EMA warmup described under "
             "--ema-warmup-epochs. Any value > 0 WINS over --ema-warmup-epochs; 0 "
             "(the default) defers to it. Use this to reproduce a run recorded "
             "before the epoch-denominated default existed, e.g. the measured "
             "--ema-warmup-steps 295.")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adam", "sgd", "rmsprop"])
    parser.add_argument("--lr-schedule", type=str, default="cosine_decay",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--gradient-clipping", type=float, default=3.0)

    # Monitoring
    parser.add_argument("--early-stopping-patience", type=int, default=30,
                        help="Patience on the TRAINING loss (there is no val_loss)")

    # k-NN probe + collapse diagnostic
    parser.add_argument("--knn-eval-every", type=int, default=1,
                        help=(
                            "Epochs between frozen-feature k-NN evaluations. 0 turns "
                            "the probe OFF, which also turns off the collapse "
                            "diagnostic -- a decreasing loss then proves nothing about "
                            "the representation."
                        ))
    parser.add_argument("--knn-bank-batches", type=int, default=16,
                        help="Memory-bank batches, drawn from the TRAIN split")
    parser.add_argument("--knn-query-batches", type=int, default=8,
                        help="Query batches, drawn from the VALIDATION split (disjoint "
                             "from the bank by construction)")
    parser.add_argument("--knn-temperature", type=float,
                        default=DEFAULT_KNN_TEMPERATURE,
                        help="Temperature of the exp(sim / T) neighbour weighting")
    parser.add_argument("--random-init-repeats", type=int, default=2,
                        help=(
                            "Repeats of the ZERO-OPTIMIZER-STEP k-NN control "
                            "written to <run_dir>/random_init_control.json before "
                            "fit() performs a single update. 0 disables it. This "
                            "is the baseline a dino_knn_top1_* delta must be read "
                            "against: an UNTRAINED ViT already scores ~0.28 on "
                            "imagenette here, so the 0.10 chance line is mostly "
                            "architecture. The epoch-0 row of training_log.csv is "
                            "NOT this number -- it is post-one-epoch."
                        ))

    # Debug
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap steps_per_epoch. Smoke runs only.")
    parser.add_argument("--smoke", action="store_true",
                        help=(
                            "Pin the MEASURED shape-validation scale: variant=tiny, "
                            "global-crop-size=96, n-local-crops=4, batch-size=32, "
                            "dino-out-dim=4096, 2 epochs x 5 steps. Peaks at 1518.6 MiB of "
                            "10001 MiB on an RTX 4070 -- per-step working set from a "
                            "dedicated get_memory_info('peak') probe (D-026), NOT what "
                            # `%%` is NOT a typo: argparse runs every help string
                            # through `help % params`, so a literal percent sign
                            # must be doubled or `--help` dies with
                            # `ValueError: unsupported format character`.
                            "nvidia-smi shows during a run (TF pre-allocates ~85%%; see "
                            "the module docstring). This validates SHAPES and wiring; it "
                            "is NOT a paper reproduction (the paper uses out_dim=65536, "
                            "224px globals and hundreds of epochs). Explicit flags still "
                            "win over the preset."
                        ))

    # Output
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Repo-root results/, never src/results/")
    parser.add_argument("--experiment-name", type=str, default=None)

    # Runtime
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds weight initialization, dataset shuffling and the "
                             "multi-crop generator. ON ITS OWN it does NOT make a run "
                             "bit-reproducible: the stateful multi-crop transform's seed "
                             "reproduces a SERIAL .map() only and this pipeline uses "
                             "num_parallel_calls=AUTOTUNE (maxdiff 1.5312), and the TFDS "
                             "file interleave is unseeded. Add BOTH "
                             "--seed-training-stream AND --stateless-augmentation and two "
                             "same-seed processes produce BIT-IDENTICAL batches "
                             "(MEASURED); either flag alone is not enough. See "
                             "dl_techniques/datasets/vision/multi_crop.py's module "
                             "docstring")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")

    return parser.parse_args(argv)


# Config-field -> argparse dest, for the handful that differ. Used by the --smoke preset
# to tell "the caller left this at the default" from "the caller asked for this value".
_ARG_FOR_FIELD: Dict[str, str] = {
    "optimizer_type": "optimizer",
    "lr_schedule_type": "lr_schedule",
}


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Map a parsed ``Namespace`` onto a :class:`TrainingConfig`. PURE -- no side effects.

    Every flag in :func:`parse_arguments` must land in a field here. A flag that does not is
    a SILENT NO-OP: the run trains at the dataclass default while the command line says
    otherwise, and nothing fails. That trap has bitten this repo before (``train/bfunet``'s
    ``high_freq_blocks`` and ``filter_multiplier`` were both silent no-ops for real runs),
    which is why this mapping is an importable function with a dedicated structural test
    (``tests/test_train/test_dino/test_train_dino.py``) rather than a block inside
    ``main()``.

    ``--smoke`` is the ONE flag that feeds no field of its own: it is a preset that
    overrides other fields, applied here where the test can see it.
    """
    values: Dict[str, Any] = dict(
        dataset=args.dataset.lower(),
        global_crop_size=args.global_crop_size,
        local_crop_size=args.local_crop_size,
        source_image_size=args.source_image_size,
        stateless_augmentation=args.stateless_augmentation,
        seed_training_stream=args.seed_training_stream,
        n_local_crops=args.n_local_crops,
        variant=args.variant,
        patch_size=args.patch_size,
        dino_out_dim=args.dino_out_dim,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        teacher_temp_final=args.teacher_temp_final,
        teacher_temp_warmup_epochs=args.teacher_temp_warmup_epochs,
        center_momentum=args.center_momentum,
        ema_decay_start=args.ema_decay_start,
        ema_decay_end=args.ema_decay_end,
        ema_warmup_steps=args.ema_warmup_steps,
        ema_warmup_epochs=args.ema_warmup_epochs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        early_stopping_patience=args.early_stopping_patience,
        knn_eval_every=args.knn_eval_every,
        knn_bank_batches=args.knn_bank_batches,
        knn_query_batches=args.knn_query_batches,
        knn_temperature=args.knn_temperature,
        random_init_repeats=args.random_init_repeats,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
        gpu=args.gpu,
    )

    if args.smoke:
        # An EXPLICIT flag still wins over the preset -- the preset only fills fields the
        # caller left at the parser default, so `--smoke --batch-size 8` really uses 8.
        defaults = parse_arguments([])
        for field_name, smoke_value in SMOKE_OVERRIDES.items():
            arg_name = _ARG_FOR_FIELD.get(field_name, field_name)
            if getattr(args, arg_name) == getattr(defaults, arg_name):
                values[field_name] = smoke_value

    return TrainingConfig(**values)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    config = config_from_args(parse_arguments())

    logger.info(
        f"Config: variant={config.variant}, dataset={config.dataset}, "
        f"crop={config.global_crop_size}, views={config.n_views}, "
        f"out_dim={config.dino_out_dim}, {config.epochs} epochs, "
        f"batch={config.batch_size}, lr={config.learning_rate}"
    )

    try:
        result = train_dino(config)
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        raise

    logger.info(
        f"=== DINO PRETRAINING DONE === first_loss={result['first_loss']:.6f} "
        f"final_loss={result['final_loss']:.6f} run_dir={result['run_dir']}"
    )


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
