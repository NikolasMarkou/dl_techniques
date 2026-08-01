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

-------------------------------------------------------------------------------
THE ONE RULE THAT IS NOT OPTIONAL: this trainer NEVER passes ``validation_data``
-------------------------------------------------------------------------------
`DINOLoss` maintains its centering statistic by ``.assign()``-ing a ``keras.Variable``
inside ``call()``. That is correct under stock ``fit()`` -- but ``call()`` runs on EVERY
batch, and Keras runs the loss on validation batches too, so each one performs a full,
unwanted centering EMA update. This is not a rounding error: MEASURED, a 4-sample
validation set at ``batch_size=2`` doubled an epoch's update count from 2 to 4 and pushed
the center **81% past** its correct value -- silently, with a finite loss and a clean exit.
``validation_batch_size`` defaults to ``batch_size``, so the corruption scales with the
validation batch COUNT.

Consequences, both deliberate:

* ``model.fit(...)`` below takes no ``validation_data`` and no ``validation_steps``.
* There is therefore no ``val_loss``, so the callbacks monitor the TRAINING ``loss``.
  Real validation for SSL pretraining is a k-NN probe on frozen features, which does not
  invoke the loss at all (that callback is a separate module; see "Seams" below).

-------------------------------------------------------------------------------
Scale
-------------------------------------------------------------------------------
``--smoke`` pins the MEASURED shape-validation scale: ``variant=tiny``,
``global_crop_size=96``, ``n_local_crops=4``, ``batch_size=32``, ``dino_out_dim=4096``,
a handful of steps. On GPU 1 (RTX 4070) one full train step at that scale peaks at
**1518.6 MiB of 10001 MiB**. That is a SHAPE-VALIDATION scale, **NOT a paper
reproduction** -- DINO uses ``dino_out_dim=65536``, 224px globals and hundreds of epochs.
The defaults (no ``--smoke``) are the paper-shaped ones and are correspondingly expensive.

Seams deliberately left open
    A k-NN evaluation callback (frozen-feature top-1 + a collapse diagnostic) belongs in
    ``callbacks`` next to ``TeacherEMACallback``. It is not imported here yet; nothing in
    this module forward-references a module that does not exist.

Usage::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.dino.train_dino --smoke
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.dino.train_dino \\
        --variant small --global-crop-size 224 --dino-out-dim 65536 --epochs 100 --gpu 1
"""

import gc
import json
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
from dl_techniques.datasets.vision.multi_crop import make_multi_crop_map_fn
from dl_techniques.models.dino.dino_training import (
    N_GLOBAL_VIEWS,
    create_dino_training_model,
)
from dl_techniques.models.depth_anything.teacher_ema import (
    TeacherEMACallback,
    cosine_ema_schedule,
    linear_ema_schedule,
)

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

VARIANTS: Tuple[str, ...] = ("tiny", "small", "base", "large", "giant")

# The MEASURED shape-validation scale (D-026): one full train step peaks at 1518.6 MiB of
# the 10001 MiB free on GPU 1, i.e. ~15%. The three pre-committed memory reductions
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

    # Model
    variant: str = "small"
    patch_size: Optional[int] = None  # None defers to the variant (D-017)
    dino_out_dim: int = 65536  # paper scale; the smoke scale uses 4096

    # Loss
    student_temp: float = 0.1
    teacher_temp: float = 0.04  # start of the warmup
    teacher_temp_final: float = 0.07  # end of the warmup (paper: 0.04 -> 0.07)
    teacher_temp_warmup_epochs: int = 30
    center_momentum: float = 0.9

    # Teacher EMA
    ema_decay_start: float = 0.996
    ema_decay_end: float = 0.9999
    ema_warmup_steps: int = 0

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
    map_fn = make_multi_crop_map_fn(
        global_crop_size=config.global_crop_size,
        local_crop_size=config.local_crop_size,
        n_local_crops=config.n_local_crops,
        seed=config.seed,
    )

    # augment=False on purpose: build_raw_image_dataset's own flip / pad-crop runs BEFORE
    # normalization and would stack a second, non-DINO augmentation under the multi-crop
    # transform. The DINO recipe's augmentation lives entirely in map_fn.
    train_ds, num_train, _ = build_raw_image_dataset(
        config.dataset,
        config.global_crop_size,
        config.batch_size,
        is_training=True,
        augment=False,
        element_map_fn=map_fn,
        seed=config.seed,
    )

    steps_per_epoch = max(1, num_train // config.batch_size)
    if config.max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_steps)
    return train_ds, steps_per_epoch


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
        warmup_steps=config.ema_warmup_steps,
        log_every=max(1, steps_per_epoch),
    ))

    callbacks.append(create_teacher_temp_callback(config, loss))

    # SEAM: the k-NN evaluation callback (frozen-feature top-1 + the collapse diagnostic)
    # is appended here once it exists. Nothing above forward-references it.
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
    parser.add_argument("--teacher-temp-final", type=float, default=0.07,
                        help="Teacher temperature after the warmup")
    parser.add_argument("--teacher-temp-warmup-epochs", type=int, default=30)
    parser.add_argument("--center-momentum", type=float, default=0.9)

    # Teacher EMA
    parser.add_argument("--ema-decay-start", type=float, default=0.996)
    parser.add_argument("--ema-decay-end", type=float, default=0.9999)
    parser.add_argument("--ema-warmup-steps", type=int, default=0)

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

    # Debug
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap steps_per_epoch. Smoke runs only.")
    parser.add_argument("--smoke", action="store_true",
                        help=(
                            "Pin the MEASURED shape-validation scale: variant=tiny, "
                            "global-crop-size=96, n-local-crops=4, batch-size=32, "
                            "dino-out-dim=4096, 2 epochs x 5 steps. Peaks at 1518.6 MiB of "
                            "10001 MiB on an RTX 4070. This validates SHAPES and wiring; it "
                            "is NOT a paper reproduction (the paper uses out_dim=65536, "
                            "224px globals and hundreds of epochs). Explicit flags still "
                            "win over the preset."
                        ))

    # Output
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Repo-root results/, never src/results/")
    parser.add_argument("--experiment-name", type=str, default=None)

    # Runtime
    parser.add_argument("--seed", type=int, default=42)
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
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        early_stopping_patience=args.early_stopping_patience,
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
