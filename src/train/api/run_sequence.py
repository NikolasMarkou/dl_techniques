"""The sequence a run follows once its directory is claimed and its lock is held.

``TaskTrainer.run`` resolves the request, claims ``out``, takes ``run.lock``, selects the device
and calls :func:`execute`; everything from the first line of ``run.log`` to the
:class:`~train.api.trainer.TrainResult` is here, cut at its seams so that no function is longer
than what one reads at a sitting (D-070: ``_train`` had grown to 345 lines past D-030's trigger of
300). The sequence is still ONE sequence, written once: a task supplies hooks, never stages.

Interface contract
------------------
``execute(trainer, family, variant, run_dir, facts, config, context, resume=None)``
    :param trainer: the task's :class:`~train.api.trainer.TaskTrainer`; its hooks are called here.
    :param config: the RESOLVED config (``trainer.config`` is the one the caller built).
    :param resume: the :class:`~train.common.resume_state.ResumePoint` of a ``resume_interrupted``
        run, else ``None``.
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigError: for a request the loaded data, the model factory or the files refuse
        (the run directory is then used: the message says so).
    :raises RuntimeError: for a non-finite loss (the diverged summary and the history are written
        first) or a best checkpoint that did not load (the summary is written first).

The stages, in order, each a function of one :class:`Run` (the state they share, filled as they go):

1. :func:`announce`: what the log opens with, ``config.json``, the recipe notices.
2. :func:`load_data`: ``build_data``, ``input_check``, the refusals that need the data's length, and
   the resume's checks of that data (``resume_decisions``).
3. :func:`build_model`: the model, its optional ``checkpoint`` weights, the optimizer, ``compile``.
4. :func:`epoch_zero`: the untrained model's evaluation (or the sidecar's copy of it on a resume),
   the checks of the monitor against the names ``compile`` produced, the summary head.
5. :func:`train_or_restore`: ``fit`` (or the files, for a run with no epoch left), then
   ``summarize_fit``; a non-finite loss raises here.
6. :func:`report`: ``evaluate_and_report``, the summary, the success verdict.

No stage holds an ``except``: the three context managers below are the only handlers on the path
(a bad value of the loader or the factory becomes a ``ConfigError``, an undecodable image is named,
a failure is logged to ``run.log`` and propagates), and ``test_task_trainer`` censuses the stages.
``train_or_restore`` is the one function that calls ``fit`` and the one that writes the history of a
diverged run: ``test_a_history_write_follows_the_model_save`` accepts a history write only in the
function that calls ``fit(callbacks=...)`` (the callbacks, ``ModelCheckpoint`` among them, are built
by :func:`train.api.fit_callbacks.build_callbacks`).

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); ``keras``, ``tensorflow``
and ``train.common`` names are imported inside the functions that use them.
"""

import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from . import registry, resume_decisions
from .args import AnalysisArgs, CheckpointArgs, MetricArgs, OptimArgs
from .config import StrictConfig, to_json
from .data_args import DataFacts
from .errors import ConfigError
from .optim import build_optimizer_and_schedule
from .run_dir import (
    CHECKPOINTS_DIR,
    TRAINING_LOG_NAME,
    write_requested,
    write_resolved,
    write_task,
)
from .trainer import FitState, TrainResult, _monitor_choices


# ---------------------------------------------------------------------
# The three handlers on the path
# ---------------------------------------------------------------------


def _claimed_note(run_dir: Path) -> str:
    """Return the sentence that tells a user a refused run still used ``out``."""
    return (
        f"The run directory {run_dir} was claimed before this failed, so 'out' is now "
        "used; choose a new 'out' ('out=auto' picks a new timestamped directory) after "
        "fixing the value."
    )


@contextlib.contextmanager
def _bad_value_is_a_config_error(run_dir: Path, step: str) -> Iterator[None]:
    """Turn a ``ValueError`` or ``TypeError`` of ``step`` into a ``ConfigError``.

    :param run_dir: the claimed run directory, named in the message.
    :param step: what ran, e.g. ``"the data loader"``.
    :raises ConfigError: carrying the original message and :func:`_claimed_note`.
    """
    # DECISION plan-2026-10-30T132432-a865fa08/D-046: the data loader and the model
    # factory refuse a bad value with a ValueError or a TypeError. Do NOT let it
    # reach the user as a traceback and exit 1, the code for a failed run: the
    # request was wrong, which is exit 2. The handler is HERE and not in ``run``,
    # ``execute`` or ``_resolve``, which hold no ``except`` (their census), and
    # it wraps only these two calls (review-iter-2 concern 5).
    try:
        yield
    except (ValueError, TypeError) as exc:
        # DECISION plan-2026-10-30T132432-a865fa08/D-053: a genuine bug that raises
        # ``TypeError`` or ``ValueError`` here is reported as a configuration error, and
        # the command line prints only the message. Do NOT drop the original traceback:
        # ``-v`` (DEBUG) must show where it was raised, in ``run.log`` too.
        from dl_techniques.utils.logger import logger

        logger.debug(f"Original error of {step}:", exc_info=True)
        raise ConfigError(
            f"Invalid argument for {step}: {exc}. {_claimed_note(run_dir)}"
        ) from exc


@contextlib.contextmanager
def _a_corrupt_file_is_named(
    run_dir: Path, files: Optional[Sequence[str]], step: str
) -> Iterator[None]:
    """Turn TensorFlow's ``InvalidArgumentError`` of an undecodable image into a ``ConfigError``.

    Wraps the first evaluation and ``fit``, the calls that run the data pipelines. When the
    error is raised the ``files`` are decoded one by one to find the culprit; if one is found
    it is named, if none is, the original error propagates unchanged (an
    ``InvalidArgumentError`` that is not about a file is somebody else's bug).

    :param run_dir: the claimed run directory, named in the message.
    :param files: the files the data pipelines read (the task's
        :meth:`~train.api.trainer.TaskTrainer.files_to_scan`); ``None`` or empty, as for the
        built-in datasets, has nothing to scan.
    :param step: what ran, e.g. ``"fit"``.
    :raises ConfigError: naming the file, and that the run directory is used.
    """
    import tensorflow as tf

    # DECISION plan-2026-10-30T132432-a865fa08/D-070: ONE narrow class is caught, here and
    # nowhere else in the run. Do NOT add ``tf.data``'s ``ignore_errors`` (a skipped image
    # changes the epoch, the statistics and the split with nothing said) and do NOT catch
    # ``Exception``: an error that is not a decode failure of a listed file is re-raised
    # as it came. The scan costs a decode pass only when a file has already failed.
    try:
        yield
    except tf.errors.InvalidArgumentError as exc:
        from train.common.classification_data import scan_for_undecodable

        found = scan_for_undecodable(files) if files else None
        if found is None:
            raise
        raise ConfigError(
            f"The image {found[0]!r} cannot be decoded ({found[1]}); it stopped {step}. Repair "
            f"or remove the file. {_claimed_note(run_dir)}"
        ) from exc


@contextlib.contextmanager
def _a_failure_is_logged(
    stage: str, hint: Optional[Callable[[], str]] = None
) -> Iterator[None]:
    """Write an exception raised inside the block to ``run.log`` as an ERROR, then let it go.

    Interface contract: wraps ``fit`` and the reporting of :func:`execute`
    (``evaluate_and_report``, the summary and its write), inside ``attach_run_log``. It has no
    ``except`` clause: the exception is read from ``sys.exc_info()`` in a ``finally`` and keeps
    propagating, so it cannot be swallowed, narrowed or replaced, the exit code stays the
    interpreter's 1 for an uncaught exception and stderr still gets the traceback.
    ``SystemExit`` and ``KeyboardInterrupt`` are not ``Exception`` and are not logged as failures.

    :param stage: ``"Training"`` or ``"Reporting"``, the first word of the line.
    :param hint: called only when an exception is on its way; returns the sentence appended
        after the exception (``""`` for none). A hint that raises ``ConfigError``, ``OSError``
        or ``ValueError`` gives no sentence and never replaces the failure being logged.
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-041
    # `finally` + `sys.exc_info()`, NOT `except Exception: ...; raise`. The API source rule
    # `no_broad_handler` forbids any broad handler under src/train/api (a re-raising one
    # included, test_the_api_source_rules.py), and the stages hold no `except` (their census).
    # Do NOT turn this into a handler: the log line is a side effect of an exception already
    # on its way, and a `finally` cannot change where it goes. Before this the OOM of the
    # gate's nano run was in stderr only and run.log ended on its last INFO line. D-064 puts
    # `fit` under the same context (an exception in `on_train_end` was in stderr only).
    try:
        yield
    finally:
        failure = sys.exc_info()[1]
        if isinstance(failure, Exception):
            from dl_techniques.utils.logger import logger

            try:
                again = hint() if hint is not None else ""
            except (ConfigError, OSError, ValueError):
                again = ""
            what = f"{stage} failed: {type(failure).__name__}: {failure}".rstrip()
            logger.error(what + ("" if what.endswith(".") else ".") + again)


# ---------------------------------------------------------------------
# The state the stages share
# ---------------------------------------------------------------------


@dataclasses.dataclass
class Run:
    """What the stages of one run share: the inputs first, then what each stage fills in.

    Mutable on purpose (a stage assigns, a later stage reads); nothing outside this module holds
    one. ``resume`` is REBOUND by :func:`load_data` when the backup settles on its previous pair.
    """

    trainer: Any
    family: str
    variant: str
    run_dir: Path
    facts: DataFacts
    config: StrictConfig
    context: Any
    resume: Any
    #: Why no epoch is left to train (``resume_decisions.training_complete_reason``), else ``None``.
    complete: Optional[str]
    sidecar_path: Path
    experiment_name: str
    devices: Dict[str, Any] = dataclasses.field(default_factory=dict)
    data: Any = None
    steps_per_epoch: int = 0
    model: Any = None
    params: int = 0
    baseline: Dict[str, float] = dataclasses.field(default_factory=dict)
    initial_ratio: float = 0.0
    init_warning: Optional[bool] = None
    best_weights: Optional[List[Any]] = None
    summary_head: Dict[str, Any] = dataclasses.field(default_factory=dict)
    vis_dir: Optional[Path] = None
    last_weights: Any = None
    history: Any = None
    epoch_times: List[float] = dataclasses.field(default_factory=list)
    fit_wall_seconds: float = 0.0
    hist: Dict[str, List[float]] = dataclasses.field(default_factory=dict)
    epochs_run: int = 0
    lr_reductions: Optional[List[int]] = None
    best_epoch: int = 0

    @property
    def reporting_only(self) -> bool:
        """``True`` for a resume of a run with no epoch left: ``fit`` is not run, the files are."""
        return self.complete is not None


# ---------------------------------------------------------------------
# The sequence
# ---------------------------------------------------------------------


def execute(
    trainer: Any,
    family: str,
    variant: str,
    run_dir: Path,
    facts: DataFacts,
    config: StrictConfig,
    context: Any,
    resume: Any = None,
) -> TrainResult:
    """Run the stages in order inside ``run.log``; see the module contract."""
    from dl_techniques.utils.logger import logger
    from train.common.resume_state import FIT_STATE_NAME
    from train.common.run_artifacts import attach_run_log

    if resume is None:
        write_requested(run_dir, trainer.config)
    sidecar_path = run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME
    with attach_run_log(run_dir, mode="w" if resume is None else "a"):
        logger.info(f"Run directory: {run_dir}")
        run = Run(
            trainer=trainer,
            family=family,
            variant=variant,
            run_dir=run_dir,
            facts=facts,
            config=config,
            context=context,
            resume=resume,
            complete=(
                None if resume is None
                else resume_decisions.training_complete_reason(resume, config)
            ),
            sidecar_path=sidecar_path,
            experiment_name=run_dir.name,
        )
        announce(run)
        load_data(run)
        build_model(run)
        epoch_zero(run)
        train_or_restore(run)
        return report(run)


def announce(run: Run) -> None:
    """Open the log: the resume line, ``Args``, the devices, ``Resolved args``, ``config.json``, the notices."""
    from dl_techniques.utils.logger import logger
    from train.common import run_summary

    trainer, config, resume = run.trainer, run.config, run.resume
    if run.reporting_only:
        # DECISION plan-2026-10-31T180033-2faf890a/D-041 (D-064: the same for a run whose
        # process died between its last epoch and `fit` returning)
        # A run whose `fit` returned and whose reporting died is NOT retrained: `fit`
        # is skipped, the weights come from the files, and only the reporting runs
        # again. Do NOT send it through `fit` with nothing left to train (no epoch
        # ends, so nothing is reported) and do NOT refuse it (the days of training
        # were the cost of a reporting bug, the gate's nano run). training_log.csv is
        # not copied or rewritten: it must be byte-for-byte what the training wrote.
        logger.info(
            f"Resuming a run whose training is complete ({resume.start} epochs of "
            f"{config.epochs}: {run.complete}) and whose reporting did not finish: fit is "
            "NOT run again. The best weights come from best_model.keras, the last epoch's from "
            "checkpoints/backup/latest.weights.h5, the history from training_log.csv; "
            "config_requested.json, config.json and training_log.csv stay as they are, "
            "run.log continues"
        )
    elif resume is not None:
        logger.info(
            (f"Resuming after epoch {resume.start} of {config.epochs}: "
             if resume.start else
             f"Resuming inside epoch 0 of {config.epochs} (a backup_every_steps copy): ")
            + "config_requested.json and config.json stay as they are, run.log and "
            "training_log.csv continue"
        )
    logger.info(f"Args:\n{to_json(trainer.config)}")
    run.devices = run_summary.describe_devices()
    logger.info(
        f"Devices: CUDA_VISIBLE_DEVICES={run.devices['cuda_visible_devices']!r}, "
        f"TensorFlow sees {run.devices['tf_visible_devices']} ({run.devices['gpu_names']})"
    )

    # DECISION plan-2026-10-30T132432-a865fa08/D-041: the data is DESCRIBED
    # (in ``run``) and LOADED after ``resolve_auto``. Do NOT load first and
    # resolve from the loaded data (the plan's order): the augmentation hook
    # and the recipe's ``loader`` arguments are arguments of the loader, and
    # both are decided by the recipe, which needs the class count and image size.
    logger.info(f"Resolved args:\n{to_json(config)}")
    with _bad_value_is_a_config_error(run.run_dir, "the model arguments"):
        used = trainer.used_arguments(config, run.context)
    if used:
        logger.info(f"Used args:\n{json.dumps(used, indent=2)}")
    if resume is None:
        with _bad_value_is_a_config_error(run.run_dir, "the config stamps"):
            stamps = trainer.config_stamps(config, registry.MODELS[run.family])
        write_resolved(run.run_dir, config, used, stamps)
    for level, text in trainer._notices:
        (logger.warning if level == "warning" else logger.info)(text)


def load_data(run: Run) -> None:
    """Load the data, refuse what its length shows is wrong, and check it against a resumed run's."""
    trainer, config, run_dir = run.trainer, run.config, run.run_dir
    checkpoint_cfg: CheckpointArgs = config.checkpoint_args
    with _bad_value_is_a_config_error(run_dir, "the data loader"):
        run.data = trainer.build_data(config, run.context)
    trainer.input_check(run.facts, run.data)
    run.steps_per_epoch = run.data.steps_per_epoch
    steps_per_epoch = run.steps_per_epoch
    every = checkpoint_cfg.backup_every_steps
    if every is not None and every >= steps_per_epoch:
        # The key needs the length of an epoch, known only now. Refuse, do not run
        # with a key that saves nothing inside an epoch (it would look like it worked).
        # The key's smallest value is 2, so the values that work are 2 .. steps - 1.
        if steps_per_epoch > 3:
            choice = f"Choose a value from 2 to {steps_per_epoch - 1} or null"
        elif steps_per_epoch == 3:
            choice = "Choose 2 or null"
        else:
            choice = (
                f"An epoch of {steps_per_epoch} steps has no value from 2 up that is below "
                "it, so choose null"
            )
        raise ConfigError(
            f"Invalid value for 'checkpoint_args.backup_every_steps': {every} is not below "
            f"the {steps_per_epoch} steps of an epoch of this data, so no batch inside an "
            f"epoch would be backed up (each epoch end always is). {choice}. "
            f"{_claimed_note(run_dir)}"
        )
    if run.resume is not None:
        # Before anything of the earlier run is touched (the CSV is copied aside
        # in `epoch_zero`): the data is the killed run's data, and the backup is a pair that
        # belongs together (D-069).
        resume_decisions.check_data_unchanged(
            run_dir, config, trainer.task_record(config, run.context, run.data), run.data,
            run.resume,
        )
        if run.reporting_only:
            resume_decisions.check_final_weights(run_dir, run.resume, steps_per_epoch)
        else:
            run.resume = resume_decisions.settle_backup_pair(
                run_dir, config, trainer.config, run.resume, steps_per_epoch
            )


def build_model(run: Run) -> None:
    """Build the model, load the optional ``checkpoint`` weights, build the optimizer and ``compile``."""
    from dl_techniques.utils.logger import logger

    trainer, config, data = run.trainer, run.config, run.data
    optim_cfg: OptimArgs = config.optim_args
    with _bad_value_is_a_config_error(run.run_dir, "the model factory"):
        run.model = trainer.build_model(config, run.context, data)
        if not run.model.built:
            run.model.build((None, *data.input_shape))
    if config.checkpoint is not None:
        trainer._load_checkpoint(run.model, config.checkpoint)
    optimizer, _schedule = build_optimizer_and_schedule(
        optim_cfg, run.steps_per_epoch, config.epochs
    )
    # DECISION plan-2026-10-30T132432-a865fa08/D-052: do NOT wrap the optimizer in
    # ``LossScaleOptimizer`` here. ``compile`` does it under ``mixed_float16``
    # (keras/src/trainers/trainer.py wraps a plain optimizer when the policy asks
    # for loss scaling), and a second wrap is a second place to get it wrong.
    loss, metrics = trainer.build_loss_and_metrics(config, data)
    # DECISION plan-2026-11-10T095307-d483ffab/D-012
    # The task's compile options are spread into THIS call and nowhere else; the default is ``{}``, so
    # a task that says nothing makes the three-argument call it always made (a spy in
    # ``test_the_sequence_seams_default_to_nothing`` proves it). Do NOT override ``compile`` in a model
    # class instead (a plain functional model has no class, and a ``compile`` override also needs
    # ``compile_from_config``), and do NOT pass ``jit_compile=None`` for "no option": Keras reads
    # None as a value.
    with _bad_value_is_a_config_error(run.run_dir, "the compile options"):
        options = dict(trainer.compile_options(config, registry.MODELS[run.family]))
    reserved = sorted(set(options) & {"optimizer", "loss", "metrics"})
    if reserved:
        raise ConfigError(
            f"compile_options of {type(trainer).__name__} names {reserved}, which the run passes "
            f"itself. {_claimed_note(run.run_dir)}"
        )
    run.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **options)
    run.params = int(run.model.count_params())
    logger.info(
        f"  {run.family}/{run.variant}: params {run.params:,}, {run.steps_per_epoch} steps/epoch, "
        f"batch {config.batch_size}, {config.epochs} epochs"
    )


def epoch_zero(run: Run) -> None:
    """Get the epoch-0 numbers (evaluated, or the sidecar's), check the monitor against them, build the summary head."""
    from dl_techniques.utils.logger import logger
    from train.common.resume_state import copy_log_aside, update_fit_state

    trainer, config, data, resume, run_dir = run.trainer, run.config, run.data, run.resume, run.run_dir
    metric_cfg: MetricArgs = config.metric_args
    monitor = metric_cfg.watch_metric

    # DECISION plan-2026-09-19T040641-db6932ec/D-014: the epoch-0 numbers (the
    # summary's initial-loss ratio AND the dashboard baseline) come from THIS one
    # evaluation. Do NOT let the dashboard call ``model.evaluate`` itself.
    run.best_weights = None
    if resume is None:
        with _a_corrupt_file_is_named(
            run_dir, trainer.files_to_scan(data), "the first evaluation"
        ):
            run.baseline, run.initial_ratio, run.init_warning = trainer.initial_loss_check(
                config, run.model, data
            )
        # The sidecar keeps the epoch-0 numbers: a resumed model is not an untrained
        # one, so a resume reads them here and evaluates nothing (D-058).
        update_fit_state(
            run.sidecar_path,
            initial={
                "metrics": run.baseline,
                "ratio": float(run.initial_ratio),
                "warned": None if run.init_warning is None else bool(run.init_warning),
            },
            data_counts=resume_decisions.data_counts(data),
            **resume_decisions.data_fingerprint_field(config, data),
        )
    else:
        initial = resume.state["initial"]
        run.baseline = initial["metrics"]
        run.initial_ratio, run.init_warning = initial["ratio"], initial["warned"]
        run.best_weights = resume_decisions.best_weights_of(run_dir, resume)
        if run.reporting_only:
            logger.info(
                f"Resume: {TRAINING_LOG_NAME} is read and not written (epochs "
                f"0..{resume.start - 1})"
            )
        else:
            kept = copy_log_aside(run_dir / TRAINING_LOG_NAME)
            logger.info(
                "Resume: " + (f"epochs 0..{resume.start - 1} come back from {TRAINING_LOG_NAME}"
                               if resume.start else "no epoch had finished, so no row comes back")
                + f"; the file as it was is kept as {kept.name}"
            )
    baseline = run.baseline
    if monitor not in _monitor_choices([k for k in baseline if k != "loss"]):
        # The pre-claim check compares the monitor with the names the task
        # NAMED; this one with the names ``compile`` produced, so a task whose
        # list and metrics disagree is stopped before ``fit`` and not after it.
        raise ConfigError(
            f"Invalid value for 'metric_args.watch_metric': {monitor!r} is not a "
            f"metric the compiled model reports ({', '.join(sorted(baseline))}). "
            f"{_claimed_note(run_dir)}"
        )
    trainer._check_success_metric(
        _monitor_choices([k for k in baseline if k != "loss"]),
        "the compiled model reports", _claimed_note(run_dir),
    )
    run.summary_head = trainer._summary_head(
        config, run.family, run.variant, run_dir, data, run.params, baseline,
        run.initial_ratio, run.init_warning, run.devices,
    )


def train_or_restore(run: Run) -> None:
    """Run ``fit`` (or put the model where it left off, from the files) and summarize it.

    The one function that calls ``fit`` and the one that writes the history of a diverged run.
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-070
    # `fit`, `summarize_fit` and the diverged branch (summary, history, raise) are ONE stage.
    # Do NOT move the `non_finite` branch into `report`, or `fit` into its own stage: the
    # history-write census (test_a_history_write_follows_the_model_save) accepts a
    # `save_training_history_json` only in the function that calls `fit(callbacks=...)`, and a
    # history written from a stage that never fitted would also be a write the census cannot
    # place after a model save. The split that cut `_train` at its other seams kept this one whole.
    from dl_techniques.utils.logger import logger
    from train.common import classification_run, run_summary
    from train.common.gpu import log_gpu_peak_memory
    from train.common.resume_state import merge_history, update_fit_state
    from train.common.run_io import save_training_history_json

    trainer, config, data, model, resume, run_dir = (
        run.trainer, run.config, run.data, run.model, run.resume, run.run_dir
    )
    optim_cfg: OptimArgs = config.optim_args
    metric_cfg: MetricArgs = config.metric_args
    checkpoint_cfg: CheckpointArgs = config.checkpoint_args
    analysis_cfg: AnalysisArgs = config.analysis_args
    monitor = metric_cfg.watch_metric
    epochs = config.epochs

    run.vis_dir = run_dir / "visualizations"
    if analysis_cfg.figures:
        run.vis_dir.mkdir(parents=True, exist_ok=True)
    run.last_weights = classification_run.LastEpochWeights()
    if run.reporting_only:
        # No callbacks (they would open `training_log.csv` for writing), no `fit`, no
        # `task.json` (it was written before the training and is left as it is).
        run.history, run.last_weights = resume_decisions.restore_finished_fit(
            model, run_dir, resume, run.best_weights
        )
        run.epoch_times = [float("nan")] * len(resume.rows)
        run.fit_wall_seconds = float("nan")
        update_fit_state(
            run.sidecar_path,
            reporting_resumes=int(resume.state.get("reporting_resumes", 0)) + 1,
        )
    else:
        # The callbacks, `ModelCheckpoint` among them, are `fit_callbacks.build_callbacks`'.
        epoch_line, callbacks = trainer._callbacks(
            config, data, run_dir, run.vis_dir, run.baseline, run.experiment_name,
            run.last_weights, resume, run.best_weights,
        )

        # DECISION plan-2026-10-30T132432-a865fa08/D-061: task.json is written BEFORE `fit`, and a resume
        # leaves an existing one as it is. Do NOT move it after `fit` or into the
        # post-fit reporting: a run killed in training would leave models that cannot
        # be used, and do NOT rewrite it on a resume: the earlier epochs trained on
        # the statistics it holds, and a rewrite would state others.
        record = trainer.task_record(config, run.context, data)
        if record is not None:
            if not write_task(run_dir, record):
                logger.info("task.json is already in the run directory: left as it is")

        started = time.perf_counter()
        with _a_failure_is_logged("Training"), _a_corrupt_file_is_named(
            run_dir, trainer.files_to_scan(data), "fit"
        ):
            run.history = model.fit(
                data.train_ds,
                validation_data=data.select_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
            )
        run.fit_wall_seconds = time.perf_counter() - started
        update_fit_state(run.sidecar_path, fit_finished=True)
        if resume is not None:
            merge_history(resume.rows, run.history)
        run.epoch_times = epoch_line.epoch_times
        log_gpu_peak_memory()
    run.hist, run.epochs_run, non_finite, run.lr_reductions = classification_run.summarize_fit(
        run.history, monitor, constant_lr=optim_cfg.schedule == "constant"
    )
    if non_finite:
        message = trainer._write_diverged_summary(
            run_dir, run.summary_head, run.hist, run.epochs_run, non_finite,
            run.lr_reductions, run.epoch_times, run.fit_wall_seconds, run.initial_ratio,
        )
        # The history is written HERE, in the function that calls ``fit`` with its
        # checkpoint callbacks: the history-write census accepts a write that follows
        # a ``fit(callbacks=...)`` in the same function and nothing else.
        save_training_history_json(run.history, str(run_dir))
        raise RuntimeError(message)
    if run.epochs_run < epochs:
        logger.info(
            f"EarlyStopping: stopped after epoch {run.epochs_run} of {epochs} "
            f"(patience {checkpoint_cfg.early_stopping_patience} on {monitor}); "
            "the best weights were restored"
        )
    # DECISION plan-2026-10-30T132432-a865fa08/D-048: the epoch reported is
    # chosen with the SAME direction the checkpoint callbacks were given
    # (``watch_mode``, resolved). Do NOT call ``best_epoch(hist, monitor)``
    # alone: it re-derives the direction from the name, so an explicit
    # ``watch_mode`` saved one epoch and the summary named another.
    # DECISION plan-2026-11-10T095307-d483ffab/D-013
    # The epoch reported is chosen from the task's selection window on, the one the checkpoint and
    # ``EarlyStopping`` were given: with a window, the best epoch of the whole run was never saved, so
    # naming it would say a file holds weights it does not. Do NOT call ``best_epoch`` over every epoch
    # when ``selection_start_epoch`` is above 0 (``first_epoch`` is 0, the whole run, by default).
    run.best_epoch = run_summary.best_epoch(
        run.hist, monitor, metric_cfg.watch_mode,
        first_epoch=trainer.selection_start_epoch(config),
    )


def report(run: Run) -> TrainResult:
    """Run the task's reporting, write the summary, judge the run and return its record."""
    from dl_techniques.utils.logger import logger
    from train.common import classification_run, run_summary
    from train.common.gpu import log_gpu_peak_memory
    from train.common.run_artifacts import write_summary_json

    trainer, config, data, resume, run_dir = run.trainer, run.config, run.data, run.resume, run.run_dir
    metric_cfg: MetricArgs = config.metric_args
    monitor = metric_cfg.watch_metric
    hist = run.hist

    analysis_file = run_dir / "model_analysis" / "analysis_results.json"
    analysis_before = analysis_file.stat().st_mtime_ns if analysis_file.exists() else None
    with _a_failure_is_logged(
        "Reporting",
        lambda: resume_decisions.resume_hint(
            run_dir, config, trainer.config, run.epochs_run, run.steps_per_epoch
        ),
    ):
        fit = FitState(
            model=run.model,
            history=run.history,
            last_weights=run.last_weights,
            epochs_run=run.epochs_run,
            run_dir=run_dir,
            vis_dir=run.vis_dir,
            experiment_name=run.experiment_name,
        )
        evaluated = trainer.evaluate_and_report(config, data, fit)

        reached = trainer.success_value(hist)
        smoke = config.debug_args.smoke
        # Under smoke one epoch on at most 512 samples cannot reach twice chance, so
        # the bar is not applied: the run is a success when it finished, and the
        # summary says "not applicable" rather than a miss (review-iter-2 concern 6).
        threshold = None if smoke else trainer.success_threshold(config)
        met = True if smoke else trainer.bar_is_met(config, reached, threshold)
        # DECISION plan-2026-11-10T095307-d483ffab/D-004
        # A check that failed after training blocks the verdict whatever the bar says, and the run is
        # kept: ``success`` false, ``success_blocked_by`` names the checks, the exit code is 1 (not 2).
        # Do NOT raise for it (the models and the summary would not be written) and do NOT fold it into
        # ``bar_is_met`` (the bar would then read "missed" for a run that reached it).
        blocked = tuple(evaluated.failed_checks)
        success = False if blocked else met
        summary = classification_run.base_summary(
            run.summary_head, hist, run.epochs_run, config.epochs, run.best_epoch,
            run.lr_reductions, evaluated.test_metrics_best, evaluated.test_metrics_final,
            evaluated.best_load_error, evaluated.checkpoint_max_diff,
            run.epoch_times, run.fit_wall_seconds, evaluated.visualizations,
            evaluated.load_check, evaluated.conformal, evaluated.analysis,
            notes=trainer._notes(
                config, run.baseline, run.initial_ratio, run.init_warning, data, resume
            ),
            load_detail=evaluated.load_detail,
        )
        if run.reporting_only:
            stale = resume_decisions.older_analysis_note(
                analysis_file, analysis_before, evaluated.analysis
            )
            if stale is not None:
                summary["notes"].append(stale)
        if resume is not None:
            summary["resumed_after_epochs"] = resume.start
        if run.reporting_only:
            summary["reporting_resumed"] = True
        summary.update(
            success=None if smoke and not blocked else success,
            success_metric=trainer.success_metric,
            success_threshold=threshold,
            success_value=reached,
        )
        if blocked:
            summary["success_blocked_by"] = list(blocked)
        # DECISION plan-2026-11-10T095307-d483ffab/D-004
        # The task's post-fit keys come through the VALUE ``evaluate_and_report`` returns, merged last
        # and never over a key the summary holds. Do NOT stash them on the trainer during
        # ``evaluate_and_report`` and read them here (a hidden coupling), and do NOT let an extra
        # replace a key (``success`` or ``task`` would then lie about the run).
        taken = sorted(set(evaluated.extras) & set(summary))
        if taken:
            raise ConfigError(
                f"Evaluation.extras of {type(trainer).__name__} names {taken}, which the summary "
                "already holds; give each task-specific key a name of its own"
            )
        summary.update(evaluated.extras)
        write_summary_json(run_dir, summary)
        run_summary.fail_on_best_load_error(run_dir, evaluated.best_load_error)
    if run.reporting_only:
        log_gpu_peak_memory()
    if smoke:
        logger.info(
            f"success_threshold not applicable under debug_args.smoke: best "
            f"{trainer.success_metric} {reached:.4f}"
        )
    else:
        logger.info(
            f"{'Reached' if met else 'Missed'} success_threshold {threshold:g} "
            f"on {trainer.success_metric}: best {reached:.4f}"
        )
    if blocked:
        logger.warning(f"The run is not a success: failed checks {list(blocked)}")
    return TrainResult(
        run_dir=run_dir,
        success=success,
        best_metric=float(hist[monitor][run.best_epoch - 1]),
        test_metrics_best=evaluated.test_metrics_best,
        test_metrics_final=evaluated.test_metrics_final,
        summary_path=run_dir / "results_summary.json",
    )
