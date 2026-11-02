"""The one sequence every training task runs, and the record it returns.

``TaskTrainer.run()`` is the sequence of the legacy ConvNeXt trainer
(``train.convnext.common.train``) cut at its seams: the shared helpers live in
``train.common.classification_run`` and are called in the legacy order, so a run
started through the API and one started by the legacy script do the same things
in the same order (D-014). A task subclasses :class:`TaskTrainer` and supplies
the hooks; it never re-writes the sequence.

Interface contract
------------------
``TrainResult``
    Frozen record returned by :meth:`TaskTrainer.run`.

    :ivar run_dir: the run directory.
    :ivar success: the task's ``success_value`` (the best ``success_metric``) reached
        its ``success_threshold``; ``True`` for a ``debug_args.smoke``
        run that finished, where no threshold applies (the summary's ``success``
        is then ``null``).
    :ivar best_metric: the best value of ``metric_args.watch_metric``
        (lowest or highest as the metric's direction says).
    :ivar test_metrics_best: the reloaded ``best_model.keras`` on the test split;
        ``None`` when ``checkpoint_args.save_best`` is false.
    :ivar test_metrics_final: the last epoch's weights on the test split.
    :ivar summary_path: ``<run_dir>/results_summary.json``.

``TaskTrainer(config)``
    :param config: a validated task config (a frozen dataclass). It must carry
        every field named in :data:`CONFIG_FIELDS`; ``TypeError`` otherwise.

``TaskTrainer.run()``
    :return: a :class:`TrainResult`.
    :raises ConfigError: for a request the run cannot honour, found before the
        run directory is claimed (a ``resume_interrupted`` whose run is finished,
        missing, or was started with another request; ``smoke`` with an
        explicit ``epochs``, a field out of range, an optimizer configuration
        that cannot be built, a ``watch_metric`` the task does not report, a
        ``devices`` index outside the visible GPUs) or after (an unusable
        ``checkpoint``, a value the data loader or the model factory refuses,
        a monitor the compiled model does not report); a refusal after the
        claim says that ``out`` is now used and to choose a new one.
    :raises RuntimeError: for a non-finite initial loss, a diverged fit (the
        diverged summary and the history are written first), or a best
        checkpoint that did not load (the summary is written first).

    The sequence, with no ``except`` anywhere in it:

    1. ``parse_model_name``, ``describe_data`` -> the family recipe ->
       ``resolve_auto`` on every group -> ``assert_resolved`` -> the checks that
       need no data (``problems`` of the config, ``check_optim_args``, the
       monitor against ``metric_names``, ``devices``); a refusal here leaves no
       trace: the directory is not claimed yet (D-042, D-047); then
       ``resolve_out`` and ``open_run`` (claims the directory); ``take_run_lock`` (``run.lock``: created
       with ``O_EXCL`` by a fresh run; a ``flock`` held until the run ends; a resume refuses a
       live holder and takes over a dead one's);
    2. ``setup_gpu``, ``set_seeds(config.seed)``, the precision policy (restored
       in a ``finally``); the lock is released when the run ends, an exception included;
    3. ``write_requested``; ``run.log`` opens; ``Args:`` is logged;
       ``Resolved args:`` is logged and ``write_resolved`` writes ``config.json``;
    4. (the data is described in step 1 and loaded in step 5)
    5. ``build_data`` (a resume then checks the data against ``task.json`` and settles the
       backup, D-069), ``build_model``, the optional ``checkpoint`` weights,
       ``compile`` through ``optim.py``;
    6. ``initial_loss_check``;
    7. callbacks: ``LearningRateLogger(at_epoch_start=True)`` first, then
       ``create_callbacks``, ``LastEpochWeights``, ``extra_callbacks``,
       ``EpochLogLine``, the task's ``figure_callbacks`` and the dashboard (both only
       while ``analysis_args.figures`` is true), and the resume trio last
       (``RestoreFitState`` on a resume only, ``FitStateSidecar``,
       ``BackupAndRestore``; the order is ``train.common.resume_state``'s);
    8. ``task.json`` (the task's :meth:`~TaskTrainer.task_record`, when it gives one;
       left alone on a resume), then ``fit`` on ``data.select_ds``; the sidecar records
       ``fit_finished``;
       ``summarize_fit``; a non-finite loss
       writes the diverged summary and the history, then raises;
    9. ``evaluate_and_report``;
    10. ``base_summary`` -> ``write_summary_json`` -> ``fail_on_best_load_error``;
    11. the :class:`TrainResult`.

Resume (``resume_interrupted=true``)
------------------------------------
The same function and the same ``out`` continue a run that was killed. Every
run writes ``checkpoints/backup/`` (``BackupAndRestore``, kept) and
``checkpoints/fit_state.json`` (``EarlyStopping``'s best and wait per epoch, the
epoch-0 evaluation, and ``fit_finished`` once ``fit`` returned). A resume:

* refuses, before anything is written, a run with no backup (killed before its first
  epoch ended, and before the first ``checkpoint_args.backup_every_steps`` copy when that
  is set: nothing to resume, choose a new ``out``), a run that is finished
  (``results_summary.json``), a run a live process holds
  (``run.lock``), a request that differs from ``config_requested.json``
  (``resume_interrupted``, ``out``, ``devices`` and ``analysis_args.eval_batch_size`` may
  differ), a run with ``checkpoint_args.save_best`` false, a ``best_model.keras`` cut short
  by a kill, and a run with no epoch left that did NOT finish its ``fit`` (all trained, or
  ``EarlyStopping`` had already decided to stop, and the process died before ``fit``
  returned);
* after the data is loaded and before the CSV is touched, refuses data that is not the run's
  (class names, geometry, mean and std against ``task.json``; split sizes against the
  sidecar) and settles the backup (``resume_state.settle_backup``: the latest pair when the
  optimizer's step count equals epoch times steps (plus the batches taken, for a copy
  written inside an epoch), else the previous pair of ``double_checkpoint``, else a
  refusal naming both files);
* appends to ``run.log``, copies ``training_log.csv`` to
  ``training_log.pre_resume_<k>.csv``, and rewrites it with epochs ``0 .. start-1``
  from the copy, so a row of an epoch the backup never saw is re-trained, once;
* gives ``ModelCheckpoint`` the best value, ``EarlyStopping`` its best, wait and
  best weights (read from ``best_model.keras``), and takes the epoch-0 baseline
  from the sidecar instead of evaluating the untrained model;
* merges the earlier epochs into the ``History`` after ``fit``, so every later
  step (summary, history file, figures) sees all epochs;
* writes neither ``config_requested.json`` nor ``config.json`` again.

Reporting-only resume (D-041, supersedes D-021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A run whose sidecar says ``fit_finished`` and which has no ``results_summary.json`` died while
reporting (the ImageNet gate's nano run: 256 images of 224 px asked 18 GB). Its training is
whole, so a resume of it does not train: ``fit`` is skipped, no callback is built, and
``training_log.csv`` is read and never written (no ``pre_resume`` copy either). Everything after
``fit`` runs as in a fresh run, on weights taken from the files: the BEST weights from
``best_model.keras`` (what EarlyStopping's restore left in the model), the LAST epoch's from
``checkpoints/backup/latest.weights.h5`` (what ``LastEpochWeights`` held in memory, lost with
the process), the history from the CSV rows. The last epoch's weights are used only when the
latest backup copy is the epoch-end copy of the epochs that ran, with the optimizer step count
that implies (``resume_state.final_copy_problem``); a missing, torn, mid-epoch or older copy is
a refusal (exit 2) that names ``latest.weights.h5``, and the previous ``.bkp`` copy is never
substituted: a final model that was one epoch old would look like the final model. An existing
``final_model.keras`` is written over (atomically), never read. The data checks are those of any
resume. The summary says ``reporting_resumed: true``, ``resumed_after_epochs`` is the epoch count,
``epoch_times`` and ``fit_wall_seconds`` are null, the sidecar counts ``reporting_resumes``, and
the GPU peak of the pass is logged (it is the reporting's own). A reporting exception is
logged as ``ERROR`` in ``run.log`` (type and message) before it propagates (exit code 1).

Residuals, stated in ``resume_interrupted``'s help (a test pins the text): the training
data order replays the order of epoch 0 (the pipeline is seeded, not resumed), an epoch
killed midway is trained again from its start (with ``checkpoint_args.backup_every_steps`` it
is continued from the last copy taken inside it, ``resume_state.StepBackupAndRestore``, D-063: only
the batches that copy had not taken run, from a fresh pass of the data, so some examples of that
epoch repeat and some are missed, and its row holds the train metrics of those batches only),
a kill between the best checkpoint's save and
the backup can leave ``best_model.keras`` one epoch LATER than the resumed state's best (the
epoch that is then trained again; never an earlier one, D-021, D-069), and the summary's
``epoch_times`` and ``fit_wall_seconds`` cover the resumed epochs only. ``ReduceLROnPlateau``'s
best, wait and cooldown come back through the sidecar and the learning rate through the
optimizer (D-069); the best checkpoint is written atomically.

Hooks (override in a task)
--------------------------
``describe_data(config) -> DataFacts``
    Dataset name, class count and image size WITHOUT loading anything. They fill
    the ``AutoContext``, which has to exist before the data is loaded: the
    augmentation hook, a recipe decision, is an argument of the loader. This
    deviates from the plan's order "build_data, then resolve_auto" (D-041).
    The resolution runs before the run directory is claimed (D-042), so a
    request that cannot resolve (an explicit ``random_erasing_prob`` under a
    preset that takes none, an unknown augmentation preset) is refused without
    touching ``out``.
``build_data(config, context)``
    The loaded data, an object with the attributes of
    ``train.common.classification_data.ClassificationData``. ``config`` is
    RESOLVED, ``context.defaults`` is the recipe (read ``loader`` from it).
    ``num_classes`` and ``input_shape[0]`` must equal what ``describe_data``
    said, else ``RuntimeError``. File-based data also carries ``image_files``
    (the files ``fit`` and the first evaluation read, so an undecodable one can be
    named in a ``ConfigError``) and ``fingerprint`` (recorded in the sidecar and
    compared at resume, D-070, with the ``data.fingerprint`` mode beside it); data without them is not scanned and carries none.
``input_check(facts, data)``
    Raise ``RuntimeError`` when the loaded data is not what ``describe_data`` said (the
    class count, the image side). Required.
``build_model(config, context, data)``
    The model; built or not (the trainer builds it from ``data.input_shape``).
``metric_names(config, facts)``
    The names of the metrics ``compile`` will report, without ``loss``. A
    ``watch_metric`` that is not one of them (or ``loss``, or the ``val_`` twin
    of either) is refused before the run is claimed.
``build_loss_and_metrics(config, data)``
    ``(loss, metrics)`` for ``compile``. The names must be the ones ``metric_names`` gave.
``initial_loss_check(config, model, data)``
    ``(baseline, ratio, warned)``: the epoch-0 evaluation of the compiled, untrained model
    (the dashboard's baseline and the summary's initial-loss ratio come from this one
    evaluation); ``warned`` is the summary's ``init_scale_warning``, a bool, or ``None`` (not
    judged) when the evaluation was not a measurement of the initialization (BatchNorm
    statistics still unset, D-045); raises ``RuntimeError`` for a non-finite loss. Required.
``success_value(hist)`` / ``success_threshold(config)``
    The number the run is judged by (read from ``hist``, the history of ``summarize_fit``, under
    the ``success_metric`` class variable) and the bar it must reach. Required, and a task must
    set ``success_metric`` (it is written to the summary). Smoke runs apply no bar.
``extra_callbacks(config, data)``
    Callbacks inserted after ``LastEpochWeights``. Default: none.
``figure_callbacks(config, data, vis_dir)``
    The task's per-epoch figure callbacks (a prediction grid drawn every few epochs, say). Called
    only while ``analysis_args.figures`` is true, with ``vis_dir`` already created; the callbacks
    go after ``EpochLogLine`` and before the dashboard, so they must not edit ``logs``. The end
    of run figures belong to ``evaluate_and_report``, which reads the same switch through
    ``FitState.vis_dir``. The training dashboard is not part of this hook: it is the base
    class's, it reads whichever of ``loss``, ``accuracy``, ``top_5_accuracy``, ``miou`` and
    ``lr`` the logs hold (only ``loss`` or ``val_loss`` is needed), and a task that returns
    nothing here still gets it. Default: none.
``epoch_line_metrics(config)``
    The history keys the one-line-per-epoch log prints. Default: ``("loss", "val_loss")``.
``evaluate_and_report(config, data, fit)``
    The reporting sequence after ``fit``: best-checkpoint test, figures and analysis of the best
    weights, the last epoch's weights. Returns an :class:`Evaluation`. Required.
``run_analysis(config, model, test_data, history, experiment_name, run_dir)``
    The model analyzer call, made by the task's ``evaluate_and_report`` through
    ``classification_run.report_best_weights(run_analysis=..., name_failed_analyses=True)``.
    Returns what ``run_model_analysis`` returns. The default picks the analyzer configuration
    by the model's output rank: a classifier (rank 2) gets the stock one, a per-pixel model
    the pixel path (D-020) and a 256 MB budget for information flow. A task with void labels
    overrides it to pass ``ignore_label``. A failed analysis is named in the ``analyzer``
    block of the summary as ``failed_analyses`` (a ``{name: reason}`` object, present only
    when non-empty), and ``analyzer.status`` then reads ``partial`` instead of ``success``.
``task_metadata(config, data)``
    Extra keys for the summary head. Default: none.
``summary_fields(config, data)``
    Keys the summary head carries because of what this task's data holds (they sit between
    ``n_test`` and ``epochs_requested``). Default: none.
``task_notes(config, baseline, ratio, data)``
    The plain-language lines of ``results_summary.json``'s ``notes`` that belong to the task, in
    order; the trainer appends the resume and smoke lines after them. Default: none.
``files_to_scan(data)``
    The files ``fit`` and the first evaluation read, so a decode failure can name the culprit
    (``None`` or empty: nothing to scan, and the original error propagates). Default: ``None``.
``recipe_notices(facts, family)``
    ``(level, text)`` lines about how the family recipe meets this data, logged once ``run.log``
    is open. Default: none. Called before ``setup_gpu``: no tensor.
``task_record(config, context, data)``
    The ``task.json`` record of a run: what the saved models need to be used without
    this config (class names, preprocessing, the statistics the data was standardized
    with). Default: ``None``, which writes no ``task.json``. It is written once, before
    ``fit``, so a killed run has it and a resume leaves it as it is.
``check_model_args(config, context)``
    Optional. Raises ``ConfigError`` for a model argument the run cannot honour, called with the
    resolved config before the run directory is claimed. Default: nothing is checked.
``used_arguments(config, context)``
    A JSON-plain mapping written to ``config.json`` under the key ``used``: what
    the run is built with beyond the config's fields (D-050). Default: empty, which
    writes no key.

The knobs of D-022 act here: ``save_best=False`` removes ``ModelCheckpoint`` and
skips the best-checkpoint test; ``save_last=False`` passes ``save=False`` to
``finalize_last_epoch``; ``tensorboard=None`` leaves TensorBoard out;
``epoch_analysis_every=N`` turns the epoch analyzer on with frequency and start
N; ``figures=False`` writes no figure and no dashboard; ``terminate_on_nan``
adds ``TerminateOnNaN``; ``smoke`` runs 1 epoch on at most 512 samples and does not apply the success
threshold (``TrainResult.success`` is ``True``, the summary's ``success`` is
``null``).

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); every
``keras``, ``tensorflow`` and ``train.common`` name is imported inside a method.
"""

import collections.abc
import contextlib
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Sequence, Tuple

from . import registry
from .args import (
    AnalysisArgs,
    AutoContext,
    CheckpointArgs,
    DebugArgs,
    LoggerArgs,
    MetricArgs,
    OptimArgs,
)
from .config import AUTO, StrictConfig, assert_resolved, join_path, to_json
from .data_args import DataFacts
from .errors import ConfigError, ConfigValidationError
from .optim import build_optimizer_and_schedule, check_optim_args
from .run_dir import (
    BACKUP_DIR,
    CHECKPOINTS_DIR,
    TASK_FILE,
    TRAINING_LOG_NAME,
    check_model_file,
    load_resume_point,
    open_run,
    resolve_out,
    take_run_lock,
    write_requested,
    write_resolved,
    write_task,
)

# ---------------------------------------------------------------------

#: The fields of a task config the trainer reads. A subclass of ``TaskTrainer``
#: is built from a config that has all of them.
CONFIG_FIELDS = (
    "out",
    "model",
    "data",
    "epochs",
    "batch_size",
    "seed",
    "devices",
    "precision",
    "resume_interrupted",
    "checkpoint",
    "model_args",
    "optim_args",
    "transform_args",
    "metric_args",
    "checkpoint_args",
    "logger_args",
    "analysis_args",
    "debug_args",
)

#: The hooks a task must override: the base class has no body for them that would be right
#: for every task. ``TaskTrainer(config)`` of a subclass lacking one raises ``TypeError``.
REQUIRED_HOOKS = (
    "describe_data",
    "build_data",
    "input_check",
    "build_model",
    "metric_names",
    "build_loss_and_metrics",
    "initial_loss_check",
    "success_value",
    "success_threshold",
    "evaluate_and_report",
)

#: ``debug_args.smoke`` runs this many epochs on at most this many samples.
SMOKE_EPOCHS = 1
SMOKE_MAX_SAMPLES = 512

#: The largest seed ``numpy`` takes (``np.random.seed`` refuses anything outside
#: ``[0, 2**32 - 1]``), so the largest ``set_seeds`` takes.
MAX_SEED = 2**32 - 1

# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrainResult:
    """What a finished run reports; see the module contract."""

    run_dir: Path
    success: bool
    best_metric: float
    test_metrics_best: Optional[Dict[str, float]]
    test_metrics_final: Optional[Dict[str, float]]
    summary_path: Path


@dataclasses.dataclass(frozen=True)
class FitState:
    """What ``evaluate_and_report`` needs from the finished ``fit``."""

    model: Any
    history: Any
    last_weights: Any
    epochs_run: int
    run_dir: Path
    vis_dir: Path
    experiment_name: str


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """What ``evaluate_and_report`` returns, in the legacy order of the steps."""

    test_metrics_best: Optional[Dict[str, float]]
    best_load_error: Optional[str]
    checkpoint_max_diff: Optional[float]
    conformal: Optional[Dict[str, Any]]
    visualizations: Dict[str, Any]
    analysis: Dict[str, Any]
    test_metrics_final: Dict[str, float]
    load_check: Optional[bool]


def _data_counts(data: Any) -> Dict[str, int]:
    """The sizes of the three splits a run trained and was selected and tested on (D-069)."""
    return {
        "n_train": int(data.n_train),
        "n_select": int(data.n_select),
        "n_test": int(data.n_test),
    }


def _data_fingerprint_field(config: Any, data: Any) -> Dict[str, str]:
    """The sidecar's fingerprint keys of a file-based data record, ``{}`` for one with none (D-070).

    ``data_fingerprint`` is the digest and ``data_fingerprint_mode`` the ``data.fingerprint``
    value that made it (``listing`` or ``content``), so a resume can say which of the two
    changed. ``data_fingerprint_scheme`` is how the rows were ordered and hashed
    (``classification_data.FINGERPRINT_SCHEME``): a sidecar without it, or with another one,
    was written by a fingerprint that could not tell a reordered CSV, and is refused.
    """
    fingerprint = getattr(data, "fingerprint", None)
    if fingerprint is None:
        return {}
    from train.common.classification_data import FINGERPRINT_SCHEME

    return {
        "data_fingerprint": fingerprint,
        "data_fingerprint_mode": getattr(config.data, "fingerprint", None),
        "data_fingerprint_scheme": FINGERPRINT_SCHEME,
    }


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value == AUTO


def _non_json_values(value: Any, path: str, found: List[Tuple[str, str]]) -> None:
    """Append ``(path, type name)`` for every part of ``value`` JSON cannot hold."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _non_json_values(item, f"{path}[{index}]", found)
    elif isinstance(value, collections.abc.Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                found.append((path, f"a {type(key).__name__} key"))
            _non_json_values(item, join_path(path, str(key)), found)
    else:
        found.append((path, f"a {type(value).__name__}"))


def config_problems(config: StrictConfig, prefix: str = "") -> List[str]:
    """Return one line per top-level field of a task config that cannot be right.

    A task config calls this from its ``problems`` so :func:`validate` reports
    the lines together with every other bad key, before anything runs.

    :param config: a task config carrying the fields of :data:`CONFIG_FIELDS`;
        an ``"auto"`` field is skipped (it has no value yet).
    :param prefix: the dotted path of ``config``; ``""`` at the top.
    :return: lines for ``epochs`` or ``batch_size`` below 1, a ``seed`` outside
        ``[0, MAX_SEED]``, a negative ``devices``, and a ``model_args`` value
        that JSON cannot hold (``config.json`` records the arguments, and a Keras
        object would fail that write after the run directory is claimed).
    """
    lines: List[str] = []

    def bad(name: str, expected: str, value: Any) -> None:
        lines.append(
            f"Invalid value for {join_path(prefix, name)!r}: expected {expected}, got {value!r}"
        )

    for name in ("epochs", "batch_size"):
        value = getattr(config, name)
        if not _is_auto(value) and value < 1:
            bad(name, "an int >= 1", value)
    if not 0 <= config.seed <= MAX_SEED:
        bad("seed", f"an int in [0, {MAX_SEED}]", config.seed)
    devices = config.devices
    if devices is not None and not _is_auto(devices) and devices < 0:
        bad("devices", "null, auto or an int >= 0", devices)
    offenders: List[Tuple[str, str]] = []
    _non_json_values(config.model_args, join_path(prefix, "model_args"), offenders)
    for path, kind in offenders:
        lines.append(
            f"Invalid value for {path!r}: expected a JSON value (null, bool, number, "
            f"string, list or mapping) because config.json records it, got {kind}; "
            "an L2 regularizer for sgd comes from optim_args.weight_decay"
        )
    return lines


def _monitor_choices(names: List[str]) -> List[str]:
    """Return the keys a callback can monitor when ``compile`` reports ``names``.

    :param names: the metric names without ``loss``; ``loss`` is added.
    :return: each name and its ``val_`` twin, the loss included.
    """
    reported = ["loss", *names]
    return [*reported, *(f"val_{name}" for name in reported)]


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
    # ``_train`` or ``_resolve``, which hold no ``except`` (their census), and
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
        :meth:`TaskTrainer.files_to_scan`); ``None`` or empty, as for the built-in datasets,
        has nothing to scan.
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


def _eval_batch_size_of(config: Any) -> Optional[int]:
    """Return the config's resolved ``analysis_args.eval_batch_size``, or ``None`` when it has none.

    ``None`` for a config without the group and for an unresolved ``"auto"`` (a config built by
    hand that skipped ``resolve_auto``): no request, so the analyzer keeps its own batch.
    """
    size = getattr(getattr(config, "analysis_args", None), "eval_batch_size", None)
    return size if isinstance(size, int) and not isinstance(size, bool) else None


@contextlib.contextmanager
def _a_reporting_failure_is_logged(run_dir: Path, epochs_run: int) -> Iterator[None]:
    """Write an exception raised inside the block to ``run.log`` as an ERROR, then let it go.

    Interface contract: wraps the reporting of ``_train`` (``evaluate_and_report``, the summary
    and its write), inside ``attach_run_log``. It has no ``except`` clause: the exception is
    read from ``sys.exc_info()`` in a ``finally`` and keeps propagating, so it cannot be
    swallowed, narrowed or replaced, the exit code stays the interpreter's 1 for an uncaught
    exception and stderr still gets the traceback. ``SystemExit`` and ``KeyboardInterrupt`` are
    not logged as failures.

    :param run_dir: the run directory; a summary in it means a resume is refused, so the hint
        to resume is given only while there is none.
    :param epochs_run: the epochs that trained, named in the hint.
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-041
    # `finally` + `sys.exc_info()`, NOT `except Exception: ...; raise`. The API source rule
    # `no_broad_handler` forbids any broad handler under src/train/api (a re-raising one
    # included, test_the_api_source_rules.py), and `_train` holds no `except` (its census).
    # Do NOT turn this into a handler: the log line is a side effect of an exception already
    # on its way, and a `finally` cannot change where it goes. Before this the OOM of the
    # gate's nano run was in stderr only and run.log ended on its last INFO line.
    try:
        yield
    finally:
        failure = sys.exc_info()[1]
        if isinstance(failure, Exception):
            from dl_techniques.utils.logger import logger
            from train.common.run_artifacts import RESULTS_SUMMARY_NAME

            again = (
                ""
                if (run_dir / RESULTS_SUMMARY_NAME).exists()
                else f" The training is kept ({epochs_run} epochs, best_model.keras, "
                "checkpoints/backup): run the same request with resume_interrupted=true to "
                "run only the reporting again"
            )
            what = f"Reporting failed: {type(failure).__name__}: {failure}".rstrip()
            logger.error(what + ("" if what.endswith(".") else ".") + again)


class TaskTrainer:
    """The training sequence of a task; see the module contract.

    The base class holds the sequence and no task: a hook in :data:`REQUIRED_HOOKS` raises
    until a subclass supplies it, and the others have the neutral default their docstring
    names.

    :cvar task: the task's command word; a subclass sets it.
    :cvar success_metric: the history key ``success_value`` reads and the summary names; a
        subclass sets it.
    """

    task: ClassVar[str] = ""
    success_metric: ClassVar[str] = ""

    def __init__(self, config: StrictConfig) -> None:
        missing = [name for name in CONFIG_FIELDS if not hasattr(config, name)]
        if missing:
            raise TypeError(
                f"{type(config).__name__} lacks the field(s) {', '.join(missing)} "
                "that TaskTrainer reads"
            )
        # DECISION plan-2026-10-31T180033-2faf890a/D-021: a task that forgets a required hook
        # is refused HERE, when it is built, naming the hook. Do NOT give these hooks a
        # classification body in the base class "so old tasks keep working": a second task
        # would then run the classification sequence on its own data and fail late, or not
        # at all, and the one success_metric default would score it by the wrong number.
        forgotten = [
            name
            for name in REQUIRED_HOOKS
            if getattr(type(self), name) is getattr(TaskTrainer, name)
        ]
        # DECISION plan-2026-10-31T180033-2faf890a/D-029
        # A hook must be CALLABLE, a task and a success metric non-empty texts. Do NOT stop at
        # "differs from the base": `input_check = None` passed that, and raised `'NoneType'
        # object is not callable` after the run directory was claimed, and `task = ""` ran to a
        # summary (review-iter-1 concern 12).
        uncallable = [
            name for name in REQUIRED_HOOKS
            if name not in forgotten and not callable(getattr(type(self), name))
        ]
        for name in ("task", "success_metric"):
            value = getattr(type(self), name)
            if not isinstance(value, str) or not value:
                forgotten.append(f"{name} (class variable)")
        if forgotten or uncallable:
            parts = []
            if forgotten:
                parts.append(
                    f"does not supply: {', '.join(forgotten)}; TaskTrainer has no default for them"
                )
            if uncallable:
                parts.append(f"has hooks that are not callable: {', '.join(uncallable)}")
            raise TypeError(f"{type(self).__name__} {'; and '.join(parts)}")
        self.config = config
        #: ``(level, text)`` lines found while the request is resolved, logged
        #: once ``run.log`` is open.
        self._notices: List[Tuple[str, str]] = []

    # -----------------------------------------------------------------
    # Hooks
    # -----------------------------------------------------------------

    def describe_data(self, config: StrictConfig) -> DataFacts:
        raise NotImplementedError

    def build_data(self, config: StrictConfig, context: AutoContext) -> Any:
        raise NotImplementedError

    def build_model(self, config: StrictConfig, context: AutoContext, data: Any) -> Any:
        raise NotImplementedError

    def metric_names(self, config: StrictConfig, facts: DataFacts) -> List[str]:
        """Return the names of the metrics ``compile`` will report, without ``loss``."""
        raise NotImplementedError

    def build_loss_and_metrics(self, config: StrictConfig, data: Any) -> Tuple[Any, List[Any]]:
        raise NotImplementedError

    def extra_callbacks(self, config: StrictConfig, data: Any) -> List[Any]:
        return []

    def figure_callbacks(self, config: StrictConfig, data: Any, vis_dir: Path) -> List[Any]:
        """Return the task's per-epoch figure callbacks; see the module contract.

        :param vis_dir: the existing ``visualizations`` directory of the run; the only place
            these callbacks write under.
        :return: callbacks placed after ``EpochLogLine`` and before the dashboard. Called only
            while ``analysis_args.figures`` is true. Default: none.
        """
        return []

    def task_metadata(self, config: StrictConfig, data: Any) -> Dict[str, Any]:
        return {}

    def task_record(
        self, config: StrictConfig, context: AutoContext, data: Any
    ) -> Optional[Dict[str, Any]]:
        """Return the ``task.json`` record, or ``None`` for a task that writes none."""
        return None

    def check_model_args(self, config: StrictConfig, context: AutoContext) -> None:
        """Raise :class:`ConfigError` for a model argument the run cannot honour, before the claim.

        Called with the RESOLVED config (every ``"auto"`` filled) and the context, after the
        other pre-claim checks and before ``open_run``, so a request that would fail while the
        model is built is refused with no trace. Default: nothing is checked.
        """

    def used_arguments(self, config: StrictConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run is built with beyond the config, for ``config.json``.

        The value is a JSON-plain mapping written under the key ``used``; empty
        writes no key. Default: empty.
        """
        return {}

    def run_analysis(
        self,
        config: StrictConfig,
        model: Any,
        test_data: Tuple[Any, Any],
        history: Any,
        experiment_name: str,
        run_dir: str,
    ) -> Any:
        """Run the model analyzer; called by the task's ``evaluate_and_report``.

        :param test_data: ``(x, y)``: the images and their labels (class indices for a
            classifier, a mask of class indices for a per-pixel model).
        :param run_dir: the run directory; the analyzer writes ``model_analysis/`` under it.
        :return: what ``run_model_analysis`` returns, the ``AnalysisResults`` or ``None`` when
            the analysis raised; ``run_summary.failed_analyses`` reads it.

        The default asks ``analysis_config_for`` for the configuration: the stock one for a
        model whose output has rank 2, and for a per-pixel model a ``memory_limit_mb`` of 256
        for information flow with the pixel path of the calibration analyzer. Both carry the
        run's ``seed`` as the analyzer's ``random_state`` (step 14.3, D-027), so the bootstrap
        ``pl_pvalue`` of the spectral analysis is reproducible between runs of one seed. A
        subclassed model, whose output rank is only known from a call, is given one image to
        be probed with. A task whose labels mark void pixels overrides this and passes
        ``ignore_label``; without it a void label is not dropped and the calibration analyzer
        raises on it, which ``failed_analyses`` then names.

        The analyzer's own predictions run at Keras' 32 images a pass unless
        ``analysis_args.eval_batch_size`` is lower (``evaluation.predict_batch_cap``), in which
        case they run at that, on the rank-2 path.
        """
        from train.common import evaluation

        return evaluation.run_model_analysis(
            model,
            test_data,
            history,
            experiment_name,
            run_dir,
            config=evaluation.analysis_config_for(
                model,
                sample=evaluation.first_sample(test_data),
                random_state=getattr(config, "seed", None),
                prediction_batch_size=evaluation.predict_batch_cap(
                    _eval_batch_size_of(config)
                ),
            ),
        )

    def evaluate_and_report(
        self, config: StrictConfig, data: Any, fit: FitState
    ) -> Evaluation:
        """Run the task's reporting after ``fit`` and return its :class:`Evaluation`; required."""
        raise NotImplementedError

    def input_check(self, facts: DataFacts, data: Any) -> None:
        """Raise ``RuntimeError`` when ``data`` is not what ``describe_data`` said; required."""
        raise NotImplementedError

    def initial_loss_check(
        self, config: StrictConfig, model: Any, data: Any
    ) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return ``(baseline, ratio, warned)`` of the untrained, compiled ``model``; required.

        ``baseline`` maps each metric name (``loss`` included) to its epoch-0 value on the
        selection split, ``ratio`` is the number the summary reports as
        ``initial_loss_ratio`` and ``warned`` the summary's ``init_scale_warning``: a bool,
        or ``None`` when the epoch-0 pass was not a measurement of the initialization (a
        model whose BatchNorm statistics are unset) and no verdict is given. Raise
        ``RuntimeError`` for a non-finite loss.
        """
        raise NotImplementedError

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        """Return the number the run is judged by, from ``hist`` (``summarize_fit``); required."""
        raise NotImplementedError

    def success_threshold(self, config: StrictConfig) -> float:
        """Return the bar ``success_value`` must reach (a smoke run applies none); required."""
        raise NotImplementedError

    def epoch_line_metrics(self, config: StrictConfig) -> Tuple[str, ...]:
        """Return the history keys the per-epoch log line prints; default ``loss``, ``val_loss``."""
        return ("loss", "val_loss")

    def summary_fields(self, config: StrictConfig, data: Any) -> Dict[str, Any]:
        """Return the summary-head keys this task's data supplies; default: none."""
        return {}

    def task_notes(
        self, config: StrictConfig, baseline: Dict[str, float], ratio: float, data: Any
    ) -> List[str]:
        """Return the task's lines of the summary's ``notes``, in order; default: none."""
        return []

    def files_to_scan(self, data: Any) -> Optional[Sequence[str]]:
        """Return the files the pipelines read, to name a corrupt one; default ``None``."""
        return None

    def recipe_notices(self, facts: DataFacts, family: str) -> List[Tuple[str, str]]:
        """Return ``(level, text)`` lines on the recipe meeting this data; default: none."""
        return []

    # -----------------------------------------------------------------
    # The sequence
    # -----------------------------------------------------------------

    def run(self) -> TrainResult:
        """Run the training sequence; see the module contract."""
        import keras

        from train.common.gpu import setup_gpu, setup_mixed_precision
        from train.common.seed import set_seeds

        config = self.config
        debug_cfg: DebugArgs = config.debug_args
        if debug_cfg.smoke and not _is_auto(config.epochs):
            raise ConfigError(
                "Invalid value for 'epochs': debug_args.smoke runs "
                f"{SMOKE_EPOCHS} epoch and cannot be combined with an explicit "
                f"epochs, got {config.epochs!r}; remove one of them"
            )

        family, variant = registry.parse_model_name(config.model)
        # DECISION plan-2026-10-30T132432-a865fa08/D-042: the request is RESOLVED
        # before the run directory is claimed. Do NOT move this below ``open_run``:
        # a request that cannot resolve is a configuration error, and refusing it
        # after the claim would leave ``config_requested.json`` behind and block
        # the same ``out`` for a mistake the user has not yet been told about.
        facts = self.describe_data(config)
        resolved, context = self._resolve(facts, family, variant)
        run_dir = open_run(
            resolve_out(config.out, self.task, family, variant), config.resume_interrupted
        )
        # Reading the earlier run's files is plain file I/O: no TensorFlow op, so it may
        # come before `setup_gpu` (D-055). Loading its best weights may not, and waits.
        resume = self._resume_point(run_dir, resolved) if config.resume_interrupted else None
        # The run lock (D-011 of this plan): a kernel flock held for the life of the run, so a
        # fresh run and a resume exclude each other, and the descriptor is released when the
        # `with` ends (also on an exception) or the process dies. After every refusal that
        # needs no tensor, before the first thing is written.
        with take_run_lock(run_dir, resume=resume is not None):
            setup_gpu(None if _is_auto(config.devices) else config.devices, flag="devices")
            set_seeds(config.seed)
            previous_policy = keras.mixed_precision.global_policy().name
            setup_mixed_precision(config.precision != "float32", config.precision)
            try:
                return self._train(family, variant, run_dir, facts, resolved, context, resume)
            finally:
                keras.mixed_precision.set_global_policy(previous_policy)

    def _resolve(
        self, facts: DataFacts, family: str, variant: str
    ) -> Tuple[StrictConfig, AutoContext]:
        """Return the config with every ``"auto"`` filled, and the context that did it."""
        config = self.config
        recipe = registry.MODELS[family].recipe(facts.dataset, variant)
        context = AutoContext(
            family=family,
            variant=variant,
            dataset=facts.dataset,
            num_classes=facts.num_classes,
            image_size=facts.image_size,
            defaults=recipe,
        )
        changes: Dict[str, Any] = {}
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            if isinstance(value, StrictConfig):
                changes[field.name] = value.resolve_auto(context)
        for name in ("epochs", "batch_size"):
            if _is_auto(getattr(config, name)) and name in recipe:
                changes[name] = recipe[name]
        if _is_auto(config.devices):
            changes["devices"] = None
        debug_cfg: DebugArgs = config.debug_args
        if debug_cfg.smoke:
            changes["epochs"] = SMOKE_EPOCHS
            data_cfg = changes["data"]
            if data_cfg.max_samples is None:
                changes["data"] = dataclasses.replace(data_cfg, max_samples=SMOKE_MAX_SAMPLES)
        resolved = dataclasses.replace(config, **changes)
        assert_resolved(resolved)
        self._notices = self._recipe_notices(config, resolved, facts, family, recipe)
        self._refuse_before_the_claim(resolved, facts)
        self.check_model_args(resolved, context)
        return resolved, context

    def _recipe_notices(
        self,
        config: StrictConfig,
        resolved: StrictConfig,
        facts: DataFacts,
        family: str,
        recipe: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        """Return the ``(level, text)`` lines the resolution owes the user.

        A recipe value an ``"auto"`` field did not take is listed (the recipe
        offered a number and the resolver chose another), the task's
        :meth:`recipe_notices` follow (a family with no overlay for the dataset is
        named, for a classifier), and a warmup that never ends is warned.
        """
        notices: List[Tuple[str, str]] = []
        for group_field in dataclasses.fields(config):
            before = getattr(config, group_field.name)
            after = getattr(resolved, group_field.name)
            table = recipe.get(group_field.name)
            if not isinstance(before, StrictConfig) or not isinstance(table, dict):
                continue
            for field in dataclasses.fields(before):
                if (
                    _is_auto(getattr(before, field.name))
                    and field.name in table
                    and table[field.name] != getattr(after, field.name)
                ):
                    notices.append(
                        (
                            "info",
                            f"Recipe value not used: {group_field.name}.{field.name} "
                            f"{table[field.name]!r} -> {getattr(after, field.name)!r} "
                            f"({family} recipe)",
                        )
                    )
        notices.extend(self.recipe_notices(facts, family))
        optim_cfg: OptimArgs = resolved.optim_args
        debug_cfg: DebugArgs = resolved.debug_args
        if (
            not debug_cfg.smoke
            and optim_cfg.schedule != "constant"
            and optim_cfg.decay_over != "post_warmup"
            and optim_cfg.warmup_epochs >= resolved.epochs
        ):
            notices.append(
                (
                    "warning",
                    f"optim_args.warmup_epochs {optim_cfg.warmup_epochs} is not below epochs "
                    f"{resolved.epochs}: the learning rate never leaves its warmup and does "
                    f"not reach lr {optim_cfg.lr!r}",
                )
            )
        return notices

    def _refuse_before_the_claim(self, resolved: StrictConfig, facts: DataFacts) -> None:
        """Refuse every request that needs no loaded data to be seen as wrong.

        :raises ConfigValidationError: for a field out of range (the config's
            own ``problems``) or an optimizer configuration that cannot be built.
        :raises ConfigError: for a ``watch_metric`` the run does not report, or
            a ``devices`` index outside the visible GPUs.
        """
        # DECISION plan-2026-10-30T132432-a865fa08/D-047: everything checkable
        # without data is refused HERE, before ``open_run``. Do NOT leave a range
        # or a combination to the optimizer builder, the loader or ``fit``: each
        # fails after the claim, so the user's next try with the value fixed is
        # refused for a directory the first try only half used, and a 200-epoch
        # run can be lost to a typo in the monitor (review-iter-2 concerns 1, 5).
        # The config and each of its groups, again: ``validate`` ran them at the
        # function's entry, but a config built by hand (or a task whose groups
        # have no ``resolve_auto``) never passed through it.
        lines = resolved.problems("")
        for field in dataclasses.fields(resolved):
            value = getattr(resolved, field.name)
            if isinstance(value, StrictConfig):
                lines.extend(value.problems(field.name))
        if lines:
            raise ConfigValidationError(lines)
        check_optim_args(resolved.optim_args, resolved.epochs)
        monitor = resolved.metric_args.watch_metric
        choices = _monitor_choices(self.metric_names(resolved, facts))
        if monitor not in choices:
            raise ConfigError(
                f"Invalid value for 'metric_args.watch_metric': {monitor!r} is not a metric "
                f"this run reports; choose one of {', '.join(sorted(choices))}"
            )
        self._check_success_metric(choices, "this run reports", "")
        self._check_devices(resolved.devices)

    def _check_success_metric(self, choices: Sequence[str], where: str, claimed: str) -> None:
        """Raise :class:`ConfigError` unless ``success_metric`` is one of ``choices``.

        Interface contract: ``choices`` are the history keys the run can report (the metric names
        and their ``val_`` twins, :func:`_monitor_choices`); ``where`` names their source and
        ``claimed`` is appended to the message (the "out is now used" note after the claim).
        Without it a misspelt name cost the whole run: ``success_value`` raised ``KeyError``
        after ``fit`` and all the reporting, with no ``results_summary.json`` and ``fit_finished``
        true, so the run could not be resumed (review-iter-1 concern 11).
        """
        # DECISION plan-2026-10-31T180033-2faf890a/D-029
        # Checked twice, like `watch_metric`: BEFORE the claim against the names the task
        # NAMED, and before `fit` against the names `compile` produced. Do NOT leave it to
        # `success_value`: that runs after the training it would have spared.
        if self.success_metric not in choices:
            raise ConfigError(
                f"Invalid {type(self).__name__}.success_metric: {self.success_metric!r} is not a "
                f"metric {where}; choose one of {', '.join(sorted(choices))}. {claimed}".rstrip()
            )

    @staticmethod
    def _check_devices(devices: Optional[int]) -> None:
        """Refuse a ``devices`` index that ``setup_gpu`` would refuse, before the claim."""
        if devices is None:
            return
        from train.common.gpu import VISIBLE_DEVICES, visible_device_for

        try:
            visible_device_for(devices, os.environ.get(VISIBLE_DEVICES))
        except ValueError as exc:
            raise ConfigError(f"Invalid value for 'devices': {exc}") from exc

    def _resume_point(self, run_dir: Path, resolved: StrictConfig) -> Any:
        """Read where the killed run stopped, or refuse to resume it.

        :param run_dir: the directory :func:`open_run` accepted for a resume.
        :param resolved: this call's config, every ``"auto"`` filled.
        :return: the :class:`~train.common.resume_state.ResumePoint`.
        :raises ConfigError: see the module contract, "Resume".
        """
        return self._check_resume_point(run_dir, resolved, load_resume_point(run_dir, self.config))

    @staticmethod
    def _check_resume_point(run_dir: Path, resolved: StrictConfig, point: Any) -> Any:
        """Apply the refusals of :meth:`_resume_point` to ``point`` and return it.

        Called again when the backup settles on its previous copy (one epoch earlier).
        """
        from train.common.callbacks import best_checkpoint_path

        checkpoint_cfg: CheckpointArgs = resolved.checkpoint_args
        if not checkpoint_cfg.save_best:
            raise ConfigError(
                f"Cannot resume {run_dir}: checkpoint_args.save_best is false, so the best "
                "weights of the killed segment were never written and a resumed run could "
                "not return to them"
            )
        if point.state.get("fit_finished"):
            return TaskTrainer._check_finished_fit(run_dir, point)
        # DECISION plan-2026-10-30T132432-a865fa08/D-058
        # A fit with nothing left to train is REFUSED, not run. Do NOT let it reach
        # `fit`: with `start >= epochs` no epoch ends, `LastEpochWeights` holds nothing
        # and the post-fit steps fail on `None`; with EarlyStopping's patience already
        # reached (it decides in `on_epoch_end`, which the kill outran) one more epoch
        # that improves would carry on past the stop an uninterrupted run made.
        if point.start >= resolved.epochs:
            raise ConfigError(
                f"Cannot resume {run_dir}: all {point.start} of its {resolved.epochs} epochs "
                "were trained and the run died after the last one; a run that crashed while "
                "reporting is not finished by a resume (D-021). Choose a new 'out'"
            )
        if point.start > 1 and point.wait >= checkpoint_cfg.early_stopping_patience:
            raise ConfigError(
                f"Cannot resume {run_dir}: after epoch {point.start} EarlyStopping had waited "
                f"{point.wait} epochs (patience {checkpoint_cfg.early_stopping_patience}) and "
                "had already decided to stop; the run died before 'fit' returned. Choose a "
                "new 'out'"
            )
        if point.best is not None and not Path(best_checkpoint_path(str(run_dir))).is_file():
            raise ConfigError(
                f"Cannot resume {run_dir}: the sidecar records a best value but "
                f"{best_checkpoint_path(str(run_dir))} does not exist"
            )
        if point.best is not None:
            # Plain file I/O, so before the GPU: a best checkpoint cut short by a kill is
            # named here, not met as a Keras traceback after the model is built (D-069).
            check_model_file(Path(best_checkpoint_path(str(run_dir))), f"Cannot resume {run_dir}")
        if "initial" not in point.state:
            raise ConfigError(
                f"Cannot resume {run_dir}: the sidecar holds no epoch-0 evaluation"
            )
        return point

    @staticmethod
    def _check_finished_fit(run_dir: Path, point: Any) -> Any:
        """Apply the refusals of a REPORTING-ONLY resume to ``point`` and return it.

        A run whose sidecar says ``fit_finished`` and which has no ``results_summary.json``
        crashed (or was killed) while reporting. Its training is whole, so nothing of
        :meth:`_check_resume_point`'s "no epoch left" rules applies: there is none left by
        design, and EarlyStopping having decided to stop is how most runs end. What the
        reporting needs is refused here, before anything is written: the best checkpoint (the
        model after ``fit`` held the best weights), the epoch-0 evaluation, and a sidecar,
        a backup and a CSV that agree on how many epochs ran. Plain file reading.

        :raises ConfigError: naming the file and the fact, and that the training is kept.
        """
        from train.common.callbacks import best_checkpoint_path

        recorded = sorted(int(key) for key in point.state.get("epochs", {}))
        epochs_run = len(recorded)
        if epochs_run == 0 or recorded != list(range(epochs_run)):
            raise ConfigError(
                f"Cannot resume {run_dir}: fit_state.json is marked fit_finished but holds the "
                f"epochs {recorded}, not 0..n-1, so the number of epochs that ran is unknown. "
                "Choose a new 'out'"
            )
        if point.start != epochs_run:
            raise ConfigError(
                f"Cannot resume {run_dir}: fit_state.json says {epochs_run} epochs ran but the "
                f"backup's latest metadata says {point.start}; the files do not agree on where "
                "the training ended, so the last epoch's weights are not known. Choose a new "
                "'out'"
            )
        best_path = Path(best_checkpoint_path(str(run_dir)))
        if point.best is None or not best_path.is_file():
            raise ConfigError(
                f"Cannot resume {run_dir}: the reporting needs the best weights and "
                f"{best_path} does not exist (no epoch had a finite best). Choose a new 'out'"
            )
        check_model_file(best_path, f"Cannot resume {run_dir}")
        if "initial" not in point.state:
            raise ConfigError(
                f"Cannot resume {run_dir}: the sidecar holds no epoch-0 evaluation"
            )
        return point

    @staticmethod
    def _check_final_weights(run_dir: Path, resume: Any, steps_per_epoch: int) -> None:
        """Refuse a reporting-only resume whose last epoch's weights cannot be shown to be right.

        The LAST epoch's weights are not in ``best_model.keras``; the only copy is the epoch-end
        backup ``checkpoints/backup/latest.weights.h5`` (``LastEpochWeights`` was in memory and
        died with the process). It is used only when it is the pair the run's own end wrote:
        metadata ``epoch`` equal to the epochs that ran and ``batch`` 0 (an epoch-end copy), and
        an optimizer step count equal to ``epochs x steps_per_epoch``
        (``resume_state.final_copy_problem``). The previous copy (``.bkp``) is NEVER used here: it is
        one epoch or some batches behind, and ``settle_backup`` (which would put it in place)
        is not called, so nothing in the backup is changed.

        :raises ConfigError: naming ``latest.weights.h5`` and what is wrong with it, and that
            ``best_model.keras`` is untouched.
        """
        from train.common.resume_state import final_copy_problem

        backup = run_dir / BACKUP_DIR
        problem = final_copy_problem(backup, resume.start, steps_per_epoch)
        if problem is not None:
            raise ConfigError(
                f"Cannot resume {run_dir}: its training finished but the last epoch's weights "
                f"cannot be recovered ({problem}). They exist only in "
                f"{backup / 'latest.weights.h5'}, and the reporting evaluates them as the "
                "final model, so it will not run on any other weights. best_model.keras is "
                "untouched and can be loaded by hand. Choose a new 'out' to train again"
            )

    @staticmethod
    def _restore_finished_fit(
        model: Any, run_dir: Path, resume: Any, best_weights: List[Any]
    ) -> Tuple[Any, Any]:
        """Put the model where ``fit`` left it, from the files, and return ``(history, last)``.

        After ``fit`` the in-memory model holds the BEST weights (EarlyStopping restores them)
        and ``LastEpochWeights`` holds the last epoch's. Here the best weights are
        ``best_model.keras``'s (``best_weights``, read by :meth:`_best_weights_of`) and the last
        epoch's are ``checkpoints/backup/latest.weights.h5`` (verified by
        :meth:`_check_final_weights`). The history is ``training_log.csv``'s rows.

        :return: ``(History, LastEpochWeights)``; the model holds the best weights.
        """
        import warnings

        import keras

        from train.common import classification_run
        from train.common.resume_state import BACKUP_WEIGHTS_NAME, rows_as_history

        with warnings.catch_warnings():
            # `load_weights` reads the optimizer's variables too and skips them with a warning
            # when the optimizer is not built; only the model's weights are wanted here.
            warnings.filterwarnings(
                "ignore", message="Skipping variable loading for optimizer", category=UserWarning
            )
            model.load_weights(str(run_dir / BACKUP_DIR / BACKUP_WEIGHTS_NAME))
        last = classification_run.LastEpochWeights()
        last.weights = model.get_weights()
        model.set_weights(best_weights)
        history = keras.callbacks.History()
        history.history = rows_as_history(resume.rows)
        history.epoch = [epoch for epoch, _ in resume.rows]
        return history, last

    def _train(
        self,
        family: str,
        variant: str,
        run_dir: Path,
        facts: DataFacts,
        config: StrictConfig,
        context: AutoContext,
        resume: Any = None,
    ) -> TrainResult:
        from dl_techniques.utils.logger import logger
        from train.common import classification_run, run_summary
        from train.common.gpu import log_gpu_peak_memory
        from train.common.resume_state import (
            FIT_STATE_NAME,
            copy_log_aside,
            merge_history,
            update_fit_state,
        )
        from train.common.run_artifacts import attach_run_log, write_summary_json
        from train.common.run_io import save_training_history_json

        if resume is None:
            write_requested(run_dir, self.config)
        sidecar_path = run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME
        with attach_run_log(run_dir, mode="w" if resume is None else "a"):
            logger.info(f"Run directory: {run_dir}")
            reporting_only = resume is not None and bool(resume.state.get("fit_finished"))
            if reporting_only:
                # DECISION plan-2026-10-31T180033-2faf890a/D-041
                # A run whose `fit` returned and whose reporting died is NOT retrained: `fit`
                # is skipped, the weights come from the files, and only the reporting runs
                # again. Do NOT send it through `fit` with nothing left to train (no epoch
                # ends, so nothing is reported) and do NOT refuse it (the days of training
                # were the cost of a reporting bug, the gate's nano run). training_log.csv is
                # not copied or rewritten: it must be byte-for-byte what the training wrote.
                logger.info(
                    f"Resuming a run whose fit had finished ({resume.start} epochs of "
                    f"{config.epochs}) and whose reporting did not: fit is NOT run again. The "
                    "best weights come from best_model.keras, the last epoch's from "
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
            logger.info(f"Args:\n{to_json(self.config)}")
            devices = run_summary.describe_devices()
            logger.info(
                f"Devices: CUDA_VISIBLE_DEVICES={devices['cuda_visible_devices']!r}, "
                f"TensorFlow sees {devices['tf_visible_devices']} ({devices['gpu_names']})"
            )

            # DECISION plan-2026-10-30T132432-a865fa08/D-041: the data is DESCRIBED
            # (in ``run``) and LOADED after ``resolve_auto``. Do NOT load first and
            # resolve from the loaded data (the plan's order): the augmentation hook
            # and the recipe's ``loader`` arguments are arguments of the loader, and
            # both are decided by the recipe, which needs the class count and image size.
            logger.info(f"Resolved args:\n{to_json(config)}")
            with _bad_value_is_a_config_error(run_dir, "the model arguments"):
                used = self.used_arguments(config, context)
            if used:
                logger.info(f"Used args:\n{json.dumps(used, indent=2)}")
            if resume is None:
                write_resolved(run_dir, config, used)
            for level, text in self._notices:
                (logger.warning if level == "warning" else logger.info)(text)

            optim_cfg: OptimArgs = config.optim_args
            metric_cfg: MetricArgs = config.metric_args
            checkpoint_cfg: CheckpointArgs = config.checkpoint_args
            analysis_cfg: AnalysisArgs = config.analysis_args
            debug_cfg: DebugArgs = config.debug_args
            epochs = config.epochs
            monitor = metric_cfg.watch_metric
            experiment_name = run_dir.name

            with _bad_value_is_a_config_error(run_dir, "the data loader"):
                data = self.build_data(config, context)
            self.input_check(facts, data)
            steps_per_epoch = data.steps_per_epoch
            every = checkpoint_cfg.backup_every_steps
            if every is not None and every >= steps_per_epoch:
                # The key needs the length of an epoch, known only now. Refuse, do not run
                # with a key that saves nothing inside an epoch (it would look like it worked).
                raise ConfigError(
                    f"Invalid value for 'checkpoint_args.backup_every_steps': {every} is not below "
                    f"the {steps_per_epoch} steps of an epoch of this data, so no batch inside an "
                    f"epoch would be backed up (each epoch end always is). Choose a value from 2 to "
                    f"{max(steps_per_epoch - 1, 2)} or null. {_claimed_note(run_dir)}"
                )
            if resume is not None:
                # Before anything of the earlier run is touched (the CSV is copied aside
                # below): the data is the killed run's data, and the backup is a pair that
                # belongs together (D-069).
                self._check_data_unchanged(run_dir, config, context, data, resume)
                if reporting_only:
                    self._check_final_weights(run_dir, resume, steps_per_epoch)
                else:
                    resume = self._settle_backup(run_dir, config, resume, steps_per_epoch)

            with _bad_value_is_a_config_error(run_dir, "the model factory"):
                model = self.build_model(config, context, data)
                if not model.built:
                    model.build((None, *data.input_shape))
            if config.checkpoint is not None:
                self._load_checkpoint(model, config.checkpoint)
            optimizer, _schedule = build_optimizer_and_schedule(
                optim_cfg, steps_per_epoch, epochs
            )
            # DECISION plan-2026-10-30T132432-a865fa08/D-052: do NOT wrap the optimizer in
            # ``LossScaleOptimizer`` here. ``compile`` does it under ``mixed_float16``
            # (keras/src/trainers/trainer.py wraps a plain optimizer when the policy asks
            # for loss scaling), and a second wrap is a second place to get it wrong.
            loss, metrics = self.build_loss_and_metrics(config, data)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            params = int(model.count_params())
            logger.info(
                f"  {family}/{variant}: params {params:,}, {steps_per_epoch} steps/epoch, "
                f"batch {config.batch_size}, {epochs} epochs"
            )

            # DECISION plan-2026-09-19T040641-db6932ec/D-014: the epoch-0 numbers (the
            # summary's initial-loss ratio AND the dashboard baseline) come from THIS one
            # evaluation. Do NOT let the dashboard call ``model.evaluate`` itself.
            best_weights = None
            if resume is None:
                with _a_corrupt_file_is_named(
                    run_dir, self.files_to_scan(data), "the first evaluation"
                ):
                    baseline, initial_ratio, init_warning = self.initial_loss_check(
                        config, model, data
                    )
                # The sidecar keeps the epoch-0 numbers: a resumed model is not an untrained
                # one, so a resume reads them here and evaluates nothing (D-058).
                update_fit_state(
                    sidecar_path,
                    initial={
                        "metrics": baseline,
                        "ratio": float(initial_ratio),
                        "warned": None if init_warning is None else bool(init_warning),
                    },
                    data_counts=_data_counts(data),
                    **_data_fingerprint_field(config, data),
                )
            else:
                initial = resume.state["initial"]
                baseline = initial["metrics"]
                initial_ratio, init_warning = initial["ratio"], initial["warned"]
                best_weights = self._best_weights_of(run_dir, resume)
                if reporting_only:
                    logger.info(
                        f"Resume: {TRAINING_LOG_NAME} is read and not written (epochs "
                        f"0..{resume.start - 1})"
                    )
                else:
                    kept = copy_log_aside(run_dir / TRAINING_LOG_NAME)
                    logger.info(
                        f"Resume: " + (f"epochs 0..{resume.start - 1} come back from {TRAINING_LOG_NAME}"
                                       if resume.start else "no epoch had finished, so no row comes back")
                        + f"; the file as it was is kept as {kept.name}"
                    )
            if monitor not in _monitor_choices([k for k in baseline if k != "loss"]):
                # The pre-claim check compares the monitor with the names the task
                # NAMED; this one with the names ``compile`` produced, so a task whose
                # list and metrics disagree is stopped before ``fit`` and not after it.
                raise ConfigError(
                    f"Invalid value for 'metric_args.watch_metric': {monitor!r} is not a "
                    f"metric the compiled model reports ({', '.join(sorted(baseline))}). "
                    f"{_claimed_note(run_dir)}"
                )
            self._check_success_metric(
                _monitor_choices([k for k in baseline if k != "loss"]),
                "the compiled model reports", _claimed_note(run_dir),
            )
            summary_head = self._summary_head(
                config, family, variant, run_dir, data, params, baseline,
                initial_ratio, init_warning, devices,
            )

            vis_dir = run_dir / "visualizations"
            if analysis_cfg.figures:
                vis_dir.mkdir(parents=True, exist_ok=True)
            last_weights = classification_run.LastEpochWeights()
            if reporting_only:
                # No callbacks (they would open `training_log.csv` for writing), no `fit`, no
                # `task.json` (it was written before the training and is left as it is).
                history, last_weights = self._restore_finished_fit(
                    model, run_dir, resume, best_weights
                )
                epoch_times = [float("nan")] * len(resume.rows)
                fit_wall_seconds = float("nan")
                update_fit_state(
                    sidecar_path,
                    reporting_resumes=int(resume.state.get("reporting_resumes", 0)) + 1,
                )
            else:
                epoch_line, callbacks = self._callbacks(
                    config, data, run_dir, vis_dir, baseline, experiment_name, last_weights,
                    resume, best_weights,
                )

                # DECISION plan-2026-10-30T132432-a865fa08/D-061: task.json is written BEFORE `fit`, and a resume
                # leaves an existing one as it is. Do NOT move it after `fit` or into the
                # post-fit reporting: a run killed in training would leave models that cannot
                # be used, and do NOT rewrite it on a resume: the earlier epochs trained on
                # the statistics it holds, and a rewrite would state others.
                record = self.task_record(config, context, data)
                if record is not None:
                    if not write_task(run_dir, record):
                        logger.info("task.json is already in the run directory: left as it is")

                started = time.perf_counter()
                with _a_corrupt_file_is_named(run_dir, self.files_to_scan(data), "fit"):
                    history = model.fit(
                        data.train_ds,
                        validation_data=data.select_ds,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=1,
                    )
                fit_wall_seconds = time.perf_counter() - started
                update_fit_state(sidecar_path, fit_finished=True)
                if resume is not None:
                    merge_history(resume.rows, history)
                epoch_times = epoch_line.epoch_times
                log_gpu_peak_memory()
            hist, epochs_run, non_finite, lr_reductions = classification_run.summarize_fit(
                history, monitor, constant_lr=optim_cfg.schedule == "constant"
            )
            if non_finite:
                message = self._write_diverged_summary(
                    run_dir, summary_head, hist, epochs_run, non_finite,
                    lr_reductions, epoch_times, fit_wall_seconds, initial_ratio,
                )
                # The history is written HERE, in the function that calls ``fit`` with its
                # checkpoint callbacks: the history-write census accepts a write that follows
                # a ``fit(callbacks=...)`` in the same function and nothing else.
                save_training_history_json(history, str(run_dir))
                raise RuntimeError(message)
            if epochs_run < epochs:
                logger.info(
                    f"EarlyStopping: stopped after epoch {epochs_run} of {epochs} "
                    f"(patience {checkpoint_cfg.early_stopping_patience} on {monitor}); "
                    "the best weights were restored"
                )
            # DECISION plan-2026-10-30T132432-a865fa08/D-048: the epoch reported is
            # chosen with the SAME direction the checkpoint callbacks were given
            # (``watch_mode``, resolved). Do NOT call ``best_epoch(hist, monitor)``
            # alone: it re-derives the direction from the name, so an explicit
            # ``watch_mode`` saved one epoch and the summary named another.
            best_epoch = run_summary.best_epoch(hist, monitor, metric_cfg.watch_mode)

            with _a_reporting_failure_is_logged(run_dir, epochs_run):
                fit = FitState(
                    model=model,
                    history=history,
                    last_weights=last_weights,
                    epochs_run=epochs_run,
                    run_dir=run_dir,
                    vis_dir=vis_dir,
                    experiment_name=experiment_name,
                )
                evaluated = self.evaluate_and_report(config, data, fit)

                reached = self.success_value(hist)
                smoke = debug_cfg.smoke
                # Under smoke one epoch on at most 512 samples cannot reach twice chance, so
                # the bar is not applied: the run is a success when it finished, and the
                # summary says "not applicable" rather than a miss (review-iter-2 concern 6).
                threshold = None if smoke else self.success_threshold(config)
                success = True if smoke else bool(reached >= threshold)
                summary = classification_run.base_summary(
                    summary_head, hist, epochs_run, epochs, best_epoch, lr_reductions,
                    evaluated.test_metrics_best, evaluated.test_metrics_final,
                    evaluated.best_load_error, evaluated.checkpoint_max_diff,
                    epoch_times, fit_wall_seconds, evaluated.visualizations,
                    evaluated.load_check, evaluated.conformal, evaluated.analysis,
                    notes=self._notes(config, baseline, initial_ratio, init_warning, data, resume),
                )
                if resume is not None:
                    summary["resumed_after_epochs"] = resume.start
                if reporting_only:
                    summary["reporting_resumed"] = True
                summary.update(
                    success=None if smoke else success,
                    success_metric=self.success_metric,
                    success_threshold=threshold,
                    success_value=reached,
                )
                write_summary_json(run_dir, summary)
                run_summary.fail_on_best_load_error(run_dir, evaluated.best_load_error)
            if reporting_only:
                log_gpu_peak_memory()
            if smoke:
                logger.info(
                    f"success_threshold not applicable under debug_args.smoke: best "
                    f"{self.success_metric} {reached:.4f}"
                )
            else:
                logger.info(
                    f"{'Reached' if success else 'Missed'} success_threshold {threshold:g} "
                    f"on {self.success_metric}: best {reached:.4f}"
                )
            return TrainResult(
                run_dir=run_dir,
                success=success,
                best_metric=float(hist[monitor][best_epoch - 1]),
                test_metrics_best=evaluated.test_metrics_best,
                test_metrics_final=evaluated.test_metrics_final,
                summary_path=run_dir / "results_summary.json",
            )

    # -----------------------------------------------------------------
    # Pieces
    # -----------------------------------------------------------------

    def _check_data_unchanged(
        self, run_dir: Path, config: StrictConfig, context: AutoContext, data: Any, resume: Any
    ) -> None:
        """Refuse a resume whose data is not the data the killed run trained on.

        Compares the class names, the geometry, the normalization and the per-channel
        statistics the loader reports now with the run's ``task.json``, and the split
        sizes with the sidecar's ``data_counts``. The loader has already made its one
        statistics pass by this point, so the check costs nothing more.

        :raises ConfigError: naming each difference as ``key: earlier -> now``, and that
            the run cannot be continued on other data.
        """
        from .predictor import read_record, record_differences

        # DECISION plan-2026-10-30T132432-a865fa08/D-069
        # Class names and statistics against `task.json`, split sizes against the sidecar.
        # Do NOT compare the statistics alone: a renamed class leaves them unchanged when
        # its images keep their pixels, and an added image moves them by less than the
        # tolerance in a large set; the names and the sizes catch what they miss. The
        # statistics catch a swap of two class folders (same names, same sizes). A swap
        # that leaves the fit split's statistics equal is caught by the data fingerprint
        # (D-070, D-025): relative path, class and file size of every listed file IN LISTING
        # ORDER (a reordered CSV splits differently), no pixel read; a swap
        # of two same-sized files is caught only by data.fingerprint=content, which hashes bytes.
        lines: List[str] = []
        record = self.task_record(config, context, data)
        task_path = run_dir / TASK_FILE
        if record is not None and task_path.is_file():
            lines.extend(record_differences(read_record(task_path), record))
        earlier_counts = resume.state.get("data_counts")
        if not isinstance(earlier_counts, dict):
            raise ConfigError(
                f"Cannot resume {run_dir}: the sidecar holds no record of the data's size, so "
                "it cannot be shown that the data is unchanged. Choose a new 'out'"
            )
        now_counts = _data_counts(data)
        for key in sorted(set(earlier_counts) | set(now_counts)):
            if earlier_counts.get(key) != now_counts.get(key):
                lines.append(f"{key}: {earlier_counts.get(key)} -> {now_counts.get(key)}")
        earlier_print = resume.state.get("data_fingerprint")
        now_print = getattr(data, "fingerprint", None)
        now_fields = _data_fingerprint_field(config, data)
        earlier_mode = resume.state.get("data_fingerprint_mode")
        now_mode = now_fields.get("data_fingerprint_mode")
        earlier_scheme = resume.state.get("data_fingerprint_scheme")
        now_scheme = now_fields.get("data_fingerprint_scheme")
        if earlier_scheme != now_scheme:
            # DECISION plan-2026-10-31T180033-2faf890a/D-025
            # A sidecar of another scheme is refused, and its digest is NOT compared: the
            # earlier scheme sorted the rows, so it equals nothing the loader computes now and
            # a "fallback to the old comparison" would pass exactly the reordered CSV this
            # exists to refuse. No legacy path (user rule).
            lines.append(
                f"data_fingerprint_scheme: {earlier_scheme or 'none'} -> {now_scheme or 'none'} "
                "(the run recorded its files by a rule that could not tell a reordered CSV, so "
                "it cannot be shown that the split is the same)")
        elif earlier_mode != now_mode:
            lines.append(
                f"data_fingerprint_mode: {earlier_mode or 'none'} -> {now_mode or 'none'} (the "
                "run recorded its data by another rule, so the two digests cannot be compared)")
        if earlier_scheme == now_scheme and earlier_print != now_print:
            lines.append(
                "data_fingerprint: "
                f"{'none' if earlier_print is None else str(earlier_print)[:12]} -> "
                f"{'none' if now_print is None else str(now_print)[:12]} (a file, its class, "
                "its size, the order of the files or, with data.fingerprint=content, its bytes is "
                "not the one the run started with)")
        if lines:
            raise ConfigError(
                f"Cannot resume {run_dir}: the data is not the data the run started with ("
                + "; ".join(lines)
                + "). The earlier epochs trained on the earlier data, and a resume would "
                "standardize, split and label differently; restore the data as it was, or "
                "choose a new 'out'"
            )

    def _settle_backup(
        self, run_dir: Path, resolved: StrictConfig, resume: Any, steps_per_epoch: int
    ) -> Any:
        """Make the backup a consistent pair, or refuse; see ``resume_state.settle_backup``.

        :return: ``resume``, or, when the previous copy had to be chosen (so the run goes
            back one epoch), the resume point re-read for that epoch with every refusal of
            :meth:`_resume_point` applied to it again: the CSV rows, the sidecar entry and
            the best value are those of the epoch ``fit`` will start at.
        :raises ConfigError: when neither the latest nor the previous copy is sound,
            naming both files. Nothing has been changed then.
        """
        from train.common.resume_state import settle_backup

        try:
            label, epoch = settle_backup(run_dir / BACKUP_DIR, steps_per_epoch)
        except ValueError as exc:
            raise ConfigError(
                f"Cannot resume {run_dir}: {exc}. The run cannot be continued from its "
                "backup; choose a new 'out'"
            ) from exc
        if label == "latest" and epoch == resume.start:
            return resume
        # DECISION plan-2026-10-30T132432-a865fa08/D-069
        # When the previous pair is the one that is sound, the run goes back to ITS epoch
        # and the resume point is read AGAIN (CSV rows, sidecar entry, best, wait) for that
        # epoch, with every refusal applied to it. Do NOT keep the point read from the
        # latest metadata: its rows include the epoch `fit` is about to train again, so the
        # CSV would hold that epoch twice (the duplicate row a harness copy of Keras's
        # steps produced in review-iter-3). The point is read again for the previous pair
        # even when its epoch is the same: its `batch` (how much of the epoch the copy had
        # taken, D-063) is the one the resumed epoch is shortened by.
        return self._check_resume_point(
            run_dir, resolved, load_resume_point(run_dir, self.config)
        )

    @staticmethod
    def _best_weights_of(run_dir: Path, resume: Any) -> Optional[List[Any]]:
        """Return the weights of ``best_model.keras`` (``get_weights`` order), or ``None``.

        ``None`` when the killed segment had no finite best, so no checkpoint was
        written. Loaded AFTER ``setup_gpu``: it creates tensors.
        """
        import keras

        from train.common.callbacks import best_checkpoint_path

        if resume.best is None:
            return None
        path = best_checkpoint_path(str(run_dir))
        try:
            return keras.models.load_model(path, compile=False).get_weights()
        except (OSError, ValueError, KeyError) as exc:
            raise ConfigError(
                f"Cannot resume {run_dir}: {path} cannot be loaded ({str(exc).splitlines()[0][:150]}); "
                "the best weights of the killed segment are unreadable. Choose a new 'out'"
            ) from exc

    def _load_checkpoint(self, model: Any, checkpoint: str) -> None:
        """Copy weights from ``checkpoint`` into the built ``model``, or refuse."""
        from dl_techniques.utils.logger import logger
        from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

        if not Path(checkpoint).is_file():
            raise ConfigError(
                f"Invalid value for 'checkpoint': {checkpoint!r} is not a file; give "
                "the path of a .keras model"
            )
        try:
            report = load_weights_from_checkpoint(model, checkpoint)
        except ValueError as exc:
            raise ConfigError(f"Invalid value for 'checkpoint': {exc}") from exc
        if report.num_loaded == 0:
            raise ConfigError(
                f"Invalid value for 'checkpoint': {checkpoint!r} loaded no layer into "
                "the model, so it is not a checkpoint of this architecture"
            )
        logger.info(report.summary_string())

    def _callbacks(
        self,
        config: StrictConfig,
        data: Any,
        run_dir: Path,
        vis_dir: Path,
        baseline: Dict[str, float],
        experiment_name: str,
        last_weights: Any,
        resume: Any = None,
        best_weights: Optional[List[Any]] = None,
    ) -> Tuple[Any, List[Any]]:
        """Return the epoch-line callback and the full callback list, in the D-022 order.

        :param resume: the :class:`~train.common.resume_state.ResumePoint` of a resumed
            run, else ``None``.
        :param best_weights: the best epoch's weights of a resumed run, else ``None``.
        """
        import keras

        from train.common.callbacks import EpochLogLine, LearningRateLogger, create_callbacks
        from train.common.classification_viz import TrainingDashboardCallback
        from train.common.resume_state import (
            FIT_STATE_NAME,
            FitStateSidecar,
            RestoreFitState,
            StepBackupAndRestore,
            rows_as_history,
        )

        optim_cfg: OptimArgs = config.optim_args
        metric_cfg: MetricArgs = config.metric_args
        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        logger_cfg: LoggerArgs = config.logger_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        debug_cfg: DebugArgs = config.debug_args
        every = analysis_cfg.epoch_analysis_every
        resumed: Dict[str, Any] = {}
        if resume is not None:
            # The earlier segment's epoch-0 analysis and epoch-1 analysis are on disk; the
            # analyzer would otherwise redo both, on the weights of a model BackupAndRestore
            # has not yet restored, and overwrite them.
            resumed = dict(
                initial_value_threshold=resume.best,
                csv_initial_rows=resume.rows,
                analyzer_on_train_begin=False,
                # A resume inside epoch 0 (a `backup_every_steps` copy) has not yet reached
                # the epoch-1 analysis; any later resume is past it.
                analyzer_first_epoch=resume.start == 0,
            )

        callbacks, _ = create_callbacks(
            model_name=experiment_name,
            results_dir_prefix=self.task,
            run_dir=str(run_dir),
            monitor=metric_cfg.watch_metric,
            monitor_mode=metric_cfg.watch_mode,
            patience=checkpoint_cfg.early_stopping_patience,
            use_lr_schedule=optim_cfg.schedule != "constant",
            analyzer_epoch_frequency=max(every, 1),
            include_tensorboard=logger_cfg.tensorboard is not None,
            include_terminate_on_nan=debug_cfg.terminate_on_nan,
            include_analyzer=every > 0,
            analyzer_start_epoch=max(every, 1),
            atomic_best_checkpoint=True,
            **resumed,
        )
        if not checkpoint_cfg.save_best:
            callbacks = [
                callback
                for callback in callbacks
                if not isinstance(callback, keras.callbacks.ModelCheckpoint)
            ]
        # Index 0: `lr` must be in `logs` before the CSV logger reads it, and it is the
        # rate at the START of the epoch (plan-2026-09-19T040641-db6932ec/D-020: the
        # default reads the NEXT epoch's first-step rate under a per-step schedule).
        callbacks.insert(0, LearningRateLogger(at_epoch_start=True))
        callbacks.append(last_weights)
        callbacks.extend(self.extra_callbacks(config, data))
        epoch_line = EpochLogLine(self.epoch_line_metrics(config))
        # After every callback that edits `logs`, before the dashboard (whose redraw
        # must stay outside this line's clock).
        callbacks.append(epoch_line)
        if analysis_cfg.figures:
            # The task's own per-epoch figures: after every callback that edits `logs`,
            # before the dashboard (`SegmentationGridCallback` in
            # `train.convunext.segmentation_viz` documents the same order).
            callbacks.extend(self.figure_callbacks(config, data, vis_dir))
            # DECISION plan-2026-10-31T180033-2faf890a/D-022: the dashboard stays HERE, not
            # behind `figure_callbacks`. It reads only the keys the logs hold (`loss`,
            # `accuracy`, `miou`, `lr`, ...) and draws a panel per key found, so a task
            # without `accuracy` needs nothing; and it needs the baseline, the experiment
            # name and the resumed rows, which the hook's signature does not carry. Do NOT
            # move it into the classification task: a second task would then have to
            # remember it, and its position (last, after `EpochLogLine`) would be each
            # task's to get right.
            # Last: it reads `logs['lr']`, written by LearningRateLogger above.
            callbacks.append(
                TrainingDashboardCallback(
                    out_path=vis_dir / "training_dashboard.png",
                    baseline_fn=lambda _model: dict(baseline),
                    title=f"{experiment_name} (seed {config.seed})",
                    initial_history=None if resume is None else rows_as_history(resume.rows),
                    initial_baseline=None if resume is None else dict(baseline),
                )
            )
        # The resume trio goes last, in the order `train.common.resume_state` documents:
        # the sidecar after EarlyStopping (it reads its best and wait), the backup after
        # the sidecar, so a kill at any instant leaves a sidecar entry for every epoch the
        # backup claims.
        stopper = next(c for c in callbacks if isinstance(c, keras.callbacks.EarlyStopping))
        plateau = next(
            (c for c in callbacks if isinstance(c, keras.callbacks.ReduceLROnPlateau)), None
        )
        if resume is not None:
            callbacks.append(
                RestoreFitState(
                    stopper, resume.best, resume.wait, best_weights, resume.best_epoch,
                    reduce_lr=plateau, reduce_lr_state=resume.reduce_lr,
                )
            )
        callbacks.append(
            FitStateSidecar(stopper, run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME, reduce_lr=plateau)
        )
        # DECISION plan-2026-10-30T132432-a865fa08/D-069
        # `double_checkpoint=True`: the backup keeps the previous pair as `*.bkp`. Do NOT
        # drop it to save the second copy's disk space: a kill inside the weights write
        # leaves the only copy torn, and the run can never be resumed (review-iter-3
        # concern 4, measured exposure 0.6 to 3.3 s of every epoch). `settle_backup`
        # chooses between the pair before `fit`; Keras's own fallback (on `OSError` only)
        # stays as the second line.
        # DECISION plan-2026-10-31T180033-2faf890a/D-038
        # `backup_every_steps` swaps in `StepBackupAndRestore`; unset, the stock callback is
        # built exactly as before (the golden). Do NOT pass `save_freq=N` to the stock one:
        # it skips the epoch end, records the wrong epoch after a resume and reruns the whole
        # killed epoch (resume_state.StepBackupAndRestore says what was measured; D-063 for
        # what the resume runs instead).
        if checkpoint_cfg.backup_every_steps is None:
            callbacks.append(
                keras.callbacks.BackupAndRestore(
                    str(run_dir / BACKUP_DIR), double_checkpoint=True, delete_checkpoint=False
                )
            )
        else:
            backup = StepBackupAndRestore(
                str(run_dir / BACKUP_DIR),
                save_every_steps=checkpoint_cfg.backup_every_steps,
                steps_per_epoch=data.steps_per_epoch,
            )
            callbacks.append(backup)
            # DECISION plan-2026-10-31T180033-2faf890a/D-063
            # `FinishKilledEpoch` goes directly BEFORE `EarlyStopping` (so after `TerminateOnNaN`,
            # which `create_callbacks` put in front of it). Its batch-end half must see a stop request
            # `TerminateOnNaN` made, and its epoch-end half must give the stop flag back before
            # `EarlyStopping` decides, or a request of EarlyStopping's would be undone or an
            # unrelated one kept. Do NOT append it with the others.
            callbacks.insert(callbacks.index(stopper), backup.finish)
        return epoch_line, callbacks

    def _summary_head(
        self,
        config: StrictConfig,
        family: str,
        variant: str,
        run_dir: Path,
        data: Any,
        params: int,
        baseline: Dict[str, float],
        initial_ratio: float,
        init_warning: Optional[bool],
        devices: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keys every summary carries, whether the run finished or diverged."""
        metric_cfg: MetricArgs = config.metric_args
        optim_cfg: OptimArgs = config.optim_args
        return {
            "run_dir": str(run_dir),
            "task": self.task,
            "experiment_name": run_dir.name,
            "model": config.model,
            "model_family": family,
            "variant": variant,
            "dataset": config.data.dataset if config.data.format == "builtin" else config.data.format,
            "data_format": config.data.format,
            "has_test_split": data.test_ds is not None,
            "params": params,
            "input_shape": list(data.input_shape),
            "num_classes": data.num_classes,
            "optim_args": dataclasses.asdict(optim_cfg),
            "steps_per_epoch": data.steps_per_epoch,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "precision": config.precision,
            "selection_split": data.selection_split,
            "test_split": data.test_split,
            "n_train": data.n_train,
            "n_val": data.n_select,
            "n_test": data.n_test,
            **self.summary_fields(config, data),
            "epochs_requested": config.epochs,
            "monitor": metric_cfg.watch_metric,
            "monitor_mode": metric_cfg.watch_mode,
            "initial_loss_sanity_eval": {
                "loss": baseline["loss"],
                "n_samples": data.n_select,
                "split": "val",
                "before_fit": True,
            },
            "initial_loss_ratio": initial_ratio,
            "init_scale_warning": init_warning,
            "gpu_name": devices["gpu_name"],
            "tf_visible_devices": devices["tf_visible_devices"],
            "cuda_visible_devices": devices["cuda_visible_devices"],
            **self.task_metadata(config, data),
        }

    def _notes(
        self,
        config: StrictConfig,
        baseline: Dict[str, float],
        ratio: float,
        init_warning: Optional[bool],
        data: Any,
        resume: Any = None,
    ) -> List[str]:
        """Return the summary's ``notes``: the task's, then the lines of the sequence itself."""
        notes = list(self.task_notes(config, baseline, ratio, data))
        if init_warning is None:
            notes.append(
                "`init_scale_warning` is null (not judged): the model has BatchNorm layers whose "
                "moving statistics were still at mean 0 and variance 1 for the epoch-0 "
                "evaluation, which ran in inference mode and so normalized nothing; "
                "`initial_loss_ratio` is that evaluation's loss over ln(C), kept as the "
                "baseline, and it is not the loss of the first training step (which uses batch "
                "statistics). No training-mode pass was run to measure it: it would move the "
                "moving statistics and the random generators"
            )
        if resume is not None and resume.state.get("fit_finished"):
            notes.append(
                f"the training finished ({resume.start} epochs) and an earlier reporting died, "
                "so this reporting ran again from the files and `fit` did not run: the best "
                "weights are best_model.keras's and `best_checkpoint_max_abs_diff` compares that "
                "file with itself; the last epoch's weights are checkpoints/backup/"
                "latest.weights.h5 (the epoch-end copy, checked against the epoch count and the "
                "optimizer's step count); `epoch_times` and `fit_wall_seconds` are null (no "
                "epoch ran in this segment); training_log.csv was not touched"
            )
        elif resume is not None:
            earlier = (
                f"resumed after epoch {resume.start}: epochs 0..{resume.start - 1} come from the "
                "killed segment's training_log.csv"
                if resume.start else
                "resumed inside epoch 0 from a backup_every_steps copy: no epoch had finished, "
                "so the killed segment's training_log.csv held none"
            )
            notes.append(
                f"{earlier} (its torn or surplus rows are in "
                "training_log.pre_resume_<k>.csv), the training data order of the resumed "
                "epochs replays the order of epoch 0, and `epoch_times` and "
                "`fit_wall_seconds` cover the resumed epochs only"
            )
            if resume.batch:
                left = data.steps_per_epoch - resume.batch - 1
                notes.append(
                    f"epoch {resume.start} was killed after {resume.batch + 1} of its "
                    f"{data.steps_per_epoch} batches; the resume ran its remaining {left} from a "
                    "fresh pass of the data, so some of its examples were seen twice and some not "
                    f"at all, and the epoch's row (index {resume.start}) holds the train metrics "
                    "(loss, accuracy, ...), `lr` and `epoch_times` entry of those "
                    f"{left} batches only and the validation metrics of the whole validation set"
                )
        if config.debug_args.smoke:
            notes.append(
                "debug_args.smoke is true: the success threshold is not applied, so "
                "`success` and `success_threshold` are null (not applicable)"
            )
        return notes

    def _write_diverged_summary(
        self,
        run_dir: Path,
        summary_head: Dict[str, Any],
        hist: Dict[str, List[float]],
        epochs_run: int,
        non_finite: List[str],
        lr_reductions: Optional[List[int]],
        epoch_times: List[float],
        fit_wall_seconds: float,
        initial_ratio: float,
    ) -> str:
        """Write the diverged summary and return the failure message."""
        from dl_techniques.utils.logger import logger
        from train.common.classification_run import STATUS_DIVERGED
        from train.common.run_artifacts import write_summary_json

        message = (
            f"Training diverged: {non_finite} hold a non-finite or missing value after "
            f"{epochs_run} epoch(s); no evaluation, figures, analysis or final_model.keras "
            f"were produced. Initial-loss ratio was {initial_ratio:.3g}."
        )
        logger.error(message)
        write_summary_json(
            run_dir,
            {
                "status": STATUS_DIVERGED,
                **summary_head,
                "epochs_run": epochs_run,
                "stopped_early": None,
                "best_epoch": None,
                "non_finite_metrics": non_finite,
                "lr_reduction_epochs": lr_reductions,
                "history": hist,
                "epoch_times": epoch_times,
                "fit_wall_seconds": fit_wall_seconds,
                "notes": [
                    message,
                    "non-finite values are written as null (strict JSON)",
                    "`stopped_early` is null: the run ended on a non-finite loss, not on "
                    "EarlyStopping, and no best weights were restored",
                ],
            },
        )
        return message
