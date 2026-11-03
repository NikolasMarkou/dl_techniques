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
  differ), a run with ``checkpoint_args.save_best`` false, and a ``best_model.keras`` cut short
  by a kill; a run with no epoch left (see the next section) is not refused for that, it is
  resumed to report;
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

Reporting-only resume (D-041, D-064)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A run with no epoch left to train and no ``results_summary.json`` died after its training:
its sidecar says ``fit_finished`` (the ImageNet gate's nano run died while reporting: 256 images of
224 px asked 18 GB), or every epoch ended, or ``EarlyStopping`` had waited out its patience after the
last epoch that ended, and the process died before ``fit`` returned (``resume_decisions.training_complete_reason``;
the window holds the dashboard draw and the restore of the best weights). Its training is
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
``final_model.keras`` is written over (atomically), never read; it holds the last epoch's weights AND
its optimizer state (the optimizer is built before ``latest.weights.h5`` is loaded), as a whole
run's does. The data checks are those of any resume. The summary says ``reporting_resumed: true``, ``resumed_after_epochs`` is the epoch count,
``epoch_times`` and ``fit_wall_seconds`` are null, the sidecar counts ``reporting_resumes``, and
the GPU peak of the pass is logged (it is the reporting's own). An exception of ``fit`` or of the
reporting is logged as ``ERROR`` in ``run.log`` (``Training failed:`` / ``Reporting failed:``, type
and message) before it propagates (exit code 1; ``SystemExit`` and ``KeyboardInterrupt`` are not
logged as failures); a reporting failure ends with a sentence that says whether ``resume_interrupted=true``
can run the reporting again, worded from the checks the resume applies. If the analyzer of a
reporting-only resume writes nothing, the summary says that ``analyzer.path`` names the earlier
segment's file.

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

Where the sequence lives
------------------------
This module holds the hooks, ``run`` (resolve, claim, lock, device) and the pieces the hooks feed
(``_resolve``, the summary head and notes, the diverged summary). The rest was cut out of it in step
22.5 (D-070) because ``_train`` (345 lines) and the class (1,518 lines, 46 methods) had passed
D-030's triggers: ``train.api.run_sequence`` holds the stages that run once the directory is
claimed (``execute``, ``announce``, ``load_data``, ``build_model``, ``epoch_zero``,
``train_or_restore``, ``report``) and the three handlers on that path;
``train.api.resume_decisions`` holds what a resume refuses, proves and restores;
``train.api.fit_callbacks`` builds the callback list (``_callbacks`` below is its seam). A task
overrides hooks, never stages.

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
import dataclasses
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

from . import registry, resume_decisions
from .args import AutoContext, DebugArgs, MetricArgs, OptimArgs
from .config import AUTO, StrictConfig, assert_resolved, join_path
from .data_args import DataFacts
from .errors import ConfigError, ConfigValidationError
from .optim import check_optim_args
from .run_dir import open_run, resolve_out, take_run_lock

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


def _eval_batch_size_of(config: Any) -> Optional[int]:
    """Return the config's resolved ``analysis_args.eval_batch_size``, or ``None`` when it has none.

    ``None`` for a config without the group and for an unresolved ``"auto"`` (a config built by
    hand that skipped ``resolve_auto``): no request, so the analyzer keeps its own batch.
    """
    size = getattr(getattr(config, "analysis_args", None), "eval_batch_size", None)
    return size if isinstance(size, int) and not isinstance(size, bool) else None


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

        # Imported here, not at module scope: `run_sequence` imports this module for `FitState`,
        # `TrainResult` and `_monitor_choices`.
        from . import run_sequence

        config = self.config
        debug_cfg: DebugArgs = config.debug_args
        if debug_cfg.smoke and not _is_auto(config.epochs):
            raise ConfigError(
                "Invalid value for 'epochs': debug_args.smoke runs "
                f"{SMOKE_EPOCHS} epoch and cannot be combined with an explicit "
                f"epochs, got {config.epochs!r}; remove one of them"
            )

        family, variant = registry.parse_model_name(config.model, task=self.task)
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
        resume = (
            resume_decisions.resume_point(run_dir, resolved, config)
            if config.resume_interrupted else None
        )
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
                return run_sequence.execute(
                    self, family, variant, run_dir, facts, resolved, context, resume
                )
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


    # -----------------------------------------------------------------
    # Pieces
    # -----------------------------------------------------------------


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

        The list is built by :func:`train.api.fit_callbacks.build_callbacks`; this method is
        the seam a task or a test overrides (call ``super()`` and edit the list it returns).

        :param resume: the :class:`~train.common.resume_state.ResumePoint` of a resumed
            run, else ``None``.
        :param best_weights: the best epoch's weights of a resumed run, else ``None``.
        """
        from . import fit_callbacks

        return fit_callbacks.build_callbacks(
            self, config, data, run_dir, vis_dir, baseline, experiment_name, last_weights,
            resume, best_weights,
        )

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
                "moving statistics and the random generators. The verdict is withheld for every "
                "such model, so a really mis-scaled BatchNorm model is NOT caught here: only its "
                "first epochs show it"
            )
        complete = None if resume is None else resume_decisions.training_complete_reason(resume, config)
        if complete is not None:
            notes.append(
                f"the training was complete ({resume.start} epochs: {complete}) and the earlier "
                "process died after its last epoch, in `fit`'s return or in the reporting, "
                "so this reporting ran again from the files and `fit` did not run: the best "
                "weights are best_model.keras's and `best_checkpoint_max_abs_diff` compares that "
                "file with itself; the last epoch's weights are checkpoints/backup/"
                "latest.weights.h5 (the epoch-end copy, checked against the epoch count and the "
                "optimizer's step count), and so is the optimizer state saved in "
                "final_model.keras; `epoch_times` and `fit_wall_seconds` are null (no "
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
