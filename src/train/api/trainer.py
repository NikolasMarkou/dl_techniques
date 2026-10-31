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
    :ivar success: the best ``success_metric`` (``val_accuracy``) reached
        ``metric_args.success_threshold``; ``True`` for a ``debug_args.smoke``
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
        run directory is claimed (``resume_interrupted``, ``smoke`` with an
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
       ``resolve_out`` and ``open_run`` (claims the directory);
    2. ``setup_gpu``, ``set_seeds(config.seed)``, the precision policy (restored
       in a ``finally``);
    3. ``write_requested``; ``run.log`` opens; ``Args:`` is logged;
       ``Resolved args:`` is logged and ``write_resolved`` writes ``config.json``;
    4. (the data is described in step 1 and loaded in step 5)
    5. ``build_data``, ``build_model``, the optional ``checkpoint`` weights,
       ``compile`` through ``optim.py``;
    6. ``check_initial_loss``;
    7. callbacks: ``LearningRateLogger(at_epoch_start=True)`` first, then
       ``create_callbacks``, ``LastEpochWeights``, ``extra_callbacks``,
       ``EpochLogLine``, and the dashboard last;
    8. ``fit`` on ``data.select_ds``; ``summarize_fit``; a non-finite loss
       writes the diverged summary and the history, then raises;
    9. ``evaluate_and_report``;
    10. ``base_summary`` -> ``write_summary_json`` -> ``fail_on_best_load_error``;
    11. the :class:`TrainResult`.

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
    said, else ``RuntimeError``.
``build_model(config, context, data)``
    The model; built or not (the trainer builds it from ``data.input_shape``).
``metric_names(config, facts)``
    The names of the metrics ``compile`` will report, without ``loss``. A
    ``watch_metric`` that is not one of them (or ``loss``, or the ``val_`` twin
    of either) is refused before the run is claimed.
``build_loss_and_metrics(config, data)``
    ``(loss, metrics)`` for ``compile``. Metric names must include
    ``accuracy`` and, when ``num_classes > metric_args.top_k``,
    ``top{k}_accuracy``.
``extra_callbacks(config, data)``
    Callbacks inserted after ``LastEpochWeights``. Default: none.
``evaluate_and_report(config, data, fit)``
    Step 9's classification sequence: best-checkpoint test, conformal sets,
    figures and analysis of the best weights, the last epoch's weights. Returns
    an :class:`Evaluation`.
``task_metadata(config, data)``
    Extra keys for the summary head. Default: none.
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
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple

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
from .run_dir import open_run, resolve_out, write_requested, write_resolved

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
    test_metrics_final: Dict[str, float]
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


class TaskTrainer:
    """The training sequence of a task; see the module contract.

    :cvar task: the task's command word; a subclass sets it.
    :cvar success_metric: the history key the success threshold is read from.
    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    """

    task: ClassVar[str] = ""
    success_metric: ClassVar[str] = "val_accuracy"
    initial_loss_hint: ClassVar[str] = (
        "Lower the dropout or drop-path rates in model_args, or check the model "
        "and the data settings."
    )

    def __init__(self, config: StrictConfig) -> None:
        missing = [name for name in CONFIG_FIELDS if not hasattr(config, name)]
        if missing:
            raise TypeError(
                f"{type(config).__name__} lacks the field(s) {', '.join(missing)} "
                "that TaskTrainer reads"
            )
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

    def task_metadata(self, config: StrictConfig, data: Any) -> Dict[str, Any]:
        return {}

    def used_arguments(self, config: StrictConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run is built with beyond the config, for ``config.json``.

        The value is a JSON-plain mapping written under the key ``used``; empty
        writes no key. Default: empty.
        """
        return {}

    def evaluate_and_report(
        self, config: StrictConfig, data: Any, fit: FitState
    ) -> Evaluation:
        """Run the classification reporting sequence; see the module contract."""
        from train.common import classification_run, evaluation
        from train.common.conformal import run_conformal

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        test_best: Optional[Dict[str, float]] = None
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        if checkpoint_cfg.save_best:
            # The in-memory model holds the BEST weights here (EarlyStopping restored them).
            test_best, load_error, max_diff = classification_run.check_best_checkpoint(
                fit.model, fit.run_dir, data.test_ds, evaluate=classification_run.evaluate_dict
            )
        conformal: Optional[Dict[str, Any]] = None
        if analysis_cfg.conformal_alpha is not None:
            conformal = run_conformal(
                fit.model,
                data.test_ds,
                alpha=analysis_cfg.conformal_alpha,
                seed=config.seed,
                out_dir=fit.run_dir,
                input_shape=data.input_shape,
                model_name=fit.experiment_name,
                weights=f"best weights, selected on {data.selection_split}",
                selection_overlaps_calibration=False,
            )
        visualizations, analysis = classification_run.report_best_weights(
            fit.model,
            data,
            fit.history,
            fit.run_dir,
            fit.vis_dir,
            fit.experiment_name,
            analysis_cfg.model_analysis,
            run_analysis=evaluation.run_model_analysis,
            figures=analysis_cfg.figures,
            analysis_off_hint="analysis_args.model_analysis=false",
        )
        # The last epoch's weights replace the best ones here.
        test_final, load_check = classification_run.finalize_last_epoch(
            fit.model,
            fit.last_weights,
            data,
            fit.run_dir,
            fit.history,
            fit.epochs_run,
            evaluate=classification_run.evaluate_dict,
            save=checkpoint_cfg.save_last,
        )
        return Evaluation(
            test_metrics_best=test_best,
            best_load_error=load_error,
            checkpoint_max_diff=max_diff,
            conformal=conformal,
            visualizations=visualizations,
            analysis=analysis,
            test_metrics_final=test_final,
            load_check=load_check,
        )

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
        if config.resume_interrupted:
            raise ConfigError(
                "Invalid value for 'resume_interrupted': resuming a run is not "
                "available yet; start a new run with a new 'out'"
            )
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
        setup_gpu(None if _is_auto(config.devices) else config.devices, flag="devices")
        set_seeds(config.seed)
        previous_policy = keras.mixed_precision.global_policy().name
        setup_mixed_precision(config.precision != "float32", config.precision)
        try:
            return self._train(family, variant, run_dir, facts, resolved, context)
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
        offered a number and the resolver chose another), a family with no
        overlay for the dataset is named, and a warmup that never ends is warned.
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
        if facts.dataset not in registry.MODELS[family].defaults.get("per_dataset", {}):
            notices.append(
                (
                    "warning",
                    f"No recipe overlay for {family} on {facts.dataset}: the family's base "
                    "recipe applies unchanged, with no dataset-specific augmentation, loader "
                    "or model arguments. Check transform_args, data and model_args.",
                )
            )
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
        self._check_devices(resolved.devices)

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

    def _train(
        self,
        family: str,
        variant: str,
        run_dir: Path,
        facts: DataFacts,
        config: StrictConfig,
        context: AutoContext,
    ) -> TrainResult:
        from dl_techniques.utils.logger import logger
        from train.common import classification_run, run_summary
        from train.common.gpu import log_gpu_peak_memory
        from train.common.run_artifacts import attach_run_log, write_summary_json
        from train.common.run_io import save_training_history_json

        write_requested(run_dir, self.config)
        with attach_run_log(run_dir):
            logger.info(f"Run directory: {run_dir}")
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
            if (
                data.num_classes != facts.num_classes
                or data.input_shape[0] != facts.image_size
            ):
                raise RuntimeError(
                    f"{type(self).__name__}.describe_data said {facts.num_classes} classes "
                    f"at {facts.image_size} px, but build_data loaded {data.num_classes} "
                    f"classes with input shape {tuple(data.input_shape)}"
                )
            steps_per_epoch = data.steps_per_epoch

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
            baseline, initial_ratio, init_warning = classification_run.check_initial_loss(
                model,
                data.select_ds,
                data.n_select,
                data.num_classes,
                evaluate=classification_run.evaluate_dict,
                hint=self.initial_loss_hint,
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
            summary_head = self._summary_head(
                config, family, variant, run_dir, data, params, baseline,
                initial_ratio, init_warning, devices,
            )

            vis_dir = run_dir / "visualizations"
            if analysis_cfg.figures:
                vis_dir.mkdir(parents=True, exist_ok=True)
            last_weights = classification_run.LastEpochWeights()
            epoch_line, callbacks = self._callbacks(
                config, data, run_dir, vis_dir, baseline, experiment_name, last_weights
            )

            started = time.perf_counter()
            history = model.fit(
                data.train_ds,
                validation_data=data.select_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
            )
            fit_wall_seconds = time.perf_counter() - started
            log_gpu_peak_memory()
            hist, epochs_run, non_finite, lr_reductions = classification_run.summarize_fit(
                history, monitor, constant_lr=optim_cfg.schedule == "constant"
            )
            if non_finite:
                message = self._write_diverged_summary(
                    run_dir, summary_head, hist, epochs_run, non_finite,
                    lr_reductions, epoch_line.epoch_times, fit_wall_seconds, initial_ratio,
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

            reached = max(hist[self.success_metric])
            smoke = debug_cfg.smoke
            # Under smoke one epoch on at most 512 samples cannot reach twice chance, so
            # the bar is not applied: the run is a success when it finished, and the
            # summary says "not applicable" rather than a miss (review-iter-2 concern 6).
            threshold = None if smoke else metric_cfg.success_threshold
            success = True if smoke else bool(reached >= threshold)
            summary = classification_run.base_summary(
                summary_head, hist, epochs_run, epochs, best_epoch, lr_reductions,
                evaluated.test_metrics_best, evaluated.test_metrics_final,
                evaluated.best_load_error, evaluated.checkpoint_max_diff,
                epoch_line.epoch_times, fit_wall_seconds, evaluated.visualizations,
                evaluated.load_check, evaluated.conformal, evaluated.analysis,
                notes=self._notes(config, baseline, initial_ratio, data),
            )
            summary.update(
                success=None if smoke else success,
                success_metric=self.success_metric,
                success_threshold=threshold,
                success_value=reached,
            )
            write_summary_json(run_dir, summary)
            run_summary.fail_on_best_load_error(run_dir, evaluated.best_load_error)
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
    ) -> Tuple[Any, List[Any]]:
        """Return the epoch-line callback and the full callback list, in the D-022 order."""
        import keras

        from train.common.callbacks import EpochLogLine, LearningRateLogger, create_callbacks
        from train.common.classification_viz import TrainingDashboardCallback

        optim_cfg: OptimArgs = config.optim_args
        metric_cfg: MetricArgs = config.metric_args
        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        logger_cfg: LoggerArgs = config.logger_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        debug_cfg: DebugArgs = config.debug_args
        every = analysis_cfg.epoch_analysis_every

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
        top_k = metric_cfg.top_k
        epoch_line = EpochLogLine(
            (
                "loss",
                "accuracy",
                f"top{top_k}_accuracy",
                "val_loss",
                "val_accuracy",
                f"val_top{top_k}_accuracy",
            )
        )
        # After every callback that edits `logs`, before the dashboard (whose redraw
        # must stay outside this line's clock).
        callbacks.append(epoch_line)
        if analysis_cfg.figures:
            # Last: it reads `logs['lr']`, written by LearningRateLogger above.
            callbacks.append(
                TrainingDashboardCallback(
                    out_path=vis_dir / "training_dashboard.png",
                    baseline_fn=lambda _model: dict(baseline),
                    title=f"{experiment_name} (seed {config.seed})",
                )
            )
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
        init_warning: bool,
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
            "dataset": config.data.dataset,
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
            "figure_samples": int(len(data.x_fig)),
            "input_normalization": {"mean": data.mean, "std": data.std},
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
        self, config: StrictConfig, baseline: Dict[str, float], ratio: float, data: Any
    ) -> List[str]:
        """Plain-language lines for ``results_summary.json``'s ``notes``."""
        import numpy as np

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        notes = [
            f"initial loss {baseline['loss']:.4f} vs ln(C)={np.log(data.num_classes):.4f} "
            f"(ratio {ratio:.2f}), a plain evaluate on the validation split",
            "CSV `epoch` is 0-based, `best_epoch` is 1-based (`best_epoch_csv_index` = "
            "`best_epoch` - 1)",
            "CSV `lr` is the rate at the START of the epoch (its first step)",
            "`test_metrics_best` is the reloaded best_model.keras, `test_metrics_final` "
            "the last epoch's weights; the test set never influenced selection",
        ]
        if not checkpoint_cfg.save_best:
            notes.append(
                "checkpoint_args.save_best is false: no best_model.keras was written, so "
                "`test_metrics_best` is null; figures and analysis use the in-memory best weights"
            )
        if not checkpoint_cfg.save_last:
            notes.append(
                "checkpoint_args.save_last is false: no final_model.keras was written and "
                "`model_loading_validated` is null"
            )
        if not analysis_cfg.figures:
            notes.append("analysis_args.figures is false: no figure was written and `ece` is null")
        if not analysis_cfg.model_analysis:
            notes.append("analysis_args.model_analysis is false: `analyzer.status` is 'skipped'")
        if analysis_cfg.conformal_alpha is None:
            notes.append("analysis_args.conformal_alpha is null: `conformal` is null")
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
