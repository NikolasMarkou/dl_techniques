"""The ``dino`` task: a config, a trainer and the keyword-only function.

``train_dino(...)`` is the function a Python caller uses and the function ``python -m train dino key=value ...`` calls, so the two cannot
disagree. It validates its keywords into a :class:`DinoConfig` and hands that to :class:`DinoTrainer`, a
:class:`~train.api.trainer.TaskTrainer` that supplies the hooks of this task and nothing else (a task overrides hooks, never stages). One family
trains it, ``dino_v1`` (:mod:`train.api.dino_models`): the library ``DINOTrainingModel``, a student and an EMA teacher ``DINOv1`` with projection
heads. The model has no ``train_step`` of its own and neither does the task: a run is stock ``compile`` and ``fit`` with the library ``DINOLoss``
(which keeps its centre in a variable that its own ``call`` moves) plus three callbacks the task adds (the compiled teacher EMA, the teacher
temperature schedule and the centre file) and one k-NN probe that is the only validation signal.

Interface contract
------------------
``MARGIN`` / ``margin_of(control_sd, null_p99)`` / ``CONTROL_SD`` / ``NULL_P99``
    The first part of the bar's margin per dataset, in k-NN accuracy units, and where it comes from. ``CONTROL_SD`` is the standard deviation of
    ``dino_knn_top1_k20`` over six random-init models of one dataset (plan scratch ``p4a_baselines.json``) and ``NULL_P99`` the 99th percentile
    of the best of 60 random-init draws minus the epoch-zero value (``p4b_null.json``, ``p4c_null.json``): the margin is
    ``ceil_to_0.005(max(3 sqrt(2) x CONTROL_SD, NULL_P99))``, 0.050 for imagenette and 0.085 for cifar10 (a function of the two constants, so a
    constant cannot move without its arithmetic).

``judge(best, baseline, epoch_zero, margin, collapse_flags, centred_flags)`` and :class:`Verdict`
    The bar (D-004) as one pure function; ``bar_is_met``, the summary's ``bar`` block and the notes read it. :return: a :class:`Verdict` with the
    two parts (``part1_met``, ``part2_met``), the threshold ``max(baseline, epoch_zero) + margin``, the first epoch (1-based, in the window) at
    which part 2 failed and its reason, and a sentence that names the part(s) missed.

``DinoConfig``
    Frozen :class:`~train.api.config.StrictConfig`; one field per keyword of the function. ``out``, ``model`` and ``data`` are required. Groups:
    ``data`` (:class:`~train.api.dino_args.DinoData`), ``transform_args`` (:class:`~train.api.dino_args.DinoTransformArgs`, the view size),
    ``metric_args`` (:class:`~train.api.dino_args.DinoMetricArgs`) and the shared optimizer, checkpoint, logger, analysis and debug groups.
    Refused before the run directory is claimed: ``analysis_args.conformal_alpha`` (a conformal set is a set of classes), a ``precision`` other than
    ``float32``, a constant schedule, ``checkpoint_args.backup_every_steps`` (the centre is not in the weights file) and ``init`` (any value).

``DinoTrainer(config)``
    :class:`~train.api.trainer.TaskTrainer` with ``task = "dino"`` and ``success_metric = "dino_knn_top1_k20"``.

    What a run is. An epoch is ``steps_per_epoch`` batches of an INFINITE multi-crop stream; the model sees ``2 + n_local_crops`` views of one image;
    there is NO validation data (``validation_stream`` returns ``None``: a loss call outside training would advance the loss's centre) and the k-NN
    probe of the student's CLS features, run at every epoch end on the fixed selection rows against a fixed memory bank, is the monitor. The
    checkpoint is selected on it, so ``best_model.keras`` is the epoch the verdict describes (the whole TRAINING model: ``predict`` and ``export``
    derive the feature model). A resume derives a fresh train stream from ``(seed, completed steps)`` (``resume_stream``) and restores the centre
    from a file written at each epoch end; it is NOT the unkilled run's trajectory (``resume_stream_exact`` stays false).

    The bar (D-004) has TWO parts and both are required. (1) The best ``dino_knn_top1_k20`` of the selection window is STRICTLY above
    ``max(baseline, epoch_zero) + MARGIN[dataset]``: the baseline is the k-NN accuracy of constant features against the real labels, epoch zero the
    untrained model's probe, in accuracy units. (2) The collapse flag is 0 and ``dino_teacher_entropy_is_centered`` is 1 at EVERY epoch of the
    window (the flag has measured blind spots in both directions, and the entropy cannot say a teacher is collapsed if it is NaN). A model with
    exactly-zero features, a collapsed teacher, a bank with the wrong labels, an untrained model and a run that meets only one part each miss; the
    run is kept, exits 1 and its summary says which part. Under ``debug_args.smoke`` no bar is applied.

    ``evaluate_and_report`` scores the best checkpoint (reloaded from ``best_model.keras``) and the last epoch's weights on the TEST split by
    :func:`train.common.dino_run.knn_of` (never ``model.evaluate``), runs the data-free analysis and saves ``final_model.keras``. ``Evaluation.extras``
    carries ``baseline``, ``epoch_zero``, ``margin``, ``best``, ``last3_mean``, ``collapse_max``, ``selection`` and ``bar`` (both parts with their
    numbers). The record (``task.json``) is ``build_dino_record``: the view geometry, the patch, the local crops, the head and the embedding width.

``train_dino(*, out, model, data, ...)``
    :param out: run directory (a ``str`` or a ``Path``), or ``"auto"``.
    :param model: ``"dino_v1/tiny"`` or ``"dino_v1/small"``; ``python -m train list_models task=dino`` lists them.
    :param data: a mapping (``{"dataset": "imagenette"}``), a :class:`~train.api.dino_args.DinoData`, or the path of a YAML file holding one.
    :param epochs: / ``batch_size`` / ``devices`` / ``seed`` / ``resume_interrupted`` / ``model_args`` and the groups: as in
        :func:`~train.api.tasks.image_classification.train_image_classification`; ``precision`` is ``float32`` only; ``init`` is refused (a dino
        run is no ``init`` target).
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigValidationError: for every bad key, listed together.
    :raises ConfigError: for a request that cannot be honoured; see :meth:`~train.api.trainer.TaskTrainer.run`.
    :raises RuntimeError: for a diverged or non-finite run.

Import rule
-----------
Module scope imports only the standard library and ``train.api``: ``python -m train dino help`` imports this module to print the config's keys and
must stay free of TensorFlow. Every ``keras`` and ``train.common`` name is imported inside a method.
"""

import dataclasses
import functools
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .. import dino_models, recipes, registry
from ..args import AnalysisArgs, AutoContext, CheckpointArgs, DebugArgs, LoggerArgs, OptimArgs
from ..config import (
    AUTO,
    DEFAULT_SEED,
    PATH_FORM,
    RUN_HELP,
    Auto,
    Precision,
    StrictConfig,
    resume_help,
    run_request,
    validate,
)
from ..data_args import DataFacts
from ..dino_args import (
    FAMILY,
    TASK,
    WATCH_METRIC,
    DinoData,
    DinoMetricArgs,
    DinoTransformArgs,
    check_geometry,
    dino_facts,
    load_dino_data_config,
    read_dino_data_file,
)
from ..dino_models import backup_problems, precision_problems, recipe_notices_of, schedule_problems
from ..errors import ConfigError
from ..predictor import build_dino_record, overrides_of_variant
from ..run_dir import CHECKPOINTS_DIR
from ..trainer import Evaluation, FitState, TaskTrainer, TrainResult, check_holds_no_classes, config_problems
from .pretrain import selection_block

# ---------------------------------------------------------------------
# The bar
# ---------------------------------------------------------------------

#: The probe's keys, in the order the callback writes them (``KNNEvalCallback.log_keys``; a test holds the two equal). Names only: the
#: monitor check runs before TensorFlow may be initialised.
COLLAPSE_FLAG = "dino_collapse_flag"
IS_CENTERED = "dino_teacher_entropy_is_centered"
PROBE_KEYS: Tuple[str, ...] = (
    "dino_knn_top1_k10", WATCH_METRIC, "dino_feat_mean_cos", "dino_teacher_entropy", "dino_teacher_entropy_norm", IS_CENTERED, COLLAPSE_FLAG,
)

#: What a random-init model's k-NN accuracy varies by, per dataset: the standard deviation over six seeds, and the 99th percentile of the best of
#: 60 draws above the epoch-zero value (the estimator the bar uses: the BEST epoch of a run). Plan-2026-11-26T224507-e3381378 scratch_step2:
#: ``p4a_baselines.json`` (CONTROL_SD, six seeds each), ``p4b_null.json`` (imagenette NULL_P99) and ``p4c_null.json`` (cifar10 NULL_P99).
CONTROL_SD: Dict[str, float] = {"imagenette": 0.0103, "cifar10": 0.0198}
NULL_P99: Dict[str, float] = {"imagenette": 0.0487, "cifar10": 0.0416}

#: Standard deviations of the difference of two random-init draws that the margin clears: 3 of ``sqrt(2) x sd``.
SD_MULTIPLE = 3.0
MARGIN_STEP = 0.005


def margin_of(control_sd: float, null_p99: float) -> float:
    """Return ``ceil_to_0.005(max(3 sqrt(2) x control_sd, null_p99))``; see the module contract."""
    # DECISION plan-2026-11-26T224507-e3381378/D-026
    # The margin is a FUNCTION of the two measured spreads, and the function is pinned by a test with its arithmetic. The first form (3 sd of the
    # null delta, 0.045 for imagenette) was 14 percent too small once P4 measured the best of 60 draws (its 99th percentile, 0.0487, reached it),
    # so the larger of the two binds. Do NOT type a margin: a number with no derivation moves without its reason. cifar10's stays at the same
    # function's output, 0.085, and no cifar10 DINO run has been trained (D-032).
    raw = max(SD_MULTIPLE * math.sqrt(2.0) * control_sd, null_p99)
    return round(math.ceil(round(raw / MARGIN_STEP, 9)) * MARGIN_STEP, 6)


MARGIN: Dict[str, float] = {name: margin_of(CONTROL_SD[name], NULL_P99[name]) for name in CONTROL_SD}


@dataclasses.dataclass(frozen=True)
class Verdict:
    """The two parts of the bar with the numbers behind them; see :func:`judge`."""

    threshold: float
    best: float
    part1_met: bool
    part2_met: bool
    first_offending_epoch: Optional[int]
    part2_reason: Optional[str]
    text: str

    @property
    def met(self) -> bool:
        return self.part1_met and self.part2_met


def judge(
    *, best: float, baseline: float, epoch_zero: float, margin: float, collapse_flags: Sequence[float], centred_flags: Sequence[float],
    first_epoch: int = 1,
) -> Verdict:
    """Return the :class:`Verdict` of one run; the bar of the module contract.

    :param best: the best ``dino_knn_top1_k20`` of the selection window. :param baseline: the constant-feature k-NN accuracy. :param epoch_zero: the
        untrained model's probe. :param margin: ``MARGIN[dataset]``. :param collapse_flags, centred_flags: the probe's flags at every epoch of the
        window, in order. :param first_epoch: the 1-based epoch number of the first flag.
    """
    # DECISION plan-2026-11-26T224507-e3381378/D-055
    # BOTH parts (D-004), each STRICT. Part 1 is `best > max(baseline, epoch_zero) + margin` on plain floats in accuracy units (a k-NN accuracy moves in
    # steps of 1/946 and the forecast bars' float32 tie bands do not fit it): a value equal to the threshold misses. Part 2 reads the flag AND
    # `is_centered` at every epoch, and a value that is not exactly 0 (flag) or 1 (centred), a NaN included, is an offence: the flag cannot see
    # exactly-zero features and `collapse_metrics` never trips on a NaN entropy. Do NOT use `>=`, do NOT drop the baseline or the epoch-zero
    # term (the larger binds: a run must beat what a constant and an untrained model score), do NOT drop either part or either flag, and do NOT
    # judge part 2 on the last epoch alone.
    threshold = max(baseline, epoch_zero) + margin
    part1 = bool(best > threshold)
    offending: Optional[int] = None
    reason: Optional[str] = None
    for offset, (flag, centred) in enumerate(zip(collapse_flags, centred_flags)):
        problems = []
        if not flag == 0.0:
            problems.append(f"{COLLAPSE_FLAG} is {flag!r}, not 0")
        if not centred == 1.0:
            problems.append(f"{IS_CENTERED} is {centred!r}, not 1")
        if problems:
            offending, reason = first_epoch + offset, "; ".join(problems)
            break
    part2 = offending is None and len(collapse_flags) > 0 and len(collapse_flags) == len(centred_flags)
    words = []
    if not part1:
        words.append(
            f"part 1 missed: best {WATCH_METRIC} {best:.4f} is not above max(baseline {baseline:.4f}, epoch zero {epoch_zero:.4f}) + margin "
            f"{margin:.3f} = {threshold:.4f}")
    if not part2:
        words.append(
            f"part 2 missed: {reason} at epoch {offending}" if offending is not None else "part 2 missed: the window holds no probe flag")
    text = "both parts met: " + f"best {best:.4f} is above {threshold:.4f} and no epoch of the window collapsed" if not words else "; ".join(words)
    return Verdict(threshold, best, part1, part2, offending, reason, text)


# ---------------------------------------------------------------------

#: The file the loss's centre is kept in, inside the run's ``checkpoints`` directory.
CENTRE_FILE = "dino_centre.npy"

_RESUME_ORDER = (
    "the training shuffle and the crops continue on a fresh stream derived from the seed and the steps completed, never epoch 0's order and "
    "never a selection row, the loss's centre is restored from the file written at the last finished epoch, and the selection, test and "
    "probe streams stay fixed"
)

_STREAM_RULE = (
    "the selection, test and probe streams are keyed on the split seed and never move; the train shuffle and the crops on the stream_seed; "
    "a resume derives stream_seed from (seed, completed steps), the first 4 bytes of sha256, and the split never follows it"
)

_CONFORMAL = f"a conformal prediction set is a set of classes, which a {TASK} run does not produce; its output is a feature vector"

_INIT_REFUSED = (
    "a dino run is no `init` target: its model holds a student and a teacher at the same weight paths, so a copy would need a teacher re-sync "
    "that this task does not do, and epoch zero under `init` would call `model.evaluate`, which advances the loss's centre; a dino run can seed "
    "a `vit` classification run of its own width and depth instead (model=vit/vit_tiny after dino_v1/tiny)"
)


@dataclasses.dataclass(frozen=True)
class DinoConfig(StrictConfig):
    """Everything a dino run is told; see the module contract."""

    out: str = dataclasses.field(metadata={"help": RUN_HELP["out"]})
    model: str = dataclasses.field(
        metadata={"help": f"family/variant, e.g. {FAMILY}/tiny; python -m train list_models task={TASK} lists them"}
    )
    data: DinoData = dataclasses.field(
        metadata={
            "help": "the images and the views: data.dataset=imagenette (or cifar10), with data.patch_size and data.n_local_crops; or the path of "
            "a YAML file holding them (data=data.yaml), over which dotted keys win (data=data.yaml data.max_samples=100)",
            PATH_FORM: read_dino_data_file,
        }
    )
    epochs: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["epochs"]})
    batch_size: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["batch_size"]})
    seed: int = dataclasses.field(default=DEFAULT_SEED, metadata={"help": RUN_HELP["seed"]})
    devices: Auto[Optional[int]] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["devices"]})
    precision: Precision = dataclasses.field(
        default="float32",
        metadata={"help": "float32, the only value accepted: the loss's centre update and the teacher's EMA were never measured in half precision"},
    )
    resume_interrupted: bool = dataclasses.field(
        default=False,
        metadata={"help": resume_help("an epoch of the larger dataset takes minutes", order=_RESUME_ORDER)},
    )
    init: Optional[str] = dataclasses.field(default=None, metadata={"help": RUN_HELP["init"]})
    model_args: Dict[str, Any] = dataclasses.field(default_factory=dict, metadata={"help": RUN_HELP["model_args"]})
    optim_args: OptimArgs = dataclasses.field(default_factory=OptimArgs)
    transform_args: DinoTransformArgs = dataclasses.field(default_factory=DinoTransformArgs)
    metric_args: DinoMetricArgs = dataclasses.field(default_factory=DinoMetricArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)

    def problems(self, prefix: str) -> List[str]:
        """Name the top-level fields out of range, a conformal level, a precision, a schedule and a backup interval the task refuses."""
        return [
            *config_problems(self, prefix, no_conformal_for=TASK, because=_CONFORMAL),
            *precision_problems(self.precision, prefix),
            *schedule_problems(self.optim_args.schedule),
            *backup_problems(self.checkpoint_args.backup_every_steps),
        ]


class DinoTrainer(TaskTrainer):
    """The hooks of a dino run; see the module contract.

    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    :cvar resume_order: what a resume does to the training stream and the centre, for the summary's resume note.
    """

    task = TASK
    success_metric = WATCH_METRIC
    resume_order = _RESUME_ORDER
    initial_loss_hint = "Lower optim_args.lr, or check the model arguments and the images."

    def __init__(self, config: StrictConfig) -> None:
        super().__init__(config)
        #: The run's ``DINOLoss`` (``build_loss_and_metrics``): the probe, the centre file and the schedules read the very object compiled.
        self._loss: Optional[Any] = None
        #: ``{teacher_temp, teacher_temp_final, ema_warmup_epochs}`` as the request resolves them, set by ``build_model``.
        self._values: Optional[Dict[str, Any]] = None
        self._out_dim: Optional[int] = None
        #: The constant-feature k-NN accuracy and the epoch-zero probe value (the sidecar's), and the window's flags: set by ``evaluate_and_report``.
        self._baseline: Optional[float] = None
        self._epoch_zero: Optional[float] = None
        self._flags: Optional[Tuple[List[float], List[float]]] = None
        self._first_epoch: int = 1

    # -- data and model -------------------------------------------------

    def describe_data(self, config: DinoConfig) -> DataFacts:
        size = config.transform_args.image_size
        return dino_facts(config.data, None if size == AUTO else size)

    def build_data(self, config: DinoConfig, context: AutoContext) -> Any:
        from train.common import dino_data

        data_cfg: DinoData = config.data
        return dino_data.load_dino_batches(
            dataset=data_cfg.dataset, image_size=config.transform_args.image_size, n_local_crops=data_cfg.n_local_crops,
            batch_size=config.batch_size, seed=config.seed, stream_seed=None, validation_split=data_cfg.validation_split,
            max_samples=data_cfg.max_samples, knn_bank_batches=dino_models.FIXED["knn_bank_batches"],
            knn_query_batches=dino_models.FIXED["knn_query_batches"])

    def input_check(self, facts: DataFacts, data: Any) -> None:
        check_holds_no_classes(self, facts, data)

    def build_model(self, config: DinoConfig, context: AutoContext, data: Any) -> Any:
        # The adapter returns a model BUILT with the rank-5 input whose views and grid it proved, so the sequence's own `build` call is skipped.
        kwargs = self._model_kwargs(config, context)
        self._values, self._out_dim = self._task_values(config, context), int(kwargs["dino_out_dim"])
        return recipes.build_model(registry.MODELS[context.family], context.variant, None, data.input_shape, kwargs)

    @staticmethod
    def _task_values(config: DinoConfig, context: AutoContext) -> Dict[str, Any]:
        return dino_models.task_values(context, config.optim_args, config.model_args)

    @staticmethod
    def _model_kwargs(config: DinoConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the factory's keywords: the recipe's, then the caller's (the resolver refuses the rest), and the data's view geometry."""
        kwargs = registry.MODELS[context.family].load_resolver()(context, config.optim_args, config.model_args)
        return {**kwargs, "patch_size": config.data.patch_size, "n_local_crops": config.data.n_local_crops}

    def task_record(self, config: DinoConfig, context: AutoContext, data: Any) -> Dict[str, Any]:
        """Return the ``dino`` ``task.json`` record (schema in :mod:`train.api.predictor`).

        Nothing of the model is needed, so a resume asks for it before ``build_model``.
        """
        from train.common.image_streams import stream_statistics

        height, width, channels = data.input_shape
        mean, std = stream_statistics(config.data.dataset)
        kwargs = self._model_kwargs(config, context)
        table = registry.MODELS[context.family].load_variants()
        return build_dino_record(
            model=config.model, family=context.family, variant=context.variant, input_size=(height, width), channels=channels, mean=mean,
            std=std, patch_size=kwargs["patch_size"], n_local_crops=kwargs["n_local_crops"], dino_out_dim=kwargs["dino_out_dim"],
            embedding_dim=table[context.variant]["embed_dim"],
            variant_overrides=overrides_of_variant(table[context.variant], kwargs))

    def init_identity(self, config: DinoConfig, context: AutoContext) -> Dict[str, Any]:
        # DECISION plan-2026-11-26T224507-e3381378/D-056 (the clause naming the `vit` pair: D-071)
        # Refused for ANY source (D-013), before the claim. Do NOT build an identity "for later": the student and the teacher sit at the same weight
        # paths, so a copy needs a teacher re-sync nothing here does, and the sequence's epoch zero under `init` is `evaluate`, which moves the
        # centre (the very call this task exists to avoid). A dino run is a SOURCE for a `vit` classifier (the pair of recipes.VIT_TRANSFER, D-068), never a target.
        raise ConfigError(f"Invalid value for 'init': {_INIT_REFUSED}")

    def used_arguments(self, config: DinoConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json``: the model keywords and the values the task reads."""
        return {"model_args": recipes.serialize_model_args({**self._model_kwargs(config, context), **self._task_values(config, context)})}

    def check_model_args(self, config: DinoConfig, context: AutoContext) -> None:
        # The family's own refusals first (every refused key at once), then the geometry of the views.
        self._model_kwargs(config, context)
        self._task_values(config, context)
        check_geometry(config.data.patch_size, config.transform_args.image_size)

    # -- compile --------------------------------------------------------

    def metric_names(self, config: DinoConfig, facts: DataFacts) -> List[str]:
        # DECISION plan-2026-10-30T132432-a865fa08/D-055: names only; a Keras metric allocates variables and the first eager op would
        # initialise TensorFlow's devices before setup_gpu.
        return list(PROBE_KEYS)

    def build_loss_and_metrics(self, config: DinoConfig, data: Any) -> Tuple[Any, List[Any]]:
        from dl_techniques.losses.dino_loss import DINOLoss

        if self._values is None or self._out_dim is None:
            raise RuntimeError("build_loss_and_metrics needs the values build_model resolves; it was asked for before the model was built")
        fixed = dino_models.FIXED
        self._loss = DINOLoss(
            out_dim=self._out_dim, student_temp=fixed["student_temp"], teacher_temp=self._values["teacher_temp"],
            center_momentum=fixed["center_momentum"])
        return self._loss, []

    def initial_loss_check(self, config: DinoConfig, model: Any, data: Any) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return the epoch-0 numbers: the loss of one batch and every probe key of the untrained model; no verdict on the scale.

        :raises RuntimeError: for a non-finite loss (the model diverges before training).
        """
        from dl_techniques.utils.logger import logger
        from train.common import dino_run, run_summary

        if self._loss is None:
            raise RuntimeError("initial_loss_check scores the loss compiled for the run; it was asked for before the loss was built")
        # DECISION plan-2026-11-26T224507-e3381378/D-022
        # NEVER `model.evaluate` here: the loss moves its centre inside `call`, so an evaluate of the untrained model would start the run off a
        # zero centre with a finite loss and a clean exit. `epoch_zero` runs the frozen networks directly and one eager forward through a
        # THROWAWAY loss; the run's centre is untouched.
        zero = dino_run.epoch_zero(model, data, self._loss)
        ratio = run_summary.initial_loss_ratio({"loss": zero.loss}, math.log(self._loss.out_dim), self.initial_loss_hint)
        logger.info(
            f"Untrained probe BEFORE fit: {', '.join(f'{key}={value:.4f}' for key, value in zero.probe.items())}; constant-feature "
            f"{WATCH_METRIC} {zero.baseline[WATCH_METRIC]:.4f}; loss of one batch {zero.loss:.6g}, {ratio:.3g}x ln(out_dim) "
            f"{math.log(self._loss.out_dim):.4g}")
        return {"loss": zero.loss, **zero.probe}, ratio, None

    def unjudged_note(self, config: DinoConfig) -> str:
        return (
            "`init_scale_warning` is null (not judged): `initial_loss_ratio` is the loss of ONE multi-crop batch of the untrained model, through a "
            "throwaway loss of the run's hyperparameters, over ln(dino_out_dim), kept as the baseline. The DINO loss of a random initialization "
            "is not a scale test (the teacher and the student start equal), so no verdict is given; a non-finite loss is refused"
        )

    # -- the seams of the shared sequence -------------------------------

    def fit_kwargs(self, config: DinoConfig, data: Any) -> Dict[str, Any]:
        # The train stream is INFINITE, so an epoch is `steps_per_epoch` batches by this keyword. No `validation_steps`: there is no validation.
        return {"steps_per_epoch": data.steps_per_epoch}

    def validation_stream(self, config: DinoConfig, data: Any) -> Optional[Any]:
        # DECISION plan-2026-11-26T224507-e3381378/D-007
        # NONE, never `data.select_ds`: Keras would run the loss over every validation batch and the loss advances its centre inside `call`
        # (measured: a 4-sample validation set pushed the centre 81 percent past its value with a finite loss and a clean exit). The k-NN probe
        # is the validation, and it never calls the loss. Do NOT return the selection stream "so val_loss exists": there is no val_loss.
        return None

    def resume_stream(self, config: DinoConfig, data: Any, completed_steps: int) -> Optional[Any]:
        # DECISION plan-2026-11-26T224507-e3381378/D-010
        # A resumed `fit` starts a fresh iterator, so the stream the fresh process built would replay from its first element (epoch 0's order and
        # crops). `with_stream` rebuilds the SAME request on a stream seed derived from `(seed, completed_steps)`: the split, the counts and the
        # plain and probe streams do not move, the train stream does. Do NOT return `data` (the replay), do NOT reseed the split, and do NOT set
        # `resume_stream_exact`: this is another stream, not the batches the unkilled run reads next.
        return data.with_stream(completed_steps)

    def _callbacks(
        self, config: DinoConfig, data: Any, run_dir: Path, vis_dir: Path, baseline: Dict[str, float], experiment_name: str,
        last_weights: Any, resume: Any = None, best_weights: Optional[List[Any]] = None,
    ) -> Tuple[Any, List[Any]]:
        import keras

        from dl_techniques.models.vision.depth_anything.teacher_ema import cosine_ema_schedule, linear_ema_schedule
        from train.common import dino_run
        from train.common.resume_state import FinishKilledEpoch

        epoch_line, callbacks = super()._callbacks(
            config, data, run_dir, vis_dir, baseline, experiment_name, last_weights, resume, best_weights)
        loss, values = self._loss, self._values
        if loss is None or values is None:
            raise RuntimeError("the callbacks need the loss and the task values, which the model and compile stages set; they were not run")
        # DECISION plan-2026-11-26T224507-e3381378/D-008
        # The probe goes BEFORE the first of the stopper and the killed-epoch finisher (`FinishKilledEpoch` is present only with a mid-epoch
        # backup, which this task refuses; the `min` keeps the rule true if that changes): `EarlyStopping` and the checkpoint read the monitor from
        # `logs`, and the CSV logger, which comes after them, freezes its header from the first row it sees. A probe appended after them is
        # invisible to all three (measured, P8: no best_model.keras, a CSV with no probe column, a warning at every epoch). Do NOT use
        # `extra_callbacks` (it appends after the stopper).
        stoppers = [
            index for index, callback in enumerate(callbacks) if isinstance(callback, (keras.callbacks.EarlyStopping, FinishKilledEpoch))]
        callbacks.insert(min(stoppers), dino_run.build_probe(data, loss))
        # DECISION plan-2026-11-26T224507-e3381378/D-054
        # The centre file is written BEFORE the weights backup at an epoch end (a kill between them leaves the centre one epoch AHEAD of the
        # weights, which `CentreFile` and the documents state; written after, one epoch behind: either order is off by one in that window), and
        # the EMA callback comes LAST so its `on_train_begin` binds the weights `BackupAndRestore` has restored.
        backups = [index for index, callback in enumerate(callbacks) if isinstance(callback, keras.callbacks.BackupAndRestore)]
        at = backups[0] if backups else len(callbacks)
        callbacks.insert(at, dino_run.CentreFile(
            loss, Path(run_dir) / CHECKPOINTS_DIR / CENTRE_FILE, completed_epochs=None if resume is None else resume.start))
        steps = data.steps_per_epoch
        horizon = max(1, int(dino_models.FIXED["teacher_temp_warmup_epochs"]))
        temperature = linear_ema_schedule(decay_start=values["teacher_temp"], decay_end=values["teacher_temp_final"], total_steps=horizon)
        callbacks.append(keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch, logs=None: loss.set_teacher_temp(temperature(epoch))))
        callbacks.append(dino_run.CompiledTeacherEMA(
            schedule=cosine_ema_schedule(
                decay_start=dino_models.FIXED["ema_decay_start"], decay_end=dino_models.FIXED["ema_decay_end"],
                total_steps=max(1, steps * config.epochs)),
            warmup_steps=dino_models.ema_warmup_steps(values["ema_warmup_epochs"], steps), log_every=max(1, steps)))
        return epoch_line, callbacks

    # -- the verdict ----------------------------------------------------

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        """Return the best (the highest) ``dino_knn_top1_k20`` of the selection window: the value the checkpoint saved."""
        import numpy as np

        return float(np.nanmax(np.asarray(hist[self.success_metric][self.selection_start_epoch(self.config):], dtype=np.float64)))

    def success_threshold(self, config: DinoConfig) -> float:
        """Return ``max(baseline, epoch_zero) + MARGIN[dataset]``; ``evaluate_and_report`` has read the two terms."""
        if self._baseline is None or self._epoch_zero is None:
            raise RuntimeError(
                "success_threshold needs the constant-feature baseline and the epoch-zero probe value, which evaluate_and_report works out; it "
                "was asked for before the reporting"
            )
        return float(max(self._baseline, self._epoch_zero) + MARGIN[config.data.dataset])

    def _verdict(self, config: DinoConfig, reached: float) -> Verdict:
        if self._flags is None or self._baseline is None or self._epoch_zero is None:
            raise RuntimeError("the verdict needs the baseline, the epoch-zero value and the window's flags, which evaluate_and_report works out")
        return judge(
            best=reached, baseline=self._baseline, epoch_zero=self._epoch_zero, margin=MARGIN[config.data.dataset],
            collapse_flags=self._flags[0], centred_flags=self._flags[1], first_epoch=self._first_epoch)

    def bar_is_met(self, config: DinoConfig, reached: float, threshold: float, loaded: Optional[float] = None) -> bool:
        # DECISION plan-2026-11-26T224507-e3381378/D-004
        # `threshold` is the one the sequence cut (`success_threshold`); the verdict recomputes it from its terms and the two must agree, so a
        # task that changed one and not the other fails loudly instead of judging against a number nobody reported.
        verdict = self._verdict(config, reached)
        if not math.isclose(verdict.threshold, threshold, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"the bar's threshold {threshold!r} is not the verdict's {verdict.threshold!r}")
        return verdict.met

    # -- the log and the summary ---------------------------------------

    def epoch_line_metrics(self, config: DinoConfig) -> Tuple[str, ...]:
        return ("loss", WATCH_METRIC)

    def summary_fields(self, config: DinoConfig, data: Any) -> Dict[str, Any]:
        return {
            "patch_size": config.data.patch_size,
            "n_local_crops": config.data.n_local_crops,
            "n_views": data.n_views,
            "probe": {"bank_batches": data.bank_batches, "query_batches": data.query_batches},
            "stream": {"seed": data.seed, "stream_seed": data.stream_seed, "rule": _STREAM_RULE},
        }

    def recipe_notices(self, facts: DataFacts, family: str) -> List[Tuple[str, str]]:
        return recipe_notices_of(facts.dataset)

    def task_notes(self, config: DinoConfig, baseline: Dict[str, float], ratio: float, data: Any) -> List[str]:
        """Return the dino lines of ``results_summary.json``'s ``notes``."""
        from train.common import classification_run

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        margin = MARGIN[config.data.dataset]
        notes = [
            f"initial loss {baseline['loss']:.6g} is {ratio:.3g}x ln(dino_out_dim): the loss of one multi-crop batch of the untrained model through "
            "a throwaway loss, never `model.evaluate` (which advances the loss's centre); see the `init_scale_warning` note",
            *classification_run.RUN_NOTES[:2],
            f"there is no validation data: the k-NN probe of the student's CLS features at every epoch end (a bank of {data.bank_batches} batches "
            f"of the fit rows, queries from the {data.n_select} selection rows) is the only signal, so the history holds no `val_` key and "
            f"`{WATCH_METRIC}` picks the best epoch",
            f"the bar has two parts, and both are required: (1) `success_value`, the best `{WATCH_METRIC}` of the selection window, is strictly "
            f"above `success_threshold`, max(`baseline`, `epoch_zero`) + {margin:.3f} (`bar.part1`: the margin is `margin_of` the measured spread "
            "of a random-init model's probe, see `train.api.tasks.dino`); (2) the collapse flag is 0 and the teacher entropy is centred at every "
            "epoch of the window (`bar.part2`); a model of zero features, a collapsed teacher, a bank with the wrong labels, an untrained model "
            "and a run that meets one part only each miss",
            "`test_metrics_best` is the reloaded best_model.keras (the whole training model) and `test_metrics_final` the last epoch's weights, "
            "both scored by the k-NN probe on the test split, which never influenced selection",
        ]
        notes.append(
            "the model analysis is data-free (weights and spectral only; `analyzer.analyzers` lists them)"
            if analysis_cfg.model_analysis else "analysis_args.model_analysis is false: `analyzer.status` is 'skipped'"
        )
        if not checkpoint_cfg.save_best:
            notes.append("checkpoint_args.save_best is false: no best_model.keras was written, so the reload check did not run")
        if not checkpoint_cfg.save_last:
            notes.append("checkpoint_args.save_last is false: no final_model.keras was written and `model_loading_validated` is null")
        return notes

    # -- the reporting --------------------------------------------------

    def run_analysis(
        self, config: DinoConfig, model: Any, test_data: Tuple[Any, Any], history: Any, experiment_name: str, run_dir: str
    ) -> Any:
        """Run the data-free analysis (weights and spectral only); see the module contract."""
        from train.common import run_summary

        # The stock analysis reads the output as class probabilities; this model's output is the packed logits of two heads (measured, P8).
        return run_summary.data_free_model_analysis(model, test_data, history, experiment_name, run_dir, config.seed)

    def evaluate_and_report(self, config: DinoConfig, data: Any, fit: FitState) -> Evaluation:
        """Run the dino reporting sequence; see the module contract."""
        import numpy as np

        from train.common import classification_run, dino_run, pretrain_run, run_summary
        from train.common.resume_state import FIT_STATE_NAME, read_fit_state

        if self._loss is None:
            raise RuntimeError("evaluate_and_report needs the loss compiled for the run; the model stage did not run")
        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        metric_cfg: DinoMetricArgs = config.metric_args
        sidecar = fit.run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME
        # The epoch-zero term from the SIDECAR, always (a resume never evaluates the untrained model), and the constant-feature baseline from the
        # labels (no model: a resume reads it the same way).
        self._epoch_zero = pretrain_run.epoch_zero_val_loss(sidecar, key=WATCH_METRIC)
        initial = read_fit_state(sidecar)["initial"]["metrics"]
        self._baseline = dino_run.constant_feature_baseline(data)[WATCH_METRIC]
        score = functools.partial(dino_run.knn_of, data=data, loss=self._loss)
        test_best: Optional[Dict[str, float]] = None
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        if checkpoint_cfg.save_best:
            # The in-memory model holds the BEST weights here (EarlyStopping restored them): the reloaded file must score them, to the tolerance.
            test_best, load_error, max_diff = classification_run.check_best_checkpoint(
                fit.model, fit.run_dir, data.test_ds, evaluate=score, split_label="Test results")
        # A rank-5 sample for the load check and the analysis: the model takes whole multi-crop view sets. One batch of the selection rows, put
        # through the same multi-crop map, the same on every pass.
        views, labels = next(iter(data.loss_batch_ds))
        sample = np.asarray(views)[: classification_run.LOAD_CHECK_SAMPLES]
        from types import SimpleNamespace

        analysis, test_final, load_check, load_detail = run_summary.report_data_free_run(
            fit.model,
            SimpleNamespace(x_fig=np.asarray(views), y_fig=np.asarray(labels)),
            fit.history,
            fit.run_dir,
            fit.vis_dir,
            fit.experiment_name,
            run_analysis=functools.partial(self.run_analysis, config),
            model_analysis=analysis_cfg.model_analysis,
            eval_batch_size=analysis_cfg.eval_batch_size,
            last_weights=fit.last_weights,
            data=data,
            epochs_run=fit.epochs_run,
            save_last=checkpoint_cfg.save_last,
            load_sample=sample,
            evaluate=score,
        )
        visualizations: Dict[str, Any] = {"files": [], "failed": [], "skipped": not analysis_cfg.figures}
        if analysis_cfg.figures and fit.vis_dir.is_dir():
            visualizations["files"] = sorted(path.name for path in fit.vis_dir.iterdir() if path.is_file())
        hist = {key: [float(v) for v in values] for key, values in fit.history.history.items()}
        start = self.selection_start_epoch(config)
        best = run_summary.best_epoch(hist, metric_cfg.watch_metric, metric_cfg.watch_mode, first_epoch=start)
        self._flags = ([float(v) for v in hist[COLLAPSE_FLAG][start:]], [float(v) for v in hist[IS_CENTERED][start:]])
        self._first_epoch = start + 1
        reached = self.success_value(hist)
        verdict = self._verdict(config, reached)
        window = hist[WATCH_METRIC][start:]
        extras: Dict[str, Any] = {
            "baseline": {WATCH_METRIC: self._baseline, "kind": "constant features against the real labels of the bank and the queries"},
            "epoch_zero": {**{key: initial[key] for key in PROBE_KEYS if key in initial}, "loss": initial.get("loss"),
                           "binds": bool(self._epoch_zero > self._baseline)},
            "margin": {"value": MARGIN[config.data.dataset], "control_sd": CONTROL_SD[config.data.dataset],
                       "null_p99": NULL_P99[config.data.dataset], "rule": "ceil_to_0.005(max(3 sqrt(2) x control_sd, null_p99))"},
            "best": reached,
            "last3_mean": float(np.mean(window[-3:])),
            "collapse_max": float(np.nanmax(self._flags[0])),
            "selection": selection_block(start, fit.epochs_run, best),
            "bar": {
                "applied": not config.debug_args.smoke,
                "part1": {"success_value": reached, "success_threshold": verdict.threshold, "met": verdict.part1_met},
                "part2": {"met": verdict.part2_met, "first_offending_epoch": verdict.first_offending_epoch, "reason": verdict.part2_reason,
                          "epochs_checked": len(self._flags[0])},
                "met": verdict.met,
                "verdict": verdict.text,
            },
        }
        return Evaluation(
            test_metrics_best=test_best,
            best_load_error=load_error,
            checkpoint_max_diff=max_diff,
            conformal=None,
            visualizations=visualizations,
            analysis=analysis,
            test_metrics_final=test_final,
            load_check=load_check,
            load_detail=load_detail,
            extras=extras,
        )


def train_dino(
    *,
    out: Union[str, Path],
    model: str,
    data: Union[DinoData, Mapping[str, Any], str, Path],
    epochs: Union[int, str] = AUTO,
    batch_size: Union[int, str] = AUTO,
    seed: int = DEFAULT_SEED,
    devices: Union[int, str, None] = AUTO,
    precision: str = "float32",
    resume_interrupted: bool = False,
    init: Union[str, Path, None] = None,
    model_args: Optional[Mapping[str, Any]] = None,
    optim_args: Union[OptimArgs, Mapping[str, Any], None] = None,
    transform_args: Union[DinoTransformArgs, Mapping[str, Any], None] = None,
    metric_args: Union[DinoMetricArgs, Mapping[str, Any], None] = None,
    checkpoint_args: Union[CheckpointArgs, Mapping[str, Any], None] = None,
    logger_args: Union[LoggerArgs, Mapping[str, Any], None] = None,
    analysis_args: Union[AnalysisArgs, Mapping[str, Any], None] = None,
    debug_args: Union[DebugArgs, Mapping[str, Any], None] = None,
) -> TrainResult:
    """Train a DINOv1 self-supervised ViT; see the module contract."""
    recipes.register_builtin_models()
    if isinstance(data, (str, Path)):
        data = load_dino_data_config(data)
    given = run_request(
        out=out, model=model, data=data, epochs=epochs, batch_size=batch_size, seed=seed,
        devices=devices, precision=precision, resume_interrupted=resume_interrupted,
        init=init, model_args=model_args,
        groups={
            "optim_args": optim_args,
            "transform_args": transform_args,
            "metric_args": metric_args,
            "checkpoint_args": checkpoint_args,
            "logger_args": logger_args,
            "analysis_args": analysis_args,
            "debug_args": debug_args,
        },
    )
    config = validate(DinoConfig, given)
    return DinoTrainer(config).run()
