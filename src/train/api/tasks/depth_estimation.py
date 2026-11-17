"""The depth_estimation task: a config, a trainer and the keyword-only function.

``train_depth_estimation(...)`` is the function a Python caller uses and the function
``python -m train depth_estimation key=value ...`` calls, so the two cannot disagree. It validates its keywords into a
:class:`DepthEstimationConfig` and hands that to :class:`DepthEstimationTrainer`, a :class:`~train.api.trainer.TaskTrainer`
that supplies the hooks of this task and nothing else (a task overrides hooks, never stages). One family trains it,
``depth_anything`` (``python -m train list_models task=depth_estimation``), on MegaDepth read in place.

Interface contract
------------------
``DepthEstimationConfig``
    Frozen :class:`~train.api.config.StrictConfig`; one field per keyword of the function. ``out``, ``model`` and ``data``
    are required. ``data`` is :class:`~train.api.depth_args.DepthData`; ``optim_args``, ``transform_args`` and
    ``metric_args`` are the depth groups of :mod:`train.api.depth_args`. ``precision`` other than ``float32``,
    ``analysis_args.conformal_alpha`` and ``checkpoint_args.backup_every_steps`` are refused by name.

``DepthEstimationTrainer(config)``
    :class:`~train.api.trainer.TaskTrainer` with ``task = "depth_estimation"`` and ``success_metric = "val_loss"``.

    What a run is. An epoch is ``data.steps_per_epoch`` batches of an INFINITE train stream (each batch a pure function of
    the stream seed and its global index, :mod:`train.common.depth_data`), so ``fit_kwargs`` gives ``steps_per_epoch`` and
    ``validation_steps``. The selection and test crops are fixed by ``(seed, salt, pool index)`` and come from scenes that
    never train; selection crops pick the best epoch and set the bar, test crops are scored once after ``fit``. A resume
    trains on the stream of ``(seed, completed steps)`` (``resume_stream``) and never moves the fixed crops.

    The bar (D-006). ``success_value`` is the minimum selection ``val_loss`` from the selection start (the rule of the
    best checkpoint). The threshold is the minimum of the constant baseline and the average layout baseline, both
    ``val_loss`` of a weightless predictor under the compiled loss on the selection crops (measured in ``build_data``),
    and of the untrained model's ``val_loss``, read from the fit sidecar (so a resume reads the same number).
    ``bar_is_met`` is strictly below in float32 beyond a tie band (:func:`train.common.forecast_run.beats_bar`). There is
    no override field.

    The optimizer (D-003). From a pretrained start ``optimizer_options`` gives every trainable variable of the DPT head
    10x the encoder rate (the legacy table, ``decoder_lr_multipliers``); from random weights it gives nothing.

    Before the claim (D-007). ``check_model_args`` runs the family resolver: a refused or unknown model key, an optimizer
    other than adamw on a pretrained start and a missing official DINOv2 weights file are refused there.

    Reporting. ``evaluate_and_report`` reads the epoch-zero term, scores the best weights on the test crops
    (``test_metrics_best`` and the ``held_out`` block of :func:`train.common.depth_run.held_out_report`), checks that
    ``best_model.keras`` reloads to the same outputs, and runs the data-free analysis and the last-epoch report
    (``report_data_free_run``). The summary's ``baselines``, ``epoch_zero`` and ``held_out`` blocks come back through
    ``Evaluation.extras``.

    The record (D-009). ``task_record`` is the depth ``task.json`` (schema in :mod:`train.api.predictor`): the size the
    model is built at, 3 channels, ``normalize`` ``imagenet`` and ``output`` ``relative_disparity``, written before
    ``fit``. ``predict`` of the run directory resizes each image to that size and the map back
    (:class:`~train.api.depth_predictor.DepthPredictor`).

``train_depth_estimation(*, out, model, data, ...)``
    :param out: run directory (a ``str`` or a ``Path``), or ``"auto"``.
    :param model: ``"depth_anything/<variant>"``, e.g. ``"depth_anything/vit_s"``.
    :param data: a mapping (``{"root": "/data/megadepth"}``), a :class:`~train.api.depth_args.DepthData`, or the path of a
        YAML file holding one.
    :param epochs: / ``batch_size`` / ``devices`` / ``seed`` / ``precision`` / ``resume_interrupted`` / ``checkpoint`` /
        ``model_args`` and the groups: as in :func:`~train.api.tasks.image_classification.train_image_classification`.
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigValidationError: for every bad key, listed together.
    :raises ConfigError: for a request that cannot be honoured, before the run directory is claimed where it can be seen
        without data: a missing root, fewer than 3 validation scenes, a missing weights file, a refused model key.
    :raises RuntimeError: for a diverged or non-finite run.

Import rule
-----------
Module scope imports only the standard library and ``train.api``: ``python -m train depth_estimation help`` imports this
module and must stay free of TensorFlow. Every ``keras`` and ``train.common`` name is imported inside a method.
"""

import dataclasses
import functools
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .. import recipes, registry
from ..args import AnalysisArgs, AutoContext, CheckpointArgs, DebugArgs, LoggerArgs
from ..config import (
    AUTO,
    DEFAULT_SEED,
    PATH_FORM,
    RUN_HELP,
    Auto,
    Precision,
    StrictConfig,
    join_path,
    resume_help,
    run_request,
    validate,
)
from ..data_args import DataFacts
from ..depth_args import (
    BUILTIN,
    TASK,
    WATCH_METRIC,
    DepthData,
    DepthMetricArgs,
    DepthOptimArgs,
    DepthTransformArgs,
    depth_facts,
    load_depth_data_config,
    pretrained_start,
    read_depth_data_file,
)
from ..depth_models import JIT_COMPILE
from ..predictor import build_depth_record, overrides_of_variant
from ..run_dir import CHECKPOINTS_DIR
from ..trainer import Evaluation, FitState, TaskTrainer, TrainResult, check_holds_no_classes, config_problems

# ---------------------------------------------------------------------

#: The compiled metrics, by name only (``metric_names`` must build nothing); a test holds them equal to the objects
#: ``train.common.depth_run.loss_and_metrics`` builds.
METRICS: Tuple[str, ...] = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta_1.25")

_RESUME_ORDER = (
    "the train crops continue on a fresh stream derived from the seed and the steps completed, never epoch 0's crops "
    "and never a selection or test scene, and the selection and test crops stay fixed"
)

_CONFORMAL = (
    f"a conformal prediction set is a set of classes, which a {TASK} run does not produce; its output is relative "
    "disparity per pixel"
)


@dataclasses.dataclass(frozen=True)
class DepthEstimationConfig(StrictConfig):
    """Everything a depth run is told; see the module contract."""

    out: str = dataclasses.field(metadata={"help": RUN_HELP["out"]})
    model: str = dataclasses.field(
        metadata={"help": "family/variant, e.g. depth_anything/vit_s; python -m train list_models task=depth_estimation "
                          "lists them"})
    data: DepthData = dataclasses.field(
        metadata={
            "help": "MegaDepth and the counts: data.root (read in place), data.n_select and data.n_test (the fixed "
            "selection and test crops), data.layout_samples, data.steps_per_epoch; or the path of a YAML file holding "
            "them (data=data.yaml), over which dotted keys win",
            PATH_FORM: read_depth_data_file,
        }
    )
    epochs: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["epochs"]})
    batch_size: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["batch_size"]})
    seed: int = dataclasses.field(default=DEFAULT_SEED, metadata={"help": RUN_HELP["seed"]})
    devices: Auto[Optional[int]] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["devices"]})
    precision: Precision = dataclasses.field(
        default="float32",
        metadata={"help": "float32, the only value accepted: the masked median of the loss was never measured in "
                          "float16 or bfloat16"},
    )
    resume_interrupted: bool = dataclasses.field(
        default=False,
        metadata={"help": resume_help(
            "an epoch of 100 batches takes about half a minute on a GPU", order=_RESUME_ORDER,
            backup="this task refuses checkpoint_args.backup_every_steps, so a resume restarts the killed epoch from "
                   "the backup taken at the last epoch end; a backup inside an epoch would make Keras fail the next "
                   "epoch of this task's train stream",
        )},
    )
    checkpoint: Optional[str] = dataclasses.field(default=None, metadata={"help": RUN_HELP["checkpoint"]})
    model_args: Dict[str, Any] = dataclasses.field(
        default_factory=dict,
        metadata={"help": RUN_HELP["model_args"] + "; model_args.pretrained (default true) starts the encoder from the "
                          "official DINOv2 weights"},
    )
    optim_args: DepthOptimArgs = dataclasses.field(default_factory=DepthOptimArgs)
    transform_args: DepthTransformArgs = dataclasses.field(default_factory=DepthTransformArgs)
    metric_args: DepthMetricArgs = dataclasses.field(default_factory=DepthMetricArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)

    def problems(self, prefix: str) -> List[str]:
        """Name the fields out of range, a conformal level, a precision other than float32 and a mid-epoch backup."""
        lines = config_problems(self, prefix, no_conformal_for=TASK, because=_CONFORMAL)
        if self.precision != "float32":
            lines.append(
                f"Invalid value for {join_path(prefix, 'precision')!r}: expected float32 for {TASK} (the masked median "
                f"of the affine-invariant loss was never measured in a mixed precision), got {self.precision!r}"
            )
        # DECISION plan-2026-11-17T110935-9c7b9096/D-031: a mid-epoch resume cuts an epoch of the PyDataset stream and
        # Keras 3.8 raises at the next one. Do NOT lift this while the Keras pin in test_depth_task.py (test_keras_fails_*)
        # passes; do NOT change the shared FinishKilledEpoch for it (five tf.data tasks rely on it).
        if self.checkpoint_args.backup_every_steps is not None:
            lines.append(
                f"Invalid value for {join_path(prefix, 'checkpoint_args.backup_every_steps')!r}: expected null for "
                f"{TASK}, got {self.checkpoint_args.backup_every_steps!r}. A resume from a backup taken inside an epoch "
                "ends that epoch early, and Keras then fails the next epoch of this task's train stream (a "
                "keras.utils.PyDataset); a resume of this task uses the backup taken at each epoch end"
            )
        return lines


class DepthEstimationTrainer(TaskTrainer):
    """The hooks of a depth run; see the module contract.

    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    :cvar resume_order: what a resume does to the train crops, for the summary's resume note.
    """

    task = TASK
    success_metric = WATCH_METRIC
    resume_order = _RESUME_ORDER
    initial_loss_hint = "Lower optim_args.lr, or check the model arguments and the MegaDepth root."
    #: ``{constant, layout}``: the two baselines on the selection crops, set by ``build_data`` (a resume builds them again;
    #: they are deterministic). ``None`` until then, also on a bare trainer in a test.
    _baselines: Optional[Dict[str, float]] = None
    #: The untrained model's ``val_loss``, read from the fit sidecar by :meth:`read_epoch_zero` and by nothing else.
    _epoch_zero: Optional[float] = None
    #: Batches of the selection crops, set by ``build_data``: the width of the bar's tie band follows it (``None`` reads as
    #: one batch, the narrowest band).
    _select_batches: Optional[int] = None
    #: The adapter keywords of the built model, set by ``build_model`` (``optimizer_options`` reads its start).
    _built_kwargs: Optional[Dict[str, Any]] = None

    # -- data and model -------------------------------------------------

    def describe_data(self, config: DepthEstimationConfig) -> DataFacts:
        return depth_facts(config.data)

    def build_data(self, config: DepthEstimationConfig, context: AutoContext) -> Any:
        from train.common import depth_data, depth_run

        data_cfg: DepthData = config.data
        transform_cfg: DepthTransformArgs = config.transform_args
        if data_cfg.format != BUILTIN:
            # `format` is a Literal of one value today; a second source adds its loader here, not a silent fall-through.
            raise ValueError(f"data.format {data_cfg.format!r} has no loader in the depth task: only {BUILTIN!r} does")
        caps = [cap for cap in (data_cfg.max_train_files, data_cfg.max_samples) if cap is not None]
        # Every field is read by name (the fields-are-read guard sees an attribute, never a `getattr` over the fields).
        data = depth_data.load_depth_batches(
            data_cfg.root, seed=config.seed, batch_size=config.batch_size, image_size=transform_cfg.image_size,
            steps_per_epoch=data_cfg.steps_per_epoch, n_select=data_cfg.n_select, n_test=data_cfg.n_test,
            layout_samples=data_cfg.layout_samples, min_valid_ratio=data_cfg.min_valid_ratio,
            max_train_files=min(caps) if caps else None, augment=transform_cfg.augment == "flip",
            workers=data_cfg.workers, train_split=data_cfg.train_split,
        )
        self._baselines = depth_run.selection_baselines(data)
        self._select_batches = data.n_select // config.batch_size
        return data

    def input_check(self, facts: DataFacts, data: Any) -> None:
        check_holds_no_classes(self, facts, data)

    def build_model(self, config: DepthEstimationConfig, context: AutoContext, data: Any) -> Any:
        # The adapter returns a model BUILT at (None, s, s, 3), so the sequence's own `build` call is skipped.
        kwargs = self._model_kwargs(config, context)
        model = recipes.build_model(registry.MODELS[context.family], context.variant, None, data.input_shape, kwargs)
        self._built_kwargs = kwargs
        return model

    @staticmethod
    def _model_kwargs(config: DepthEstimationConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the family adapter's keywords: the recipe's, then the caller's (the resolver refuses the rest)."""
        return registry.MODELS[context.family].load_resolver()(context, config.optim_args, config.model_args)

    def task_record(self, config: DepthEstimationConfig, context: AutoContext, data: Any) -> Dict[str, Any]:
        """Return the depth ``task.json`` record: the size the model is built at (``data.input_shape``, which
        ``build_model`` builds it at) and its channels; nothing of the model is needed, so a resume asks for it before
        ``build_model``."""
        height, width, channels = data.input_shape
        table = registry.MODELS[context.family].load_variants()
        return build_depth_record(
            model=config.model, family=context.family, variant=context.variant, input_size=(height, width),
            channels=channels,
            variant_overrides=overrides_of_variant(table[context.variant], self._model_kwargs(config, context)),
        )

    def used_arguments(self, config: DepthEstimationConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json``: the model keywords."""
        return {"model_args": recipes.serialize_model_args(self._model_kwargs(config, context))}

    def check_model_args(self, config: DepthEstimationConfig, context: AutoContext) -> None:
        # DECISION plan-2026-11-17T110935-9c7b9096/D-007
        # The family resolver runs HERE, before the claim: a refused or unknown model key, an optimizer other than adamw on
        # a pretrained start and a missing official DINOv2 file are refused with no run directory. Do NOT leave the weights
        # to `build_model` (the directory is claimed by then, and `from_variant` raises inside the build), and do NOT fall
        # back to a random encoder.
        self._model_kwargs(config, context)

    # -- compile --------------------------------------------------------

    def metric_names(self, config: DepthEstimationConfig, facts: DataFacts) -> List[str]:
        # DECISION plan-2026-10-30T132432-a865fa08/D-055: names only; a Keras metric allocates variables and the first eager
        # op would initialise TensorFlow's devices before setup_gpu.
        return list(METRICS)

    def build_loss_and_metrics(self, config: DepthEstimationConfig, data: Any) -> Tuple[Any, List[Any]]:
        from train.common import depth_run

        return depth_run.loss_and_metrics()

    def compile_options(self, config: DepthEstimationConfig, spec: registry.ModelSpec) -> Dict[str, Any]:
        # `jit_compile` is passed EXPLICITLY (D-016, anchored at the constant): Keras resolves "auto" to XLA on a GPU.
        return {"jit_compile": JIT_COMPILE}

    def optimizer_options(self, config: DepthEstimationConfig, model: Any) -> Dict[str, Any]:
        """Return the head's 10x table from a pretrained start, nothing from random weights."""
        from train.depth_anything.train_depth_anything import decoder_lr_multipliers

        if self._built_kwargs is None:
            raise RuntimeError("optimizer_options reads the start build_model resolved; it was asked for before it")
        # DECISION plan-2026-11-17T110935-9c7b9096/D-003
        # The legacy table (every trainable variable of the DPT head at 10x, from a pretrained start only) reaches the ONE
        # optimizer the sequence builds, through this seam. Do NOT build an optimizer in the task, and do NOT give a random
        # start the 10x (the paper's 10x is for a random head on a pretrained encoder). An empty table is no option.
        table = decoder_lr_multipliers(model, pretrained_start(self._built_kwargs))
        return {"lr_multipliers": table} if table else {}

    def initial_loss_check(
        self, config: DepthEstimationConfig, model: Any, data: Any
    ) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return the epoch-0 numbers; the ratio is the loss over the constant baseline's, and no verdict is given.

        :raises RuntimeError: for a non-finite loss (the model diverges before training).
        """
        from dl_techniques.utils.logger import logger
        from train.common import classification_run, run_summary

        if self._baselines is None:
            raise RuntimeError(
                "initial_loss_check compares with the constant baseline, which build_data works out; it was asked for first")
        metrics = classification_run.evaluate_dict(model, data.select_ds)
        constant = self._baselines["constant"]
        ratio = run_summary.initial_loss_ratio(metrics, constant, self.initial_loss_hint)
        logger.info(
            f"Untrained evaluate BEFORE fit: val loss {metrics['loss']:.6g} on {data.n_select} selection crops, "
            f"{ratio:.3g}x the constant baseline's {constant:.6g}"
        )
        return metrics, ratio, None

    def unjudged_note(self, config: DepthEstimationConfig) -> str:
        return (
            "`init_scale_warning` is null (not judged): `initial_loss_ratio` is the loss of the untrained model over the "
            "constant baseline's (1.0 under the masked affine-invariant loss) on the same fixed selection crops. A random "
            "DPT head on the pretrained encoder sat at 1.55 times it (measured) and then trained, so no threshold on that "
            "ratio separates a well-scaled model from a mis-scaled one; a non-finite initial loss is refused"
        )

    # -- the two seams of the shared sequence ---------------------------

    def fit_kwargs(self, config: DepthEstimationConfig, data: Any) -> Dict[str, Any]:
        # The train stream is INFINITE: an epoch is `steps_per_epoch` batches by this keyword (the sequence refuses an infinite
        # stream without it); `validation_steps` is the selection stream's one pass.
        return {"steps_per_epoch": data.steps_per_epoch, "validation_steps": data.n_select // data.batch_size}

    def resume_stream(self, config: DepthEstimationConfig, data: Any, completed_steps: int) -> Optional[Any]:
        # DECISION plan-2026-11-17T110935-9c7b9096/D-004
        # A resumed `fit` asks the stream for global index 0 again, so the stream the fresh process built would replay the
        # first batches. `with_stream` keeps the fixed crops and the layout and moves only the train stream, to the seed
        # derived from `(seed, completed_steps)`. Do NOT return `data` (that is the replay) and do NOT reseed the fixed sets.
        return data.with_stream(completed_steps)

    # -- the verdict ----------------------------------------------------

    def read_epoch_zero(self, run_dir: Union[str, Path]) -> float:
        """Read the untrained model's ``val_loss`` from the run's fit sidecar, keep it for the threshold and return it.

        :raises ValueError: when the sidecar holds no ``initial.metrics.loss``.
        """
        from train.common import pretrain_run
        from train.common.resume_state import FIT_STATE_NAME

        # DECISION plan-2026-11-17T110935-9c7b9096/D-006
        # The epoch-zero term is read from the SIDECAR, always: a resume never runs `initial_loss_check`, and evaluating
        # the weights a resume restored would measure the trained model. Do NOT cache it on the trainer from the first
        # process's evaluate.
        self._epoch_zero = pretrain_run.epoch_zero_val_loss(Path(run_dir) / CHECKPOINTS_DIR / FIT_STATE_NAME)
        return self._epoch_zero

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        """Return the best (the lowest) ``val_loss`` of the selection window."""
        start = self.selection_start_epoch(self.config)
        return float(min(hist[self.success_metric][start:]))

    def success_threshold(self, config: DepthEstimationConfig) -> float:
        """Return ``min(constant, layout, epoch-zero)``; ``evaluate_and_report`` has read the third term."""
        if self._baselines is None:
            raise RuntimeError(
                "success_threshold is cut from the baselines (constant and layout on the selection crops), which "
                "build_data works out; it was asked for before the data was built"
            )
        if self._epoch_zero is None:
            raise RuntimeError(
                "success_threshold needs the epoch-zero val_loss, which evaluate_and_report reads from the fit sidecar; "
                "it was asked for before the reporting"
            )
        # DECISION plan-2026-11-17T110935-9c7b9096/D-006
        # The MINIMUM of the three terms, all on the selection crops: a constant output scores the first, an average-layout
        # output the second (the from-scratch model's measured end point), and the untrained model the third. Do NOT drop
        # a term because it "never binds" (Step 1 measured the constant binding, a smaller tree can make the layout bind).
        return float(min(self._baselines["constant"], self._baselines["layout"], self._epoch_zero))

    def bar_is_met(self, config: DepthEstimationConfig, reached: float, threshold: float) -> bool:
        from train.common import forecast_run

        # DECISION plan-2026-11-17T110935-9c7b9096/D-006
        # STRICTLY below, in float32, beyond the tie band of the selection batch count (`forecast_run.beats_bar`): the base
        # class's `>=` reads a loss backwards, and `<=` would pass a constant model, which scores the constant baseline.
        # Do NOT compare the Python floats and do NOT shrink the band.
        return forecast_run.beats_bar(reached, threshold, self._select_batches or 1)

    # -- the log and the summary ---------------------------------------

    def epoch_line_metrics(self, config: DepthEstimationConfig) -> Tuple[str, ...]:
        return ("loss", "val_loss", "val_abs_rel")

    def summary_fields(self, config: DepthEstimationConfig, data: Any) -> Dict[str, Any]:
        return {
            "image_size": data.image_size,
            "scenes": {"train": len(data.train_scenes), "select": len(data.select_scenes), "test": len(data.test_scenes)},
            "train_files": len(data.train_rgb),
            "stream": {
                "seed": data.seed,
                "stream_seed": data.stream_seed,
                "rule": (
                    "the selection and test crops are keyed on (seed, salt, pool index) in scenes that never train and "
                    "never move; train batch i on (stream_seed, i); a resume derives stream_seed from (seed, completed "
                    "steps), the first 4 bytes of sha256"
                ),
            },
        }

    def task_notes(self, config: DepthEstimationConfig, baseline: Dict[str, float], ratio: float, data: Any) -> List[str]:
        """Return the depth lines of ``results_summary.json``'s ``notes``."""
        from train.common import classification_run

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        notes = [
            f"initial loss {baseline['loss']:.6g} is {ratio:.3g}x the constant baseline on the {data.n_select} fixed "
            "selection crops: a plain evaluate in inference mode; see the `init_scale_warning` note",
            *classification_run.RUN_NOTES[:2],
            "`val_loss` is the masked affine-invariant loss of Depth Anything V1 on the fixed selection crops (a function of "
            "the weights alone); the compiled `abs_rel`, `sq_rel`, `rmse`, `rmse_log` and `delta_1.25` score the prediction "
            "after a per-crop scale and shift, on the normalized-disparity scale, not the paper's metric depth",
            "`success_value` is the lowest `val_loss` of the selection window; `success_threshold` is min(`baselines.constant`, "
            "`baselines.layout`, `epoch_zero.val_loss`), the first two the loss of a constant and of the average layout map "
            "on the same selection crops, the third read from the fit sidecar; the bar is STRICTLY below, compared in "
            "float32 beyond `train.common.forecast_run.tie_band` float32 steps",
            "the selection crops select and are not a held-out estimate; `held_out` scores the best weights on the test "
            "crops of other scenes, once: loss, per-crop Pearson over valid pixels, the constant and the layout, and the "
            "paired model-minus-layout difference with its standard error",
            "`task.json` records the size the model is built at: `predict` resizes each image to it, standardizes it as "
            "training did and resizes the map back to the image's size; a map is relative disparity, defined up to scale "
            "and shift per image; `export` writes keras, saved_model, tflite or onnx at that size only (another size is "
            "refused in saved_model, tflite and onnx; a keras file is the trained model itself and ignores input_shape), "
            "a file that takes ImageNet-standardized images of that size and returns the map at it",
        ]
        notes.append(
            "the model analysis is data-free (weights and spectral only; `analyzer.analyzers` lists them): the stock "
            "analyses read class probabilities, which a disparity map is not"
            if analysis_cfg.model_analysis else "analysis_args.model_analysis is false: `analyzer.status` is 'skipped'"
        )
        if not checkpoint_cfg.save_best:
            notes.append("checkpoint_args.save_best is false: no best_model.keras was written, so the reload check did not run")
        if not checkpoint_cfg.save_last:
            notes.append("checkpoint_args.save_last is false: no final_model.keras was written and `model_loading_validated` is null")
        return notes

    # -- the figures ----------------------------------------------------

    def figure_callbacks(self, config: DepthEstimationConfig, data: Any, vis_dir: Path) -> List[Any]:
        """Return the prediction grid of the first fixed selection crops (rgb, target, aligned prediction)."""
        from train.common import depth_run

        return [depth_run.prediction_grid(data, vis_dir)]

    # -- the reporting --------------------------------------------------

    def run_analysis(
        self,
        config: DepthEstimationConfig,
        model: Any,
        test_data: Tuple[Any, Any],
        history: Any,
        experiment_name: str,
        run_dir: str,
    ) -> Any:
        """Run the data-free analysis (weights and spectral only); see the module contract."""
        from train.common import run_summary

        # DECISION plan-2026-11-17T110935-9c7b9096/D-017
        # The stock analysis reads the output as class probabilities and the target as labels; a disparity map is neither.
        # The data-free analysis completes on vit_s (measured, 39 s on CPU). Do NOT fall back to the default hook.
        return run_summary.data_free_model_analysis(model, test_data, history, experiment_name, run_dir, config.seed)

    def evaluate_and_report(self, config: DepthEstimationConfig, data: Any, fit: FitState) -> Evaluation:
        """Run the depth reporting sequence; see the module contract."""
        from types import SimpleNamespace

        from train.common import classification_run, depth_run, evaluation, run_summary
        from train.common.callbacks import best_checkpoint_path

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        baselines = self._baselines
        if baselines is None:
            raise RuntimeError("evaluate_and_report needs the baselines build_data works out; the data was not built")
        epoch_zero = self.read_epoch_zero(fit.run_dir)
        # The in-memory model holds the BEST weights here (EarlyStopping restored them): everything about them is taken
        # before `report_data_free_run` puts the last epoch's weights in.
        sample_x = data.x_select[: data.batch_size]
        sample_y = data.y_select[: data.batch_size]
        test_best = classification_run.evaluate_dict(fit.model, data.test_ds)
        held_out = depth_run.held_out_report(fit.model, data)
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        if checkpoint_cfg.save_best:
            # `check_model_loading` reloads with compile=False (the loss and the metrics are custom objects); the crops are
            # fixed, so the in-memory best weights and the file give the same outputs.
            expected = fit.model.predict(sample_x, batch_size=data.batch_size, verbose=0)
            checked = evaluation.check_model_loading(
                best_checkpoint_path(str(fit.run_dir)), sample_x, expected, batch_size=data.batch_size,
                relative_tolerance=evaluation.LOAD_RTOL, reference_model=fit.model,
            )
            max_diff = checked.max_abs_diff
            if not checked.ok:
                load_error = f"best_model.keras reloads to other outputs than the in-memory best weights: {checked.as_record()}"
        analysis, test_final, load_check, load_detail = run_summary.report_data_free_run(
            fit.model,
            SimpleNamespace(x_fig=sample_x, y_fig=sample_y),
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
            load_sample=sample_x,
            predict_batch_size=data.batch_size,
        )
        visualizations: Dict[str, Any] = {"files": [], "failed": [], "skipped": not analysis_cfg.figures}
        if analysis_cfg.figures and fit.vis_dir.is_dir():
            visualizations["files"] = sorted(path.name for path in fit.vis_dir.iterdir() if path.is_file())
        terms = min(baselines["constant"], baselines["layout"])
        extras: Dict[str, Any] = {
            "baselines": dict(baselines),
            "epoch_zero": {"val_loss": epoch_zero, "binds": bool(math.isfinite(epoch_zero) and epoch_zero < terms)},
            "held_out": held_out,
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


def train_depth_estimation(
    *,
    out: Union[str, Path],
    model: str,
    data: Union[DepthData, Mapping[str, Any], str, Path],
    epochs: Union[int, str] = AUTO,
    batch_size: Union[int, str] = AUTO,
    seed: int = DEFAULT_SEED,
    devices: Union[int, str, None] = AUTO,
    precision: str = "float32",
    resume_interrupted: bool = False,
    checkpoint: Union[str, Path, None] = None,
    model_args: Optional[Mapping[str, Any]] = None,
    optim_args: Union[DepthOptimArgs, Mapping[str, Any], None] = None,
    transform_args: Union[DepthTransformArgs, Mapping[str, Any], None] = None,
    metric_args: Union[DepthMetricArgs, Mapping[str, Any], None] = None,
    checkpoint_args: Union[CheckpointArgs, Mapping[str, Any], None] = None,
    logger_args: Union[LoggerArgs, Mapping[str, Any], None] = None,
    analysis_args: Union[AnalysisArgs, Mapping[str, Any], None] = None,
    debug_args: Union[DebugArgs, Mapping[str, Any], None] = None,
) -> TrainResult:
    """Train a relative-disparity model on MegaDepth; see the module contract."""
    recipes.register_builtin_models()
    if isinstance(data, (str, Path)):
        data = load_depth_data_config(data)
    given = run_request(
        out=out, model=model, data=data, epochs=epochs, batch_size=batch_size, seed=seed,
        devices=devices, precision=precision, resume_interrupted=resume_interrupted,
        checkpoint=checkpoint, model_args=model_args,
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
    config = validate(DepthEstimationConfig, given)
    return DepthEstimationTrainer(config).run()
