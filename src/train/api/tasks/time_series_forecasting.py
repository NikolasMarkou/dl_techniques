"""The time_series_forecasting task: a config, a trainer and the keyword-only function.

``train_time_series_forecasting(...)`` is the function a Python caller uses and the function
``python -m train time_series_forecasting key=value ...`` calls, so the two cannot disagree. It validates its keywords into a
:class:`TimeSeriesForecastingConfig` and hands that to :class:`TimeSeriesForecastingTrainer`, a
:class:`~train.api.trainer.TaskTrainer` that supplies the hooks of this task and nothing else (a task overrides hooks, never
stages). Seven families train it (``python -m train list_models task=time_series_forecasting``); ``deepar`` and
``adaptive_ema`` are refused by name (:data:`train.api.forecast_models.UNSUPPORTED`).

Interface contract
------------------
``TimeSeriesForecastingConfig``
    Frozen :class:`~train.api.config.StrictConfig`; one field per keyword of the function. ``out``, ``model`` and ``data``
    are required. ``data`` is :class:`~train.api.forecast_args.TimeSeriesData` (the synthetic bank or one csv, the window,
    the split by time); ``transform_args``, ``metric_args`` and ``loss_args`` are the forecasting groups; ``jit_compile``
    (default false) opts into XLA. ``analysis_args.conformal_alpha`` is refused (a conformal set is a set of classes).

``TimeSeriesForecastingTrainer(config)``
    :class:`~train.api.trainer.TaskTrainer` with ``task = "time_series_forecasting"`` and ``success_metric = "val_mae"``.

    What a run is. An epoch is ``data.steps_per_epoch`` steps of an INFINITE train stream (windows keyed on
    ``(stream seed, counter)``), so ``fit_kwargs`` gives ``steps_per_epoch`` and ``validation_steps``. The select and test
    windows are fixed by ``(seed, salt, index)``, so ``val_mae`` is a function of the weights alone; a resume replaces the train
    stream (``resume_stream``) by one derived from ``(seed, completed steps)`` and never moves the split.

    The bar (D-008, D-037). ``success_value`` is the minimum ``val_mae`` of the selection window; the threshold is the minimum
    of the MAE of naive-last, of seasonal-naive and of the untrained model, all three on the SAME fixed select windows (the
    baselines are numpy, measured in ``build_data``; the untrained model's is read from the fit sidecar in
    ``evaluate_and_report``); ``bar_is_met`` is strictly below in float32 with a tie band
    (:func:`train.common.forecast_run.beats_bar`). There is no override field. Under ``debug_args.smoke`` no bar is applied.

    Reporting. ``evaluate_and_report`` scores the best weights on the test windows, checks that ``best_model.keras`` reloads
    to the same outputs, computes the canonical post-hoc metrics (MAE, RMSE, sMAPE, rMAE, MASE from the context, coverage and
    sharpness for a quantile head) on the select and test windows through the family's decode row, draws one forecast
    figure, runs the data-free analysis and scores, saves and reloads the last epoch (``report_data_free_run``). The
    summary's ``baselines``, ``epoch_zero`` and ``canonical_metrics`` blocks come back through ``Evaluation.extras``.

    The run writes ``task.json`` before ``fit`` (``task_record``: the window, the channels, the scale floor, the head, the
    levels and the decode row of the model, where the data came from), so ``predict`` loads a
    :class:`~train.api.forecast_predictor.ForecastPredictor` from the run directory.

``train_time_series_forecasting(*, out, model, data, ...)``
    :param out: run directory (a ``str`` or a ``Path``), or ``"auto"``.
    :param model: ``"family/variant"``, e.g. ``"nbeats/base"``; ``python -m train list_models task=time_series_forecasting``
        lists them.
    :param data: a mapping (``{"lookback": 168, "horizon": 24}``), a :class:`~train.api.forecast_args.TimeSeriesData`, or the
        path of a YAML file holding one.
    :param jit_compile: compile with XLA (false by default: one graph on every device); refused for ``ets``.
    :param epochs: / ``batch_size`` / ``devices`` / ``seed`` / ``precision`` / ``resume_interrupted`` / ``checkpoint`` /
        ``model_args`` and the groups: as in :func:`~train.api.tasks.image_classification.train_image_classification`.
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigValidationError: for every bad key, listed together.
    :raises ConfigError: for a request that cannot be honoured; see :meth:`~train.api.trainer.TaskTrainer.run`. A family that
        does not train this task, a model key a family refuses, a second channel for ``tirex``, ``ets`` and ``mdn``, a csv or
        a series too short for one window, a loss name for a head whose loss is fixed and ``jit_compile`` for ``ets`` are
        refused before the run directory is claimed.
    :raises RuntimeError: for a diverged or non-finite run.

Import rule
-----------
Module scope imports only the standard library and ``train.api``: ``python -m train time_series_forecasting help`` imports
this module and must stay free of TensorFlow. Every ``keras`` and ``train.common`` name is imported inside a method.
"""

import dataclasses
import functools
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .. import recipes, registry
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
from ..errors import ConfigError, ConfigValidationError
from ..forecast_args import (
    TASK,
    WATCH_METRIC,
    ForecastLossArgs,
    ForecastMetricArgs,
    ForecastTransformArgs,
    TimeSeriesData,
    forecast_facts,
    load_forecast_data_config,
    read_forecast_data_file,
)
from ..forecast_models import (
    ETS,
    FAMILIES,
    NBEATS,
    check_supported_family,
    flat_target,
    geometry_problems,
    jit_problem,
    loss_problem,
)
from ..forecast_series import check_csv_level, read_csv_series
from ..predictor import build_forecast_record, overrides_of_variant
from ..run_dir import CHECKPOINTS_DIR
from ..trainer import Evaluation, FitState, TaskTrainer, TrainResult, config_problems

# ---------------------------------------------------------------------

#: The name of the compiled metric (``val_<name>`` is the watched key): derived, so the two cannot disagree.
METRIC = WATCH_METRIC.removeprefix("val_")

_RESUME_ORDER = (
    "the training windows continue on a fresh stream derived from the seed and the steps completed, never epoch 0's "
    "windows and never a selection or test window, and the selection and test windows stay fixed"
)

# DECISION plan-2026-11-13T105600-3a8319ef/D-016
# A conformal set is a set of classes; the interval forecast of this task is a quantile head. The shared sentence is about
# images, so this task hands `config_problems` a clause of its own. Do NOT reuse the image sentence here.
_CONFORMAL = (
    f"a conformal prediction set is a set of classes, which a {TASK} run does not produce; the interval forecast of this "
    "task is a quantile head (model_args.quantile_levels)"
)

#: ``(level, text)`` lines the recipe owes a family's user.
_NOTICES: Dict[str, Tuple[str, str]] = {
    NBEATS: (
        "info",
        "nbeats trains the forecast branch only (the residual output has no loss, D-007): Keras warns that two kernels "
        "receive no gradient, which is expected (measured, D-024)",
    ),
    ETS: (
        "info",
        "ets has one smoothing parameter per component: its untrained forecast can already beat naive-last, so the bar "
        "(the minimum of three terms) can mean 'beat the untrained model' for it (D-029)",
    ),
}


@dataclasses.dataclass(frozen=True)
class TimeSeriesForecastingConfig(StrictConfig):
    """Everything a forecasting run is told; see the module contract."""

    out: str = dataclasses.field(metadata={"help": RUN_HELP["out"]})
    model: str = dataclasses.field(
        metadata={
            "help": "family/variant, e.g. nbeats/base or prism/tiny; python -m train list_models "
            "task=time_series_forecasting lists them"
        }
    )
    data: TimeSeriesData = dataclasses.field(
        metadata={
            "help": "the series and the window: data.lookback and data.horizon (auto: the family recipe's), the synthetic "
            "bank (data.realizations, data.series_length) or a csv (data.format=csv data.path=<file>), the split by time "
            "(data.select_fraction, data.test_fraction); or the path of a YAML file holding them (data=data.yaml), over "
            "which dotted keys win (data=data.yaml data.max_samples=100)",
            PATH_FORM: read_forecast_data_file,
        }
    )
    epochs: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["epochs"]})
    batch_size: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["batch_size"]})
    seed: int = dataclasses.field(default=DEFAULT_SEED, metadata={"help": RUN_HELP["seed"]})
    devices: Auto[Optional[int]] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["devices"]})
    precision: Precision = dataclasses.field(default="float32", metadata={"help": RUN_HELP["precision"]})
    resume_interrupted: bool = dataclasses.field(
        default=False,
        metadata={"help": resume_help("an epoch of the synthetic bank takes seconds to minutes", order=_RESUME_ORDER)},
    )
    checkpoint: Optional[str] = dataclasses.field(default=None, metadata={"help": RUN_HELP["checkpoint"]})
    model_args: Dict[str, Any] = dataclasses.field(
        default_factory=dict,
        metadata={"help": RUN_HELP["model_args"] + "; model_args.quantile_levels gives prism and xlstm a quantile head"},
    )
    jit_compile: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "compile with XLA: false (the default) builds one graph on CPU and GPU, true is faster on a GPU "
            "(3.6x to 11.4x per step, measured) and is refused for ets, whose gradient XLA cannot compile"
        },
    )
    optim_args: OptimArgs = dataclasses.field(default_factory=OptimArgs)
    transform_args: ForecastTransformArgs = dataclasses.field(default_factory=ForecastTransformArgs)
    metric_args: ForecastMetricArgs = dataclasses.field(default_factory=ForecastMetricArgs)
    loss_args: ForecastLossArgs = dataclasses.field(default_factory=ForecastLossArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)

    def problems(self, prefix: str) -> List[str]:
        """Name the top-level fields out of range, a conformal level and a family this task refuses by name."""
        # `deepar` and `adaptive_ema` are refused here, the first place the model name is seen and before any model code is
        # imported; the family is not registered, so the shared parse would only say "unknown family".
        check_supported_family(self.model)
        return config_problems(self, prefix, no_conformal_for=TASK, because=_CONFORMAL)


class TimeSeriesForecastingTrainer(TaskTrainer):
    """The hooks of a forecasting run; see the module contract.

    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    :cvar resume_order: what a resume does to the training windows, for the summary's resume note (the help of
        ``resume_interrupted`` carries the same clause).
    """

    task = TASK
    success_metric = WATCH_METRIC
    resume_order = _RESUME_ORDER
    initial_loss_hint = "Lower optim_args.lr, or check the model arguments and the series."
    #: Batches of the fixed selection windows, set by ``build_data``; the width of the bar's tie band follows it (D-043). ``None``
    #: until the data is built, which a bare trainer in a test reads as one batch (the narrowest band, 16 float32 steps).
    _select_batches: Optional[int] = None

    def __init__(self, config: StrictConfig) -> None:
        super().__init__(config)
        #: ``{naive_last, seasonal_naive, context_mean, n_windows}`` MAE on the fixed select windows, set by ``build_data``
        #: (a resume builds the data again and the numbers are deterministic).
        self._baselines: Optional[Dict[str, float]] = None
        #: The untrained model's ``val_mae``, read from the fit sidecar by ``evaluate_and_report`` and by nothing else.
        self._epoch_zero: Optional[float] = None
        #: The built model and its decode row, set by ``build_model`` (``build_loss_and_metrics`` receives neither).
        self._model: Optional[Any] = None
        self._decoder: Optional[Any] = None

    # -- data and model -------------------------------------------------

    def describe_data(self, config: TimeSeriesForecastingConfig) -> DataFacts:
        return forecast_facts(config.data)

    def build_data(self, config: TimeSeriesForecastingConfig, context: AutoContext) -> Any:
        import time

        import numpy as np

        from dl_techniques.utils.logger import logger
        from train.common import forecast_data, forecast_run

        data_cfg: TimeSeriesData = config.data
        # Every field is read by name (the fields-are-read guard sees an attribute, never a `getattr` over the fields).
        data = forecast_data.load_forecast_batches(
            format=data_cfg.format, dataset=data_cfg.dataset, path=data_cfg.path, value_cols=data_cfg.value_cols,
            time_col=data_cfg.time_col, lookback=data_cfg.lookback, horizon=data_cfg.horizon,
            realizations=data_cfg.realizations, series_length=data_cfg.series_length, noise_level=data_cfg.noise_level,
            patterns=data_cfg.patterns, select_fraction=data_cfg.select_fraction, test_fraction=data_cfg.test_fraction,
            n_select=data_cfg.n_select, n_test=data_cfg.n_test, steps_per_epoch=data_cfg.steps_per_epoch,
            max_samples=data_cfg.max_samples, batch_size=config.batch_size, seed=config.seed, stream_seed=None,
            scale_floor=config.transform_args.scale_floor,
        )
        started = time.perf_counter()
        # DECISION plan-2026-11-13T105600-3a8319ef/D-008
        # The baselines are measured on the SAME fixed select windows the checkpoint is selected on (the dataset the compiled
        # metric scores), never on the test windows and never on another draw: a bar from another window set would say
        # nothing about the number it is compared with. Do NOT move this to the test split "to keep selection clean": the
        # bar is a learning-and-wiring check on the selection windows (D-004), the test windows are scored once.
        pairs = [(x.numpy(), y.numpy()) for x, y in data.select_ds]
        baselines = forecast_run.baseline_maes(np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs]))
        if not all(math.isfinite(baselines[key]) and baselines[key] > 0 for key in ("naive_last", "seasonal_naive")):
            # A ValueError: `run_sequence.load_data` turns it into a ConfigError that says `out` is now used.
            raise ValueError(
                f"the naive-last and seasonal-naive MAE of the {data.n_select} selection windows are "
                f"{baselines['naive_last']} and {baselines['seasonal_naive']}: a bar cut from them would be vacuous or "
                "undefined (the selection windows are flat or hold a non-finite value)"
            )
        self._baselines = baselines
        self._select_batches = data.n_select // config.batch_size
        if flat_target(context.family, self._model_kwargs(config, context)):
            # The pinball loss of tirex and of a quantile xlstm takes the target as (batch, horizon); the baselines above
            # were measured on the window layout, and the decode row reshapes either layout (`decode_dataset`).
            data = forecast_data.flatten_targets(data)
        logger.info(
            f"Baselines on the {data.n_select} fixed selection windows: naive-last {baselines['naive_last']:.6g}, "
            f"seasonal-naive {baselines['seasonal_naive']:.6g}, context mean {baselines['context_mean']:.6g} "
            f"({time.perf_counter() - started:.1f} s)"
        )
        return data

    def input_check(self, facts: DataFacts, data: Any) -> None:
        if data.num_classes is not None or facts.num_classes is not None:
            raise RuntimeError(
                f"{type(self).__name__} holds no classes, but describe_data said {facts.num_classes!r} and "
                f"build_data loaded {data.num_classes!r}"
            )

    def build_model(self, config: TimeSeriesForecastingConfig, context: AutoContext, data: Any) -> Any:
        from train.common import forecast_run

        # The adapter returns a model BUILT at (None, lookback, channels), so the sequence's own `build` call is skipped; a
        # static-batch family (timestral) is built at the run's batch.
        kwargs = {**self._model_kwargs(config, context), "horizon": config.data.horizon}
        if FAMILIES[context.family].static_batch:
            kwargs["batch_size"] = config.batch_size
        model = recipes.build_model(registry.MODELS[context.family], context.variant, None, data.input_shape, kwargs)
        self._model, self._decoder = model, forecast_run.decoder_for(context.family, model)
        return model

    @staticmethod
    def _model_kwargs(config: TimeSeriesForecastingConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the family adapter's keywords: the recipe's, then the caller's."""
        return registry.MODELS[context.family].load_resolver()(context, config.optim_args, config.model_args)

    def task_record(self, config: TimeSeriesForecastingConfig, context: AutoContext, data: Any) -> Dict[str, Any]:
        """Return the forecasting ``task.json`` record (schema in :mod:`train.api.predictor`).

        The head, the quantile levels, the median index and the decode row are the BUILT model's (:func:`train.common.forecast_run.decoder_for`:
        prism and xlstm choose theirs by ``model_args.quantile_levels``, tirex and timestral carry their own), so a resume,
        which asks for the record before ``build_model``, has the model built for it here.
        """
        # DECISION plan-2026-11-13T105600-3a8319ef/D-010
        # The levels and the median index are read off the model by the decode row (one rule for the record, the Predictor
        # and the metric). Do NOT recompute them from `model_args` here (tirex's and timestral's levels are the library's, and
        # a second rule would drift), and do NOT return a record without them on a resume: `check_data_unchanged` compares the
        # data keys only, but a record that is complete on one path and partial on another is the partial record the
        # denoising and pretrain decisions refuse. The sequence builds the model again right after; the extra build is the
        # price and its weights are never used (the backup restores them).
        if self._decoder is None:
            self.build_model(config, context, data)
        decoder = self._decoder
        table = registry.MODELS[context.family].load_variants()
        return build_forecast_record(
            model=config.model, family=context.family, variant=context.variant, lookback=data.lookback,
            horizon=data.horizon, channels=data.input_shape[1], value_cols=data.value_cols,
            scale_floor=config.transform_args.scale_floor, decode=decoder.row, quantile_levels=decoder.levels,
            point_index=decoder.median_index, data_format=config.data.format, data_sha256=data.csv_sha256,
            variant_overrides=(overrides_of_variant(table[context.variant], self._model_kwargs(config, context))
                               if isinstance(table, Mapping) else None),
        )

    def used_arguments(self, config: TimeSeriesForecastingConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json``: the model keywords."""
        return {"model_args": recipes.serialize_model_args(self._model_kwargs(config, context))}

    def check_model_args(self, config: TimeSeriesForecastingConfig, context: AutoContext) -> None:
        # Before the claim: the family's own refusals (a key it does not offer, a double normalization, an sgd decay), XLA for
        # a family that cannot take it, a loss name for a head with a fixed loss, and a channel count or horizon it cannot take.
        kwargs = self._model_kwargs(config, context)
        loss_cfg: ForecastLossArgs = config.loss_args
        problem = jit_problem(context.family, config.jit_compile) or loss_problem(context.family, kwargs, loss_cfg.name)
        if problem:
            raise ConfigError(problem)
        data_cfg: TimeSeriesData = config.data
        channels = 1
        if data_cfg.format == "csv":
            values, names, _ = read_csv_series(data_cfg.path, data_cfg.value_cols, data_cfg.time_col)
            # DECISION plan-2026-11-13T105600-3a8319ef/D-040
            # A column that varies by less than `scale_floor` of its level cannot be resolved in float32: refused HERE, before the
            # claim, by the one function the loader calls too. Do NOT leave it to the loader (the run directory is claimed by then).
            check_csv_level(values, names, config.transform_args.scale_floor)
            channels = int(values.shape[1])
        lines = geometry_problems(context.family, channels, data_cfg.horizon, kwargs)
        if lines:
            raise ConfigValidationError(lines)

    # -- compile --------------------------------------------------------

    def metric_names(self, config: TimeSeriesForecastingConfig, facts: DataFacts) -> List[str]:
        # DECISION plan-2026-10-30T132432-a865fa08/D-055: names only; a Keras metric allocates variables and the first eager
        # op would initialise TensorFlow's devices before setup_gpu.
        # DECISION plan-2026-11-13T105600-3a8319ef/D-024
        # ONE metric named `mae` for every family gives the key `val_mae` for all nine heads, the two-output nbeats included
        # (measured). Do NOT name it per family or per output: the watched key would then differ by family.
        return [METRIC]

    def build_loss_and_metrics(self, config: TimeSeriesForecastingConfig, data: Any) -> Tuple[Any, List[Any]]:
        from train.common import forecast_run

        if self._model is None or self._decoder is None:
            raise RuntimeError("build_loss_and_metrics reads the model build_model made; it was asked for before it")
        loss_cfg: ForecastLossArgs = config.loss_args
        built = forecast_run.build_loss_and_metrics(self._decoder, self._model, loss_cfg.name)
        return built.loss, built.metrics

    def compile_options(self, config: TimeSeriesForecastingConfig, spec: registry.ModelSpec) -> Dict[str, Any]:
        # DECISION plan-2026-11-13T105600-3a8319ef/D-014
        # DECISION plan-2026-11-13T105600-3a8319ef/D-028
        # `jit_compile` is passed EXPLICITLY, False unless the field asks: Keras 3.8 resolves "auto" to XLA whenever a GPU is
        # present, so a GPU run and a CPU run would build different graphs (measured: ets fails under XLA on the GPU, the
        # other families run 3.6x to 11.4x slower without it). Do NOT drop the key "because False is the default", and do
        # NOT let `true` through for a family that refuses it.
        problem = jit_problem(spec.family, config.jit_compile)
        if problem:
            raise ConfigError(problem)
        return {"jit_compile": config.jit_compile}

    def initial_loss_check(
        self, config: TimeSeriesForecastingConfig, model: Any, data: Any
    ) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return the epoch-0 numbers; the ratio is the ``mae`` over naive-last's, and no verdict is given.

        :raises RuntimeError: for a non-finite loss or ``mae`` (the model diverges before training).
        """
        # DECISION plan-2026-11-13T105600-3a8319ef/D-029
        # Only a non-finite value is refused. The untrained model sits between 1.1x and 9.0x naive-last (measured) and ets
        # below it, so no ratio threshold separates a well-scaled model from a mis-scaled one. Do NOT add one.
        from dl_techniques.utils.logger import logger
        from train.common import classification_run

        if self._baselines is None:
            raise RuntimeError("initial_loss_check compares with the naive-last MAE, which build_data works out; it was asked for first")
        metrics = classification_run.evaluate_dict(model, data.select_ds)
        for name in ("loss", METRIC):
            if not math.isfinite(metrics.get(name, math.nan)):
                raise RuntimeError(
                    f"Initial validation {name} is {metrics.get(name)}: the model diverges before training. {self.initial_loss_hint}"
                )
        ratio = metrics[METRIC] / self._baselines["naive_last"]
        logger.info(
            f"Untrained evaluate BEFORE fit: val {METRIC} {metrics[METRIC]:.6g} on {data.n_select} selection windows, "
            f"{ratio:.3g}x naive-last's {self._baselines['naive_last']:.6g}"
        )
        return metrics, ratio, None

    def unjudged_note(self, config: TimeSeriesForecastingConfig) -> str:
        return (
            "`init_scale_warning` is null (not judged): `initial_loss_ratio` is the `mae` of the untrained model over the "
            "naive-last MAE on the same fixed selection windows. An untrained forecaster sits between 1.1 and 9.0 times "
            "naive-last (measured, nbeats the highest) and ets below it, so no threshold on that ratio separates a "
            "well-scaled model from a mis-scaled one; a non-finite initial loss is refused"
        )

    # -- the two seams of the shared sequence ---------------------------

    def fit_kwargs(self, config: TimeSeriesForecastingConfig, data: Any) -> Dict[str, Any]:
        # DECISION plan-2026-11-13T105600-3a8319ef/D-013
        # The train stream is INFINITE, so an epoch is `steps_per_epoch` batches by this keyword (the sequence refuses an
        # infinite stream without it); `validation_steps` is the selection stream's one pass, stated so that Keras never has
        # to guess a length. Do NOT drop either "because the defaults work": without `steps_per_epoch` the epoch never ends,
        # and a different `steps_per_epoch` moves the schedule, the backup arithmetic and the epoch-zero comparison.
        return {"steps_per_epoch": data.steps_per_epoch, "validation_steps": data.n_select // config.batch_size}

    def resume_stream(self, config: TimeSeriesForecastingConfig, data: Any, completed_steps: int) -> Optional[Any]:
        # DECISION plan-2026-11-13T105600-3a8319ef/D-013
        # A resumed `fit` starts a fresh iterator, so the stream the fresh process built would replay from its first
        # element (epoch 0's windows). `with_stream` rebuilds the SAME request on a stream seed derived from
        # `(seed, completed_steps)`: the bank, the split, the counts and the select and test windows do not move, the train
        # stream does. Do NOT return `data` ("nothing to rebuild": that is the replay), and do NOT reseed the split.
        return data.with_stream(completed_steps)

    # -- the verdict ----------------------------------------------------

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        """Return the best (the lowest) ``val_mae`` of the selection window."""
        # DECISION plan-2026-11-13T105600-3a8319ef/D-008
        # The MINIMUM over the window the checkpoint was selected from, the same rule the saved file follows. Do NOT take
        # the last epoch or the maximum, and do NOT read epochs before `selection_start_epoch` (no file holds them).
        start = self.selection_start_epoch(self.config)
        return float(min(hist[self.success_metric][start:]))

    def success_threshold(self, config: TimeSeriesForecastingConfig) -> float:
        """Return ``min(naive-last, seasonal-naive, epoch-zero)``; ``evaluate_and_report`` has read the third term."""
        if self._baselines is None:
            raise RuntimeError(
                "success_threshold is cut from the baselines (naive-last and seasonal-naive on the select windows), which "
                "build_data works out; it was asked for before the data was built"
            )
        if self._epoch_zero is None:
            raise RuntimeError(
                "success_threshold needs the epoch-zero val_mae, which evaluate_and_report reads from the fit sidecar; "
                "it was asked for before the reporting"
            )
        return float(min(self._baselines["naive_last"], self._baselines["seasonal_naive"], self._epoch_zero))

    def bar_is_met(self, config: TimeSeriesForecastingConfig, reached: float, threshold: float) -> bool:
        from train.common import forecast_run

        # DECISION plan-2026-11-13T105600-3a8319ef/D-008
        # DECISION plan-2026-11-13T105600-3a8319ef/D-037
        # DECISION plan-2026-11-13T105600-3a8319ef/D-043
        # STRICTLY below, in float32, with a tie band that widens with the select batch count (`forecast_run.beats_bar`, D-043):
        # the base class's `>=` reads an error backwards, `<=` would call a model that merely EQUALS naive-last a success, and a
        # float32 `val_mae` of a model that outputs naive-last sits 1 step below the float64 numpy baseline on some seeds. There
        # is no `success_threshold` field to weaken it (D-037). Do NOT compare the Python floats, do NOT shrink the band to 0 and
        # do NOT pin it at 16 steps again.
        return forecast_run.beats_bar(reached, threshold, self._select_batches or 1)

    # -- the log and the summary ---------------------------------------

    def epoch_line_metrics(self, config: TimeSeriesForecastingConfig) -> Tuple[str, ...]:
        return ("loss", METRIC, "val_loss", WATCH_METRIC)

    def summary_fields(self, config: TimeSeriesForecastingConfig, data: Any) -> Dict[str, Any]:
        return {
            "lookback": data.lookback,
            "horizon": data.horizon,
            "channels": data.input_shape[1],
            "n_series": data.n_series,
            "series_dropped": [list(pair) for pair in data.dropped],
            "segments": {name: list(bounds) for name, bounds in data.segments.items()},
            "stream": {
                "seed": data.seed,
                "stream_seed": data.stream_seed,
                "first_train_batch_sha256": data.first_batch_digest,
                "rule": (
                    "the selection and test windows are keyed on (seed, split, index) and never move; the train windows "
                    "on (stream_seed, counter), inside the train segment; a resume derives stream_seed from (seed, "
                    "completed steps), the first 4 bytes of sha256, and the split never follows it"
                ),
            },
        }

    def recipe_notices(self, facts: DataFacts, family: str) -> List[Tuple[str, str]]:
        return [_NOTICES[family]] if family in _NOTICES else []

    def task_notes(self, config: TimeSeriesForecastingConfig, baseline: Dict[str, float], ratio: float, data: Any) -> List[str]:
        """Return the forecasting lines of ``results_summary.json``'s ``notes``."""
        from train.common import classification_run

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        notes = [
            f"initial `mae` {baseline[METRIC]:.6g} is {ratio:.3g}x the naive-last MAE ({self._baselines['naive_last']:.6g}) on the "
            f"{data.n_select} fixed selection windows: a plain evaluate in inference mode; see the `init_scale_warning` note",
            *classification_run.RUN_NOTES[:2],
            "every number is in the units of the window's CONTEXT statistics (each window and its target are divided by the "
            "context's own mean and scale); `val_mae` is the mean absolute error of the point forecast (the median of a "
            "quantile head) and is a function of the weights alone (the selection windows are fixed by (seed, split, index))",
            "`success_value` is the lowest `val_mae` of the selection window; `success_threshold` is min(`baselines.naive_last`, "
            "`baselines.seasonal_naive`, `epoch_zero.val_mae`), the first two numpy on the same fixed windows and the third read "
            "from the fit sidecar; the bar is STRICTLY below, compared in float32 with a band of "
            "`train.common.forecast_run.tie_band` float32 steps (`TIE_ULPS` = 16 up to 16 select batches, wider by the square root of "
            "the batch count above; a model that outputs the naive-last forecast sits within it)",
            "`test_metrics_best` is the in-memory best weights and `test_metrics_final` the last epoch's, on the test windows, "
            "which never influenced selection; `canonical_metrics` holds the post-hoc MAE, RMSE, sMAPE, rMAE, MASE (scaled by "
            "the context's one-step differences) and, for a quantile head, coverage and sharpness",
        ]
        notes.append(
            "the model analysis is data-free (weights and spectral only; `analyzer.analyzers` lists them): the stock analyses "
            "read class probabilities, which a forecast is not"
            if analysis_cfg.model_analysis else "analysis_args.model_analysis is false: `analyzer.status` is 'skipped'"
        )
        if not checkpoint_cfg.save_best:
            notes.append("checkpoint_args.save_best is false: no best_model.keras was written, so the reload check did not run")
        if not checkpoint_cfg.save_last:
            notes.append("checkpoint_args.save_last is false: no final_model.keras was written and `model_loading_validated` is null")
        return notes

    # -- the reporting --------------------------------------------------

    def run_analysis(
        self,
        config: TimeSeriesForecastingConfig,
        model: Any,
        test_data: Tuple[Any, Any],
        history: Any,
        experiment_name: str,
        run_dir: str,
    ) -> Any:
        """Run the data-free analysis (weights and spectral only); see the module contract."""
        from train.common import run_summary

        # DECISION plan-2026-11-13T105600-3a8319ef/D-025
        # The stock analysis reads the output as class probabilities and the target as labels; a forecast is neither
        # (measured: the data-free analysis completes on 6 of 7 families, ets, with one parameter, reports `missing`). Do NOT
        # fall back to the default hook.
        return run_summary.data_free_model_analysis(model, test_data, history, experiment_name, run_dir, config.seed)

    def evaluate_and_report(self, config: TimeSeriesForecastingConfig, data: Any, fit: FitState) -> Evaluation:
        """Run the forecasting reporting sequence; see the module contract."""
        from types import SimpleNamespace

        from train.common import classification_run, evaluation, forecast_run, run_summary
        from train.common.callbacks import best_checkpoint_path
        from train.common.resume_state import FIT_STATE_NAME

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        baselines, decoder = self._baselines, self._decoder
        if baselines is None or decoder is None:
            raise RuntimeError("evaluate_and_report needs the baselines and the decoder the data and the model made")
        # The epoch-zero term of the bar, from the sidecar: see `epoch_zero_mae` (a resume reads the same number).
        self._epoch_zero = forecast_run.epoch_zero_mae(fit.run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME)
        # One full select batch: the sample of the reload checks and of the analysis (a static-batch family predicts only at
        # exactly its batch, so the sample is never cut).
        sample_x, sample_y = (array.numpy() for array in next(iter(data.select_ds)))
        # The in-memory model holds the BEST weights here (EarlyStopping restored them). The test windows are fixed, so the
        # reloaded file reproduces them: the reload gap is 0.0, not a tolerance.
        test_best = classification_run.evaluate_dict(fit.model, data.test_ds)
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        if checkpoint_cfg.save_best:
            # DECISION plan-2026-11-13T105600-3a8319ef/D-007
            # `check_model_loading` reloads with compile=False (a compiled reload fails for the families with a custom metric
            # or a bound-method loss, D-024) and compares EVERY output of the model: nbeats returns (forecast, residual), and
            # its list branch checks the second as well as the first. Do NOT use `load_best_metrics` here (it reloads compiled
            # and fails for five families), and do NOT compare the first output only.
            expected = fit.model.predict(sample_x, batch_size=data.batch_size, verbose=0)
            checked = evaluation.check_model_loading(
                best_checkpoint_path(str(fit.run_dir)), sample_x, expected, batch_size=data.batch_size,
                relative_tolerance=evaluation.LOAD_RTOL, reference_model=fit.model,
            )
            max_diff = checked.max_abs_diff
            if not checked.ok:
                load_error = f"best_model.keras reloads to other outputs than the in-memory best weights: {checked.as_record()}"
        select_report = forecast_run.canonical_report(fit.model, decoder, data.select_ds)
        test_windows = forecast_run.decode_dataset(fit.model, decoder, data.test_ds)
        canonical = {"select": select_report, "test": forecast_run.canonical_metrics(*test_windows, decoder.levels)}
        visualizations: Dict[str, Any] = {"files": [], "failed": [], "skipped": not analysis_cfg.figures}
        if analysis_cfg.figures:
            forecast_run.plot_forecasts(
                fit.vis_dir / "forecast_test.png", *test_windows, decoder.levels, title=f"{config.model}: best weights, test windows")
        # The best weights are analysed, then the last epoch's replace them (the sequence the denoising task shares).
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
        if analysis.get("status") == "missing":
            # DECISION plan-2026-11-13T105600-3a8319ef/D-025
            # ets has one parameter, and the analyzer finds nothing to analyse (measured: `missing`, no crash). The summary
            # says so as `skipped`, which is what it is; the analyzer's own note stays in `error`.
            analysis = {**analysis, "status": "skipped"}
        if analysis_cfg.figures and fit.vis_dir.is_dir():
            visualizations["files"] = sorted(path.name for path in fit.vis_dir.iterdir() if path.is_file())
        extras: Dict[str, Any] = {
            "baselines": baselines,
            "epoch_zero": {
                "val_mae": self._epoch_zero,
                "binds": bool(self._epoch_zero < min(baselines["naive_last"], baselines["seasonal_naive"])),
            },
            "canonical_metrics": canonical,
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


def train_time_series_forecasting(
    *,
    out: Union[str, Path],
    model: str,
    data: Union[TimeSeriesData, Mapping[str, Any], str, Path],
    epochs: Union[int, str] = AUTO,
    batch_size: Union[int, str] = AUTO,
    seed: int = DEFAULT_SEED,
    devices: Union[int, str, None] = AUTO,
    precision: str = "float32",
    resume_interrupted: bool = False,
    checkpoint: Union[str, Path, None] = None,
    model_args: Optional[Mapping[str, Any]] = None,
    jit_compile: bool = False,
    optim_args: Union[OptimArgs, Mapping[str, Any], None] = None,
    transform_args: Union[ForecastTransformArgs, Mapping[str, Any], None] = None,
    metric_args: Union[ForecastMetricArgs, Mapping[str, Any], None] = None,
    loss_args: Union[ForecastLossArgs, Mapping[str, Any], None] = None,
    checkpoint_args: Union[CheckpointArgs, Mapping[str, Any], None] = None,
    logger_args: Union[LoggerArgs, Mapping[str, Any], None] = None,
    analysis_args: Union[AnalysisArgs, Mapping[str, Any], None] = None,
    debug_args: Union[DebugArgs, Mapping[str, Any], None] = None,
) -> TrainResult:
    """Train a forecaster; see the module contract."""
    recipes.register_builtin_models()
    if isinstance(data, (str, Path)):
        data = load_forecast_data_config(data)
    given = run_request(
        out=out, model=model, data=data, epochs=epochs, batch_size=batch_size, seed=seed,
        devices=devices, precision=precision, resume_interrupted=resume_interrupted,
        checkpoint=checkpoint, model_args=model_args,
        groups={
            "optim_args": optim_args,
            "transform_args": transform_args,
            "metric_args": metric_args,
            "loss_args": loss_args,
            "checkpoint_args": checkpoint_args,
            "logger_args": logger_args,
            "analysis_args": analysis_args,
            "debug_args": debug_args,
        },
    )
    given["jit_compile"] = jit_compile
    config = validate(TimeSeriesForecastingConfig, given)
    return TimeSeriesForecastingTrainer(config).run()
