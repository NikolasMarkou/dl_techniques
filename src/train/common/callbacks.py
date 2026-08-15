"""Common callback and learning rate schedule utilities for training scripts."""

import os
import re
import json
import keras
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any

from dl_techniques.utils.logger import logger
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback
from train.common.config_io import json_numpy_default
from train.common.evaluation import generate_training_curves

# `create_learning_rate_schedule` now lives in dl_techniques.optimization.schedule
# alongside the other LR-schedule construction. It is re-exported here so the
# ~14 existing `from train.common import create_learning_rate_schedule` call
# sites keep working unchanged.
from dl_techniques.optimization.schedule import create_learning_rate_schedule  # noqa: F401


# ---------------------------------------------------------------------

# DECISION plan-2026-08-12T123743-e798a9e1/D-021
# This is the ONE producer of the best-checkpoint path. `create_callbacks`
# below configures `ModelCheckpoint` with `best_checkpoint_path(results_dir)`
# and every reader (currently
# `train.common.nlp.run_finetune_post_training_analysis`) resolves the same
# way, so a writer and a reader CANNOT disagree about the name.
# DO NOT re-spell this literal at a read site. That is exactly how F-23
# happened: a reader hard-coded `best_sentiment_model.keras` while the writer
# here wrote `best_model.keras`, a filename mismatch (on top of a directory
# mismatch) that nothing could detect until a real run crashed at the very end.
# See decisions.md D-021 (which supersedes D-017).
BEST_CHECKPOINT_FILENAME = 'best_model.keras'


def best_checkpoint_path(results_dir: str) -> str:
    """Path `create_callbacks`' `ModelCheckpoint` writes the best model to.

    Args:
        results_dir: A run directory -- the SECOND element of the tuple
            returned by :func:`create_callbacks` (or by
            :func:`train.common.nlp.create_nlp_callbacks`). It is the
            timestamped `results/{prefix}_{model}_{timestamp}` directory, NOT a
            config's static `save_dir`.

    Returns:
        `os.path.join(results_dir, BEST_CHECKPOINT_FILENAME)`.

    Failure mode: none -- this is pure path arithmetic and never touches the
    filesystem, so it returns a path for a checkpoint that may not exist yet
    (e.g. when `fit()` had no validation split and `ModelCheckpoint` never
    fired). Callers that need the file must check for it themselves.
    """
    return os.path.join(results_dir, BEST_CHECKPOINT_FILENAME)


# ---------------------------------------------------------------------
# Monitor-direction resolution
# ---------------------------------------------------------------------

# Metric-name tokens whose direction is DOWN (smaller is better).
#
# Both registries hold only tokens that are UNAMBIGUOUS as metric names,
# because a Keras multi-output monitor key embeds the OUTPUT LAYER's name
# (`val_<output_name>_<metric_name>`). `darkir`'s first output is the
# `layers.Add(name="final_residual")`, so `val_final_residual_psnr` is a real
# monitor key in this repo -- a generic token like `residual`, `score`, `std`
# or `variance` in either set would resolve it off the layer name and get the
# direction exactly backwards. Do not add one.
_MINIMIZE_METRIC_TOKENS = frozenset({
    'loss', 'error', 'err', 'mae', 'mse', 'rmse', 'nrmse', 'mape', 'smape',
    'msle', 'nll', 'nlpd', 'logloss', 'crossentropy', 'perplexity', 'ppl',
    'wer', 'cer', 'fid', 'kid', 'lpips', 'absrel', 'sqrel', 'rmselog',
    'crps', 'pinball', 'divergence', 'kld', 'entropy', 'quantization',
})

# Metric-name tokens whose direction is UP (larger is better).
_MAXIMIZE_METRIC_TOKENS = frozenset({
    'accuracy', 'acc', 'psnr', 'ssim', 'msssim', 'iou', 'miou', 'jaccard',
    'map', 'ap', 'f1', 'fbeta', 'dice', 'auc', 'auroc', 'auprc', 'prauc',
    'precision', 'recall', 'sensitivity', 'specificity', 'r2', 'bleu',
    'rouge', 'meteor', 'cider', 'kappa', 'mcc', 'purity', 'nmi', 'ari',
    'delta1', 'delta2', 'delta3', 'topk', 'top1', 'top3', 'top5',
    'likelihood',
})


# DECISION plan-2026-08-14T233721-d4f9beb2/D-051
# Resolve the selection DIRECTION from a metric-name token registry, never from
# a substring test. DO NOT restore `monitor_mode = 'max' if 'accuracy' in
# monitor else 'min'`: every maximize metric whose name lacks the substring
# "accuracy" -- `val_psnr`, `val_iou`, `val_box_iou`, `val_map`, `val_f1`,
# `val_dice`, `val_auc`, `val_ssim` -- got `mode='min'`, so
# `EarlyStopping(restore_best_weights=True)` RESTORED and `ModelCheckpoint`
# SAVED the WORST epoch. `train_sam3.py` (D-041) measured the cost of exactly
# this class of selection error at box IoU 0.2360 vs an achievable 0.2724.
# DO NOT delegate to Keras' own `mode='auto'` either: it is resolved at
# `on_train_begin`/`on_epoch_end` by looking for a compiled metric OBJECT whose
# `.name` equals the monitor minus its `val_` prefix and reading that object's
# `_direction`, and it RAISES `ValueError` when it cannot find one. Every
# multi-output monitor in this repo (`train_darkir.py`'s
# `f"val_{model.output_names[0]}_psnr"`) is prefixed with the output name and
# therefore matches no metric object -- so `auto` would turn a silently-wrong
# selection into a mid-run crash. Minimize tokens are tested FIRST so a
# composite name like `dice_loss` or `iou_loss` still resolves to `min`.
# See decisions.md D-051.
def resolve_monitor_mode(monitor: str, mode: Optional[str] = None) -> str:
    """Resolve the ``mode`` for a monitored metric from its name.

    Args:
        monitor: The metric key that will be read out of the epoch ``logs``,
            e.g. ``'val_loss'``, ``'val_accuracy'``, ``'val_psnr'``,
            ``'val_output_1_psnr'``. Split into lowercase alphanumeric tokens;
            a leading ``val`` token is dropped.
        mode: Explicit override. ``'min'`` or ``'max'`` is returned unchanged;
            ``None`` (the default) means "infer from ``monitor``".

    Returns:
        ``'min'`` or ``'max'``, suitable for ``keras.callbacks.EarlyStopping``
        and ``keras.callbacks.ModelCheckpoint``.

    Failure mode: raises ``ValueError`` for an explicit ``mode`` that is
    neither ``'min'`` nor ``'max'`` (a typo there would otherwise make Keras
    warn and silently fall back to ``'auto'``). An UNRECOGNIZED metric name
    never raises -- it falls back to ``'min'``, which is what the old
    substring heuristic returned for it, and logs a WARNING naming the metric
    so the caller can pass ``mode=`` explicitly. Add the token to
    ``_MAXIMIZE_METRIC_TOKENS`` / ``_MINIMIZE_METRIC_TOKENS`` rather than
    re-deriving a direction at a call site.
    """
    if mode is not None:
        if mode not in ('min', 'max'):
            raise ValueError(
                f"monitor_mode must be 'min', 'max' or None; got {mode!r}."
            )
        return mode

    tokens = {token for token in re.split(r'[^a-z0-9]+', monitor.lower()) if token}
    tokens.discard('val')

    if tokens & _MINIMIZE_METRIC_TOKENS:
        return 'min'
    if tokens & _MAXIMIZE_METRIC_TOKENS:
        return 'max'

    logger.warning(
        f"Unrecognized monitor metric '{monitor}': its optimization direction "
        f"is not in the token registry, so checkpoint selection falls back to "
        f"mode='min'. If '{monitor}' is a metric to MAXIMIZE, pass "
        f"monitor_mode='max' to create_callbacks (or add its token to "
        f"train.common.callbacks._MAXIMIZE_METRIC_TOKENS)."
    )
    return 'min'


def create_callbacks(
        model_name: str,
        results_dir_prefix: str = "model",
        output_root: str = "results",
        run_dir: Optional[str] = None,
        monitor: str = 'val_accuracy',
        monitor_mode: Optional[str] = None,
        patience: int = 15,
        use_lr_schedule: bool = True,
        analyzer_epoch_frequency: int = 1,
        include_tensorboard: bool = False,
        include_terminate_on_nan: bool = False,
        include_analyzer: bool = True,
        analyzer_config: Optional[Any] = None,
        analyzer_start_epoch: int = 1,
) -> Tuple[List, str]:
    """
    Create standard training callbacks.

    Parameters
    ----------
    model_name : str
        Name identifier for the model (used in directory naming).
    results_dir_prefix : str
        Prefix for the results directory (e.g., 'convnext_v1', 'convnext_v2').
    output_root : str
        Base directory under which the timestamped run dir is created. Default 'results'.
    run_dir : Optional[str]
        Exact run directory to write artifacts into. When provided, it is used
        verbatim as the results directory and the ``{prefix}_{model_name}_{timestamp}``
        construction (and ``output_root``) is bypassed. Use this when the caller
        already owns a run directory, to avoid creating a second orphan dir.
        Default None preserves the timestamped-dir behavior.
    monitor : str
        Metric to monitor for checkpointing/early stopping.
    monitor_mode : Optional[str]
        Selection direction for `monitor`: 'min', 'max', or None (default) to
        infer it from the metric name via :func:`resolve_monitor_mode`. Pass it
        explicitly for a metric whose name that function does not recognize --
        it logs a WARNING naming the metric and falls back to 'min'.
    patience : int
        Early stopping patience.
    use_lr_schedule : bool
        If True, skip ReduceLROnPlateau (assumes external LR schedule).
    analyzer_epoch_frequency : int
        How often to run the EpochAnalyzerCallback (every N epochs).
    include_tensorboard : bool
        If True, add TensorBoard callback.
    include_terminate_on_nan : bool
        If True, add TerminateOnNaN callback.
    include_analyzer : bool
        If True, add EpochAnalyzerCallback. Set False to disable.
    analyzer_config : Optional[AnalysisConfig]
        Custom AnalysisConfig for EpochAnalyzerCallback. None uses defaults.
    analyzer_start_epoch : int
        Epoch to start running the analyzer (default: 1).

    Returns
    -------
    Tuple[List, str]
        List of callbacks and the results directory path.
    """
    if run_dir is not None:
        results_dir = run_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(output_root, f"{results_dir_prefix}_{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    monitor_mode = resolve_monitor_mode(monitor, monitor_mode)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode=monitor_mode,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path(results_dir),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            mode=monitor_mode,
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv')
        ),
    ]

    if include_terminate_on_nan:
        callbacks.insert(0, keras.callbacks.TerminateOnNaN())

    if include_analyzer:
        analyzer_kwargs = dict(
            output_dir=os.path.join(results_dir, "epoch_analysis"),
            model_name=model_name,
            epoch_frequency=analyzer_epoch_frequency,
            start_epoch=analyzer_start_epoch,
        )
        if analyzer_config is not None:
            analyzer_kwargs["analysis_config"] = analyzer_config
        callbacks.append(EpochAnalyzerCallback(**analyzer_kwargs))

    if include_tensorboard:
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, "tensorboard"),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
        ))

    if not use_lr_schedule:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )

    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir


# ---------------------------------------------------------------------

# NOT @keras.saving.register_keras_serializable: callbacks are never
# serialized as part of a model (StepCheckpointCallback precedent, SYSTEM.md).
class EpochMetricsPlotCallback(keras.callbacks.Callback):
    """Accumulate per-epoch metrics and emit mid-training curve PNGs.

    Replaces the three near-identical hand-rolled per-epoch matplotlib
    metrics callbacks previously local to ``resnet``, ``vit``, and ``bfunet``
    (plan_2026-06-02_35651564, F10). Accumulates ``loss`` plus each requested
    metric (and its ``val_`` counterpart) across epochs, then on a fixed
    cadence delegates plotting to
    :func:`train.common.evaluation.generate_training_curves` so no raw
    matplotlib lives in ``common``.

    The plot guard is fail-soft-but-LOUD: a plotting failure is logged at
    WARNING with a traceback and never aborts the (multi-hour) training run.

    Args:
        viz_dir: Directory the per-epoch PNGs (and optional JSON) are written
            into. Created with ``exist_ok=True`` at construction time.
        metric_names: Metric keys (besides ``loss``) to accumulate and plot,
            e.g. ``["accuracy", "top5_accuracy"]``. Each entry's ``val_``
            counterpart is also tracked when present in the epoch logs.
        every_n: Plot cadence. A plot is produced when
            ``(epoch + 1) % every_n == 0`` or on the first epoch
            (``epoch == 0``). Defaults to ``5``.
        write_json: If ``True``, also dump the latest accumulated metrics to
            ``viz_dir/latest_metrics.json`` (serialized via
            :func:`train.common.config_io.json_numpy_default`). Defaults to
            ``False``.
    """

    def __init__(
            self,
            viz_dir: str,
            metric_names: List[str],
            every_n: int = 5,
            write_json: bool = False,
    ) -> None:
        super().__init__()
        # LESSON: makedirs at the top of every save-capable component.
        os.makedirs(viz_dir, exist_ok=True)

        self.viz_dir = viz_dir
        self.metric_names = list(metric_names)
        self.every_n = every_n
        self.write_json = write_json

        # Accumulators: 'loss' + each metric, plus their 'val_' counterparts.
        # val_ keys are appended only when present in a given epoch's logs.
        self.train_metrics: Dict[str, List[float]] = {
            "loss": [],
            **{name: [] for name in self.metric_names},
        }
        self.val_metrics: Dict[str, List[float]] = {
            "val_loss": [],
            **{f"val_{name}": [] for name in self.metric_names},
        }

    def on_epoch_end(
            self, epoch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Bucket float-coercible logs, then plot on the configured cadence."""
        if logs is None:
            logs = {}

        for metric_name, metric_value in logs.items():
            try:
                val = float(metric_value)
            except (ValueError, TypeError):
                continue
            if metric_name in self.train_metrics:
                self.train_metrics[metric_name].append(val)
            elif metric_name in self.val_metrics:
                self.val_metrics[metric_name].append(val)

        if (epoch + 1) % self.every_n == 0 or epoch == 0:
            self._plot(epoch)

    def _build_history(self) -> Dict[str, List[float]]:
        """Assemble the dict ``generate_training_curves`` consumes.

        Drops empty accumulators so a never-populated ``val_`` key does not
        produce a zero-length series.
        """
        history: Dict[str, List[float]] = {}
        for key, values in self.train_metrics.items():
            if values:
                history[key] = values
        for key, values in self.val_metrics.items():
            if values:
                history[key] = values
        return history

    def _plot(self, epoch: int) -> None:
        """Delegate per-epoch curve plotting; fail-soft-but-loud."""
        # LESSON: makedirs again at save time (dir may have been removed).
        os.makedirs(self.viz_dir, exist_ok=True)
        try:
            history = self._build_history()
            if not history.get("loss"):
                return

            generate_training_curves(
                history,
                self.viz_dir,
                filename=f"epoch_{epoch + 1:03d}_metrics",
            )

            if self.write_json:
                metrics_data = {
                    "epoch": epoch + 1,
                    "train_metrics": self.train_metrics,
                    "val_metrics": self.val_metrics,
                }
                json_path = os.path.join(self.viz_dir, "latest_metrics.json")
                with open(json_path, "w") as f:
                    json.dump(
                        metrics_data, f, indent=2, default=json_numpy_default
                    )
        except Exception as e:
            logger.warning(
                f"EpochMetricsPlotCallback: failed to create metrics plots: {e}",
                exc_info=True,
            )


# ---------------------------------------------------------------------

def read_current_lr(model: keras.Model) -> float:
    """Read the current learning rate from ``model``'s live optimizer.

    ``optimizer.learning_rate`` already returns the CURRENT VALUE -- Keras
    evaluates an attached ``LearningRateSchedule`` at the optimizer's step
    before handing it back -- so a plain ``float()`` of it tracks the schedule
    correctly. The isinstance branch below is defensive, for the case where an
    optimizer hands back the schedule object itself.

    Returns ``float('nan')`` if the optimizer or its learning rate is
    unavailable (e.g. called before ``compile``), so a logging path can never
    abort training.
    """
    try:
        optimizer = model.optimizer
        lr = optimizer.learning_rate
        if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
            return float(keras.ops.convert_to_numpy(lr(optimizer.iterations)))
        return float(keras.ops.convert_to_numpy(lr))
    except Exception:
        return float("nan")


# NOT @keras.saving.register_keras_serializable: callbacks are never
# serialized as part of a model (StepCheckpointCallback precedent, SYSTEM.md).
class LearningRateLogger(keras.callbacks.Callback):
    """Record the current learning rate into ``logs`` each epoch.

    Consolidates the copies that lived in the power_mlp and capsnet trainers
    (a third, in ``clip/train_clip.py``, was never instantiated and was
    deleted). Their bodies were equivalent; this is a de-duplication, NOT a
    behaviour change -- see :func:`read_current_lr` on why the plain
    ``float(optimizer.learning_rate)`` they used was already correct.

    ``bfunet/common.py`` keeps its own ``LRLoggerCallback``: it deliberately
    leaves ``logs['lr']`` UNSET on a non-finite read rather than writing a NaN
    into the CSV row, and documents a callback-ordering contract.

    Args:
        log_key: Key written into ``logs``. The default ``'lr'`` matches what
            the adopting trainers already recorded in their history; changing
            it renames the series in every downstream plot and CSV column.
    """

    def __init__(self, log_key: str = "lr") -> None:
        super().__init__()
        self.log_key = log_key

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is None:
            logs = {}
        logs[self.log_key] = read_current_lr(self.model)
