"""Common callback and learning rate schedule utilities for training scripts."""

import os
import json
import keras
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any

from dl_techniques.utils.logger import logger
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from train.common.config_io import json_numpy_default
from train.common.evaluation import generate_training_curves


# ---------------------------------------------------------------------

def create_callbacks(
        model_name: str,
        results_dir_prefix: str = "model",
        output_root: str = "results",
        run_dir: Optional[str] = None,
        monitor: str = 'val_accuracy',
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

    monitor_mode = 'max' if 'accuracy' in monitor else 'min'

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode=monitor_mode,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
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

def create_learning_rate_schedule(
        initial_lr: float,
        schedule_type: str = 'cosine',
        total_epochs: int = 100,
        warmup_epochs: int = 5,
        steps_per_epoch: Optional[int] = None,
        warmup_steps: int = 0,
        warmup_start_lr: float = 1e-8,
) -> keras.optimizers.schedules.LearningRateSchedule:
    """
    Create learning rate schedule.

    Parameters
    ----------
    initial_lr : float
        Initial learning rate.
    schedule_type : str
        Type of schedule ('cosine', 'exponential', 'constant').
    total_epochs : int
        Total number of training epochs.
    warmup_epochs : int
        Number of warmup epochs. RESERVED / no-op — kept only for backward
        positional compatibility. Use ``warmup_steps`` to activate warmup.
    steps_per_epoch : Optional[int]
        Steps per epoch (for step-based schedules like ImageNet).
    warmup_steps : int
        Active warmup control. When ``> 0`` (cosine schedule only), the cosine
        decay is wrapped in a :class:`WarmupSchedule` that linearly ramps from
        ``warmup_start_lr`` to ``initial_lr`` over ``warmup_steps`` steps before
        decaying. ``0`` (default) means NO warmup — existing callers are
        unaffected. Requires ``steps_per_epoch`` to be set.
    warmup_start_lr : float
        Learning rate at the start of the warmup ramp (only used when
        ``warmup_steps > 0``). Defaults to ``1e-8``.

    Returns
    -------
    Learning rate schedule or float for constant.
    """
    if schedule_type == 'cosine':
        # DECISION plan_2026-06-02_cc4d4e14/D-004: warmup is wired ONLY through the
        # explicit warmup_steps param (default 0 = no-op). Do NOT activate via
        # warmup_epochs — dozens of existing callers rely on the plain cosine path
        # and would silently gain warmup (behavior regression). warmup engages only
        # when warmup_steps>0, reproducing the inline CosineDecay+WarmupSchedule
        # block (alpha=0.01, max(1, total_steps-warmup_steps) guard) at the 11 C1
        # sites. See decisions.md D-004.
        if warmup_steps > 0:
            if steps_per_epoch is None:
                raise ValueError(
                    "create_learning_rate_schedule: warmup_steps>0 requires steps_per_epoch"
                )
            total_steps = total_epochs * steps_per_epoch
            primary = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_lr,
                decay_steps=max(1, total_steps - warmup_steps),
                alpha=0.01,
            )
            return WarmupSchedule(
                warmup_steps=warmup_steps,
                warmup_start_lr=warmup_start_lr,
                primary_schedule=primary,
            )
        decay_steps = total_epochs if steps_per_epoch is None else total_epochs * steps_per_epoch
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=0.01
        )
    elif schedule_type == 'exponential':
        decay_steps = (total_epochs // 4) if steps_per_epoch is None else (total_epochs // 4) * steps_per_epoch
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=0.9
        )
    else:  # constant
        return initial_lr


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
