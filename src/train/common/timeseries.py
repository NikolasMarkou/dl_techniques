"""Shared time-series training scaffolding for the synthetic-pattern forecasters.

This module is the single home for the duplication that survived across the
time-series training scripts (``nbeats`` / ``prism`` / ``tirex`` / ``mdn``):

- :class:`BaseTimeSeriesTrainingConfig` — the shared dataclass fields (data
  splits, optimizer/warmup knobs, pattern selection, category weights,
  visualization + deep-analysis flags) with a ratio-sum invariant. Each script
  subclasses this and adds its architecture-specific fields.
- :func:`_fill_nans` — the forward-fill-then-zero-fill NaN cleaner (the complete
  version shared by the forecasting trio; mdn's divergent copy that omitted the
  trailing zero-fill is unified onto this).
- :class:`WindowedTimeSeriesProcessor` — the buffered streaming + fixed val/test
  data processor base. Generalized so all four scripts subclass it via exactly
  two override hooks (``_make_sample`` and ``output_signature``) plus a handful
  of constructor parameters. The base absorbs:
    * weighted-or-uniform pattern sampling (``pattern_to_category`` present →
      category-weighted; absent → uniform, matching mdn),
    * a ``normalize`` flag and ``normalize_method`` (STANDARD for nbeats/prism,
      none for tirex, ROBUST for mdn),
    * nested arbitrary I/O via ``tf.nest`` stacking (mdn's
      ``((seq, task_id), y)`` and nbeats's ``(x, (forecast, recon))``).

These are plain classes / functions (not Keras layers or models) so there is no
``@keras.saving.register_keras_serializable()`` decoration here.
"""

import os
import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.analyzer import AnalysisConfig
from dl_techniques.datasets.time_series import (
    TimeSeriesNormalizer,
    NormalizationMethod,
    TimeSeriesGenerator,
)
from dl_techniques.utils.logger import logger
from train.common.config_io import json_numpy_default
from train.common.evaluation import generate_training_curves
from train.common.callbacks import create_callbacks as create_common_callbacks

# Default category -> sampling-weight map shared by nbeats / prism / tirex.
# Centralized here so the three trio configs no longer each carry a copy.
_DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "trend": 1.0, "seasonal": 1.0, "composite": 1.2,
    "financial": 1.5, "weather": 1.3, "biomedical": 1.2,
    "industrial": 1.3, "intermittent": 1.0, "volatility": 1.1,
    "regime": 1.2, "structural": 1.1,
}


@dataclass
class BaseTimeSeriesTrainingConfig:
    """Shared configuration fields for synthetic time-series training scripts.

    All fields carry defaults so that subclass dataclasses can add their own
    (also-defaulted) architecture fields without running into the
    "non-default argument follows default argument" ordering constraint.

    Subclasses MUST call ``super().__post_init__()`` from their own
    ``__post_init__`` so the data-ratio invariant is always enforced, then add
    their architecture-specific checks.
    """

    result_dir: str = "results"
    save_results: bool = True
    experiment_name: str = "timeseries"
    target_categories: Optional[List[str]] = None

    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Training
    epochs: int = 150
    batch_size: int = 128
    steps_per_epoch: int = 500
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    optimizer: str = 'adamw'

    # Warmup schedule
    use_warmup: bool = True
    warmup_steps: int = 1000
    warmup_start_lr: float = 1e-6

    # Pattern selection
    max_patterns: Optional[int] = None
    max_patterns_per_category: int = 10
    normalize_per_instance: bool = True

    # Category weights for balanced sampling
    category_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_CATEGORY_WEIGHTS)
    )

    # Visualization
    visualize_every_n_epochs: int = 5
    save_interim_plots: bool = True
    plot_top_k_patterns: int = 12
    create_learning_curves: bool = True
    create_prediction_plots: bool = True

    # Deep analysis
    perform_deep_analysis: bool = True
    analysis_frequency: int = 10
    analysis_start_epoch: int = 1

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Data ratios must sum to 1.0, got {total}")


def _fill_nans(data: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along axis 0, then zero-fill any remaining NaNs.

    This is the complete version shared by the nbeats/prism/tirex processors
    (the trailing ``out[np.isnan(out)] = 0`` handles leading NaNs that no prior
    value can forward-fill). mdn's former copy omitted that line and is now
    unified onto this one.

    Args:
        data: 1-D (or leading-axis) numpy array possibly containing NaNs.

    Returns:
        Array of the same shape with NaNs forward-filled then zero-filled.
    """
    # DECISION plan_2026-06-09_a3c7304c/D-007: build the axis-0 ramp with a shape
    # that broadcasts ONLY along axis 0, and gather with take_along_axis. The
    # prior `np.where(~mask, np.arange(N), 0)` + `data[idx]` form silently
    # BROADCAST a 2-D `(N, 1)` input to `(N, N, 1)` (arange is `(N,)`, mask is
    # `(N, 1)`), corrupting every windowed sample once the D-003 change made this
    # call unconditional (the originals guarded it behind `if isnan.any()`, so
    # clean 2-D series never hit it). Do NOT revert to `np.arange(mask.shape[0])`
    # / `data[idx]` — that re-introduces the shape blow-up. See decisions.md D-007.
    mask = np.isnan(data)
    ramp = np.arange(mask.shape[0]).reshape([-1] + [1] * (mask.ndim - 1))
    idx = np.where(~mask, ramp, 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = np.take_along_axis(data, idx, axis=0)
    out[np.isnan(out)] = 0
    return out


class WindowedTimeSeriesProcessor:
    """Buffered streaming + fixed val/test data processor for windowed forecasting.

    Subclasses customize behavior through two hooks — :meth:`_make_sample` and
    the :attr:`output_signature` property — plus constructor parameters. The base
    implements the shared buffered training generator, the fixed val/test
    pre-computation (nested-aware via ``tf.nest``), the raw test generator, and
    ``tf.data`` assembly.

    Args:
        config: A :class:`BaseTimeSeriesTrainingConfig` (or subclass). Supplies
            ``train_ratio``, ``val_ratio``, ``batch_size``, and ``category_weights``.
        generator: Object exposing ``generate_task_data(pattern_name) -> np.ndarray``.
        selected_patterns: Ordered list of pattern names to sample from.
        pattern_to_category: Optional pattern-name -> category map. When provided,
            sampling is category-weighted; when ``None`` sampling is uniform.
        context_len: Number of input timesteps per window.
        horizon_len: Number of target (horizon) timesteps per window.
        num_features: Feature dimension of the series (default 1).
        normalize: When ``True`` apply per-instance normalization in
            :meth:`_safe_normalize`; when ``False`` only clip/fill (tirex).
        normalize_method: Normalization method when ``normalize`` is ``True``.
        patterns_to_mix: How many patterns to draw per buffer refill.
        windows_per_pattern: How many windows to sample per drawn pattern.
        min_length_multiplier: Minimum series length is
            ``(context_len + horizon_len) * min_length_multiplier``.
        require_finite: When ``True`` skip non-finite series in the generators.
    """

    def __init__(
            self,
            config: BaseTimeSeriesTrainingConfig,
            generator: Any,
            selected_patterns: List[str],
            pattern_to_category: Optional[Dict[str, str]] = None,
            *,
            context_len: int,
            horizon_len: int,
            num_features: int = 1,
            normalize: bool = True,
            normalize_method: NormalizationMethod = NormalizationMethod.STANDARD,
            patterns_to_mix: int = 50,
            windows_per_pattern: int = 5,
            min_length_multiplier: int = 1,
            require_finite: bool = True,
    ):
        self.config = config
        self.ts_generator = generator
        self.selected_patterns = selected_patterns
        self.pattern_to_category = pattern_to_category
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.num_features = num_features
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.patterns_to_mix = patterns_to_mix
        self.windows_per_pattern = windows_per_pattern
        self.min_length_multiplier = min_length_multiplier
        self.require_finite = require_finite

        # Always built (used by multi-task subclasses such as mdn; harmless otherwise).
        self.pattern_to_id = {name: i for i, name in enumerate(selected_patterns)}
        self.num_tasks = len(selected_patterns)

        self.weighted_patterns, self.weights = self._prepare_weighted_sampling()

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _prepare_weighted_sampling(self) -> Tuple[List[str], Optional[List[float]]]:
        """Build the sampling population and (optional) per-pattern weights.

        Returns ``(patterns, None)`` for uniform sampling when no
        ``pattern_to_category`` was given (``random.choices(pop, weights=None)``
        is uniform). Otherwise returns ``(patterns, normalized_weights)`` derived
        from ``config.category_weights``.
        """
        if self.pattern_to_category is None:
            return list(self.selected_patterns), None

        patterns, weights = [], []
        for name in self.selected_patterns:
            category = self.pattern_to_category.get(name, "unknown")
            weights.append(self.config.category_weights.get(category, 1.0))
            patterns.append(name)
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(patterns)] * len(patterns)
        return patterns, weights

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #
    def _safe_normalize(self, series: np.ndarray) -> np.ndarray:
        """Clip, NaN-fill (always), optionally per-instance normalize, to float32.

        NaNs are filled unconditionally here (no-op when none are present). This
        is the documented micro-change relative to nbeats/mdn, which filled only
        inside their normalize branch; at their default ``normalize_per_instance=True``
        it makes no difference, and filling always is strictly safer.

        D-003 (decision log): two independent axes the originals keep separate.
        (a) the per-instance STANDARD/ROBUST transform is gated on
        ``config.normalize_per_instance``; (b) the ``clip(-10, 10)`` is applied by
        every *normalizing* processor (``self.normalize=True`` → nbeats/prism/mdn)
        whenever it runs, INDEPENDENT of the per-instance toggle — disabling
        per-instance normalization still clips. tirex passes ``normalize=False`` so
        neither fires (clip(1e6)+fill+float32 only, matching tirex original). This
        reproduces all four originals across BOTH ``normalize_per_instance``
        settings. Do NOT re-couple clip(10) to ``normalize_per_instance``.
        """
        series = np.clip(series, -1e6, 1e6)
        series = _fill_nans(series)
        if self.normalize:
            if getattr(self.config, 'normalize_per_instance', True):
                series = TimeSeriesNormalizer(method=self.normalize_method).fit_transform(series)
            series = np.clip(series, -10.0, 10.0)
        return series.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Hooks (overridable by subclasses)
    # ------------------------------------------------------------------ #
    def _make_sample(self, window: np.ndarray, pattern_name: str) -> Tuple[Any, Any]:
        """Split a normalized window into an ``(inputs, targets)`` sample.

        Default: ``inputs`` is the context reshaped to ``(context_len, num_features)``
        and ``targets`` is the horizon reshaped to ``(horizon_len, num_features)``.
        ``pattern_name`` is ignored by default; multi-task subclasses (mdn) use it
        to emit a task id. Subclasses may return arbitrary nested structures, as
        long as they match :attr:`output_signature`.
        """
        inputs = window[:self.context_len].reshape(-1, self.num_features).astype(np.float32)
        targets = window[self.context_len:].reshape(-1, self.num_features).astype(np.float32)
        return inputs, targets

    @property
    def output_signature(self) -> Tuple[Any, Any]:
        """``tf.TensorSpec`` structure matching :meth:`_make_sample` output."""
        return (
            tf.TensorSpec(shape=(self.context_len, self.num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(self.horizon_len, self.num_features), dtype=tf.float32),
        )

    # ------------------------------------------------------------------ #
    # Generators
    # ------------------------------------------------------------------ #
    def _training_generator(self):
        """Infinite generator with buffered pattern mixing.

        Each refill draws ``patterns_to_mix`` patterns (weighted or uniform),
        and for each pattern that passes the length / finiteness gates, samples
        ``windows_per_pattern`` random windows from the training split, then
        shuffles the buffer.
        """
        total_len = self.context_len + self.horizon_len
        buffer: List[Tuple[Any, Any]] = []

        while True:
            if not buffer:
                for name in random.choices(
                        self.weighted_patterns, weights=self.weights, k=self.patterns_to_mix):
                    try:
                        data = self.ts_generator.generate_task_data(name)
                    except Exception:
                        continue
                    if len(data) < total_len * self.min_length_multiplier:
                        continue
                    if self.require_finite and not np.isfinite(data).all():
                        continue
                    train_data = data[:int(self.config.train_ratio * len(data))]
                    max_start = len(train_data) - total_len
                    if max_start <= 0:
                        continue
                    for _ in range(self.windows_per_pattern):
                        start = random.randint(0, max_start)
                        window = self._safe_normalize(train_data[start:start + total_len])
                        buffer.append(self._make_sample(window, name))
                random.shuffle(buffer)
            if buffer:
                yield buffer.pop()

    def _generate_fixed_dataset(self, split: str, num_samples: int) -> Tuple[Any, Any]:
        """Pre-compute a fixed val/test dataset, stacked nested-aware.

        Cycles through ``selected_patterns``, slicing each series by the
        train/val/test ratios, and collects samples until ``num_samples`` is
        reached. Leaves are stacked with ``tf.nest.map_structure`` so nested
        sample structures (mdn's ``((seq, task), y)``, nbeats's
        ``(x, (forecast, recon))``) stack generically; per-leaf dtype follows the
        sample arrays (set in :meth:`_make_sample`).
        """
        logger.info(f"Pre-computing {split} dataset ({num_samples} samples)")
        total_len = self.context_len + self.horizon_len
        samples: List[Tuple[Any, Any]] = []
        cycle = 0

        while len(samples) < num_samples:
            name = self.selected_patterns[cycle % len(self.selected_patterns)]
            cycle += 1
            try:
                data = self.ts_generator.generate_task_data(name)
            except Exception:
                continue
            if len(data) < total_len * self.min_length_multiplier:
                continue
            if self.require_finite and not np.isfinite(data).all():
                continue

            train_end = int(self.config.train_ratio * len(data))
            val_end = train_end + int(self.config.val_ratio * len(data))
            split_data = data[train_end:val_end] if split == 'val' else data[val_end:]

            max_start = len(split_data) - total_len
            if max_start <= 0:
                continue
            window = self._safe_normalize(split_data[random.randint(0, max_start):][:total_len])
            samples.append(self._make_sample(window, name))

        return tf.nest.map_structure(lambda *xs: np.stack(xs), *samples)

    def _test_generator_raw(self):
        """Fresh test-split samples for visualization (shuffled pattern order)."""
        total_len = self.context_len + self.horizon_len
        viz_patterns = list(self.selected_patterns)
        random.shuffle(viz_patterns)

        for name in viz_patterns:
            try:
                data = self.ts_generator.generate_task_data(name)
            except Exception:
                continue
            if len(data) < total_len * self.min_length_multiplier:
                continue
            if self.require_finite and not np.isfinite(data).all():
                continue
            test_data = data[int((self.config.train_ratio + self.config.val_ratio) * len(data)):]
            if len(test_data) < total_len:
                continue
            start = random.randint(0, len(test_data) - total_len)
            window = self._safe_normalize(test_data[start:start + total_len])
            yield self._make_sample(window, name)

    # ------------------------------------------------------------------ #
    # Dataset assembly
    # ------------------------------------------------------------------ #
    def prepare_datasets(self) -> Dict[str, Any]:
        """Assemble train/val/test ``tf.data.Dataset`` objects.

        Returns a dict with ``train_ds`` (infinite streaming),
        ``val_ds`` / ``test_ds`` (pre-computed, cached), ``validation_steps``,
        ``test_steps``, and ``test_data_raw`` (the stacked test arrays).
        """
        batch_size = self.config.batch_size

        train_ds = tf.data.Dataset.from_generator(
            self._training_generator, output_signature=self.output_signature
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_steps = max(50, len(self.selected_patterns))
        test_steps = max(20, len(self.selected_patterns))

        val_stacked = self._generate_fixed_dataset('val', val_steps * batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices(val_stacked).batch(
            batch_size).cache().prefetch(tf.data.AUTOTUNE)

        test_stacked = self._generate_fixed_dataset('test', test_steps * batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices(test_stacked).batch(
            batch_size).cache().prefetch(tf.data.AUTOTUNE)

        return {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
            'validation_steps': val_steps,
            'test_steps': test_steps,
            'test_data_raw': test_stacked,
        }


# --------------------------------------------------------------------------- #
# Per-epoch visualization callback (shared scaffolding)
# --------------------------------------------------------------------------- #
def _prepare_viz_data_from_processor(
        processor: WindowedTimeSeriesProcessor, k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect up to ``k`` fixed test windows from a processor for visualization.

    Absorbs the identical 10-line ``_test_generator_raw`` collection loop that
    nbeats / prism / tirex each carried in their callback's viz-data prep. Each
    yielded ``(x, y)`` is appended verbatim — so nbeats's reconstruction tuple
    target (``(forecast, backcast)`` when ``reconstruction_loss_weight > 0``)
    passes through unchanged.

    Args:
        processor: A :class:`WindowedTimeSeriesProcessor` exposing
            ``_test_generator_raw()``.
        k: Maximum number of windows to collect (``plot_top_k_patterns``).

    Returns:
        ``(viz_x, viz_y)`` stacked numpy arrays, or two empty arrays if the
        generator yielded nothing.
    """
    viz_x, viz_y = [], []
    for x, y in processor._test_generator_raw():
        viz_x.append(x)
        viz_y.append(y)
        if len(viz_x) >= k:
            break
    if not viz_x:
        return np.array([]), np.array([])
    return np.array(viz_x), np.array(viz_y)


class TimeSeriesPerformanceCallback(keras.callbacks.Callback):
    """Abstract base for the per-epoch visualization callbacks of the TS trainers.

    Owns the scaffolding that was ~70% identical across the four time-series
    training scripts (nbeats / prism / tirex / mdn): the ``__init__`` +
    ``makedirs``, the ``loss`` / ``val_loss`` history accumulation, the
    ``(epoch + 1) % visualize_every_n_epochs`` gate, the
    ``create_learning_curves`` / ``create_prediction_plots`` flag checks, and the
    delegation of learning-curve rendering to
    :func:`train.common.evaluation.generate_training_curves`. Only the
    genuinely model-specific pieces are left to subclasses, via three hooks:

    - :meth:`_prepare_viz_data` — load the fixed test samples used for prediction
      plots. Default returns empty arrays; the forecasting trio override it via
      :func:`_prepare_viz_data_from_processor`, while mdn overrides it to return
      a pre-built ``viz_data`` tuple (it never touches a processor).
    - :meth:`_extend_history` — append per-model metric keys (and optionally the
      learning rate via :meth:`_track_lr`) to ``self.training_history`` each
      epoch. Default is a no-op (mdn / nbeats track nothing extra).
    - :meth:`_plot_predictions` — render the model-specific prediction plot
      (mixture PDF / backcast-forecast / 4D or 3D quantile bands). Abstract:
      the default raises :class:`NotImplementedError`.

    Per SYSTEM.md, Keras callbacks are NOT
    ``@keras.saving.register_keras_serializable`` — this class carries no
    serialization decorator.

    Args:
        config: A :class:`BaseTimeSeriesTrainingConfig` (or subclass). Supplies
            ``visualize_every_n_epochs`` and the optional
            ``create_learning_curves`` / ``create_prediction_plots`` flags
            (read via ``getattr(..., True)`` so configs lacking them — mdn —
            keep "always plot" behavior).
        save_dir: Directory for the emitted plot files (created if absent).
        model_name: Short label used by subclasses when titling plots.
    """

    BASE_HISTORY_KEYS = ['loss', 'val_loss']

    def __init__(self, config, save_dir: str, model_name: str = "model"):
        super().__init__()
        self.config = config
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(self.save_dir, exist_ok=True)
        self.training_history: Dict[str, List[float]] = {
            k: [] for k in self.BASE_HISTORY_KEYS
        }
        self.viz_test_data = self._prepare_viz_data()

    # ------------------------------------------------------------------ #
    # Hooks (overridable by subclasses)
    # ------------------------------------------------------------------ #
    def _prepare_viz_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the fixed test samples for prediction plots. Default: empties."""
        return np.array([]), np.array([])

    def _extend_history(self, logs: dict) -> None:
        """Append model-specific metrics to ``self.training_history``. Default: no-op."""
        pass

    def _track_lr(self, logs: dict) -> None:
        """Opt-in helper: append the current learning rate under ``'lr'``.

        Subclasses that track LR (prism / tirex) call this from their
        :meth:`_extend_history`. The value comes from ``logs['lr']`` when Keras
        provides it, else from the optimizer's ``learning_rate`` (guarded — on
        any failure the epoch is simply skipped, never raising).
        """
        lr = logs.get('lr')
        if lr is None:
            try:
                lr = float(keras.ops.convert_to_numpy(
                    self.model.optimizer.learning_rate))
            except Exception:
                return
        self.training_history.setdefault('lr', []).append(lr)

    def _plot_predictions(self, epoch: int) -> None:
        """Render the model-specific prediction plot. Abstract — must override."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared epoch loop + learning-curve delegation
    # ------------------------------------------------------------------ #
    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        self.training_history['loss'].append(logs.get('loss', 0.0))
        self.training_history['val_loss'].append(logs.get('val_loss', 0.0))
        self._extend_history(logs)

        if (epoch + 1) % self.config.visualize_every_n_epochs == 0:
            if getattr(self.config, 'create_learning_curves', True):
                self._plot_learning_curves(epoch)
            if getattr(self.config, 'create_prediction_plots', True):
                self._plot_predictions(epoch)

    def _plot_learning_curves(self, epoch: int) -> None:
        """Delegate learning-curve rendering to the shared evaluation helper."""
        generate_training_curves(
            history=self.training_history,
            results_dir=self.save_dir,
            filename=f"learning_curves_epoch_{epoch + 1:03d}",
        )


# --------------------------------------------------------------------------- #
# Trainer orchestration (shared skeleton)
# --------------------------------------------------------------------------- #
class BaseTimeSeriesTrainer:
    """Abstract base for the four synthetic time-series trainers.

    Absorbs the ~70% of the ``nbeats`` / ``prism`` / ``tirex`` / ``mdn`` trainer
    classes that was structurally identical: the ``__init__`` (build the
    :class:`~dl_techniques.datasets.time_series.TimeSeriesGenerator`, the
    ``pattern_to_category`` map, the selected-pattern list, and the data
    processor), the byte-identical ``_select_patterns`` body (promoted verbatim
    from the nbeats original), ``_create_experiment_dir``, the
    ``create_common_callbacks`` assembly in :meth:`_make_callbacks`, the shared
    ``model.fit`` + ``model.evaluate`` in :meth:`_train_model`, the ``results.json``
    write in :meth:`_save_results`, the prism/tirex ONNX export, and the
    :meth:`run_experiment` skeleton.

    Three abstract hooks carry the genuine per-model variation and MUST be
    overridden by every subclass:

    - :meth:`_build_processor` — returns the
      :class:`WindowedTimeSeriesProcessor` subclass with model-specific
      normalization / output signature.
    - :meth:`_build_model` — full model construction + compile (loss, metrics,
      LR schedule, jit).
    - :meth:`_build_performance_callback` — the domain
      :class:`TimeSeriesPerformanceCallback` subclass holding the model-specific
      prediction-plot body.

    Two methods are overridable but have working defaults:
    :meth:`_build_results_prefix` (default ``config.experiment_name``) and
    :meth:`_make_callbacks` (mdn overrides it for ``include_analyzer=False`` /
    ``patience=15``). :meth:`_save_results` is overridable too (nbeats overrides
    it for the D-005 ``primary_loss`` serializer).

    Args:
        config: A :class:`BaseTimeSeriesTrainingConfig` (or subclass).
        generator_config: A ``TimeSeriesGeneratorConfig`` passed straight to
            :class:`~dl_techniques.datasets.time_series.TimeSeriesGenerator`.
    """

    def __init__(self, config, generator_config) -> None:
        self.config = config
        self.generator_config = generator_config
        self.generator = TimeSeriesGenerator(generator_config)

        self.all_patterns = self.generator.get_task_names()
        self.pattern_categories = self.generator.get_task_categories()
        self.pattern_to_category = {
            task: cat
            for cat in self.pattern_categories
            for task in self.generator.get_tasks_by_category(cat)
        }
        self.selected_patterns = self._select_patterns()
        self.processor = self._build_processor()
        self.model: Optional[keras.Model] = None
        self.exp_dir: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Pattern selection (canonical body, verbatim from nbeats)
    # ------------------------------------------------------------------ #
    def _select_patterns(self) -> List[str]:
        if self.config.target_categories:
            candidates = {
                p for c in self.config.target_categories
                for p in self.generator.get_tasks_by_category(c)
            }
        else:
            candidates = self.all_patterns

        selected: List[str] = []
        cat_counts: Dict[str, int] = {}
        for pattern in sorted(candidates):
            cat = self.pattern_to_category.get(pattern)
            if cat and cat_counts.get(cat, 0) < self.config.max_patterns_per_category:
                selected.append(pattern)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        if self.config.max_patterns and len(selected) > self.config.max_patterns:
            selected = random.sample(selected, self.config.max_patterns)

        logger.info(f"Selected {len(selected)} patterns for training")
        return selected

    # ------------------------------------------------------------------ #
    # Experiment dir + results prefix
    # ------------------------------------------------------------------ #
    def _build_results_prefix(self) -> str:
        """Directory-name prefix. Default: the experiment name.

        prism / tirex override this to fold in ``preset``/``mode`` or
        ``model_type``.
        """
        return self.config.experiment_name

    def _create_experiment_dir(self, prefix: Optional[str] = None) -> str:
        """Create ``{result_dir}/{prefix}_{timestamp}`` and return it."""
        prefix = prefix or self._build_results_prefix()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(self.config.result_dir, f"{prefix}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _make_callbacks(self, exp_dir: str) -> List:
        """Assemble Pattern-2 callbacks + the model-specific performance callback.

        Reproduces the trio's ``create_common_callbacks`` call (8 kwargs,
        ``monitor='val_loss'``, ``include_terminate_on_nan=True``,
        ``patience=25``, conditional analyzer with the lightweight
        :class:`AnalysisConfig`) and appends
        :meth:`_build_performance_callback`. mdn overrides this for
        ``include_analyzer=False`` / ``patience=15``.
        """
        viz_dir = os.path.join(exp_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        callbacks, _ = create_common_callbacks(
            model_name=self.config.experiment_name,
            results_dir_prefix=exp_dir,
            monitor="val_loss",
            patience=25,
            use_lr_schedule=self.config.use_warmup,
            include_terminate_on_nan=True,
            include_analyzer=self.config.perform_deep_analysis,
            analyzer_config=AnalysisConfig(
                analyze_weights=True, analyze_spectral=True,
                analyze_calibration=False, analyze_information_flow=False,
                analyze_training_dynamics=False, verbose=False),
            analyzer_start_epoch=self.config.analysis_start_epoch,
            analyzer_epoch_frequency=self.config.analysis_frequency,
        )
        callbacks.append(self._build_performance_callback(viz_dir))
        return callbacks

    # ------------------------------------------------------------------ #
    # Train / evaluate
    # ------------------------------------------------------------------ #
    def _train_model(self, data_pipeline: Dict[str, Any], exp_dir: str) -> Dict[str, Any]:
        """Shared ``model.fit`` + ``model.evaluate`` (identical across all four)."""
        callbacks = self._make_callbacks(exp_dir)

        history = self.model.fit(
            data_pipeline['train_ds'],
            validation_data=data_pipeline['val_ds'],
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_steps=data_pipeline['validation_steps'],
            callbacks=callbacks, verbose=1
        )

        logger.info("Evaluating on test set")
        test_results = self.model.evaluate(
            data_pipeline['test_ds'], steps=data_pipeline['test_steps'],
            verbose=1, return_dict=True
        )

        return {
            'history': history.history,
            'test_metrics': {k: float(v) for k, v in test_results.items()},
            'final_epoch': len(history.history['loss'])
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _save_results(self, results: Dict[str, Any], exp_dir: str,
                      extra_fields: Optional[Dict[str, Any]] = None) -> None:
        """Write the standard 4-key ``results.json`` (+ optional ``extra_fields``).

        Uses :data:`train.common.config_io.json_numpy_default` (prism/tirex/mdn).
        nbeats overrides this for the D-005 ``primary_loss`` str-fallback.
        """
        serializable = {
            'history': results['history'],
            'test_metrics': results['test_metrics'],
            'final_epoch': results['final_epoch'],
            'config': self.config.__dict__,
        }
        if extra_fields:
            serializable.update(extra_fields)
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(serializable, f, indent=4, default=json_numpy_default)

    def _export_to_onnx(self, model_path: str, exp_dir: str) -> Optional[str]:
        """Export the trained model to ONNX (prism/tirex). No-op otherwise.

        Guarded by ``getattr(config, 'export_onnx', False)`` (mdn/nbeats lack the
        flag → skip) and ``os.path.exists(model_path)`` (early-stop may never fire
        → checkpoint absent). Conversion failures are caught + logged (matching
        the originals' silent-catch), returning ``None``.
        """
        if not getattr(self.config, 'export_onnx', False):
            return None
        if not os.path.exists(model_path):
            logger.warning(f"ONNX export skipped: checkpoint absent ({model_path})")
            return None

        onnx_path = os.path.join(exp_dir, 'model.onnx')
        try:
            logger.info(f"Exporting to ONNX: {onnx_path}")
            best_model = keras.saving.load_model(model_path, compile=False)
            input_signature = [
                keras.InputSpec(
                    shape=(None, self.config.context_len, self.processor.num_features),
                    dtype="float32"
                )
            ]
            best_model.export(
                onnx_path, format="onnx",
                input_signature=input_signature,
                opset_version=self.config.onnx_opset_version, verbose=True
            )
            logger.info(f"ONNX export successful: {onnx_path}")
            return onnx_path
        except Exception as e:
            logger.error(f"ONNX export failed: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Orchestration skeleton
    # ------------------------------------------------------------------ #
    def run_experiment(self) -> Dict[str, Any]:
        """Shared skeleton: dir → datasets → build → train → save → return."""
        logger.info(f"Starting {self.config.experiment_name} training experiment")
        self.exp_dir = self._create_experiment_dir()
        logger.info(f"Results: {self.exp_dir}")

        data_pipeline = self.processor.prepare_datasets()
        self.model = self._build_model()
        logger.info(f"Model params: {self.model.count_params():,}")
        self.model.summary(print_fn=logger.info)

        training_results = self._train_model(data_pipeline, self.exp_dir)
        if self.config.save_results:
            self._save_results(training_results, self.exp_dir)

        return {
            'config': self.config, 'experiment_dir': self.exp_dir,
            'training_results': training_results, 'results_dir': self.exp_dir
        }

    # ------------------------------------------------------------------ #
    # Abstract hooks (must override)
    # ------------------------------------------------------------------ #
    def _build_processor(self) -> WindowedTimeSeriesProcessor:
        """Return the model-specific data processor. Abstract — must override."""
        raise NotImplementedError

    def _build_model(self) -> keras.Model:
        """Construct + compile the model. Abstract — must override."""
        raise NotImplementedError

    def _build_performance_callback(self, viz_dir: str):
        """Return the domain performance callback. Abstract — must override."""
        raise NotImplementedError
