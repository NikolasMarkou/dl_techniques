"""End-to-end stub test for ``BaseTimeSeriesTrainer`` (iter-2 Step 3, SC3).

The four real TS trainers cannot be run for the test bar (GPU / hours). The
correctness bar is a throwaway stub subclass whose ``_build_model`` is a tiny
``(context, 1) -> (horizon, 1)`` Keras model, run for 1 epoch / 2 steps on a
real (small) :class:`TimeSeriesGenerator`. It exercises the full skeleton:
``_select_patterns`` -> ``_build_processor`` -> ``_make_callbacks`` -> fit ->
evaluate -> ``_save_results``, asserting completion, the four ``results.json``
keys, at least one learning-curve file, and that ONNX is skipped.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import json
import glob
from dataclasses import dataclass

import keras
import numpy as np

from dl_techniques.datasets.time_series import TimeSeriesGeneratorConfig

from train.common.timeseries import (
    BaseTimeSeriesTrainer,
    BaseTimeSeriesTrainingConfig,
    WindowedTimeSeriesProcessor,
    TimeSeriesPerformanceCallback,
)

CONTEXT_LEN = 16
HORIZON_LEN = 4


@dataclass
class _StubConfig(BaseTimeSeriesTrainingConfig):
    """Tiny training config: 1 epoch, 2 steps, no deep analysis."""

    experiment_name: str = "stub_ts"
    epochs: int = 1
    batch_size: int = 4
    steps_per_epoch: int = 2
    perform_deep_analysis: bool = False
    visualize_every_n_epochs: int = 1
    use_warmup: bool = False
    max_patterns_per_category: int = 2


class _StubProcessor(WindowedTimeSeriesProcessor):
    """Flat default processor with small context/horizon."""


class _StubPerformanceCallback(TimeSeriesPerformanceCallback):
    """Minimal performance callback whose prediction plot is a no-op sentinel."""

    def __init__(self, config, processor, save_dir):
        self._processor = processor
        super().__init__(config, save_dir, model_name="stub")

    def _plot_predictions(self, epoch: int) -> None:
        # No-op: write a sentinel so we can prove the hook fired without plotting.
        with open(os.path.join(self.save_dir, f"pred_sentinel_{epoch + 1:03d}.txt"), "w") as f:
            f.write("ok")


class _StubTrainer(BaseTimeSeriesTrainer):
    """Throwaway trainer with a tiny compiled model and a flat processor."""

    def _build_processor(self) -> WindowedTimeSeriesProcessor:
        return _StubProcessor(
            self.config,
            self.generator,
            self.selected_patterns,
            self.pattern_to_category,
            context_len=CONTEXT_LEN,
            horizon_len=HORIZON_LEN,
            num_features=1,
            patterns_to_mix=4,
            windows_per_pattern=4,
        )

    def _build_model(self) -> keras.Model:
        model = keras.Sequential([
            keras.layers.Input(shape=(CONTEXT_LEN, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(HORIZON_LEN),
            keras.layers.Reshape((HORIZON_LEN, 1)),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def _build_performance_callback(self, viz_dir: str):
        return _StubPerformanceCallback(self.config, self.processor, viz_dir)


def _make_trainer(tmp_path):
    config = _StubConfig(result_dir=str(tmp_path))
    gen_config = TimeSeriesGeneratorConfig(
        n_samples=2000, random_seed=42, default_noise_level=0.1
    )
    return _StubTrainer(config, gen_config)


def test_select_patterns_nonempty(tmp_path):
    trainer = _make_trainer(tmp_path)
    assert isinstance(trainer.selected_patterns, list)
    assert len(trainer.selected_patterns) > 0


def test_run_experiment_end_to_end(tmp_path):
    trainer = _make_trainer(tmp_path)
    results = trainer.run_experiment()

    # 1. run_experiment returns a dict carrying the training_results.
    assert isinstance(results, dict)
    training_results = results["training_results"]
    for key in ("history", "test_metrics", "final_epoch"):
        assert key in training_results, f"missing {key} in training_results"

    # 2. exp_dir was created under tmp_path and holds results.json with the keys.
    exp_dir = results["experiment_dir"]
    assert exp_dir is not None
    assert str(tmp_path) in exp_dir
    results_json = os.path.join(exp_dir, "results.json")
    assert os.path.exists(results_json), "results.json not written"
    with open(results_json) as f:
        loaded = json.load(f)
    for key in ("history", "test_metrics", "final_epoch", "config"):
        assert key in loaded, f"missing {key} in results.json"

    # 3. at least one learning-curve file was written by the base callback.
    viz_dir = os.path.join(exp_dir, "visualizations")
    curves = glob.glob(os.path.join(viz_dir, "learning_curves_epoch_*"))
    assert len(curves) >= 1, f"no learning-curve files in {viz_dir}"

    # 4. no ONNX export (export_onnx absent on the stub config).
    onnx_files = glob.glob(os.path.join(exp_dir, "*.onnx"))
    assert onnx_files == [], f"unexpected ONNX export: {onnx_files}"
