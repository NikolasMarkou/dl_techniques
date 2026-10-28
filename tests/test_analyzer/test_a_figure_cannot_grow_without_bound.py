"""A saved analyzer figure cannot grow without bound.

Measured: a trained classifier whose confidences are all saturated near 1.0 gave the
calibration figure a 4,215 x 18,346,848 px tight bounding box, because the mean-confidence
labels were placed at ``ylim[1] * 0.98`` in DATA coordinates and the y-range of that panel
was about 1e-6. ``savefig(bbox_inches='tight')`` then allocated a ~37 GB Agg buffer and the
kernel OOM-killed the trainer (``train_cliffordnet`` end-of-run analysis, about half of
the runs of one tiny job).

Two guards: the labels no longer depend on the data range, and
``BaseVisualizer._save_figure`` refuses a figure past ``MAX_FIGURE_SIDE_PIXELS`` instead of
allocating it. The regression arm measures the tight bounding box and never saves the
oversized figure, so a mutation cannot reproduce the allocation.
"""

import matplotlib

matplotlib.use("Agg")

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dl_techniques.analyzer import AnalysisConfig, DataInput, ModelAnalyzer  # noqa: E402
from dl_techniques.analyzer.visualizers import base as vbase  # noqa: E402
from dl_techniques.analyzer.visualizers import BaseVisualizer  # noqa: E402
from dl_techniques.visualization import PlotConfig, VisualizationContext  # noqa: E402
from dl_techniques.visualization.training_performance import (  # noqa: E402
    OverfittingAnalysis,
    TrainingHistory,
)


def _saturated_model_and_data():
    """Confidences ``1 - 2*exp(-x)`` for x in 12..16: all within about 1e-5 of 1.0."""
    inp = keras.Input((1,))
    dense = keras.layers.Dense(3, use_bias=False, kernel_initializer="zeros")
    model = keras.Model(inp, dense(inp), name="saturated")
    dense.kernel.assign(np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    x = np.linspace(12.0, 16.0, 8, dtype="float32")[:, None]
    y = np.array([0, 1, 2, 0, 1, 2, 0, 0])
    return model, x, y


def _measure_figures(monkeypatch, tmp_path):
    """Run the calibration analysis; return {figure name: (width_px, height_px)}.

    Replaces ``_save_figure`` by a measurement, so no figure is ever rasterized.
    """
    sizes = {}

    def measure(self, fig, name):
        box = fig.get_tightbbox()
        sizes[name] = (box.width * self.config.dpi, box.height * self.config.dpi)

    monkeypatch.setattr(vbase.BaseVisualizer, "_save_figure", measure)
    model, x, y = _saturated_model_and_data()
    config = AnalysisConfig(
        analyze_weights=False, analyze_calibration=True, analyze_information_flow=False,
        analyze_training_dynamics=False, analyze_spectral=False, save_plots=True,
    )
    analyzer = ModelAnalyzer(
        models={"saturated": model}, config=config, output_dir=str(tmp_path / "out"),
        training_history={"saturated": {"loss": [1.0], "val_loss": [1.0],
                                        "accuracy": [0.3], "val_accuracy": [0.3]}},
    )
    analyzer.analyze(data=DataInput(x_data=x, y_data=y))
    return sizes


def test_the_confidences_of_the_probe_model_really_are_saturated():
    model, x, _ = _saturated_model_and_data()
    conf = keras.ops.convert_to_numpy(keras.ops.softmax(model(x), axis=-1)).max(axis=1)
    assert 1.0 - conf.min() < 2e-5 and conf.max() <= 1.0 and conf.min() > 0.99998


def test_the_calibration_figure_stays_a_normal_size_for_saturated_confidences(
    monkeypatch, tmp_path
):
    sizes = _measure_figures(monkeypatch, tmp_path)
    assert "confidence_calibration_analysis" in sizes  # anti-vacuity: it was drawn
    for name, (w, h) in sizes.items():
        assert max(w, h) < 10_000, f"{name} would be {w:.0f} x {h:.0f} px"


class _Plain(BaseVisualizer):
    def create_visualizations(self):
        pass

    def _get_models_with_data(self):
        return []


def _plain(tmp_path):
    from dl_techniques.analyzer.data_types import AnalysisResults

    return _Plain(AnalysisResults(), AnalysisConfig(), tmp_path, {})


def test_an_ordinary_figure_is_saved(tmp_path):
    fig = plt.figure(figsize=(4, 3))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    _plain(tmp_path)._save_figure(fig, "ok")
    plt.close(fig)
    assert (tmp_path / "ok.png").is_file()


def test_a_figure_past_the_limit_is_refused_not_allocated(tmp_path, monkeypatch, caplog):
    # The limit, not a real giant, is what varies: 4,200 px is "too big" at 1,000.
    monkeypatch.setattr(vbase, "MAX_FIGURE_SIDE_PIXELS", 1_000)
    fig = plt.figure(figsize=(14, 10))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    _plain(tmp_path)._save_figure(fig, "too_big")
    plt.close(fig)
    assert not (tmp_path / "too_big.png").exists()


def test_the_refusal_is_logged_with_the_figure_name_and_size(tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(vbase, "MAX_FIGURE_SIDE_PIXELS", 1_000)
    monkeypatch.setattr(vbase.logger, "error", lambda m, *a, **k: messages.append(str(m)))
    fig = plt.figure(figsize=(14, 10))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    _plain(tmp_path)._save_figure(fig, "too_big")
    plt.close(fig)
    assert len(messages) == 1
    assert "too_big" in messages[0] and "1000 px limit" in messages[0]


# 100 data-units out is ~70,000 px at 300 dpi: past the limit, yet small enough that
# Agg would still rasterize it (so removing the guard is visible as a file, not an OOM).
# Past 2^23 px Agg refuses by itself; a label 1000 units out is ~2.9 GB.
@pytest.mark.parametrize("position", [(0.5, -100), (100, 0.5)], ids=["below", "right"])
def test_a_label_far_outside_its_axes_is_caught_by_the_guard(tmp_path, position):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(*position, "stray", transform=ax.transData)
    _plain(tmp_path)._save_figure(fig, "stray")
    plt.close(fig)
    assert not (tmp_path / "stray.png").exists()


def test_the_overfit_label_stays_inside_the_figure_for_a_nearly_constant_loss(tmp_path):
    # Loss ~5.0 +- 1e-7: its y-range is ~1e-7, so a label at ``ylim[1] * 0.9`` in data
    # units (4.5) would sit millions of axes-heights below the axes.
    epochs = list(range(30))
    val = [5.0 + 1e-7 * abs(e - 8) for e in epochs]  # best at epoch 8, worse after
    history = TrainingHistory(epochs=epochs, train_loss=[5.0] * 30, val_loss=val)
    plugin = OverfittingAnalysis(
        config=PlotConfig(), context=VisualizationContext(experiment_name="", output_dir=tmp_path)
    )
    fig = plugin.create_visualization(history, patience=5)
    try:
        assert any("Overfit @" in t.get_text() for ax in fig.axes for t in ax.texts)  # drawn
        box = fig.get_tightbbox()
        assert max(box.width, box.height) < 100, f"{box.width:.0f} x {box.height:.0f} in"
    finally:
        plt.close(fig)
