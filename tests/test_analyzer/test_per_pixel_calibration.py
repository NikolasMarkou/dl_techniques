"""The analyzer on per-pixel (mask) outputs: ``(N, H, W, C)`` scores against ``(N, H, W)`` labels.

Plan 2faf890a step 12, D-020. Measured in step 9 (``findings/analyzer-on-masks.md``): before this
change calibration read ``shape[1]`` (H) as the class count and raised on every rank-4 input,
the raise was swallowed, the summary dashboard printed ECE 0.000 / Brier 0.000, and a one-hot
mask gave a wrong ECE (0.5375 for a hand 0.35).

The hand computation. Two images of 2x2 pixels, two classes. The model has no weights: the input
holds ``p`` = P(class 1) per pixel and the output is ``[1 - p, p]``, so the probabilities are
exact and sit away from every edge of the 10 equal-width bins::

    pixel  p     conf  bin  pred  label(clean)  correct
    A      .95   .95   9    1     1             yes
    B      .85   .85   8    1     1             yes
    C      .25   .75   7    0     0             yes
    D      .05   .95   9    0     1             NO
    E      .65   .65   6    1     0             NO
    F      .45   .55   5    0     0             yes
    G      .75   .75   7    1     1             yes
    H      .15   .85   8    0     0 / void      yes

clean ECE   2/8*|.95-.5| + 2/8*|.85-1| + 2/8*|.75-1| + 1/8*|.65-0| + 1/8*|.55-1| = 0.35
void ECE    (H dropped, 7 pixels) 2.65 / 7 = 0.378571...
clean Brier (2/8) * (.05^2 + .15^2 + .25^2 + .95^2 + .65^2 + .45^2 + .25^2 + .15^2) = 0.425
void Brier  (2/7) * (1.7 - .15^2) = 0.479286...

The rank-2 baseline (ECE 0.08518462230761842, Brier 0.6595650636382939) is the control of the
step-9 probe (``scratch_analyzer/probe.py`` section ``rank2``), measured BEFORE the change.
"""

import json
import logging

import keras
import numpy as np
import pytest

from dl_techniques.analyzer import AnalysisConfig, DataInput, ModelAnalyzer
from dl_techniques.analyzer.visualizers.summary_visualizer import SummaryVisualizer

VOID = 255

P1 = np.array([[[0.95, 0.85], [0.25, 0.05]],
               [[0.65, 0.45], [0.75, 0.15]]], dtype=np.float32)
Y_CLEAN = np.array([[[1, 1], [0, 1]], [[0, 0], [1, 0]]], dtype=np.int32)
Y_VOID = Y_CLEAN.copy()
Y_VOID[1, 1, 1] = VOID  # pixel H

HAND = {
    "clean": {"ece": 0.35, "brier": 0.425, "total": 8, "void": 0, "used": 8},
    "void": {"ece": 2.65 / 7.0, "brier": (2.0 / 7.0) * 1.6775, "total": 8, "void": 1, "used": 7},
}


def _x(p1=P1):
    return p1[..., None].astype(np.float32)


def _fixed_probability_model(loss=None):
    """Output ``[1 - x, x]`` at every pixel: a model with no weights."""
    inputs = keras.Input((2, 2, 1))
    one_minus = keras.layers.Rescaling(scale=-1.0, offset=1.0)(inputs)
    outputs = keras.layers.Concatenate(axis=-1)([one_minus, inputs])
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss=loss or keras.losses.SparseCategoricalCrossentropy())
    return model


def _calibration_config(**overrides):
    kwargs = dict(
        analyze_weights=False, analyze_calibration=True, analyze_information_flow=False,
        analyze_training_dynamics=False, analyze_spectral=False, verbose=False,
        random_state=0, calibration_bins=10,
    )
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


def _analyze(tmp_path, model, x, y, config, name="fixed"):
    analyzer = ModelAnalyzer({name: model}, config, output_dir=str(tmp_path / "analysis"))
    results = analyzer.analyze(data=DataInput(x_data=x, y_data=y))
    return analyzer, results


# ---------------------------------------------------------------------
# the hand computation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("tag, y", [("clean", Y_CLEAN), ("void", Y_VOID)])
def test_calibration_of_a_mask_model_equals_the_hand_computation(tmp_path, tag, y):
    _, results = _analyze(tmp_path, _fixed_probability_model(), _x(), y,
                          _calibration_config(ignore_label=VOID))
    cal = results.calibration_metrics["fixed"]
    hand = HAND[tag]
    assert cal["ece"] == pytest.approx(hand["ece"], abs=1e-6)
    assert cal["brier_score"] == pytest.approx(hand["brier"], abs=1e-6)
    assert (cal["pixels_total"], cal["pixels_void"], cal["pixels_used"]) == (
        hand["total"], hand["void"], hand["used"])


def test_a_void_pixel_moves_no_calibration_number(tmp_path):
    """Changing what the model predicts AT the void pixel changes nothing at all."""
    config = _calibration_config(ignore_label=VOID)
    _, base = _analyze(tmp_path / "a", _fixed_probability_model(), _x(), Y_VOID, config)
    moved = P1.copy()
    moved[1, 1, 1] = 0.99  # pixel H was 0.15
    _, other = _analyze(tmp_path / "b", _fixed_probability_model(), _x(moved), Y_VOID, config)
    assert other.calibration_metrics["fixed"] == base.calibration_metrics["fixed"]
    assert other.confidence_metrics["fixed"].keys() == base.confidence_metrics["fixed"].keys()
    for key, value in base.confidence_metrics["fixed"].items():
        assert np.array_equal(np.asarray(other.confidence_metrics["fixed"][key]), np.asarray(value)), key


def test_the_void_pixel_is_not_counted_as_a_wrong_prediction(tmp_path):
    """Kept in the flat arrays, label 255 gives ECE 0.4000 (measured in step 9)."""
    _, results = _analyze(tmp_path, _fixed_probability_model(), _x(), Y_VOID,
                          _calibration_config(ignore_label=VOID))
    assert results.calibration_metrics["fixed"]["ece"] != pytest.approx(0.4, abs=1e-3)


def test_an_unconfigured_void_label_is_loud_not_a_miss(tmp_path):
    """No ``ignore_label``: 255 is outside [0, C), so calibration refuses the model."""
    analyzer, results = _analyze(tmp_path, _fixed_probability_model(), _x(), Y_VOID,
                                 _calibration_config(ignore_label=None))
    assert "fixed" not in results.calibration_metrics
    assert "fixed" not in results.confidence_metrics


def test_trailing_label_axis_of_one_is_squeezed(tmp_path):
    config = _calibration_config(ignore_label=VOID)
    _, flat = _analyze(tmp_path / "a", _fixed_probability_model(), _x(), Y_VOID, config)
    _, trailing = _analyze(tmp_path / "b", _fixed_probability_model(), _x(), Y_VOID[..., None], config)
    assert trailing.calibration_metrics["fixed"] == flat.calibration_metrics["fixed"]


# ---------------------------------------------------------------------
# one-hot masks and a wrong shape: refused with both shapes named
# ---------------------------------------------------------------------

def test_a_one_hot_mask_is_refused_naming_both_shapes(tmp_path):
    """One-hot was WRONG (ECE 0.5375 against the hand 0.35) and then raised; it is refused now."""
    one_hot = np.eye(2, dtype=np.float32)[Y_CLEAN]  # (2, 2, 2, 2)
    analyzer, results = _analyze(tmp_path, _fixed_probability_model(), _x(), one_hot,
                                 _calibration_config(ignore_label=VOID))
    assert "fixed" not in results.calibration_metrics
    entry = analyzer._prediction_cache["fixed"]
    assert entry["status"] == "error"
    assert "(2, 2, 2, 2)" in entry["error"] and "(2, 2, 2)" in entry["error"]


def test_the_shape_check_raises_a_value_error_naming_both_shapes(tmp_path):
    analyzer = ModelAnalyzer({"fixed": _fixed_probability_model()}, _calibration_config(),
                             output_dir=str(tmp_path / "analysis"))
    wrong = np.zeros((2, 2, 3), dtype=np.int32)
    with pytest.raises(ValueError) as error:
        analyzer._cache_pixel_predictions("fixed", analyzer.models["fixed"], _x(), wrong)
    assert "(2, 2, 3)" in str(error.value) and "(2, 2, 2, 2)" in str(error.value)


def test_a_mask_of_only_void_pixels_is_an_error_not_an_empty_calibration(tmp_path):
    all_void = np.full_like(Y_CLEAN, VOID)
    analyzer, results = _analyze(tmp_path, _fixed_probability_model(), _x(), all_void,
                                 _calibration_config(ignore_label=VOID))
    assert "fixed" not in results.calibration_metrics
    assert "no labelled pixel" in analyzer._prediction_cache["fixed"]["error"]


# ---------------------------------------------------------------------
# the cap: an exactly uniform draw without replacement, seeded, across batches
# ---------------------------------------------------------------------

class _StubModel:
    """Predicts logits ``log([1 - p, p])`` from a one-channel image holding ``p``."""

    output_shape = (None, 2, 2, 2)

    def predict(self, x, verbose=0):
        p = np.asarray(x)[..., 0]
        return np.log(np.stack([1.0 - p, p], axis=-1))


def _cache_entry(stub_images, labels, max_pixels, seed, ignore=VOID):
    analyzer = ModelAnalyzer.__new__(ModelAnalyzer)
    analyzer.config = _calibration_config(max_pixels=max_pixels, random_state=seed, ignore_label=ignore)
    analyzer._multi_input_models = set()
    return analyzer._cache_pixel_predictions("stub", _StubModel(), stub_images, labels)


def test_the_cap_keeps_exactly_max_pixels_of_the_labelled_pixels():
    entry = _cache_entry(_x(), Y_VOID, max_pixels=3, seed=0)
    assert entry["pixel_counts"] == {"pixels_total": 8, "pixels_void": 1, "pixels_used": 3}
    assert entry["predictions"].shape == (3, 2) and entry["y_data"].shape == (3,)
    assert VOID not in entry["y_data"]


def test_the_draw_is_seeded_and_the_kept_pixels_stay_in_image_order():
    first = _cache_entry(_x(), Y_VOID, max_pixels=4, seed=7)
    again = _cache_entry(_x(), Y_VOID, max_pixels=4, seed=7)
    other = _cache_entry(_x(), Y_VOID, max_pixels=4, seed=8)
    assert np.array_equal(first["predictions"], again["predictions"])
    assert not np.array_equal(first["predictions"], other["predictions"])
    kept = first["predictions"][:, 1]  # p of each kept pixel
    original = P1.reshape(-1)
    positions = [int(np.flatnonzero(np.isclose(original, p))[0]) for p in kept]
    assert positions == sorted(positions)


def test_the_cap_is_a_uniform_draw_across_batches():
    """40 images = 3 batches of 16, 160 pixels, 40 kept: every pixel is kept 1 time in 4."""
    p = (0.05 + 0.005 * np.arange(160)).astype(np.float32).reshape(40, 2, 2)  # every pixel distinct
    labels = np.zeros((40, 2, 2), dtype=np.int32)
    draws = 300
    counts = np.zeros(160)
    for seed in range(draws):
        kept = _cache_entry(p[..., None], labels, max_pixels=40, seed=seed)["predictions"][:, 1]
        for value in kept:
            counts[int(round((float(value) - 0.05) / 0.005))] += 1
    expected, sigma = draws * 0.25, np.sqrt(draws * 0.25 * 0.75)
    assert counts.sum() == draws * 40
    assert np.abs(counts - expected).max() < 5 * sigma
    per_batch = counts.reshape(40, 4).sum(axis=1)  # per image
    assert per_batch[:16].sum() == pytest.approx(per_batch[16:32].sum(), rel=0.1)


def test_a_cap_above_the_pixel_count_keeps_everything():
    entry = _cache_entry(_x(), Y_CLEAN, max_pixels=1000, seed=0)
    assert entry["pixel_counts"]["pixels_used"] == 8
    assert np.allclose(entry["predictions"][:, 1], P1.reshape(-1))


# ---------------------------------------------------------------------
# a compile with a stock ignore_class loss
# ---------------------------------------------------------------------

def test_model_metrics_come_from_the_compile_with_a_stock_ignore_class_loss(tmp_path):
    """``model_metrics`` is ``model.evaluate``: with ``ignore_class`` it is the loss over the kept pixels.

    Steps 27 and 35 owe the arm for the void-aware metrics (accuracy, mIoU): a stock
    ``SparseCategoricalAccuracy`` scores a void pixel as wrong and stock ``MeanIoU`` raises on a
    255 label, so no stock metric is compiled here.
    """
    loss = keras.losses.SparseCategoricalCrossentropy(ignore_class=VOID)
    model = _fixed_probability_model(loss=loss)
    _, results = _analyze(tmp_path, model, _x(), Y_VOID, _calibration_config(ignore_label=VOID))
    metrics = results.model_metrics["fixed"]
    kept = Y_VOID != VOID
    p_correct = np.where(Y_VOID == 1, P1, 1.0 - P1)[kept]
    hand = float(-np.log(p_correct).mean())
    assert metrics["status"] == "success"
    assert metrics["loss"] == pytest.approx(hand, abs=1e-5)


# ---------------------------------------------------------------------
# the artifact and the figures
# ---------------------------------------------------------------------

def test_a_mask_run_writes_its_figures_and_the_counts(tmp_path):
    analyzer, _ = _analyze(tmp_path, _fixed_probability_model(), _x(), Y_VOID,
                           _calibration_config(ignore_label=VOID))
    out = analyzer.output_dir
    assert (out / "confidence_calibration_analysis.png").is_file()
    assert (out / "summary_dashboard.png").is_file()
    payload = json.loads((out / "analysis_results.json").read_text())
    assert payload["calibration_metrics"]["fixed"]["pixels_used"] == 7
    assert payload["config"]["ignore_label"] == VOID and payload["config"]["max_pixels"] == 100_000


def test_a_full_five_analysis_run_on_a_convunext_mask_model(tmp_path):
    """All five analyses, void labels, a real network: calibration present, nothing swallowed."""
    from dl_techniques.models.vision.convunext.model import create_convunext_variant

    size, classes = 16, 3
    model = create_convunext_variant(
        "tiny", (size, size, 3), use_bias=True, output_channels=classes, final_activation="linear")
    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=VOID))
    rng = np.random.default_rng(0)
    x = rng.random((20, size, size, 3), dtype=np.float32)
    y = np.digitize(x[..., 0], [1 / 3, 2 / 3]).astype(np.int32)
    y[:, :4, :4] = VOID
    config = AnalysisConfig(verbose=False, random_state=0, ignore_label=VOID,
                            memory_limit_mb=256, analyze_fisher=False)
    analyzer, results = _analyze(tmp_path, model, x, y, config, name="seg")
    cal = results.calibration_metrics["seg"]
    assert (cal["pixels_total"], cal["pixels_void"], cal["pixels_used"]) == (
        20 * size * size, 20 * 16, 20 * size * size - 20 * 16)
    assert 0.0 <= cal["ece"] <= 1.0
    assert "seg" in results.weight_stats and "seg" in results.information_flow
    payload = json.loads((analyzer.output_dir / "analysis_results.json").read_text())
    assert payload["config"]["memory_limit_mb"] == 256


# ---------------------------------------------------------------------
# rank 2 is untouched
# ---------------------------------------------------------------------

def _rank2_control():
    """The step-9 probe's control: a two-conv classifier on 120 images, 6 epochs."""
    rng = np.random.default_rng(3)
    x = rng.random((120, 8, 8, 3), dtype=np.float32)
    y = np.digitize(x[..., 0].mean(axis=(1, 2)), [0.49, 0.51]).astype(np.int32)
    keras.utils.set_random_seed(0)
    inputs = keras.Input((8, 8, 3))
    h = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    h = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(h)
    h = keras.layers.GlobalAveragePooling2D()(h)
    h = keras.layers.Dense(16, activation="relu")(h)
    model = keras.Model(inputs, keras.layers.Dense(3)(h))
    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    history = model.fit(x, y, epochs=6, batch_size=16, verbose=0, validation_split=0.25)
    return model, x, y, history.history


def test_a_rank_2_model_returns_the_numbers_it_returned_before_the_change(tmp_path):
    model, x, y, history = _rank2_control()
    config = AnalysisConfig(analyze_weights=True, analyze_calibration=True,
                            analyze_information_flow=True, analyze_training_dynamics=True,
                            analyze_spectral=True, verbose=False, random_state=0)
    analyzer = ModelAnalyzer({"clf": model}, config, output_dir=str(tmp_path / "rank2"),
                             training_history={"clf": history})
    results = analyzer.analyze(data=DataInput(x_data=x, y_data=y))
    cal = results.calibration_metrics["clf"]
    # Measured before the change, in scratch_analyzer/probe.py section rank2.
    assert cal["ece"] == pytest.approx(0.08518462230761842, abs=1e-9)
    assert cal["brier_score"] == pytest.approx(0.6595650636382939, abs=1e-9)
    assert results.model_metrics["clf"]["accuracy"] == pytest.approx(0.4416666626930237, abs=1e-6)
    assert [round(v, 4) for v in cal["per_class_ece"]] == [0.0852, 0.0502, 0.0350]
    # No per-pixel residue: no counts, nothing in the cache, and the two new config
    # fields are not in the published config block.
    assert not {"pixels_total", "pixels_void", "pixels_used"} & set(cal)
    assert "pixel_counts" not in analyzer._prediction_cache["clf"]
    assert analyzer._prediction_cache["clf"]["logits"] is not None
    payload = json.loads((analyzer.output_dir / "analysis_results.json").read_text())
    assert not {"ignore_label", "max_pixels"} & set(payload["config"])


@pytest.mark.parametrize("shape, expected", [
    ((None, 10), False),
    ((None, 8, 10), False),
    ((None, 8, 8, 3), True),
    ((None, 4, 8, 8, 3), True),
    ([(None, 4), (None, 4), (None, 4), (None, 4)], False),  # four outputs, none per-pixel
])
def test_only_a_single_output_of_rank_four_or_more_takes_the_pixel_path(shape, expected):
    class Fake:
        output_shape = shape

    assert ModelAnalyzer.outputs_per_pixel(Fake()) is expected


def test_a_model_without_a_static_output_shape_keeps_the_original_path():
    class Subclassed:
        @property
        def output_shape(self):
            raise AttributeError("not built")

    assert ModelAnalyzer.outputs_per_pixel(Subclassed()) is False


# ---------------------------------------------------------------------
# the dashboard prints n/a for a missing calibration, never 0.0
# ---------------------------------------------------------------------

def _dashboard_row(tmp_path, results, name):
    visualizer = SummaryVisualizer(results, results.config, tmp_path, {name: "#1f77b4"})
    row = []
    visualizer._add_standard_metrics_to_row(row, name, {"loss": 0.5, "accuracy": 0.9})
    return row


def test_the_dashboard_row_prints_n_a_when_calibration_is_absent(tmp_path):
    from dl_techniques.analyzer.data_types import AnalysisResults

    results = AnalysisResults(config=AnalysisConfig(verbose=False))
    row = _dashboard_row(tmp_path, results, "m")
    assert row[2:4] == ["n/a", "n/a"]
    assert "0.000" not in row[2:4]


def test_the_dashboard_row_prints_the_numbers_when_calibration_is_present(tmp_path):
    from dl_techniques.analyzer.data_types import AnalysisResults

    results = AnalysisResults(config=AnalysisConfig(verbose=False))
    results.calibration_metrics["m"] = {"ece": 0.08518462, "brier_score": 0.65956506}
    row = _dashboard_row(tmp_path, results, "m")
    assert row[2:4] == ["0.085", "0.660"]


def test_a_failed_calibration_logs_the_error_it_swallowed(tmp_path, caplog):
    """The isolation stays (F-046); what changes is that the summary can name it."""
    with caplog.at_level(logging.WARNING):
        _analyze(tmp_path, _fixed_probability_model(), _x(), Y_VOID,
                 _calibration_config(ignore_label=None))
    assert any("fixed" in record.getMessage() for record in caplog.records)
