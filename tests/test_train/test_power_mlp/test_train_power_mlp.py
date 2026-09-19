"""Behavioural guards for ``src/train/power_mlp/`` (trainer + visualization).

Each guard below exists because the pre-fix trainer got the thing wrong, and each
was proven RED by injecting the pre-fix behaviour (see the plan's progress.md):

- ``effective_hidden_units`` must START with the input width. ``PowerMLP`` reads
  ``hidden_units[0]`` as the input width and drops it, so the old table
  ``[256, 128, 64, 10]`` silently trained ``[128, 64]``.
- Labels are integer and the loss is ``SparseCategoricalCrossentropy``. The old
  one-hot + ``categorical_crossentropy`` made ``ModelAnalyzer`` fail with a rank
  mismatch, which it swallows into ``status: evaluation_failed`` while the log
  says "completed successfully". The smoke test therefore reads the status from
  the file ON DISK, never from the trainer's own log line.
- ``lr`` must be a column of ``training_log.csv`` (callback ordering).
- The dashboard draws ``Smoothed loss`` iff more than 5 epochs exist and never a
  blank panel.
- Every run lands in ``<repo>/results/<name>/`` from any cwd; every test here
  routes through ``tmp_path`` (a conftest autouse guard fails on any write to the
  repo-root ``results/``).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import types
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import train.common.classification_viz as shared_viz  # noqa: E402
import train.common.run_artifacts as run_artifacts  # noqa: E402
from train.common import run_summary  # noqa: E402
import train.power_mlp.train_power_mlp as tpm  # noqa: E402
import train.power_mlp.visualization as viz  # noqa: E402
from dl_techniques.models.general_purpose.power_mlp.model import PowerMLP  # noqa: E402
from keras.src.trainers.trainer import Trainer  # noqa: E402

REPO_ROOT = Path(tpm.__file__).resolve().parents[3]


# ---------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------


def _synthetic_mnist_split(n: int, rng: np.random.Generator):
    """``(x, y)`` shaped like ``load_dataset('mnist')``: 28x28x3 float in [0, 1].

    Every class occurs, the class signal is a bright patch at a class-specific
    place buried in noise (learnable in 2 epochs, not perfectly), and the 3
    channels are identical, like the real loader.
    """
    y = (np.arange(n) % 10).astype(np.uint8)
    rng.shuffle(y)
    x = rng.random((n, 28, 28, 1), dtype=np.float32) * 0.6
    for c in range(10):
        x[y == c, 2 * c:2 * c + 7, 2 * c:2 * c + 7, :] += 0.5
    x = np.clip(x, 0.0, 1.0)
    return np.repeat(x, 3, axis=-1), y


def _fake_load_dataset(n_train: int = 640, n_test: int = 1200):
    """A drop-in for ``train.common.load_dataset`` returning synthetic MNIST."""

    def load(name, *args, **kwargs):
        assert name == "mnist", name
        rng = np.random.default_rng(7)
        train = _synthetic_mnist_split(n_train, rng)
        test = _synthetic_mnist_split(n_test, rng)
        return train, test, (28, 28, 3), 10

    return load


@pytest.fixture
def synthetic_mnist(monkeypatch):
    monkeypatch.setattr(tpm, "load_dataset", _fake_load_dataset())


# ---------------------------------------------------------------------
# (ii) architecture: the input width is prepended, the model reads it back
# ---------------------------------------------------------------------


@pytest.mark.parametrize("architecture", list(tpm.ARCHITECTURE_HIDDEN_WIDTHS))
@pytest.mark.parametrize("dataset,input_dim", [("mnist", 784), ("cifar10", 3072)])
def test_effective_hidden_units_is_input_then_hidden_then_classes(
        architecture, dataset, input_dim) -> None:
    units = tpm.effective_hidden_units(dataset, architecture, input_dim, 10)
    assert units[0] == input_dim, "entry 0 is the input width (PowerMLP drops it)"
    assert units[-1] == 10
    assert units[1:-1] == tpm.ARCHITECTURE_HIDDEN_WIDTHS[architecture][dataset]


def test_the_default_config_builds_the_486218_param_model() -> None:
    """The DEFAULT model (BN on, D-025) has 486,218 params; without BN 484,426.

    Derived from the code, not typed in: 484,426 + 4 * (256 + 128 + 64) = 486,218
    (gamma, beta, moving mean, moving variance per hidden unit).
    """
    config = tpm.TrainingConfig()
    units = tpm.effective_hidden_units("mnist", "default", 784, 10)
    counts = {}
    for bn in (False, True):
        model = PowerMLP(
            hidden_units=units, k=config.k, batch_normalization=bn,
            output_activation="softmax",
        )
        model.build((None, 784))
        counts[bn] = model.count_params()
    assert config.batch_normalization is True
    assert counts == {False: 484_426, True: 486_218}
    assert counts[True] - counts[False] == 4 * (256 + 128 + 64)


def test_the_cifar10_default_config_builds_the_3475594_param_model_without_bn() -> None:
    """CIFAR-10 defaults to BN OFF (D-028): 3,475,594 params, 3,479,178 with BN.

    Derived from the code: 3,475,594 + 4 * (512 + 256 + 128) = 3,479,178 (the 6 grid
    runs report exactly these two counts).
    """
    config = tpm.TrainingConfig(dataset="cifar10")
    units = tpm.effective_hidden_units("cifar10", "default", 3072, 10)
    counts = {}
    for bn in (False, True):
        model = PowerMLP(
            hidden_units=units, k=config.k, batch_normalization=bn,
            output_activation="softmax",
        )
        model.build((None, 3072))
        counts[bn] = model.count_params()
    assert config.batch_normalization is False
    assert counts == {False: 3_475_594, True: 3_479_178}
    assert counts[True] - counts[False] == 4 * (512 + 256 + 128)


def test_the_built_model_has_the_preset_widths_and_params() -> None:
    """784 -> 256/128/64 -> 10 for mnist default: 3 hidden layers, 484,426 params."""
    units = tpm.effective_hidden_units("mnist", "default", 784, 10)
    assert units == [784, 256, 128, 64, 10]
    model = PowerMLP(hidden_units=units, k=2, output_activation="softmax")
    model.build((None, 784))
    assert [layer.units for layer in model.hidden_layers] == [256, 128, 64]
    assert model.output_layer.units == 10
    assert model.count_params() == 484_426


def test_unknown_architecture_or_dataset_raises() -> None:
    with pytest.raises(ValueError):
        tpm.effective_hidden_units("mnist", "gigantic", 784, 10)
    with pytest.raises(ValueError):
        tpm.effective_hidden_units("svhn", "default", 784, 10)


# ---------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------


def test_prepare_data_shapes_dtypes_and_integer_labels(synthetic_mnist) -> None:
    (xt, yt), (xv, yv), (xs, ys), info = tpm.prepare_data("mnist", 0.1, 0, "unit")
    assert xt.shape == (576, 784) and xv.shape == (64, 784) and xs.shape == (1200, 784)
    assert xt.dtype == np.float32
    for y in (yt, yv, ys):
        assert y.ndim == 1, "labels are (N,) integers, not one-hot (N, C)"
        assert np.issubdtype(y.dtype, np.integer)
    assert info["input_dim"] == 784 and info["num_classes"] == 10
    assert info["image_shape"] == (28, 28, 1)


def test_prepare_data_standardizes_with_the_mnist_constants(monkeypatch) -> None:
    """A constant image at the MNIST mean maps to 0, one std above maps to 1."""
    def load(name, *a, **k):
        x = np.zeros((20, 28, 28, 3), dtype=np.float32)
        x[:10] = tpm.MNIST_MEAN
        x[10:] = tpm.MNIST_MEAN + tpm.MNIST_STD
        y = np.arange(20, dtype=np.uint8) % 10
        return (x, y), (x[:4].copy(), y[:4].copy()), (28, 28, 3), 10

    monkeypatch.setattr(tpm, "load_dataset", load)
    (xt, _), (xv, _), (xs, _), info = tpm.prepare_data("mnist", 0.5, 0, "standardize")
    everything = np.concatenate([xt, xv])
    values = np.unique(np.round(everything, 4))
    assert set(values.tolist()) == {0.0, 1.0}
    assert xs.shape == (4, 784)
    np.testing.assert_allclose(info["mean"], [0.1307], rtol=1e-6)
    np.testing.assert_allclose(info["std"], [0.3081], rtol=1e-6)


def _split_identities(synthetic_split_seed: int):
    (xt, yt), (xv, yv), _, _ = tpm.prepare_data("mnist", 0.1, synthetic_split_seed, "unit")
    return xt, xv


def test_split_is_disjoint_seed_deterministic_and_seed_dependent(synthetic_mnist) -> None:
    def row_keys(x):
        return {row.tobytes() for row in x}

    xt0, xv0 = _split_identities(0)
    xt0b, xv0b = _split_identities(0)
    xt1, xv1 = _split_identities(1)

    assert row_keys(xt0).isdisjoint(row_keys(xv0)), "train and val overlap"
    assert len(row_keys(xt0)) == len(xt0) and len(row_keys(xv0)) == len(xv0)
    np.testing.assert_array_equal(xv0, xv0b)
    np.testing.assert_array_equal(xt0, xt0b)
    assert not np.array_equal(xv0, xv1), "the split ignores the seed"


def test_validation_is_never_the_test_set(synthetic_mnist) -> None:
    (_, _), (xv, _), (xs, _), _ = tpm.prepare_data("mnist", 0.1, 0, "unit")
    test_rows = {row.tobytes() for row in xs}
    assert not any(row.tobytes() in test_rows for row in xv)


@pytest.mark.parametrize("split", [0.0, 1.0, 1.5, -0.2])
def test_out_of_range_validation_split_raises(synthetic_mnist, split) -> None:
    with pytest.raises(ValueError, match="validation_split"):
        tpm.prepare_data("mnist", split, 0, "unit")
    with pytest.raises(ValueError, match="validation_split"):
        tpm.TrainingConfig(validation_split=split)


def test_a_split_that_holds_out_zero_samples_raises(synthetic_mnist) -> None:
    with pytest.raises(ValueError, match="0 of"):
        tpm.prepare_data("mnist", 1e-6, 0, "unit")


@pytest.mark.skipif(
    not (Path.home() / ".keras" / "datasets" / "mnist.npz").exists(),
    reason="real MNIST is not cached; the suite never downloads",
)
def test_real_mnist_is_784_features_standardized_and_channels_are_identical(monkeypatch) -> None:
    """Loads the cached real MNIST once (about a second) through the real loader."""
    real = tpm.load_dataset
    seen = {}

    def spy(name, *a, **k):
        out = real(name, *a, **k)
        (x_train, _), _, shape, _ = out
        seen["channels_equal"] = bool(
            np.array_equal(x_train[:500, ..., 0], x_train[:500, ..., 1])
            and np.array_equal(x_train[:500, ..., 0], x_train[:500, ..., 2])
        )
        seen["shape"] = tuple(shape)
        return out

    monkeypatch.setattr(tpm, "load_dataset", spy)
    (xt, yt), (xv, yv), (xs, ys), info = tpm.prepare_data("mnist", 0.1, 0, "standardize")

    assert seen["channels_equal"], "loader no longer triplicates the channel: channel 0 is lossy"
    assert (len(xt), len(xv), len(xs)) == (54000, 6000, 10000)
    assert xt.shape[1] == xv.shape[1] == xs.shape[1] == 784
    assert abs(float(xt.mean())) < 0.01, float(xt.mean())
    assert abs(float(xt.std()) - 1.0) < 0.01, float(xt.std())
    assert np.issubdtype(yt.dtype, np.integer) and int(yt.max()) == 9
    assert info["image_shape"] == (28, 28, 1)


# ---------------------------------------------------------------------
# Optimizer / regularizer wiring
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", tpm.OPTIMIZERS)
def test_every_optimizer_choice_builds_and_carries_weight_decay_and_clipnorm(name) -> None:
    opt = tpm.create_optimizer(name, 3e-4, 0.05)
    assert isinstance(opt, keras.optimizers.Optimizer)
    assert opt.weight_decay == pytest.approx(0.05)
    assert opt.clipnorm == pytest.approx(tpm.GRADIENT_CLIP_NORM)
    assert float(keras.ops.convert_to_numpy(opt.learning_rate)) == pytest.approx(3e-4, rel=1e-5)


@pytest.mark.parametrize("name", tpm.OPTIMIZERS)
def test_default_weight_decay_is_zero_for_every_optimizer(name) -> None:
    """The default builds NO decay, notably for ``adamw`` (whose builder falls
    back to 0.004 when it is handed no ``weight_decay`` key at all)."""
    assert tpm.create_optimizer(name, 3e-4, tpm.TrainingConfig().weight_decay).weight_decay == 0.0


@pytest.mark.parametrize("name", tpm.OPTIMIZERS)
def test_weight_decay_actually_shrinks_weights_under_a_zero_gradient(name) -> None:
    """The attribute is set AND applied: with a zero gradient every optimizer's
    update is 0, so the only movement is ``w * (1 - lr * wd)``."""
    lr, wd = 0.1, 0.5
    for decay, expected in ((wd, 1.0 - lr * wd), (0.0, 1.0)):
        opt = tpm.create_optimizer(name, lr, decay)
        w = keras.Variable(np.ones((4,), dtype="float32"))
        opt.apply_gradients([(keras.ops.zeros((4,)), w)])
        np.testing.assert_allclose(keras.ops.convert_to_numpy(w), expected, rtol=1e-5)


def test_unknown_optimizer_name_raises_instead_of_falling_back() -> None:
    with pytest.raises(ValueError, match="Unknown optimizer"):
        tpm.create_optimizer("lion", 1e-3, 0.0)
    with pytest.raises(ValueError, match="optimizer"):
        tpm.TrainingConfig(optimizer="lion")


# ---------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------


def test_default_run_dir_is_repo_root_results_from_any_cwd(monkeypatch, tmp_path) -> None:
    """PATH only: nothing is created under the repo ``results/``."""
    monkeypatch.chdir(tmp_path)
    config = tpm.TrainingConfig()
    assert config.output_dir == "results"
    run_dir = tpm.resolved_run_dir(config)
    assert run_dir.parent == REPO_ROOT / "results"
    assert run_dir.name == config.experiment_name
    assert tmp_path not in run_dir.parents
    assert (REPO_ROOT / "src") not in run_dir.parents


class _Probe(Exception):
    pass


def test_train_model_hands_the_repo_root_path_to_prepare_run_dir(monkeypatch, tmp_path) -> None:
    """The wiring, not just the helper: ``train_model`` resolves before it creates."""
    monkeypatch.chdir(tmp_path)
    seen = {}

    def spy(config, output_dir=None, **kwargs):
        seen["output_dir"] = Path(output_dir)
        raise _Probe

    monkeypatch.setattr(tpm, "prepare_run_dir", spy)
    config = tpm.TrainingConfig(experiment_name="probe_run")
    with pytest.raises(_Probe):
        tpm.train_model(config)
    assert seen["output_dir"] == REPO_ROOT / "results" / "probe_run"
    assert not (tmp_path / "results").exists()


# ---------------------------------------------------------------------
# (iv) end-to-end smoke through tmp_path
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """One real 2-epoch run on the DEFAULTS (lecun_normal, unit, epoch analysis off, FINAL ``run_model_analysis`` on).

    ``load_dataset`` is patched to synthetic arrays; nothing else is mocked.
    ``compile`` and ``fit`` are wrapped (call-through) only to record what the trainer really passed.
    """
    out_root = tmp_path_factory.mktemp("power_mlp_smoke")
    seen: dict = {}
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(tpm, "load_dataset", _fake_load_dataset())
        # Patched on Keras' ``Trainer`` mixin, not on PowerMLP: Keras compares
        # ``model.__class__.compile`` with ``Trainer.compile`` and refuses to
        # call compile() while loading a model that differs (the trainer
        # reloads best_model.keras). The first call is the trainer's own
        # compile / fit, so `setdefault` records that one.
        real_compile, real_fit = Trainer.compile, keras.Model.fit

        def compile_spy(self, *args, **kwargs):
            seen.setdefault("loss", kwargs.get("loss"))
            return real_compile(self, *args, **kwargs)

        def fit_spy(self, x=None, y=None, *args, **kwargs):
            seen.setdefault("y_dtype", np.asarray(y).dtype)
            seen.setdefault("y_ndim", np.asarray(y).ndim)
            return real_fit(self, x, y, *args, **kwargs)

        real_grid = run_summary.plot_confident_errors

        def grid_spy(x, y_true, probs, out_path, image_shape=None, mean=None, std=None, *a, **k):
            seen["grid"] = (image_shape, mean, std)
            return real_grid(x, y_true, probs, out_path, image_shape, mean, std, *a, **k)

        mp.setattr(Trainer, "compile", compile_spy)
        mp.setattr(keras.Model, "fit", fit_spy)
        mp.setattr(run_summary, "plot_confident_errors", grid_spy)

        config = tpm.config_from_args(tpm.parse_arguments([
            "--epochs", "2", "--batch-size", "64", "--seed", "3",
            "--output-dir", str(out_root),
            "--experiment-name", "smoke_run",
        ]))
        summary = tpm.train_model(config)
        data = tpm.prepare_data(
            "mnist", config.validation_split, config.seed, config.input_scaling
        )
    finally:
        mp.undo()
    return types.SimpleNamespace(
        out_root=out_root, run_dir=out_root / "smoke_run", summary=summary,
        config=config, seen=seen, data=data,
    )


def test_smoke_hands_the_shared_figure_writer_the_data_shape_and_statistics(smoke) -> None:
    """The trainer must forward ``image_shape``, ``mean`` and ``std`` (flat rows need the
    shape; the mean and std un-standardize the grid)."""
    image_shape, mean, std = smoke.seen["grid"]
    assert image_shape == (28, 28, 1)
    assert mean is not None and std is not None
    info = smoke.data[3]
    np.testing.assert_array_equal(mean, info["mean"])
    np.testing.assert_array_equal(std, info["std"])


def test_smoke_run_directory_inventory(smoke) -> None:
    run = smoke.run_dir
    assert [p.name for p in smoke.out_root.iterdir()] == ["smoke_run"], (
        "one run must create exactly one directory (no orphan from create_callbacks)"
    )
    expected = [
        "config.json", "training_log.csv", "training_history.json", "best_model.keras",
        "final_model.keras", "results_summary.json", "run.log",
        "visualizations/training_dashboard.png", "visualizations/confusion_matrix.png",
        "visualizations/per_class_metrics.png", "visualizations/confidence_calibration.png",
        "visualizations/misclassifications.png", "visualizations/classification_report.json",
        "model_analysis/analysis_results.json",
    ]
    missing = [name for name in expected if not (run / name).is_file()]
    assert missing == [], f"missing from the run dir: {missing}"
    for name in expected:
        assert (run / name).stat().st_size > 0, name
    assert not (run / "epoch_analysis").exists(), (
        "the per-epoch analyzer is opt-in (--epoch-analysis); the default run must not create it"
    )
    assert not (run / "training_history.png").exists()
    assert not (run / "training_summary.txt").exists()
    assert list(run.glob("powermlp_*_final.keras")) == []
    assert str(run).startswith(str(smoke.out_root)), "the run must land under --output-dir"


def test_smoke_csv_has_the_lr_column_with_the_lr_used(smoke) -> None:
    with open(smoke.run_dir / "training_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert "lr" in rows[0], f"header lacks lr: {list(rows[0])}"
    assert len(rows) == 2
    lrs = [float(r["lr"]) for r in rows]
    assert all(np.isfinite(lrs))
    assert lrs[0] == pytest.approx(smoke.config.learning_rate, rel=1e-5)
    history = json.loads((smoke.run_dir / "training_history.json").read_text())
    for key in ("lr", "loss", "val_loss", "accuracy", "val_accuracy"):
        assert len(history[key]) == 2, key


def test_smoke_labels_are_integer_and_the_loss_is_sparse(smoke) -> None:
    assert np.issubdtype(smoke.seen["y_dtype"], np.integer)
    assert smoke.seen["y_ndim"] == 1
    loss = smoke.seen["loss"]
    assert isinstance(loss, keras.losses.SparseCategoricalCrossentropy), type(loss)
    assert loss.get_config()["from_logits"] is False


def test_smoke_analyzer_status_read_from_disk_is_success_and_matches_evaluate(smoke) -> None:
    """The status in ``analysis_results.json`` is the assertion: the analyzer
    swallows evaluation errors and the run log still says "completed"."""
    path = smoke.run_dir / "model_analysis" / "analysis_results.json"
    metrics = json.loads(path.read_text())["model_metrics"][smoke.config.experiment_name]
    assert metrics["status"] != "evaluation_failed", metrics
    assert metrics["status"] == "success", metrics
    assert metrics["loss"] is not None and metrics["accuracy"] is not None

    # Same slice run_model_analysis uses: the FIRST 1000 test samples.
    _, _, (x_test, y_test), _ = smoke.data
    x, y = x_test[:1000], y_test[:1000]
    model = keras.models.load_model(smoke.run_dir / "final_model.keras")
    probs = model.predict(x, verbose=0)
    accuracy = float(np.mean(probs.argmax(axis=1) == y))
    assert metrics["accuracy"] == pytest.approx(accuracy, abs=1e-3)
    ce = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-7, 1.0))))
    assert metrics["loss"] == pytest.approx(ce, rel=1e-2)

    # And the trainer reports the very same thing in its own summary.
    assert smoke.summary["analyzer"]["status"] == "success"
    on_disk = json.loads((smoke.run_dir / "results_summary.json").read_text())
    assert on_disk["analyzer"]["status"] == "success"


def test_smoke_summary_is_self_consistent(smoke) -> None:
    on_disk = json.loads((smoke.run_dir / "results_summary.json").read_text())
    with open(smoke.run_dir / "training_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    val_losses = [float(r["val_loss"]) for r in rows]

    assert on_disk["effective_hidden_units"] == [784, 256, 128, 64, 10]
    # 484,426 is the no-BN count; the default (BN on, D-025) adds 2 * (256 + 128 + 64)
    # BatchNormalization gamma/beta (trainable) plus the same number of moving
    # mean/variance (counted by Keras): 486,218 = 484,426 + 1,792.
    assert on_disk["batch_normalization"] is True
    assert on_disk["params"] == 486_218
    assert on_disk["epochs_run"] == 2 and on_disk["epochs_requested"] == 2
    assert on_disk["best_epoch"] == int(np.argmin(val_losses)) + 1
    assert Path(on_disk["run_dir"]) == smoke.run_dir.resolve()
    assert on_disk["seed"] == 3 and on_disk["weight_decay"] == 0.0
    assert on_disk["test_metrics_final"]["accuracy"] > 0.0
    assert on_disk["test_metrics_best"]["accuracy"] == pytest.approx(
        on_disk["test_metrics_final"]["accuracy"], abs=1e-6
    )
    assert on_disk["model_loading_validated"] is True
    config = json.loads((smoke.run_dir / "config.json").read_text())
    assert config["epochs"] == 2 and config["experiment_name"] == "smoke_run"


def test_smoke_records_the_init_scale_diagnostics(smoke) -> None:
    """The defaults start near ``ln(C)``: ratio == loss / ln(10), and no warning."""
    on_disk = json.loads((smoke.run_dir / "results_summary.json").read_text())
    loss = on_disk["initial_loss_sanity_eval"]["loss"]
    assert on_disk["initial_loss_ratio"] == pytest.approx(loss / np.log(10.0), rel=1e-9)
    assert on_disk["init_scale_warning"] is False
    assert 0.5 < on_disk["initial_loss_ratio"] <= 3.0, on_disk["initial_loss_ratio"]
    assert on_disk["kernel_initializer"] == smoke.config.kernel_initializer == "lecun_normal"
    assert on_disk["input_scaling"] == smoke.config.input_scaling == "unit"
    config = json.loads((smoke.run_dir / "config.json").read_text())
    assert config["kernel_initializer"] == "lecun_normal"
    assert config["input_scaling"] == "unit"
    assert config["epoch_analysis"] is False


def _no_constants(token: str):
    raise ValueError(f"non-strict JSON constant {token!r}")


def test_smoke_summary_is_strict_json_with_status_ok_and_no_failed_figure(smoke) -> None:
    """``status`` distinguishes a finished run from a diverged one; a figure that
    failed is swallowed into ``visualizations.failed``, so it is asserted EMPTY here."""
    text = (smoke.run_dir / "results_summary.json").read_text()
    on_disk = json.loads(text, parse_constant=_no_constants)
    assert on_disk["status"] == "ok" == tpm.STATUS_OK
    assert smoke.summary["status"] == "ok"
    assert on_disk["visualizations"]["failed"] == [], on_disk["visualizations"]
    assert on_disk["initial_loss_mode"] == "training", "the default model is batch-normalized (D-025)"
    assert on_disk["lr_reduction_epochs"] == [], "ReduceLROnPlateau cannot fire in 2 epochs"
    assert on_disk["epochs_run"] == 2 and on_disk["stopped_early"] is False


def test_smoke_notes_state_the_guard_floor_the_ece_slice_and_what_final_means(smoke) -> None:
    """Review C6, C8, C9: three readings a reader would otherwise get wrong."""
    notes = " || ".join(json.loads((smoke.run_dir / "results_summary.json").read_text())["notes"])
    assert "guard is a floor" in notes and "not healthy" in notes, "3x-10x is not 'healthy'"
    assert "top-level `ece` is computed on the FULL test set" in notes
    assert "first 1000 test samples" in notes
    assert "equals `test_metrics_best` by construction" in notes
    assert "`final_val_metrics` is the LAST epoch's validation metrics" in notes
    assert "initial loss mode: training" in notes


def test_smoke_run_log_is_in_the_run_dir_and_its_handler_is_gone(smoke) -> None:
    text = (smoke.run_dir / "run.log").read_text()
    assert "Run directory" in text and "Sanity evaluate BEFORE fit (training mode)" in text, text[:400]
    assert "Test results (final weights)" in text
    assert "Analyzer status (read back from disk): success" in text
    leaked = [h for h in logging.getLogger("dl").handlers if isinstance(h, logging.FileHandler)]
    assert leaked == [], f"train_model left a file handler on the dl logger: {leaked}"


def test_smoke_best_epoch_csv_index_is_the_zero_based_csv_row(smoke) -> None:
    """``best_epoch`` is 1-based, the CSV ``epoch`` column is 0-based; the summary
    carries both and says so."""
    on_disk = json.loads((smoke.run_dir / "results_summary.json").read_text())
    with open(smoke.run_dir / "training_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    val_losses = [float(r["val_loss"]) for r in rows]
    argmin = int(np.argmin(val_losses))

    assert on_disk["best_epoch_csv_index"] == argmin
    assert on_disk["best_epoch"] == on_disk["best_epoch_csv_index"] + 1
    assert int(rows[on_disk["best_epoch_csv_index"]]["epoch"]) == argmin, (
        "the CSV `epoch` column must be 0-based for best_epoch_csv_index to index it"
    )
    assert any("0-based" in note for note in on_disk["notes"]), on_disk["notes"]


def test_smoke_no_model_layer_carries_a_kernel_regularizer(smoke) -> None:
    """Weight decay lives in the optimizer only; a regularizer would double it."""
    model = keras.models.load_model(smoke.run_dir / "final_model.keras")
    dense_like = [sub for layer in model.hidden_layers
                  for sub in (layer.main_dense, layer.basis_dense)] + [model.output_layer]
    assert len(dense_like) == 7
    assert all(sub.kernel_regularizer is None for sub in dense_like)
    assert all(layer.kernel_regularizer is None for layer in model.hidden_layers)
    assert model.losses == []


# ---------------------------------------------------------------------
# Summary writer and best-epoch selection (review C5)
# ---------------------------------------------------------------------


def test_write_summary_json_is_strict_json_and_turns_every_non_finite_value_into_null(tmp_path) -> None:
    """``json.dump`` writes ``NaN`` tokens by default and jq / most non-Python readers
    reject them. numpy scalars and arrays must still serialize."""
    summary = {
        "status": "diverged",
        "metric": float("nan"),
        "curve": [1.0, float("inf"), float("-inf")],
        "nested": {"f32": np.float32("nan"), "array": np.array([0.5, np.nan]), "int": np.int64(3)},
        "ok": np.float32(0.5),
        "text": "x",
    }
    written = run_artifacts.write_summary_json(tmp_path, summary)

    text = (tmp_path / "results_summary.json").read_text()
    parsed = json.loads(text, parse_constant=_no_constants)
    assert parsed == written == {
        "status": "diverged", "metric": None, "curve": [1.0, None, None],
        "nested": {"f32": None, "array": [0.5, None], "int": 3}, "ok": 0.5, "text": "x",
    }
    for token in ("NaN", "Infinity"):
        assert token not in text


def test_a_nan_test_metric_reaches_the_summary_file_as_null(tmp_path) -> None:
    run_artifacts.write_summary_json(tmp_path, {"test_metrics_final": {"loss": float("nan"), "accuracy": 0.9}})
    parsed = json.loads((tmp_path / "results_summary.json").read_text(), parse_constant=_no_constants)
    assert parsed["test_metrics_final"] == {"loss": None, "accuracy": 0.9}


# ---------------------------------------------------------------------
# (v) dashboard and visualization functions
# ---------------------------------------------------------------------


def _history(n: int) -> dict:
    e = np.arange(1, n + 1, dtype=np.float64)
    return {
        "loss": (2.0 / e).tolist(), "val_loss": (2.2 / e).tolist(),
        "accuracy": (1.0 - 0.5 / e).tolist(), "val_accuracy": (0.98 - 0.5 / e).tolist(),
        "lr": [3e-4 * 0.5 ** (i // 3) for i in range(n)],
    }


class CapturedFigure:
    """What one figure looked like at the moment it was about to be saved.

    Read INSIDE the spy: ``_save_and_close`` closes the figure, so nothing can be
    inspected afterwards. ``axes`` keeps ``(title, has_data)`` for every axes
    (twin and colorbar axes have an empty title); the dicts are keyed by the
    non-empty axes title.
    """

    def __init__(self, fig) -> None:
        self.axes = [(ax.get_title(), ax.has_data()) for ax in fig.axes]
        titled = [ax for ax in fig.axes if ax.get_title()]
        self.ylim = {ax.get_title(): tuple(ax.get_ylim()) for ax in titled}
        self.axes_texts = {ax.get_title(): [t.get_text() for t in ax.texts] for ax in titled}
        self.fig_texts = [t.get_text() for t in fig.texts]
        # One entry per axes in ``fig.axes`` order (image axes AND colorbar axes,
        # which have no title): does any x or y grid line show?
        self.grid_visible = [
            any(gl.get_visible() for gl in ax.get_xgridlines() + ax.get_ygridlines())
            for ax in fig.axes
        ]
        # Same, split by direction, for axes that legitimately keep one direction.
        self.xgrid_visible = [any(gl.get_visible() for gl in ax.get_xgridlines()) for ax in fig.axes]
        self.ygrid_visible = [any(gl.get_visible() for gl in ax.get_ygridlines()) for ax in fig.axes]
        # (hatch, alpha, height) of every bar patch, keyed by the axes title.
        self.bars = {
            ax.get_title(): [(p.get_hatch(), p.get_alpha(), p.get_height()) for p in ax.patches]
            for ax in titled
        }


@pytest.fixture
def captured_figures(monkeypatch):
    """Record every figure the visualization module is about to save."""
    figures = []
    real = viz._save_and_close

    def spy(fig, out_path):
        figures.append(CapturedFigure(fig))
        return real(fig, out_path)

    monkeypatch.setattr(viz, "_save_and_close", spy)
    return figures


def _loss_axis_labels(tmp_path, monkeypatch, low, high, n_epochs=8):
    """Render a dashboard and return the drawn y tick labels of the Loss and Smoothed loss
    panels: ``{title: (visible major labels, visible minor labels)}``.

    Labels exist only after a draw, and ``_save_and_close`` closes the figure, so the spy
    draws the canvas and reads them first.
    """
    read = {}
    real = viz._save_and_close

    def spy(fig, out_path):
        fig.canvas.draw()
        for ax in fig.axes:
            if ax.get_title() in ("Loss", "Smoothed loss"):
                lo, hi = ax.get_ylim()  # matplotlib also labels ticks outside the view
                read[ax.get_title()] = tuple(
                    [t.get_text() for t, loc in zip(labels, locs) if lo <= loc <= hi]
                    for labels, locs in (
                        (ax.yaxis.get_majorticklabels(), ax.yaxis.get_majorticklocs()),
                        (ax.yaxis.get_minorticklabels(), ax.yaxis.get_minorticklocs()),
                    )
                )
        return real(fig, out_path)

    monkeypatch.setattr(viz, "_save_and_close", spy)
    loss = np.geomspace(high, low, n_epochs)
    viz.render_training_dashboard(
        {"loss": loss.tolist(), "val_loss": (loss * 1.05).tolist()}, tmp_path / "d.png")
    return read


def test_dashboard_loss_axis_reads_plain_numbers_for_a_narrow_range(
        tmp_path, monkeypatch) -> None:
    """A loss of 1.1 to 2.5 used to be labelled ``1.8x10^0``, ``2x10^0`` (audit F9)."""
    read = _loss_axis_labels(tmp_path, monkeypatch, 1.1, 2.5)
    assert set(read) == {"Loss", "Smoothed loss"}
    for title, (major, minor) in read.items():
        shown = [t for t in major + minor if t]
        assert shown, (title, "no tick label at all: the guard would pass vacuously")
        assert not any("10^" in t or "x" in t or "\u00d7" in t for t in shown), (title, shown)
        assert all(0.0 < float(t) < 3.0 for t in shown), (title, shown)  # plain, parseable
    assert any(t for t in read["Loss"][1]), "a narrow range keeps its minor labels"


def test_dashboard_loss_axis_keeps_only_decades_for_a_wide_range(
        tmp_path, monkeypatch) -> None:
    read = _loss_axis_labels(tmp_path, monkeypatch, 0.001, 10.0)
    major, minor = read["Loss"]
    assert {"0.001", "0.01", "0.1", "1", "10"} <= {t for t in major if t}, major
    assert [t for t in minor if t] == [], minor


@pytest.mark.parametrize("n_epochs", [1, 2, 3, 5, 6, 7, 12])
def test_dashboard_smoothed_loss_panel_iff_more_than_five_epochs(
        tmp_path, captured_figures, n_epochs) -> None:
    out = tmp_path / "dash.png"
    titles = viz.render_training_dashboard(
        _history(n_epochs), out, "t", epoch_times=[1.0] * n_epochs,
        baseline={"loss": 5.0, "accuracy": 0.1},
    )
    assert out.is_file() and out.stat().st_size > 0
    assert ("Smoothed loss" in titles) == (n_epochs > 5), titles
    assert {"Loss", "Accuracy", "Learning rate", "Generalization gap", "Per-epoch time"} <= set(titles)

    (figure,) = captured_figures
    axes = figure.axes
    drawn = [title for title, _ in axes if title]
    assert drawn == titles, "the returned titles must be the panels actually drawn"
    blank = [title or "<twin axis>" for title, has_data in axes if not has_data]
    assert blank == [], f"blank panels: {blank}"


def test_dashboard_draws_only_panels_that_have_data(tmp_path, captured_figures) -> None:
    """No lr and no epoch times -> no lr panel and no time panel, and no blank cells."""
    history = {k: v for k, v in _history(3).items() if k != "lr"}
    titles = viz.render_training_dashboard(history, tmp_path / "d.png")
    assert titles == ["Loss", "Accuracy", "Generalization gap"]
    assert all(has_data for _, has_data in captured_figures[0].axes)


@pytest.mark.parametrize("n_epochs", [1, 3, 7])
def test_dashboard_carries_the_train_vs_val_caption(
        tmp_path, captured_figures, n_epochs) -> None:
    """Train metrics are a running mean over the epoch, val is measured at its end;
    the caption says so on the figure itself, whatever the panel layout."""
    viz.render_training_dashboard(_history(n_epochs), tmp_path / "cap.png", "t")
    (figure,) = captured_figures
    assert viz.TRAIN_METRIC_CAPTION in figure.fig_texts, figure.fig_texts
    assert "running mean" in viz.TRAIN_METRIC_CAPTION
    assert "end of epoch" in viz.TRAIN_METRIC_CAPTION


def _near_one_history() -> dict:
    """Curves that hug 1.0 (max 1.1), like a real run after a huge epoch-0 loss."""
    history = _history(3)
    history["loss"] = [1.0, 0.95, 0.9]
    history["val_loss"] = [1.1, 1.0, 0.95]
    return history


#: The SHIPPED headline shape (iteration-2 real run): epoch-1 running-mean loss
#: 0.4972 and an epoch-0 baseline of ln(10) = 2.312, a ratio of 4.65. A clip
#: factor of 4.0 mislabelled this correct baseline as clipped (review C3).
SHIPPED_CURVE_MAX = 0.4972
SHIPPED_BASELINE = 2.312


def _shipped_history() -> dict:
    history = _history(3)
    history["loss"] = [SHIPPED_CURVE_MAX, 0.20, 0.12]
    history["val_loss"] = [0.46, 0.19, 0.11]
    return history


def _loss_panel(tmp_path, captured_figures, baseline, history=None) -> CapturedFigure:
    viz.render_training_dashboard(
        _near_one_history() if history is None else history,
        tmp_path / f"d{len(captured_figures)}.png", "t",
        epoch_times=[1.0] * 3, baseline=baseline,
    )
    return captured_figures[-1]


def test_an_extreme_epoch_0_baseline_is_clipped_and_annotated_with_its_true_value(
        tmp_path, captured_figures) -> None:
    """Iteration 1: an init loss of 218.6 over curves near 1.0 flattened the log axis."""
    figure = _loss_panel(tmp_path, captured_figures, {"loss": 218.6, "accuracy": 0.1})
    low, high = figure.ylim["Loss"]
    curve_max = 1.1

    assert high <= viz.BASELINE_CLIP_FACTOR * curve_max * 1.5, (
        f"Loss axis reaches {high:.1f}: the 218.6 baseline flattened the curves"
    )
    assert low > 0.0
    annotations = figure.axes_texts["Loss"]
    assert any("218.6" in t and "clipped" in t for t in annotations), annotations
    assert figure.axes_texts["Accuracy"] == [], "an accuracy of 0.1 is never clipped"


def test_the_shipped_headline_baseline_is_drawn_where_it_is_and_not_annotated(
        tmp_path, captured_figures, monkeypatch) -> None:
    """The healthy run (baseline ln(10) over a curve max of about 0.5) is untouched:
    same limits as with the clip disabled, the true baseline inside the axis, and no
    "(clipped)" annotation. A factor below the shipped ratio 4.65 (the old 4.0) fails
    here, so the arm is not vacuous."""
    baseline = {"loss": SHIPPED_BASELINE, "accuracy": 0.1}
    history = _shipped_history()
    assert SHIPPED_BASELINE / SHIPPED_CURVE_MAX > 4.0, "the shape must stay above the old factor"

    clipped_run = _loss_panel(tmp_path, captured_figures, baseline, history)
    monkeypatch.setattr(viz, "BASELINE_CLIP_FACTOR", 1e12)
    unclipped_run = _loss_panel(tmp_path, captured_figures, baseline, history)

    low, high = clipped_run.ylim["Loss"]
    assert low < SHIPPED_BASELINE < high, "the true baseline must be inside the axis"
    for panel in ("Loss", "Accuracy"):
        assert clipped_run.ylim[panel] == unclipped_run.ylim[panel], panel
        assert clipped_run.axes_texts[panel] == unclipped_run.axes_texts[panel] == [], panel


def test_a_history_with_no_finite_value_still_draws_the_baseline(
        tmp_path, captured_figures) -> None:
    """A diverged run: every plotted curve value is NaN, so there is no ceiling to clip
    at. The baseline branch used to take ``max()`` of an empty array and raise."""
    history = {"loss": [float("nan")] * 3, "val_loss": [float("nan")] * 3}
    out = tmp_path / "nan.png"
    titles = viz.render_training_dashboard(history, out, "t", baseline={"loss": 2.3})

    assert out.stat().st_size > 0
    assert titles == ["Loss", "Generalization gap"]
    assert captured_figures[-1].axes_texts["Loss"] == [], "an unclipped baseline is not annotated"


def _time_panel(tmp_path, captured_figures, times) -> CapturedFigure:
    viz.render_training_dashboard(
        _history(len(times)), tmp_path / f"t{len(captured_figures)}.png", "t",
        epoch_times=times,
    )
    return captured_figures[-1]


def test_a_warmup_epoch_time_is_clipped_and_annotated_with_its_true_value(
        tmp_path, captured_figures) -> None:
    """Epoch 1 pays the XLA warmup (about 22.8 s against 1.8 s). Unclipped, that one
    bar sets the whole y-scale and the real per-epoch variation collapses."""
    figure = _time_panel(tmp_path, captured_figures, [22.8] + [1.8] * 9)
    low, high = figure.ylim["Per-epoch time"]

    assert high <= viz.TIME_CLIP_FACTOR * 1.8 * 1.3, f"time axis reaches {high:.1f} s"
    assert high > viz.TIME_CLIP_FACTOR * 1.8, "the clipped bar must fit under the top limit"
    annotations = figure.axes_texts["Per-epoch time"]
    assert len(annotations) == 1 and "22.8" in annotations[0] and "clipped" in annotations[0], annotations


def test_ordinary_epoch_times_are_drawn_exactly_as_before(
        tmp_path, captured_figures, monkeypatch) -> None:
    """Times within ``TIME_CLIP_FACTOR`` x the median: same limits as with the clip
    disabled and no annotation. A factor of 0.5 clips every bar and fails here."""
    times = [2.4, 1.8, 2.0, 1.9, 2.1, 1.8]
    clipped_run = _time_panel(tmp_path, captured_figures, times)
    monkeypatch.setattr(viz, "TIME_CLIP_FACTOR", 1e12)
    unclipped_run = _time_panel(tmp_path, captured_figures, times)

    assert clipped_run.ylim["Per-epoch time"] == unclipped_run.ylim["Per-epoch time"]
    assert clipped_run.axes_texts["Per-epoch time"] == unclipped_run.axes_texts["Per-epoch time"] == []


def test_a_single_epoch_time_has_no_reference_and_is_not_clipped(
        tmp_path, captured_figures) -> None:
    """The reference is the median of epochs >= 2: with one epoch there is none."""
    figure = _time_panel(tmp_path, captured_figures, [22.8])
    assert figure.ylim["Per-epoch time"][1] >= 22.8
    assert figure.axes_texts["Per-epoch time"] == []


def _figure_from_counts(counts: np.ndarray):
    """Labels whose confusion matrix is exactly ``counts`` (rows true, columns predicted)."""
    y_true = np.repeat(np.arange(len(counts)), counts.sum(axis=1))
    y_pred = np.concatenate([np.repeat(np.arange(len(counts)), row) for row in counts])
    return y_true, y_pred


def test_the_normalized_confusion_panel_hides_cells_at_or_below_half_a_percent(
        tmp_path, captured_figures) -> None:
    """``f"{0.003:.0%}"`` is ``"0%"``: a normalized panel that annotates every cell prints
    a wall of them. Cells > 0.5% keep their text; 0.5% itself (5 of 1000) also formats
    as "0%" (round half to even), so the floor is strict."""
    counts = np.array([
        [990, 5, 3, 2],     # 99%, then 0.5% (boundary), 0.3%, 0.2%: only 99% is annotated
        [6, 940, 54, 0],    # 0.6% -> "1%", 94%, 5.4% -> "5%"
        [0, 0, 100, 0],     # 100%
        [0, 0, 2, 98],      # 2%, 98%
    ])
    y_true, y_pred = _figure_from_counts(counts)
    cm = viz.plot_confusion_matrix(y_true, y_pred, list("abcd"), tmp_path / "cm.png")
    np.testing.assert_array_equal(cm, counts)

    (figure,) = captured_figures
    normalized = figure.axes_texts["Row-normalized (recall per true class)"]
    assert "0%" not in normalized, normalized
    assert sorted(normalized) == sorted(["99%", "1%", "94%", "5%", "100%", "2%", "98%"]), normalized
    assert len(figure.axes_texts["Counts"]) == 16, "the counts panel annotates every cell"
    assert "0" in figure.axes_texts["Counts"]


def test_confusion_matrix_grid_lines_are_off_whatever_the_ambient_style_says(
        tmp_path, captured_figures) -> None:
    """``dl_techniques.visualization.core`` sets a whitegrid style at import, making
    ``axes.grid`` True process-wide, and the grid then cut through the cell text (review
    C7). The function must switch the lines off itself on both image axes; the two
    colorbar axes (four axes in all) are asserted too, so a matplotlib that grids them
    would fail here."""
    counts = np.array([[9, 1], [2, 8]])
    y_true, y_pred = _figure_from_counts(counts)
    with matplotlib.rc_context({"axes.grid": True}):
        # Control: the ambient setting really does grid a plain axes, so a passing
        # assertion below is not a consequence of the context doing nothing.
        control_fig, control_ax = plt.subplots()
        control_ax.plot([0, 1])
        control_grid = any(gl.get_visible() for gl in control_ax.get_xgridlines())
        plt.close(control_fig)
        viz.plot_confusion_matrix(y_true, y_pred, ["a", "b"], tmp_path / "cm_grid.png")

    assert control_grid is True
    (figure,) = captured_figures
    assert len(figure.grid_visible) == 4, "two image axes and two colorbar axes"
    assert figure.grid_visible == [False] * 4, figure.grid_visible


def test_per_class_chart_has_no_vertical_grid_whatever_the_ambient_style_says(
        tmp_path, captured_figures) -> None:
    """The per-class bar chart showed vertical grid lines through the bars: it set
    ``ax.grid(axis="y")`` only, so the x grid followed the ambient ``axes.grid=True``
    that ``dl_techniques.visualization.core`` sets at import. The horizontal grid is
    deliberate and stays (asserted, so switching the whole grid off cannot pass)."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1])
    with matplotlib.rc_context({"axes.grid": True}):
        control_fig, control_ax = plt.subplots()
        control_ax.plot([0, 1])
        control_x = any(gl.get_visible() for gl in control_ax.get_xgridlines())
        plt.close(control_fig)
        viz.plot_per_class_metrics(y_true, y_pred, ["a", "b", "c"], tmp_path / "pc.png")
    assert control_x is True, "the ambient setting really grids a plain axes"
    (figure,) = captured_figures
    assert figure.xgrid_visible == [False], figure.xgrid_visible
    assert figure.ygrid_visible == [True], figure.ygrid_visible


def _reliability_groups():
    """``(n, top-class confidence, n_correct)`` per bin; 20 is dense, 19 is sparse."""
    return [(150, 0.97, 147), (20, 0.85, 17), (19, 0.75, 14), (3, 0.65, 0), (2, 0.55, 1)]


def _reliability_inputs():
    y, probs = [], []
    for n, conf, n_correct in _reliability_groups():
        y += [0] * n_correct + [1] * (n - n_correct)
        probs += [[conf, 1.0 - conf]] * n
    return np.array(y), np.array(probs)


def test_reliability_bins_under_20_samples_are_hatched_lighter_and_annotated(
        tmp_path, captured_figures) -> None:
    y, probs = _reliability_inputs()
    ece = viz.plot_calibration(y, probs, tmp_path / "cal.png")
    (figure,) = captured_figures
    bars = figure.bars["Reliability diagram"]
    assert len(bars) == 5, "one bar per populated bin"
    dense = [b for b in bars if not b[0]]
    sparse = [b for b in bars if b[0]]
    # Exactly the 19-, 3- and 2-sample bins are marked; the 20-sample bin is not
    # (the threshold is strict on n < 20).
    assert len(dense) == 2 and len(sparse) == 3, bars
    assert all(alpha == 0.8 for _, alpha, _ in dense)
    assert all(alpha < 0.5 for _, alpha, _ in sparse), "lighter than the dense bars"
    assert sorted(round(h, 4) for _, _, h in sparse) == [0.0, 0.5, round(14 / 19, 4)]
    assert sorted(figure.axes_texts["Reliability diagram"]) == sorted(
        ["ECE = %.4f" % ece, "n=19", "n=3", "n=2"]), figure.axes_texts["Reliability diagram"]
    # The ECE is the untouched population-weighted one (hand-derived from the groups).
    total = sum(n for n, _, _ in _reliability_groups())
    expected = sum(n / total * abs(k / n - conf) for n, conf, k in _reliability_groups())
    assert ece == pytest.approx(expected, abs=1e-9)


def test_reliability_diagram_with_only_dense_bins_has_no_hatch_and_no_n_labels(
        tmp_path, captured_figures) -> None:
    rng = np.random.default_rng(0)
    y, _, probs = _predictions(rng, n=400)
    viz.plot_calibration(y, probs, tmp_path / "cal2.png")
    (figure,) = captured_figures
    assert all(not hatch for hatch, _, _ in figure.bars["Reliability diagram"])
    assert not [t for t in figure.axes_texts["Reliability diagram"] if t.startswith("n=")]


def test_dashboard_with_no_epochs_writes_nothing(tmp_path) -> None:
    assert viz.render_training_dashboard({}, tmp_path / "none.png") == []
    assert not (tmp_path / "none.png").exists()


def test_dashboard_callback_accumulates_lr_and_writes_the_png(tmp_path) -> None:
    cb = viz.TrainingDashboardCallback(tmp_path / "cb.png", title="cb")
    for epoch in range(3):
        cb.on_epoch_begin(epoch)
        cb.on_epoch_end(epoch, {"loss": 1.0 / (epoch + 1), "val_loss": 1.1 / (epoch + 1),
                                "accuracy": 0.5 + 0.1 * epoch, "val_accuracy": 0.4 + 0.1 * epoch,
                                "lr": 3e-4})
    assert (tmp_path / "cb.png").stat().st_size > 0
    assert cb.history["lr"] == [3e-4] * 3 and len(cb.epoch_times) == 3
    assert len(cb.history["loss"]) == 3


def test_a_raising_renderer_does_not_abort_on_epoch_end(monkeypatch, tmp_path) -> None:
    def boom(*a, **k):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(viz, "render_training_dashboard", boom)
    cb = viz.TrainingDashboardCallback(tmp_path / "x.png")
    cb.on_epoch_begin(0)
    cb.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.0})  # must not raise
    assert cb.history["loss"] == [1.0]


def _drive_dashboard(monkeypatch, tmp_path, planned, run, with_params=True):
    """Run a stubbed callback for ``run`` epochs; return the epoch counts it drew.

    ``render_training_dashboard`` is replaced by a spy that records how many epochs
    the history held at each call, so a draw at epoch ``n`` shows up as ``n``.
    """
    drawn: list[int] = []

    def spy(history, out_path, title="", epoch_times=None, baseline=None, **kwargs):
        assert len(epoch_times) == len(history["loss"]), "every epoch is accumulated"
        drawn.append(len(history["loss"]))
        return ["Loss"]

    monkeypatch.setattr(viz, "render_training_dashboard", spy)
    cb = viz.TrainingDashboardCallback(tmp_path / "c.png")
    if with_params:
        cb.set_params({"epochs": planned, "steps": 5, "verbose": 0})
    cb.on_train_begin()
    for epoch in range(run):
        cb.on_epoch_begin(epoch)
        cb.on_epoch_end(epoch, {"loss": 1.0 / (epoch + 1), "val_loss": 1.0, "lr": 3e-4})
    cb.on_train_end()
    return drawn, cb


@pytest.mark.parametrize(
    "planned, run, expected",
    [
        (3, 3, [1, 2, 3]),  # 3 // 20 = 0 -> every epoch
        (10, 10, list(range(1, 11))),  # 10 // 20 = 0 -> every epoch
        (39, 39, list(range(1, 40))),  # 39 // 20 = 1 -> every epoch
        (40, 40, [1] + list(range(2, 41, 2))),  # interval 2
        # Early stop at 37 of 100: interval 5, so 35 is the last cadence draw and the
        # final state (37) is drawn by on_train_end.
        (100, 37, [1, 5, 10, 15, 20, 25, 30, 35, 37]),
        # Full 100 epochs: 21 draws, epoch 100 is on the cadence so no extra draw.
        (100, 100, [1] + list(range(5, 101, 5))),
    ],
)
def test_dashboard_draws_on_a_cadence_and_always_shows_the_final_state(
        monkeypatch, tmp_path, planned, run, expected) -> None:
    drawn, _ = _drive_dashboard(monkeypatch, tmp_path, planned, run)
    assert drawn == expected, drawn
    assert drawn[0] == 1 and drawn[-1] == run, "epoch 1 first, the final state last"


def test_dashboard_without_planned_epochs_draws_every_epoch(monkeypatch, tmp_path) -> None:
    drawn, _ = _drive_dashboard(monkeypatch, tmp_path, None, 25, with_params=False)
    assert drawn == list(range(1, 26))
    drawn, _ = _drive_dashboard(monkeypatch, tmp_path, None, 7)  # params present, epochs None
    assert drawn == list(range(1, 8))


def test_a_hundred_epoch_run_draws_21_times_not_100(monkeypatch, tmp_path) -> None:
    """The saving: 79 renders of about 1.1-1.7 s each (measured on this machine)."""
    drawn, cb = _drive_dashboard(monkeypatch, tmp_path, 100, 100)
    assert len(drawn) == 21 and len(cb.epoch_times) == 100


def test_the_final_draw_is_skipped_only_when_the_last_epoch_is_already_on_disk(
        monkeypatch, tmp_path) -> None:
    """``on_train_end`` must draw when the last epoch is off-cadence, and must not
    draw a second identical image when it is on it. A run with no epochs draws
    nothing."""
    drawn, _ = _drive_dashboard(monkeypatch, tmp_path, 100, 37)
    assert drawn.count(37) == 1 and drawn[-1] == 37
    drawn, _ = _drive_dashboard(monkeypatch, tmp_path, 100, 100)
    assert drawn.count(100) == 1
    drawn, _ = _drive_dashboard(monkeypatch, tmp_path, 100, 0)
    assert drawn == []


def test_a_failed_draw_is_retried_in_on_train_end_and_never_raises(
        monkeypatch, tmp_path) -> None:
    calls = []

    def flaky(history, out_path, title="", epoch_times=None, baseline=None, **kwargs):
        calls.append(len(history["loss"]))
        if len(calls) == 1:
            raise RuntimeError("first render fails")
        return ["Loss"]

    monkeypatch.setattr(viz, "render_training_dashboard", flaky)
    cb = viz.TrainingDashboardCallback(tmp_path / "f.png")
    cb.set_params({"epochs": 100})
    cb.on_epoch_begin(0)
    cb.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.0})  # draws, raises inside, is caught
    cb.on_train_end()  # the PNG never showed epoch 1, so it is drawn again
    assert calls == [1, 1]


def test_real_fit_hands_the_callback_its_planned_epoch_count(monkeypatch, tmp_path) -> None:
    """``keras.Model.fit`` must put ``epochs`` in ``callback.params`` under the key the
    callback reads, or the cadence silently degrades to every-epoch (and the saving
    disappears without any test on the stub path noticing)."""
    drawn: list[int] = []

    def spy(history, out_path, title="", epoch_times=None, baseline=None, **kwargs):
        drawn.append(len(history["loss"]))
        return ["Loss"]

    monkeypatch.setattr(viz, "render_training_dashboard", spy)
    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(2, activation="softmax")])
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy())
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(16, 4)).astype("float32"), rng.integers(0, 2, 16)
    cb = viz.TrainingDashboardCallback(tmp_path / "r.png")
    model.fit(x, y, validation_data=(x, y), epochs=40, batch_size=16, verbose=0, callbacks=[cb])
    assert cb.params["epochs"] == 40
    assert drawn == [1] + list(range(2, 41, 2)), drawn


def test_dashboard_callback_evaluates_the_epoch_0_baseline(tmp_path) -> None:
    model = keras.Sequential([keras.layers.Input((6,)), keras.layers.Dense(3, activation="softmax")])
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(32, 6)).astype("float32"), rng.integers(0, 3, 32)
    cb = viz.TrainingDashboardCallback(tmp_path / "b.png", baseline_data=(x, y))
    cb.set_model(model)
    cb.on_train_begin()
    assert set(cb.baseline) >= {"loss", "accuracy"}
    assert all(np.isfinite(v) for v in cb.baseline.values())


def _predictions(rng, n=200, c=5, accuracy=0.8):
    y = rng.integers(0, c, n)
    pred = np.where(rng.random(n) < accuracy, y, (y + 1) % c)
    probs = np.full((n, c), 0.02)
    probs[np.arange(n), pred] = 1.0 - 0.02 * (c - 1)
    return y, pred, probs


def test_confusion_matrix_counts_are_right(tmp_path) -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 2])
    cm = viz.plot_confusion_matrix(y_true, y_pred, ["a", "b", "c"], tmp_path / "cm.png")
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 2, 0], [1, 0, 2]])
    assert (tmp_path / "cm.png").stat().st_size > 0


def test_per_class_metrics_report_matches_sklearn_hand_values(tmp_path) -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 2])
    report = viz.plot_per_class_metrics(y_true, y_pred, ["a", "b", "c"], tmp_path / "pc.png")
    assert report["accuracy"] == pytest.approx(5 / 7)
    assert report["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert report["per_class"]["c"]["recall"] == pytest.approx(2 / 3)
    assert report["per_class"]["a"]["support"] == 2
    json.dumps(report)  # JSON-serializable
    assert (tmp_path / "pc.png").stat().st_size > 0


def test_a_class_never_predicted_scores_zero_instead_of_crashing(tmp_path) -> None:
    report = viz.plot_per_class_metrics(
        np.array([0, 1, 2]), np.array([0, 1, 1]), ["a", "b", "c"], tmp_path / "z.png")
    assert report["per_class"]["c"]["precision"] == 0.0


def test_calibration_ece_is_zero_when_confidence_equals_accuracy_and_one_when_all_wrong(tmp_path) -> None:
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]  # confidence 1.0, all correct
    assert viz.plot_calibration(y, perfect, tmp_path / "ok.png") == pytest.approx(0.0, abs=1e-9)
    wrong = np.eye(3)[(y + 1) % 3]  # confidence 1.0, all wrong
    assert viz.plot_calibration(y, wrong, tmp_path / "bad.png") == pytest.approx(1.0)
    assert (tmp_path / "ok.png").stat().st_size > 0


def test_confident_errors_grid_renders_and_is_skipped_without_errors(tmp_path) -> None:
    rng = np.random.default_rng(0)
    y, pred, probs = _predictions(rng, n=60, c=10, accuracy=0.7)
    x = rng.normal(size=(60, 784)).astype("float32")
    mean, std = np.array([0.1307], "float32"), np.array([0.3081], "float32")
    out = viz.plot_confident_errors(x, y, probs, tmp_path / "e.png", (28, 28, 1), mean, std)
    assert out is not None and (tmp_path / "e.png").stat().st_size > 0

    rgb = viz.plot_confident_errors(
        rng.normal(size=(60, 3072)).astype("float32"), y, probs, tmp_path / "e3.png",
        (32, 32, 3), np.zeros(3, "float32"), np.ones(3, "float32"))
    assert rgb is not None

    perfect = np.eye(10)[y]
    assert viz.plot_confident_errors(x, y, perfect, tmp_path / "none.png",
                                     (28, 28, 1), mean, std) is None
    assert not (tmp_path / "none.png").exists()


# ---------------------------------------------------------------------
# Shared helpers promoted to train/common/ (run_artifacts, classification_viz)
# ---------------------------------------------------------------------


def test_the_power_mlp_visualization_module_is_the_shared_module() -> None:
    """A copy of the names would let ``monkeypatch.setattr(viz, ...)`` patch a name the
    dashboard code never reads (D-010). Identity, not equality."""
    assert viz is shared_viz
    assert viz.TrainingDashboardCallback is shared_viz.TrainingDashboardCallback


def test_write_summary_json_writes_nothing_when_a_non_finite_value_survives_sanitizing(
        monkeypatch, tmp_path) -> None:
    """``allow_nan=False`` is the second line of defence behind the sanitizing pass: with
    the sanitizer defeated, a NaN must raise and must not leave a half-written file."""
    monkeypatch.setattr(run_artifacts, "np", types.SimpleNamespace(isfinite=lambda value: True))
    with pytest.raises(ValueError):
        run_artifacts.write_summary_json(tmp_path, {"loss": float("nan")})
    assert not (tmp_path / "results_summary.json").exists()


def test_attach_run_log_tees_the_logger_in_write_mode_and_detaches_on_exit(tmp_path) -> None:
    log = tmp_path / "run.log"
    log.write_text("stale line from an earlier run\n")

    with run_artifacts.attach_run_log(tmp_path) as handler:
        assert handler in tpm.logger.handlers
        assert handler.mode == "w"
        tpm.logger.info("inside the block")
    assert handler not in tpm.logger.handlers

    text = log.read_text()
    assert "inside the block" in text and "stale line" not in text
    assert "INFO" in text  # the repo LOGGER_FORMAT, not a bare message


def test_attach_run_log_detaches_when_the_block_raises(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with run_artifacts.attach_run_log(tmp_path) as handler:
            raise RuntimeError("boom")
    assert handler not in tpm.logger.handlers
    assert handler.stream is None  # closed


def test_confident_errors_accepts_nhwc_images_and_matches_the_flat_rendering(tmp_path) -> None:
    rng = np.random.default_rng(0)
    y, pred, probs = _predictions(rng, n=60, c=10, accuracy=0.7)
    nhwc = rng.random((60, 8, 8, 3)).astype("float32")
    flat = nhwc.reshape(60, -1)
    mean, std = np.zeros(3, "float32"), np.ones(3, "float32")

    a = viz.plot_confident_errors(nhwc, y, probs, tmp_path / "nhwc.png")
    b = viz.plot_confident_errors(flat, y, probs, tmp_path / "flat.png", (8, 8, 3), mean, std)

    # An NHWC input is used as it is: a stale ``image_shape`` must not be consulted.
    c = viz.plot_confident_errors(nhwc, y, probs, tmp_path / "nhwc_stale_shape.png", (1, 1, 1))

    assert a is not None and b is not None and c is not None
    assert (tmp_path / "nhwc.png").read_bytes() == (tmp_path / "flat.png").read_bytes()
    assert (tmp_path / "nhwc_stale_shape.png").read_bytes() == (tmp_path / "flat.png").read_bytes()


def test_confident_errors_flat_input_without_image_shape_is_an_error(tmp_path) -> None:
    rng = np.random.default_rng(0)
    y, pred, probs = _predictions(rng, n=20, c=10, accuracy=0.5)
    with pytest.raises(ValueError, match="image_shape"):
        viz.plot_confident_errors(rng.random((20, 192)).astype("float32"), y, probs, tmp_path / "x.png")
