"""Initial-scale and input-scaling guards for ``src/train/power_mlp/train_power_mlp.py``.

The defect class
----------------
The iteration-1 trainer hardcoded ``glorot_normal`` on standardized input. ReLU-k
composes to degree ``k**depth``, so on real MNIST the untrained ``default``
preset started at loss 218.6 against ``ln(10) = 2.30`` (ratio 94.9) and the
3-epoch test accuracy fell below the floor. Nothing in the trainer said so, and
every unit test used tiny synthetic data on which the blow-up is invisible.
This file pins the repair:

- ``_check_initial_loss`` WARNS (strictly above ``INITIAL_LOSS_WARN_FACTOR`` x
  ``ln(C)``) and returns ``(loss, ratio, warned, mode)``; a batch-normalized model
  is measured in TRAINING mode with its moving statistics restored (review C4:
  the inference-mode number was 4.2e11 where the first training step sees 2.7);
- the CHOSEN defaults (``TrainingConfig()``) start near ``ln(C)`` on real-shaped
  data, and would not if they were set back to ``glorot_normal`` + ``standardize``;
- ``--input-scaling`` really changes what ``prepare_data`` returns;
- ``train_model`` forwards ``kernel_initializer`` / ``input_scaling`` /
  ``epoch_analysis`` instead of hardcoding them.

Data regime of the guards (MEASURED, CPU, first 1024 training samples, default
preset, ``k=2``, seeds 0-4, loss / ln(10)):

======================  ===============================  =========================
configuration           real cached MNIST                synthetic MNIST-like
======================  ===============================  =========================
glorot + standardize    94.9 / 138.3 / 118.5 / 122.1 /   29.4 / 30.6 / 27.7 /
                        114.9                            33.5 / 32.2
lecun  + unit           1.00 - 1.01                      1.00
glorot + unit           1.04 - 1.08                      1.01 - 1.02
======================  ===============================  =========================

Both regimes clear the ``> 10`` firing threshold with margin (the synthetic one
by 2.8x, the real one by 9.5x), so the guard is not vacuous on either. The real
regime runs when ``~/.keras/datasets/mnist.npz`` exists (the suite never
downloads); the synthetic one always runs. The synthetic batch is MNIST-shaped
(784 features, mean 0.12, ``E[x^2]`` 0.09), NOT i.i.d. ``N(0, 1)``: it goes
through the real ``prepare_data`` so both ``input_scaling`` values apply to it.
"""

from __future__ import annotations

import functools
import json
import os
import types
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import train.power_mlp.train_power_mlp as tpm  # noqa: E402

REAL_MNIST_CACHED = (Path.home() / ".keras" / "datasets" / "mnist.npz").exists()

REGIMES = [
    pytest.param(
        "real",
        marks=pytest.mark.skipif(
            not REAL_MNIST_CACHED, reason="real MNIST is not cached; the suite never downloads"
        ),
    ),
    "synthetic",
]

#: The iteration-1 configuration whose pre-fit loss was 218.6 (ratio 94.9).
ITERATION_1_PAIR = ("glorot_normal", "standardize")
#: The pair the pre-registered rule (D-014) chose from the measured 3-epoch grid.
CHOSEN_PAIR = ("lecun_normal", "unit")

GUARD_SAMPLES = 1024


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def _synthetic_mnist_like(n: int, seed: int):
    """``(x, y)`` shaped like ``load_dataset('mnist')``: 28x28x3 in [0, 1].

    Sparse bright pixels (19% lit, uniform in [0.3, 1]) give mean 0.12 and
    ``E[x^2]`` 0.09, close to real MNIST (0.13 / 0.11); the 3 channels are
    identical like the real loader.
    """
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 10).astype(np.uint8)
    rng.shuffle(y)
    lit = rng.random((n, 28, 28, 1), dtype=np.float32) < 0.19
    x = (lit * rng.uniform(0.3, 1.0, (n, 28, 28, 1))).astype(np.float32)
    return np.repeat(x, 3, axis=-1), y


def _synthetic_load_dataset(n_train: int = 2000, n_test: int = 400):
    def load(name, *args, **kwargs):
        assert name == "mnist", name
        return (
            _synthetic_mnist_like(n_train, 7),
            _synthetic_mnist_like(n_test, 8),
            (28, 28, 3),
            10,
        )

    return load


@functools.lru_cache(maxsize=None)
def _guard_slice(regime: str, input_scaling: str) -> Tuple[np.ndarray, np.ndarray]:
    """The first ``GUARD_SAMPLES`` training rows of ``prepare_data`` for a regime."""
    with pytest.MonkeyPatch.context() as mp:
        if regime == "synthetic":
            mp.setattr(tpm, "load_dataset", _synthetic_load_dataset())
        (x, y), _, _, _ = tpm.prepare_data("mnist", 0.1, 0, input_scaling)
    return x[:GUARD_SAMPLES], y[:GUARD_SAMPLES]


def _build_model(
        kernel_initializer: str, k: int = 2, batch_normalization: bool = False
) -> keras.Model:
    """The ``default`` MNIST preset built and compiled the way ``train_model`` does."""
    model = tpm.PowerMLP(
        hidden_units=tpm.effective_hidden_units("mnist", "default", 784, 10),
        k=k,
        dropout_rate=0.1,
        batch_normalization=batch_normalization,
        output_activation="softmax",
        kernel_initializer=kernel_initializer,
        bias_initializer="zeros",
    )
    model.build((None, 784))
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


@pytest.fixture
def recorded_warnings(monkeypatch):
    """Replace ``tpm.logger.warning`` with a recorder (the project logger may not
    propagate to ``caplog``, so a handler-based capture could see nothing)."""
    seen = []
    monkeypatch.setattr(tpm.logger, "warning", lambda msg, *a, **k: seen.append(str(msg)))
    return seen


def _initial_loss_check(
        pair: Tuple[str, str], regime: str, k: int = 2, batch_normalization: bool = False
):
    keras.utils.set_random_seed(0)
    kernel_initializer, input_scaling = pair
    x, y = _guard_slice(regime, input_scaling)
    model = _build_model(kernel_initializer, k, batch_normalization)
    return tpm._check_initial_loss(model, x, y)


# ---------------------------------------------------------------------
# The initial-loss guard
# ---------------------------------------------------------------------


@pytest.mark.parametrize("regime", REGIMES)
def test_initial_loss_guard_fires_on_the_iteration_1_configuration(
        regime, recorded_warnings) -> None:
    loss, ratio, warned, mode = _initial_loss_check(ITERATION_1_PAIR, regime)

    assert warned is True
    assert ratio > 10.0, f"glorot_normal + standardize started at {ratio:.1f}x ln(C)"
    assert ratio == pytest.approx(loss / np.log(10.0))
    assert len(recorded_warnings) == 1, recorded_warnings
    text = recorded_warnings[0]
    assert "--kernel-initializer" in text and "--input-scaling" in text, text
    assert f"{loss:.4f}" in text and f"{ratio:.1f}x" in text, text


@pytest.mark.parametrize("regime", REGIMES)
def test_the_chosen_default_pair_does_not_warn(regime, recorded_warnings) -> None:
    """The POSITIVE arm: the guard must stay quiet on the configuration it recommends."""
    loss, ratio, warned, mode = _initial_loss_check(CHOSEN_PAIR, regime)

    assert warned is False
    assert ratio <= 3.0, f"{CHOSEN_PAIR} started at {ratio:.2f}x ln(C)"
    assert mode == "inference", "the non-BN default keeps model.evaluate"
    assert recorded_warnings == []


@pytest.mark.parametrize("regime", REGIMES)
def test_trainingconfig_defaults_start_near_the_uniform_loss(
        regime, recorded_warnings) -> None:
    """The default-pair pin: what ``TrainingConfig()`` actually selects, not a literal.

    Setting the two defaults back to ``glorot_normal`` + ``standardize`` (the
    iteration-1 pair) makes this fail while the literal-pair test above still passes.
    """
    defaults = tpm.TrainingConfig()
    loss, ratio, warned, mode = _initial_loss_check(
        (defaults.kernel_initializer, defaults.input_scaling), regime,
        batch_normalization=defaults.batch_normalization,
    )

    assert ratio <= 3.0, (
        f"TrainingConfig() defaults ({defaults.kernel_initializer}, "
        f"{defaults.input_scaling}) start at {ratio:.1f}x ln(C)"
    )
    assert warned is False
    assert recorded_warnings == []


def test_training_config_defaults_are_the_d025_outcome() -> None:
    """Literal pin of the D-021 grid outcome recorded in D-025.

    The 3-seed x 10-epoch MNIST grid chose lecun_normal + BN on (unit inputs).
    Changing a default must go through a new grid and a new decision, so it must
    fail here first; the pin is literal on purpose (the guards above read the
    defaults and would follow a silent change).
    """
    defaults = tpm.TrainingConfig()
    assert (defaults.kernel_initializer, defaults.batch_normalization) == (
        "lecun_normal", True,
    )
    assert defaults.input_scaling == "unit"
    args = tpm.parse_arguments([])
    assert (args.kernel_initializer, args.batch_normalization, args.input_scaling) == (
        "lecun_normal", True, "unit",
    )


# ---------------------------------------------------------------------
# BatchNorm: the guard measures the regime of the first training step (C4)
# ---------------------------------------------------------------------

#: The review's discriminating case: glorot_normal + standardize + k=3 is a huge
#: initial scale. Without BN it is hazardous (ratio 1e8-1e11) and must warn; with BN
#: the first TRAINING step sees batch statistics and starts near ln(C) (1.14-1.16),
#: while the inference-mode pass sees the untrained (0, 1) moving statistics
#: (4.2e11 on real MNIST). k=3 is what makes the two modes differ this much.
BN_PAIR = ("glorot_normal", "standardize")


@pytest.mark.parametrize("regime", REGIMES)
def test_a_batch_normalized_model_is_measured_in_training_mode_and_does_not_warn(
        regime, recorded_warnings) -> None:
    loss, ratio, warned, mode = _initial_loss_check(
        BN_PAIR, regime, k=3, batch_normalization=True)

    assert mode == "training"
    assert ratio <= 3.0, f"BN training-mode ratio {ratio:.3g} (inference mode gives ~1e8-1e11)"
    assert ratio == pytest.approx(loss / np.log(10.0))
    assert warned is False
    assert recorded_warnings == []


@pytest.mark.parametrize("regime", REGIMES)
def test_the_same_configuration_without_batch_norm_warns(regime, recorded_warnings) -> None:
    """The discriminating twin: identical pair, k and data, BN off. If this did not
    warn, the BN arm above would pass on any measurement and prove nothing."""
    loss, ratio, warned, mode = _initial_loss_check(
        BN_PAIR, regime, k=3, batch_normalization=False)

    assert mode == "inference"
    assert warned is True and ratio > 1e3, f"non-BN ratio {ratio:.3g}"
    assert len(recorded_warnings) == 1, recorded_warnings


@pytest.mark.parametrize("regime", REGIMES)
def test_the_batch_norm_sanity_pass_leaves_every_weight_bit_identical(regime) -> None:
    """A training-mode pass assigns ``moving_mean`` / ``moving_variance``; the check
    must snapshot and restore, or a BN run would start with drifted statistics."""
    keras.utils.set_random_seed(0)
    x, y = _guard_slice(regime, "standardize")
    model = _build_model("glorot_normal", k=3, batch_normalization=True)

    # Control: a bare training-mode call DOES change the moving statistics, so the
    # equality below is a statement about the restore and not about BN being inert.
    before_control = model.get_weights()
    model(x, training=True)
    changed = [not np.array_equal(a, b) for a, b in zip(before_control, model.get_weights())]
    assert sum(changed) >= 6, "3 BN layers x (moving_mean, moving_variance) must move"
    model.set_weights(before_control)

    tpm._check_initial_loss(model, x, y)

    after = model.get_weights()
    assert len(after) == len(before_control)
    for i, (a, b) in enumerate(zip(before_control, after)):
        np.testing.assert_array_equal(a, b, err_msg=f"weight {i} changed by the sanity check")


@pytest.mark.parametrize("regime", REGIMES)
def test_a_model_without_batch_norm_keeps_model_evaluate_bit_identically(
        regime, recorded_warnings) -> None:
    """The non-BN path is untouched: ``mode == "inference"`` and the loss is the
    number ``model.evaluate`` returns for the same batch and weights."""
    keras.utils.set_random_seed(0)
    x, y = _guard_slice(regime, "unit")
    model = _build_model("lecun_normal")

    loss, ratio, warned, mode = tpm._check_initial_loss(model, x, y)
    reference = float(model.evaluate(x, y, batch_size=tpm.SANITY_SAMPLES, verbose=0,
                                     return_dict=True)["loss"])

    assert mode == "inference"
    assert loss == reference
    assert warned is False


def test_lr_reduction_epochs_are_the_epochs_trained_at_a_lower_rate() -> None:
    """1-based; ``lr`` is the rate USED in each epoch, so a drop at index i (0-based)
    is reported as epoch i + 1."""
    assert tpm._lr_reduction_epochs([3e-4, 3e-4, 1.5e-4, 1.5e-4, 7.5e-5]) == [3, 5]
    assert tpm._lr_reduction_epochs([3e-4] * 6) == []
    assert tpm._lr_reduction_epochs([]) == [] and tpm._lr_reduction_epochs([3e-4]) == []
    assert tpm._lr_reduction_epochs([1e-3, 5e-4]) == [2]


class _StubModel:
    """Just what ``_check_initial_loss`` reads: ``evaluate``, ``hidden_units`` and
    ``batch_normalization`` (False: the inference-mode ``evaluate`` path)."""

    def __init__(self, loss: float, classes: int = 10) -> None:
        self._loss = loss
        self.hidden_units = [784, 64, classes]
        self.batch_normalization = False

    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        return {"loss": self._loss}


@pytest.mark.parametrize(
    "factor,expected_warned",
    [(9.99, False), (10.0, False), (10.01, True)],
    ids=["below", "exactly-at-the-threshold", "above"],
)
def test_the_warning_threshold_is_strict(factor, expected_warned, recorded_warnings) -> None:
    """``ratio == INITIAL_LOSS_WARN_FACTOR`` does NOT warn; only ``>`` does."""
    assert tpm.INITIAL_LOSS_WARN_FACTOR == 10.0
    uniform = float(np.log(10))
    x, y = np.zeros((4, 784), np.float32), np.zeros(4, np.int32)

    loss, ratio, warned, mode = tpm._check_initial_loss(_StubModel(factor * uniform), x, y)

    assert ratio == factor, "the stub must land exactly on the ratio under test"
    assert warned is expected_warned
    assert len(recorded_warnings) == int(expected_warned)


# ---------------------------------------------------------------------
# input_scaling
# ---------------------------------------------------------------------


def _ramp_load_dataset(name, *args, **kwargs):
    """MNIST-shaped images whose pixels sweep the whole [0, 1] range (0 and 1 included)."""
    assert name == "mnist", name
    n = 20
    ramp = np.linspace(0.0, 1.0, n * 28 * 28, dtype=np.float32).reshape(n, 28, 28, 1)
    x = np.repeat(ramp, 3, axis=-1)
    y = (np.arange(n) % 10).astype(np.uint8)
    return (x, y), (x[:4].copy(), y[:4].copy()), (28, 28, 3), 10


def _ramp_pixels() -> np.ndarray:
    return np.linspace(0.0, 1.0, 20 * 28 * 28, dtype=np.float32)


def test_unit_scaling_returns_the_raw_pixels_and_an_identity_unscale(monkeypatch) -> None:
    monkeypatch.setattr(tpm, "load_dataset", _ramp_load_dataset)
    (xt, _), (xv, _), _, info = tpm.prepare_data("mnist", 0.5, 0, "unit")

    everything = np.sort(np.concatenate([xt, xv]).ravel())
    assert everything.min() >= 0.0 and everything.max() <= 1.0
    np.testing.assert_array_equal(everything, np.sort(_ramp_pixels()))
    np.testing.assert_array_equal(info["mean"], [0.0])
    np.testing.assert_array_equal(info["std"], [1.0])
    assert info["input_scaling"] == "unit"


def test_standardize_scaling_subtracts_the_mnist_mean_and_divides_by_the_std(monkeypatch) -> None:
    monkeypatch.setattr(tpm, "load_dataset", _ramp_load_dataset)
    (xt, _), (xv, _), _, info = tpm.prepare_data("mnist", 0.5, 0, "standardize")

    everything = np.sort(np.concatenate([xt, xv]).ravel())
    expected = np.sort((_ramp_pixels() - tpm.MNIST_MEAN) / tpm.MNIST_STD)
    np.testing.assert_allclose(everything, expected, rtol=1e-5, atol=1e-6)
    assert everything.min() < 0.0, "standardized pixels leave [0, 1]"
    np.testing.assert_allclose(info["mean"], [tpm.MNIST_MEAN], rtol=1e-6)
    np.testing.assert_allclose(info["std"], [tpm.MNIST_STD], rtol=1e-6)
    assert info["input_scaling"] == "standardize"


def _cifar_load_dataset(name, *args, **kwargs):
    assert name == "cifar10", name
    n = 20
    base = np.linspace(0.0, 1.0, n * 32 * 32, dtype=np.float32).reshape(n, 32, 32, 1)
    x = np.concatenate([base, base ** 2, 1.0 - base], axis=-1)  # three different channels
    y = (np.arange(n) % 10).astype(np.uint8)
    return (x, y), (x[:4].copy(), y[:4].copy()), (32, 32, 3), 10


@pytest.mark.parametrize("input_scaling", tpm.INPUT_SCALINGS)
def test_cifar10_scaling_is_per_channel_and_unit_is_the_identity(
        monkeypatch, input_scaling) -> None:
    """The never-run CIFAR-10 path: 3072 features, per-channel mean/std, RGB layout."""
    monkeypatch.setattr(tpm, "load_dataset", _cifar_load_dataset)
    (xt, _), (xv, _), (xs, _), info = tpm.prepare_data("cifar10", 0.5, 0, input_scaling)
    (raw, _), _, _, _ = _cifar_load_dataset("cifar10")

    assert xt.shape[1] == xv.shape[1] == xs.shape[1] == info["input_dim"] == 3072
    assert info["image_shape"] == (32, 32, 3)
    got = np.concatenate([xt, xv]).reshape(-1, 32, 32, 3)
    if input_scaling == "unit":
        want_mean, want_std = np.zeros(3), np.ones(3)
    else:
        want_mean, want_std = np.asarray(tpm.CIFAR10_MEAN), np.asarray(tpm.CIFAR10_STD)
    np.testing.assert_allclose(info["mean"], want_mean, rtol=1e-6)
    np.testing.assert_allclose(info["std"], want_std, rtol=1e-6)
    for c in range(3):
        expected = np.sort(((raw[..., c] - want_mean[c]) / want_std[c]).ravel())
        np.testing.assert_allclose(np.sort(got[..., c].ravel()), expected, rtol=1e-5, atol=1e-6)


def test_prepare_data_refuses_an_unknown_input_scaling(monkeypatch) -> None:
    monkeypatch.setattr(tpm, "load_dataset", _ramp_load_dataset)
    with pytest.raises(ValueError, match="input_scaling"):
        tpm.prepare_data("mnist", 0.1, 0, "whiten")


def test_the_config_refuses_an_unknown_input_scaling() -> None:
    with pytest.raises(ValueError, match="input_scaling"):
        tpm.TrainingConfig(input_scaling="whiten")


def test_the_config_refuses_an_unknown_kernel_initializer() -> None:
    with pytest.raises(ValueError, match="kernel_initializer"):
        tpm.TrainingConfig(kernel_initializer="orthogonal")


# ---------------------------------------------------------------------
# train_model forwards the knobs
# ---------------------------------------------------------------------


class _Probe(Exception):
    """Raised by a sentinel so ``train_model`` stops at the call under test."""


def _config(tmp_path: Path, **overrides: Any) -> tpm.TrainingConfig:
    return tpm.TrainingConfig(
        epochs=1, batch_size=64, seed=3, output_dir=str(tmp_path),
        experiment_name="forward_probe", **overrides,
    )


@pytest.mark.parametrize("name", tpm.KERNEL_INITIALIZERS)
def test_train_model_forwards_the_configured_kernel_initializer(
        monkeypatch, tmp_path, name) -> None:
    monkeypatch.setattr(tpm, "load_dataset", _synthetic_load_dataset())
    seen: Dict[str, Any] = {}

    def sentinel(*args, **kwargs):
        seen.update(kwargs)
        raise _Probe

    monkeypatch.setattr(tpm, "PowerMLP", sentinel)
    with pytest.raises(_Probe):
        tpm.train_model(_config(tmp_path, kernel_initializer=name))
    assert seen["kernel_initializer"] == name


@pytest.mark.parametrize("scaling", tpm.INPUT_SCALINGS)
def test_train_model_forwards_the_configured_input_scaling(
        monkeypatch, tmp_path, scaling) -> None:
    seen: Dict[str, Any] = {}

    def sentinel(dataset, validation_split, seed, input_scaling):
        seen["input_scaling"] = input_scaling
        raise _Probe

    monkeypatch.setattr(tpm, "prepare_data", sentinel)
    with pytest.raises(_Probe):
        tpm.train_model(_config(tmp_path, input_scaling=scaling))
    assert seen["input_scaling"] == scaling


@pytest.mark.parametrize("epoch_analysis", [False, True])
def test_train_model_forwards_epoch_analysis_into_create_callbacks(
        monkeypatch, tmp_path, epoch_analysis) -> None:
    monkeypatch.setattr(tpm, "load_dataset", _synthetic_load_dataset())
    seen: Dict[str, Any] = {}

    def sentinel(*args, **kwargs):
        seen.update(kwargs)
        raise _Probe

    monkeypatch.setattr(tpm, "create_callbacks", sentinel)
    with pytest.raises(_Probe):
        tpm.train_model(_config(tmp_path, epoch_analysis=epoch_analysis))
    assert seen["include_analyzer"] is epoch_analysis


# ---------------------------------------------------------------------
# End to end: the iteration-1 configuration through the real trainer
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def iteration_1_run(tmp_path_factory):
    """One real 1-epoch ``train_model`` on ``glorot_normal`` + ``standardize``.

    The synthetic batch (measured ratio ~30) is what makes the guard fire, so the
    hop ``_check_initial_loss -> summary`` is exercised on the path users run.
    """
    out_root = tmp_path_factory.mktemp("power_mlp_iter1_config")
    seen = []
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(tpm, "load_dataset", _synthetic_load_dataset())
        mp.setattr(tpm.logger, "warning", lambda msg, *a, **k: seen.append(str(msg)))
        config = tpm.config_from_args(tpm.parse_arguments([
            "--epochs", "1", "--batch-size", "64", "--seed", "3",
            "--kernel-initializer", "glorot_normal", "--input-scaling", "standardize",
            "--no-batch-normalization", "--output-dir", str(out_root), "--experiment-name", "iter1_config",
        ]))
        summary = tpm.train_model(config)
    finally:
        mp.undo()
    return types.SimpleNamespace(
        run_dir=out_root / "iter1_config", summary=summary, warnings=seen
    )


def test_the_real_trainer_records_the_init_scale_warning_in_the_summary(iteration_1_run) -> None:
    on_disk = json.loads((iteration_1_run.run_dir / "results_summary.json").read_text())
    assert on_disk["init_scale_warning"] is True
    assert on_disk["initial_loss_ratio"] > tpm.INITIAL_LOSS_WARN_FACTOR
    assert on_disk["kernel_initializer"] == "glorot_normal"
    assert on_disk["input_scaling"] == "standardize"
    config = json.loads((iteration_1_run.run_dir / "config.json").read_text())
    assert config["kernel_initializer"] == "glorot_normal"
    assert config["input_scaling"] == "standardize"


def test_the_real_trainer_logs_the_init_scale_warning_once(iteration_1_run) -> None:
    init_warnings = [w for w in iteration_1_run.warnings if "--kernel-initializer" in w]
    assert len(init_warnings) == 1, iteration_1_run.warnings
    assert "--input-scaling" in init_warnings[0]
