"""Guards for the batch shape ``eval_psnr_vs_noise._predict_denoised`` feeds ``model.predict``.

A run whose best epoch is not its last scores two models. A short last batch (100 patches at
batch 16 leave 4) is a second input shape, so each model traces its predict function twice, and
TensorFlow counts those traces across ALL models of the process: with the two predicts the
trainer makes before its test evaluation, the fifth trace inside a window of ten calls printed
``5 out of the last 10 calls ... triggered tf.function retracing`` (measured on a real
40-epoch run, D-057). The padding keeps one shape per model; these tests count the traces
(the warning itself has a margin of one trace, so a count is the guard) and pin that the
padding changes no number.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

from train.bfunet import eval_psnr_vs_noise as ev  # noqa: E402

SIZE = 8
BATCH = 4
N_PATCHES = 10           # 4 + 4 + 2: a short last batch


def _model(outputs: int = 1) -> keras.Model:
    """A tiny denoiser with batch norm whose moving statistics are not the identity."""
    inputs = keras.Input((SIZE, SIZE, 3))
    x = keras.layers.Conv2D(4, 3, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    heads = [keras.layers.Conv2D(3, 3, padding="same")(x) for _ in range(outputs)]
    model = keras.Model(inputs, heads if outputs > 1 else heads[0])
    bn = next(layer for layer in model.layers if isinstance(layer, keras.layers.BatchNormalization))
    bn.moving_mean.assign(np.linspace(-0.3, 0.3, 4).astype("float32"))
    bn.moving_variance.assign(np.linspace(0.5, 2.0, 4).astype("float32"))
    return model


def _write_pngs(directory: Path, count: int) -> List[str]:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for index in range(count):
        path = directory / f"img_{index}.png"
        path.write_bytes(tf.io.encode_png(rng.integers(1, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)).numpy())
        paths.append(str(path))
    return paths


def _config() -> "ev.EvalConfig":
    return ev.EvalConfig(
        models={}, datasets={}, sigmas_255=[15, 25, 50], num_samples=N_PATCHES, patch_size=SIZE,
        channels=3, batch_size=BATCH, seed=1,
    )


@pytest.fixture
def exact_float32():
    """TF32 convolutions pick their kernel by batch size and differ at 3e-4 between a batch of 1 and of 4
    on an Ampere or newer GPU (measured); turn them off for the numeric pin and restore the flag."""
    enabled = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    yield
    tf.config.experimental.enable_tensor_float_32_execution(enabled)


class TestPredictTracesOncePerModel:
    """``TestPredictTracesOncePerModel``: the guard named by the D-057 anchor in ``_predict_denoised``."""

    def test_two_models_three_sigmas_two_datasets_trace_the_predict_function_once_each(self, tmp_path) -> None:
        models: Dict[str, keras.Model] = {"best": _model(), "final": _model()}
        cfg = _config()
        for name in ("kodak", "cbsd"):
            paths = _write_pngs(tmp_path / name, 6)
            rows = ev.evaluate_dataset(models, cfg, name, paths, np.random.RandomState(cfg.seed))
            assert len(rows) == 6, "two models x three sigmas"
        traces = {key: model.predict_function.experimental_get_tracing_count() for key, model in models.items()}
        assert traces == {"best": 1, "final": 1}, (
            f"the short last batch ({N_PATCHES} patches at batch {BATCH}) traced a second time: {traces}")

    @pytest.mark.parametrize("n", [1, 3, 8, N_PATCHES])
    @pytest.mark.parametrize("outputs", [1, 2])
    def test_the_padding_changes_no_number_and_keeps_every_row(self, exact_float32, n: int, outputs: int) -> None:
        """``n`` below, at and above the batch, a multiple of it or not, one output or a deep-supervision
        pair (output 0 is kept): the padded prediction equals the plain forward pass within 1e-5."""
        model = _model(outputs)
        x = np.random.default_rng(3).uniform(0.0, 1.0, size=(n, SIZE, SIZE, 3)).astype("float32")
        plain = model(x, training=False)
        plain = np.asarray(plain[0] if outputs > 1 else plain)
        predicted = ev._predict_denoised(model, x, BATCH)
        assert predicted.shape == plain.shape == (n, SIZE, SIZE, 3)
        assert float(np.max(np.abs(predicted - plain))) < 1e-5
