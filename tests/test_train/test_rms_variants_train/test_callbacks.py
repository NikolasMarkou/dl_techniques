"""Unit tests for the 4 probe callbacks.

Each callback is exercised on a 2-layer dummy model with mixed norm variants,
covering:
- CSV files are created with the right header.
- Rows are appended on ``on_epoch_end``.
- Targeted norm layers are detected via ``isinstance``.
- The activation callback's intermediate-Model trick does not mutate the
  original model's trainable state.
"""
from __future__ import annotations

import csv
import os
from typing import List

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.norms.factory import create_normalization_layer
from train.rms_variants_train.callbacks import (
    GradientNormCallback,
    NormInternalStatsCallback,
    NormLayerActivationCallback,
    NormLayerActivationCallback as _ActCb,  # alias
    NormInternalStatsCallback as _IntCb,
    NORM_LAYER_CLASSES,
    WeightNormTrajectoryCallback,
    _find_norm_layers,
    _walk_layers,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _build_dummy_model(d: int = 16) -> keras.Model:
    """Functional dummy: x → Dense → RMSNorm → Dense → BandRMS → Dense(1)."""
    inputs = keras.Input(shape=(d,))
    x = keras.layers.Dense(d, name="dense1")(inputs)
    x = create_normalization_layer("rms_norm", epsilon=1e-6, use_scale=True, name="norm1")(x)
    x = keras.layers.Dense(d, name="dense2")(x)
    x = create_normalization_layer(
        "band_rms", epsilon=1e-6, max_band_width=0.1, band_regularizer=None, name="norm2"
    )(x)
    outputs = keras.layers.Dense(1, name="out")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


@pytest.fixture
def dummy_data() -> tuple:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 16)).astype(np.float32)
    y = rng.normal(size=(8, 1)).astype(np.float32)
    return x, y


# ---------------------------------------------------------------------
# Layer-walker helpers
# ---------------------------------------------------------------------


def test_walk_layers_yields_norm_instances() -> None:
    model = _build_dummy_model()
    found = _find_norm_layers(model)
    assert len(found) == 2
    classes = {type(layer) for layer in found}
    # We expect one of RMSNorm-family and one of BandRMS-family
    assert any(isinstance(layer, NORM_LAYER_CLASSES) for layer in found)
    assert any(layer.name == "norm1" for layer in found)
    assert any(layer.name == "norm2" for layer in found)


def test_walk_layers_handles_models_without_subblocks() -> None:
    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    # No norm layers → empty
    assert _find_norm_layers(model) == []


# ---------------------------------------------------------------------
# GradientNormCallback
# ---------------------------------------------------------------------


def test_grad_norm_callback_writes_rows(tmp_path, dummy_data) -> None:
    model = _build_dummy_model()
    cb = GradientNormCallback(
        calibration_data=dummy_data,
        out_dir=str(tmp_path),
        loss_fn=keras.losses.MeanSquaredError(),
    )
    cb.set_model(model)
    cb.on_epoch_end(0)
    cb.on_epoch_end(1)
    csv_path = os.path.join(str(tmp_path), "grad_norm.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    # header + 2 rows
    assert rows[0] == ["epoch", "grad_norm_global", "grad_norm_max"]
    assert len(rows) == 3
    # Values should be finite numbers > 0
    for r in rows[1:]:
        assert float(r[1]) >= 0.0
        assert float(r[2]) >= 0.0


# ---------------------------------------------------------------------
# WeightNormTrajectoryCallback
# ---------------------------------------------------------------------


def test_weight_norm_callback_tracks_norm_layers_only(tmp_path, dummy_data) -> None:
    model = _build_dummy_model()
    # Run a forward to build the norm layers' weights.
    _ = model(tf.convert_to_tensor(dummy_data[0]))

    cb = WeightNormTrajectoryCallback(out_dir=str(tmp_path))
    cb.set_model(model)
    cb.on_epoch_end(0)
    csv_path = os.path.join(str(tmp_path), "weight_norm.csv")
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["epoch", "layer_name", "weight_name", "l2"]
    # Each row's layer_name should be one of the norm layers, never a Dense.
    body = rows[1:]
    assert len(body) >= 2  # at least one weight per norm layer
    for r in body:
        assert r[1] in {"norm1", "norm2"}
        assert float(r[3]) >= 0.0


# ---------------------------------------------------------------------
# NormLayerActivationCallback
# ---------------------------------------------------------------------


def test_activation_callback_records_per_layer(tmp_path, dummy_data) -> None:
    model = _build_dummy_model()
    x = tf.convert_to_tensor(dummy_data[0])
    # Capture pre-callback trainable vars to confirm the intermediate-model
    # trick does NOT mutate them.
    pre = [v.numpy().copy() for v in model.trainable_variables]
    cb = NormLayerActivationCallback(calibration_data=x, out_dir=str(tmp_path))
    cb.set_model(model)
    cb.on_epoch_end(0)
    post = [v.numpy() for v in model.trainable_variables]
    for a, b in zip(pre, post):
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)

    csv_path = os.path.join(str(tmp_path), "activation_stats.csv")
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "epoch",
        "layer_name",
        "mean",
        "per_sample_rms_std",
        "per_sample_rms_mean",
        "per_sample_rms_min",
        "per_sample_rms_max",
    ]
    body = rows[1:]
    # 2 norm layers
    assert len(body) == 2


def test_activation_callback_band_layer_rms_in_band(tmp_path) -> None:
    """For a model with only BandRMS, per_sample_rms_max <= 1.0 + slack."""
    inputs = keras.Input(shape=(32,))
    x = keras.layers.Dense(32)(inputs)
    x = create_normalization_layer(
        "band_rms", max_band_width=0.1, epsilon=1e-7, band_regularizer=None, name="b1"
    )(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    rng = np.random.default_rng(0)
    x_cal = rng.normal(size=(16, 32)).astype(np.float32) + 3.0  # biased
    cb = NormLayerActivationCallback(
        calibration_data=tf.convert_to_tensor(x_cal), out_dir=str(tmp_path)
    )
    cb.set_model(model)
    cb.on_epoch_end(0)
    with open(os.path.join(str(tmp_path), "activation_stats.csv")) as f:
        rows = list(csv.reader(f))
    body = rows[1:]
    assert len(body) == 1
    per_sample_rms_max = float(body[0][6])
    # BandRMS upper bound is 1.0, allow 1e-3 numerical slack
    assert per_sample_rms_max <= 1.0 + 1e-3


def test_activation_callback_zero_centered_has_zero_mean(tmp_path) -> None:
    """ZeroCenteredRMSNorm output must have ~0 mean even on biased input."""
    inputs = keras.Input(shape=(32,))
    x = keras.layers.Dense(32)(inputs)
    x = create_normalization_layer(
        "zero_centered_rms_norm", epsilon=1e-6, use_scale=False, name="zc1"
    )(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    rng = np.random.default_rng(0)
    x_cal = rng.normal(size=(16, 32)).astype(np.float32) + 5.0
    cb = NormLayerActivationCallback(
        calibration_data=tf.convert_to_tensor(x_cal), out_dir=str(tmp_path)
    )
    cb.set_model(model)
    cb.on_epoch_end(0)
    with open(os.path.join(str(tmp_path), "activation_stats.csv")) as f:
        rows = list(csv.reader(f))
    mean = float(rows[1][2])
    assert abs(mean) < 1e-3


# ---------------------------------------------------------------------
# NormInternalStatsCallback
# ---------------------------------------------------------------------


def test_internal_stats_callback_kinds(tmp_path, dummy_data) -> None:
    model = _build_dummy_model()
    _ = model(tf.convert_to_tensor(dummy_data[0]))
    cb = NormInternalStatsCallback(out_dir=str(tmp_path))
    cb.set_model(model)
    cb.on_epoch_end(0)
    with open(os.path.join(str(tmp_path), "norm_internal.csv")) as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "epoch",
        "layer_name",
        "kind",
        "scale_l2_or_raw",
        "post_sigmoid_scale",
    ]
    body = rows[1:]
    kinds = {r[2] for r in body}
    assert "rms_family" in kinds
    assert "band_family" in kinds
    # rms_family rows have NaN post_sigmoid_scale
    rms_rows = [r for r in body if r[2] == "rms_family"]
    for r in rms_rows:
        assert r[4] == "nan" or r[4].lower() == "nan" or r[4] == "" or "nan" in r[4].lower()
    # band_family rows: post_sigmoid in [1-alpha, 1]
    band_rows = [r for r in body if r[2] == "band_family"]
    for r in band_rows:
        post = float(r[4])
        assert 0.0 < post <= 1.0 + 1e-4


def test_internal_stats_param_matched_rms_scale_l2_zero(tmp_path) -> None:
    """When use_scale=False, scale_l2_or_raw should be 0.0 for RMSNorm-family."""
    inputs = keras.Input(shape=(16,))
    x = create_normalization_layer(
        "rms_norm", epsilon=1e-6, use_scale=False, name="r1"
    )(inputs)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    _ = model(tf.zeros((2, 16)))
    cb = NormInternalStatsCallback(out_dir=str(tmp_path))
    cb.set_model(model)
    cb.on_epoch_end(0)
    with open(os.path.join(str(tmp_path), "norm_internal.csv")) as f:
        rows = list(csv.reader(f))
    body = rows[1:]
    assert len(body) == 1
    assert body[0][2] == "rms_family"
    assert float(body[0][3]) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
