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


# ---------------------------------------------------------------------
# Phase 2: 3 new norm classes are detected and produce the expected `kind`
# ---------------------------------------------------------------------


def test_norm_layer_classes_tuple_extended() -> None:
    """Tuple is the 8-variant Phase 3 universe (4 originals + 4 new)."""
    assert len(NORM_LAYER_CLASSES) == 8
    names = {cls.__name__ for cls in NORM_LAYER_CLASSES}
    assert {
        "RMSNorm",
        "BandRMS",
        "ZeroCenteredRMSNorm",
        "ZeroCenteredBandRMSNorm",
        "AdaptiveBandRMS",
        "BandLogitNorm",
        "DynamicTanh",
        "ZeroCenteredAdaptiveBandRMS",
    } == names


def test_norm_layer_classes_count() -> None:
    """SC-3: NORM_LAYER_CLASSES length is 8."""
    assert len(NORM_LAYER_CLASSES) == 8


def _build_new_variants_model(d: int = 16) -> keras.Model:
    """Dummy: x → Dense → AdaptiveBandRMS → Dense → BandLogitNorm → Dense → DynamicTanh → Dense(1)."""
    inputs = keras.Input(shape=(d,))
    x = keras.layers.Dense(d, name="dense1")(inputs)
    x = create_normalization_layer(
        "adaptive_band_rms",
        epsilon=1e-6,
        max_band_width=0.1,
        band_regularizer=None,
        name="adaptive",
    )(x)
    x = keras.layers.Dense(d, name="dense2")(x)
    x = create_normalization_layer(
        "band_logit_norm",
        epsilon=1e-6,
        max_band_width=0.1,
        name="blogit",
    )(x)
    x = keras.layers.Dense(d, name="dense3")(x)
    x = create_normalization_layer(
        "dynamic_tanh",
        axis=-1,
        alpha_init_value=0.5,
        name="dyt",
    )(x)
    outputs = keras.layers.Dense(1, name="out")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def test_find_norm_layers_detects_three_new_variants(dummy_data) -> None:
    """`_find_norm_layers` returns each of the 3 new outer classes exactly once."""
    model = _build_new_variants_model()
    _ = model(tf.convert_to_tensor(dummy_data[0]))
    found = _find_norm_layers(model)
    found_names = {layer.name for layer in found}
    # The OUTER 3 norm layers — and crucially NOT the inner LayerNormalization
    # sublayer of BandLogitNorm (its name is "blogit_layer_norm", and the
    # vanilla `keras.layers.LayerNormalization` is not in NORM_LAYER_CLASSES).
    assert {"adaptive", "blogit", "dyt"}.issubset(found_names)
    assert "blogit_layer_norm" not in found_names


def test_internal_stats_kinds_for_new_variants(tmp_path, dummy_data) -> None:
    model = _build_new_variants_model()
    _ = model(tf.convert_to_tensor(dummy_data[0]))
    cb = NormInternalStatsCallback(out_dir=str(tmp_path))
    cb.set_model(model)
    cb.on_epoch_end(0)
    with open(os.path.join(str(tmp_path), "norm_internal.csv")) as f:
        rows = list(csv.reader(f))
    body = rows[1:]
    kind_by_layer = {r[1]: r[2] for r in body}
    assert kind_by_layer["adaptive"] == "adaptive_band_family"
    assert kind_by_layer["blogit"] == "band_logit_family"
    assert kind_by_layer["dyt"] == "dyt_family"

    # adaptive_band_family: scale_l2 ≥ 0 (Dense kernel L2 is real-valued);
    # post in [1-α, 1]
    adaptive_row = next(r for r in body if r[1] == "adaptive")
    adaptive_l2 = float(adaptive_row[3])
    adaptive_post = float(adaptive_row[4])
    assert adaptive_l2 >= 0.0
    assert 0.9 - 1e-4 <= adaptive_post <= 1.0 + 1e-4

    # band_logit_family: gamma L2 ≥ 0; post_sigmoid_scale is NaN by design
    blogit_row = next(r for r in body if r[1] == "blogit")
    assert float(blogit_row[3]) >= 0.0
    assert blogit_row[4].lower() == "nan"

    # dyt_family: raw alpha == 0.5 (default); post NaN
    dyt_row = next(r for r in body if r[1] == "dyt")
    np.testing.assert_allclose(float(dyt_row[3]), 0.5, atol=1e-6)
    assert dyt_row[4].lower() == "nan"


def test_weight_norm_callback_tracks_new_variants(tmp_path, dummy_data) -> None:
    """WeightNormTrajectoryCallback auto-detects the 3 new classes via NORM_LAYER_CLASSES."""
    model = _build_new_variants_model()
    _ = model(tf.convert_to_tensor(dummy_data[0]))
    cb = WeightNormTrajectoryCallback(out_dir=str(tmp_path))
    cb.set_model(model)
    cb.on_epoch_end(0)
    with open(os.path.join(str(tmp_path), "weight_norm.csv")) as f:
        rows = list(csv.reader(f))
    layer_names = {r[1] for r in rows[1:]}
    assert {"adaptive", "blogit", "dyt"}.issubset(layer_names)


def test_activation_callback_tracks_new_variants(tmp_path, dummy_data) -> None:
    """NormLayerActivationCallback records one row per outer norm layer."""
    model = _build_new_variants_model()
    x = tf.convert_to_tensor(dummy_data[0])
    cb = NormLayerActivationCallback(calibration_data=x, out_dir=str(tmp_path))
    cb.set_model(model)
    cb.on_epoch_end(0)
    with open(os.path.join(str(tmp_path), "activation_stats.csv")) as f:
        rows = list(csv.reader(f))
    body = rows[1:]
    layer_names = {r[1] for r in body}
    assert {"adaptive", "blogit", "dyt"}.issubset(layer_names)


# ---------------------------------------------------------------------
# Phase 3: ZeroCenteredAdaptiveBandRMS isinstance branch (SC-4)
# ---------------------------------------------------------------------


def _build_zc_adaptive_model(d: int = 16) -> keras.Model:
    """Dummy model wrapping ZeroCenteredAdaptiveBandRMS (8th variant)."""
    inputs = keras.Input(shape=(d,))
    x = keras.layers.Dense(d, name="dense1")(inputs)
    x = create_normalization_layer(
        "zero_centered_adaptive_band_rms_norm",
        epsilon=1e-6,
        max_band_width=0.1,
        band_regularizer=None,
        name="zc_adaptive",
    )(x)
    outputs = keras.layers.Dense(1, name="out")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def test_zc_adaptive_band_family_branch(tmp_path, dummy_data) -> None:
    """SC-4: NormInternalStatsCallback emits kind='zc_adaptive_band_family' row."""
    model = _build_zc_adaptive_model()
    _ = model(tf.convert_to_tensor(dummy_data[0]))
    cb = NormInternalStatsCallback(out_dir=str(tmp_path))
    cb.set_model(model)
    cb.on_epoch_end(0)
    with open(os.path.join(str(tmp_path), "norm_internal.csv")) as f:
        rows = list(csv.reader(f))
    body = rows[1:]
    kind_by_layer = {r[1]: r[2] for r in body}
    assert kind_by_layer["zc_adaptive"] == "zc_adaptive_band_family"
    zc_row = next(r for r in body if r[1] == "zc_adaptive")
    # kernel L2 ≥ 0, post in [1-α, 1]
    assert float(zc_row[3]) >= 0.0
    post = float(zc_row[4])
    assert 0.9 - 1e-4 <= post <= 1.0 + 1e-4


# ---------------------------------------------------------------------
# Phase 3: CalibrationCallback + RobustnessProbe schema tests (SC-6)
# ---------------------------------------------------------------------


class TestCalibrationCallback:
    def _model_and_data(self, n: int = 32, d: int = 8, c: int = 3):
        inp = keras.Input(shape=(d,))
        out = keras.layers.Dense(c)(inp)
        model = keras.Model(inp, out)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        rng = np.random.RandomState(0)
        x = rng.randn(n, d).astype("float32")
        y = rng.randint(0, c, n).astype("int64")
        return model, x, y, c

    def test_sparse_labels_writes_expected_schema(self, tmp_path) -> None:
        from train.rms_variants_train.callbacks import CalibrationCallback
        model, x, y, c = self._model_and_data()
        cb = CalibrationCallback(
            val_data=(x, y), num_classes=c, out_dir=str(tmp_path)
        )
        model.fit(x, y, epochs=1, batch_size=8, verbose=0, callbacks=[cb])
        path = os.path.join(str(tmp_path), "calibration.csv")
        assert os.path.exists(path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert rows[0] == [
            "ece_15", "brier_score", "n_samples", "n_classes", "n_bins",
        ]
        assert len(rows) == 2
        ece = float(rows[1][0])
        brier = float(rows[1][1])
        assert 0.0 <= ece <= 1.0
        assert brier >= 0.0
        assert int(rows[1][3]) == c
        assert int(rows[1][4]) == 15

    def test_skips_when_num_classes_lt_2(self, tmp_path) -> None:
        from train.rms_variants_train.callbacks import CalibrationCallback
        model, x, y, _ = self._model_and_data()
        cb = CalibrationCallback(
            val_data=(x, y), num_classes=1, out_dir=str(tmp_path)
        )
        model.fit(x, y, epochs=1, batch_size=8, verbose=0, callbacks=[cb])
        # Gate triggered → no CSV emitted.
        assert not os.path.exists(os.path.join(str(tmp_path), "calibration.csv"))


class TestRobustnessProbe:
    def _model_and_data(self, n: int = 32, d: int = 8, c: int = 3):
        inp = keras.Input(shape=(d,))
        out = keras.layers.Dense(c)(inp)
        model = keras.Model(inp, out)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        rng = np.random.RandomState(0)
        x = rng.randn(n, d).astype("float32")
        y = rng.randint(0, c, n).astype("int64")
        return model, x, y

    def test_writes_one_row_per_sigma_plus_clean(self, tmp_path) -> None:
        from train.rms_variants_train.callbacks import RobustnessProbe
        model, x, y = self._model_and_data()
        sigmas = (0.05, 0.1)
        cb = RobustnessProbe(
            val_data=(x, y), out_dir=str(tmp_path), sigmas=sigmas
        )
        model.fit(x, y, epochs=1, batch_size=8, verbose=0, callbacks=[cb])
        path = os.path.join(str(tmp_path), "robustness.csv")
        assert os.path.exists(path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["sigma", "val_acc", "n_samples"]
        # 1 header + clean baseline + len(sigmas) noisy rows
        assert len(rows) == 1 + 1 + len(sigmas)
        # First data row is the clean baseline (sigma = 0).
        assert float(rows[1][0]) == 0.0
        # Each row's val_acc is in [0, 1].
        for r in rows[1:]:
            acc = float(r[1])
            assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------
# DistributionShiftProbe (plan_e1f12eab Step 4)
# ---------------------------------------------------------------------


class TestDistributionShiftProbe:
    """Tests for the CIFAR-10-C-style distribution-shift probe.

    Two soft-fail paths are exercised here without touching TFDS:
    (a) missing dataset name → reason='dataset_missing:...';
    (b) ctor / CSV schema verification.
    Network-dependent end-to-end coverage is left to the Step 9 smoke gate.
    """

    def test_constructor_does_not_crash(self):
        from train.rms_variants_train.callbacks import DistributionShiftProbe
        probe = DistributionShiftProbe(
            out_dir="/tmp/_dist_shift_ctor_test",
            corruptions=("gaussian_noise",),
            severity=3,
        )
        assert probe._severity == 3
        assert probe._corruptions == ("gaussian_noise",)
        assert probe._csv_path.endswith("dist_shift.csv")

    def test_soft_fail_on_nonexistent_dataset_template(self, tmp_path):
        """A nonsense dataset_name_template that TFDS cannot resolve → the
        probe must emit dist_shift.csv with `reason` populated, NOT raise."""
        from train.rms_variants_train.callbacks import DistributionShiftProbe

        # Build a trivial model so .predict won't run if a row makes it that
        # far (the dataset load should fail first).
        inputs = keras.Input(shape=(4,))
        outputs = keras.layers.Dense(2)(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        probe = DistributionShiftProbe(
            out_dir=str(tmp_path),
            dataset_name_template="totally_fake_corrupted_dataset_xyz/{corruption}_{severity}",
            corruptions=("gaussian_noise", "defocus_blur"),
            severity=3,
        )
        probe.set_model(model)
        # Must NOT raise.
        probe.on_train_end()
        csv_path = tmp_path / "dist_shift.csv"
        assert csv_path.exists(), "dist_shift.csv not written on soft-fail"
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        # Header + 2 rows (one per corruption attempted).
        assert rows[0] == ["corruption", "severity", "val_acc", "n_samples", "reason"]
        assert len(rows) == 3, f"Expected 3 rows (header + 2 corruptions); got {len(rows)}"
        # Each row's reason field should be non-empty.
        for r in rows[1:]:
            reason = r[4]
            assert reason, f"Soft-fail row missing reason: {r}"

    def test_soft_fail_on_missing_tfds_dependency(self, tmp_path, monkeypatch):
        """Simulate `import tensorflow_datasets` raising ImportError → the
        probe must emit a single soft-fail row, not raise."""
        from train.rms_variants_train.callbacks import DistributionShiftProbe

        # Force the lazy `import tensorflow_datasets` inside on_train_end to
        # raise. We do this by injecting a sentinel into sys.modules.
        import sys
        # Save real module if loaded.
        real = sys.modules.pop("tensorflow_datasets", None)

        class _RaisingModule:
            def __getattr__(self, name):
                raise ImportError("simulated missing tfds for test")

        sys.modules["tensorflow_datasets"] = _RaisingModule()
        try:
            inputs = keras.Input(shape=(4,))
            outputs = keras.layers.Dense(2)(inputs)
            model = keras.Model(inputs, outputs)
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

            probe = DistributionShiftProbe(out_dir=str(tmp_path))
            probe.set_model(model)
            probe.on_train_end()

            csv_path = tmp_path / "dist_shift.csv"
            assert csv_path.exists()
            with open(csv_path, newline="") as f:
                rows = list(csv.reader(f))
            # Header + 1 soft-fail row with the import-error reason OR a
            # broken-import path. In either case dist_shift.csv exists with
            # a non-empty reason.
            assert rows[0] == ["corruption", "severity", "val_acc", "n_samples", "reason"]
            # The probe may either hit the ImportError soft-fail path
            # (single row) or, depending on how the sentinel interacts with
            # tfds.core.registered access, fall through to per-corruption
            # soft-fails. Either way the CSV must contain at least one row
            # with a non-empty `reason`.
            assert len(rows) >= 2, "Expected at least one data row"
            assert all(r[4] for r in rows[1:]), "All data rows must have a reason"
        finally:
            sys.modules.pop("tensorflow_datasets", None)
            if real is not None:
                sys.modules["tensorflow_datasets"] = real

    def test_appears_in__all__(self):
        from train.rms_variants_train import callbacks
        assert "DistributionShiftProbe" in callbacks.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
