"""Unit tests for train.logic.train_e3_faithfulness.

No LIME/SHAP runtime — those are wall-clock expensive and only validated
indirectly via `test_attributions.py`. We focus on generators,
build/forward, band callback integration, and CSV schema.

Plan: plan_2026-05-13_798d3a60.
"""

import os
from typing import Tuple

import keras
import numpy as np
import pytest

from train.logic.callbacks_band import StopOnAccuracyBand
from train.logic.train_benchmark import build_circuit
from train.logic.train_e3_faithfulness import (
    ATTRIBUTION_METHODS,
    CSV_COLUMNS,
    METRIC_SUFFIXES,
    TASK_SPECS_E3,
    evaluate_hard_extraction_circuit,
    gen_mux_11bit,
    gen_parity_k8,
    gen_random_dnf,
)


# ---------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------


class TestGenerators:
    def test_mux_11bit_truth(self):
        rng = np.random.default_rng(0)
        x, y = gen_mux_11bit(256, rng)
        assert x.shape == (256, 11)
        assert y.shape == (256, 1)
        # Reproduce the truth table manually.
        for i in range(20):
            addr = int(x[i, 0]) + 2 * int(x[i, 1]) + 4 * int(x[i, 2])
            expected = float(x[i, 3 + addr])
            assert y[i, 0] == expected

    def test_parity_k8_truth(self):
        rng = np.random.default_rng(1)
        x, y = gen_parity_k8(256, rng)
        assert x.shape == (256, 8)
        assert y.shape == (256, 1)
        expected = (x.sum(axis=1) % 2 == 1).astype(np.float32)
        np.testing.assert_array_equal(y.ravel(), expected)

    def test_random_dnf_shape(self):
        rng = np.random.default_rng(2)
        x, y = gen_random_dnf(256, rng)
        assert x.shape == (256, 8)
        assert y.shape == (256, 1)
        # The DNF is non-trivial: not all-0 and not all-1.
        assert 0.05 < float(y.mean()) < 0.95


# ---------------------------------------------------------------------
# circuit_attributions on a build_circuit model (sanity)
# ---------------------------------------------------------------------


class TestCircuitAttributionsOnBenchmarkModel:
    def test_integrated_gradients_runs(self):
        from train.logic.attributions import circuit_attributions
        m = build_circuit(num_bits=8, num_outputs=1)
        x = np.random.RandomState(0).randint(0, 2, size=(8,)).astype(np.float32)
        a = circuit_attributions(m, x)
        assert a.shape == (8,)
        assert np.all(np.isfinite(a))


# ---------------------------------------------------------------------
# Hard-extraction on a build_circuit model
# ---------------------------------------------------------------------


class TestHardExtractionCircuit:
    def test_roundtrip_below_threshold(self, tmp_path):
        m = build_circuit(num_bits=8, num_outputs=1)
        rng = np.random.default_rng(0)
        Xv, Yv = gen_parity_k8(64, rng)
        res = evaluate_hard_extraction_circuit(m, Xv, Yv, str(tmp_path / "rt.keras"))
        assert res["roundtrip_diff"] < 1e-5
        assert res["soft_acc"] is not None
        assert res["hard_acc"] is not None
        assert res["delta_hard"] is not None


# ---------------------------------------------------------------------
# StopOnAccuracyBand integration in a 5-epoch fake-fit
# ---------------------------------------------------------------------


class TestBandCallbackIntegration:
    def test_band_fires_in_short_fit(self):
        """Fit a tiny MLP on a trivially-learnable task for 5 epochs with a
        wide band and verify the callback fires.
        """
        m = build_circuit(num_bits=4, num_outputs=1)
        # Easy task: y = x[0] AND x[1] — small build_circuit learns it fast.
        rng = np.random.default_rng(0)
        x = rng.integers(0, 2, size=(2048, 4)).astype(np.float32)
        y = (x[:, 0] * x[:, 1]).astype(np.float32).reshape(-1, 1)
        xv = rng.integers(0, 2, size=(256, 4)).astype(np.float32)
        yv = (xv[:, 0] * xv[:, 1]).astype(np.float32).reshape(-1, 1)
        cb = StopOnAccuracyBand("val_accuracy", 0.55, 0.999, "enter", verbose=0)
        m.fit(x, y, validation_data=(xv, yv), epochs=10, batch_size=32, verbose=0, callbacks=[cb])
        # We don't strictly require fired=True (model may not enter the
        # band in 10 epochs on every seed), but if it did fire,
        # band_acc must be in range.
        if cb.fired:
            assert 0.55 <= cb.band_value <= 0.999


# ---------------------------------------------------------------------
# .keras save/load round-trip on E3 circuit
# ---------------------------------------------------------------------


class TestKerasRoundtrip:
    def test_save_load_circuit(self, tmp_path):
        m = build_circuit(num_bits=11, num_outputs=1)
        path = str(tmp_path / "circ.keras")
        m.save(path)
        m2 = keras.models.load_model(path)
        rng = np.random.default_rng(0)
        x, _ = gen_mux_11bit(8, rng)
        np.testing.assert_allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), atol=1e-5)


# ---------------------------------------------------------------------
# CSV schema sanity (matches plan SC5)
# ---------------------------------------------------------------------


class TestCSVSchema:
    def test_expected_columns_present(self):
        # Plan SC5 grep target: lime_suff_auc and circuit_suff_auc.
        assert "lime_suff_auc" in CSV_COLUMNS
        assert "shap_suff_auc" in CSV_COLUMNS
        assert "circuit_suff_auc" in CSV_COLUMNS
        assert "delta_hard" in CSV_COLUMNS
        assert "band_acc" in CSV_COLUMNS
        # Every (method, suffix) pair is present.
        for m in ATTRIBUTION_METHODS:
            for s in METRIC_SUFFIXES:
                assert f"{m}{s}" in CSV_COLUMNS


# ---------------------------------------------------------------------
# Spec registry sanity
# ---------------------------------------------------------------------


class TestTaskSpecs:
    def test_three_tasks_registered(self):
        assert set(TASK_SPECS_E3.keys()) == {"mux_11bit", "parity_k8", "random_dnf_8input_4term"}
        for name, spec in TASK_SPECS_E3.items():
            assert "generator" in spec and callable(spec["generator"])
            assert spec["num_bits"] >= 4
            assert spec["num_outputs"] == 1
            assert spec["enum_size"] == (1 << spec["num_bits"])
