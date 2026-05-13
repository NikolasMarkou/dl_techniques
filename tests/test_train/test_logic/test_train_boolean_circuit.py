"""Unit tests for train.logic.train_boolean_circuit.

These tests do not train to convergence — they only exercise the data
generators, the model factory, a single fit step, and the save/load round
trip. The real convergence check happens via the smoke run in S5.
"""

import os
import tempfile
from typing import Any

import keras
import numpy as np
import pytest

from train.logic.train_boolean_circuit import (
    TASK_REGISTRY,
    build_model,
    exact_accuracy_for_parity,
    gen_majority,
    gen_parity,
    gen_random_dnf,
    round_trip_check,
)


# ---------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------

class TestGenerators:
    def test_gen_parity_shape_and_label(self):
        rng = np.random.default_rng(0)
        x, y = gen_parity(num_samples=32, num_bits=4, rng=rng)
        assert x.shape == (32, 4) and y.shape == (32, 1)
        assert x.dtype == np.float32 and y.dtype == np.float32
        # parity == sum % 2 == 1
        expected = ((x.sum(axis=1) % 2) == 1).astype(np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(y, expected)

    def test_gen_majority_shape_and_label(self):
        rng = np.random.default_rng(0)
        x, y = gen_majority(num_samples=32, num_bits=5, rng=rng)
        assert x.shape == (32, 5) and y.shape == (32, 1)
        expected = (x.sum(axis=1) >= 2.5).astype(np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(y, expected)

    def test_gen_random_dnf_deterministic_under_seed(self):
        x1, y1 = gen_random_dnf(num_samples=64, num_bits=4, rng=np.random.default_rng(7))
        x2, y2 = gen_random_dnf(num_samples=64, num_bits=4, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_gen_random_dnf_label_values_are_binary(self):
        x, y = gen_random_dnf(num_samples=128, num_bits=4, rng=np.random.default_rng(11))
        assert set(np.unique(y).tolist()).issubset({0.0, 1.0})

    def test_task_registry_keys(self):
        assert set(TASK_REGISTRY) == {"parity", "majority", "random_dnf"}


# ---------------------------------------------------------------------
# Model + training
# ---------------------------------------------------------------------

class TestModel:
    def test_build_model_compiles_and_predicts(self):
        m = build_model(num_bits=4, channels=8, circuit_depth=1)
        x = np.random.randint(0, 2, (4, 4)).astype("float32")
        y_pred = m.predict(x, verbose=0)
        assert y_pred.shape == (4, 1)
        assert (y_pred >= 0).all() and (y_pred <= 1).all()

    def test_one_epoch_smoke_runs(self):
        """End-to-end .fit() runs without error on tiny data."""
        rng = np.random.default_rng(0)
        x, y = gen_majority(num_samples=64, num_bits=4, rng=rng)
        m = build_model(num_bits=4, channels=8, circuit_depth=1)
        hist = m.fit(x, y, epochs=1, batch_size=16, verbose=0)
        assert "loss" in hist.history and np.isfinite(hist.history["loss"][-1])

    def test_save_load_round_trip(self):
        rng = np.random.default_rng(0)
        x, _ = gen_majority(num_samples=32, num_bits=4, rng=rng)
        m = build_model(num_bits=4, channels=8, circuit_depth=1,
                        gate_entropy_coef=0.05, diversity_coef=0.05)
        _ = m.predict(x, verbose=0)  # build all sublayers
        with tempfile.TemporaryDirectory() as td:
            diff = round_trip_check(m, x, td)
        assert diff < 1e-6, f"round-trip diff too large: {diff}"

    def test_to_symbolic_after_one_epoch(self):
        rng = np.random.default_rng(0)
        x, y = gen_majority(num_samples=64, num_bits=4, rng=rng)
        m = build_model(num_bits=4, channels=8, circuit_depth=2)
        m.fit(x, y, epochs=1, batch_size=16, verbose=0)
        nc = m.get_layer("neural_circuit")
        s = nc.to_symbolic(top_k=2)
        assert "depth 0:" in s
        assert "combination:" in s


class TestExactAccuracy:
    def test_exact_parity_accuracy_against_constant_predictor(self):
        """Constant-0 predictor has exactly 0.5 accuracy on K=4 parity
        (8 of 16 inputs have odd parity). This verifies the eval helper."""
        m = build_model(num_bits=4, channels=4, circuit_depth=1)
        # Override head bias to -1000 so sigmoid output is ~0 always.
        head = m.get_layer("head")
        w, b = head.get_weights()
        head.set_weights([np.zeros_like(w), np.full_like(b, -1000.0)])
        acc = exact_accuracy_for_parity(m, num_bits=4)
        assert acc == 0.5
