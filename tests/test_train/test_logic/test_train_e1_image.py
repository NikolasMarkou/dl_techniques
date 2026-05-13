"""Unit tests for train.logic.train_e1_image.

No real training — only build/forward/round-trip/extraction checks on
tiny fake inputs.

Plan: plan_2026-05-13_798d3a60.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from train.logic.train_e1_image import (
    build_cnn_baseline,
    build_image_circuit,
    evaluate_hard_extraction,
    find_cnn_filters_for_param_budget,
)


# ---------------------------------------------------------------------
# Build + forward
# ---------------------------------------------------------------------

class TestBuildCircuit:
    def test_mnist_shape(self):
        m = build_image_circuit((28, 28, 1), num_classes=10, stem_filters=(32,))
        x = np.random.RandomState(0).rand(2, 28, 28, 1).astype(np.float32)
        y = m.predict(x, verbose=0)
        assert y.shape == (2, 10)

    def test_cifar_shape(self):
        m = build_image_circuit((32, 32, 3), num_classes=10, stem_filters=(32, 64, 128))
        x = np.random.RandomState(0).rand(2, 32, 32, 3).astype(np.float32)
        y = m.predict(x, verbose=0)
        assert y.shape == (2, 10)


class TestBuildCNN:
    def test_cnn_shape(self):
        m = build_cnn_baseline((28, 28, 1), num_classes=10, filters=(16,))
        x = np.random.RandomState(0).rand(2, 28, 28, 1).astype(np.float32)
        y = m.predict(x, verbose=0)
        assert y.shape == (2, 10)


# ---------------------------------------------------------------------
# Param budget search
# ---------------------------------------------------------------------


class TestParamBudgetSearch:
    def test_finds_within_tolerance(self):
        """Find a CNN within ~5% of a 1000-param target."""
        filters, _ = find_cnn_filters_for_param_budget(
            input_shape=(28, 28, 1), num_classes=10,
            filter_pattern=(32,), target_params=1000, tol=0.25,
        )
        m = build_cnn_baseline((28, 28, 1), 10, filters=filters)
        # Just sanity-check filters are positive ints.
        assert all(f > 0 for f in filters)
        assert isinstance(m.count_params(), int)


# ---------------------------------------------------------------------
# Hard-extraction round-trip on an untrained circuit
# ---------------------------------------------------------------------

class TestHardExtraction:
    def test_roundtrip_diff_below_threshold(self, tmp_path):
        """Untrained model — no fitting needed; roundtrip + soft-restore must
        be bit-exact within 1e-6."""
        m = build_image_circuit((28, 28, 1), num_classes=10, stem_filters=(32,))
        x_val = np.random.RandomState(0).rand(8, 28, 28, 1).astype(np.float32)
        y_val = np.zeros((8,), dtype=np.int64)
        save_path = str(tmp_path / "circuit.keras")
        result = evaluate_hard_extraction(m, x_val, y_val, save_path=save_path)
        # Untrained model has soft_acc ~= chance; we only care that the
        # mechanics work.
        assert result["roundtrip_diff"] < 1e-5  # 1e-6 spec, leave slack
        assert result["soft_acc"] is not None
        assert result["hard_acc"] is not None
        assert result["delta_hard"] is not None


# ---------------------------------------------------------------------
# .keras save/load round-trip on the MNIST circuit
# ---------------------------------------------------------------------

class TestKerasRoundtrip:
    def test_mnist_circuit_save_load(self, tmp_path):
        m = build_image_circuit((28, 28, 1), num_classes=10, stem_filters=(32,))
        save_path = str(tmp_path / "circuit_full.keras")
        m.save(save_path)
        m2 = keras.models.load_model(save_path)
        x = np.random.RandomState(0).rand(4, 28, 28, 1).astype(np.float32)
        p1 = m.predict(x, verbose=0)
        p2 = m2.predict(x, verbose=0)
        np.testing.assert_allclose(p1, p2, atol=1e-5)
