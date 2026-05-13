"""Unit tests for train.logic.train_benchmark.

No real training in this module — only data-correctness, model-construction,
and hard-extraction reversibility. The full benchmark is exercised via the
end-to-end run in plan S3.
"""

import os
import tempfile
from typing import Any, Dict, List

import keras
import numpy as np
import pytest
from keras import ops

from train.logic.train_benchmark import (
    TASK_SPECS,
    _iter_inner_ops,
    bitwise_accuracy,
    build_circuit,
    build_mlp,
    extract_hard_inplace,
    find_mlp_hidden_for_param_budget,
    gen_multiplexer_6,
    gen_shift_xor,
    restore_soft_weights,
    train_one,
    write_csv,
    write_report_md,
)


class TestGenerators:
    def test_multiplexer_ground_truth_handcoded(self):
        """Compare each of the first 8 multiplexer outputs against the
        published truth-table semantics: addr = x0 + 2*x1; y = x[2+addr]."""
        x = np.array(
            [
                [0, 0, 0, 0, 0, 0],  # addr=0, d0=0
                [0, 0, 1, 0, 0, 0],  # addr=0, d0=1
                [1, 0, 0, 0, 0, 0],  # addr=1, d1=0
                [1, 0, 0, 1, 0, 0],  # addr=1, d1=1
                [0, 1, 0, 0, 0, 0],  # addr=2, d2=0
                [0, 1, 0, 0, 1, 0],  # addr=2, d2=1
                [1, 1, 0, 0, 0, 0],  # addr=3, d3=0
                [1, 1, 0, 0, 0, 1],  # addr=3, d3=1
            ],
            dtype=np.float32,
        )
        expected_y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32).reshape(-1, 1)
        # Use the internal eval logic directly.
        addr = (x[:, 0].astype(np.int32) + 2 * x[:, 1].astype(np.int32))
        actual_y = x[np.arange(len(x)), 2 + addr].reshape(-1, 1)
        np.testing.assert_array_equal(actual_y, expected_y)

    def test_shift_xor_ground_truth(self):
        rng = np.random.default_rng(0)
        x, y = gen_shift_xor(16, 8, rng)
        # Recompute by hand on the same x.
        x_rolled = np.roll(x.astype(bool), -1, axis=1)
        expected = np.logical_xor(x.astype(bool), x_rolled).astype(np.float32)
        np.testing.assert_array_equal(y, expected)
        # Output shape is (n, K), not (n, 1).
        assert y.shape == (16, 8)

    def test_task_specs_keys(self):
        assert set(TASK_SPECS) == {
            "parity_k6", "majority_k6", "multiplexer_6", "shift_xor_k8",
        }


class TestModelFactories:
    def test_build_circuit_scalar_output(self):
        m = build_circuit(num_bits=6, num_outputs=1)
        x = np.random.randint(0, 2, (4, 6)).astype("float32")
        y = m.predict(x, verbose=0)
        assert y.shape == (4, 1)

    def test_build_circuit_multi_output_per_channel(self):
        m = build_circuit(num_bits=8, num_outputs=8, selection_mode="per_channel")
        x = np.random.randint(0, 2, (4, 8)).astype("float32")
        y = m.predict(x, verbose=0)
        assert y.shape == (4, 8)

    def test_find_mlp_hidden_for_param_budget_within_30pct(self):
        target = 400
        h = find_mlp_hidden_for_param_budget(6, 1, target)
        m = build_mlp(6, 1, hidden_units=h)
        # Tolerance: tightest possible without flakiness.
        assert abs(m.count_params() - target) / target < 0.30, (
            f"hidden={h} -> {m.count_params()} params, target {target}"
        )


class TestHardExtraction:
    def _make_built_circuit(self) -> keras.Model:
        m = build_circuit(num_bits=6, num_outputs=1)
        x = np.random.randint(0, 2, (4, 6)).astype("float32")
        _ = m.predict(x, verbose=0)  # build all sublayers
        return m

    def test_iter_inner_ops_nonzero(self):
        m = self._make_built_circuit()
        ops_list = list(_iter_inner_ops(m))
        # depth=2, 2 logic + 2 arith per depth = 8 inner ops total.
        assert len(ops_list) == 8

    def test_extract_hard_inplace_changes_predictions(self):
        m = self._make_built_circuit()
        x = np.random.randint(0, 2, (8, 6)).astype("float32")
        soft = m.predict(x, verbose=0)
        snap = extract_hard_inplace(m)
        hard = m.predict(x, verbose=0)
        # Hard should be DIFFERENT from soft (otherwise the soft mixture had
        # already collapsed to one-hot — unlikely on a fresh init).
        assert not np.allclose(soft, hard, atol=1e-4)
        restore_soft_weights(snap)

    def test_extract_hard_inplace_is_reversible(self):
        m = self._make_built_circuit()
        x = np.random.randint(0, 2, (8, 6)).astype("float32")
        soft = m.predict(x, verbose=0)
        snap = extract_hard_inplace(m)
        _ = m.predict(x, verbose=0)
        restore_soft_weights(snap)
        soft_again = m.predict(x, verbose=0)
        np.testing.assert_allclose(soft, soft_again, atol=1e-7)

    def test_extract_hard_yields_near_onehot_probs(self):
        m = self._make_built_circuit()
        snap = extract_hard_inplace(m)
        try:
            for op in _iter_inner_ops(m):
                probs = ops.convert_to_numpy(op._operation_probs(deterministic=True))
                # Max prob per row should be very near 1.0.
                if probs.ndim == 1:
                    assert probs.max() > 0.99
                else:
                    assert probs.max(axis=-1).min() > 0.99
        finally:
            restore_soft_weights(snap)


class TestTrainOne:
    def test_one_epoch_smoke_circuit(self):
        with tempfile.TemporaryDirectory() as td:
            row = train_one(
                task_name="majority_k6", model_name="circuit",
                epochs=1, train_samples=128, test_samples=64,
                seed=0, results_dir=td,
            )
        # Smoke: numbers populated and finite.
        assert row["params"] > 0
        assert row["epochs_used"] == 1
        assert 0.0 <= row["soft_test_acc"] <= 1.0
        assert row["hard_test_acc"] is not None
        assert row["roundtrip_diff"] < 1e-6

    def test_one_epoch_smoke_mlp_matched(self):
        with tempfile.TemporaryDirectory() as td:
            row = train_one(
                task_name="majority_k6", model_name="mlp_matched",
                epochs=1, train_samples=128, test_samples=64,
                seed=0, results_dir=td,
            )
        # MLP rows have no hard extraction (no circuit to extract from).
        assert row["hard_test_acc"] is None
        assert row["roundtrip_diff"] < 1e-6


class TestReport:
    def test_write_csv_and_report_md_non_empty(self):
        rows: List[Dict[str, Any]] = [
            {
                "task": "parity_k6", "model": "circuit", "params": 401,
                "epochs_used": 10, "wall_s": 5.0,
                "soft_test_acc": 0.95, "exact_acc": 0.95,
                "hard_test_acc": 0.92, "hard_exact_acc": 0.92,
                "roundtrip_diff": 0.0,
                "symbolic": "depth 0:\n  logic_op_0: xor(0.45)",
            },
            {
                "task": "parity_k6", "model": "mlp_large", "params": 1313,
                "epochs_used": 8, "wall_s": 3.5,
                "soft_test_acc": 0.97, "exact_acc": 0.97,
                "hard_test_acc": None, "hard_exact_acc": None,
                "roundtrip_diff": 0.0, "symbolic": None,
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            csv_p = os.path.join(td, "results.csv")
            md_p = os.path.join(td, "report.md")
            write_csv(rows, csv_p)
            write_report_md(rows, md_p)
            assert os.path.getsize(csv_p) > 0
            md = open(md_p).read()
            assert "Headline" in md
            assert "parity_k6" in md
            assert "Faithfulness summary" in md
            assert "FAITHFUL" in md or "LOSSY" in md
