"""Smoke tests for ``train.rms_variants_train.norm_overhead_bench`` (step 2).

The bench is a standalone telemetry script; these tests confirm import,
canonical-row emission for a single norm under tiny iter counts, and the
end-to-end ``write_csv`` round-trip. Heavy timing assertions are deliberately
out of scope (timings are observational, not enforced).

Plan: ``plans/plan_2026-05-18_6776f8ba`` (step 2 gate; D-001 anchor present).
"""
from __future__ import annotations

import csv
import os
import tempfile

import pytest


def test_import_module() -> None:
    """The bench module must import cleanly without side effects."""
    from train.rms_variants_train import norm_overhead_bench as nob

    assert hasattr(nob, "run_bench")
    assert hasattr(nob, "write_csv")
    assert hasattr(nob, "main")


def test_decision_anchor_present() -> None:
    """SC15 spot-check: D-001 anchor must live at the file header."""
    from train.rms_variants_train import norm_overhead_bench as nob

    src_path = nob.__file__
    with open(src_path, "r") as fh:
        head = fh.read(2000)
    assert "DECISION plan_2026-05-18_6776f8ba/D-001" in head


def test_single_norm_smoke() -> None:
    """Running the bench for a single norm produces a canonical row."""
    from train.rms_variants_train.norm_overhead_bench import run_bench

    rows = run_bench(
        ("rms_norm",),
        batch=8,
        features=16,
        iters=2,
        warmup=1,
        include_fp16=False,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["norm"] == "rms_norm"
    assert row["params"] > 0
    assert row["mean_step_ms_fp32"] > 0.0
    # fp16 disabled -> NaN sentinel
    import math
    assert math.isnan(row["mean_step_ms_fp16"])


def test_write_csv_roundtrip() -> None:
    """``write_csv`` emits the canonical 6-column header + one row per dict."""
    from train.rms_variants_train.norm_overhead_bench import run_bench, write_csv

    rows = run_bench(
        ("rms_norm",),
        batch=8,
        features=16,
        iters=2,
        warmup=1,
        include_fp16=False,
    )
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "overhead.csv")
        write_csv(rows, out)
        assert os.path.exists(out)
        with open(out, "r") as fh:
            reader = csv.DictReader(fh)
            data = list(reader)
        assert len(data) == 1
        assert set(reader.fieldnames or []) == {
            "norm",
            "params",
            "mean_step_ms_fp32",
            "mean_step_ms_fp16",
            "peak_mem_mb_fp32",
            "peak_mem_mb_fp16",
        }
        assert data[0]["norm"] == "rms_norm"


if __name__ == "__main__":
    pytest.main([__file__, "-vvv"])
