"""Sanity check that the stats re-export from train.logic works end-to-end.

Heavy testing of the stats math lives in
``tests/test_train/test_logic/test_multiseed_stats.py``. Here we only confirm
that imports resolve and a happy-path call to each function returns sane
values when invoked through the re-export module.
"""
from __future__ import annotations

import numpy as np
import pytest

from train.rms_variants_train import stats as st


def test_reexport_surface() -> None:
    assert callable(st.mean_std)
    assert callable(st.bootstrap_ci)
    assert callable(st.paired_permutation_test)
    assert callable(st.format_mean_std)


def test_mean_std_happy_path() -> None:
    m, s = st.mean_std([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m == pytest.approx(3.0)
    assert s == pytest.approx(np.sqrt(2.5), rel=1e-6)


def test_bootstrap_ci_deterministic() -> None:
    rng = np.random.default_rng(20260515)
    lo, hi = st.bootstrap_ci([0.7, 0.71, 0.72, 0.73, 0.74], rng=rng, n_boot=500)
    assert 0.7 <= lo <= hi <= 0.74


def test_paired_permutation_happy_path() -> None:
    rng = np.random.default_rng(20260515)
    a = np.array([0.80, 0.81, 0.82, 0.83, 0.84])
    b = np.array([0.70, 0.71, 0.72, 0.73, 0.74])
    diff, p = st.paired_permutation_test(a, b, rng=rng, n_perm=500)
    assert diff > 0
    assert 0.0 <= p <= 1.0


def test_format_mean_std() -> None:
    s = st.format_mean_std(0.7006, 0.0123)
    assert "0.7006" in s
    assert "0.0123" in s
    assert "±" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
