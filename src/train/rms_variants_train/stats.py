"""Thin re-export of statistical helpers from ``train.logic.multiseed_stats``.

Per plan_2026-05-14_3764496e D-001: do NOT duplicate stats math here. The
``train.logic`` module is pure-function, NaN-tolerant, degenerate-case-safe
(LESSONS L78), and already pinned by 30 unit tests. We import it directly so
the report writer in this package stays a thin caller.
"""
from __future__ import annotations

from train.logic.multiseed_stats import (
    bootstrap_ci,
    format_mean_std,
    mean_std,
    paired_permutation_test,
)

__all__ = [
    "bootstrap_ci",
    "format_mean_std",
    "mean_std",
    "paired_permutation_test",
]
