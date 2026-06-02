"""Backwards-compatible shim for the RMSNorm-variants sweep seeder.

The canonical ``set_seeds`` now lives in :mod:`train.common.seed`
(plan_2026-06-02_30721a0f, F3). This module re-exports it verbatim so the
existing ``rms_variants_train/experiments/`` callers keep working.

# DECISION plan_2026-06-02_30721a0f/D-002: do NOT re-add the RNG body here.
# set_seeds was promoted to train.common.seed to deduplicate the 4-call form
# across trainers; this file stays a pure re-export. See decisions.md D-002.
"""
from __future__ import annotations

from train.common.seed import set_seeds

__all__ = ["set_seeds"]
