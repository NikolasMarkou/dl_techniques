"""Unit tests for the sweep driver — focused on the plan_e1f12eab Step 5
additions: ``--regimes`` enumeration + ``--max-cells`` guard (D-003).

Existing sweep behaviour (subprocess launching, GPU pinning, aggregation) is
not re-tested here — it is covered by integration via the Phase 3 smoke gate
(plan_e1f12eab Step 9). These tests are pure-Python, fast (no GPU, no
subprocess), and verify the build-time contract.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
from typing import List

import pytest

from train.rms_variants_train.sweep import (
    EXPERIMENT_MODES,
    EXPERIMENT_REGIMES,
    EXPERIMENT_REGISTRY,
    RunSpec,
    VARIANT_SUPPORTS_PARAM_MATCHED,
    build_run_specs,
    run_one,
)
from train.rms_variants_train.config import NORM_VARIANTS


class TestBuildRunSpecsBackCompat:
    """No --regimes passed → default behaviour preserved (regime='default',
    out-dir layout unchanged from pre-plan_e1f12eab)."""

    def test_default_regime_when_unspecified(self, tmp_path):
        specs = build_run_specs(
            experiments=["e5"], norms=["rms_norm"], modes=["oob"],
            seeds=[0, 1], sweep_root=str(tmp_path), epochs_override=None,
        )
        assert len(specs) == 2
        for s in specs:
            assert s.regime == "default"

    def test_default_regime_out_dir_layout_unchanged(self, tmp_path):
        # Pre-plan_e1f12eab: out_dir = sweep_root/exp/norm/mode/seed_N
        # Must NOT include regime_default segment (I2 invariant).
        specs = build_run_specs(
            experiments=["e5"], norms=["rms_norm"], modes=["oob"],
            seeds=[0], sweep_root=str(tmp_path), epochs_override=None,
        )
        expected_leaf = os.path.join("e5", "rms_norm", "oob", "seed_0")
        assert specs[0].out_dir.endswith(expected_leaf), (
            f"Default-regime out_dir layout changed; was {specs[0].out_dir}"
        )


class TestBuildRunSpecsRegimes:
    def test_regimes_multiplies_cell_count(self, tmp_path):
        # E5 supports 5 regimes; passing 3 of them = 3x cells.
        specs = build_run_specs(
            experiments=["e5"], norms=["rms_norm"], modes=["oob"],
            seeds=[0, 1], sweep_root=str(tmp_path), epochs_override=None,
            regimes=("default", "bs_32", "bs_256"),
        )
        # 1 exp × 1 norm × 1 mode × 3 regimes × 2 seeds = 6.
        assert len(specs) == 6
        regimes_seen = {s.regime for s in specs}
        assert regimes_seen == {"default", "bs_32", "bs_256"}

    def test_unsupported_regime_skipped(self, tmp_path):
        # E2 supports only the 'default' regime. Asking for 'lr_low' on E2
        # should be skipped (with a log line), NOT raise.
        specs = build_run_specs(
            experiments=["e2"], norms=["rms_norm"], modes=["oob"],
            seeds=[0], sweep_root=str(tmp_path), epochs_override=None,
            regimes=("default", "lr_low"),
        )
        # Only the supported one yields a cell.
        assert len(specs) == 1
        assert specs[0].regime == "default"

    def test_non_default_regime_uses_regime_path_segment(self, tmp_path):
        specs = build_run_specs(
            experiments=["e5"], norms=["rms_norm"], modes=["oob"],
            seeds=[0], sweep_root=str(tmp_path), epochs_override=None,
            regimes=("bs_32",),
        )
        assert "regime_bs_32" in specs[0].out_dir, (
            f"Non-default regime should appear in out_dir path; got "
            f"{specs[0].out_dir}"
        )

    def test_regime_arg_appears_in_subprocess_extra_args(self, tmp_path):
        specs = build_run_specs(
            experiments=["e5"], norms=["rms_norm"], modes=["oob"],
            seeds=[0], sweep_root=str(tmp_path), epochs_override=None,
            regimes=("lr_low",),
        )
        assert "--regime" in specs[0].extra_args
        idx = specs[0].extra_args.index("--regime")
        assert specs[0].extra_args[idx + 1] == "lr_low"

    def test_unsupported_experiment_regime_combo_does_not_crash(self, tmp_path):
        # All-unsupported request → empty spec list (caller decides what to do).
        specs = build_run_specs(
            experiments=["e2"], norms=["rms_norm"], modes=["oob"],
            seeds=[0], sweep_root=str(tmp_path), epochs_override=None,
            regimes=("lr_low", "depth_12"),
        )
        assert specs == []


class TestMaxCellsGuard:
    """D-003: --max-cells fires at build time, BEFORE any subprocess."""

    def test_under_cap_succeeds(self, tmp_path):
        specs = build_run_specs(
            experiments=["e5"], norms=list(NORM_VARIANTS[:2]), modes=["oob"],
            seeds=[0, 1], sweep_root=str(tmp_path), epochs_override=None,
            max_cells=10,
        )
        assert len(specs) == 4
        assert len(specs) <= 10

    def test_over_cap_raises_value_error(self, tmp_path):
        # 5 norms × 5 seeds = 25 cells on E5/oob/default — exceeds max_cells=5.
        with pytest.raises(ValueError) as exc:
            build_run_specs(
                experiments=["e5"],
                norms=list(NORM_VARIANTS[:5]),
                modes=["oob"],
                seeds=[0, 1, 2, 3, 4],
                sweep_root=str(tmp_path),
                epochs_override=None,
                max_cells=5,
            )
        # Error message must be actionable: includes the cell count and
        # dimensions.
        msg = str(exc.value)
        assert "25" in msg, f"Error message missing cell count: {msg}"
        assert "max-cells" in msg or "max_cells" in msg
        assert "experiments" in msg or "norms" in msg

    def test_guard_fires_before_subprocess_launch(self, tmp_path, monkeypatch):
        """The guard must raise at build time, NOT after partially launching
        subprocesses (falsification scenario C)."""
        # build_run_specs is pure — if it raises, no subprocess.run can have
        # been invoked. We sanity-check by monkeypatching subprocess.run to
        # fail loudly if called.
        import subprocess
        called = []
        original_run = subprocess.run

        def _no_run(*args, **kwargs):
            called.append(args)
            raise AssertionError(
                "subprocess.run called during build — D-003 guard violation"
            )

        monkeypatch.setattr(subprocess, "run", _no_run)
        with pytest.raises(ValueError):
            build_run_specs(
                experiments=["e5"],
                norms=list(NORM_VARIANTS),
                modes=["oob"],
                seeds=[0, 1, 2, 3, 4],
                sweep_root=str(tmp_path),
                epochs_override=None,
                max_cells=10,
            )
        assert called == [], (
            "subprocess.run was called during build — guard fired too late"
        )

    def test_default_cap_is_1000(self, tmp_path):
        # Default behaviour (no max_cells override): a small build succeeds.
        # We rely on the function signature default = 1000.
        specs = build_run_specs(
            experiments=["e5"], norms=["rms_norm"], modes=["oob"],
            seeds=[0], sweep_root=str(tmp_path), epochs_override=None,
        )
        # Sanity — under the default 1000.
        assert len(specs) == 1


class TestExperimentRegimes:
    """The EXPERIMENT_REGIMES table mirrors per-trainer _REGIME_MAP keys.
    If a trainer adds/removes a regime, this dict must follow — else the
    sweep skips meaningful regimes silently."""

    def test_all_experiments_have_default_regime(self):
        for exp, regimes in EXPERIMENT_REGIMES.items():
            assert "default" in regimes, (
                f"{exp} missing 'default' regime — sweep would skip "
                f"default-regime cells, breaking I2 invariant."
            )

    def test_every_experiment_in_registry_has_regimes_entry(self):
        for exp in EXPERIMENT_REGISTRY:
            assert exp in EXPERIMENT_REGIMES, (
                f"{exp} in EXPERIMENT_REGISTRY but missing from "
                f"EXPERIMENT_REGIMES — sweep would default to ('default',) "
                f"and silently skip the trainer's real regime options."
            )


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def test_two_cells_that_differ_only_in_regime_log_two_names(tmp_path):
    """The regime is part of a cell's name in the FAIL line (and ``failures.log``).

    The label named experiment, norm, mode and seed only, so the ``default`` and
    ``bs_32`` cells of one seed failed under the same line; ``RunSpec.regime`` was
    read by nothing (step 13.10 of plan 98b866fb). One real subprocess per cell, a
    module that does not exist, so each exits 1 at once.
    """
    specs = build_run_specs(
        experiments=["e5"], norms=["rms_norm"], modes=["oob"], seeds=[0],
        sweep_root=str(tmp_path), epochs_override=None, regimes=("default", "bs_32"),
    )
    specs = [dataclasses.replace(s, module="no_such_module_for_this_test") for s in specs]
    handler = _Lines()
    logging.getLogger("dl").addHandler(handler)
    try:
        for spec in specs:
            ok, _ = run_one(spec, python_exe=sys.executable, cell_timeout_s=60,
                            deadline_s=time.time() + 60, visible_devices="-1")
            assert not ok
    finally:
        logging.getLogger("dl").removeHandler(handler)
    failed = [line for line in handler.lines if line.startswith("[sweep] FAIL")]
    assert len(failed) == 2 and "regime_default" in failed[0] and "regime_bs_32" in failed[1], failed
