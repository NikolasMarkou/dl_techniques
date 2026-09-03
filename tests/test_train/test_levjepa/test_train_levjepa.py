"""Regression + REAL smoke-run tests for ``train.levjepa.train_levjepa``.

Covers:
- ``TrainingConfig.__post_init__`` validation.
- ``--smoke`` preset overrides defaults but not user-provided flags.
- ``--help`` parses without allocating anything.
- A REAL end-to-end smoke training run (``main()`` -> ``model.fit()``),
  asserting it completes and produces a finite loss -- the plan's Success
  Criterion 10 / Pre-Mortem #3 STOP-IF trigger. Not an import-only check.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from train.levjepa.train_levjepa import (
    TrainingConfig,
    _build_config,
    _SMOKE_OVERRIDES,
    parse_arguments,
    main,
)


class TestTrainingConfigValidation:
    def test_default_config_is_valid(self):
        TrainingConfig()  # should not raise

    def test_batch_size_below_two_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=1)

    def test_num_frames_one_raises(self):
        with pytest.raises(ValueError, match="num_frames"):
            TrainingConfig(num_frames=1)

    def test_num_frames_not_divisible_by_tubelet_raises(self):
        with pytest.raises(ValueError, match="tubelet_size"):
            TrainingConfig(num_frames=5, tubelet_size=2)

    def test_zero_local_crops_raises(self):
        with pytest.raises(ValueError, match="local_crops_number"):
            TrainingConfig(local_crops_number=0)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="variant"):
            TrainingConfig(variant="not_a_variant")

    def test_img_size_not_divisible_by_patch_size_raises(self):
        with pytest.raises(ValueError, match="patch_size"):
            TrainingConfig(variant="vit_tiny", img_size=17)


class TestParseArguments:
    def test_help_exits_zero_with_usage_line(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out

    def test_smoke_overrides_defaults(self):
        args = parse_arguments(["--smoke"])
        for key, value in _SMOKE_OVERRIDES.items():
            assert getattr(args, key) == value

    def test_smoke_does_not_override_explicit_flags(self):
        args = parse_arguments(["--smoke", "--batch-size", "6"])
        assert args.batch_size == 6
        # Untouched smoke overrides still apply.
        assert args.img_size == _SMOKE_OVERRIDES["img_size"]

    def test_build_config_from_smoke_args(self):
        args = parse_arguments(["--smoke"])
        config = _build_config(args)
        assert config.dataset == "synthetic_drone"
        assert config.batch_size == 2


@pytest.mark.integration
class TestSmokeTrainingRun:
    """THE REAL INTEGRATION TEST -- Success Criterion 10 / Pre-Mortem #3."""

    def test_smoke_run_completes_with_finite_loss(self, tmp_path):
        argv = [
            "--smoke",
            "--dataset", "synthetic_drone",
            "--output-dir", str(tmp_path / "levjepa_smoke_run"),
        ]
        history = main(argv)

        assert "loss" in history.history
        final_loss = history.history["loss"][-1]
        assert np.isfinite(final_loss), f"Smoke run loss was not finite: {final_loss}"

        # CSVLogger should have written a real log.
        log_path = tmp_path / "levjepa_smoke_run" / "training_log.csv"
        assert log_path.exists()
