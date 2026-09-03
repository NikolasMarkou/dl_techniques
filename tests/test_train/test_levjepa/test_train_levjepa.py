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
    _build_model,
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

    def test_unknown_attn_mode_raises(self):
        with pytest.raises(ValueError, match="attn_mode"):
            TrainingConfig(attn_mode="not_a_mode")

    def test_token_drop_rate_out_of_range_raises(self):
        with pytest.raises(ValueError, match="token_drop_rate"):
            TrainingConfig(token_drop_rate=1.0)
        with pytest.raises(ValueError, match="token_drop_rate"):
            TrainingConfig(token_drop_rate=-0.1)

    def test_default_config_exercises_the_risky_attention_config(self):
        """CRITICAL-2 regression guard (D-023): the DEFAULT config -- the one
        every real run gets unless a flag overrides it -- must already be
        the risky config (block_causal + RoPE + nonzero token drop), not the
        encoder's own degenerate constructor defaults."""
        config = TrainingConfig()
        assert config.attn_mode == "block_causal"
        assert config.use_rope is True
        assert config.token_drop_rate > 0.0


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

    def test_attn_mode_use_rope_token_drop_flags_parse(self):
        args = parse_arguments(
            ["--attn-mode", "full", "--no-use-rope", "--token-drop-rate", "0.1"]
        )
        assert args.attn_mode == "full"
        assert args.use_rope is False
        assert args.token_drop_rate == pytest.approx(0.1)

    def test_attn_mode_use_rope_token_drop_default_to_the_risky_config(self):
        args = parse_arguments([])
        assert args.attn_mode == "block_causal"
        assert args.use_rope is True
        assert args.token_drop_rate > 0.0

    def test_smoke_pins_the_risky_config_explicitly(self):
        args = parse_arguments(["--smoke"])
        assert args.attn_mode == "block_causal"
        assert args.use_rope is True
        assert args.token_drop_rate > 0.0

    def test_explicit_flag_still_wins_under_smoke(self):
        args = parse_arguments(["--smoke", "--attn-mode", "full"])
        assert args.attn_mode == "full"


class TestSmokeConfigWiring:
    """CRITICAL-2 regression guard (D-023): builds the model the smoke
    preset would actually construct and asserts the encoder was built with
    attn_mode='block_causal', use_rope=True, token_drop_rate>0.0 -- not
    just that a run completes, but that it ran with the RIGHT config. This
    is the config-wiring test the adversarial review found missing.
    """

    def test_smoke_preset_builds_encoder_with_the_risky_config(self):
        args = parse_arguments(["--smoke"])
        config = _build_config(args)
        model = _build_model(config)
        assert model.encoder.attn_mode == "block_causal"
        assert model.encoder.use_rope is True
        assert model.encoder.token_drop_rate == pytest.approx(0.5)

    def test_explicit_full_attn_mode_reaches_the_encoder(self):
        """Anti-vacuity arm: an explicit override must actually reach the
        encoder, proving _build_model does not silently ignore the config."""
        args = parse_arguments(["--smoke", "--attn-mode", "full", "--no-use-rope",
                                 "--token-drop-rate", "0.0"])
        config = _build_config(args)
        model = _build_model(config)
        assert model.encoder.attn_mode == "full"
        assert model.encoder.use_rope is False
        assert model.encoder.token_drop_rate == pytest.approx(0.0)


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
