"""``--smoke`` honours what the user typed, and the ConvUNeXt ``main()`` builds one config.

The audit of the trainer found ``--smoke`` silently ignored 23 flags: ``main()`` built a
second, hand-written ``TrainingConfig`` for smoke, so ``--smoke --epochs 3`` trained 2
epochs, ``--smoke --initial-filters 16`` trained the variant default, and the run was
named ``convunext_denoiser_smoke`` every time (a reused name silently merged into the old
run directory). ``--max-train-files`` / ``--max-val-files`` also lied: ``0`` fell through
``or`` to 10000 / 500.

Now the config is built ONCE (``config_from_args``) and smoke only supplies its preset for
the dests the user did not type. "Typed" is detected by parsing into a namespace
pre-populated with a sentinel for every dest, NOT by comparing with the parser default: a
value typed equal to the default is still typed (the trap ``plans/LESSONS.md`` records --
an override that fills a field only when it equals the parser default cannot tell an
explicit default from an omission, and the user's most likely typed value is the default).

Every expected literal is typed here, never imported from the trainer.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

import train.bfunet.train_convunext_denoiser as trainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def _main_config(argv: List[str]) -> trainer.TrainingConfig:
    """The config ``main()`` hands to ``train`` for ``argv`` (GPU setup and training stubbed)."""
    seen: List[trainer.TrainingConfig] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer, "setup_gpu", lambda gpu_id=None: None)
        patch.setattr(trainer, "train", lambda config: seen.append(config))
        patch.setattr(sys, "argv", ["train_convunext_denoiser", *argv])
        trainer.main()
    (config,) = seen
    return config


# ---------------------------------------------------------------------
# a typed value wins over the smoke preset, including a typed DEFAULT
# ---------------------------------------------------------------------

def test_smoke_honours_an_explicit_epoch_count() -> None:
    config = _main_config(["--smoke", "--epochs", "3"])
    assert config.epochs == 3
    assert config.curriculum_epochs == 3  # derived from the epochs actually run


def test_smoke_honours_explicit_values_that_equal_the_preset() -> None:
    config = _main_config(["--smoke", "--epochs", "2", "--batch-size", "2"])
    assert (config.epochs, config.batch_size) == (2, 2)


# (flag, the flag's own parser default, config attribute, the smoke preset it must beat)
TYPED_DEFAULTS = [
    ("--variant", "base", "variant", "tiny"),
    ("--epochs", "100", "epochs", 2),
    ("--batch-size", "16", "batch_size", 2),
    ("--patch-size", "256", "patch_size", 64),
    ("--patches-per-image", "4", "patches_per_image", 2),
    ("--viz-freq", "5", "viz_freq", 1),
    ("--gabor-filters", "32", "gabor_filters", 8),
    ("--self-iterate-pool-size", "2048", "self_iterate_pool_size", 32),
]


@pytest.mark.parametrize(
    "flag,default,attribute,preset", TYPED_DEFAULTS, ids=[row[0] for row in TYPED_DEFAULTS]
)
def test_a_typed_default_still_beats_the_smoke_preset(flag, default, attribute, preset) -> None:
    """The recorded trap: the flag's default is typed explicitly and must NOT be replaced."""
    config = _main_config(["--smoke", flag, default])
    typed = getattr(config, attribute)
    assert str(typed) == default, f"{flag} {default} was overridden by the smoke preset {preset!r}"
    assert typed != preset


def test_smoke_supplies_every_preset_value_the_user_did_not_type() -> None:
    config = _main_config(["--smoke"])
    assert config.variant == "tiny"
    assert config.epochs == 2
    assert config.curriculum_epochs == 2
    assert config.batch_size == 2
    assert config.patch_size == 64
    assert config.patches_per_image == 2
    assert config.max_train_files == 8
    assert config.max_val_files == 8
    assert config.steps_per_epoch == 3
    assert config.validation_steps == 2
    assert config.warmup_epochs == 0
    assert config.viz_freq == 1
    assert config.gabor_filters == 8
    assert config.learning_rate == 1e-3
    assert (config.sigma_max_start, config.sigma_max_end) == (0.025, 0.25)
    assert config.curriculum_schedule == "linear"
    assert config.self_iterate_pool_size == 32
    assert config.self_iterate_regen_freq == 1


def test_smoke_keeps_the_pool_cap_but_lets_the_user_widen_it() -> None:
    assert _main_config(["--smoke", "--self-iterate"]).self_iterate_pool_size == 32
    widened = _main_config(["--smoke", "--self-iterate", "--self-iterate-pool-size", "64"])
    assert widened.self_iterate_pool_size == 64


def test_smoke_now_honours_the_flags_it_used_to_ignore() -> None:
    """A sample of the 23 flags the old smoke branch dropped, each typed away from preset AND default."""
    config = _main_config([
        "--smoke", "--variant", "small", "--epochs", "7", "--batch-size", "5",
        "--patch-size", "32", "--learning-rate", "0.0005", "--warmup-epochs", "1",
        "--sigma-max-start", "0.03", "--sigma-max-end", "0.3",
        "--curriculum-schedule", "cosine", "--curriculum-epochs", "4",
        "--gabor-filters", "12", "--initial-filters", "24", "--filter-multiplier", "1.5",
        "--final-projection-groups", "3", "--max-train-files", "11", "--max-val-files", "13",
        "--steps-per-epoch", "9", "--validation-steps", "4", "--viz-freq", "3",
    ])
    assert config.variant == "small"
    assert (config.epochs, config.batch_size, config.patch_size) == (7, 5, 32)
    assert config.learning_rate == 0.0005
    assert config.warmup_epochs == 1
    assert (config.sigma_max_start, config.sigma_max_end) == (0.03, 0.3)
    assert (config.curriculum_schedule, config.curriculum_epochs) == ("cosine", 4)
    assert config.gabor_filters == 12
    assert config.initial_filters == 24
    assert config.filter_multiplier == 1.5
    assert config.final_projection_groups == 3
    assert (config.max_train_files, config.max_val_files) == (11, 13)
    assert (config.steps_per_epoch, config.validation_steps) == (9, 4)
    assert config.viz_freq == 3


def test_smoke_no_gabor_projection_is_no_longer_dropped() -> None:
    config = _main_config(
        ["--smoke", "--no-gabor-projection", "--gabor-filters", "32", "--initial-filters", "32"]
    )
    assert config.gabor_stem_projection is False


def test_a_bad_typed_value_is_refused_under_smoke_too() -> None:
    with pytest.raises(ValueError, match="max_val_files"):
        _main_config(["--smoke", "--max-val-files", "0"])


# ---------------------------------------------------------------------
# the smoke run has a unique, honest name
# ---------------------------------------------------------------------

def test_two_smoke_runs_get_different_names(monkeypatch) -> None:
    stamps = iter(["20260101_000001", "20260101_000002"])
    monkeypatch.setattr("train.common.run_io.run_timestamp", lambda: next(stamps))
    first = _main_config(["--smoke"]).experiment_name
    second = _main_config(["--smoke"]).experiment_name
    assert first == "convunext_denoiser_smoke_20260101_000001"
    assert second == "convunext_denoiser_smoke_20260101_000002"


def test_the_smoke_name_is_not_the_old_fixed_name() -> None:
    assert _main_config(["--smoke"]).experiment_name != "convunext_denoiser_smoke"


def test_an_explicit_experiment_name_wins_under_smoke() -> None:
    assert _main_config(["--smoke", "--experiment-name", "mine"]).experiment_name == "mine"


# ---------------------------------------------------------------------
# --max-*-files defaults stop lying
# ---------------------------------------------------------------------

def test_the_file_caps_default_to_none_and_reach_the_config_default_untouched() -> None:
    args = trainer.parse_arguments([])
    assert args.max_train_files is None and args.max_val_files is None
    config = _main_config([])
    assert config.max_train_files == 10000
    assert config.max_val_files == 500
    assert config.validation_steps == 100


@pytest.mark.parametrize("flag,field", [
    ("--max-train-files", "max_train_files"),
    ("--max-val-files", "max_val_files"),
])
def test_a_zero_file_cap_is_refused_not_turned_into_the_default(flag, field) -> None:
    with pytest.raises(ValueError, match=field):
        _main_config([flag, "0"])


def test_a_typed_file_cap_reaches_the_config() -> None:
    config = _main_config(["--max-train-files", "123", "--max-val-files", "45"])
    assert (config.max_train_files, config.max_val_files) == (123, 45)


def _run_module(*argv: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg",
           "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "train.bfunet.train_convunext_denoiser", *argv],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120,
    )


def test_a_zero_val_file_cap_exits_non_zero_from_the_command_line(tmp_path) -> None:
    """The output dir is a tmp_path so a regression that accepts 0 and starts a run leaks nowhere."""
    result = _run_module("--max-val-files", "0", "--output-dir", str(tmp_path / "out"),
                         "--experiment-name", "cap_probe")
    assert result.returncode != 0
    assert "max_val_files" in result.stderr
    assert not (tmp_path / "out" / "cap_probe").exists()


# ---------------------------------------------------------------------
# parse_arguments(argv) and the explicit-flag detection
# ---------------------------------------------------------------------

def test_parse_arguments_takes_an_argv_list() -> None:
    assert trainer.parse_arguments(["--epochs", "3"]).epochs == 3


def test_parse_arguments_defaults_are_the_parser_defaults() -> None:
    args = trainer.parse_arguments([])
    assert (args.epochs, args.batch_size, args.variant) == (100, 16, "base")
    assert args.smoke is False


def test_explicit_is_exactly_the_typed_dests_including_a_typed_default() -> None:
    _, explicit = trainer.parse_arguments_with_explicit(["--epochs", "100", "--smoke"])
    assert explicit == frozenset({"epochs", "smoke"})
    _, none_typed = trainer.parse_arguments_with_explicit([])
    assert none_typed == frozenset()


def test_a_boolean_optional_flag_counts_as_typed() -> None:
    args, explicit = trainer.parse_arguments_with_explicit(["--test-eval"])
    assert args.test_eval is True and explicit == frozenset({"test_eval"})


# ---------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------

def test_help_prints_usage_creates_no_run_directory_and_drops_the_stale_smoke_text() -> None:
    results = REPO_ROOT / "results"
    before = sorted(p.name for p in results.iterdir()) if results.is_dir() else []
    result = _run_module("--help")
    after = sorted(p.name for p in results.iterdir()) if results.is_dir() else []
    assert result.returncode == 0
    assert result.stdout.lstrip().startswith("usage:")
    assert before == after
    assert "constant LR" not in result.stdout
