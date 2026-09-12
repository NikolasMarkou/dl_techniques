"""CLI contract guard for ``src/train/zamba2/train_zamba2.py``.

Two things this file exists to catch, scoped to the minimum needed for this
step (the full exhaustive per-field sweep `train.hnet.common`'s suite runs
is a heavier investment than this step's verification budget asks for;
Step 8's real smoke run is the end-to-end proof):

1. **Exit 0 is not a passing ``--help``.** A script with no parser at all
   ignores ``--help``, runs its whole job and exits 0 anyway
   (``src/train/CLAUDE.md``). ``--help`` must print a ``usage:`` line AND
   must not reach ``setup_gpu``/``build_datasets``/``train`` -- sentinels
   installed over all three are asserted uncalled.
2. **``argv -> config`` must wire, not silently default.** A flag parsed
   but not forwarded by ``config_from_args`` is a knob that does nothing.
   One representative flag per field group is driven through the REAL
   parser and asserted to land on the REAL config object, including the
   defaults path (no flags at all).

Nothing here trains, allocates a GPU, reads the 20 GB Wikipedia cache, or
writes into the repo-root ``results/``.
"""

from __future__ import annotations

import argparse

import pytest

from train.zamba2 import common as zamba2_common
from train.zamba2 import train_zamba2 as trainer
from train.zamba2.common import DEFAULT_VARIANT, VARIANT_NAMES, Zamba2TrainingConfig
from train.zamba2.train_zamba2 import NON_CONFIG_DESTS, build_parser, config_from_argv


# ---------------------------------------------------------------------
# --help: prints usage, allocates nothing
# ---------------------------------------------------------------------


def test_help_exits_zero_and_prints_usage(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        trainer.main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("usage:"), (
        f"--help did not print a usage: line first. stdout:\n{captured.out}"
    )


def test_help_allocates_no_gpu_and_touches_no_dataset(monkeypatch, capsys) -> None:
    calls = {"setup_gpu": 0, "build_datasets": 0, "train": 0}

    def _spy_setup_gpu(*_args, **_kwargs):
        calls["setup_gpu"] += 1

    def _spy_build_datasets(*_args, **_kwargs):
        calls["build_datasets"] += 1
        raise AssertionError("build_datasets must not run on --help")

    def _spy_train(*_args, **_kwargs):
        calls["train"] += 1
        raise AssertionError("train must not run on --help")

    # `trainer` (train_zamba2.py) imports `setup_gpu`/`train` by name, so each
    # binds its OWN reference at import time; patching `zamba2_common.train`
    # would not affect the name `trainer.train` the entry point actually
    # calls. `build_datasets` is patched on `zamba2_common` too (belt-and-
    # suspenders: --help never reaches `trainer.train`, so it is never
    # exercised, but a future refactor that has `trainer` call
    # `zamba2_common.build_datasets` directly would still be caught).
    monkeypatch.setattr(trainer, "setup_gpu", _spy_setup_gpu)
    monkeypatch.setattr(zamba2_common, "build_datasets", _spy_build_datasets)
    monkeypatch.setattr(trainer, "train", _spy_train)

    with pytest.raises(SystemExit) as excinfo:
        trainer.main(["--help"])
    assert excinfo.value.code == 0

    assert calls == {"setup_gpu": 0, "build_datasets": 0, "train": 0}, (
        f"--help reached training machinery: {calls}"
    )


def test_the_exit_code_assertion_is_vacuous_without_the_usage_check() -> None:
    """A parser-less stub exits 0 on --help too; exit code alone can't tell.

    Proves assertion 1 above is not a weaker, vacuous form -- a script that
    ignores --help entirely and runs to completion also exits 0.
    """

    def fake_main(argv) -> None:
        return None  # no parser, no argparse.SystemExit, still "succeeds"

    # The exit-code-only form would pass against fake_main even though it
    # never parsed --help at all.
    fake_main(["--help"])  # does not raise -- no usage: line was ever printed


# ---------------------------------------------------------------------
# argv -> config wiring
# ---------------------------------------------------------------------


def test_no_flags_produces_the_dataclass_defaults() -> None:
    config = config_from_argv([])
    defaults = Zamba2TrainingConfig()
    assert config == defaults
    assert config.variant == DEFAULT_VARIANT


@pytest.mark.parametrize(
    ("argv", "field", "expected"),
    [
        (["--variant", "zamba2_small"], "variant", "zamba2_small"),
        (["--seq-length", "128"], "max_seq_length", 128),
        (["--max-seq-length", "256"], "max_seq_length", 256),
        (["--batch-size", "16"], "batch_size", 16),
        (["--epochs", "7"], "epochs", 7),
        (["--steps-per-epoch", "100"], "steps_per_epoch", 100),
        (["--learning-rate", "1e-3"], "learning_rate", 1e-3),
        (["--final-learning-rate", "1e-5"], "final_learning_rate", 1e-5),
        (["--weight-decay", "0.05"], "weight_decay", 0.05),
        (["--warmup-ratio", "0.1"], "warmup_ratio", 0.1),
        (["--gradient-clip-norm", "2.0"], "gradient_clip_norm", 2.0),
        (["--loss-type", "focal"], "loss_type", "focal"),
        (["--focal-gamma", "2.0"], "focal_gamma", 2.0),
        (["--label-smoothing", "0.1"], "label_smoothing", 0.1),
        (["--min-article-length", "500"], "min_article_length", 500),
        (["--max-train-samples", "1000"], "max_train_samples", 1000),
        (["--max-val-samples", "200"], "max_val_samples", 200),
        (["--val-fraction", "0.1"], "val_fraction", 0.1),
        (["--shuffle-shards", "8"], "shuffle_shards", 8),
        (["--shuffle-buffer", "128"], "shuffle_buffer", 128),
        (["--seed", "123"], "seed", 123),
        (["--patience", "9"], "patience", 9),
        (["--output-dir", "/tmp/zamba2_out"], "output_dir", "/tmp/zamba2_out"),
        (["--dataset-root", "/tmp/wiki"], "dataset_root", "/tmp/wiki"),
        (["--wikipedia-config", "20240101.en"], "wikipedia_config", "20240101.en"),
        (["--encoding-name", "gpt2"], "encoding_name", "gpt2"),
    ],
)
def test_one_flag_reaches_its_field(argv, field, expected) -> None:
    config = config_from_argv(argv)
    defaults = Zamba2TrainingConfig()
    assert getattr(config, field) == expected
    # Trap: the probe value must differ from the default, or a wiring bug
    # that drops the flag entirely would still pass this assertion.
    assert expected != getattr(defaults, field), (
        f"probe value for {field!r} equals the default; it cannot prove "
        f"the flag was forwarded"
    )


def test_variant_choices_match_the_config_validation() -> None:
    """``--variant`` choices and ``Zamba2TrainingConfig``'s own validation
    must agree, or a choice argparse accepts could still raise in
    ``__post_init__``."""
    parser = build_parser()
    variant_action = next(
        action for action in parser._actions if action.dest == "variant"
    )
    assert set(variant_action.choices) == set(VARIANT_NAMES)
    for variant in VARIANT_NAMES:
        Zamba2TrainingConfig(variant=variant)  # must not raise


def test_unknown_variant_is_rejected_by_the_config() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        Zamba2TrainingConfig(variant="not-a-real-variant")


# ---------------------------------------------------------------------
# --gpu: the one declared, deliberately non-config, exemption
# ---------------------------------------------------------------------


def test_gpu_is_the_only_non_config_dest() -> None:
    parser = build_parser()
    config_fields = {f.name for f in __import__("dataclasses").fields(Zamba2TrainingConfig)}
    dests = {
        action.dest
        for action in parser._actions
        if action.option_strings and not isinstance(action, argparse._HelpAction)
    }
    non_config = dests - config_fields
    assert non_config == set(NON_CONFIG_DESTS) == {"gpu"}
