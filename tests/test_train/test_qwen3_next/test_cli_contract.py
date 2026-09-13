"""CLI contract guard for ``src/train/qwen3_next/train_qwen3_next.py``.

Cloned from ``tests/test_train/test_qwen/test_cli_contract.py`` (same
``common.py`` + thin-entry shape). Scoped to the same two things that file
exists to catch:

1. **Exit 0 is not a passing ``--help``.** ``--help`` must print a
   ``usage:`` line AND must not reach ``setup_gpu``/``build_datasets``/
   ``train`` -- sentinels installed over all three are asserted uncalled.
2. **``argv -> config`` must wire, not silently default.** A representative
   flag per field group is driven through the REAL parser and asserted to
   land on the REAL config object, including the defaults path.

Plus one cheap, fast build check (not a training run): ``build_model`` on
the smallest shipped variant (``"tiny"``, MoE-enabled, 512 hidden / 3
blocks) with a tiny ``vocab_size`` override -- ``Qwen3Next`` already bakes
its own LM head (like plain ``Qwen3``), so this is a direct
``Qwen3Next.from_variant`` + ``compile()`` round trip.

Nothing here trains, allocates a GPU, reads the Wikipedia cache, or writes
into the repo-root ``results/``.
"""

from __future__ import annotations

import argparse

import pytest

from train.qwen3_next import common as qwen3_next_common
from train.qwen3_next import train_qwen3_next as trainer
from train.qwen3_next.common import (
    DEFAULT_VARIANT,
    RESULTS_DIR_PREFIX,
    VARIANT_NAMES,
    Qwen3NextTrainingConfig,
)
from train.qwen3_next.train_qwen3_next import NON_CONFIG_DESTS, build_parser, config_from_argv


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

    monkeypatch.setattr(trainer, "setup_gpu", _spy_setup_gpu)
    monkeypatch.setattr(qwen3_next_common, "build_datasets", _spy_build_datasets)
    monkeypatch.setattr(trainer, "train", _spy_train)

    with pytest.raises(SystemExit) as excinfo:
        trainer.main(["--help"])
    assert excinfo.value.code == 0

    assert calls == {"setup_gpu": 0, "build_datasets": 0, "train": 0}, (
        f"--help reached training machinery: {calls}"
    )


# ---------------------------------------------------------------------
# argv -> config wiring
# ---------------------------------------------------------------------


def test_no_flags_produces_the_dataclass_defaults() -> None:
    config = config_from_argv([])
    defaults = Qwen3NextTrainingConfig()
    assert config == defaults
    assert config.variant == DEFAULT_VARIANT


def test_default_variant_is_moe_enabled_and_small() -> None:
    """The plan's own falsification signal: an aggregation RED-proof against
    a variant that turns out dense (``num_experts=1``) would be vacuous. The
    default must never be ``"80b"`` (dense) or ``"80b_a3b"`` (production
    scale)."""
    from dl_techniques.models.language.qwen.qwen3_next import Qwen3Next

    assert DEFAULT_VARIANT == "tiny"
    assert DEFAULT_VARIANT not in ("80b", "80b_a3b")
    assert Qwen3Next.MODEL_VARIANTS[DEFAULT_VARIANT]["num_experts"] > 1


@pytest.mark.parametrize(
    ("argv", "field", "expected"),
    [
        (["--variant", "small"], "variant", "small"),
        (["--seq-length", "128"], "max_seq_length", 128),
        (["--max-seq-length", "256"], "max_seq_length", 256),
        (["--batch-size", "16"], "batch_size", 16),
        (["--epochs", "7"], "epochs", 7),
        (["--hidden-size", "32"], "hidden_size", 32),
        (["--num-layers", "2"], "num_layers", 2),
        (["--num-attention-heads", "4"], "num_attention_heads", 4),
        (["--num-key-value-heads", "1"], "num_key_value_heads", 1),
        (["--learning-rate", "1e-3"], "learning_rate", 1e-3),
        (["--weight-decay", "0.05"], "weight_decay", 0.05),
        (["--loss-type", "focal"], "loss_type", "focal"),
        (["--min-article-length", "500"], "min_article_length", 500),
        (["--max-train-samples", "1000"], "max_train_samples", 1000),
        (["--seed", "123"], "seed", 123),
        (["--output-dir", "/tmp/qwen3_next_out"], "output_dir", "/tmp/qwen3_next_out"),
    ],
)
def test_one_flag_reaches_its_field(argv, field, expected) -> None:
    config = config_from_argv(argv)
    defaults = Qwen3NextTrainingConfig()
    assert getattr(config, field) == expected
    # Trap: the probe value must differ from the default, or a wiring bug
    # that drops the flag entirely would still pass this assertion.
    assert expected != getattr(defaults, field), (
        f"probe value for {field!r} equals the default; it cannot prove "
        f"the flag was forwarded"
    )


def test_variant_choices_match_the_config_validation() -> None:
    """``--variant`` choices and ``Qwen3NextTrainingConfig``'s own
    validation must agree, or a choice argparse accepts could still raise
    in ``__post_init__``."""
    parser = build_parser()
    variant_action = next(
        action for action in parser._actions if action.dest == "variant"
    )
    assert set(variant_action.choices) == set(VARIANT_NAMES)
    for variant in VARIANT_NAMES:
        Qwen3NextTrainingConfig(variant=variant)  # must not raise


def test_unknown_variant_is_rejected_by_the_config() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        Qwen3NextTrainingConfig(variant="not-a-real-variant")


# ---------------------------------------------------------------------
# --gpu: the one declared, deliberately non-config, exemption
# ---------------------------------------------------------------------


def test_gpu_is_the_only_non_config_dest() -> None:
    parser = build_parser()
    config_fields = {
        f.name for f in __import__("dataclasses").fields(Qwen3NextTrainingConfig)
    }
    dests = {
        action.dest
        for action in parser._actions
        if action.option_strings and not isinstance(action, argparse._HelpAction)
    }
    non_config = dests - config_fields
    assert non_config == set(NON_CONFIG_DESTS) == {"gpu"}


# ---------------------------------------------------------------------
# results/ prefix must differ from plain Qwen3's
# ---------------------------------------------------------------------


def test_results_dir_prefix_differs_from_plain_qwen3() -> None:
    from train.qwen.common import RESULTS_DIR_PREFIX as QWEN3_RESULTS_DIR_PREFIX

    assert RESULTS_DIR_PREFIX != QWEN3_RESULTS_DIR_PREFIX
    assert RESULTS_DIR_PREFIX == "qwen3_next"


# ---------------------------------------------------------------------
# Cheap, fast model-build check (not a training run)
# ---------------------------------------------------------------------


def test_build_model_constructs_on_the_smallest_shipped_variant() -> None:
    """``build_model`` on the ``"tiny"`` variant with a tiny vocabulary.

    ``Qwen3Next`` bakes its own LM head, so no local wrapper is involved
    here (like plain Qwen3, unlike ``mamba/common.py``'s
    ``build_causal_lm_model``). ``vocab_size`` and ``max_seq_length`` are
    overridden small purely for construction speed -- this is not a
    training run.
    """
    config = Qwen3NextTrainingConfig(
        variant="tiny", hidden_size=32, num_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_seq_length=16,
    )
    model = qwen3_next_common.build_model(config, steps_per_epoch=1, vocab_size=50)
    assert model.compiled
    # Qwen3Next is a subclassed (not Functional) model -- no `.output_shape`;
    # `.vocab_size` is the constructor's own record of what was requested.
    assert model.vocab_size == 50
    assert model.aggregate_backbone_losses is True
