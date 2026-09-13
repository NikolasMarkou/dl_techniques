"""CLI contract guard for ``src/train/mamba/train_mamba_v1.py``.

Cloned from ``tests/test_train/test_mamba/test_cli_contract.py`` (the Mamba-2
CLI-contract guard, itself cloned from zamba2's), same shape, same two
things it exists to catch:

1. **Exit 0 is not a passing ``--help``.** A script with no parser at all
   ignores ``--help``, runs its whole job and exits 0 anyway
   (``src/train/CLAUDE.md``). ``--help`` must print a ``usage:`` line AND
   must not reach ``setup_gpu``/``build_datasets``/``train`` -- sentinels
   installed over all three are asserted uncalled.
2. **``argv -> config`` must wire, not silently default.** A representative
   flag per field group is driven through the REAL parser and asserted to
   land on the REAL config object, including the defaults path.

Plus one cheap, fast build check (not a training run): ``build_model``
constructs a real, tiny ``Mamba`` (v1) backbone wrapped in
``CausalLanguageModel`` -- proving the headless-backbone wiring (this plan's
D-012 ``hidden_size``/``get_embedding_matrix`` port) actually produces a
model, without running ``fit()`` or touching a dataset/GPU.

Nothing here trains, allocates a GPU, reads the Wikipedia cache, or writes
into the repo-root ``results/``.
"""

from __future__ import annotations

import argparse

import pytest

from train.mamba import common_v1 as mamba_v1_common
from train.mamba import train_mamba_v1 as trainer
from train.mamba.common_v1 import (
    DEFAULT_VARIANT,
    RESULTS_DIR_PREFIX_V1,
    VARIANT_NAMES,
    MambaV1TrainingConfig,
)
from train.mamba.train_mamba_v1 import NON_CONFIG_DESTS, build_parser, config_from_argv
from train.mamba.common import RESULTS_DIR_PREFIX


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
    monkeypatch.setattr(mamba_v1_common, "build_datasets", _spy_build_datasets)
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
    defaults = MambaV1TrainingConfig()
    assert config == defaults
    assert config.variant == DEFAULT_VARIANT


@pytest.mark.parametrize(
    ("argv", "field", "expected"),
    [
        (["--variant", "370m"], "variant", "370m"),
        (["--seq-length", "128"], "max_seq_length", 128),
        (["--batch-size", "16"], "batch_size", 16),
        (["--epochs", "7"], "epochs", 7),
        (["--d-model", "16"], "d_model", 16),
        (["--num-layers", "2"], "num_layers", 2),
        (["--d-state", "8"], "d_state", 8),
        (["--no-tie-word-embeddings"], "tie_word_embeddings", False),
        (["--learning-rate", "1e-3"], "learning_rate", 1e-3),
        (["--weight-decay", "0.05"], "weight_decay", 0.05),
        (["--loss-type", "focal"], "loss_type", "focal"),
        (["--min-article-length", "500"], "min_article_length", 500),
        (["--max-train-samples", "1000"], "max_train_samples", 1000),
        (["--seed", "123"], "seed", 123),
        (["--output-dir", "/tmp/mamba_v1_out"], "output_dir", "/tmp/mamba_v1_out"),
    ],
)
def test_one_flag_reaches_its_field(argv, field, expected) -> None:
    config = config_from_argv(argv)
    defaults = MambaV1TrainingConfig()
    assert getattr(config, field) == expected
    # Trap: the probe value must differ from the default, or a wiring bug
    # that drops the flag entirely would still pass this assertion.
    assert expected != getattr(defaults, field), (
        f"probe value for {field!r} equals the default; it cannot prove "
        f"the flag was forwarded"
    )


def test_variant_choices_match_the_config_validation() -> None:
    """``--variant`` choices and ``MambaV1TrainingConfig``'s own validation
    must agree, or a choice argparse accepts could still raise in
    ``__post_init__``."""
    parser = build_parser()
    variant_action = next(
        action for action in parser._actions if action.dest == "variant"
    )
    assert set(variant_action.choices) == set(VARIANT_NAMES)
    for variant in VARIANT_NAMES:
        MambaV1TrainingConfig(variant=variant)  # must not raise


def test_unknown_variant_is_rejected_by_the_config() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        MambaV1TrainingConfig(variant="not-a-real-variant")


def test_base_is_a_first_class_variant_key_no_alias_union_needed() -> None:
    """``findings/mamba-v1-trainer-requirements.md``: unlike Mamba-2,
    ``Mamba`` (v1) has no separate ``VARIANT_ALIASES`` -- ``"base"`` is
    already a key of ``MODEL_VARIANTS`` directly."""
    from dl_techniques.models.language.mamba.mamba_v1 import Mamba

    assert "base" in Mamba.MODEL_VARIANTS
    assert VARIANT_NAMES == tuple(Mamba.MODEL_VARIANTS)


# ---------------------------------------------------------------------
# --gpu: the one declared, deliberately non-config, exemption
# ---------------------------------------------------------------------


def test_gpu_is_the_only_non_config_dest() -> None:
    parser = build_parser()
    config_fields = {
        f.name for f in __import__("dataclasses").fields(MambaV1TrainingConfig)
    }
    dests = {
        action.dest
        for action in parser._actions
        if action.option_strings and not isinstance(action, argparse._HelpAction)
    }
    non_config = dests - config_fields
    assert non_config == set(NON_CONFIG_DESTS) == {"gpu"}


# ---------------------------------------------------------------------
# results/ prefix must differ from Mamba-2's, to avoid a collision
# ---------------------------------------------------------------------


def test_v1_results_dir_prefix_differs_from_v2() -> None:
    """Both architectures share the same variant-name set (``130m``, etc.) --
    a shared bare ``"mamba"`` prefix would make a v1 and a v2 run at the same
    variant indistinguishable in ``results/`` except by timestamp."""
    assert RESULTS_DIR_PREFIX_V1 != RESULTS_DIR_PREFIX
    assert RESULTS_DIR_PREFIX_V1 == "mamba_v1"


# ---------------------------------------------------------------------
# Cheap, fast model-build check (not a training run)
# ---------------------------------------------------------------------


def test_build_model_constructs_at_a_tiny_size() -> None:
    """``build_model`` wraps a headless ``Mamba`` (v1) in ``CausalLanguageModel``.

    ``d_model=32, num_layers=1, d_state=8`` and ``vocab_size=17`` are chosen
    purely for construction speed -- this is not a training run. Unlike
    Mamba-2's ``Mamba2Layer`` (``d_ssm`` must be divisible by ``headdim``),
    ``Mamba`` (v1) has no such divisibility constraint, so a smaller
    ``d_model`` than the Mamba-2 CLI-contract test uses is fine here.
    """
    config = MambaV1TrainingConfig(
        variant=DEFAULT_VARIANT, d_model=32, num_layers=1, d_state=8,
    )
    model = mamba_v1_common.build_model(config, steps_per_epoch=1, vocab_size=17)
    assert model.compiled
    assert model.vocab_size == 17
