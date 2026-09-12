"""CLI contract guard for ``src/train/mamba/train_mamba.py``.

Cloned from ``tests/test_train/test_zamba2/test_cli_contract.py`` (the same
``common.py`` + thin-entry shape, D-003's zamba2-clone precedent). Scoped to
the same two things that file exists to catch:

1. **Exit 0 is not a passing ``--help``.** A script with no parser at all
   ignores ``--help``, runs its whole job and exits 0 anyway
   (``src/train/CLAUDE.md``). ``--help`` must print a ``usage:`` line AND
   must not reach ``setup_gpu``/``build_datasets``/``train`` -- sentinels
   installed over all three are asserted uncalled.
2. **``argv -> config`` must wire, not silently default.** A representative
   flag per field group is driven through the REAL parser and asserted to
   land on the REAL config object, including the defaults path.

Plus one cheap, fast build check (not a training run): ``build_model``
constructs a real, tiny ``Mamba2`` backbone wrapped in ``CausalLanguageModel``
-- proving the (post plan-2026-09-12T195532-422091c3, D-004-superseding)
headless-backbone wiring actually produces a model, without running
``fit()`` or touching a dataset/GPU.

Nothing here trains, allocates a GPU, reads the Wikipedia cache, or writes
into the repo-root ``results/``.
"""

from __future__ import annotations

import argparse

import pytest

from train.mamba import common as mamba_common
from train.mamba import train_mamba as trainer
from train.mamba.common import DEFAULT_VARIANT, VARIANT_NAMES, Mamba2TrainingConfig
from train.mamba.train_mamba import NON_CONFIG_DESTS, build_parser, config_from_argv


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
    monkeypatch.setattr(mamba_common, "build_datasets", _spy_build_datasets)
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
    defaults = Mamba2TrainingConfig()
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
        (["--output-dir", "/tmp/mamba_out"], "output_dir", "/tmp/mamba_out"),
    ],
)
def test_one_flag_reaches_its_field(argv, field, expected) -> None:
    config = config_from_argv(argv)
    defaults = Mamba2TrainingConfig()
    assert getattr(config, field) == expected
    # Trap: the probe value must differ from the default, or a wiring bug
    # that drops the flag entirely would still pass this assertion.
    assert expected != getattr(defaults, field), (
        f"probe value for {field!r} equals the default; it cannot prove "
        f"the flag was forwarded"
    )


def test_variant_choices_match_the_config_validation() -> None:
    """``--variant`` choices and ``Mamba2TrainingConfig``'s own validation
    must agree, or a choice argparse accepts could still raise in
    ``__post_init__``."""
    parser = build_parser()
    variant_action = next(
        action for action in parser._actions if action.dest == "variant"
    )
    assert set(variant_action.choices) == set(VARIANT_NAMES)
    for variant in VARIANT_NAMES:
        Mamba2TrainingConfig(variant=variant)  # must not raise


def test_unknown_variant_is_rejected_by_the_config() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        Mamba2TrainingConfig(variant="not-a-real-variant")


# ---------------------------------------------------------------------
# --gpu: the one declared, deliberately non-config, exemption
# ---------------------------------------------------------------------


def test_gpu_is_the_only_non_config_dest() -> None:
    parser = build_parser()
    config_fields = {f.name for f in __import__("dataclasses").fields(Mamba2TrainingConfig)}
    dests = {
        action.dest
        for action in parser._actions
        if action.option_strings and not isinstance(action, argparse._HelpAction)
    }
    non_config = dests - config_fields
    assert non_config == set(NON_CONFIG_DESTS) == {"gpu"}


# ---------------------------------------------------------------------
# Cheap, fast model-build check (not a training run)
# ---------------------------------------------------------------------


def test_build_model_constructs_at_a_tiny_size() -> None:
    """``build_model`` wraps a headless ``Mamba2`` in ``CausalLanguageModel``.

    ``d_model=64, num_layers=1, d_state=8`` and ``vocab_size=17`` are chosen
    purely for construction speed -- this is not a training run.
    ``d_model=64`` (not smaller) is required by ``Mamba2Layer``'s own
    constraint that ``d_ssm`` (``d_model * expand``, ``expand`` defaults to
    2) be divisible by ``headdim`` (defaults to 64) --
    ``Mamba2TrainingConfig.variant_overrides`` does not expose
    ``expand``/``headdim`` to shrink further. Unlike Gemma3/Qwen3
    (``skip_head=True``, D-001), Mamba2 is genuinely headless, so this is
    the one candidate where ``CausalLanguageModel`` builds its OWN output
    head (D-004, superseding the retired local ``build_causal_lm_model``
    wrapper) -- proven here by construction succeeding and ``.compiled``
    being true, not by an ``.output_shape`` on a Functional model (this is a
    subclassed model, same as gemma/qwen post-migration).
    """
    config = Mamba2TrainingConfig(
        variant=DEFAULT_VARIANT, d_model=64, num_layers=1, d_state=8,
    )
    model = mamba_common.build_model(config, steps_per_epoch=1, vocab_size=17)
    assert model.compiled
    assert model.vocab_size == 17
