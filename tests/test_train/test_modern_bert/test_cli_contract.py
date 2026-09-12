"""CLI contract guard for ``src/train/modern_bert/pretrain.py``.

``modern_bert/pretrain.py`` is the single-file Pattern-3 MLM shape (cloned
from ``train.distilbert.pretrain``, itself cloned from
``train.bert.pretrain``, D-003's non-harmonization precedent), not the
zamba2 ``common.py`` + thin-entry split -- so unlike
``tests/test_train/test_zamba2/test_cli_contract.py`` there is no
``config_from_args``/``NON_CONFIG_DESTS`` to check: ``main()`` builds a
plain ``ModernBertTrainingConfig()`` instance and assigns four CLI-exposed
fields onto it directly. This file adapts that file's two scoped checks to
this shape:

1. **Exit 0 is not a passing ``--help``.** ``--help`` must print a
   ``usage:`` line AND must not reach ``setup_gpu``/``train_modern_bert_mlm``
   -- sentinels installed over both are asserted uncalled.
2. **``argv -> config`` must wire, not silently default.** Each of the four
   CLI-exposed knobs (``--variant``/``--epochs``/``--batch-size``/
   ``--max-samples``) is driven through the REAL parser and ``main()``, and
   asserted to land on the config object ``train_modern_bert_mlm`` actually
   receives (``train_modern_bert_mlm`` itself is stubbed so no dataset/GPU
   is touched). ``--variant base`` is used only as a WIRING probe here (the
   stub short-circuits before any model is built) -- the module docstring's
   documented ``ResourceExhaustedError`` risk for "base"/"large" never
   applies in this test.

Plus one cheap, fast build check (not a training run):
``create_modern_bert_mlm_model`` constructs a real, tiny (``variant="tiny"``,
``max_seq_length=16``) ``ModernBERT`` wrapped in ``MaskedLanguageModel`` and
returns a built model -- ``vocab_size`` is intentionally left at its
cl100k_base default (100277) because the special token ids
(``cls``/``sep``/``pad``/``mask_token_id``, all >= 100264) are only valid
against that vocabulary; shrinking ``vocab_size`` here raises ``ValueError``
from ``MaskedLanguageModel``'s own config validation, not a defect in the
trainer. "tiny" is the smallest shipped ``ModernBERT`` variant (see the
module docstring's "base"/"large" OOM warning) so this test never risks it.

Nothing here trains, allocates a GPU, reads the IMDB dataset, or writes into
the repo-root ``results/``.
"""

from __future__ import annotations

import pytest

import train.modern_bert.pretrain as trainer
from train.modern_bert.pretrain import ModernBertTrainingConfig, create_modern_bert_mlm_model


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
    calls = {"setup_gpu": 0, "train_modern_bert_mlm": 0}

    def _spy_setup_gpu(*_args, **_kwargs):
        calls["setup_gpu"] += 1

    def _spy_train(*_args, **_kwargs):
        calls["train_modern_bert_mlm"] += 1
        raise AssertionError("train_modern_bert_mlm must not run on --help")

    monkeypatch.setattr(trainer, "setup_gpu", _spy_setup_gpu)
    monkeypatch.setattr(trainer, "train_modern_bert_mlm", _spy_train)

    with pytest.raises(SystemExit) as excinfo:
        trainer.main(["--help"])
    assert excinfo.value.code == 0

    assert calls == {"setup_gpu": 0, "train_modern_bert_mlm": 0}, (
        f"--help reached training machinery: {calls}"
    )


# ---------------------------------------------------------------------
# argv -> config wiring
# ---------------------------------------------------------------------


def test_no_flags_produces_the_class_defaults(monkeypatch) -> None:
    captured = {}

    def _spy_train(config):
        captured["config"] = config
        raise SystemExit(0)  # short-circuit before the tokenizer/eval tail

    monkeypatch.setattr(trainer, "setup_gpu", lambda **_kwargs: None)
    monkeypatch.setattr(trainer, "train_modern_bert_mlm", _spy_train)

    with pytest.raises(SystemExit):
        trainer.main([])

    defaults = ModernBertTrainingConfig()
    config = captured["config"]
    assert config.modern_bert_variant == defaults.modern_bert_variant
    assert config.num_epochs == defaults.num_epochs
    assert config.batch_size == defaults.batch_size
    assert config.max_samples == defaults.max_samples


@pytest.mark.parametrize(
    ("argv", "field", "expected"),
    [
        (["--variant", "base"], "modern_bert_variant", "base"),
        (["--epochs", "7"], "num_epochs", 7),
        (["--batch-size", "16"], "batch_size", 16),
        (["--max-samples", "500"], "max_samples", 500),
    ],
)
def test_one_flag_reaches_its_field(monkeypatch, argv, field, expected) -> None:
    captured = {}

    def _spy_train(config):
        captured["config"] = config
        raise SystemExit(0)

    monkeypatch.setattr(trainer, "setup_gpu", lambda **_kwargs: None)
    monkeypatch.setattr(trainer, "train_modern_bert_mlm", _spy_train)

    with pytest.raises(SystemExit):
        trainer.main(argv)

    config = captured["config"]
    defaults = ModernBertTrainingConfig()
    assert getattr(config, field) == expected
    # Trap: the probe value must differ from the default, or a wiring bug
    # that drops the flag entirely would still pass this assertion.
    assert expected != getattr(defaults, field), (
        f"probe value for {field!r} equals the default; it cannot prove "
        f"the flag was forwarded"
    )


def test_default_variant_is_never_base_or_large() -> None:
    """The module docstring's own HARD constraint, as an executable guard.

    ``"base"``/``"large"`` are documented to raise ``ResourceExhaustedError``
    on constrained hardware; the shipped DEFAULT must never be either.
    """
    assert ModernBertTrainingConfig().modern_bert_variant not in ("base", "large")


# ---------------------------------------------------------------------
# Cheap, fast model-build check (not a training run)
# ---------------------------------------------------------------------


def test_create_mlm_model_constructs_at_the_tiny_variant() -> None:
    """The smallest shipped ``ModernBERT`` variant, wrapped, builds cleanly.

    ``vocab_size`` is left at its cl100k_base default -- the special token
    ids used for masking are only valid against that vocabulary size (see
    module docstring).
    """
    config = ModernBertTrainingConfig()
    config.modern_bert_variant = "tiny"
    config.max_seq_length = 16
    model = create_modern_bert_mlm_model(config)
    assert model.built
    assert model.encoder.count_params() > 0
