"""CLI contract guard for ``src/train/distilbert/pretrain.py``.

``distilbert/pretrain.py`` is the single-file Pattern-3 MLM shape (cloned
from ``train.bert.pretrain``, D-003's non-harmonization precedent), not the
zamba2 ``common.py`` + thin-entry split -- so unlike
``tests/test_train/test_zamba2/test_cli_contract.py`` there is no
``config_from_args``/``NON_CONFIG_DESTS`` to check: ``main()`` builds a
plain ``DistilBertTrainingConfig()`` instance and assigns four CLI-exposed
fields onto it directly. This file adapts that file's two scoped checks to
this shape:

1. **Exit 0 is not a passing ``--help``.** ``--help`` must print a
   ``usage:`` line AND must not reach ``setup_gpu``/``train_distilbert_mlm``
   -- sentinels installed over both are asserted uncalled.
2. **``argv -> config`` must wire, not silently default.** Each of the four
   CLI-exposed knobs (``--variant``/``--epochs``/``--batch-size``/
   ``--max-samples``) is driven through the REAL parser and ``main()``, and
   asserted to land on the config object ``train_distilbert_mlm`` actually
   receives (``train_distilbert_mlm`` itself is stubbed so no dataset/GPU
   is touched).

Plus one cheap, fast build check (not a training run):
``create_distilbert_mlm_model`` constructs a real, tiny (``variant="tiny"``,
``max_seq_length=16``) ``DistilBERT`` wrapped in ``MaskedLanguageModel`` and
returns a built model -- ``vocab_size`` is intentionally left at its cl100k_base
default (100277) because the special token ids
(``cls``/``sep``/``pad``/``mask_token_id``, all >= 100264) are only valid
against that vocabulary; shrinking ``vocab_size`` here raises
``ValueError`` from ``MaskedLanguageModel``'s own config validation, not a
defect in the trainer.

Nothing here trains, allocates a GPU, reads the IMDB dataset, or writes into
the repo-root ``results/``.
"""

from __future__ import annotations

import pytest

import train.distilbert.pretrain as trainer
from train.distilbert.pretrain import DistilBertTrainingConfig, create_distilbert_mlm_model


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
    calls = {"setup_gpu": 0, "train_distilbert_mlm": 0}

    def _spy_setup_gpu(*_args, **_kwargs):
        calls["setup_gpu"] += 1

    def _spy_train(*_args, **_kwargs):
        calls["train_distilbert_mlm"] += 1
        raise AssertionError("train_distilbert_mlm must not run on --help")

    monkeypatch.setattr(trainer, "setup_gpu", _spy_setup_gpu)
    monkeypatch.setattr(trainer, "train_distilbert_mlm", _spy_train)

    with pytest.raises(SystemExit) as excinfo:
        trainer.main(["--help"])
    assert excinfo.value.code == 0

    assert calls == {"setup_gpu": 0, "train_distilbert_mlm": 0}, (
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
    monkeypatch.setattr(trainer, "train_distilbert_mlm", _spy_train)

    with pytest.raises(SystemExit):
        trainer.main([])

    defaults = DistilBertTrainingConfig()
    config = captured["config"]
    assert config.distilbert_variant == defaults.distilbert_variant
    assert config.num_epochs == defaults.num_epochs
    assert config.batch_size == defaults.batch_size
    assert config.max_samples == defaults.max_samples


@pytest.mark.parametrize(
    ("argv", "field", "expected"),
    [
        (["--variant", "small"], "distilbert_variant", "small"),
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
    monkeypatch.setattr(trainer, "train_distilbert_mlm", _spy_train)

    with pytest.raises(SystemExit):
        trainer.main(argv)

    config = captured["config"]
    defaults = DistilBertTrainingConfig()
    assert getattr(config, field) == expected
    # Trap: the probe value must differ from the default, or a wiring bug
    # that drops the flag entirely would still pass this assertion.
    assert expected != getattr(defaults, field), (
        f"probe value for {field!r} equals the default; it cannot prove "
        f"the flag was forwarded"
    )


# ---------------------------------------------------------------------
# Cheap, fast model-build check (not a training run)
# ---------------------------------------------------------------------


def test_create_mlm_model_constructs_at_the_tiny_variant() -> None:
    """The smallest shipped ``DistilBERT`` variant, wrapped, builds cleanly.

    ``vocab_size`` is left at its cl100k_base default -- the special token
    ids used for masking are only valid against that vocabulary size (see
    module docstring).
    """
    config = DistilBertTrainingConfig()
    config.distilbert_variant = "tiny"
    config.max_seq_length = 16
    model = create_distilbert_mlm_model(config)
    assert model.built
    assert model.encoder.count_params() > 0
