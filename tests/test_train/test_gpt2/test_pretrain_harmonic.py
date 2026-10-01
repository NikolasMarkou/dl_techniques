"""``train.gpt2.pretrain_harmonic``: CLI contract, model factory, one real fit."""

from __future__ import annotations

from typing import Tuple

import keras
import numpy as np
import pytest

from dl_techniques.losses import HarmonicCausalLMLoss
from train.gpt2 import pretrain_harmonic as ph

from .._cli_contract import (
    CLM_PRETRAIN_ROWS,
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)

# --loss-type / --focal-gamma are removed from this parser: harmonic loss is the
# only objective, so they would be advertised and ignored.
_INHERITED: Tuple[Row, ...] = tuple(
    r for r in CLM_PRETRAIN_ROWS
    if not set(r.flags) & set(ph._INERT_FLAGS)
)

HARMONIC_ROWS: Tuple[Row, ...] = (
    Row(("--harmonic-exponent",), ("--harmonic-exponent", "48.5"),
        "harmonic_exponent", 48.5),
    Row(("--harmonic-eps",), ("--harmonic-eps", "0.001"), "harmonic_eps", 0.001),
    Row(("--weight-decay",), ("--weight-decay", "0.07"), "weight_decay", 0.07),
)

GPT2_PRETRAIN_HARMONIC = Contract(
    name="train.gpt2.pretrain_harmonic",
    build_parser=lambda monkeypatch: ph._build_harmonic_parser(),
    build_config=lambda monkeypatch: ph._harmonic_config_from_args(
        ph._build_harmonic_parser().parse_args()
    ),
    modes=(Mode("plain", (), _INHERITED + HARMONIC_ROWS),),
)

_CASES, _IDS = cases((GPT2_PRETRAIN_HARMONIC,))


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


def test_every_declared_flag_has_a_contract_row(monkeypatch) -> None:
    declared = declared_option_strings(GPT2_PRETRAIN_HARMONIC.build_parser(monkeypatch))
    assert declared == GPT2_PRETRAIN_HARMONIC.covered_flags


def test_inert_flags_are_gone_and_the_config_keeps_ce_defaults() -> None:
    parser = ph._build_harmonic_parser()
    assert not declared_option_strings(parser) & set(ph._INERT_FLAGS)
    assert "--loss-type" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--loss-type", "focal"])
    config = ph._harmonic_config_from_args(parser.parse_args([]))
    assert (config.loss_type, config.focal_gamma) == ("ce", 1.0)


def test_reference_defaults() -> None:
    config = ph._harmonic_config_from_args(ph._build_harmonic_parser().parse_args([]))
    assert config.learning_rate == 6e-3 and config.weight_decay == 0.0
    assert config.harmonic_exponent is None
    assert "harmonic" in config.save_dir


def _tiny_config(**overrides) -> ph.HarmonicTrainingConfig:
    return ph.HarmonicTrainingConfig(
        model_variant="tiny", num_layers=1, num_heads=2, vocab_size=64,
        max_seq_length=17, **overrides,
    )


def test_factory_builds_harmonic_head_and_loss() -> None:
    model = ph.create_harmonic_gpt2_model(_tiny_config(harmonic_exponent=24.0))
    backbone = model.backbone
    assert backbone.head_type == "harmonic"
    assert backbone.harmonic_exponent == 24.0
    assert isinstance(model.loss_fn, HarmonicCausalLMLoss)


def test_untied_embeddings_are_refused() -> None:
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        ph.create_harmonic_gpt2_model(_tiny_config(tie_word_embeddings=False))


def test_a_few_real_steps_reduce_the_loss_and_checkpoint_reloads(tmp_path) -> None:
    model = ph.create_harmonic_gpt2_model(_tiny_config())
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=3e-3, clipnorm=1.0))
    rng = np.random.default_rng(0)
    x = rng.integers(0, 64, (32, 16)).astype("int32")
    y = np.roll(x, -1, axis=1)
    hist = model.fit(x, y, batch_size=16, epochs=8, verbose=0)
    losses = hist.history["loss"]
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.9 * losses[0]

    path = str(tmp_path / "harm.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    assert loaded.backbone.head_type == "harmonic"
    assert isinstance(loaded.loss_fn, HarmonicCausalLMLoss)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(model(x[:2], training=False)),
        keras.ops.convert_to_numpy(loaded(x[:2], training=False)),
        atol=1e-2,
    )
