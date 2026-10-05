"""CLI contract guard for ``src/train/deepsort/train_deepsort.py``.

``build_config`` is ``config_from_args(parse_arguments())`` and every row but
``--gpu`` is a ``field`` row. See ``tests/test_train/_cli_contract.py`` for
the three traps the driver designs out.
"""

from __future__ import annotations

from typing import Tuple

import pytest

import train.deepsort.train_deepsort as train_deepsort

from .._cli_contract import (  # noqa: TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)

DEEPSORT_ROWS: Tuple[Row, ...] = (
    Row(("--data-source",), ("--data-source", "market1501"), "data_source",
        "market1501"),
    Row(("--market1501-root",), ("--market1501-root", "/probe/market"), "market1501_root",
        "/probe/market"),
    Row(("--num-synthetic-ids",), ("--num-synthetic-ids", "7"), "num_synthetic_ids", 7),
    Row(("--shots-per-id",), ("--shots-per-id", "9"), "shots_per_id", 9),
    Row(("--p-ids",), ("--p-ids", "5"), "p_ids", 5),
    Row(("--k-shots",), ("--k-shots", "3"), "k_shots", 3),
    Row(("--validation-id-fraction",), ("--validation-id-fraction", "0.25"),
        "validation_id_fraction", 0.25),
    Row(("--no-augment",), ("--no-augment",), "augment", False),
    Row(("--seed",), ("--seed", "1234"), "seed", 1234),
    Row(("--loss-mode",), ("--loss-mode", "triplet"), "loss_mode", "triplet"),
    Row(("--dropout-rate",), ("--dropout-rate", "0.1"), "dropout_rate", 0.1),
    Row(("--epochs",), ("--epochs", "7"), "epochs", 7),
    Row(("--steps-per-epoch",), ("--steps-per-epoch", "11"), "steps_per_epoch", 11),
    Row(("--val-steps",), ("--val-steps", "13"), "val_steps", 13),
    Row(("--learning-rate",), ("--learning-rate", "0.03125"), "learning_rate", 0.03125),
    Row(("--optimizer",), ("--optimizer", "sgd"), "optimizer_type", "sgd"),
    Row(("--lr-schedule",), ("--lr-schedule", "cosine_decay"), "lr_schedule_type",
        "cosine_decay"),
    Row(("--warmup-epochs",), ("--warmup-epochs", "4"), "warmup_epochs", 4),
    Row(("--weight-decay",), ("--weight-decay", "0.0625"), "weight_decay", 0.0625),
    Row(("--gradient-clipping",), ("--gradient-clipping", "0.375"), "gradient_clipping",
        0.375),
    Row(("--momentum",), ("--momentum", "0.5"), "momentum", 0.5),
    Row(("--early-stopping-patience",), ("--early-stopping-patience", "13"),
        "early_stopping_patience", 13),
    Row(("--output-dir",), ("--output-dir", "/probe/deepsort-output-dir"), "output_dir",
        "/probe/deepsort-output-dir"),
    Row(("--experiment-name",), ("--experiment-name", "probe-experiment-name"),
        "experiment_name", "probe-experiment-name"),
    Row(("--gpu",), ("--gpu", "3"), namespace_dest="gpu", expected=3),
)

DEEPSORT_TRAINER = Contract(
    name="train.deepsort.train_deepsort",
    build_parser=lambda monkeypatch: train_deepsort._build_parser(),
    build_config=lambda monkeypatch: train_deepsort.config_from_args(
        train_deepsort.parse_arguments()),
    modes=(Mode(id="plain", required_argv=(), rows=DEEPSORT_ROWS),),
)

CONTRACTS: Tuple[Contract, ...] = (DEEPSORT_TRAINER,)

_CASES, _IDS = cases(CONTRACTS)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


def test_flag_table_is_complete() -> None:
    parser = train_deepsort._build_parser()
    assert declared_option_strings(parser) == DEEPSORT_TRAINER.covered_flags


def test_help_lists_every_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        train_deepsort.parse_arguments(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    for flag in declared_option_strings(train_deepsort._build_parser()):
        assert flag in out
