"""CLI contract guard for ``src/train/dasiamrpn/train_dasiamrpn.py``.

``build_config`` is ``config_from_args(parse_arguments())`` and every row but
``--gpu`` is a ``field`` row. See ``tests/test_train/_cli_contract.py`` for
the three traps the driver designs out.
"""

from __future__ import annotations

from typing import Tuple

import pytest

import train.dasiamrpn.train_dasiamrpn as train_dasiamrpn

from .._cli_contract import (  # noqa: TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)

DASIAMRPN_ROWS: Tuple[Row, ...] = (
    Row(("--variant",), ("--variant", "big"), "variant", "big"),
    Row(("--data-source",), ("--data-source", "coco"), "data_source", "coco"),
    Row(("--coco-data-dir",), ("--coco-data-dir", "/probe/coco"), "coco_data_dir",
        "/probe/coco"),
    Row(("--coco-split",), ("--coco-split", "validation"), "coco_split", "validation"),
    Row(("--num-synthetic-train",), ("--num-synthetic-train", "7"), "num_synthetic_train", 7),
    Row(("--num-synthetic-val",), ("--num-synthetic-val", "9"), "num_synthetic_val", 9),
    Row(("--steps-per-epoch",), ("--steps-per-epoch", "11"), "steps_per_epoch", 11),
    Row(("--val-steps",), ("--val-steps", "13"), "val_steps", 13),
    Row(("--no-augment",), ("--no-augment",), "augment", False),
    Row(("--brightness-delta",), ("--brightness-delta", "0.5"), "brightness_delta", 0.5),
    Row(("--seed",), ("--seed", "1234"), "seed", 1234),
    Row(("--exemplar-size",), ("--exemplar-size", "96"), "exemplar_size", 96),
    Row(("--search-size",), ("--search-size", "300"), "search_size", 300),
    Row(("--no-batch-norm",), ("--no-batch-norm",), "use_batch_norm", False),
    Row(("--pos-iou",), ("--pos-iou", "0.7"), "pos_iou", 0.7),
    Row(("--neg-iou",), ("--neg-iou", "0.2"), "neg_iou", 0.2),
    Row(("--cls-weight",), ("--cls-weight", "2.5"), "cls_weight", 2.5),
    Row(("--reg-weight",), ("--reg-weight", "0.5"), "reg_weight", 0.5),
    Row(("--huber-delta",), ("--huber-delta", "2.0"), "huber_delta", 2.0),
    Row(("--batch-size",), ("--batch-size", "3"), "batch_size", 3),
    Row(("--epochs",), ("--epochs", "7"), "epochs", 7),
    Row(("--learning-rate",), ("--learning-rate", "0.03125"), "learning_rate", 0.03125),
    Row(("--optimizer",), ("--optimizer", "sgd"), "optimizer_type", "sgd"),
    Row(("--lr-schedule",), ("--lr-schedule", "constant"), "lr_schedule_type", "constant"),
    Row(("--warmup-epochs",), ("--warmup-epochs", "4"), "warmup_epochs", 4),
    Row(("--weight-decay",), ("--weight-decay", "0.0625"), "weight_decay", 0.0625),
    Row(("--gradient-clipping",), ("--gradient-clipping", "0.375"), "gradient_clipping",
        0.375),
    Row(("--momentum",), ("--momentum", "0.5"), "momentum", 0.5),
    Row(("--early-stopping-patience",), ("--early-stopping-patience", "13"),
        "early_stopping_patience", 13),
    Row(("--output-dir",), ("--output-dir", "/probe/dasiamrpn-output-dir"), "output_dir",
        "/probe/dasiamrpn-output-dir"),
    Row(("--experiment-name",), ("--experiment-name", "probe-experiment-name"),
        "experiment_name", "probe-experiment-name"),
    Row(("--gpu",), ("--gpu", "3"), namespace_dest="gpu", expected=3),
)

DASIAMRPN_TRAINER = Contract(
    name="train.dasiamrpn.train_dasiamrpn",
    build_parser=lambda monkeypatch: train_dasiamrpn._build_parser(),
    build_config=lambda monkeypatch: train_dasiamrpn.config_from_args(
        train_dasiamrpn.parse_arguments()),
    modes=(Mode(id="plain", required_argv=(), rows=DASIAMRPN_ROWS),),
)

CONTRACTS: Tuple[Contract, ...] = (DASIAMRPN_TRAINER,)

_CASES, _IDS = cases(CONTRACTS)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


def test_flag_table_is_complete() -> None:
    parser = train_dasiamrpn._build_parser()
    assert declared_option_strings(parser) == DASIAMRPN_TRAINER.covered_flags


def test_help_lists_every_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        train_dasiamrpn.parse_arguments(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    for flag in declared_option_strings(train_dasiamrpn._build_parser()):
        assert flag in out
