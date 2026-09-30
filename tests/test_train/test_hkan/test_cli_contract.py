"""CLI contract guard for ``src/train/hkan/train_hkan.py``.

The defect class: a flag is declared, ``--help`` advertises it, and nothing between
argparse and the point of use reads it. The shared driver ``tests/test_train/
_cli_contract.py`` documents the three traps it designs out (a probe equal to the
default, an incomplete row table, two rows sharing a value).

Shape of this trainer: there is no config dataclass. ``parse_arguments()`` returns
the ``argparse.Namespace`` and every downstream function reads ``args.<field>`` off
it, the same shape as ``train_mothnet.py``. So ``build_config`` is
``parse_arguments`` itself and every row but ``--gpu`` is a ``field`` row whose
field is the argparse destination.

Two kinds of flag need no special handling from the driver, which compares the
parsed value with ``==``:

- the seven per-layer flags parse a comma-separated string into a Python list
  through their argparse ``type``; the row's expected value is that list;
- the three ``store_true`` flags carry a one-element argv fragment and expect
  ``True``.

Rows are driven one at a time. ``--hidden-units 12,7`` makes a three-layer model
while every per-layer flag keeps its single default value; each per-layer row
carries two values against the default two-layer model. Both pass the trainer's
own length check (one value, or one per layer).

``--gpu`` is the one flag parsing cannot vouch for: ``main()`` has a further hop,
``setup_gpu(gpu_id=args.gpu)``. ``test_gpu_flag_reaches_setup_gpu`` drives
``main()`` against a sentinel. The run is pointed at ``tmp_path`` with tiny
settings, so a trainer that no longer calls ``setup_gpu`` at all completes a
two-second run there and fails on the assertion, instead of writing a run
directory under repo-root ``results/``.
"""

from __future__ import annotations

from typing import Tuple

import pytest

import train.hkan.train_hkan as train_hkan

from .._cli_contract import (  # noqa: TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)

#: One row per declared flag; every probe value is unique across the table and
#: differs from the flag's default.
HKAN_ROWS: Tuple[Row, ...] = (
    Row(("--training-mode",), ("--training-mode", "backprop"), "training_mode", "backprop"),
    Row(("--dataset",), ("--dataset", "tf3"), "dataset", "tf3"),
    Row(("--train-csv",), ("--train-csv", "/probe/train.csv"), "train_csv", "/probe/train.csv"),
    Row(("--test-csv",), ("--test-csv", "/probe/test.csv"), "test_csv", "/probe/test.csv"),
    Row(("--scale-csv",), ("--scale-csv",), "scale_csv", True),
    Row(("--num-train-samples",), ("--num-train-samples", "777"), "num_train_samples", 777),
    Row(("--num-test-samples",), ("--num-test-samples", "321"), "num_test_samples", 321),
    Row(("--repeats",), ("--repeats", "5"), "repeats", 5),
    Row(("--seed",), ("--seed", "1234"), "seed", 1234),
    Row(("--hidden-units",), ("--hidden-units", "12,7"), "hidden_units", [12, 7]),
    Row(("--num-basis",), ("--num-basis", "9,11"), "num_basis", [9, 11]),
    Row(("--basis",), ("--basis", "tanh,identity"), "basis", ["tanh", "identity"]),
    Row(("--slope",), ("--slope", "7.5,2.5"), "slope", [7.5, 2.5]),
    Row(("--centers",), ("--centers", "equally_spaced,data"), "centers",
        ["equally_spaced", "data"]),
    Row(("--l2-block",), ("--l2-block", "0.125,0.25"), "l2_block", [0.125, 0.25]),
    Row(("--l2-mix",), ("--l2-mix", "0.375,0.0625"), "l2_mix", [0.375, 0.0625]),
    Row(("--no-block-bias",), ("--no-block-bias",), "no_block_bias", True),
    Row(("--no-bias",), ("--no-bias",), "no_bias", True),
    Row(("--chunk-size",), ("--chunk-size", "13"), "chunk_size", 13),
    Row(("--epochs",), ("--epochs", "17"), "epochs", 17),
    Row(("--batch-size",), ("--batch-size", "19"), "batch_size", 19),
    Row(("--learning-rate",), ("--learning-rate", "0.03125"), "learning_rate", 0.03125),
    Row(("--val-fraction",), ("--val-fraction", "0.35"), "val_fraction", 0.35),
    Row(("--predict-batch-size",), ("--predict-batch-size", "111"), "predict_batch_size", 111),
    Row(("--output-dir",), ("--output-dir", "/probe/hkan-output-dir"), "output_dir",
        "/probe/hkan-output-dir"),
    Row(("--experiment-name",), ("--experiment-name", "probe-experiment-name"),
        "experiment_name", "probe-experiment-name"),
    # Not a `field` row: consumed once in main() by setup_gpu(gpu_id=args.gpu).
    # This row proves parsing only; test_gpu_flag_reaches_setup_gpu proves the call.
    Row(("--gpu",), ("--gpu", "3"), namespace_dest="gpu", expected=3),
)

HKAN_TRAINER = Contract(
    name="train.hkan.train_hkan",
    build_parser=lambda monkeypatch: train_hkan._build_parser(),
    build_config=lambda monkeypatch: train_hkan.parse_arguments(),
    modes=(Mode(id="plain", required_argv=(), rows=HKAN_ROWS),),
)

CONTRACTS: Tuple[Contract, ...] = (HKAN_TRAINER,)

_CASES, _IDS = cases(CONTRACTS)


# ---------------------------------------------------------------------
# flag -> config (argparse Namespace)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    """A flag that parses and arrives nowhere is the silent no-op this guards."""
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    """Trap 1: without this, every row above could pass vacuously."""
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


def test_probe_values_are_mutually_distinct() -> None:
    """Trap 3: two rows sharing a value would hide a cross-wired forward."""
    values = [repr(row.expected) for row in HKAN_ROWS if row.expected is not True]
    assert len(values) == len(set(values)), sorted(values)


# ---------------------------------------------------------------------
# completeness: every declared flag has a row, and vice versa
# ---------------------------------------------------------------------


@pytest.mark.parametrize("contract", CONTRACTS, ids=[c.name for c in CONTRACTS])
def test_every_declared_flag_has_a_contract_row(monkeypatch, contract) -> None:
    """A flag added without a row fails here. The count is the parser's, not pasted."""
    declared = declared_option_strings(contract.build_parser(monkeypatch))
    covered = contract.covered_flags
    assert declared == covered, (
        f"{contract.name}: flags with no contract row "
        f"{sorted(declared - covered)}; rows for flags the parser no longer "
        f"declares {sorted(covered - declared)}"
    )
    assert len(HKAN_ROWS) == len(declared), "a flag has two rows"


def test_every_list_valued_flag_is_probed_with_two_comma_separated_values() -> None:
    """A one-value probe would pass against a ``type`` that never splits on commas."""
    parser = train_hkan._build_parser()
    list_flags = {
        action.option_strings[0] for action in parser._actions
        if isinstance(action.default, list)
    }
    assert len(list_flags) >= 2, "the trainer no longer declares list-valued flags"
    for row in HKAN_ROWS:
        if row.id in list_flags:
            assert isinstance(row.expected, list) and len(row.expected) == 2, row.id
            assert "," in row.argv[1], row.id


# ---------------------------------------------------------------------
# --gpu: forwarding needs main(), not just the parser
# ---------------------------------------------------------------------


class _SetupGpuProbe(Exception):
    """Raised by the sentinel so ``main()`` stops at its second statement."""


def test_gpu_flag_reaches_setup_gpu(monkeypatch, tmp_path) -> None:
    """``--gpu 3`` must reach ``setup_gpu(gpu_id=3)`` inside ``main()``."""
    calls = []

    def _sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        raise _SetupGpuProbe

    monkeypatch.setattr(train_hkan, "setup_gpu", _sentinel)
    argv = [
        "--gpu", "3", "--output-dir", str(tmp_path), "--experiment-name", "gpu_probe",
        "--num-train-samples", "60", "--num-test-samples", "30",
        "--hidden-units", "4", "--num-basis", "4",
    ]
    try:
        train_hkan.main(argv)
    except _SetupGpuProbe:
        pass

    assert calls == [((), {"gpu_id": 3})], (
        f"--gpu 3 did not reach setup_gpu(gpu_id=3): recorded calls={calls!r}"
    )
    assert not (tmp_path / "gpu_probe").exists(), "setup_gpu ran after the run directory was made"
