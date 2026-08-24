"""Every flag ``train.wave_field.pretrain`` declares must reach its config field.

The trainer had no coverage of this contract before this module. Its
``_config_from_args`` is 26 hand-written forwarding lines; drop one and the
flag stays in ``--help``, still parses, and silently does nothing.

The row table is imported from ``tests/test_train/_cli_contract.py`` and SHARED
with ``tests/test_train/test_gpt2/test_cli_contract.py``: the two CLM trainers
are required to expose the same flag surface (`src/train/CLAUDE.md`, Pattern 3)
and now share one config base (D-010). One table means a flag that drifts onto
only one of them turns that trainer's completeness test RED.

``--field-size`` is wave_field's only additional flag. It is also the field
D-011 moved to LAST in the dataclass by inheritance, so a positional
construction would put the wrong value in it -- pinned here by value.
"""

from __future__ import annotations

from typing import Tuple

import pytest

from train.wave_field import pretrain as wf

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

WAVE_FIELD_ROWS: Tuple[Row, ...] = CLM_PRETRAIN_ROWS + (
    Row(("--field-size",), ("--field-size", "64"), "field_size", 64),
)

WAVE_FIELD_PRETRAIN = Contract(
    name="train.wave_field.pretrain",
    build_parser=lambda monkeypatch: wf._build_parser(),
    build_config=lambda monkeypatch: wf._config_from_args(
        wf._build_parser().parse_args()
    ),
    modes=(Mode("plain", (), WAVE_FIELD_ROWS),),
)

CONTRACTS = (WAVE_FIELD_PRETRAIN,)

_CASES, _IDS = cases(CONTRACTS)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract", CONTRACTS, ids=[c.name for c in CONTRACTS])
def test_every_declared_flag_has_a_contract_row(monkeypatch, contract) -> None:
    """The table is read against the REAL parser, so a new flag fails here."""
    declared = declared_option_strings(contract.build_parser(monkeypatch))
    covered = contract.covered_flags
    assert declared == covered, (
        f"{contract.name}: flags with no contract row {sorted(declared - covered)}; "
        f"rows for flags the parser does not declare {sorted(covered - declared)}"
    )
