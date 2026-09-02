r"""CLI contract guard for ``src/train/bit_diffusion/train_bit_diffusion.py``.

The defect class this exists to catch
-------------------------------------
A trainer declares a flag, ``--help`` advertises it, and ``main()`` never
forwards it. The flag then silently does nothing: no error, no warning, and
``config.json`` records the DEFAULT while the user believes they overrode it.
This repo has shipped that bug and pinned it in two shared instruments, both
REUSED here rather than re-implemented:

* ``tests/test_train/_cli_contract.py`` -- drives the REAL parser and the REAL
  config builder and asserts on the resulting OBJECT. Its three designed-out
  traps (a row must carry a non-default value; the row table must be complete;
  rows must be driven one at a time) are what make the assertions mean anything.
* ``tests/test_train/test_config_fields_are_live.py`` -- the other end of the
  same contract, asking "does anything READ this field?". ``TrainingConfig`` is
  REGISTERED there, so a field that only reaches ``save_config_json`` fails
  THERE rather than needing a weaker copy here.

THE ROW TABLE IS GENERATED, NOT TYPED. A hand-written table of 37 rows is 37
chances to type a probe value that is already the default (trap 1), and it goes
stale the day a flag is added. :func:`_rows` derives each row from the REAL
parser action plus the REAL dataclass default: a ``choices`` action gets a
different choice, a ``BooleanOptionalAction`` gets the flipped spelling, and a
scalar gets its default displaced by a fixed offset. The shared
``assert_row_value_is_not_the_default`` then MEASURES that every generated probe
differs from the default -- generation is not trusted, it is checked.

``--help`` MUST ALLOCATE NOTHING. Sentinels are installed over ``setup_gpu``,
``create_model``, ``build_bridge_dataset`` and ``train_bit_diffusion``; each
raises on contact and each is asserted to have been called ZERO times. Moving
``parse_arguments()`` below any of them fails by the named ``--help reached ...``
message, not by the exit code -- so this distinguishes "parsed FIRST" from
"parsed at all". Asserting only ``exit == 0`` is measurably weaker: a script
with no parser at all runs its whole job and exits 0.

Nothing here trains, allocates a GPU, or writes into the repo-root ``results/``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Tuple

import pytest

from train.bit_diffusion import train_bit_diffusion as trainer
from train.bit_diffusion.train_bit_diffusion import (
    CLI_TO_CONFIG,
    NON_CONFIG_DESTS,
    SMOKE_PRESET,
    TrainingConfig,
    build_parser,
    config_from_argv,
)

from .._cli_contract import (  # noqa: TID252 -- shared driver, one level up
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)

#: Displacement applied to a scalar default to build a probe value. Any value
#: works as long as it is not zero; ``0.375`` is exactly representable in binary
#: floating point, so ``float(str(default + 0.375))`` round-trips exactly and the
#: equality assertion cannot fail on a formatting artefact.
FLOAT_OFFSET = 0.375
INT_OFFSET = 7


def _probe_value(action: argparse.Action, default: Any) -> Any:
    """A value for ``action`` that is guaranteed to differ from ``default``.

    Interface contract: pure. Reads only the action's declared ``choices`` /
    ``type`` and the dataclass default; never parses anything.

    :param action: The real argparse action.
    :param default: The dataclass default for the field it writes.
    :return: The probe value.
    :raises ValueError: If the action declares a single-element ``choices``, in
        which case no differing value exists and the row would be vacuous.
    """
    if isinstance(action, argparse.BooleanOptionalAction):
        return not bool(default)
    if action.choices:
        alternatives = [c for c in action.choices if c != default]
        if not alternatives:
            raise ValueError(
                f"{action.option_strings[0]} has no choice other than its "
                f"default {default!r}; the row cannot distinguish forwarded "
                "from untouched"
            )
        return alternatives[0]
    if action.type is int:
        return (default if isinstance(default, int) else 0) + INT_OFFSET
    if action.type is float:
        return (default if isinstance(default, float) else 0.0) + FLOAT_OFFSET
    return f"probe-{action.dest}"


def _argv_for(action: argparse.Action, value: Any) -> Tuple[str, ...]:
    """The argv fragment that sets ``action`` to ``value``."""
    if isinstance(action, argparse.BooleanOptionalAction):
        positive = action.option_strings[0]
        negative = next(
            option for option in action.option_strings if option.startswith("--no-")
        )
        return (positive,) if value else (negative,)
    return (action.option_strings[0], str(value))


def _actions_by_dest() -> Dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in build_parser()._actions
        if action.option_strings
    }


def _rows() -> Tuple[Row, ...]:
    """One generated :class:`Row` per ``CLI_TO_CONFIG`` entry."""
    actions = _actions_by_dest()
    defaults = TrainingConfig()
    rows: List[Row] = []
    for dest, field_name in CLI_TO_CONFIG.items():
        action = actions[dest]
        value = _probe_value(action, getattr(defaults, field_name))
        rows.append(
            Row(
                flags=tuple(action.option_strings),
                argv=_argv_for(action, value),
                field=field_name,
                expected=value,
            )
        )
    # `--gpu` is deliberately NOT a config field: it acts on the process.
    gpu_action = actions["gpu"]
    rows.append(
        Row(
            flags=tuple(gpu_action.option_strings),
            argv=("--gpu", "1"),
            namespace_dest="gpu",
            expected=1,
        )
    )
    return tuple(rows)


CONTRACT = Contract(
    name="train_bit_diffusion.py",
    build_parser=lambda monkeypatch: build_parser(),
    build_config=lambda monkeypatch: config_from_argv(None),
    modes=(Mode(id="default", required_argv=(), rows=_rows()),),
)

_CASES, _IDS = cases([CONTRACT])


# ---------------------------------------------------------------------
# flag -> config
# ---------------------------------------------------------------------


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_every_cli_value_reaches_the_config(monkeypatch, contract, mode, row):
    """The repo's silent-no-op bug class: a flag that parses and is never used."""
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_the_probe_value_is_not_already_the_default(
    monkeypatch, contract, mode, row
):
    """Trap 1. Without this, every row above could pass vacuously."""
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


def test_every_declared_flag_is_accounted_for():
    """Trap 2: a flag added without a wiring row fails HERE, not in production."""
    declared = declared_option_strings(build_parser())
    covered = CONTRACT.covered_flags
    assert declared == covered, (
        f"flags with no contract row: {sorted(declared - covered)}; rows for "
        f"flags that no longer exist: {sorted(covered - declared)}"
    )


def test_every_config_field_has_a_cli_flag():
    """The other direction: a field nobody can set is a knob with no handle."""
    declared_fields = {item.name for item in dataclass_fields(TrainingConfig)}
    wired = set(CLI_TO_CONFIG.values())
    assert declared_fields == wired, (
        f"TrainingConfig fields with no flag: {sorted(declared_fields - wired)}; "
        f"CLI_TO_CONFIG rows naming no field: {sorted(wired - declared_fields)}"
    )


def test_the_non_config_dests_really_are_not_config_fields():
    """``--gpu`` and ``-h`` act on the process; naming them here is not enough."""
    declared_fields = {item.name for item in dataclass_fields(TrainingConfig)}
    overlap = NON_CONFIG_DESTS & declared_fields
    assert not overlap, (
        f"{sorted(overlap)} is listed as a non-config dest but IS a config field"
    )
    dests = {
        action.dest for action in build_parser()._actions if action.option_strings
    }
    unaccounted = dests - set(CLI_TO_CONFIG) - NON_CONFIG_DESTS
    assert not unaccounted, f"argparse dests accounted for nowhere: {sorted(unaccounted)}"


# ---------------------------------------------------------------------
# the --smoke preset
# ---------------------------------------------------------------------


def test_every_smoke_preset_key_is_a_config_field():
    """A preset key that is not a field is a setting that silently evaporates."""
    declared_fields = {item.name for item in dataclass_fields(TrainingConfig)}
    unknown = set(SMOKE_PRESET) - declared_fields
    assert not unknown, f"SMOKE_PRESET names non-fields: {sorted(unknown)}"


def test_the_smoke_preset_applies():
    """``--smoke`` alone must move every preset field onto its preset value."""
    config = config_from_argv(["--smoke"])
    for field_name, expected in SMOKE_PRESET.items():
        assert getattr(config, field_name) == expected, (
            f"--smoke left {field_name} at {getattr(config, field_name)!r}, "
            f"expected {expected!r}"
        )


@pytest.mark.parametrize("field_name", sorted(SMOKE_PRESET))
def test_a_typed_flag_wins_over_the_preset_even_at_its_own_default(field_name):
    """PROVENANCE, not value comparison.

    A flag typed AT its own parser default is indistinguishable from an omission
    in the parsed ``Namespace``; only a raw token scan can tell them apart. This
    arm drives ``--smoke`` together with the flag typed at its OWN DEFAULT and
    asserts the default survives -- so a regression to a value-vs-default
    provenance test fails here.
    """
    actions = _actions_by_dest()
    dest = next(d for d, f in CLI_TO_CONFIG.items() if f == field_name)
    action = actions[dest]
    default = getattr(TrainingConfig(), field_name)
    if default == SMOKE_PRESET[field_name]:
        pytest.skip(
            f"{field_name}'s default already equals its preset value; the arm "
            "cannot distinguish the two"
        )
    config = config_from_argv(["--smoke", *_argv_for(action, default)])
    assert getattr(config, field_name) == default, (
        f"--{dest.replace('_', '-')} typed at its own default ({default!r}) "
        f"lost to the smoke preset ({SMOKE_PRESET[field_name]!r})"
    )


# ---------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------


def test_help_exits_zero_and_allocates_nothing(monkeypatch, capsys):
    """``--help`` prints ``usage:`` and never reaches a GPU, a model or a dataset."""
    reached: List[str] = []

    def sentinel(name):
        def _raise(*args, **kwargs):
            reached.append(name)
            raise AssertionError(f"--help reached {name!r}")

        return _raise

    for name in ("setup_gpu", "set_seeds", "create_model",
                 "build_bridge_dataset", "train_bit_diffusion"):
        monkeypatch.setattr(trainer, name, sentinel(name))
    monkeypatch.setattr(sys, "argv", ["train_bit_diffusion.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        trainer.main()

    assert excinfo.value.code == 0
    assert not reached, (
        f"--help reached {reached!r} before argparse could exit. "
        "`args, config = parse_arguments(argv)` must be the FIRST statement of "
        "main(), above GPU setup, model construction and dataset construction."
    )
    printed = capsys.readouterr().out
    assert printed.startswith("usage:"), (
        "--help printed no `usage:` line. An exit code of 0 is not evidence: a "
        "script with no parser at all runs its whole job and exits 0."
    )
    assert "--smoke" in printed and "--direction" in printed
