r"""CLI contract guard for ``src/train/dit/train_dit.py``.

The defect class this exists to catch
-------------------------------------
A trainer declares a flag, ``--help`` advertises it, and ``main()`` never
forwards it. The flag then silently does nothing: no error, no warning, and
``config.json`` records the DEFAULT while the user believes they overrode it.
Step 13 of this port MEASURED that failure mode on this very trainer: with the
``num_timesteps`` row deleted from ``CLI_TO_CONFIG``, ``--num-timesteps 7``
parses with NO error and the config keeps ``1000``.
:class:`TestTheCompletenessAssertionCanFail` reproduces that measurement
executably, so the two completeness assertions below are known to discriminate
rather than merely known to be green.

Reused, not re-implemented
--------------------------
* ``tests/test_train/_cli_contract.py`` -- drives the REAL parser and the REAL
  config builder and asserts on the resulting OBJECT. Its three designed-out
  traps (a row must carry a non-default value; the row table must be complete;
  rows must be driven one at a time) are what make the assertions mean anything.
* ``tests/test_train/test_config_fields_are_live.py`` -- the other end of the
  same contract, asking "does anything READ this field?". ``TrainingConfig`` is
  REGISTERED there, so a field that only reaches ``save_config_json`` fails
  THERE rather than needing a weaker copy here.

THE ROW TABLE IS GENERATED, NOT TYPED. A hand-written table of 31 rows is 31
chances to type a probe value that is already the default (trap 1), and it goes
stale the day a flag is added. :func:`_rows` derives each row from the REAL
parser action plus the REAL dataclass default. Two fields carry an explicit
override because the generic ``default + offset`` rule produces a value
``TrainingConfig.__post_init__`` rightly REJECTS -- ``input_size`` must stay a
multiple of the variant's patch size and ``ema_decay`` must stay in ``[0, 1]``.
The shared ``assert_row_value_is_not_the_default`` then MEASURES that every
generated probe -- override or not -- differs from the default, so neither the
generator nor the override table is trusted.

``--help`` MUST ALLOCATE NOTHING. Sentinels are installed over ``setup_gpu``,
``set_seeds``, ``create_model``, ``create_datasets`` and ``train_dit``; each
raises on contact and each is asserted to have been called ZERO times. Moving
``parse_arguments()`` below any of them fails by the named ``--help reached ...``
message, not by the exit code -- so this distinguishes "parsed FIRST" from
"parsed at all". Asserting only ``exit == 0`` is measurably weaker: a script
with no parser at all runs its whole job and exits 0.

Nothing here trains, builds a model, allocates a GPU, or writes into the
repo-root ``results/``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Mapping, Set, Tuple

import pytest

from train.common.args import config_values_from_args
from train.dit import train_dit as trainer
from train.dit.train_dit import (
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

#: Fields whose generic probe would be REJECTED by ``__post_init__``, with the
#: reason. These are validation facts, not preferences: ``32 + 7 = 39`` is not a
#: multiple of ``DiT-S/2``'s patch size 2, and ``0.9999 + 0.375`` leaves ``[0, 1]``.
#: Each override is still checked against the default by trap 1.
PROBE_OVERRIDES: Dict[str, Any] = {
    "input_size": 16,   # a legal latent grid that is not the default 32
    "ema_decay": 0.5,   # inside [0, 1] and far from the 0.9999 default
}


def _probe_value(action: argparse.Action, field_name: str, default: Any) -> Any:
    """A value for ``action`` that is guaranteed to differ from ``default``.

    Interface contract: pure. Reads only :data:`PROBE_OVERRIDES`, the action's
    declared ``choices`` / ``type`` and the dataclass default; never parses
    anything and never constructs a config.

    :param action: The real argparse action.
    :param field_name: The ``TrainingConfig`` field the action writes.
    :param default: The dataclass default for that field.
    :return: The probe value.
    :raises ValueError: If the action declares a single-element ``choices``, in
        which case no differing value exists and the row would be vacuous.
    """
    if field_name in PROBE_OVERRIDES:
        return PROBE_OVERRIDES[field_name]
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
    """One generated :class:`Row` per ``CLI_TO_CONFIG`` entry, plus ``--gpu``."""
    actions = _actions_by_dest()
    defaults = TrainingConfig()
    rows: List[Row] = []
    for dest, field_name in CLI_TO_CONFIG.items():
        action = actions[dest]
        value = _probe_value(action, field_name, getattr(defaults, field_name))
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
    name="train_dit.py",
    build_parser=lambda monkeypatch: build_parser(),
    build_config=lambda monkeypatch: config_from_argv(None),
    modes=(Mode(id="default", required_argv=(), rows=_rows()),),
)

_CASES, _IDS = cases([CONTRACT])


# ---------------------------------------------------------------------
# the two completeness predicates, as PURE functions
# ---------------------------------------------------------------------


def _flag_gaps(cli_map: Mapping[str, str]) -> Tuple[Set[str], Set[str]]:
    """``(argparse dests with no row, rows naming no dest)``.

    Interface contract: pure over the REAL parser. Extracted from the test body
    so :class:`TestTheCompletenessAssertionCanFail` can drive the identical
    predicate against a deliberately broken map -- a predicate exercised only
    on the passing input is not known to discriminate.

    :param cli_map: A candidate ``CLI_TO_CONFIG``.
    :return: ``(missing_rows, stale_rows)``.
    """
    dests = {
        action.dest for action in build_parser()._actions if action.option_strings
    }
    dests -= NON_CONFIG_DESTS
    return dests - set(cli_map), set(cli_map) - dests


def _field_gaps(cli_map: Mapping[str, str]) -> Tuple[Set[str], Set[str]]:
    """``(TrainingConfig fields with no flag, rows naming no field)``.

    Interface contract: pure. Same extraction rationale as :func:`_flag_gaps`.

    :param cli_map: A candidate ``CLI_TO_CONFIG``.
    :return: ``(missing_flags, unknown_fields)``.
    """
    declared = {item.name for item in dataclass_fields(TrainingConfig)}
    wired = set(cli_map.values())
    return declared - wired, wired - declared


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


class TestBothDirections:
    """Every flag has a row, AND every field has a flag."""

    def test_every_declared_dest_has_a_cli_to_config_row(self):
        missing_rows, stale_rows = _flag_gaps(CLI_TO_CONFIG)
        assert not missing_rows and not stale_rows, (
            f"argparse dests with no CLI_TO_CONFIG row: {sorted(missing_rows)}; "
            f"rows naming no argparse dest: {sorted(stale_rows)}"
        )

    def test_every_config_field_has_a_cli_flag(self):
        missing_flags, unknown_fields = _field_gaps(CLI_TO_CONFIG)
        assert not missing_flags and not unknown_fields, (
            f"TrainingConfig fields with no flag: {sorted(missing_flags)}; "
            f"CLI_TO_CONFIG rows naming no field: {sorted(unknown_fields)}"
        )

    def test_the_non_config_dests_really_are_not_config_fields(self):
        """``--gpu`` and ``-h`` act on the process; naming them here is not enough."""
        declared = {item.name for item in dataclass_fields(TrainingConfig)}
        overlap = NON_CONFIG_DESTS & declared
        assert not overlap, (
            f"{sorted(overlap)} is listed as a non-config dest but IS a "
            "config field"
        )

    def test_the_exemption_list_is_the_only_exemption(self):
        """``NON_CONFIG_DESTS`` is the ONLY permitted way out of the contract."""
        dests = {
            action.dest
            for action in build_parser()._actions
            if action.option_strings
        }
        unaccounted = dests - set(CLI_TO_CONFIG) - NON_CONFIG_DESTS
        assert not unaccounted, (
            f"argparse dests accounted for nowhere: {sorted(unaccounted)}"
        )
        unused_exemptions = NON_CONFIG_DESTS - dests
        assert not unused_exemptions, (
            f"NON_CONFIG_DESTS exempts dests the parser does not declare: "
            f"{sorted(unused_exemptions)} -- a stale exemption silently widens "
            "the carve-out"
        )


# ---------------------------------------------------------------------
# ANTI-VACUITY: the completeness assertion is known to FAIL
# ---------------------------------------------------------------------


class TestTheCompletenessAssertionCanFail:
    """Step 13's measurement, formalised.

    A guard that has never been seen red is not known to work. These arms drive
    the SAME predicates and the SAME real merge function against a
    ``CLI_TO_CONFIG`` with one row deleted, and assert that the deletion is
    (a) named in BOTH directions and (b) otherwise SILENT at runtime -- which is
    exactly why the predicates exist.
    """

    DROPPED = "num_timesteps"

    @property
    def broken_map(self) -> Dict[str, str]:
        return {k: v for k, v in CLI_TO_CONFIG.items() if k != self.DROPPED}

    def test_a_dropped_row_is_named_in_both_directions(self):
        missing_rows, stale_rows = _flag_gaps(self.broken_map)
        missing_flags, unknown_fields = _field_gaps(self.broken_map)
        assert missing_rows == {self.DROPPED}, missing_rows
        assert missing_flags == {self.DROPPED}, missing_flags
        assert not stale_rows and not unknown_fields

    def test_the_dropped_row_is_otherwise_completely_silent(self):
        """The runtime consequence the guard exists to convert into a failure.

        With the row gone, ``--num-timesteps 7`` parses without error and the
        value simply never becomes a config value: the config keeps its default
        ``1000`` while the user believes the chain is 7 steps long. Nothing
        raises, nothing warns, and ``config.json`` records ``1000``.
        """
        parser = build_parser()
        _, values = config_values_from_args(
            parser, ["--num-timesteps", "7"], self.broken_map, SMOKE_PRESET
        )
        assert self.DROPPED not in values
        config = TrainingConfig(**values)
        assert config.num_timesteps == TrainingConfig().num_timesteps == 1000

        # The control: with the REAL map the same argv arrives.
        _, good = config_values_from_args(
            parser, ["--num-timesteps", "7"], CLI_TO_CONFIG, SMOKE_PRESET
        )
        assert good[self.DROPPED] == 7


# ---------------------------------------------------------------------
# the --smoke preset
# ---------------------------------------------------------------------


def test_every_smoke_preset_key_is_a_config_field():
    """A preset key that is not a field is a setting that silently evaporates."""
    declared = {item.name for item in dataclass_fields(TrainingConfig)}
    unknown = set(SMOKE_PRESET) - declared
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
                 "create_datasets", "train_dit"):
        monkeypatch.setattr(trainer, name, sentinel(name))
    monkeypatch.setattr(sys, "argv", ["train_dit.py", "--help"])

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
    assert "--smoke" in printed and "--num-timesteps" in printed
