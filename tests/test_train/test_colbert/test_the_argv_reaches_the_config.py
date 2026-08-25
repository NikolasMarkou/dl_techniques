"""Every ColBERT trainer flag must reach the config through the REAL parse path.

The defect class
----------------

A ``main()`` that lists its config fields explicitly turns an omitted CLI arg
into a **silent no-op**: the flag parses, ``--help`` still advertises it, the run
proceeds, and the value is ignored. Both ColBERT trainers avoid the shape by
forwarding every mapped ``dest`` in a loop inside
:func:`train.common.args.config_values_from_args` -- but "avoids the shape" is a
claim about today's source, not a guard. This module is the guard.

What is driven
--------------

The trainer's own ``main()``, with ``setup_gpu`` and the training function
replaced, so the config asserted on is the object ``main()`` would have handed
to training. Nothing here rebuilds the parser or re-implements the merge: a test
that built its own parser would measure a different option table than the one
that ships, and a test that re-implemented the ``args -> config`` hop would pass
against a ``main()`` that had lost that hop entirely.

The three vacuity traps (non-default probe values, table completeness read off
the real parser, one row at a time so a cross-wire is visible) are implemented
once in ``tests/test_train/_cli_contract.py`` and reused here rather than
restated -- see that module's docstring for why each exists.

The two parsers are NOT the same option table
---------------------------------------------

``add_common_arguments`` reads every default off :class:`TrainingConfig`, and
``train_colbert_v2.build_parser`` then calls ``parser.set_defaults(nway=4)``.
So ``--nway``'s default is **2** for v1 and **4** for v2, from one shared
registration. That divergence is pinned by name below
(:func:`test_the_two_scripts_disagree_on_the_nway_default`) instead of being
assumed away; the shared probe row uses ``3``, which is a non-default for both.
"""

from __future__ import annotations

import sys
from typing import Any, Tuple

import pytest

from train.language.colbert import train_colbert_v1 as v1
from train.language.colbert import train_colbert_v2 as v2
from train.language.colbert.common import SMOKE_PRESET, TrainingConfig

from .._cli_contract import (
    Contract,
    Mode,
    Row,
    assert_row_reaches_config,
    assert_row_value_is_not_the_default,
    cases,
    declared_option_strings,
)
from ..test_sam3.parser_help_guard import assert_no_bare_percent_help


# ---------------------------------------------------------------------
# Driving the real `main()`
# ---------------------------------------------------------------------


def _make_build_config(module: Any, train_attr: str):
    """Return a ``(monkeypatch) -> TrainingConfig`` driver for one trainer.

    Interface contract (2 callers: the v1 and the v2 :class:`Contract`, plus the
    ``--smoke`` section through :func:`_config_from_argv`):

    :param module: The trainer module, whose ``main()`` is executed.
    :param train_attr: Name of the training function attribute on ``module``
        that ``main()`` calls; it is replaced by a recorder.
    :returns: A callable that reads ``sys.argv`` (already set by the caller),
        runs ``main()`` and returns the config it built.
    :raises AssertionError: If ``main()`` returned without calling the training
        function -- which would mean the config never existed.
    """

    def build_config(monkeypatch) -> TrainingConfig:
        captured = []
        monkeypatch.setattr(module, "setup_gpu", lambda **kwargs: None)
        monkeypatch.setattr(module, train_attr, captured.append)
        module.main()
        assert captured, (
            f"{module.__name__}.main() never called {train_attr}; no config "
            f"was built, so nothing here measures the argv -> config path"
        )
        return captured[0]

    return build_config


_BUILD_V1 = _make_build_config(v1, "train_colbert_v1")
_BUILD_V2 = _make_build_config(v2, "train_colbert_v2")


def _config_from_argv(monkeypatch, module: Any, argv) -> TrainingConfig:
    """Resolve ``argv`` into a config through ``module.main()``."""
    build = _BUILD_V1 if module is v1 else _BUILD_V2
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    return build(monkeypatch)


# ---------------------------------------------------------------------
# The flag table
# ---------------------------------------------------------------------

#: Every flag ``add_common_arguments`` registers, with a probe value that is a
#: non-default for BOTH scripts. Shared by the two contracts because the two
#: parsers share the registration function -- a private copy per script is how
#: they would drift apart unnoticed.
COLBERT_COMMON_ROWS: Tuple[Row, ...] = (
    Row(("--gpu",), ("--gpu", "1"), namespace_dest="gpu", expected=1),
    Row(("--variant",), ("--variant", "small"), "colbert_variant", "small"),
    Row(("--vocab-size",), ("--vocab-size", "4096"), "vocab_size", 4096),
    Row(("--dim",), ("--dim", "48"), "dim", 48),
    Row(("--query-maxlen",), ("--query-maxlen", "12"), "query_maxlen", 12),
    Row(("--doc-maxlen",), ("--doc-maxlen", "96"), "doc_maxlen", 96),
    Row(("--nway",), ("--nway", "3"), "nway", 3),
    Row(("--batch-size",), ("--batch-size", "7"), "batch_size", 7),
    Row(("--epochs",), ("--epochs", "11"), "epochs", 11),
    Row(("--learning-rate",), ("--learning-rate", "1.5e-4"), "learning_rate", 1.5e-4),
    Row(("--warmup-epochs",), ("--warmup-epochs", "5"), "warmup_epochs", 5),
    Row(("--weight-decay",), ("--weight-decay", "0.07"), "weight_decay", 0.07),
    Row(
        ("--gradient-clipping",),
        ("--gradient-clipping", "0.25"),
        "gradient_clipping",
        0.25,
    ),
    Row(("--optimizer-type",), ("--optimizer-type", "adam"), "optimizer_type", "adam"),
    Row(
        ("--lr-schedule-type",),
        ("--lr-schedule-type", "exponential_decay"),
        "lr_schedule_type",
        "exponential_decay",
    ),
    Row(("--patience",), ("--patience", "9"), "patience", 9),
    Row(("--seed",), ("--seed", "1234"), "seed", 1234),
    Row(
        ("--num-train-groups",),
        ("--num-train-groups", "17"),
        "num_train_groups",
        17,
    ),
    Row(("--num-val-groups",), ("--num-val-groups", "13"), "num_val_groups", 13),
    Row(("--query-words",), ("--query-words", "6"), "query_words", 6),
    Row(("--doc-words",), ("--doc-words", "33"), "doc_words", 33),
    Row(("--output-root",), ("--output-root", "/probe/out"), "output_root", "/probe/out"),
    Row(
        ("--results-dir-prefix",),
        ("--results-dir-prefix", "probe_prefix"),
        "results_dir_prefix",
        "probe_prefix",
    ),
    Row(("--smoke",), ("--smoke",), "smoke", True),
)

#: v2's only additional flag. v1 must NOT declare it -- pinned by the
#: completeness test, which reads the real parser.
V2_ONLY_ROWS: Tuple[Row, ...] = (
    Row(
        ("--distillation-alpha",),
        ("--distillation-alpha", "0.35"),
        "distillation_alpha",
        0.35,
    ),
)

V1_CONTRACT = Contract(
    name="train.language.colbert.train_colbert_v1",
    build_parser=lambda monkeypatch: v1.build_parser(),
    build_config=_BUILD_V1,
    modes=(Mode("plain", (), COLBERT_COMMON_ROWS),),
)

V2_CONTRACT = Contract(
    name="train.language.colbert.train_colbert_v2",
    build_parser=lambda monkeypatch: v2.build_parser(),
    build_config=_BUILD_V2,
    modes=(Mode("plain", (), COLBERT_COMMON_ROWS + V2_ONLY_ROWS),),
)

CONTRACTS = (V1_CONTRACT, V2_CONTRACT)

_CASES, _IDS = cases(CONTRACTS)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_flag_reaches_its_config_field(monkeypatch, contract, mode, row) -> None:
    """THE wiring assertion: the typed value arrives where the row says."""
    assert_row_reaches_config(monkeypatch, contract, mode, row)


@pytest.mark.parametrize("contract,mode,row", _CASES, ids=_IDS)
def test_probe_value_differs_from_the_default(monkeypatch, contract, mode, row) -> None:
    """Anti-vacuity: a default-valued probe passes whether or not it is wired."""
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


@pytest.mark.parametrize("contract", CONTRACTS, ids=[c.name for c in CONTRACTS])
def test_help_strings_carry_no_bare_percent(monkeypatch, contract) -> None:
    """A lone ``%`` in a help string crashes ``--help`` and only ``--help``."""
    assert_no_bare_percent_help(contract.build_parser(monkeypatch), contract.name)


# ---------------------------------------------------------------------
# The two parsers are one registration with one deliberate divergence
# ---------------------------------------------------------------------


def test_the_two_scripts_disagree_on_the_nway_default() -> None:
    """``parser.set_defaults(nway=...)`` in v2, over the shared registration.

    A field-by-field argv diff that assumed one default set for both scripts
    would either miss this or falsely report it as a defect. It is neither: v1
    trains on a triple, v2 on a listwise tuple.
    """
    assert v1.build_parser().get_default("nway") == TrainingConfig.nway == 2
    assert v2.build_parser().get_default("nway") == v2.V2_DEFAULT_NWAY == 4


def test_every_other_default_is_shared_between_the_two_parsers() -> None:
    """The divergence is exactly one dest wide -- measured, not asserted.

    Reads both real parsers' defaults for every dest v1 declares. ``nway`` is
    the single documented exception; anything else appearing here means a
    default drifted onto one script only, which is invisible at ``--help``.
    """
    v1_parser, v2_parser = v1.build_parser(), v2.build_parser()
    v1_dests = {a.dest for a in v1_parser._actions} - {"help"}
    differing = {
        dest: (v1_parser.get_default(dest), v2_parser.get_default(dest))
        for dest in sorted(v1_dests)
        if v1_parser.get_default(dest) != v2_parser.get_default(dest)
    }
    assert set(differing) == {"nway"}, (
        f"v1 and v2 defaults diverge on {sorted(differing)} ({differing}); only "
        f"'nway' is a documented divergence"
    )


def test_only_v2_declares_the_distillation_flag() -> None:
    """``distillation_alpha`` is read by the v2 recipe and by nothing in v1."""
    assert "--distillation-alpha" in declared_option_strings(v2.build_parser())
    assert "--distillation-alpha" not in declared_option_strings(v1.build_parser())


def test_the_v2_wiring_table_extends_the_shared_one() -> None:
    """v2's table is built by extension, so a shared field reaches both."""
    from train.language.colbert.common import CLI_TO_CONFIG

    assert set(v2.V2_CLI_TO_CONFIG) == set(CLI_TO_CONFIG) | {"distillation_alpha"}


def test_every_wiring_table_entry_names_a_real_config_field() -> None:
    """A renamed field would otherwise raise only at run time, inside a run."""
    known = set(TrainingConfig.field_names())
    unknown = sorted(set(v2.V2_CLI_TO_CONFIG.values()) - known)
    assert not unknown, f"wiring table targets non-existent fields {unknown}"


def test_gpu_never_reaches_the_config(monkeypatch) -> None:
    """``--gpu`` is a process-level concern; a ``gpu`` field would be dead."""
    config = _config_from_argv(monkeypatch, v1, ["--gpu", "1"])
    assert not hasattr(config, "gpu")
    assert "gpu" not in set(TrainingConfig.field_names())


# ---------------------------------------------------------------------
# `--smoke`
# ---------------------------------------------------------------------

#: Preset keys whose value EQUALS the class default, so ``--smoke`` cannot be
#: observed to move them. MEASURED, not assumed: ``SMOKE_PRESET`` pins
#: ``colbert_variant="tiny"`` and ``TrainingConfig.colbert_variant`` is already
#: ``"tiny"``. The pin is deliberate -- it keeps a smoke run on the cheap
#: backbone if the class default ever moves up -- but it makes that one key's
#: "the preset moved it" arm vacuous, so the vacuity is named here rather than
#: silently tolerated, and :func:`test_the_preset_keys_that_pin_the_default_are_exactly_these`
#: fails if the set changes in either direction.
# DECISION plan-2026-08-25T121346-c71fc3ad/D-027
# This exemption is a NAMED VACUITY, not a suppression. WHAT NOT TO DO:
# (1) do NOT delete `colbert_variant` from SMOKE_PRESET to make this set empty
#     -- the key is a deliberate pin that keeps a smoke run on the cheap
#     backbone if `TrainingConfig.colbert_variant` ever moves up, and deleting
#     it trades a vacuous test arm for a real regression risk;
# (2) do NOT widen this set to silence a future failure of
#     `test_the_preset_keys_that_pin_the_default_are_exactly_these` -- a key
#     that BECOMES inert means the preset stopped reducing work, which is the
#     defect, not the test;
# (3) do NOT drop the set-equality assertion for a subset one: it is what makes
#     a newly-inert key fail instead of being absorbed.
# See decisions.md D-027.
PRESET_KEYS_THAT_PIN_THE_DEFAULT = frozenset({"colbert_variant"})


def test_the_preset_keys_that_pin_the_default_are_exactly_these() -> None:
    """Which preset keys are observable, pinned as a fact."""
    inert = {
        field
        for field, value in SMOKE_PRESET.items()
        if getattr(TrainingConfig, field) == value
    }
    assert inert == PRESET_KEYS_THAT_PIN_THE_DEFAULT, (
        f"preset keys equal to their class default are {sorted(inert)}, "
        f"expected {sorted(PRESET_KEYS_THAT_PIN_THE_DEFAULT)}. A key that "
        f"became inert has a vacuous guard; a key that became observable "
        f"should be dropped from this exemption."
    )


def test_the_preset_actually_shrinks_the_work() -> None:
    """An EMPTY preset would make every per-key arm below vacuous.

    MEASURED necessity, not caution: emptying ``SMOKE_PRESET`` leaves
    :func:`test_smoke_moves_every_observable_preset_field` iterating over
    nothing and passing. The three fields named here are the ones that decide
    how long a smoke run takes.
    """
    assert SMOKE_PRESET, "SMOKE_PRESET is empty; --smoke reduces nothing"
    for field in ("epochs", "num_train_groups", "num_val_groups"):
        assert field in SMOKE_PRESET, f"--smoke does not shrink {field}"
        assert SMOKE_PRESET[field] < getattr(TrainingConfig, field), (
            f"--smoke sets {field}={SMOKE_PRESET[field]!r}, which is not less "
            f"than the default {getattr(TrainingConfig, field)!r}"
        )


def test_the_preset_keys_are_real_config_fields() -> None:
    unknown = sorted(set(SMOKE_PRESET) - set(TrainingConfig.field_names()))
    assert not unknown, f"SMOKE_PRESET names non-fields {unknown}"


@pytest.mark.parametrize("module", [v1, v2], ids=["v1", "v2"])
def test_smoke_moves_every_observable_preset_field(monkeypatch, module) -> None:
    """``--smoke`` must actually reduce the work, field by field."""
    smoke = _config_from_argv(monkeypatch, module, ["--smoke"])
    for field, value in SMOKE_PRESET.items():
        actual = getattr(smoke, field)
        assert actual == value, (
            f"{module.__name__}: --smoke left {field}={actual!r}, "
            f"SMOKE_PRESET declares {value!r}"
        )
    observable = set(SMOKE_PRESET) - PRESET_KEYS_THAT_PIN_THE_DEFAULT
    for field in sorted(observable):
        assert getattr(smoke, field) != getattr(TrainingConfig, field), (
            f"{module.__name__}: --smoke did not move {field} off its default "
            f"{getattr(TrainingConfig, field)!r}; the preset is inert here"
        )


@pytest.mark.parametrize("module", [v1, v2], ids=["v1", "v2"])
def test_every_field_the_preset_does_not_name_is_bit_identical(
    monkeypatch, module
) -> None:
    """A preset may change how MUCH is measured, never WHAT.

    A set operation over every declared field, not a hand-listed subset, so a
    field added to :class:`TrainingConfig` later is covered automatically.
    """
    base = _config_from_argv(monkeypatch, module, [])
    smoke = _config_from_argv(monkeypatch, module, ["--smoke"])
    untouched = set(TrainingConfig.field_names()) - set(SMOKE_PRESET) - {"smoke"}
    for field in sorted(untouched):
        expected, actual = getattr(base, field), getattr(smoke, field)
        assert type(expected) is type(actual) and expected == actual, (
            f"{module.__name__}: --smoke moved {field}: {expected!r} -> "
            f"{actual!r}. A preset may change how much is measured, never what."
        )


@pytest.mark.parametrize("module", [v1, v2], ids=["v1", "v2"])
def test_the_preset_never_touches_a_field_that_shapes_the_run(module) -> None:
    """The same claim, enumerated from the SHAPING side.

    ``nway`` is in this set deliberately: it is the divisor both losses reshape
    by, so moving it would change the objective a smoke run is meant to be a
    wiring proof for -- and it is precisely the field v2 overrides.
    """
    shaping = {
        "vocab_size", "nway", "distillation_alpha", "learning_rate",
        "weight_decay", "gradient_clipping", "optimizer_type",
        "lr_schedule_type", "seed", "query_words", "doc_words",
    }
    offenders = sorted(shaping & set(SMOKE_PRESET))
    assert not offenders, (
        f"SMOKE_PRESET declares {offenders}, which change WHAT is measured. "
        f"A smoke preset may only change how much."
    )


@pytest.mark.parametrize("module", [v1, v2], ids=["v1", "v2"])
def test_an_explicitly_typed_flag_beats_the_preset(monkeypatch, module) -> None:
    config = _config_from_argv(monkeypatch, module, ["--smoke", "--epochs", "11"])
    assert config.epochs == 11 and config.smoke is True


@pytest.mark.parametrize("module", [v1, v2], ids=["v1", "v2"])
def test_every_preset_field_can_be_typed_at_its_own_default_and_win(
    monkeypatch, module
) -> None:
    """The provenance property, across the whole observable preset.

    A flag typed at its own parser default is indistinguishable from an
    omission in the Namespace, so a value-vs-default implementation silently
    overrides it. One field alone would leave the rest unproved.
    """
    for field in sorted(set(SMOKE_PRESET) - PRESET_KEYS_THAT_PIN_THE_DEFAULT):
        default_value = getattr(TrainingConfig, field)
        flag = "--" + field.replace("_", "-")
        config = _config_from_argv(
            monkeypatch, module, ["--smoke", flag, str(default_value)]
        )
        assert getattr(config, field) == default_value, (
            f"{module.__name__}: {flag} typed at its own default "
            f"({default_value!r}) lost to the preset; provenance is being "
            f"computed by VALUE, not by whether the token was typed"
        )
