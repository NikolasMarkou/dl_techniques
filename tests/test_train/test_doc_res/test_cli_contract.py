r"""CLI contract guard for ``src/train/doc_res/train_doc_res.py``.

The defect class this exists to catch
-------------------------------------
A trainer declares a flag, ``--help`` advertises it, and ``main()`` never
forwards it. The flag then silently does nothing: no error, no warning, and
``config.json`` records the DEFAULT while the user believes they overrode it.
:class:`TestTheForwardingAssertionCanFail` reproduces that failure mode
executably against a deliberately crippled ``config_from_args``, so the
assertions below are known to DISCRIMINATE rather than merely known to be green.

Reused, not re-implemented
--------------------------
* ``tests/test_train/_cli_contract.py`` -- the shared driver. It runs the REAL
  parser and the REAL config builder and asserts on the resulting OBJECT, and
  its three designed-out traps (a row must carry a non-default value; the row
  table must be complete; rows are driven ONE AT A TIME so a cross-wire leaves
  the intended field at its default) are what make these rows mean anything.
* ``tests/test_train/test_config_fields_are_live.py`` -- the other end of the
  same contract, asking "does anything READ this field?".
  ``DocResTrainingConfig`` is REGISTERED there (added in step 11), so a field
  that only reaches ``save_config_json`` fails THERE and needs no weaker copy
  here.

THE ROW TABLE IS GENERATED, NOT TYPED. A hand-typed table is one chance per row
to pick a probe value that is already the default (trap 1), and it goes stale
the day a flag is added. :func:`_rows` derives every row from the REAL parser
action plus the REAL dataclass default. Two fields carry an explicit override
because the generic ``default + offset`` rule produces a value
``DocResTrainingConfig.__post_init__`` rightly REJECTS -- see
:data:`PROBE_OVERRIDES`. The shared ``assert_row_value_is_not_the_default`` then
MEASURES that every probe, generated or overridden, differs from the default, so
neither the generator nor the override table is trusted.

``--help`` MUST ALLOCATE NOTHING, and the CORPUS CHECK MUST COME FIRST.
Sentinels are installed over ``setup_gpu``, ``require_task_data`` and ``train``
on the trainer module and over ``set_seeds``, ``build_model`` and
``create_dataset`` on ``common``; each records contact.
:func:`test_help_prints_usage_and_allocates_nothing` asserts every one of them
was reached ZERO times and that stdout starts with ``usage:``. Asserting only
``exit == 0`` is measurably weaker -- ``src/train/CLAUDE.md`` records that a
script with no parser at all ignores ``--help``, runs its whole job and exits 0.
:class:`TestStartupOrder` then drives a REAL ``main()`` at a task with no staged
corpus and asserts the ``MissingTrainingDataError`` arrives with the GPU and the
model sentinels still untouched -- an ordering claim measured through behaviour,
never by reading the source.

Nothing here trains, builds a model, allocates a GPU, downloads anything, or
writes into the repo-root ``results/``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any, Dict, List, Set, Tuple

import pytest

from train.doc_res import common
from train.doc_res import train_doc_res as trainer
from train.doc_res.common import DocResTrainingConfig
from train.doc_res.prepare_doc_res_data import MissingTrainingDataError
from train.doc_res.train_doc_res import (
    NON_CONFIG_DESTS,
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

#: Displacement applied to a scalar default to build a probe value. Any nonzero
#: value works; ``0.375`` is exactly representable in binary floating point, so
#: ``float(str(default + 0.375))`` round-trips exactly and the equality
#: assertion cannot fail on a formatting artefact.
FLOAT_OFFSET = 0.375
INT_OFFSET = 7

#: Fields whose generic probe would be REJECTED by ``__post_init__``, with the
#: reason. These are validation facts, not preferences:
#:
#: * ``256 + 7 = 263`` is not a multiple of ``SPATIAL_DIVISOR`` (8), and DocRes
#:   downsamples three times;
#: * ``1e-6 + 0.375`` exceeds the ``learning_rate`` a row driven ALONE leaves at
#:   its ``2e-4`` default, and the config requires
#:   ``0 < final_learning_rate <= learning_rate``.
#:
#: Each override is still checked against the default by trap 1, so an override
#: that accidentally equalled the default would fail rather than pass quietly.
PROBE_OVERRIDES: Dict[str, Any] = {
    "patch_size": 128,           # a legal multiple of 8 that is not 256
    "final_learning_rate": 5e-5,  # positive and below the 2e-4 default peak
}


def _config_fields() -> Set[str]:
    return {item.name for item in dataclass_fields(DocResTrainingConfig)}


def _actions_by_dest() -> Dict[str, argparse.Action]:
    """Every optional action of the REAL parser, keyed by destination."""
    return {
        action.dest: action
        for action in build_parser()._actions
        if action.option_strings and not isinstance(action, argparse._HelpAction)
    }


def _probe_value(action: argparse.Action, field_name: str, default: Any) -> Any:
    """A value for ``action`` guaranteed to differ from ``default``.

    Interface contract: pure. Reads only :data:`PROBE_OVERRIDES`, the action's
    declared ``choices`` / ``type`` and the dataclass default; parses nothing and
    constructs no config.

    :param action: The real argparse action.
    :param field_name: The config field the action writes.
    :param default: The dataclass default for that field.
    :return: The probe value.
    :raises ValueError: If the action declares a single-element ``choices``, in
        which case no differing value exists and the row would be vacuous.
    """
    if field_name in PROBE_OVERRIDES:
        return PROBE_OVERRIDES[field_name]
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


def _rows() -> Tuple[Row, ...]:
    """One generated :class:`Row` per parser dest.

    The row set is derived from the REAL parser's dest set, so a flag added
    without a config field fails the completeness arm instead of being ignored.
    """
    actions = _actions_by_dest()
    defaults = DocResTrainingConfig()
    fields = _config_fields()
    rows: List[Row] = []
    for dest, action in sorted(actions.items()):
        flags = tuple(action.option_strings)
        if dest in NON_CONFIG_DESTS:
            # `--gpu` acts on the process; it must reach the NAMESPACE.
            rows.append(
                Row(flags=flags, argv=("--gpu", "1"),
                    namespace_dest=dest, expected=1)
            )
            continue
        if dest not in fields:
            # No row can be generated for a flag that reaches no field. Do NOT
            # raise here: an exception in the generator is a COLLECTION error
            # that takes all 51 arms down and names the cause only in a
            # traceback. Skipping instead lets
            # `test_every_parser_dest_is_a_config_field_or_an_exemption` and
            # `test_every_declared_flag_has_a_contract_row` both report it as a
            # named failure while the rest of the suite still runs.
            continue
        value = _probe_value(action, dest, getattr(defaults, dest))
        rows.append(
            Row(flags=flags, argv=(action.option_strings[0], str(value)),
                field=dest, expected=value)
        )
    return tuple(rows)


CONTRACT = Contract(
    name="train_doc_res.py",
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
def test_the_probe_value_is_not_already_the_default(monkeypatch, contract, mode, row):
    """Trap 1. Without this, every row above could pass vacuously."""
    assert_row_value_is_not_the_default(monkeypatch, contract, mode, row)


class TestCompleteness:
    """Every flag has a row and a field, and every field has a flag."""

    def test_every_declared_flag_has_a_contract_row(self):
        """Trap 2: a flag added without a row fails HERE, not in production."""
        declared = declared_option_strings(build_parser())
        covered = CONTRACT.covered_flags
        assert declared == covered, (
            f"flags with no contract row: {sorted(declared - covered)}; rows "
            f"for flags that no longer exist: {sorted(covered - declared)}"
        )

    def test_every_parser_dest_is_a_config_field_or_an_exemption(self):
        dests = set(_actions_by_dest())
        unaccounted = dests - _config_fields() - NON_CONFIG_DESTS
        assert not unaccounted, (
            f"parser dests accounted for nowhere: {sorted(unaccounted)} -- "
            "each is a flag that cannot reach the config"
        )

    def test_every_config_field_has_a_cli_flag(self):
        missing = _config_fields() - set(_actions_by_dest())
        assert not missing, (
            f"DocResTrainingConfig fields with no CLI flag: {sorted(missing)}"
        )

    def test_the_exemption_list_is_neither_stale_nor_a_config_field(self):
        """``NON_CONFIG_DESTS`` is the ONLY way out of the contract."""
        overlap = NON_CONFIG_DESTS & _config_fields()
        assert not overlap, (
            f"{sorted(overlap)} is exempted as a non-config dest but IS a "
            "config field"
        )
        stale = NON_CONFIG_DESTS - set(_actions_by_dest())
        assert not stale, (
            f"NON_CONFIG_DESTS exempts dests the parser does not declare: "
            f"{sorted(stale)} -- a stale exemption silently widens the carve-out"
        )


# ---------------------------------------------------------------------
# ANTI-VACUITY: the forwarding assertion is known to FAIL
# ---------------------------------------------------------------------


class TestTheForwardingAssertionCanFail:
    """A guard never seen red is not known to work.

    Drives the identical comparison the parametrised rows make, against a
    ``config_from_args`` with ONE field dropped, and asserts that (a) the row's
    assertion fails and (b) the drop is otherwise completely SILENT at runtime.
    (b) is the reason (a) has to exist.
    """

    DROPPED = "patience"
    PROBE = 22

    def _crippled_config(self, args: argparse.Namespace) -> DocResTrainingConfig:
        """``config_from_args`` as if the author forgot one assignment."""
        full = common.config_from_args(args)
        return replace(full, **{self.DROPPED: getattr(DocResTrainingConfig(), self.DROPPED)})

    def test_a_dropped_forward_is_silent_at_runtime(self):
        args = build_parser().parse_args([f"--{self.DROPPED}", str(self.PROBE)])
        crippled = self._crippled_config(args)

        # Nothing raises, nothing warns: the user asked for 22 and gets 15.
        assert getattr(crippled, self.DROPPED) == getattr(
            DocResTrainingConfig(), self.DROPPED
        )
        assert getattr(crippled, self.DROPPED) != self.PROBE

        # The control: the REAL builder does forward it.
        assert getattr(common.config_from_args(args), self.DROPPED) == self.PROBE

    def test_the_row_assertion_reds_on_that_drop(self, monkeypatch):
        broken = Contract(
            name=CONTRACT.name,
            build_parser=CONTRACT.build_parser,
            build_config=lambda mp: self._crippled_config(
                build_parser().parse_args(sys.argv[1:])
            ),
            modes=CONTRACT.modes,
        )
        row = next(r for r in CONTRACT.modes[0].rows if r.field == self.DROPPED)
        with pytest.raises(AssertionError, match="did NOT reach"):
            assert_row_reaches_config(monkeypatch, broken, broken.modes[0], row)


# ---------------------------------------------------------------------
# a bogus --task
# ---------------------------------------------------------------------


def test_a_bogus_task_is_rejected_and_the_valid_ones_are_named(capsys):
    """A typo'd task must not fall through to an empty-glob mystery."""
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--task", "binarisation"])

    assert excinfo.value.code == 2, (
        f"a bad --task exited {excinfo.value.code!r}; argparse reports an "
        "invalid choice with exit code 2"
    )
    message = capsys.readouterr().err
    assert "binarisation" in message, message
    for name in common.task_names():
        assert name in message, (
            f"the rejection message does not name the valid task {name!r}; "
            f"stderr was {message!r}"
        )


# ---------------------------------------------------------------------
# sentinels: --help allocates nothing, and the corpus check comes first
# ---------------------------------------------------------------------


class _Sentinel:
    """Records contact instead of doing the expensive thing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError(f"sentinel {self.name!r} was called")


#: ``(module, attribute)`` pairs covering every expensive thing a DocRes run
#: does: claiming a GPU, seeding the process, walking the staging volume,
#: building the tf.data pipeline, and constructing/compiling the model.
_EXPENSIVE: Tuple[Tuple[str, str], ...] = (
    ("trainer", "setup_gpu"),
    ("trainer", "require_task_data"),
    ("trainer", "train"),
    ("common", "set_seeds"),
    ("common", "build_model"),
    ("common", "create_dataset"),
)

_MODULES = {"trainer": trainer, "common": common}


def _install_sentinels(monkeypatch) -> Dict[str, _Sentinel]:
    out: Dict[str, _Sentinel] = {}
    for module_key, attribute in _EXPENSIVE:
        name = f"{module_key}.{attribute}"
        sentinel = _Sentinel(name)
        monkeypatch.setattr(_MODULES[module_key], attribute, sentinel)
        out[name] = sentinel
    return out


def test_the_sentinels_cover_names_that_exist(monkeypatch):
    """A sentinel over a misspelled attribute would never fire.

    ``monkeypatch.setattr`` raises on an unknown attribute, so installing them
    IS the check -- this arm states it as a claim rather than leaving it as an
    accident of the other tests.
    """
    sentinels = _install_sentinels(monkeypatch)
    assert len(sentinels) == len(_EXPENSIVE)
    assert all(s.calls == 0 for s in sentinels.values())


def test_help_prints_usage_and_allocates_nothing(monkeypatch, capsys):
    """``--help`` prints ``usage:`` and reaches no GPU, model or dataset."""
    sentinels = _install_sentinels(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train_doc_res.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        trainer.main()

    reached = {name: s.calls for name, s in sentinels.items() if s.calls}
    assert not reached, (
        f"--help reached {reached} before argparse could exit. "
        "`args, config = parse_arguments(argv)` must be the FIRST statement of "
        "main(), above the corpus check, GPU setup and training."
    )
    assert excinfo.value.code == 0, f"--help exited {excinfo.value.code!r}"

    printed = capsys.readouterr().out
    assert printed.startswith("usage:"), (
        "--help printed no `usage:` line. An exit code of 0 is not evidence: a "
        "script with no parser at all runs its whole job and exits 0 anyway. "
        f"stdout was {printed[:200]!r}"
    )
    for flag in ("--task", "--gpu", "--dataset-root", "--patch-size"):
        assert flag in printed, f"--help does not advertise {flag}"


class TestStartupOrder:
    """``require_task_data`` runs BEFORE anything expensive.

    Measured through behaviour: a task with no staged corpus must raise with
    the manifest's real reason while the GPU and model sentinels are still
    untouched. Reading the source for statement order would prove nothing about
    what runs.
    """

    #: Deshadowing's corpus could not be staged (D-027: the RDD Google Drive
    #: folder cannot be paged through by a script), so the manifest carries a
    #: real, named reason for it.
    TASK = "deshadowing"

    def _argv(self, tmp_path) -> List[str]:
        return ["--task", self.TASK, "--dataset-root", str(tmp_path)]

    def test_a_task_with_no_corpus_fails_before_the_gpu_and_the_model(
        self, monkeypatch, tmp_path
    ):
        sentinels = _install_sentinels(monkeypatch)
        # `require_task_data` is the one thing that MUST run, so it keeps its
        # real implementation; everything downstream stays a sentinel.
        monkeypatch.setattr(
            trainer, "require_task_data", common.require_task_data
        )

        with pytest.raises(MissingTrainingDataError) as excinfo:
            trainer.main(self._argv(tmp_path))

        message = str(excinfo.value)
        assert self.TASK in message
        assert "no training corpus is staged" in message, message

        reached = {
            name: s.calls
            for name, s in sentinels.items()
            if s.calls and name != "trainer.require_task_data"
        }
        assert not reached, (
            f"a task with no corpus reached {reached}. The corpus check must "
            "run before setup_gpu, before seeding and before any model or "
            "dataset construction, so the failure costs nothing."
        )

    def test_the_probe_would_otherwise_have_trained(self, monkeypatch, tmp_path):
        """Anti-vacuity: the argv is a REAL run, not one argparse rejects.

        With the corpus check stubbed out, the same argv walks straight on into
        ``setup_gpu``. So the arm above is measuring the check, not a parse
        error and not a missing flag.
        """
        sentinels = _install_sentinels(monkeypatch)
        monkeypatch.setattr(trainer, "require_task_data", lambda *a, **k: ())

        with pytest.raises(AssertionError, match="trainer.setup_gpu"):
            trainer.main(self._argv(tmp_path))

        assert sentinels["trainer.setup_gpu"].calls == 1
        assert sentinels["trainer.train"].calls == 0
