r"""CLI contract guard for BOTH DocScanner trainer entry points.

The defect class this exists to catch
-------------------------------------
A trainer declares a flag, ``--help`` advertises it, and ``main()`` never
forwards it. The flag then silently does nothing: no error, no warning, and
``config.json`` records the DEFAULT while the user believes they overrode it.
:class:`TestTheForwardingAssertionCanFail` reproduces that failure mode
executably against a deliberately crippled ``config_from_args``, so the
assertions below are known to DISCRIMINATE rather than merely known to be green.

TWO ENTRY POINTS, ONE TABLE, AND THE THING THAT MAKES THAT SAFE
---------------------------------------------------------------
``train_doc_scanner_segmenter.py`` and ``train_doc_scanner_rectifier.py`` share
every flag; they differ only in the DEFAULTS bound to them and in the ``stage``
they fix. That is exactly the shape in which one entry point silently inherits
the other's recipe, so the contract is instantiated ONCE PER SCRIPT (each
driving its own real ``build_parser`` and its own real ``config_from_argv``),
and :class:`TestTheTwoStagesAreDistinct` asserts the paper's per-stage numbers
are actually different where the paper says they are -- batch 32 vs 12, warmup
0 vs non-zero -- and that ``stage`` cannot be reached from argv at all.

``--stage`` IS NOT A FLAG, and that is the ONE config field with no CLI flag.
:data:`FIXED_BY_ENTRY_POINT` is that exemption, and it is checked from both
ends: the field must not be settable from argv, and each script must bind a
different value to it.

``--help`` MUST ALLOCATE NOTHING, and the CORPUS GATE MUST COME FIRST.
Sentinels are installed over ``setup_gpu``, ``require_training_data`` and
``train`` on each trainer module and over ``set_seeds``, ``build_model`` and
``create_dataset`` on ``common``; each records contact.
:func:`test_help_prints_usage_and_allocates_nothing` asserts every one of them
was reached ZERO times and that stdout starts with ``usage:``. Asserting only
``exit == 0`` is measurably weaker -- ``src/train/CLAUDE.md`` records that a
script with no parser at all ignores ``--help``, runs its whole job and exits 0.
:class:`TestStartupOrder` then drives a REAL ``main()`` at a corpus root that
does not exist and asserts the ``MissingTrainingDataError`` arrives with the GPU
and the model sentinels still untouched -- an ordering claim measured through
behaviour, never by reading the source.

Nothing here trains, builds a model, allocates a GPU, downloads anything, or
writes into the repo-root ``results/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any, Callable, Dict, List, Set, Tuple

import pytest

from train.doc_scanner import common
from train.doc_scanner import train_doc_scanner_rectifier as rectifier_cli
from train.doc_scanner import train_doc_scanner_segmenter as segmenter_cli
from train.doc_scanner.common import (
    STAGE_RECTIFIER,
    STAGE_SEGMENTER,
    DocScannerTrainingConfig,
    MissingTrainingDataError,
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

#: The ONE config field with no CLI flag. The stage is fixed by which script you
#: ran; a ``--stage`` flag would let ``train_doc_scanner_segmenter.py --stage
#: rectifier`` train the other module under this script's name, recipe and
#: run-directory prefix. Checked from both ends by
#: :class:`TestTheTwoStagesAreDistinct`.
FIXED_BY_ENTRY_POINT: Set[str] = {"stage"}

#: Fields whose generic ``default + offset`` probe would be REJECTED by
#: ``__post_init__``. These are validation facts, not preferences:
#:
#: * ``288 + 7 = 295`` is not a multiple of 32, and the segmenter pools five
#:   times;
#: * ``0.1 + 0.375`` is a legal ``lr_drop_factor`` but ``1.0 + 0.375`` is not,
#:   and the two stages bind different defaults -- a fixed legal value keeps
#:   one table usable for both;
#: * ``0.01 + 0.375`` and ``0.05 + 0.375`` stay inside their ranges, so those
#:   need no override;
#: * ``max_pages`` / ``max_geometries`` default to ``None``, for which the
#:   generic int rule produces ``0 + 7 = 7``; that IS legal and IS not the
#:   default, so it needs no override either.
#:
#: Each override is still checked against the default by trap 1, so an override
#: that accidentally equalled the default would fail rather than pass quietly.
PROBE_OVERRIDES: Dict[str, Any] = {
    "image_size": 128,        # a legal multiple of 32 that is neither default
    "lr_drop_factor": 0.5,    # in (0, 1] for both stages' defaults
}


def _config_fields() -> Set[str]:
    return {item.name for item in dataclass_fields(DocScannerTrainingConfig)}


def _actions_by_dest(build_parser: Callable[[], argparse.ArgumentParser]):
    """Every optional action of a REAL parser, keyed by destination."""
    return {
        action.dest: action
        for action in build_parser()._actions
        if action.option_strings and not isinstance(action, argparse._HelpAction)
    }


def _probe_value(action: argparse.Action, field_name: str, default: Any) -> Any:
    """A value for ``action`` guaranteed to differ from ``default``.

    Interface contract: pure. Reads only :data:`PROBE_OVERRIDES`, the action's
    declared ``choices`` / ``type`` and the dataclass default; parses nothing
    and constructs no config.

    :param action: The real argparse action.
    :param field_name: The config field the action writes.
    :param default: The stage default for that field.
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


def _rows(module, stage: str) -> Tuple[Row, ...]:
    """One generated :class:`Row` per parser dest of one entry point.

    THE ROW TABLE IS GENERATED, NOT TYPED. A hand-typed table is one chance per
    row to pick a probe value that is already the default (trap 1), and it goes
    stale the day a flag is added. Every row is derived from the REAL parser
    action plus the REAL stage default that parser bound.
    """
    actions = _actions_by_dest(module.build_parser)
    defaults = common.stage_defaults(stage)
    fields = _config_fields()
    rows: List[Row] = []
    for dest, action in sorted(actions.items()):
        flags = tuple(action.option_strings)
        if dest in module.NON_CONFIG_DESTS:
            # `--gpu` acts on the process; it must reach the NAMESPACE.
            rows.append(
                Row(flags=flags, argv=("--gpu", "1"),
                    namespace_dest=dest, expected=1)
            )
            continue
        if dest not in fields:
            # No row can be generated for a flag that reaches no field. Do NOT
            # raise here: an exception in the generator is a COLLECTION error
            # that takes every arm down and names the cause only in a
            # traceback. Skipping instead lets the completeness arms report it
            # as a named failure while the rest of the suite still runs.
            continue
        value = _probe_value(action, dest, getattr(defaults, dest))
        rows.append(
            Row(flags=flags, argv=(action.option_strings[0], str(value)),
                field=dest, expected=value)
        )
    return tuple(rows)


#: ``(module, stage)`` for each entry point. Everything below is parametrized
#: over this, so a flag wired in one script and forgotten in the other is a
#: named failure rather than half a green suite.
ENTRY_POINTS: Tuple[Tuple[Any, str], ...] = (
    (segmenter_cli, STAGE_SEGMENTER),
    (rectifier_cli, STAGE_RECTIFIER),
)

CONTRACTS: Tuple[Contract, ...] = tuple(
    Contract(
        name=module.PROGRAM_NAME,
        build_parser=(lambda mp, m=module: m.build_parser()),
        build_config=(lambda mp, m=module: m.config_from_argv(None)),
        modes=(Mode(id=stage, required_argv=(), rows=_rows(module, stage)),),
    )
    for module, stage in ENTRY_POINTS
)

_CASES, _IDS = cases(CONTRACTS)


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


# ---------------------------------------------------------------------
# completeness, per entry point
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "module,stage", ENTRY_POINTS, ids=[stage for _, stage in ENTRY_POINTS]
)
class TestCompleteness:
    """Every flag has a row and a field, and every field has a flag."""

    def test_every_declared_flag_has_a_contract_row(self, module, stage):
        """A flag added without a row fails HERE, not in production."""
        declared = declared_option_strings(module.build_parser())
        contract = next(c for c in CONTRACTS if c.name == module.PROGRAM_NAME)
        covered = contract.covered_flags
        assert declared == covered, (
            f"flags with no contract row: {sorted(declared - covered)}; rows "
            f"for flags that no longer exist: {sorted(covered - declared)}"
        )

    def test_every_parser_dest_is_a_config_field_or_an_exemption(self, module, stage):
        dests = set(_actions_by_dest(module.build_parser))
        unaccounted = dests - _config_fields() - module.NON_CONFIG_DESTS
        assert not unaccounted, (
            f"parser dests accounted for nowhere: {sorted(unaccounted)} -- "
            "each is a flag that cannot reach the config"
        )

    def test_every_config_field_has_a_cli_flag_or_is_fixed_by_the_script(
        self, module, stage
    ):
        missing = (
            _config_fields()
            - set(_actions_by_dest(module.build_parser))
            - FIXED_BY_ENTRY_POINT
        )
        assert not missing, (
            "DocScannerTrainingConfig fields with no CLI flag: "
            f"{sorted(missing)}"
        )

    def test_the_exemption_list_is_neither_stale_nor_a_config_field(
        self, module, stage
    ):
        """``NON_CONFIG_DESTS`` is the ONLY way out of the contract."""
        overlap = module.NON_CONFIG_DESTS & _config_fields()
        assert not overlap, (
            f"{sorted(overlap)} is exempted as a non-config dest but IS a "
            "config field"
        )
        stale = module.NON_CONFIG_DESTS - set(_actions_by_dest(module.build_parser))
        assert not stale, (
            "NON_CONFIG_DESTS exempts dests the parser does not declare: "
            f"{sorted(stale)} -- a stale exemption silently widens the carve-out"
        )


# ---------------------------------------------------------------------
# the two stages are actually two
# ---------------------------------------------------------------------


class TestTheTwoStagesAreDistinct:
    """The failure mode of a shared scaffold: one recipe, two names.

    The paper trains the two modules independently and states DIFFERENT
    numbers for each (§4.3). A refactor that binds one set of defaults to both
    scripts keeps every flag, every shape and every test above green.
    """

    def test_each_script_fixes_its_own_stage(self):
        assert segmenter_cli.config_from_argv([]).stage == STAGE_SEGMENTER
        assert rectifier_cli.config_from_argv([]).stage == STAGE_RECTIFIER

    def test_the_stage_cannot_be_reached_from_argv(self, capsys):
        """``--stage rectifier`` on the segmenter script must be a parse error."""
        with pytest.raises(SystemExit) as excinfo:
            segmenter_cli.build_parser().parse_args(["--stage", STAGE_RECTIFIER])
        assert excinfo.value.code == 2, (
            f"--stage exited {excinfo.value.code!r}; argparse reports an "
            "unrecognized argument with exit code 2"
        )
        assert "stage" in capsys.readouterr().err

    def test_the_stage_field_is_the_only_flagless_field(self):
        """The exemption is exactly one field and it is the one we think."""
        assert FIXED_BY_ENTRY_POINT == {"stage"}

    def test_the_paper_batch_sizes_differ(self):
        """32 (localization) vs 12 (rectification), verbatim from §4.3."""
        assert common.stage_defaults(STAGE_SEGMENTER).batch_size == 32
        assert common.stage_defaults(STAGE_RECTIFIER).batch_size == 12

    def test_only_the_rectifier_warms_up(self):
        """The paper gives a warmup for the rectifier and none for the segmenter."""
        assert common.stage_defaults(STAGE_SEGMENTER).warmup_epochs == 0
        assert common.stage_defaults(STAGE_RECTIFIER).warmup_epochs > 0

    def test_the_segmenter_drop_is_the_papers(self):
        """"reduced by a factor of 0.1 after 30 epochs"."""
        defaults = common.stage_defaults(STAGE_SEGMENTER)
        assert defaults.lr_drop_epoch == 30
        assert defaults.lr_drop_factor == pytest.approx(0.1)
        assert defaults.learning_rate == pytest.approx(1e-4)

    def test_an_unknown_stage_is_rejected(self):
        with pytest.raises(ValueError, match="stage must be one of"):
            common.stage_defaults("localization")


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

    def _crippled_config(self, args: argparse.Namespace) -> DocScannerTrainingConfig:
        """``config_from_args`` as if the author forgot one assignment."""
        full = common.config_from_args(args, STAGE_SEGMENTER)
        return replace(
            full,
            **{self.DROPPED: getattr(DocScannerTrainingConfig(), self.DROPPED)},
        )

    def test_a_dropped_forward_is_silent_at_runtime(self):
        args = segmenter_cli.build_parser().parse_args(
            [f"--{self.DROPPED}", str(self.PROBE)]
        )
        crippled = self._crippled_config(args)

        # Nothing raises, nothing warns: the user asked for 22 and gets 15.
        assert getattr(crippled, self.DROPPED) == getattr(
            DocScannerTrainingConfig(), self.DROPPED
        )
        assert getattr(crippled, self.DROPPED) != self.PROBE

        # The control: the REAL builder does forward it.
        assert (
            getattr(common.config_from_args(args, STAGE_SEGMENTER), self.DROPPED)
            == self.PROBE
        )

    def test_the_row_assertion_reds_on_that_drop(self, monkeypatch):
        contract = next(
            c for c in CONTRACTS if c.name == segmenter_cli.PROGRAM_NAME
        )
        broken = Contract(
            name=contract.name,
            build_parser=contract.build_parser,
            build_config=lambda mp: self._crippled_config(
                segmenter_cli.build_parser().parse_args(sys.argv[1:])
            ),
            modes=contract.modes,
        )
        row = next(
            r for r in contract.modes[0].rows if r.field == self.DROPPED
        )
        with pytest.raises(AssertionError, match="did NOT reach"):
            assert_row_reaches_config(monkeypatch, broken, broken.modes[0], row)


# ---------------------------------------------------------------------
# a bogus --data-source
# ---------------------------------------------------------------------


def test_a_bogus_data_source_is_rejected_and_the_valid_ones_are_named(capsys):
    """A typo'd source must not fall through to an empty-glob mystery."""
    with pytest.raises(SystemExit) as excinfo:
        segmenter_cli.build_parser().parse_args(["--data-source", "uv-doc"])

    assert excinfo.value.code == 2
    message = capsys.readouterr().err
    assert "uv-doc" in message, message
    for name in common.DOC_SCANNER_SOURCES:
        assert name in message, (
            f"the rejection message does not name the valid source {name!r}; "
            f"stderr was {message!r}"
        )


# ---------------------------------------------------------------------
# sentinels: --help allocates nothing, and the corpus gate comes first
# ---------------------------------------------------------------------


class _Sentinel:
    """Records contact instead of doing the expensive thing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError(f"sentinel {self.name!r} was called")


#: ``(module_key, attribute)`` pairs covering every expensive thing a run does:
#: claiming a GPU, walking a corpus, seeding the process, building the tf.data
#: pipeline, and constructing/compiling the model.
_EXPENSIVE: Tuple[Tuple[str, str], ...] = (
    ("trainer", "setup_gpu"),
    ("trainer", "require_training_data"),
    ("trainer", "train"),
    ("common", "set_seeds"),
    ("common", "build_model"),
    ("common", "create_dataset"),
)


def _install_sentinels(monkeypatch, module) -> Dict[str, _Sentinel]:
    modules = {"trainer": module, "common": common}
    out: Dict[str, _Sentinel] = {}
    for module_key, attribute in _EXPENSIVE:
        name = f"{module_key}.{attribute}"
        sentinel = _Sentinel(name)
        monkeypatch.setattr(modules[module_key], attribute, sentinel)
        out[name] = sentinel
    return out


@pytest.mark.parametrize(
    "module,stage", ENTRY_POINTS, ids=[stage for _, stage in ENTRY_POINTS]
)
def test_the_sentinels_cover_names_that_exist(monkeypatch, module, stage):
    """A sentinel over a misspelled attribute would never fire.

    ``monkeypatch.setattr`` raises on an unknown attribute, so installing them
    IS the check -- this arm states it as a claim rather than leaving it as an
    accident of the other tests.
    """
    sentinels = _install_sentinels(monkeypatch, module)
    assert len(sentinels) == len(_EXPENSIVE)
    assert all(s.calls == 0 for s in sentinels.values())


@pytest.mark.parametrize(
    "module,stage", ENTRY_POINTS, ids=[stage for _, stage in ENTRY_POINTS]
)
def test_help_prints_usage_and_allocates_nothing(
    monkeypatch, capsys, module, stage
):
    """``--help`` prints ``usage:`` and reaches no GPU, model or dataset."""
    sentinels = _install_sentinels(monkeypatch, module)
    monkeypatch.setattr(sys, "argv", [module.PROGRAM_NAME, "--help"])

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    reached = {name: s.calls for name, s in sentinels.items() if s.calls}
    assert not reached, (
        f"--help reached {reached} before argparse could exit. "
        "`args, config = parse_arguments(argv)` must be the FIRST statement of "
        "main(), above the corpus gate, GPU setup and training."
    )
    assert excinfo.value.code == 0, f"--help exited {excinfo.value.code!r}"

    printed = capsys.readouterr().out
    assert printed.startswith("usage:"), (
        "--help printed no `usage:` line. An exit code of 0 is not evidence: a "
        "script with no parser at all runs its whole job and exits 0 anyway. "
        f"stdout was {printed[:200]!r}"
    )
    assert module.PROGRAM_NAME in printed, (
        "--help does not name the script it belongs to; `prog=` is what keeps "
        "`python -m` from printing `__main__`"
    )
    for flag in ("--data-source", "--gpu", "--pages-root", "--image-size"):
        assert flag in printed, f"--help does not advertise {flag}"


@pytest.mark.parametrize(
    "module,stage", ENTRY_POINTS, ids=[stage for _, stage in ENTRY_POINTS]
)
class TestStartupOrder:
    """``require_training_data`` runs BEFORE anything expensive.

    Measured through behaviour: a missing corpus must raise with the real
    reason while the GPU and model sentinels are still untouched. Reading the
    source for statement order would prove nothing about what runs.
    """

    def _argv(self, tmp_path) -> List[str]:
        return ["--pages-root", str(tmp_path / "does-not-exist")]

    def test_a_missing_corpus_fails_before_the_gpu_and_the_model(
        self, monkeypatch, tmp_path, module, stage
    ):
        sentinels = _install_sentinels(monkeypatch, module)
        # `require_training_data` is the one thing that MUST run, so it keeps
        # its real implementation; everything downstream stays a sentinel.
        monkeypatch.setattr(
            module, "require_training_data", common.require_training_data
        )

        with pytest.raises(MissingTrainingDataError) as excinfo:
            module.main(self._argv(tmp_path))

        message = str(excinfo.value)
        assert "does-not-exist" in message, message
        assert "is not a directory" in message, message

        reached = {
            name: s.calls
            for name, s in sentinels.items()
            if s.calls and name != "trainer.require_training_data"
        }
        assert not reached, (
            f"a missing corpus reached {reached}. The gate must run before "
            "setup_gpu, before seeding and before any model or dataset "
            "construction, so the failure costs nothing."
        )

    def test_an_empty_but_present_root_names_the_real_reason(
        self, monkeypatch, tmp_path, module, stage
    ):
        """A directory with no pages is a DIFFERENT reason from a missing one.

        Both would produce the same empty worklist later; the point of the gate
        is that the user is told which of the two happened.
        """
        _install_sentinels(monkeypatch, module)
        monkeypatch.setattr(
            module, "require_training_data", common.require_training_data
        )
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(MissingTrainingDataError) as excinfo:
            module.main(["--pages-root", str(empty)])

        message = str(excinfo.value)
        assert "holds no image file" in message, message
        assert "is not a directory" not in message, message

    def test_the_probe_would_otherwise_have_trained(
        self, monkeypatch, tmp_path, module, stage
    ):
        """Anti-vacuity: the argv is a REAL run, not one argparse rejects.

        With the gate stubbed out, the same argv walks straight on into
        ``setup_gpu``. So the arms above are measuring the gate, not a parse
        error and not a missing flag.
        """
        sentinels = _install_sentinels(monkeypatch, module)
        monkeypatch.setattr(module, "require_training_data", lambda *a, **k: None)

        with pytest.raises(AssertionError, match="trainer.setup_gpu"):
            module.main(self._argv(tmp_path))

        assert sentinels["trainer.setup_gpu"].calls == 1
        assert sentinels["trainer.train"].calls == 0


class TestTheUvdocGateNamesTheDownload:
    """A missing 27.5 GB corpus must say so, not glob emptily."""

    def test_a_missing_uvdoc_root_is_reported_with_the_alternative(self, tmp_path):
        # `uvdoc_cache_root` is pinned to an EMPTY tmp_path, not left at its
        # default. Step 18 gave the uvdoc source a second, precomputed corpus
        # form, and the gate accepts a staged sidecar directory WITHOUT the
        # archive on purpose (that is the point of the cache). On a machine
        # where the default cache root happens to be populated -- as it is on
        # the developer machine that staged 400 renders -- a config that leaves
        # it at the default no longer reaches the archive branch at all, and
        # this arm silently stops testing the message it names. Pinning the
        # cache root is what keeps it pointed at the archive branch; the
        # cache branch has its own arms in `test_stage_uvdoc_samples.py`.
        config = replace(
            common.stage_defaults(STAGE_RECTIFIER),
            data_source=common.SOURCE_UVDOC,
            uvdoc_root=str(tmp_path / "no-uvdoc-here.zip"),
            uvdoc_cache_root=str(tmp_path / "no-cache-here"),
        )
        with pytest.raises(MissingTrainingDataError) as excinfo:
            common.require_training_data(config)
        message = str(excinfo.value)
        assert "no-uvdoc-here.zip" in message, message
        assert "synthetic" in message, (
            "the gate must name the corpus that needs no download; otherwise "
            f"the user's only option looks like a 27.5 GB fetch. Got {message!r}"
        )

    def test_a_populated_cache_root_makes_the_archive_optional(self, tmp_path):
        """The other half, so the arm above cannot be `green` by accident.

        The gate deliberately passes with NO archive when sidecars are staged
        for this `image_size`; without this arm, someone could `fix` the arm
        above by making the gate demand the archive unconditionally and never
        notice that a sidecar-only machine can no longer train.
        """
        directory = tmp_path / "cache" / "288x288"
        (directory / common.UVDOC_CACHE_RENDER_DIR).mkdir(parents=True)
        (directory / common.UVDOC_CACHE_RENDER_DIR / "00000.npz").write_bytes(
            b""
        )
        (directory / common.UVDOC_CACHE_MANIFEST).write_text(
            json.dumps({
                "schema": common.UVDOC_CACHE_SCHEMA,
                "height": 288,
                "width": 288,
            })
        )
        config = replace(
            common.stage_defaults(STAGE_RECTIFIER),
            data_source=common.SOURCE_UVDOC,
            uvdoc_root=str(tmp_path / "no-uvdoc-here.zip"),
            uvdoc_cache_root=str(tmp_path / "cache"),
            image_size=288,
        )
        common.require_training_data(config)
