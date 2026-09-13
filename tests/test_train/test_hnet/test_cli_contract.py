r"""CLI contract guard for ``src/train/hnet/train_hnet.py``.

The defect class this exists to catch
-------------------------------------
A trainer declares a flag, ``--help`` advertises it, and ``main()`` never
forwards it. The flag then silently does nothing: no error, no warning, and
``config.json`` records the DEFAULT while the user believes they overrode it.
:class:`TestTheForwardingAssertionCanFail` reproduces that failure mode
executably against a deliberately crippled ``config_from_args``, so the
assertions below are known to DISCRIMINATE rather than merely known to be green.

The second defect class is an ADVERTISED CHOICE THAT CANNOT BE BUILT. This repo
has shipped one: a ``--loss siglip`` choice whose branch called a method only
the other loss class defined, under three passing tests that never built it.
:class:`TestEveryAdvertisedArchVariantConstructs` therefore CONSTRUCTS every
``--arch-variant`` choice through the real ``argv -> config -> build_model``
path. Checking that the string is in ``choices`` would pass against exactly the
defect it is supposed to catch.

Reused, not re-implemented
--------------------------
* ``tests/test_train/_cli_contract.py`` -- the shared driver. It runs the REAL
  parser and the REAL config builder and asserts on the resulting OBJECT, and
  its three designed-out traps (a row must carry a non-default value; the row
  table must be complete; rows are driven ONE AT A TIME so a cross-wire leaves
  the intended field at its default) are what make these rows mean anything.
* ``tests/test_train/test_config_fields_are_live.py`` -- the other end of the
  same contract, asking "does anything READ this field?".
  ``HNetTrainingConfig`` is REGISTERED there (row added in step 16 and asserted
  present by :func:`test_the_config_class_is_registered_as_live`), so a field
  that only reaches ``save_config_json`` fails THERE and needs no weaker copy
  here.
* ``tests/test_train/test_hnet/test_pipeline.py`` -- pins the same forwarding
  through ``add_common_arguments`` + ``config_from_args``. This module drives
  the ENTRY POINT's parser instead, which is a different object: it carries
  ``--gpu``, and it is the one a user actually reaches. A flag added to
  ``train_hnet.build_parser`` alone would be invisible to that module.

THE ROW TABLE IS GENERATED, NOT TYPED. A hand-typed table is one chance per row
to pick a probe value that is already the default (trap 1), and it goes stale
the day a flag is added. :func:`_rows` derives every row from the REAL parser
action plus the REAL dataclass default. Three fields carry an explicit override
because the generic ``default + offset`` rule produces a value
``HNetTrainingConfig.__post_init__`` rightly REJECTS, or a shape it cannot hold
-- see :data:`PROBE_OVERRIDES`. The shared ``assert_row_value_is_not_the_default``
then MEASURES that every probe, generated or overridden, differs from the
default, so neither the generator nor the override table is trusted.

``--help`` MUST ALLOCATE NOTHING. Sentinels are installed over ``setup_gpu``
and ``train`` on the trainer module and over ``set_seeds``, ``build_model``,
``build_datasets`` and ``build_optimizer`` on ``common``; each records contact.
:func:`test_help_prints_usage_and_allocates_nothing` asserts every one of them
was reached ZERO times and that stdout starts with ``usage:``. Asserting only
``exit == 0`` is measurably weaker -- ``src/train/CLAUDE.md`` records that a
script with no parser at all ignores ``--help``, runs its whole job and exits 0
-- and :class:`TestTheExitCodeAssertionIsVacuous` proves that weakness
executably, by passing the exit-code form against a parser-less stub.

Nothing here trains, allocates a GPU, reads the corpus, downloads anything, or
writes into the repo-root ``results/``. The models the variant arm constructs
are UNBUILT (Keras subclassed models allocate weights on first call), so the
arm costs ~2.6 s for all seven variants and no device memory.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any, Dict, List, Set, Tuple

import pytest

from dl_techniques.models.language.hnet.model import HNet
from dl_techniques.models.language.masked_language_model.clm import (
    CausalLanguageModel,
)
from train.hnet import common
from train.hnet import train_hnet as trainer
from train.hnet.common import ARCH_VARIANTS, HNetTrainingConfig, get_arch_config
from train.hnet.train_hnet import (
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

#: Fields whose generic probe would be REJECTED by ``__post_init__``, or whose
#: SHAPE the generic rule gets wrong. These are validation facts, not
#: preferences:
#:
#: * ``final_learning_rate``: ``3e-5 + 0.375`` exceeds the ``learning_rate`` a
#:   row driven ALONE leaves at its ``3e-4`` default, and the config requires
#:   ``0 < final_learning_rate <= learning_rate``.
#: * ``val_fraction`` is fine at ``0.02 + 0.375``, but ``warmup_ratio`` and it
#:   are both bounded above by 1.0, so they are left to the generic rule and
#:   trap 1 checks them.
#: * ``max_chunks`` is ``nargs="+"``: the value is a LIST of one entry per
#:   chunking level, and the default variant (``dev``) has exactly one. The
#:   generic ``int`` rule would produce the scalar ``7`` and compare it against
#:   the parsed ``[7]``.
#: * ``seq_len``: the generic ``512 + 7 = 519`` is legal, but a power of two is
#:   what a user would ever pass, and it keeps the probe honest about the knob.
#:
#: Each override is still checked against the default by trap 1, so an override
#: that accidentally equalled the default would fail rather than pass quietly.
PROBE_OVERRIDES: Dict[str, Any] = {
    "final_learning_rate": 5e-5,  # positive and below the 3e-4 default peak
    "max_chunks": [17],           # one entry: `dev` has one chunking level
    "seq_len": 128,
}


def _config_fields() -> Set[str]:
    return {item.name for item in dataclass_fields(HNetTrainingConfig)}


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
    declared ``choices`` / ``type`` and the dataclass default; parses nothing
    and constructs no config.

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


def _argv_for(action: argparse.Action, value: Any) -> Tuple[str, ...]:
    """The argv fragment that sets ``action`` to ``value``.

    A list value spreads across the fragment, which is what ``nargs="+"``
    consumes; anything else is a single token.
    """
    flag = action.option_strings[0]
    if isinstance(value, list):
        return (flag, *(str(item) for item in value))
    return (flag, str(value))


def _rows() -> Tuple[Row, ...]:
    """One generated :class:`Row` per parser dest.

    The row set is derived from the REAL parser's dest set, so a flag added
    without a config field fails the completeness arm instead of being ignored.
    """
    actions = _actions_by_dest()
    defaults = HNetTrainingConfig()
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
            # that takes every arm down and names the cause only in a
            # traceback. Skipping instead lets
            # `test_every_parser_dest_is_a_config_field_or_an_exemption` and
            # `test_every_declared_flag_has_a_contract_row` both report it as a
            # named failure while the rest of the suite still runs.
            continue
        value = _probe_value(action, dest, getattr(defaults, dest))
        rows.append(
            Row(flags=flags, argv=_argv_for(action, value),
                field=dest, expected=value)
        )
    return tuple(rows)


CONTRACT = Contract(
    name="train_hnet.py",
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
            f"HNetTrainingConfig fields with no CLI flag: {sorted(missing)}"
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

    def test_the_config_class_is_registered_as_live(self):
        """The other end of the contract must cover this config class.

        ``test_config_fields_are_live.py`` only inspects the classes in its
        ``REGISTERED`` list, and subclassing does not inherit membership. The
        row was added in step 16; this asserts it is still there rather than
        adding a second, weaker copy of that guard.
        """
        from tests.test_train.test_config_fields_are_live import REGISTERED

        assert ("train/hnet/common.py", "HNetTrainingConfig") in REGISTERED


# ---------------------------------------------------------------------
# every advertised --arch-variant CONSTRUCTS
# ---------------------------------------------------------------------


class TestEveryAdvertisedArchVariantConstructs:
    """An advertised choice that cannot be built is a dead CLI branch.

    Each variant is driven through the REAL path a user takes -- ``argv`` ->
    ``config_from_argv`` -> ``common.build_model`` -- so a layout that parses
    but cannot be assembled (a mixer letter the block builder does not handle,
    a per-stage list of the wrong length, a head dimension that does not divide
    the expanded width) fails HERE.

    The constructed models are deliberately NOT called. Keras subclassed models
    allocate weights on first call, and the six reference variants start at
    ``d_model = 1024`` with 22+ inner layers: calling them would cost minutes
    and gigabytes to prove a construction claim. ``test_pipeline.py`` runs a
    real forward and backward step on ``dev``, which is where behaviour is
    measured.
    """

    @pytest.mark.parametrize("variant", ARCH_VARIANTS)
    def test_the_choice_builds_a_model(self, variant: str):
        config = config_from_argv(["--arch-variant", variant])
        model = common.build_model(config, steps_per_epoch=1)

        assert isinstance(model, CausalLanguageModel)
        assert isinstance(model.backbone, HNet)
        assert model.backbone.arch_config == get_arch_config(variant), (
            f"--arch-variant {variant!r} built a model whose architecture is "
            "not the one that name resolves to"
        )
        assert model.optimizer is not None, (
            f"--arch-variant {variant!r} produced an uncompiled model"
        )
        assert not model.built, (
            "build_model() must not force a forward pass through the "
            "backbone: the six real reference variants (d_model=1024, "
            "22+ layers) are constructed here specifically to prove "
            "constructibility without paying for a real call -- see "
            "build_model's own D-011 comment, plan-2026-09-13T052422-19022ba2"
        )

    def test_the_parser_advertises_exactly_the_constructible_set(self):
        """A choice list and a variant table that drift apart is the defect.

        Reads the choices off the REAL parser action rather than off
        ``ARCH_VARIANTS``, so an extra choice typed straight into
        ``add_common_arguments`` would be advertised, unconstructible, and
        caught -- the arm above only iterates ``ARCH_VARIANTS``.
        """
        advertised = list(_actions_by_dest()["arch_variant"].choices)
        assert advertised == list(ARCH_VARIANTS)
        for name in advertised:
            assert get_arch_config(name).num_stages >= 2

    def test_a_fabricated_advertised_choice_is_caught(self, monkeypatch):
        """Anti-vacuity, in the exact shape of the defect.

        Widening :data:`~train.hnet.common.ARCH_VARIANTS` with a name no
        architecture table can resolve makes the parser ADVERTISE a choice that
        cannot be built -- the dead-CLI-branch shape. The parse succeeds (so the
        name really is advertised) and the construction path refuses it by name,
        which is what makes the seven passing arms above evidence rather than
        coincidence.
        """
        fabricated = "hnet_9stage_XXL"
        monkeypatch.setattr(
            common, "ARCH_VARIANTS", tuple(ARCH_VARIANTS) + (fabricated,)
        )

        advertised = list(_actions_by_dest()["arch_variant"].choices)
        assert fabricated in advertised, (
            "the fabricated name was not advertised; the probe proves nothing"
        )

        args = build_parser().parse_args(["--arch-variant", fabricated])
        assert args.arch_variant == fabricated

        with pytest.raises(ValueError, match="unknown arch_variant"):
            common.config_from_args(args)


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

    def _crippled_config(self, args: argparse.Namespace) -> HNetTrainingConfig:
        """``config_from_args`` as if the author forgot one assignment."""
        full = common.config_from_args(args)
        return replace(
            full, **{self.DROPPED: getattr(HNetTrainingConfig(), self.DROPPED)}
        )

    def test_a_dropped_forward_is_silent_at_runtime(self):
        args = build_parser().parse_args([f"--{self.DROPPED}", str(self.PROBE)])
        crippled = self._crippled_config(args)

        # Nothing raises, nothing warns: the user asked for 22 and gets 5.
        assert getattr(crippled, self.DROPPED) == getattr(
            HNetTrainingConfig(), self.DROPPED
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
# a bogus --arch-variant
# ---------------------------------------------------------------------


def test_a_bogus_arch_variant_is_rejected_and_the_valid_ones_are_named(capsys):
    """A typo'd variant must not fall through to a construction traceback."""
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--arch-variant", "hnet_2stage_xl"])

    assert excinfo.value.code == 2, (
        f"a bad --arch-variant exited {excinfo.value.code!r}; argparse reports "
        "an invalid choice with exit code 2"
    )
    message = capsys.readouterr().err
    assert "hnet_2stage_xl" in message, message
    for name in ARCH_VARIANTS:
        assert name in message, (
            f"the rejection message does not name the valid variant {name!r}; "
            f"stderr was {message!r}"
        )


# ---------------------------------------------------------------------
# sentinels: --help allocates nothing
# ---------------------------------------------------------------------


class _Sentinel:
    """Records contact -- and what it was called WITH -- instead of doing the
    expensive thing.

    The arguments are kept because "``setup_gpu`` was reached" is a weaker claim
    than "``setup_gpu`` was reached with the index the user asked for": a
    ``setup_gpu(gpu_id=None)`` that ignores ``--gpu`` passes the first and fails
    the second.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.last_args: Tuple[Any, ...] = ()
        self.last_kwargs: Dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.last_args = args
        self.last_kwargs = dict(kwargs)
        raise AssertionError(f"sentinel {self.name!r} was called")


#: ``(module, attribute)`` pairs covering every expensive thing an H-Net run
#: does: claiming a GPU, seeding the process, opening the 20 GB Arrow cache,
#: building the optimizer, constructing/compiling the model, and ``fit()``.
_EXPENSIVE: Tuple[Tuple[str, str], ...] = (
    ("trainer", "setup_gpu"),
    ("trainer", "train"),
    ("common", "set_seeds"),
    ("common", "build_datasets"),
    ("common", "build_model"),
    ("common", "build_optimizer"),
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
    monkeypatch.setattr(sys, "argv", ["train_hnet.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        trainer.main()

    reached = {name: s.calls for name, s in sentinels.items() if s.calls}
    assert not reached, (
        f"--help reached {reached} before argparse could exit. "
        "`args, config = parse_arguments(argv)` must be the FIRST statement of "
        "main(), above GPU setup and training."
    )
    assert excinfo.value.code == 0, f"--help exited {excinfo.value.code!r}"

    printed = capsys.readouterr().out
    assert printed.startswith("usage:"), (
        "--help printed no `usage:` line. An exit code of 0 is not evidence: a "
        "script with no parser at all runs its whole job and exits 0 anyway. "
        f"stdout was {printed[:200]!r}"
    )
    for flag in ("--arch-variant", "--gpu", "--seq-len", "--dataset-root"):
        assert flag in printed, f"--help does not advertise {flag}"


class TestTheExitCodeAssertionIsVacuous:
    """Why the arm above asserts ``usage:`` and not ``exit == 0``.

    ``src/train/CLAUDE.md`` states the rule; this measures it. A stub with no
    parser at all -- the exact defect -- runs its whole job on ``--help`` and
    exits 0, so the exit-code form PASSES against it while the ``usage:`` form
    fails. The claim is executable rather than quoted.
    """

    @staticmethod
    def _parserless_main(argv=None) -> None:
        """A trainer that ignores its arguments and does the expensive thing."""
        TestTheExitCodeAssertionIsVacuous.did_the_job.append(True)

    did_the_job: List[bool] = []

    def test_the_exit_code_form_passes_against_a_parserless_script(self, capsys):
        self.did_the_job.clear()

        exit_code = 0
        try:
            self._parserless_main(["--help"])
        except SystemExit as exc:  # pragma: no cover -- the stub never exits
            exit_code = exc.code

        # The weak assertion. It passes, and the job ran anyway.
        assert exit_code == 0
        assert self.did_the_job == [True], (
            "the stub was supposed to do its whole job on --help"
        )

        # The strong assertion, on the same run, fails.
        printed = capsys.readouterr().out
        assert not printed.startswith("usage:")

    def test_the_real_trainer_fails_that_stub_s_test(self, monkeypatch, capsys):
        """The control: the real entry point does print ``usage:`` and exit.

        The sentinels are installed here even though a correct ``main()`` exits
        before reaching any of them. That is deliberate and was learned the
        expensive way: while this module's own RED proof was running, a mutant
        with the parse REMOVED from ``main()`` walked past ``--help`` straight
        into ``fit()`` and started a real Wikipedia training run out of a test
        process. Every arm that calls the REAL ``main()`` sits behind sentinels
        for that reason -- a guard must not be able to train.
        """
        sentinels = _install_sentinels(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["train_hnet.py", "--help"])
        with pytest.raises(SystemExit):
            trainer.main()
        assert capsys.readouterr().out.startswith("usage:")
        assert all(s.calls == 0 for s in sentinels.values())


# ---------------------------------------------------------------------
# startup order, measured through behaviour
# ---------------------------------------------------------------------


class TestStartupOrder:
    """``setup_gpu`` runs AFTER the parse and BEFORE ``train``.

    Measured through behaviour: a real ``main()`` walks into the GPU sentinel
    with ``train`` still untouched, and a bad flag never reaches either.
    Reading the source for statement order would prove nothing about what runs.
    """

    def test_a_real_run_reaches_the_gpu_before_training(self, monkeypatch):
        sentinels = _install_sentinels(monkeypatch)

        with pytest.raises(AssertionError, match="trainer.setup_gpu"):
            trainer.main(["--arch-variant", "dev", "--gpu", "1"])

        gpu = sentinels["trainer.setup_gpu"]
        assert gpu.calls == 1
        assert (*gpu.last_args, *gpu.last_kwargs.values()) == (1,), (
            f"setup_gpu was called with {gpu.last_args}/{gpu.last_kwargs}; "
            "--gpu 1 must reach it, or the flag silently does nothing"
        )
        assert sentinels["trainer.train"].calls == 0
        assert sentinels["common.build_datasets"].calls == 0

    def test_a_bad_flag_reaches_nothing_at_all(self, monkeypatch):
        """Argparse rejects before the GPU: a typo costs no device."""
        sentinels = _install_sentinels(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            trainer.main(["--arch-variant", "not-a-variant"])

        assert excinfo.value.code == 2
        reached = {name: s.calls for name, s in sentinels.items() if s.calls}
        assert not reached, f"a rejected flag still reached {reached}"

    def test_an_invalid_knob_combination_fails_before_the_gpu(self, monkeypatch):
        """``__post_init__`` runs inside the parse step, above ``setup_gpu``.

        A floor above the peak learning rate is a knob combination the trainer
        cannot honour; it must be refused while the process is still cheap.
        """
        sentinels = _install_sentinels(monkeypatch)

        with pytest.raises(ValueError, match="final_learning_rate"):
            trainer.main(["--learning-rate", "1e-5", "--final-learning-rate", "1.0"])

        reached = {name: s.calls for name, s in sentinels.items() if s.calls}
        assert not reached, f"an invalid config still reached {reached}"
