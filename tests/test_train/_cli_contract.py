"""Driver for the trainer CLI contract guards (flag -> config field).

The defect class this exists to catch
-------------------------------------
A trainer declares a CLI flag, ``--help`` advertises it, and ``main()`` forgets
to forward it into the config. The flag then silently does nothing: no error, no
warning, and ``config.json`` records the DEFAULT while the user believes they
overrode it. It is the same class as the dead config field
``tests/test_train/test_config_fields_are_live.py`` catches, approached from the
other end -- that guard asks "does anything READ this field?", this one asks
"does the flag that claims to WRITE it actually arrive?".

This is the INVERSE of ``tests/test_train/test_ntm/test_ntm_trainers.py``'s
``assert_flag_is_not_registered`` (which pins that five inherited flags are
GONE). Same instrument, opposite polarity: drive the real parser and the real
config builder, and assert on the resulting OBJECT -- never on an exit code and
never on a ``--help`` string diff. An exit-code check is measurably weaker: that
NTM helper records that ``--dataset 1`` and ``--show-plots 1`` exit non-zero for
reasons unrelated to registration, so 2 of its 5 flags were unreadable that way.

Three traps are designed out, each one a way this guard could pass vacuously:

1. A ROW MUST CARRY A NON-DEFAULT VALUE.
   ``--epochs 3`` against a config whose default is already ``3`` cannot
   distinguish "forwarded" from "never touched". Every row is therefore
   re-driven WITHOUT its own argv fragment and the resulting field is asserted
   DIFFERENT from the expected value -- see
   ``assert_row_value_is_not_the_default``. That test is what makes the rest of
   the file mean anything.

2. THE ROW TABLE MUST BE COMPLETE.
   A guard that only checks the flags someone remembered to list goes stale the
   moment a flag is added. ``declared_option_strings`` reads the flags off the
   REAL parser and the completeness test asserts set equality against the table,
   so a new flag fails until it is given a row.

3. VALUES MUST BE MUTUALLY DISTINCT.
   If two rows use the same value, a cross-wired forward (``num_epochs=
   args.batch_size``) still passes. Rows are driven ONE AT A TIME against
   otherwise-default argv, so a cross-wire moves the wrong field and the
   intended field stays at its default -- which trap 1 has already proven is
   distinguishable.

A flag that is deliberately NOT a config field (``--gpu``, consumed by
``setup_gpu`` in ``main()``) carries ``namespace_dest`` instead of ``field`` and
is asserted to reach the argparse NAMESPACE. It is never silently skipped.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

_MISSING = object()


@dataclass(frozen=True)
class Row:
    """One flag of one trainer, and where its value must arrive.

    :param flags: Every option string this row accounts for. More than one for
        ``BooleanOptionalAction`` (``--tie-word-embeddings`` /
        ``--no-tie-word-embeddings``), which the completeness check sees as two
        declared flags while one argv fragment exercises both.
    :param argv: The argv fragment that sets it, e.g. ``("--epochs", "7")``.
        Empty when the value is supplied by the mode's ``required_argv``
        (a required flag, or one arm of a required mutually-exclusive group).
    :param field: Config attribute the value must reach, or ``None`` for a flag
        that is deliberately not a config field.
    :param expected: Value the config attribute must hold afterwards.
    :param namespace_dest: For ``field=None`` rows, the argparse destination the
        value must reach instead. Exactly one of ``field`` / ``namespace_dest``
        is set.
    """

    flags: Tuple[str, ...]
    argv: Tuple[str, ...]
    field: Optional[str] = None
    expected: Any = None
    namespace_dest: Optional[str] = None

    @property
    def id(self) -> str:
        return self.flags[0]


@dataclass(frozen=True)
class Mode:
    """One argv shape of a trainer.

    A trainer needs more than one mode when its parser has a required
    mutually-exclusive group: ``train.gpt2.finetune`` cannot exercise
    ``--hf-dataset`` and ``--text-dir`` in the same invocation.

    :param id: Short label used in the pytest node id.
    :param required_argv: Flags every invocation of this mode must carry.
    :param rows: The flag contract rows exercised in this mode.
    :param baseline_argv: argv used to measure a field's DEFAULT for trap 1.
        Defaults to ``required_argv``; override it when a row's value is itself
        supplied by ``required_argv``, so the baseline still differs.
    """

    id: str
    required_argv: Tuple[str, ...]
    rows: Tuple[Row, ...]
    baseline_argv: Optional[Tuple[str, ...]] = None

    @property
    def defaults_argv(self) -> Tuple[str, ...]:
        return self.required_argv if self.baseline_argv is None else self.baseline_argv


@dataclass(frozen=True)
class Contract:
    """A trainer entry point and its full flag table.

    :param name: Program name, used as ``sys.argv[0]`` and in node ids.
    :param build_parser: ``(monkeypatch) -> ArgumentParser`` -- the REAL parser.
        Takes monkeypatch because ``train.gpt2.finetune`` builds its parser
        inside ``main()`` and has to be intercepted to get at it.
    :param build_config: ``(monkeypatch) -> config`` -- reads ``sys.argv``, which
        the caller has already set. Drives the real parser and the real config
        builder, so anything the trainer does between them is exercised.
    :param modes: One or more argv shapes.
    """

    name: str
    build_parser: Callable[[Any], argparse.ArgumentParser]
    build_config: Callable[[Any], Any]
    modes: Tuple[Mode, ...]

    @property
    def covered_flags(self) -> set:
        return {f for mode in self.modes for row in mode.rows for f in row.flags}


def declared_option_strings(parser: argparse.ArgumentParser) -> set:
    """Every option string the parser declares, minus ``-h`` / ``--help``.

    Reads ``parser._actions`` because argparse exposes no public accessor for
    its action list. Reading the REAL parser rather than a hand-written list is
    the point: it is what makes the completeness check fail on a newly added
    flag instead of ignoring it.
    """
    out = set()
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        out.update(action.option_strings)
    return out


def cases(contracts) -> Tuple[List[Tuple[Contract, Mode, Row]], List[str]]:
    """Flatten ``(contract, mode, row)`` triples plus their pytest ids."""
    params: List[Tuple[Contract, Mode, Row]] = []
    ids: List[str] = []
    for contract in contracts:
        for mode in contract.modes:
            for row in mode.rows:
                params.append((contract, mode, row))
                ids.append(f"{contract.name}[{mode.id}]{row.id}")
    return params, ids


def _drive(monkeypatch, contract: Contract, argv: Tuple[str, ...]) -> Any:
    monkeypatch.setattr(sys, "argv", [contract.name, *argv])
    return contract.build_config(monkeypatch)


def assert_row_reaches_config(
    monkeypatch, contract: Contract, mode: Mode, row: Row
) -> None:
    """The value given on the command line must arrive where the row says."""
    if row.field is None:
        parser = contract.build_parser(monkeypatch)
        args = parser.parse_args([*mode.required_argv, *row.argv])
        actual = getattr(args, row.namespace_dest, _MISSING)
        assert actual is not _MISSING, (
            f"{contract.name}: {row.id} declares no argparse destination "
            f"{row.namespace_dest!r}"
        )
        assert actual == row.expected, (
            f"{contract.name}: {row.id} {row.argv} reached "
            f"args.{row.namespace_dest} = {actual!r}, expected {row.expected!r}"
        )
        return

    config = _drive(monkeypatch, contract, (*mode.required_argv, *row.argv))
    actual = getattr(config, row.field, _MISSING)
    assert actual is not _MISSING, (
        f"{contract.name}: config has no attribute {row.field!r} -- the field "
        f"{row.id} forwards into was renamed or removed"
    )
    assert actual == row.expected, (
        f"{contract.name}: {row.id} {row.argv} did NOT reach "
        f"config.{row.field} (got {actual!r}, expected {row.expected!r}). "
        "The flag is advertised in --help and silently does nothing."
    )


def assert_row_value_is_not_the_default(
    monkeypatch, contract: Contract, mode: Mode, row: Row
) -> None:
    """Trap 1: the row must be able to tell "forwarded" from "never touched".

    Drives the trainer with the row's own argv fragment REMOVED and asserts the
    field does not already hold the expected value. A row that fails here is
    vacuous by construction: it would pass against a ``_config_from_args`` that
    never mentions the flag at all.
    """
    if row.field is None:
        parser = contract.build_parser(monkeypatch)
        args = parser.parse_args(list(mode.defaults_argv))
        actual = getattr(args, row.namespace_dest, _MISSING)
    else:
        config = _drive(monkeypatch, contract, mode.defaults_argv)
        actual = getattr(config, row.field, _MISSING)
    assert actual != row.expected, (
        f"{contract.name}: {row.id} probes with {row.expected!r}, which is "
        f"ALREADY the default of {row.field or row.namespace_dest}. The row "
        "cannot distinguish a forwarded value from an untouched one -- pick a "
        "different probe value."
    )


# ---------------------------------------------------------------------
# The shared CLM flag surface
# ---------------------------------------------------------------------

#: The 27 flags `train.gpt2.pretrain` and `train.wave_field.pretrain` share.
#:
#: ONE table, consumed by both packages' contract modules, because the two
#: parsers are the same surface by design -- `src/train/CLAUDE.md` § Pattern 3
#: requires every CLM consumer to expose the same flags so users can switch
#: scripts without relearning, and both configs now inherit the single
#: `ClmPretrainConfig` (D-010). A private copy per package is how that
#: agreement would drift apart unnoticed; sharing the table means a flag added
#: to only one trainer fails that trainer's completeness test.
#:
#: `--resume` -> `resume_from` is the one row whose flag and field names differ,
#: which is exactly the kind of hand-written hop this guard exists to pin.
CLM_PRETRAIN_ROWS: Tuple[Row, ...] = (
    Row(("--gpu",), ("--gpu", "1"), namespace_dest="gpu", expected=1),
    Row(("--variant",), ("--variant", "medium"), "model_variant", "medium"),
    Row(("--num-layers",), ("--num-layers", "5"), "num_layers", 5),
    Row(("--num-heads",), ("--num-heads", "3"), "num_heads", 3),
    Row(
        ("--tie-word-embeddings", "--no-tie-word-embeddings"),
        ("--no-tie-word-embeddings",),
        "tie_word_embeddings",
        False,
    ),
    Row(("--epochs",), ("--epochs", "7"), "num_epochs", 7),
    Row(("--batch-size",), ("--batch-size", "3"), "batch_size", 3),
    Row(("--max-seq-length",), ("--max-seq-length", "128"), "max_seq_length", 128),
    Row(("--learning-rate",), ("--learning-rate", "1.5e-5"), "learning_rate", 1.5e-5),
    Row(("--loss-type",), ("--loss-type", "focal"), "loss_type", "focal"),
    Row(("--focal-gamma",), ("--focal-gamma", "2.5"), "focal_gamma", 2.5),
    Row(("--label-smoothing",), ("--label-smoothing", "0.15"), "label_smoothing", 0.15),
    Row(("--dataset-source",), ("--dataset-source", "tfds"), "dataset_source", "tfds"),
    Row(
        ("--dataset-name",),
        ("--dataset-name", "probe_corpus"),
        "dataset_name",
        "probe_corpus",
    ),
    Row(("--max-samples",), ("--max-samples", "321"), "max_samples", 321),
    Row(
        ("--hf-cache-dir",),
        ("--hf-cache-dir", "/probe/hf-cache"),
        "hf_cache_dir",
        "/probe/hf-cache",
    ),
    Row(
        ("--max-train-samples",),
        ("--max-train-samples", "654"),
        "max_train_samples",
        654,
    ),
    Row(("--val-fraction",), ("--val-fraction", "0.33"), "val_fraction", 0.33),
    Row(
        ("--min-article-length",),
        ("--min-article-length", "500"),
        "min_article_length",
        500,
    ),
    Row(("--shuffle-shards",), ("--shuffle-shards", "9"), "shuffle_shards", 9),
    Row(("--seed",), ("--seed", "1234"), "seed", 1234),
    Row(
        ("--checkpoint-every-steps",),
        ("--checkpoint-every-steps", "111"),
        "checkpoint_every_steps",
        111,
    ),
    Row(
        ("--analyze-every-steps",),
        ("--analyze-every-steps", "222"),
        "analyze_every_steps",
        222,
    ),
    Row(("--max-checkpoints",), ("--max-checkpoints", "8"), "max_checkpoints", 8),
    Row(("--steps-per-epoch",), ("--steps-per-epoch", "777"), "steps_per_epoch", 777),
    Row(
        ("--resume",),
        ("--resume", "/probe/ckpt.keras"),
        "resume_from",
        "/probe/ckpt.keras",
    ),
    Row(("--save-dir",), ("--save-dir", "/probe/save"), "save_dir", "/probe/save"),
)
