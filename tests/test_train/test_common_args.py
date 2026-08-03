"""Direct tests for ``train.common.args.explicitly_set_flags``.

The helper answers PROVENANCE ("did the caller type this flag?") rather than
VALUE ("does this differ from the default?"), which is the only way a
``--smoke``-style preset can lose to a flag typed at the flag's OWN default.
Its blast radius is three trainers (``train_dino``, ``train_video_jepa``,
``train_lewm``), and until this module existed it was exercised only
INDIRECTLY, through the DINO parser, in ``tests/test_train/test_dino/``.

The helper's docstring enumerates the token shapes it does and does not see.
This module pins that enumeration as a table.

Two assertions per reachable row:

1. the helper reports exactly the expected ``dest`` set, and
2. argparse ACCEPTS the same argv — so the row is a vector a real user can
   actually type, not a look-alike. Rows argparse rejects (an ambiguous
   prefix; any abbreviation under ``allow_abbrev=False``) are marked
   ``reachable=False`` and only the first assertion applies: they cannot occur
   in an argv that parsed, so the helper's answer for them is unobservable and
   is pinned only to document the branch.

Deliberately NOT re-done here: the exhaustive differential oracle that wraps
each argparse ``Action`` and compares dispatched dests against this helper over
thousands of generated vectors. That was executed once against all three real
parsers during this plan's adversarial review (9,233 accepted vectors, zero
disagreements) and lives in that review, not in the suite — it is a one-off
proof of equivalence, not a regression gate. What belongs in the suite is the
enumerated contract, which is what this module is.

One documented shape has no row, because it is unreachable through a successful
parse: a VALUE that happens to spell a registered option
(``--experiment-name --smoke``). argparse rejects that argv ("expected one
argument") before the helper's answer can matter, and distinguishing it would
require reimplementing nargs/type handling.

The single-dash shapes DO have rows, in ``_SHORT_DASH_CASES``: argparse accepts
``-b8`` and ``-vb 8``, the token scan sees neither, and the helper therefore
REFUSES any parser registering a single-dash option other than ``-h``.
"""

from __future__ import annotations

import argparse

import pytest


def _probe_parser(
        allow_abbrev: bool = True,
        short_options: bool = False,
) -> argparse.ArgumentParser:
    """A throwaway parser carrying every shape the helper's docstring names.

    Deliberately LOCAL and tiny so the table below runs without importing a
    trainer. It reproduces the four structural features that matter:

    * a strict PREFIX PAIR (``--ema-warmup-epochs`` / ``--ema-warmup-steps``),
      so ``--ema-w`` is genuinely AMBIGUOUS and ``--ema-warmup-ep`` genuinely
      is not;
    * an ``argparse.BooleanOptionalAction``, which registers TWO option
      strings (``--x`` and ``--no-x``) on ONE dest;
    * ``-h`` / ``--help``, the one single-dash option every real trainer
      registers;
    * a POSITIONAL, without which a bare ``--`` separator cannot appear in an
      argv argparse accepts (none of the three real trainers has one, so this
      is the only place the ``--`` break is exercised on a parse that
      SUCCEEDS).

    ``short_options=True`` additionally registers ``-n/--n-workers`` (takes a
    value) and ``-v/--verbose`` (a flag) — the pair needed to type an ATTACHED
    ``-n8`` and a GROUPED ``-vn 8``, both of which argparse accepts and the
    token scan cannot see. No real trainer registers either; the flag exists
    only to drive the refusal.
    """
    parser = argparse.ArgumentParser(prog="probe", allow_abbrev=allow_abbrev)
    if short_options:
        parser.add_argument("-n", "--n-workers", type=int, default=1)
        parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ema-warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--stateless-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("passthrough", nargs="*")
    return parser


# (test id, allow_abbrev, argv, expected dests, argparse accepts this argv)
_CASES = [
    ("exact-long-flag", True, ["--batch-size", "8"], {"batch_size"}, True),
    # The motivating shape: typed AT the flag's own parser default. A
    # value-vs-default comparison cannot see this one at all.
    ("exact-long-flag-at-its-own-default", True,
     ["--ema-warmup-epochs", "1.0"], {"ema_warmup_epochs"}, True),
    ("equals-form", True, ["--batch-size=8"], {"batch_size"}, True),
    ("unambiguous-prefix", True, ["--batch-s", "8"], {"batch_size"}, True),
    ("equals-form-on-an-abbreviated-flag", True,
     ["--ema-warmup-ep=1.5"], {"ema_warmup_epochs"}, True),
    ("boolean-optional-positive", True,
     ["--stateless-augmentation"], {"stateless_augmentation"}, True),
    ("boolean-optional-negative", True,
     ["--no-stateless-augmentation"], {"stateless_augmentation"}, True),
    ("boolean-optional-positive-abbreviated", True,
     ["--stateless-aug"], {"stateless_augmentation"}, True),
    ("boolean-optional-negative-abbreviated", True,
     ["--no-stateless-aug"], {"stateless_augmentation"}, True),
    ("two-flags-at-once", True,
     ["--batch-size", "8", "--seed", "7"], {"batch_size", "seed"}, True),
    # The bare `--` STOPS the scan: `--batch-size` after it is a positional
    # VALUE, not a typed flag, and reporting it would let a preset lose to a
    # word the caller never meant as a flag.
    ("bare-double-dash-stops-the-scan", True,
     ["--seed", "1", "--", "--batch-size", "8"], {"seed"}, True),
    # `-h` IS an exact match, so it is reported (dest `help`). Harmless: the
    # help action prints and exits, so `parse_args` never returns and no
    # caller ever consumes the report; and `help` is not a config field.
    ("single-dash-help-is-an-exact-match", True, ["-h"], {"help"}, False),
    # `--ema-w` prefixes BOTH `--ema-warmup-epochs` and `--ema-warmup-steps`.
    ("ambiguous-prefix-is-not-explicit", True, ["--ema-w", "1"], set(), False),
    # allow_abbrev is READ from the parser, never assumed. With it off, an
    # abbreviation is an argparse ERROR and must not count as typed.
    ("no-abbrev-parser-rejects-the-prefix", False,
     ["--batch-s", "8"], set(), False),
    ("no-abbrev-parser-rejects-the-abbreviated-equals-form", False,
     ["--ema-warmup-ep=1.5"], set(), False),
    ("no-abbrev-parser-still-sees-the-exact-flag", False,
     ["--batch-size", "8"], {"batch_size"}, True),
    ("no-abbrev-parser-still-sees-the-negative-boolean-spelling", False,
     ["--no-stateless-augmentation"], {"stateless_augmentation"}, True),
]

# Same shape, on a probe that registers SINGLE-dash options. Here `expected` is
# an exception TYPE: argparse accepts every one of these argvs, the token scan
# resolves none of the short forms, so the helper refuses the PARSER outright
# rather than return a wrong set. The third row shows the refusal is a property
# of the parser, not of the argv.
_SHORT_DASH_CASES = [
    ("single-dash-attached-value-is-refused-not-missed", True,
     ["-n8"], ValueError, True),
    ("single-dash-grouped-flags-are-refused-not-missed", True,
     ["-vn", "8"], ValueError, True),
    ("single-dash-registration-refuses-even-a-plain-long-flag", True,
     ["--batch-size", "8"], ValueError, True),
]


@pytest.mark.parametrize(
    "allow_abbrev,argv,expected,reachable,short_options",
    [c[1:] + (False,) for c in _CASES]
    + [c[1:] + (True,) for c in _SHORT_DASH_CASES],
    ids=[c[0] for c in _CASES + _SHORT_DASH_CASES],
)
def test_explicitly_set_flags_reports_exactly_the_documented_shapes(
        allow_abbrev, argv, expected, reachable, short_options):
    from train.common.args import explicitly_set_flags

    parser = _probe_parser(
        allow_abbrev=allow_abbrev, short_options=short_options)
    if isinstance(expected, type):
        with pytest.raises(expected, match="single-dash"):
            explicitly_set_flags(parser, argv)
    else:
        assert explicitly_set_flags(parser, argv) == expected

    if reachable:
        # The row is a real vector, not a look-alike: argparse accepts it.
        parser.parse_args(argv)


def test_the_real_dino_parser_agrees_on_its_own_prefix_pairs():
    """Same helper, a REAL trainer parser, on the shapes a probe cannot fake.

    ``train_dino``'s parser carries three strict prefix pairs — ``--seed`` is a
    prefix of ``--seed-training-stream``, ``--teacher-temp`` of both
    ``--teacher-temp-final`` and ``--teacher-temp-warmup-epochs``. An EXACT
    match must win over being the prefix of a longer option, exactly as
    argparse resolves it, or typing ``--seed 42`` would report a flag the
    caller never passed.

    ``parse_arguments`` attaches the helper's result to the Namespace as
    ``explicit_flags``, so this drives the shipped wiring, not a re-creation
    of it.
    """
    from train.dino.train_dino import parse_arguments

    assert parse_arguments(["--seed", "42"]).explicit_flags == {"seed"}
    assert parse_arguments(["--teacher-temp", "0.04"]).explicit_flags == {
        "teacher_temp"}
    assert parse_arguments(["--seed-training-stream"]).explicit_flags == {
        "seed_training_stream"}
    assert parse_arguments(["--no-seed-training-stream"]).explicit_flags == {
        "seed_training_stream"}
    assert parse_arguments(["--ema-warmup-ep", "1.0"]).explicit_flags == {
        "ema_warmup_epochs"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
