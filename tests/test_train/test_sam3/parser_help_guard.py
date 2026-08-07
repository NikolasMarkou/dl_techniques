"""ONE home for the bare-``%`` help-string check, shared by BOTH SAM 3 CLIs.

Why this module exists rather than a copy in each test file. ``argparse`` runs
every ``help=`` string through ``help % params`` when ``--help`` is formatted, so
a lone ``%`` (as in "85% of the card") crashes ``--help`` -- and ONLY ``--help``.
No other code path formats those strings, so no other test can see it. The check
was written once for ``train_sam3.build_parser`` and a later step added a flag to
``train.sam3.baselines``, whose parser the check did not walk: a bare ``%`` in
``--prompt-swap``'s help made ``python -m train.sam3.baselines --help`` exit 1
with ``TypeError: %o format: an integer is required, not dict`` while
``pytest tests/test_train/test_sam3/test_baselines.py`` reported 72 passed.

So the check lives here, in one implementation, and each parser's own test file
calls it. Adding a THIRD parser is one call, not a fourth copy.

Interface contract: :func:`bare_percent_offenders` is pure and returns the list
of offending ``"<option strings>: <help text>"`` descriptions (empty when
clean); :func:`assert_no_bare_percent_help` is the NAMED assertion the RED proof
fires and raises ``AssertionError`` naming the parser and every offender.
``%%`` and the ``%(default)s`` / ``%(prog)s`` interpolations are legal argparse
spellings and are excluded by both.
"""

import argparse
from typing import List

#: Interpolations argparse resolves itself; these are NOT crashes.
_LEGAL_PERCENT_SPELLINGS = ("%%", "%(default)s", "%(prog)s")


def bare_percent_offenders(parser: argparse.ArgumentParser) -> List[str]:
    """Every action of ``parser`` whose ``help=`` carries a bare ``%``."""
    offenders = []
    for action in parser._actions:
        text = action.help or ""
        stripped = text
        for legal in _LEGAL_PERCENT_SPELLINGS:
            stripped = stripped.replace(legal, "")
        if "%" in stripped:
            offenders.append(f"{action.option_strings}: {text!r}")
    return offenders


def assert_no_bare_percent_help(parser: argparse.ArgumentParser,
                                parser_name: str) -> None:
    """THE named assertion, in one place so a RED proof can name it."""
    offenders = bare_percent_offenders(parser)
    assert not offenders, (
        f"{parser_name} has help strings with a bare '%', which argparse "
        f"formats at --help time and would CRASH --help on: {offenders}")
