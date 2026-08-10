"""Executable enforcement of `REPO_MAP.md`'s "Numbers, and how to re-derive them".

`REPO_MAP.md` claims every digit in its prose is a Value from that table and is
"mechanically enforced". Until this file, that enforcement was an ad-hoc script
somebody re-typed each time the map changed — the exact rot that already killed
`verify_map.py` (see `REPO_MAP.md`'s own note, and `plans/LESSONS.md`). Landing
it as a test is the point: a checker nobody runs enforces nothing.

What it does — literally what REPO_MAP.md's § "The cheap sweep" prescribes:
extract every ``| Quantity | Value | `command` |`` row whose Value is all
digits, run the command with ``bash -c`` from the repo root, and compare.

Failure means one of two things, and the message says which is which:

* the repo changed and the map did not — fix `REPO_MAP.md`;
* the command in the map is wrong — fix the command.

Do NOT "fix" a failure by editing the expected Value alone: REPO_MAP.md warns
that a prose digit and its table Value must move in the SAME edit, and that a
single directory addition has moved 19 of the ~68 rows at once. Re-derive the
WHOLE table.

Landed 2026-08-10 by plan-2026-08-10-3649c19e/iter-2/step-13 (decisions.md
D-032), from the throwaway checker step 11 used to reach 68/68.
"""

from __future__ import annotations

import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_MAP = os.path.join(REPO_ROOT, "REPO_MAP.md")

# `| Quantity | Value | `command` |` — Value must be all digits, and the command
# cell is a single backticked run of shell. Rows whose Value is prose ("two",
# "59 / 7.3G") are deliberately NOT enforceable and are skipped by this regex.
_ROW = re.compile(
    r"^\|\s*(?P<quantity>[^|]+?)\s*\|\s*(?P<value>\d+)\s*\|\s*`(?P<cmd>.+?)`\s*\|\s*$"
)

# A quantity may legitimately be tabulated more than once (a Numbers row and a
# subpackage-table row); pytest ids must still be unique.
_MIN_ROWS = 50


def _rows():
    """Parse the enforceable rows out of REPO_MAP.md."""
    out = []
    with open(REPO_MAP, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            match = _ROW.match(line.rstrip("\n"))
            if match is None:
                continue
            out.append(
                (
                    lineno,
                    match.group("quantity"),
                    int(match.group("value")),
                    # The table escapes pipes for Markdown; unescape for bash.
                    match.group("cmd").replace(r"\|", "|"),
                )
            )
    return out


ROWS = _rows()


def test_the_table_is_still_parseable():
    """Anti-vacuity guard: an empty parse must never read as "all rows pass".

    If a future Markdown reformat changes the row shape, this test fails
    LOUDLY instead of the parametrized sweep below silently collecting zero
    cases.
    """
    assert len(ROWS) >= _MIN_ROWS, (
        f"only {len(ROWS)} enforceable rows parsed out of REPO_MAP.md "
        f"(expected >= {_MIN_ROWS}). Either the Numbers table lost most of its "
        "rows, or its `| Quantity | Value | `command` |` shape changed and the "
        "parser in this file needs updating. Do not lower the floor to make "
        "this pass."
    )


@pytest.mark.parametrize(
    "lineno,quantity,expected,cmd",
    ROWS,
    ids=[f"L{r[0]}:{r[1][:48]}" for r in ROWS],
)
def test_repo_map_number_reproduces(lineno, quantity, expected, cmd):
    """Each tabulated number must equal what its own command prints."""
    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"REPO_MAP.md:{lineno} — the command for {quantity!r} exited "
        f"{proc.returncode}:\n  {cmd}\n{proc.stderr[-800:]}"
    )
    printed = proc.stdout.strip()
    assert printed.isdigit(), (
        f"REPO_MAP.md:{lineno} — the command for {quantity!r} did not print a "
        f"bare number:\n  {cmd}\n  got: {printed[:200]!r}"
    )
    assert int(printed) == expected, (
        f"REPO_MAP.md:{lineno} — {quantity!r} is tabulated as {expected} but "
        f"its own command re-derives {int(printed)}.\n  {cmd}\n"
        "Either the repo changed and the map did not, or the command is wrong. "
        "Re-derive the WHOLE table, not just this row (REPO_MAP.md § 'Numbers': "
        "one src/train/ addition once moved 19 of the 68 rows), and move any "
        "prose digit sourced from this row in the SAME edit."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
