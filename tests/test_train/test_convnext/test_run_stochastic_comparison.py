"""The stochastic-mode driver forwards ``--strides`` only when the user gave it.

Its own ``--strides`` default used to be a hard-coded 4 that shadowed the trainer's
(the help text also wrongly said 4 "crashes"); after D-033 the trainer default (2) is
the one source, so an absent flag must not reach the child command line at all.
"""

from __future__ import annotations

import os
import subprocess
from typing import List

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

import train.convnext.run_stochastic_comparison as driver  # noqa: E402


class _Stop(Exception):
    """Raised by the fake ``subprocess.run`` so no training is launched."""


def _first_command(monkeypatch, argv: List[str]) -> List[str]:
    seen: List[List[str]] = []

    def fake_run(cmd, *args, **kwargs):
        seen.append(list(cmd))
        raise _Stop

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = driver.build_argument_parser().parse_args(argv)
    with pytest.raises(_Stop):
        driver.run_comparison(args)
    return seen[0]


def test_the_driver_has_no_strides_default_of_its_own() -> None:
    assert driver.build_argument_parser().parse_args([]).strides is None


def test_an_absent_strides_flag_is_not_forwarded_to_the_trainer(monkeypatch) -> None:
    cmd = _first_command(monkeypatch, [])
    assert "--strides" not in cmd, cmd


def test_an_explicit_strides_flag_is_forwarded_verbatim(monkeypatch) -> None:
    cmd = _first_command(monkeypatch, ["--strides", "4"])
    assert cmd[cmd.index("--strides") + 1] == "4", cmd


def test_the_strides_help_no_longer_claims_that_strides_4_crashes() -> None:
    text = driver.build_argument_parser().format_help()
    assert "crashes" not in text and "collapses spatial dims" not in text
