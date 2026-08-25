"""``--help`` on both ColBERT entry points must print usage and train NOTHING.

Why a subprocess, and why not an exit-code check
------------------------------------------------

A green import proves nothing about ``--help``: the failure this guards is a
module whose ``main()`` does work BEFORE parsing. ``src/train/CLAUDE.md`` records
two measured instances -- ``bert/wikipedia/*`` reached ``MirroredStrategy`` and a
full TFDS dataset build, and ``train.tabm.train_tabm`` ran all five example
pipelines to completion and **exited 0 with no usage line**, so a repo-wide
exit-code sweep read it as healthy. Exit 0 is therefore not a passing ``--help``;
the assertions here are on the ARTEFACTS and on the OUTPUT.

The detector, and its liveness arm
----------------------------------

Each ``--help`` subprocess runs with its working directory inside ``tmp_path``,
so the trainers' default ``--output-root results`` would materialise
``<cwd>/results/`` if training started. The detector is therefore:

1. exit status 0,
2. a ``usage:`` line on stdout,
3. no ``<cwd>/results/`` directory,
4. none of the trainer's own run banners in the output.

An absence assertion with no liveness arm passes on a broken detector, so
:func:`test_a_real_run_does_trip_every_absence_signal` runs the SAME machinery
against a real (tiny) training invocation and requires signals 3 and 4 to fire
and the usage line to be absent. That arm is the reason the four assertions above
mean anything.

Repo-root ``results/`` is never involved: every subprocess is chdir'd into
``tmp_path``, and ``tests/conftest.py``'s autouse fixture errors in teardown if
anything lands in the real tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import pytest

MODULES = (
    "train.language.colbert.train_colbert_v1",
    "train.language.colbert.train_colbert_v2",
)

#: Log lines only a started run emits. Both trainers log a banner before the
#: first ``fit`` step, and ``build_datasets`` logs the synthetic-corpus line.
RUN_BANNERS: Tuple[str, ...] = (
    "ColBERT v1 --",
    "ColBERT v2 --",
    "Synthetic data:",
    "Run directory:",
)


def _run(module: str, argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    """Execute ``python -m <module> <argv>`` in ``cwd`` on CPU.

    Interface contract (2 callers: the ``--help`` arms and the liveness arm,
    which must share one invocation shape or the liveness proof measures a
    different detector):

    :param module: Dotted module path passed to ``-m``.
    :param argv: Tokens after the module name.
    :param cwd: Working directory. A relative ``--output-root`` resolves here,
        which is what makes ``<cwd>/results`` a usable side-effect signal.
    :returns: The completed process, stdout and stderr captured as text.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", module, *argv],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def _side_effect_signals(result: subprocess.CompletedProcess, cwd: Path) -> List[str]:
    """Every training side effect visible in ``result`` / ``cwd``.

    :returns: Human-readable signal descriptions; empty when nothing trained.
    """
    output = result.stdout + result.stderr
    signals = []
    results_dir = cwd / "results"
    if results_dir.exists():
        signals.append(f"created {results_dir} containing {sorted(os.listdir(results_dir))}")
    for banner in RUN_BANNERS:
        if banner in output:
            signals.append(f"logged run banner {banner!r}")
    return signals


@pytest.mark.parametrize("module", MODULES, ids=["v1", "v2"])
def test_help_exits_zero_and_prints_usage(module, tmp_path) -> None:
    result = _run(module, ["--help"], tmp_path)
    assert result.returncode == 0, (
        f"{module} --help exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "usage:" in result.stdout, (
        f"{module} --help printed no 'usage:' line -- a script with no parser "
        f"ignores --help entirely and still exits 0.\nstdout:\n{result.stdout}"
    )


@pytest.mark.parametrize("module", MODULES, ids=["v1", "v2"])
def test_help_starts_no_training(module, tmp_path) -> None:
    result = _run(module, ["--help"], tmp_path)
    signals = _side_effect_signals(result, tmp_path)
    assert not signals, (
        f"{module} --help produced training side effects: {signals}. Parsing "
        f"must be the FIRST thing main() does."
    )


@pytest.mark.parametrize("module", MODULES, ids=["v1", "v2"])
def test_help_advertises_every_flag_the_wiring_table_maps(module, tmp_path) -> None:
    """A flag consumed by the config must be discoverable from ``--help``."""
    from train.language.colbert.train_colbert_v2 import V2_CLI_TO_CONFIG

    result = _run(module, ["--help"], tmp_path)
    v2_only = {"distillation_alpha"}
    dests = set(V2_CLI_TO_CONFIG)
    if module.endswith("v1"):
        dests -= v2_only
    missing = sorted(
        dest for dest in dests
        if "--" + dest.replace("_", "-") not in result.stdout
    )
    assert not missing, f"{module} --help does not mention {missing}"


def test_a_real_run_does_trip_every_absence_signal(tmp_path) -> None:
    """Liveness: the detector the three tests above rely on is not blind.

    Runs the v1 trainer for real at the smallest geometry the pipeline accepts
    and requires that the ``results/`` directory appears, that the run banners
    are logged and that no ``usage:`` line is printed. Without this arm, the
    absence assertions would pass against a detector that could never fire.
    """
    result = _run(
        MODULES[0],
        ["--smoke", "--num-train-groups", "2", "--num-val-groups", "2"],
        tmp_path,
    )
    assert result.returncode == 0, (
        f"the liveness run failed ({result.returncode}); it cannot prove the "
        f"detector fires\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    signals = _side_effect_signals(result, tmp_path)
    assert any(s.startswith("created") for s in signals), (
        f"a real training run left no results/ directory under {tmp_path}; the "
        f"artefact signal in the --help tests is blind. Signals: {signals}"
    )
    assert any("banner" in s for s in signals), (
        f"a real training run logged none of {RUN_BANNERS}; the log signal in "
        f"the --help tests is blind"
    )
    assert "usage:" not in result.stdout, (
        "a real training run printed a 'usage:' line, so that assertion cannot "
        "distinguish --help from training"
    )
