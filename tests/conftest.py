import os
import sys
from pathlib import Path

import pytest

# Add src to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )


# --- Golden-value reference device -----------------------------------------
#
# The single source of truth for the device a stored-reference ("golden value")
# probe must be built and run on. Centralized here because the *policy* and its
# justification must not be restated per module: there are already two golden
# modules (``test_models/test_scunet/test_golden_values.py`` and
# ``test_models/test_swin_transformer/test_golden_values.py``) with two pinned
# sites each, and a per-file copy would be four places to drift.
#
# WHY A PIN IS REQUIRED, measured (not assumed) on GPU 1 (RTX 4070) at the Swin
# golden config, GPU output vs the CPU reference:
#
#   | regime                    | max|diff| vs CPU |
#   |---------------------------|------------------|
#   | float32, TF32 ON (default)| 2.254173e-05     |
#   | float32, TF32 OFF         | 1.490116e-08     |
#   | float64                   | 2.081668e-17     |
#
# So the deviation is ordinary reduced-precision matmul reassociation -- TF32
# accounts for ~99.93% of it -- and NOT a device-dependent difference in the
# Swin code path. It is still fatal to these guards: the signal they exist to
# catch is 1.04e-07 (the D-006 single-window-rule neuter), so their tolerance is
# 1e-8, and even the TF32-OFF residual exceeds it. Widening the tolerance to
# accommodate a GPU would make the guards blind to exactly what they watch.
#
# WHAT NOT TO DO: do not "fix" a golden-value failure on GPU by relaxing atol,
# and do not skip these modules when a GPU is visible -- a skipped guard on the
# machine the models are developed on is not a guard. Pin the probe instead.
# The cost is recorded honestly: these guards pin CPU numerics only.
GOLDEN_REFERENCE_DEVICE = "cpu"


@pytest.fixture(scope="session")
def golden_reference_device() -> str:
    """Device string that golden-value probes must build and run inside.

    Use as ``with keras.device(golden_reference_device): ...`` at BOTH the
    reference-producing site and any site whose output is compared against it --
    pinning only one of the two makes the comparison cross-device, which is the
    failure this fixture exists to prevent.

    :return: a ``keras.device``-compatible device string (never ``None``).
    """
    return GOLDEN_REFERENCE_DEVICE


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# --- D-1: no test may write into the repo-root results/ directory ----------
#
# The defect CLASS, not one instance. Trainer entry points resolve a relative
# `output_dir` (typically the literal default `"results"`) against the REPO
# ROOT, so any test that calls a real trainer -- or a real `ModelCheckpoint`,
# or a real `model.save()` through one -- deposits a run directory into the
# user's `results/` tree. That tree is gitignored AND untracked, so anything
# that lands there is indistinguishable from a real experiment, and anything
# deleted from there is UNRECOVERABLE (62 run directories, including a
# published paper's subject checkpoint, were destroyed exactly once this way).
#
# `tests/test_train/test_sam3/test_train_sam3.py` carried this snapshot inline
# in a single test body. It is promoted here so the next test anywhere in the
# suite that reaches a trainer entry point is watched by default rather than by
# remembering.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


def _results_entries() -> frozenset:
    """Names directly under the repo-root ``results/``; empty if absent.

    One ``os.scandir`` on ONE directory, no recursion and no ``stat``, because
    this runs twice per test for the whole suite.

    :return: entry names, or an empty frozenset when ``results/`` does not
        exist (a fresh clone) or is unreadable.
    """
    try:
        with os.scandir(RESULTS_ROOT) as it:
            return frozenset(entry.name for entry in it)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return frozenset()


@pytest.fixture(autouse=True)
def no_repo_root_results_writes():
    """Fail any test that adds an entry to the repo-root ``results/``.

    ASSERT-ONLY, AND THIS IS NOT NEGOTIABLE. This fixture MUST NEVER delete,
    move, truncate, rename or otherwise "clean up" anything under ``results/``
    -- not even an entry it can prove the test under it just created, and not
    in a ``finally``. ``results/`` is gitignored and untracked: there is no
    history and no backup, so a deletion here is permanent, and a cleanup step
    that names that directory is precisely how 62 run directories (including a
    published paper's subject checkpoint) were destroyed on 2026-08-12. When
    this assertion fires, the repair is to route the test's config through
    ``tmp_path``; the reported directory is then removed BY A HUMAN, if at all.

    Failing rather than skipping is deliberate: a run directory in ``results/``
    is not a cosmetic mess, it is an artefact a later human reads as an
    experiment.

    :raises AssertionError: if new entries appear directly under ``results/``
        while the test runs.
    """
    before = _results_entries()
    yield
    new = sorted(_results_entries() - before)
    assert not new, (
        f"this test wrote into the repo-root results/ directory: {new}. "
        "A trainer's relative output_dir resolves against the repo root; route "
        "the config through tmp_path (e.g. dataclasses.replace(config, "
        "output_dir=str(tmp_path))). results/ is gitignored and untracked -- do "
        "NOT delete what this assertion reports, and do NOT make this fixture "
        "clean up after itself.")
