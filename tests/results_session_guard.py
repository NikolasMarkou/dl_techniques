"""Session-level guard: a pytest session may not add a name to ``results/``.

READ-ONLY, ALWAYS. Nothing in this module deletes, moves, renames or cleans
anything, under ``results/`` or anywhere else. ``results/`` is gitignored and
untracked, so a deletion there is unrecoverable; on 2026-08-12 a cleanup step
that named that directory destroyed 62 run directories. When this guard
fires, the repair is to route the offending test through ``tmp_path``; the
reported directory is removed BY A HUMAN, if at all.

Why a session guard on top of the per-test ``no_repo_root_results_writes``
fixture in ``tests/conftest.py``: that fixture snapshots around each test's
own setup/call/teardown window, so it cannot see a write made by a
module- or session-scoped fixture set up before its snapshot, by an import
at collection time, or by anything that runs between tests. This guard
snapshots once at ``pytest_sessionstart`` and compares once at
``pytest_sessionfinish``, and fails the session (exit status 1) if any name
was added (plan-2026-10-14T062616-81938e81 D-048, D-084).

The watched root is ``<repo>/results`` unless the environment variable
:data:`ROOT_ENV` names another directory. That override exists so the guard
itself can be RED-proven against a ``tmp_path`` root without anything ever
writing into the real ``results/``.

``tests/conftest.py`` imports :func:`pytest_sessionstart` and
:func:`pytest_sessionfinish` (which registers them), and the per-test fixture
reads the same :func:`results_root`/:func:`results_entries` pair, so the two
guards always watch the same directory. The guard's own test loads this module
as a plugin (``-p tests.results_session_guard``) in a child pytest.
"""

import os
from pathlib import Path

import pytest

#: Environment variable that points the guards at another root (tests only).
ROOT_ENV = "DL_TECHNIQUES_RESULTS_GUARD_ROOT"

REPO_ROOT = Path(__file__).resolve().parent.parent

_SNAPSHOT_KEY = pytest.StashKey[frozenset]()
_ROOT_KEY = pytest.StashKey[Path]()


def results_root() -> Path:
    """The directory both guards watch.

    Interface contract (callers: this module's hooks, the per-test fixture in
    ``tests/conftest.py``): returns ``Path(os.environ[ROOT_ENV])`` when that
    variable is set and non-empty, else ``<repo>/results``. Never creates the
    directory.

    :return: The watched directory.
    """
    override = os.environ.get(ROOT_ENV, "")
    return Path(override) if override else REPO_ROOT / "results"


def results_entries(root: Path) -> frozenset:
    """Names directly under ``root``; empty if it is absent or unreadable.

    Interface contract (same callers): one ``os.scandir`` of ONE directory,
    no recursion and no ``stat``, because the per-test fixture calls it twice
    per test for the whole suite. Read-only.

    :param root: Directory to list.
    :return: Entry names.
    """
    try:
        with os.scandir(root) as it:
            return frozenset(entry.name for entry in it)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return frozenset()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Snapshot the watched root's names once, before collection."""
    root = results_root()
    session.stash[_ROOT_KEY] = root
    session.stash[_SNAPSHOT_KEY] = results_entries(root)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the session if the watched root gained a name. Never deletes."""
    if _SNAPSHOT_KEY not in session.stash:
        return
    root = session.stash[_ROOT_KEY]
    added = sorted(results_entries(root) - session.stash[_SNAPSHOT_KEY])
    if not added:
        return
    message = (
        f"RESULTS GUARD: this pytest session added {len(added)} name(s) to "
        f"{root}: {added}. A trainer's relative output dir resolves against "
        "the working directory; route the test through tmp_path. These "
        "entries are NOT deleted by the guard -- do NOT delete them from a "
        "script either; results/ is untracked and a deletion is permanent."
    )
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(message, red=True, bold=True)
    else:
        print(message)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
