"""The session-level ``results/`` guard fails a session that adds a name, and deletes nothing.

Guard for ``tests/results_session_guard.py`` (plan-2026-10-14T062616-81938e81
D-048, D-084). Each case runs a CHILD pytest in ``tmp_path`` with the guard
loaded as a plugin and its watched root pointed at a ``tmp_path`` directory
through ``DL_TECHNIQUES_RESULTS_GUARD_ROOT``. Nothing here touches the real
``results/``: the planted writes go to the tmp root, and the guard is proven
on that root, never on the real one.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.results_session_guard import ROOT_ENV

REPO_ROOT = Path(__file__).resolve().parent.parent

WRITES_IN_A_TEST = """
    import os
    def test_writes():
        os.makedirs(os.path.join(os.environ["{env}"], "run_from_a_test"))
"""

WRITES_IN_A_MODULE_FIXTURE = """
    import os
    import pytest

    @pytest.fixture(scope="module")
    def trained():
        os.makedirs(os.path.join(os.environ["{env}"], "run_from_a_module_fixture"))

    def test_uses_it(trained):
        pass
"""

WRITES_NOTHING = """
    def test_nothing():
        pass
"""


def _child(tmp_path: Path, body: str, preexisting=()) -> tuple:
    watched = tmp_path / "watched"
    watched.mkdir()
    for name in preexisting:
        (watched / name).mkdir()
    work = tmp_path / "work"
    work.mkdir()
    (work / "pytest.ini").write_text("[pytest]\n")
    (work / "test_planted.py").write_text(textwrap.dedent(body.format(env=ROOT_ENV)))
    env = dict(os.environ)
    env[ROOT_ENV] = str(watched)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.results_session_guard",
         "-p", "no:cacheprovider", "-q", "--rootdir", str(work), "-c",
         str(work / "pytest.ini"), str(work / "test_planted.py")],
        cwd=work, env=env, capture_output=True, text=True, timeout=300,
    )
    return proc, watched


@pytest.mark.parametrize("body, name", [
    (WRITES_IN_A_TEST, "run_from_a_test"),
    (WRITES_IN_A_MODULE_FIXTURE, "run_from_a_module_fixture"),
], ids=["in_a_test", "in_a_module_fixture"])
def test_a_session_that_adds_a_name_fails_and_the_name_survives(tmp_path, body, name):
    proc, watched = _child(tmp_path, body)
    output = proc.stdout + proc.stderr
    assert proc.returncode == pytest.ExitCode.TESTS_FAILED, output
    assert "1 passed" in output, f"the planted test itself should pass:\n{output}"
    assert "RESULTS GUARD" in output and name in output, output
    # Read-only: the guard reports the name and leaves it in place.
    assert (watched / name).is_dir()


def test_a_clean_session_passes_and_preexisting_names_are_not_reported(tmp_path):
    proc, _ = _child(tmp_path, WRITES_NOTHING, preexisting=("old_run_a", "old_run_b"))
    output = proc.stdout + proc.stderr
    assert proc.returncode == pytest.ExitCode.OK, output
    assert "RESULTS GUARD" not in output, output


def test_the_watched_root_defaults_to_the_repo_results(monkeypatch):
    from tests.results_session_guard import results_root

    monkeypatch.delenv(ROOT_ENV, raising=False)
    assert results_root() == REPO_ROOT / "results"
    monkeypatch.setenv(ROOT_ENV, "/probe/root")
    assert results_root() == Path("/probe/root")


def test_the_guard_module_never_deletes():
    """A static belt: no deleting call appears in the guard's source."""
    source = (REPO_ROOT / "tests" / "results_session_guard.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    for banned in ("rmtree", "os.remove(", "os.unlink(", ".unlink(", "os.rmdir(",
                   ".rmdir(", "shutil.move(", "os.rename(", ".rename("):
        assert banned not in code, f"the results guard must stay read-only: found {banned}"
