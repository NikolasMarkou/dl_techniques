"""``run.lock`` that is a link, a FIFO or a directory is refused, never followed or waited on (review-iter-2 concern 17).

The lock trusted its path. Reproduced before the change (``probes/t7b.py``, ``t11.py``):

* a SYMLINKED ``run.lock`` passed ``open_run`` and ``take_run_lock`` and the payload (pid, start,
  created) was written THROUGH it over the target's contents, so a run directory with a link in it
  edited the file the link named;
* a FIFO blocked ``refuse_live_holder`` (``open(O_RDONLY)`` waits for a writer) forever;
* a DIRECTORY opened read-only without complaint, so ``refuse_live_holder`` let a resume go on.

Now every open is ``O_NOFOLLOW | O_NONBLOCK`` followed by an ``fstat`` that wants a regular file,
and a refusal is a ``ConfigError`` that names the lock and what it is. A regular lock, held or
stale, and an absent one behave as before (the other run-lock files pin that).

Tests write only under ``tmp_path``; a FIFO test is bounded by an alarm so a regression fails
instead of hanging.
"""

import contextlib
import os
import signal
from pathlib import Path

import pytest

from train.api.errors import ConfigError
from train.api.run_dir import LOCK_NAME, read_lock, refuse_live_holder, take_run_lock

VICTIM = "precious contents " * 20


@contextlib.contextmanager
def within(seconds):
    """Fail with a ``TimeoutError`` if the block blocks for ``seconds`` (a FIFO waiting for a peer)."""

    def alarm(*_):
        raise TimeoutError(f"blocked more than {seconds} s")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def odd_lock(tmp_path, kind):
    run = tmp_path / "run"
    run.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text(VICTIM)
    lock = run / LOCK_NAME
    if kind == "symlink":
        lock.symlink_to(victim)
    elif kind == "dangling":
        lock.symlink_to(tmp_path / "nowhere")
    elif kind == "fifo":
        os.mkfifo(lock)
    elif kind == "directory":
        lock.mkdir()
    return run, victim


KINDS = ["symlink", "dangling", "fifo", "directory"]
WORDS = {"symlink": "symbolic link", "dangling": "symbolic link", "fifo": "not a regular file",
         "directory": "not a regular file"}


@pytest.mark.parametrize("kind", KINDS)
def test_the_probe_a_resume_makes_refuses_an_odd_lock_and_does_not_wait(tmp_path, kind):
    run, victim = odd_lock(tmp_path, kind)
    with within(20), pytest.raises(ConfigError) as caught:
        refuse_live_holder(run)
    assert str(run / LOCK_NAME) in str(caught.value) and WORDS[kind] in str(caught.value)
    assert victim.read_text() == VICTIM


@pytest.mark.parametrize("resume", [True, False], ids=["resume", "fresh"])
@pytest.mark.parametrize("kind", KINDS)
def test_taking_the_lock_refuses_an_odd_lock_and_writes_nothing_through_it(tmp_path, kind, resume):
    """Mutant ``no O_NOFOLLOW``: the payload was written over the symlink target."""
    run, victim = odd_lock(tmp_path, kind)
    with within(20), pytest.raises(ConfigError) as caught:
        take_run_lock(run, resume)
    assert LOCK_NAME in str(caught.value)
    if resume:
        # The wording matters: a FIFO that blocks until the alarm raises ``TimeoutError``, an ``OSError``
        # the old code turned into a ConfigError too ("Cannot open ...: timed out"); only the new
        # check says WHAT the lock is.
        assert WORDS[kind] in str(caught.value), str(caught.value)
    assert victim.read_text() == VICTIM


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_reading_the_label_of_an_odd_lock_is_refused_not_followed(tmp_path, kind):
    run, _ = odd_lock(tmp_path, kind)
    with within(20), pytest.raises(ConfigError, match=LOCK_NAME) as caught:
        read_lock(run)
    assert WORDS[kind] in str(caught.value), str(caught.value)  # not a timeout turned into a ConfigError


def test_open_run_refuses_a_resume_whose_lock_is_a_link(tmp_path):
    from train.api.run_dir import open_run

    run, victim = odd_lock(tmp_path, "symlink")
    (run / "checkpoints" / "backup").mkdir(parents=True)
    with within(20), pytest.raises(ConfigError, match="symbolic link"):
        open_run(run, True)
    assert victim.read_text() == VICTIM


def test_a_regular_lock_is_still_taken_probed_and_read(tmp_path):
    """Control: every refusal above is the same call on a lock that is not a regular file."""
    run = tmp_path / "run"
    run.mkdir()
    refuse_live_holder(run)  # absent: nobody holds it
    with take_run_lock(run, False):
        assert read_lock(run)["pid"] == os.getpid()
        with pytest.raises(ConfigError, match="holds run.lock"):
            refuse_live_holder(run)
    refuse_live_holder(run)  # released: the file stays, nobody holds it
    with take_run_lock(run, True):
        pass
    assert isinstance(Path(run / LOCK_NAME).stat().st_size, int)
