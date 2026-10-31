"""The run lock is a mutex the kernel arbitrates, not a file that two resumes can both replace (D-011).

The lock of the earlier design was read, replaced and read back, so two processes that read
the same stale ``run.lock`` before either replaced it could both end up running in one
directory. These tests run REAL processes:

* eight processes take one stale lock at the same instant, twenty times: exactly one holds it
  each round (the instant is a shared wall-clock time every child spins to, so the
  read-replace-read window is hit by all of them at once);
* a SIGKILLed holder's lock is taken over at once, with no wait and no liveness guess;
* a pid that another live process now owns does not block a resume;
* the descriptor is released at the end of a run, also when the run raised, and is not
  inherited across ``exec``;
* a filesystem where ``flock`` fails is a ``ConfigError``.

Tests write only under ``tmp_path``.
"""

import errno
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import keras
import pytest

from train.api import run_dir as run_dir_module
from train.api.errors import ConfigError
from train.api.run_dir import LOCK_NAME, process_stamp, read_lock, refuse_live_holder, take_run_lock

from .test_resume_interrupted import (  # noqa: F401  (fixtures are used by name)
    REQUEST,
    Scripted,
    killed,
    whole,
    world,
)
from .test_task_trainer import make_config

ROUNDS = 20
PROCESSES = 8

#: A child: import first (the slow part), report ready, read the shared start instant, spin
#: to it, take the lock, report the outcome, then keep the handle until the parent says stop.
RACER = textwrap.dedent(
    """
    import sys
    import time
    from pathlib import Path

    from train.api.errors import ConfigError
    from train.api.run_dir import take_run_lock

    directory = Path(sys.argv[1])
    print("ready", flush=True)
    start = float(sys.stdin.readline())
    while time.time() < start:
        pass
    try:
        handle = take_run_lock(directory, resume=True)
    except ConfigError as exc:
        print("refused", flush=True)
    else:
        print("held", flush=True)
        sys.stdin.readline()
    """
)

HOLDER = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from train.api.run_dir import take_run_lock

    handle = take_run_lock(Path(sys.argv[1]), resume=sys.argv[2] == "resume")
    print("held", flush=True)
    sys.stdin.read()
    """
)


def dead_pid() -> int:
    """The pid of a process that ran and was reaped (it may be reused much later, not within a test)."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=60)
    return process.pid


def plant_stale_lock(directory: Path) -> None:
    """A ``run.lock`` whose holder is gone: what a killed run leaves."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / LOCK_NAME).write_text(
        json.dumps({"pid": dead_pid(), "start": None, "created": "2026-01-01T00:00:00+0000"})
    )


def start_holder(directory: Path, mode: str = "fresh") -> subprocess.Popen:
    process = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(directory), mode],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert process.stdout.readline().strip() == "held", process.stderr.read()
    return process


def finish(process: subprocess.Popen) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except BrokenPipeError:  # the child already exited with our last line unread
                pass
    process.wait(timeout=60)


def race(directory: Path) -> list:
    """Start ``PROCESSES`` racers on ``directory``; return their outcomes (``held`` / ``refused``)."""
    racers = [
        subprocess.Popen(
            [sys.executable, "-c", RACER, str(directory)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(PROCESSES)
    ]
    try:
        for racer in racers:
            assert racer.stdout.readline().strip() == "ready", racer.stderr.read()
        start = time.time() + 0.3
        for racer in racers:
            racer.stdin.write(f"{start}\n")
            racer.stdin.flush()
        outcomes = []
        for racer in racers:
            line = racer.stdout.readline().strip()
            assert line in ("held", "refused"), (line, racer.stderr.read())
            outcomes.append(line)
        return outcomes
    finally:
        for racer in racers:
            try:
                racer.stdin.write("stop\n")
                racer.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass
        for racer in racers:
            finish(racer)


def test_eight_processes_on_one_stale_lock_leave_exactly_one_holder_in_every_round(tmp_path):
    held_per_round = []
    for index in range(ROUNDS):
        directory = tmp_path / f"round_{index}"
        plant_stale_lock(directory)
        outcomes = race(directory)
        held_per_round.append(outcomes.count("held"))
        assert outcomes.count("refused") == PROCESSES - outcomes.count("held")
    assert held_per_round == [1] * ROUNDS, held_per_round


def test_a_sigkilled_holder_lets_the_next_process_take_over_at_once(tmp_path):
    holder = start_holder(tmp_path)
    try:
        with pytest.raises(ConfigError, match=rf"process {holder.pid} holds run\.lock"):
            take_run_lock(tmp_path, resume=True)
        holder.kill()
        holder.wait(timeout=60)  # killed and reaped; the kernel released the lock when it died
        taken = take_run_lock(tmp_path, resume=True)
        try:
            assert read_lock(tmp_path)["pid"] == os.getpid()
        finally:
            taken.close()
    finally:
        finish(holder)


def test_a_sigkilled_holder_that_nobody_reaped_has_already_lost_the_lock(tmp_path):
    holder = start_holder(tmp_path)
    holder.kill()
    deadline = time.time() + 30
    while True:
        try:
            taken = take_run_lock(tmp_path, resume=True)
            break
        except ConfigError:
            assert time.time() < deadline, "a killed process still holds the lock after 30 s"
            time.sleep(0.05)
    taken.close()
    finish(holder)


def test_a_lock_naming_a_live_unrelated_process_does_not_block_a_resume(tmp_path):
    """A pid is reused by another process: only the kernel's lock says who runs here."""
    parent = os.getppid()
    (tmp_path / LOCK_NAME).write_text(
        json.dumps({"pid": parent, "start": process_stamp(parent), "created": "2026-01-01T00:00:00+0000"})
    )
    refuse_live_holder(tmp_path)  # does not raise: nobody holds the lock
    taken = take_run_lock(tmp_path, resume=True)
    try:
        assert read_lock(tmp_path)["pid"] == os.getpid()
    finally:
        taken.close()


def test_a_second_resume_in_this_process_is_refused_while_the_first_holds_the_lock(tmp_path):
    first = take_run_lock(tmp_path, resume=False)
    try:
        with pytest.raises(ConfigError, match="holds run\\.lock"):
            take_run_lock(tmp_path, resume=True)
        with pytest.raises(ConfigError, match="holds run\\.lock"):
            refuse_live_holder(tmp_path)
    finally:
        first.close()
    refuse_live_holder(tmp_path)  # released: the next one may come
    take_run_lock(tmp_path, resume=True).close()


def test_close_is_idempotent_and_a_with_block_releases(tmp_path):
    with take_run_lock(tmp_path, resume=False) as lock:
        with pytest.raises(ConfigError, match="holds run\\.lock"):
            refuse_live_holder(tmp_path)
        lock.close()
        lock.close()
        refuse_live_holder(tmp_path)  # closed inside the block: released
    with take_run_lock(tmp_path, resume=True):
        pass
    refuse_live_holder(tmp_path)  # and released again at the end of the block


def test_a_dropped_handle_releases_the_lock(tmp_path):
    take_run_lock(tmp_path, resume=False)  # the handle is dropped at once
    take_run_lock(tmp_path, resume=True).close()


def test_the_descriptor_is_not_inherited_across_exec(tmp_path):
    """A child started with ``close_fds=False`` must not keep the lock alive after we close it."""
    lock = take_run_lock(tmp_path, resume=False)
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE, close_fds=False,
    )
    try:
        lock.close()
        take_run_lock(tmp_path, resume=True).close()  # would be refused if the child shared the lock
    finally:
        child.stdin.close()
        child.wait(timeout=60)


def test_a_filesystem_where_flock_fails_is_a_config_error_naming_the_lock(tmp_path, monkeypatch):
    def refuse(descriptor, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(run_dir_module.fcntl, "flock", refuse)
    for resume in (False, True):
        directory = tmp_path / f"resume_{resume}"
        directory.mkdir()
        with pytest.raises(ConfigError, match=r"run\.lock.*flock.*No locks available"):
            take_run_lock(directory, resume=resume)
    with pytest.raises(ConfigError, match=r"run\.lock.*flock"):
        refuse_live_holder(directory)


def test_the_lock_is_released_when_a_run_ends_normally_and_when_it_raises(world, whole, killed):
    _, finished = whole
    take_run_lock(finished, resume=True).close()  # a finished run in this very process
    take_run_lock(killed, resume=True).close()  # a run that raised in this very process


def test_two_runs_one_after_the_other_in_this_process_both_get_the_lock(world, tmp_path):
    for name in ("first", "second"):
        Scripted(make_config(tmp_path / name, **REQUEST)).run()
        take_run_lock(tmp_path / name, resume=True).close()


class ProbeTheLock(keras.callbacks.Callback):
    """At each epoch start, record whether another taker of the run's lock is refused."""

    def __init__(self, run_dir):
        super().__init__()
        self.run_dir = run_dir
        self.refused = []

    def on_epoch_begin(self, epoch, logs=None):
        try:
            take_run_lock(self.run_dir, resume=True).close()
        except ConfigError as exc:
            self.refused.append("holds run.lock" in str(exc))
        else:
            self.refused.append(False)


def test_the_run_holds_the_lock_from_the_first_epoch_to_the_last(world, tmp_path):
    run_dir = tmp_path / "held_during_the_run"
    probe = ProbeTheLock(run_dir)

    class Probed(Scripted):
        def extra_callbacks(self, config, data):
            return [probe]

    Probed(make_config(run_dir, **REQUEST)).run()
    assert probe.refused and all(probe.refused), probe.refused
    take_run_lock(run_dir, resume=True).close()  # and it is free once the run has ended
