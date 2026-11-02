"""A refused run lock says what is true and names the lock, whatever failed (review-iter-1 concerns 5, 6, 16).

Three things the earlier lock got wrong, each reproduced with real processes and real files:

* **Concern 5.** ``open_run`` looked for the backup and for ``fit_finished`` BEFORE it asked who
  holds the lock, so a second process on a run that was alive said "killed before its first
  epoch ended" (first-epoch window) or "stopped early or crashed while reporting" (reporting
  window) and told the user to abandon a healthy run. The lock is asked first, in both windows.
* **Concern 6.** An open, a write or a read of ``run.lock`` that failed (read-only file or
  directory, a full disk, a payload that is not UTF-8) reached the user as a raw
  ``PermissionError`` / ``OSError`` / ``UnicodeDecodeError``, a failed write left a 0-byte lock,
  and a non-UTF-8 payload broke the refusal path itself. Each is a ``ConfigError`` naming the
  lock path now, and a failed write never zeroes a lock that existed (the label is written
  before the file is cut). A lock file that THIS call created stays, empty: removing it would
  need a deletion call, which the API source rules forbid (invariant 5).
* **Concern 16.** The kernel keeps a ``flock`` while any descriptor of the open file lives, so a
  holder killed while a forked child (no ``exec``) lives still blocks a resume. The message says
  a child of that process may hold the lock.

Tests write only under ``tmp_path``.
"""

import errno
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from train.api import run_dir as run_dir_module
from train.api.errors import ConfigError
from train.api.run_dir import (
    BACKUP_DIR,
    LOCK_NAME,
    REQUESTED_CONFIG_NAME,
    open_run,
    read_lock,
    refuse_live_holder,
    take_run_lock,
)

from .test_the_run_lock_cannot_race import finish, start_holder

NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")

FIRST_EPOCH = "killed before its first epoch ended"
REPORTING = "stopped early or crashed while reporting"


def lock_path_named(text: str, directory: Path) -> bool:
    return str(directory / LOCK_NAME) in text


# ---------------------------------------------------------------------
# Concern 5: the lock is asked first, in both windows
# ---------------------------------------------------------------------


def first_epoch_run(directory: Path) -> Path:
    """What a live run has during its first epoch: the request file and no backup yet."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / REQUESTED_CONFIG_NAME).write_text("{}")
    return directory


def reporting_run(directory: Path) -> Path:
    """What a live run has while it reports: the backup and a sidecar marking ``fit_finished``."""
    first_epoch_run(directory)
    (directory / BACKUP_DIR).mkdir(parents=True)
    (directory / "checkpoints" / "fit_state.json").write_text(
        json.dumps({"epochs": {}, "fit_finished": True})
    )
    return directory


@pytest.mark.parametrize(
    "build, wrong",
    [(first_epoch_run, FIRST_EPOCH), (reporting_run, REPORTING)],
    ids=["first_epoch_window", "reporting_window"],
)
def test_a_second_process_on_a_live_run_is_told_a_live_process_holds_it(tmp_path, build, wrong):
    run_dir = build(tmp_path / "run")
    holder = start_holder(run_dir, "resume")  # a REAL process holding the flock
    try:
        with pytest.raises(ConfigError) as caught:
            open_run(run_dir, resume=True)
        text = str(caught.value)
        assert f"process {holder.pid} holds run.lock and is still running" in text, text
        assert wrong not in text and "Choose a new 'out'" not in text, text
    finally:
        finish(holder)


def test_the_same_first_epoch_run_with_no_holder_keeps_its_own_refusal(tmp_path):
    """The control: asking the lock first changes nothing when nobody holds it."""
    run_dir = first_epoch_run(tmp_path / "run")
    with pytest.raises(ConfigError, match=FIRST_EPOCH):
        open_run(run_dir, resume=True)


def test_the_same_reporting_run_with_no_holder_is_opened_for_a_reporting_resume(tmp_path):
    """D-041: with nobody holding the lock a run that crashed while reporting is no longer
    refused, so the live-process refusal above is the only thing that stops a second process
    from reporting over a run that is still reporting."""
    run_dir = reporting_run(tmp_path / "run")
    assert open_run(run_dir, resume=True) == run_dir


def test_a_resume_of_a_directory_that_does_not_exist_is_still_the_missing_backup_refusal(tmp_path):
    with pytest.raises(ConfigError, match="does not exist, so there is no interrupted run"):
        open_run(tmp_path / "nowhere", resume=True)


# ---------------------------------------------------------------------
# Concern 16: a forked child keeps the flock
# ---------------------------------------------------------------------

FORKER = textwrap.dedent(
    """
    import os
    import sys
    import time
    from pathlib import Path

    from train.api.run_dir import take_run_lock

    handle = take_run_lock(Path(sys.argv[1]), resume=False)
    child = os.fork()
    if child == 0:  # a worker that never execs: it shares the open file, so it shares the lock
        os.close(1)
        time.sleep(300)
        os._exit(0)
    print(child, flush=True)
    sys.stdin.read()
    """
)


def test_a_dead_holder_whose_forked_child_lives_is_refused_with_the_child_named_as_the_reason(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c", FORKER, str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    child = int(holder.stdout.readline())
    try:
        holder.kill()
        holder.wait(timeout=60)  # the holder is dead and reaped; the child still holds the lock
        os.kill(child, 0)  # and the child is alive
        for attempt in (
            lambda: take_run_lock(tmp_path, resume=True),
            lambda: refuse_live_holder(tmp_path),
        ):
            with pytest.raises(ConfigError) as caught:
                attempt()
            text = str(caught.value)
            assert f"process {holder.pid}" in text and "child" in text, text
            assert "may still hold" in text or "inherited" in text, text
    finally:
        os.kill(child, signal.SIGKILL)  # only the recorded pid
        finish(holder)
    deadline = time.time() + 30
    while True:
        try:
            take_run_lock(tmp_path, resume=True).close()
            break
        except ConfigError:
            assert time.time() < deadline, "the killed child still holds the lock after 30 s"
            time.sleep(0.05)


def test_a_live_holder_message_names_the_child_possibility_too(tmp_path):
    holder = start_holder(tmp_path)
    try:
        with pytest.raises(ConfigError, match=rf"process {holder.pid} holds run\.lock.*child"):
            refuse_live_holder(tmp_path)
    finally:
        finish(holder)


# ---------------------------------------------------------------------
# Concern 6: open, write and read failures are a ConfigError naming the lock
# ---------------------------------------------------------------------


@NOT_ROOT
def test_a_read_only_lock_file_is_a_config_error_naming_the_lock_on_resume(tmp_path):
    lock = tmp_path / LOCK_NAME
    lock.write_text("{}")
    lock.chmod(0o444)
    try:
        with pytest.raises(ConfigError) as caught:
            take_run_lock(tmp_path, resume=True)
        assert lock_path_named(str(caught.value), tmp_path), str(caught.value)
        assert lock.read_text() == "{}"
    finally:
        lock.chmod(0o644)


@NOT_ROOT
def test_a_read_only_directory_without_a_lock_is_a_config_error_naming_the_lock(tmp_path):
    directory = tmp_path / "ro"
    directory.mkdir()
    directory.chmod(0o555)
    try:
        for resume in (True, False):
            with pytest.raises(ConfigError) as caught:
                take_run_lock(directory, resume=resume)
            assert lock_path_named(str(caught.value), directory), str(caught.value)
        assert list(directory.iterdir()) == []
    finally:
        directory.chmod(0o755)


@NOT_ROOT
def test_an_unreadable_lock_is_a_config_error_naming_the_lock_when_asked_who_holds_it(tmp_path):
    lock = tmp_path / LOCK_NAME
    lock.write_text("{}")
    lock.chmod(0o000)
    try:
        with pytest.raises(ConfigError) as caught:
            refuse_live_holder(tmp_path)
        assert lock_path_named(str(caught.value), tmp_path), str(caught.value)
    finally:
        lock.chmod(0o644)


def test_a_lock_that_cannot_be_read_is_a_config_error_naming_it(tmp_path):
    (tmp_path / LOCK_NAME).mkdir()  # opening it for reading works; reading it fails
    with pytest.raises(ConfigError) as caught:
        read_lock(tmp_path)
    assert lock_path_named(str(caught.value), tmp_path), str(caught.value)


@pytest.mark.parametrize(
    "payload", [b"\xff\xfe garbage", b"\x80", b"[1, 2]", b'{"pid": "abc"}', b""],
    ids=["non_utf8", "one_bad_byte", "list", "pid_string", "empty"],
)
def test_whatever_the_lock_holds_a_live_holder_is_refused_with_a_config_error(tmp_path, payload):
    """A payload is only a label: the refusal path must not die reading it."""
    first = take_run_lock(tmp_path, resume=False)
    try:
        (tmp_path / LOCK_NAME).write_bytes(payload)
        assert not isinstance((read_lock(tmp_path) or {}).get("pid"), int)  # no pid to name
        for attempt in (lambda: refuse_live_holder(tmp_path), lambda: take_run_lock(tmp_path, resume=True)):
            with pytest.raises(ConfigError, match=r"another process holds run\.lock"):
                attempt()
    finally:
        first.close()
    refuse_live_holder(tmp_path)  # released: the garbage is stale
    with take_run_lock(tmp_path, resume=True):
        assert read_lock(tmp_path)["pid"] == os.getpid()


def failing_write(*arguments, **keywords):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_a_failed_write_on_a_fresh_run_is_a_config_error_and_the_empty_file_is_a_stale_label(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dir_module.os, "pwrite", failing_write)
    with pytest.raises(ConfigError) as caught:
        take_run_lock(tmp_path, resume=False)
    text = str(caught.value)
    assert lock_path_named(text, tmp_path) and "No space left" in text, text
    # Nothing here deletes (invariant 5): the file this call created stays, empty, a stale label.
    assert (tmp_path / LOCK_NAME).read_bytes() == b"" and read_lock(tmp_path) == {}
    monkeypatch.undo()
    refuse_live_holder(tmp_path)  # nobody holds it
    with take_run_lock(tmp_path, resume=True):  # and the next taker locks and labels it
        assert read_lock(tmp_path)["pid"] == os.getpid()


def test_a_failed_write_on_a_resume_leaves_the_earlier_lock_unchanged_and_released(tmp_path, monkeypatch):
    earlier = json.dumps({"pid": 1, "start": None, "created": "2026-01-01T00:00:00+0000"}).encode()
    (tmp_path / LOCK_NAME).write_bytes(earlier)
    monkeypatch.setattr(run_dir_module.os, "pwrite", failing_write)
    with pytest.raises(ConfigError, match="No space left"):
        take_run_lock(tmp_path, resume=True)
    assert (tmp_path / LOCK_NAME).read_bytes() == earlier  # never a 0-byte lock
    monkeypatch.undo()
    take_run_lock(tmp_path, resume=True).close()  # the failed taker released the lock


def test_a_failed_write_on_a_resume_with_no_lock_file_is_a_config_error_and_releases_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dir_module.os, "pwrite", failing_write)
    with pytest.raises(ConfigError, match="No space left"):
        take_run_lock(tmp_path, resume=True)
    monkeypatch.undo()
    refuse_live_holder(tmp_path)  # the failed taker released its flock


def test_a_short_write_is_a_failure_too(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dir_module.os, "pwrite", lambda descriptor, data, offset: len(data) - 1)
    with pytest.raises(ConfigError, match=r"run\.lock"):
        take_run_lock(tmp_path, resume=False)


FULL_DISK = textwrap.dedent(
    """
    import resource
    import signal
    import sys
    from pathlib import Path

    from train.api.errors import ConfigError
    from train.api.run_dir import take_run_lock

    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # a write past the limit is EFBIG, not a signal
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, hard))
    try:
        take_run_lock(Path(sys.argv[1]), resume=sys.argv[2] == "resume")
    except ConfigError as exc:
        print("ConfigError", exc)
    else:
        print("taken")
    """
)


@pytest.mark.parametrize("mode", ["fresh", "resume"])
def test_a_write_the_kernel_refuses_is_a_config_error_and_releases_the_lock(tmp_path, mode):
    """A REAL failing write (RLIMIT_FSIZE 0 gives EFBIG, standing in for ENOSPC), in a child."""
    done = subprocess.run(
        [sys.executable, "-c", FULL_DISK, str(tmp_path), mode],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("ConfigError"), (done.stdout, done.stderr)
    assert str(tmp_path / LOCK_NAME) in done.stdout
    refuse_live_holder(tmp_path)  # the failed taker released its flock


# ---------------------------------------------------------------------
# A longer earlier payload must not survive under the new one (lock_no_truncate)
# ---------------------------------------------------------------------


def test_a_resume_replaces_a_longer_earlier_payload_whole(tmp_path):
    """Without a truncate the new bytes land on the old ones and the tail of the old stays."""
    earlier = json.dumps(
        {"pid": 1, "start": "x" * 400, "created": "2026-01-01T00:00:00+0000"}
    ).encode()
    (tmp_path / LOCK_NAME).write_bytes(earlier)
    with take_run_lock(tmp_path, resume=True):
        data = (tmp_path / LOCK_NAME).read_bytes()
        record = json.loads(data)  # raises on a tail of the earlier payload
        assert record["pid"] == os.getpid()
        assert len(data) < len(earlier) and b"xxxx" not in data
        assert read_lock(tmp_path)["pid"] == os.getpid()


def test_a_held_lock_whose_label_cannot_be_read_is_still_refused_as_held(tmp_path):
    """A read failure of the LABEL must not replace the fact that someone holds the lock."""
    (tmp_path / LOCK_NAME).mkdir()  # reading it fails; the kernel lock on it still works
    descriptor = os.open(tmp_path / LOCK_NAME, os.O_RDONLY)
    run_dir_module.fcntl.flock(descriptor, run_dir_module.fcntl.LOCK_EX | run_dir_module.fcntl.LOCK_NB)
    try:
        with pytest.raises(ConfigError, match=r"another process holds run\.lock"):
            refuse_live_holder(tmp_path)
    finally:
        os.close(descriptor)


def test_a_lock_that_appears_between_the_open_and_the_create_is_a_lost_claim(tmp_path, monkeypatch):
    """A resume with no lock file creates it with ``O_EXCL``: a creator that got in first wins."""
    real_open = os.open
    state = {"first": True}

    def racing_open(path, flags, *arguments, **keywords):
        if state["first"] and str(path).endswith(LOCK_NAME):
            state["first"] = False
            real_open(path, os.O_WRONLY | os.O_CREAT, 0o666)  # the other process creates it now
            raise FileNotFoundError(errno.ENOENT, "No such file", str(path))
        return real_open(path, flags, *arguments, **keywords)

    monkeypatch.setattr(run_dir_module.os, "open", racing_open)
    with pytest.raises(ConfigError, match="claimed by another run just now"):
        take_run_lock(tmp_path, resume=True)
