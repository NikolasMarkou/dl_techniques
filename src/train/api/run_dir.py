"""Where a run lives, and the rule that a run is never overwritten.

Nothing in this module deletes a file (invariant 5): a run directory only
grows, and a directory that already holds a run is refused. Every ``train.common``
import happens inside a function, because importing it loads TensorFlow and
the help path may not (D-012).

Interface contract
------------------
``resolve_out(out, task, family, variant)``
    :param out: the ``out`` config value. ``"auto"`` names a fresh
        ``<repo>/results/<task>_<family>_<variant>_<timestamp>``; a relative path
        is anchored at the repo root, not the working directory; an absolute
        path is kept.
    :return: the run directory as a ``Path``. Nothing is created.
    :raises ConfigError: for an empty ``out`` (or only blanks), ``.``, any ``..``
        segment, a leading ``~``, the filesystem root, the repo root, the results
        root, a path naming an existing file, a dangling or looping symbolic
        link, and any ``OSError`` met while looking (not a directory, no
        permission, name too long). The text names ``out``.

``run_artifact_names()``
    :return: the names whose presence marks a directory as holding a run: the
        shared ``RUN_ARTIFACT_NAMES`` plus this API's own files.

``open_run(run_dir, resume)``
    :param run_dir: from :func:`resolve_out`.
    :param resume: ``False`` for a new run, ``True`` to continue one.
    :return: ``run_dir``. A new run needs a missing or EMPTY directory and
        claims it before returning, by creating an empty
        ``config_requested.json`` with ``O_CREAT | O_EXCL``, so two callers
        cannot both succeed. A resumed run touches nothing (the trainer takes
        ``run.lock`` later, :func:`take_run_lock`).
    :raises ConfigError: a new run into a non-empty directory, into a directory
        holding any artifact, or under an ancestor directory holding one; a lost
        claim; a resume without ``checkpoints/backup``, with
        ``results_summary.json`` present, or whose sidecar records
        ``fit_finished`` (a finished run has nothing to resume, D-016, D-021), or
        whose ``run.lock`` names a live process (D-069); a resume of a run killed before
        its first epoch ended says so and to choose a new ``out``; any ``OSError`` met
        on the way. Nothing is created before a refusal.
        The text names ``out``.

``take_run_lock(run_dir, resume)`` / ``refuse_live_holder(run_dir)``
    The run lock ``run.lock`` (D-011 of plan-2026-10-31T180033-2faf890a): an exclusive
    ``fcntl.flock`` on the file, held for the life of the run by the returned
    :class:`RunLock` and released by the kernel when the process ends, SIGKILL included.
    The file also holds ``{"pid", "start", "created"}`` (``start`` is the process start
    time from ``/proc/<pid>/stat`` with the boot id, ``null`` where ``/proc`` is missing),
    written only after the lock is held; it is for the refusal message and never decides
    anything, so a pid that another process now owns blocks nobody.
    :param run_dir: the run directory.
    :param resume: ``False`` for a fresh run: ``run.lock`` is created with ``O_EXCL``, and
        its existing is a lost claim. ``True``: the file is opened (created when absent)
        and locked without waiting; a lock held by a live process is refused.
    :return: the :class:`RunLock`. It is a context manager; ``close()`` releases the
        lock and may be called twice. Dropping it releases the lock too, so a caller that
        ignores the return value holds nothing. The descriptor is not inherited across
        ``exec``.
    :raises ConfigError: a lost claim, a live holder (naming its pid when the file says
        it, and saying that a child of that process may still hold the lock), a filesystem
        on which ``flock`` fails, or a lock file that cannot be opened, written or read (a
        read-only file or directory, a full disk, an unreadable file); each names the lock
        path. A failed ``take`` leaves the earlier file as it was: the label is written over
        the old one BEFORE the file is cut to its length, so a failed write never zeroes a lock
        that existed. A file that this very call created stays, empty (a stale label): nothing
        here deletes. A finished or killed run's lock file stays and its payload is
        overwritten by the next holder. A payload that is empty, garbled or not UTF-8 is a
        stale label (``read_lock`` gives ``{}``), never an error.
    ``refuse_live_holder`` is the read-only check of the same lock (a shared ``flock``
    taken and dropped at once, so it writes nothing): it raises the same refusals and is
    what ``open_run`` calls FIRST on a resume, before any other file is read, so a live
    holder is named as such in the first-epoch and the reporting windows alike. While it runs, a process
    taking the lock at that instant is refused as if the probe were a holder.

``check_model_file(path, what)``
    :raises ConfigError: when ``path`` is not an intact ``.keras`` archive (a zip whose
        every member passes its CRC), naming the file.

``load_resume_point(run_dir, mapping)``
    :param run_dir: a directory :func:`open_run` accepted for a resume.
    :param mapping: the requested configuration of THIS call.
    :return: the :class:`~train.common.resume_state.ResumePoint` read from the
        run's files. Reads only; touches nothing and calls no TensorFlow op.
    :raises ConfigError: when ``config_requested.json`` is missing, empty (the
        run died before writing it) or unreadable; when ``mapping`` differs from
        it in any key except ``resume_interrupted``, ``out`` and ``devices`` (the
        keys are named); or when the backup, the CSV and the sidecar do not agree on where
        the run stopped (D-058).

``write_requested(run_dir, mapping)`` / ``write_resolved(run_dir, config)``
    Write ``config_requested.json`` (replacing the claim; the mapping as the
    caller gave it, AUTO values included, group instances turned into plain
    values) and ``config.json`` (the resolved config). ``write_requested`` raises
    ``ConfigError`` for a NaN or an infinity, and writes nothing then; both raise
    ``OSError`` or ``TypeError`` on a failed write, which neither wraps.

``write_task(run_dir, record)``
    :param record: the ``task.json`` record (:mod:`train.api.predictor`).
    :return: ``True`` when ``task.json`` was written; ``False`` when the directory
        already held one (a resume), which is then left untouched.
"""

import collections.abc
import dataclasses
import errno
import fcntl
import json
import math
import numbers
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional, Tuple

from .errors import ConfigError

# ---------------------------------------------------------------------

#: The ``out`` value that asks for a timestamped directory under ``results/``.
AUTO_OUT = "auto"

#: Where ``out="auto"`` runs go, relative to the repo root.
AUTO_PARENT = "results"

#: The file a request is written to before any data is loaded.
REQUESTED_CONFIG_NAME = "config_requested.json"

#: A run that can be resumed holds this directory (written by ``BackupAndRestore``).
BACKUP_DIR = "checkpoints/backup"

#: Where the fit-state sidecar lives, relative to the run directory
#: (``train.common.resume_state.FIT_STATE_NAME`` inside ``checkpoints``).
CHECKPOINTS_DIR = "checkpoints"

#: What a model needs to be used without its config (class names, preprocessing);
#: schema and reader in :mod:`train.api.predictor`.
TASK_FILE = "task.json"

#: The ending of the record ``export`` writes beside an exported model (``<out>.task.json``);
#: ``predict`` reads it for a ``.keras`` file (D-068).
TASK_SUFFIX = ".task.json"

#: The training log the CSV logger writes, relative to the run directory.
TRAINING_LOG_NAME = "training_log.csv"

#: The run lock (D-069): who is running in this directory.
LOCK_NAME = "run.lock"

#: Request keys a resume does not compare with the earlier request: the flag that
#: asks for the resume, ``out`` (the directory was found from it, and a run
#: started with ``out=auto`` or moved since is resumed by its real path) and
#: ``devices`` (the weights and the optimizer do not depend on which GPU holds
#: them, so a run may move to another one; D-069).
RESUME_UNCOMPARED_KEYS = ("resume_interrupted", "out", "devices")

# ---------------------------------------------------------------------


def _anchor(output_dir: str) -> Path:
    """Return ``output_dir`` anchored the way ``resolve_out`` anchors a relative path."""
    from train.common.args import resolved_run_dir

    return resolved_run_dir(SimpleNamespace(output_dir=output_dir, experiment_name=""))


def _unusable(path: Path, exc: BaseException) -> ConfigError:
    """Turn an ``OSError`` (or a symlink-loop ``RuntimeError``) into a ``ConfigError``."""
    reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
    return ConfigError(f"Invalid value for 'out': {path} cannot be used: {reason}")


def _refuse_links(run_dir: Path) -> None:
    """Refuse a dangling or looping symbolic link at ``run_dir`` or any parent.

    ``mkdir(parents=True)`` follows a dangling link and creates its target, so the
    link must be found before anything is created.
    """
    for part in (run_dir, *run_dir.parents):
        if part.is_symlink() and not part.exists():
            raise ConfigError(
                f"Invalid value for 'out': {part} is a dangling or looping symbolic "
                "link, so no directory can be made there"
            )


def resolve_out(out: Any, task: str, family: str, variant: str) -> Path:
    """Turn the ``out`` config value into a run directory path.

    :param out: ``"auto"``, a relative path or an absolute path.
    :param task: the task name, e.g. ``"image_classification"``.
    :param family: the model family, e.g. ``"vit"``.
    :param variant: the model variant, e.g. ``"vit_pico"``.
    :return: the run directory. Nothing is created or checked for existing runs.
    :raises ConfigError: see the module contract.
    """
    if not isinstance(out, (str, Path)):
        raise ConfigError(
            f"Invalid value for 'out': expected a path, got {out!r} ({type(out).__name__})"
        )
    text = str(out)
    if text.strip() == "":
        raise ConfigError("Invalid value for 'out': it is empty; use a path or 'auto'")

    from train.common.run_io import default_experiment_name

    # DECISION plan-2026-10-30T132432-a865fa08/D-030
    # A relative `out` is anchored at the REPO ROOT through the shared
    # `resolved_run_dir`, so `out=results/x` lands in the same tree wherever the
    # user stands. Do NOT use `Path(out).resolve()` (it anchors at the working
    # directory and strays a `results/` tree anywhere), and do NOT re-derive
    # `parents[3]` here: that index is a layout coincidence owned by the helper
    # (plan-2026-08-12T123743-e798a9e1/D-014).
    if text == AUTO_OUT:
        run_dir = _anchor(AUTO_PARENT) / default_experiment_name(task, family, variant)
    else:
        # DECISION plan-2026-10-30T132432-a865fa08/D-033
        # A `..` segment and a leading `~` are REFUSED, not resolved. Do NOT
        # normalize the path and accept it: `missing/../finished_run` made
        # `mkdir(parents=True)` create `missing`, after which the path reached the
        # finished run and its files were overwritten (review-iter-1 concern 1).
        # A user wanting a parent directory writes the absolute path.
        path = Path(text)
        if not path.parts or ".." in path.parts:
            raise ConfigError(
                f"Invalid value for 'out': {text!r} does not name a run directory "
                "(no '.', and no '..' segment; write an absolute path instead)"
            )
        if text.startswith("~"):
            raise ConfigError(
                f"Invalid value for 'out': {text!r} starts with '~', which is not "
                "expanded here; write an absolute path"
            )
        run_dir = _anchor(text)

    if run_dir.parent == run_dir:
        raise ConfigError(f"Invalid value for 'out': {text!r} is the filesystem root")
    try:
        _refuse_links(run_dir)
        resolved = run_dir.resolve()
        for label, root in (("the repo root", "."), ("the results root", AUTO_PARENT)):
            if resolved == _anchor(root).resolve():
                raise ConfigError(
                    f"Invalid value for 'out': {text!r} is {label}, which holds other "
                    "work; name a run directory inside it, or use 'auto'"
                )
        if run_dir.exists() and not run_dir.is_dir():
            raise ConfigError(
                f"Invalid value for 'out': {run_dir} is an existing file, not a directory"
            )
    except (OSError, RuntimeError) as exc:
        raise _unusable(run_dir, exc) from exc
    return run_dir


def run_artifact_names() -> Tuple[str, ...]:
    """Return the names whose presence marks a directory as holding a run.

    :return: the shared ``RUN_ARTIFACT_NAMES`` followed by this API's files.
    """
    from train.common.run_artifacts import RUN_ARTIFACT_NAMES

    # DECISION plan-2026-10-30T132432-a865fa08/D-030
    # The API's own files are added HERE, in a tuple built per call. Do NOT widen
    # `RUN_ARTIFACT_NAMES` itself: seven other callers and the power_mlp
    # parametrized refusal test read it, and a legacy trainer's refusal must not
    # change when this package grows a file.
    return (*RUN_ARTIFACT_NAMES, REQUESTED_CONFIG_NAME, "final_model.keras", TASK_FILE, BACKUP_DIR)


def _refuse_holder(directory: Path, run_dir: Path) -> None:
    """Call the shared refusal on ``directory`` and say it in terms of ``out``."""
    from train.common.run_artifacts import refuse_existing_run

    names = run_artifact_names()
    try:
        refuse_existing_run(directory, names)
    except FileExistsError as exc:
        found = ", ".join(name for name in names if (directory / name).exists())
        place = (
            f"{directory} already holds a run"
            if directory == run_dir
            else f"{run_dir} lies inside {directory}, which already holds a run"
        )
        raise ConfigError(
            f"Invalid value for 'out': {place} ({found}). Nothing was written. "
            "Choose a new 'out' ('out=auto' picks a new timestamped directory); "
            "existing results are never overwritten or deleted."
        ) from exc


def _refuse_used(run_dir: Path) -> None:
    """Refuse ``run_dir`` for a new run when it, or a parent, is already used."""
    _refuse_holder(run_dir, run_dir)
    if run_dir.exists() and not run_dir.is_dir():
        raise ConfigError(
            f"Invalid value for 'out': {run_dir} is an existing file, not a directory"
        )
    if run_dir.is_dir():
        held = sorted(entry.name for entry in run_dir.iterdir())
        if held:
            shown = ", ".join(held[:5]) + (", ..." if len(held) > 5 else "")
            raise ConfigError(
                f"Invalid value for 'out': {run_dir} is not empty ({shown}). Nothing "
                "was written. A new run needs a missing or empty directory; choose a "
                "new 'out' ('out=auto' picks a new timestamped directory)."
            )
    for ancestor in run_dir.resolve().parents:
        _refuse_holder(ancestor, run_dir)


def _claim(run_dir: Path) -> None:
    """Create an empty ``config_requested.json`` that only one caller can create."""
    # DECISION plan-2026-10-30T132432-a865fa08/D-033
    # The claim is `O_CREAT | O_EXCL` on the request file, because the directory
    # may legitimately exist already (empty), so `mkdir` without `exist_ok` cannot
    # arbitrate. Do NOT replace it with "check the artifact names, then write
    # later": two callers with no write between them both passed (review-iter-1
    # concern 7; `setup_gpu` and seeding sit between open_run and write_requested).
    try:
        fd = os.open(
            run_dir / REQUESTED_CONFIG_NAME, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666
        )
    except FileExistsError as exc:
        raise ConfigError(
            f"Invalid value for 'out': {run_dir} was claimed by another run just now "
            f"({REQUESTED_CONFIG_NAME} exists). Nothing was written. Choose a new "
            "'out' ('out=auto' picks a new timestamped directory)."
        ) from exc
    os.close(fd)


def _fit_state_path(run_dir: Path) -> Path:
    """Return the sidecar file of ``run_dir`` (the name is the sidecar's own)."""
    from train.common.resume_state import FIT_STATE_NAME

    return run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME


def _refuse_finished_fit(run_dir: Path) -> None:
    """Refuse a resume of a run whose ``fit`` already returned (D-021).

    A run that crashed after ``fit`` but before its summary has no summary to
    show it finished; the sidecar's ``fit_finished`` marker does.
    """
    from train.common.resume_state import read_fit_state

    path = _fit_state_path(run_dir)
    try:
        finished = read_fit_state(path)["fit_finished"]
    except ValueError as exc:
        raise ConfigError(f"Cannot resume: {path} is unreadable ({exc})") from exc
    if finished:
        raise ConfigError(
            f"Cannot resume: {run_dir} is marked fit_finished, so its training "
            "ended (it was stopped early or crashed while reporting); choose a new "
            "'out' to train again"
        )


def _first_epoch_message(run_dir: Path) -> str:
    """The refusal for a run that was killed before its first epoch ended (nothing to resume)."""
    return (
        f"Cannot resume {run_dir}: the run was killed before its first epoch ended, so nothing "
        "was saved that a resume could continue from. Choose a new 'out' (this directory is "
        "kept as it is; nothing in it is deleted)"
    )


# ---------------------------------------------------------------------
# The run lock
# ---------------------------------------------------------------------


def _proc_stat(pid: int) -> Optional[Tuple[str, str]]:
    """Return ``(state, start time in clock ticks)`` of ``pid`` from ``/proc``, or ``None``."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # `pid (comm) state ppid ...`: comm may hold spaces and parentheses, so cut at the LAST ")".
    fields = text[text.rindex(")") + 2:].split()
    try:
        return fields[0], fields[19]
    except IndexError:
        return None


def process_stamp(pid: int) -> Optional[str]:
    """Return the start time of ``pid`` (boot id and clock ticks), or ``None`` without ``/proc``."""
    stat = _proc_stat(pid)
    if stat is None:
        return None
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = ""
    return f"{boot}:{stat[1]}"


def read_lock(run_dir: Path) -> Optional[dict]:
    """Return the parsed ``run.lock`` of ``run_dir``; ``None`` if absent; ``{}`` if unreadable text.

    An empty, garbled or non-UTF-8 payload is the lock of a process that died while writing
    it (or of a disk that damaged it): ``{}``, a label that names nobody.

    :raises ConfigError: naming the lock when the file exists but cannot be read at all
        (no permission, a directory, an I/O error).
    """
    lock = run_dir / LOCK_NAME
    try:
        data = lock.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _lock_unusable(lock, "read", exc) from exc
    try:
        record = json.loads(data.decode("utf-8"))
    except ValueError:  # UnicodeDecodeError and JSONDecodeError are both ValueErrors
        return {}
    return record if isinstance(record, dict) else {}


def _holder_text(run_dir: Path) -> str:
    """Name the process the lock file says holds ``run_dir`` (the file is only a label)."""
    try:
        pid = (read_lock(run_dir) or {}).get("pid")
    except ConfigError:  # a holder whose label cannot be read is still a holder
        pid = None
    return f"process {pid}" if isinstance(pid, int) and not isinstance(pid, bool) else "another process"


def _lock_unusable(lock: Path, doing: str, exc: OSError) -> ConfigError:
    """The refusal for an open, a write or a read of ``lock`` that the operating system refused."""
    return ConfigError(
        f"Cannot {doing} {lock}: {exc.strerror or exc}. The run lock needs a writable run "
        "directory on a local disk; fix the permission or space, or choose another 'out'"
    )


def _flock_failed(lock: Path, exc: OSError) -> ConfigError:
    """The refusal for a filesystem on which ``flock`` itself fails (not a held lock)."""
    return ConfigError(
        f"Cannot lock {lock}: flock failed ({exc.strerror or exc}). The run lock needs a "
        "filesystem that supports flock; choose an 'out' on a local disk"
    )


def _is_held(exc: OSError) -> bool:
    """Whether a failed non-blocking ``flock`` means "another descriptor holds it"."""
    return exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN)


def _held_by_another(run_dir: Path) -> ConfigError:
    # DECISION plan-2026-10-31T180033-2faf890a/D-024
    # The second sentence is true and needed: a flock lives as long as ANY descriptor of the open
    # file does, so a holder killed while a forked worker (no exec) lives still blocks the lock,
    # and the pid named is then dead (review-iter-1 concern 16). Do NOT word it as "process N is
    # running": it is the kernel that holds the lock, and the pid is only a label.
    return ConfigError(
        f"Cannot resume {run_dir}: {_holder_text(run_dir)} holds {LOCK_NAME} and is still "
        "running in it. Two processes in one run directory corrupt its log and its "
        "checkpoints; wait for that process to end, or stop it if it is not meant to run. "
        "If that process was killed and this message stays, a child it started (a forked "
        "worker) may still hold the lock; it is released when that child ends"
    )


def refuse_live_holder(run_dir: Path) -> None:
    """Refuse when a process holds ``run_dir``'s lock; write nothing.

    :raises ConfigError: naming the pid when the lock file says it, naming the lock when
        ``flock`` fails or the file cannot be opened.
    """
    lock = run_dir / LOCK_NAME
    try:
        descriptor = os.open(lock, os.O_RDONLY)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise _lock_unusable(lock, "open", exc) from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            if _is_held(exc):
                raise _held_by_another(run_dir) from exc
            raise _flock_failed(lock, exc) from exc
    finally:
        os.close(descriptor)  # closing drops the shared lock


def _lock_payload() -> bytes:
    pid = os.getpid()
    return json.dumps(
        {"pid": pid, "start": process_stamp(pid), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    ).encode()


class RunLock:
    """The ``flock`` on ``run.lock``, held until :meth:`close`, a ``with`` block's end or the drop.

    Made by :func:`take_run_lock`. Closing the descriptor releases the lock; the file is
    left in place.
    """

    def __init__(self, descriptor: int) -> None:
        self._descriptor: Optional[int] = descriptor

    def close(self) -> None:
        """Release the lock; calling it again does nothing."""
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is not None:
            os.close(descriptor)

    def __enter__(self) -> "RunLock":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _open_lock(run_dir: Path, resume: bool) -> int:
    """Open ``run.lock`` for writing and return the descriptor.

    A fresh run creates it with ``O_EXCL``. A resume opens the existing file, and creates it
    (also with ``O_EXCL``, so a creator that got in between wins) only when it is absent.
    """
    lock = run_dir / LOCK_NAME
    try:
        if resume:
            try:
                return os.open(lock, os.O_WRONLY)
            except FileNotFoundError:
                pass
        return os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError as exc:
        raise ConfigError(
            f"Invalid value for 'out': {run_dir} was claimed by another run just now "
            f"({LOCK_NAME} exists). Nothing was written. Choose a new 'out'"
        ) from exc
    except OSError as exc:
        raise _lock_unusable(lock, "open", exc) from exc


def _write_payload(descriptor: int) -> None:
    """Write the label over the start of the file, then cut the file to its length, then sync.

    The write comes BEFORE the cut, so a write that fails leaves the earlier payload in
    place (never an empty file), and the cut removes a longer earlier payload's tail.

    :raises OSError: a failed or short write, cut or sync.
    """
    payload = _lock_payload()
    if os.pwrite(descriptor, payload, 0) != len(payload):
        raise OSError(errno.ENOSPC, "short write")
    os.ftruncate(descriptor, len(payload))
    os.fsync(descriptor)


def take_run_lock(run_dir: Path, resume: bool) -> RunLock:
    """Take the run lock of ``run_dir`` for this process; see the module contract.

    :param run_dir: the run directory.
    :param resume: ``False`` for a fresh run (``O_EXCL`` creates the file), ``True`` to take
        over a lock whose holder is gone.
    :return: the held :class:`RunLock`.
    :raises ConfigError: a lost claim, a live holder, a failing ``flock``, or a lock file that
        cannot be opened or written (each names the lock path).
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-018
    # The lock is a kernel `flock` (LOCK_EX | LOCK_NB) on `run.lock`, held on a descriptor for
    # the life of the run. It supersedes the read / `os.replace` / read-back lock of
    # plan-2026-10-30T132432-a865fa08/D-069, whose stated residual (two resumes that read
    # one stale lock before either replaced it both ran) was measured: eight processes on a
    # stale lock left two or three holders in 20 of 20 rounds. Do NOT go back to a pid and a
    # liveness guess, and do NOT take over by rename-then-create: both are check-then-act.
    # The pid in the file is a label for the message only. Do NOT add an unlink for the lock:
    # the package deletes nothing (invariant 5); a stale lock is simply one nobody holds.
    # A failed write leaves the earlier payload in place (the label is written BEFORE the file
    # is cut, D-024) and never zeroes a lock that existed. A file this very call created stays
    # empty (a stale label, like the empty claim file of `_claim`): removing it would need a
    # deletion call, which `test_the_api_source_rules` forbids, so do NOT add an unlink.
    # The descriptor is opened by `os.open`, which is close-on-exec, so a child process does
    # not keep the lock alive. A fresh run still decides its claim by `O_EXCL`.
    lock = run_dir / LOCK_NAME
    descriptor = _open_lock(run_dir, resume)
    handle = RunLock(descriptor)  # owns the descriptor from here: any refusal below closes it
    taken = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if not _is_held(exc):
                raise _flock_failed(lock, exc) from exc
            if not resume:
                raise ConfigError(
                    f"Invalid value for 'out': {run_dir} was claimed by another run just now "
                    f"({LOCK_NAME} is held). Nothing was written. Choose a new 'out'"
                ) from exc
            raise _held_by_another(run_dir) from exc
        try:
            _write_payload(descriptor)
        except OSError as exc:
            raise _lock_unusable(lock, "write", exc) from exc
        taken = True
    finally:
        if not taken:
            handle.close()
    return handle


def check_model_file(path: Path, what: str) -> None:
    """Raise a :class:`ConfigError` naming ``path`` unless it is an intact ``.keras`` archive.

    A ``.keras`` file is a zip whose central directory is written last, so a file cut
    short by a kill or a full disk fails here, in plain file I/O, instead of as a
    traceback from Keras ("File not found" for a file that exists). Every member is read
    once and checked against its CRC.

    :param what: the sentence head, e.g. ``"Cannot resume <dir>"``.
    """
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            damaged = archive.testzip()
    except (OSError, zipfile.BadZipFile) as exc:
        raise ConfigError(
            f"{what}: {path} is not a readable .keras archive ({exc}); it was probably cut "
            "short by a kill or a full disk"
        ) from exc
    if damaged is not None:
        raise ConfigError(f"{what}: {path} is damaged (member {damaged!r} fails its checksum)")


def open_run(run_dir: Path, resume: bool) -> Path:
    """Refuse a run directory that cannot be used, then claim it.

    :param run_dir: the directory from :func:`resolve_out`.
    :param resume: ``True`` to continue an interrupted run.
    :return: ``run_dir``.
    :raises ConfigError: see the module contract.
    """
    from train.common.run_artifacts import RESULTS_SUMMARY_NAME

    try:
        if resume:
            # DECISION plan-2026-10-31T180033-2faf890a/D-024
            # The lock is asked FIRST: a live holder is the one fact that makes every other
            # finding (no backup yet, fit_finished) a state of a healthy run, not a dead one
            # (review-iter-1 concern 5). Do NOT move it below those checks.
            refuse_live_holder(run_dir)
            if not (run_dir / BACKUP_DIR).is_dir():
                if (run_dir / REQUESTED_CONFIG_NAME).exists() and not (
                    run_dir / RESULTS_SUMMARY_NAME
                ).exists():
                    raise ConfigError(_first_epoch_message(run_dir))
                raise ConfigError(
                    f"Cannot resume: {run_dir / BACKUP_DIR} does not exist, so there is "
                    "no interrupted run to continue"
                )
            if (run_dir / RESULTS_SUMMARY_NAME).exists():
                raise ConfigError(
                    f"Cannot resume: {run_dir} holds {RESULTS_SUMMARY_NAME}, so the run "
                    "has finished; choose a new 'out' to train again"
                )
            _refuse_finished_fit(run_dir)
            return run_dir

        _refuse_links(run_dir)
        _refuse_used(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        _claim(run_dir)
    except (OSError, RuntimeError) as exc:
        raise _unusable(run_dir, exc) from exc
    return run_dir


def _plain(value: Any, path: str, non_finite: List[str]) -> Any:
    """Return ``value`` as plain JSON values, listing each non-finite number's path."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _plain(getattr(value, f.name), f"{path}.{f.name}" if path else f.name, non_finite)
            for f in dataclasses.fields(value)
        }
    if isinstance(value, collections.abc.Mapping):
        return {
            key: _plain(item, f"{path}.{key}" if path else str(key), non_finite)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{path}[{i}]", non_finite) for i, item in enumerate(value)]
    if (
        isinstance(value, numbers.Real)
        and not isinstance(value, numbers.Integral)
        and not math.isfinite(value)
    ):
        non_finite.append(path or "<top level>")
    return value


def _plain_request(mapping: Any) -> Any:
    """Return ``mapping`` as plain JSON values, or raise for a NaN or an infinity."""
    non_finite: List[str] = []
    plain = _plain(mapping, "", non_finite)
    if non_finite:
        raise ConfigError(
            f"Cannot write {REQUESTED_CONFIG_NAME}: JSON cannot hold NaN or infinity, "
            "found at " + ", ".join(repr(path) for path in non_finite)
        )
    return plain


def write_requested(run_dir: Path, mapping: Any) -> Path:
    """Write ``config_requested.json``: the request as the caller gave it.

    The file replaces the empty claim :func:`open_run` made.

    :param run_dir: an existing run directory.
    :param mapping: the requested configuration, AUTO values left as ``"auto"``.
        Nested group instances are written as the plain mappings they hold.
    :return: ``run_dir``.
    :raises ConfigError: when a number is NaN or infinite, since JSON cannot hold
        it; nothing is written then.
    """
    from train.common.config_io import save_config_json

    save_config_json(_plain_request(mapping), str(run_dir), REQUESTED_CONFIG_NAME)
    return run_dir


def _leaves(value: Any, path: str, found: dict) -> None:
    """Flatten mappings into ``{dotted path: leaf}``; a list or scalar is one leaf."""
    if isinstance(value, dict):
        for key, item in value.items():
            _leaves(item, f"{path}.{key}" if path else str(key), found)
    else:
        found[path] = value


#: Stands for a key one side of a request comparison does not have.
_ABSENT = object()


def _shown(value: Any) -> str:
    """Return ``value`` for an error line, cut at 60 characters."""
    if value is _ABSENT:
        return "<absent>"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def request_differences(earlier: Any, mapping: Any) -> List[str]:
    """List the keys where ``mapping`` differs from the ``earlier`` request.

    :param earlier: the parsed ``config_requested.json``.
    :param mapping: the requested configuration of this call.
    :return: one ``"<dotted key>: <earlier> -> <now>"`` line per differing key,
        in key order, skipping :data:`RESUME_UNCOMPARED_KEYS`. Both sides are
        compared in JSON-normal form (a tuple is a list after the round trip
        through the file), so a request equal to the earlier one gives ``[]``.
    :raises ConfigError: when ``mapping`` holds a NaN or an infinity.
    """
    from train.common.config_io import json_numpy_default

    now = json.loads(json.dumps(_plain_request(mapping), default=json_numpy_default))
    before_leaves: dict = {}
    now_leaves: dict = {}
    _leaves(earlier, "", before_leaves)
    _leaves(now, "", now_leaves)
    lines = []
    for key in sorted(set(before_leaves) | set(now_leaves)):
        if key.split(".")[0] in RESUME_UNCOMPARED_KEYS:
            continue
        old, new = before_leaves.get(key, _ABSENT), now_leaves.get(key, _ABSENT)
        if old != new or type(old) is not type(new):
            lines.append(f"{key}: {_shown(old)} -> {_shown(new)}")
    return lines


def load_resume_point(run_dir: Path, mapping: Any) -> Any:
    """Check the request against the run's own, then read where the run stopped.

    :param run_dir: a directory :func:`open_run` accepted for a resume.
    :param mapping: the requested configuration of this call.
    :return: the :class:`~train.common.resume_state.ResumePoint`.
    :raises ConfigError: see the module contract.
    """
    from train.common.resume_state import read_resume_point

    # DECISION plan-2026-10-30T132432-a865fa08/D-058
    # The comparison reads `config_requested.json`, NEVER `config.json`: the second
    # holds the resolved config plus `used` (D-050), so it cannot be loaded as a
    # request. An empty `config_requested.json` is the claim of a run that died
    # before writing it; refuse it by name rather than parse it. `out` is not
    # compared (see RESUME_UNCOMPARED_KEYS); every other difference is a different
    # run and is refused, because a changed `lr` or `batch_size` would resume a
    # trajectory it did not start.
    path = run_dir / REQUESTED_CONFIG_NAME
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(
            f"Cannot resume: {path} cannot be read ({exc.strerror or exc}), so the "
            "request this run started with is unknown"
        ) from exc
    if not text.strip():
        raise ConfigError(
            f"Cannot resume: {path} is empty, so the run died before it wrote its "
            "request and there is nothing to compare; choose a new 'out'"
        )
    try:
        earlier = json.loads(text)
    except ValueError as exc:
        raise ConfigError(f"Cannot resume: {path} is not valid JSON ({exc})") from exc
    if not isinstance(earlier, dict):
        raise ConfigError(f"Cannot resume: {path} does not hold a JSON object")
    lines = request_differences(earlier, mapping)
    if lines:
        raise ConfigError(
            f"Cannot resume {run_dir}: the request differs from {REQUESTED_CONFIG_NAME} "
            "in " + "; ".join(lines) + ". Resume with the request the run started "
            "with (only resume_interrupted and out may change)"
        )
    try:
        return read_resume_point(
            run_dir / TRAINING_LOG_NAME, run_dir / BACKUP_DIR, _fit_state_path(run_dir)
        )
    except (OSError, ValueError) as exc:
        from train.common.resume_state import BACKUP_METADATA_NAME, PREVIOUS_SUFFIX

        backup = run_dir / BACKUP_DIR
        if not any((backup / (BACKUP_METADATA_NAME + tail)).exists() for tail in ("", PREVIOUS_SUFFIX)):
            raise ConfigError(_first_epoch_message(run_dir)) from exc
        raise ConfigError(
            f"Cannot resume {run_dir}: the run's files do not agree on where it stopped "
            f"({exc})"
        ) from exc


def write_resolved(run_dir: Path, config: Any, used: Optional[Mapping[str, Any]] = None) -> str:
    """Write ``config.json``: the configuration after AUTO resolution.

    :param run_dir: an existing run directory.
    :param config: a dataclass instance (or a JSON-serializable mapping).
    :param used: what the run was built with beyond the config's fields (the
        model keywords its resolver produced, the loader mapping); written under
        the key ``used``. ``None`` or empty adds no key.
    :return: the path written, as text.
    """
    from train.common.config_io import save_config_json

    payload = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config
    # DECISION plan-2026-10-30T132432-a865fa08/D-050: what the run was built with goes
    # under its OWN key, ``used``. Do NOT overwrite the config's ``model_args`` with the
    # resolver's output: that field is what the caller typed (and what a replay of this
    # file would feed back through the resolver), and a regularizer object is not JSON.
    if used:
        payload = {**payload, "used": dict(used)}
    return save_config_json(payload, str(run_dir), "config.json")


def write_task(run_dir: Path, record: Mapping[str, Any]) -> bool:
    """Write ``task.json`` unless the run directory already holds one.

    :param run_dir: an existing run directory.
    :param record: the checked record from :func:`train.api.predictor.build_task_record`.
    :return: ``True`` when the file was written; ``False`` when it was already there, which
        is a resume: the earlier segment trained on the statistics it holds, so the file is
        left exactly as it is and a resume never rewrites it.
    """
    from train.common.config_io import save_config_json

    if (run_dir / TASK_FILE).exists():
        return False
    save_config_json(dict(record), str(run_dir), TASK_FILE)
    return True
