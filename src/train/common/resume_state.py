"""State a killed ``fit`` loses and Keras 3.8 ``BackupAndRestore`` does not keep.

``BackupAndRestore`` restores the weights, the optimizer and the next epoch
index, and nothing else (measured, plan-2026-10-30T132432-a865fa08 findings,
section C). A resumed ``fit`` therefore starts with ``EarlyStopping`` at
``best=None, wait=0``, a fresh ``ModelCheckpoint`` best, a CSV that rewrites
itself empty and a ``History`` holding only the new epochs. This module holds
the primitives that put those back:

* :class:`FitStateSidecar` writes ``{best, wait, best_epoch}`` per epoch to a
  JSON file, atomically, in the position that guarantees the entry for the
  backup's epoch exists whenever the backup does.
* :class:`RestoreFitState` sets ``best``, ``wait`` and the in-memory best
  weights on the resumed ``EarlyStopping``, and the ``best``, ``wait`` and
  ``cooldown_counter`` of a ``ReduceLROnPlateau`` when the run has one (the
  learning rate itself comes back through the optimizer's weights).
* :func:`read_backup_epoch`, :func:`read_fit_state`, :func:`prior_rows` and
  :func:`merge_history` read the previous segment back; :func:`read_resume_point`
  gathers what a resume needs into one :class:`ResumePoint`, and
  :func:`copy_log_aside` keeps the killed segment's CSV before it is rewritten.
* :func:`settle_backup` checks the two copies ``BackupAndRestore(double_checkpoint=True)``
  keeps against each other and the run's step count, and puts the previous copy in
  place when the latest is torn or from a different epoch than its metadata.

Callback order for a resumable run (every position is load-bearing)::

    EarlyStopping, ModelCheckpoint, EpochCSVLogger, ..., RestoreFitState,
    FitStateSidecar, BackupAndRestore

``RestoreFitState.on_train_begin`` must come after ``EarlyStopping``'s
(Keras resets ``best``, ``wait`` and ``best_weights`` there);
``FitStateSidecar.on_epoch_end`` must come after ``EarlyStopping``'s and before
``BackupAndRestore``'s, so that a kill at any instant leaves a sidecar entry for
every epoch the backup claims is done.

Importing this module imports ``keras`` and nothing else from the repository.
"""

import csv
import dataclasses
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import keras

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

PathLike = Union[str, "os.PathLike[str]"]

#: File name of the sidecar inside ``<run>/checkpoints``.
FIT_STATE_NAME = "fit_state.json"

#: Name of the metadata file ``BackupAndRestore`` writes next to its weights.
BACKUP_METADATA_NAME = "training_metadata.json"

#: Name of the weights file ``BackupAndRestore`` writes; the previous copy it keeps
#: under ``double_checkpoint=True`` is the same name plus :data:`PREVIOUS_SUFFIX`.
BACKUP_WEIGHTS_NAME = "latest.weights.h5"
PREVIOUS_SUFFIX = ".bkp"

#: Top-level keys of the sidecar file.
_EPOCHS_KEY = "epochs"
_FINISHED_KEY = "fit_finished"


# ---------------------------------------------------------------------

def _write_json_atomically(path: PathLike, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` through a temp file and ``os.replace``.

    A kill at any instant leaves either the old file or the new one, never a
    torn one. The temp file lives in the same directory so the rename is atomic.
    Failure mode: an unwritable directory raises ``OSError``; the temp file is
    left behind only if the process dies between the write and the rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_fit_state(path: PathLike) -> Dict[str, Any]:
    """Read the sidecar written by :class:`FitStateSidecar`.

    Args:
        path: The sidecar file, ``<run>/checkpoints/fit_state.json``.

    Returns:
        ``{"epochs": {"<epoch index>": {"best", "wait", "best_epoch"}},
        "fit_finished": bool}``. A missing file gives the empty state
        (``epochs`` empty, ``fit_finished`` False), because a run killed before
        its first epoch ended has none. Extra top-level keys written by
        :func:`update_fit_state` are returned as written.

    Raises:
        ValueError: The file exists and is not a JSON object.
    """
    path = Path(path)
    if not path.exists():
        return {_EPOCHS_KEY: {}, _FINISHED_KEY: False}
    state = json.loads(path.read_text())
    if not isinstance(state, dict):
        raise ValueError(f"{path} is not a JSON object (got {type(state).__name__})")
    state.setdefault(_EPOCHS_KEY, {})
    state.setdefault(_FINISHED_KEY, False)
    return state


def update_fit_state(path: PathLike, **fields: Any) -> Dict[str, Any]:
    """Merge top-level ``fields`` into the sidecar, atomically, and return it.

    The one writer of non-epoch keys (``fit_finished=True`` once ``fit``
    returned; a caller may add its own, such as the epoch-0 baseline). The
    per-epoch entries are kept. Creates the file when it does not exist.
    """
    state = read_fit_state(path)
    state.update(fields)
    _write_json_atomically(path, state)
    return state


def read_backup_epoch(backup_dir: PathLike, previous: bool = False) -> int:
    """Number of epochs the ``BackupAndRestore`` in ``backup_dir`` completed.

    This is the epoch index a resumed ``fit`` starts at (Keras stores
    ``epoch + 1`` after epoch ``epoch`` ended).

    Args:
        backup_dir: The backup directory.
        previous: Read the metadata of the previous copy
            (``training_metadata.json.bkp``) instead of the latest.

    Raises:
        FileNotFoundError: No ``training_metadata.json`` (no epoch ever ended).
        ValueError: The file holds no integer ``epoch`` (or is not JSON: a torn write).
    """
    path = Path(backup_dir) / (BACKUP_METADATA_NAME + (PREVIOUS_SUFFIX if previous else ""))
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: no epoch of this run was backed up")
    metadata = json.loads(path.read_text())
    epoch = metadata.get("epoch") if isinstance(metadata, dict) else None
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"{path} holds no non-negative integer 'epoch': {metadata!r}")
    return epoch


def prior_rows(csv_path: PathLike, start_epoch: int) -> List[Tuple[int, Dict[str, float]]]:
    """The rows of ``training_log.csv`` for epochs ``0 .. start_epoch - 1``.

    Args:
        csv_path: A log written by :class:`train.common.callbacks.EpochCSVLogger`.
        start_epoch: The epoch the resumed ``fit`` starts at
            (:func:`read_backup_epoch`). Rows at or after it belong to epochs the
            resume trains again, and are dropped.

    Returns:
        ``[(epoch, {key: float})]`` in epoch order, ready for ``initial_rows``.
        An empty cell (a key that epoch did not log) is left out of the dict.

    Raises:
        ValueError: The file does not hold the epochs ``0 .. start_epoch - 1``
            as its first rows, in order (a row is missing or out of order). A
            torn final row is harmless: reading stops once ``start_epoch`` rows
            were taken.
    """
    rows: List[Tuple[int, Dict[str, float]]] = []
    if start_epoch > 0:
        with open(csv_path, newline="") as handle:
            for record in csv.DictReader(handle):
                if len(rows) == start_epoch:
                    break
                epoch = int(record["epoch"])
                if epoch != len(rows):
                    raise ValueError(
                        f"{csv_path}: expected the row of epoch {len(rows)}, found epoch {epoch}")
                logs = {key: float(value) for key, value in record.items()
                        if key != "epoch" and value not in (None, "")}
                rows.append((epoch, logs))
    if len(rows) != start_epoch:
        raise ValueError(f"{csv_path} holds {len(rows)} complete epoch rows, the backup needs {start_epoch}")
    return rows


def rows_as_history(rows: Sequence[Tuple[int, Mapping[str, float]]]) -> Dict[str, List[float]]:
    """The ``History.history`` shape of ``rows``: each key to its per-epoch values.

    A key an epoch did not log is skipped for that epoch, as Keras itself lists
    a key only for the epochs that logged it.
    """
    keys = {key for _, logs in rows for key in logs}
    return {key: [logs[key] for _, logs in rows if key in logs] for key in keys}


def merge_history(
        rows: Sequence[Tuple[int, Mapping[str, float]]], history: "keras.callbacks.History"
) -> "keras.callbacks.History":
    """Put the earlier segment's epochs in front of a resumed ``History``, in place.

    ``history.history`` maps each key to its per-epoch list. For every key the
    earlier segment logged, its values are prepended; ``history.epoch`` is
    prefixed with the earlier epoch indices. The same object is returned so
    every consumer holding it (analysis, JSON, summary) sees all epochs. A key
    missing from some earlier epochs is prepended as far as it was logged, as
    Keras itself lists a key only for the epochs that logged it.
    """
    for key, earlier in rows_as_history(rows).items():
        history.history[key] = earlier + list(history.history.get(key, []))
    history.epoch = [epoch for epoch, _ in rows] + list(history.epoch or [])
    return history


@dataclasses.dataclass(frozen=True)
class ResumePoint:
    """Where a killed run stopped, as read from its files.

    :ivar start: The epoch index the resumed ``fit`` starts at (epochs ``0 ..
        start - 1`` are done).
    :ivar best: ``EarlyStopping``'s best value after epoch ``start - 1``, or
        ``None`` when none was finite.
    :ivar wait: Its ``wait`` after that epoch.
    :ivar best_epoch: Zero-based index of the best epoch.
    :ivar rows: ``(epoch, logs)`` for epochs ``0 .. start - 1``, from the CSV.
    :ivar state: The whole sidecar, for the keys a caller stored with
        :func:`update_fit_state` (such as the epoch-0 evaluation).
    :ivar reduce_lr: ``ReduceLROnPlateau``'s ``{"best", "wait", "cooldown_counter"}``
        after epoch ``start - 1``, or ``None`` when the run had none.
    """

    start: int
    best: Optional[float]
    wait: int
    best_epoch: int
    rows: Tuple[Tuple[int, Dict[str, float]], ...]
    state: Dict[str, Any]
    reduce_lr: Optional[Dict[str, Any]] = None


def read_resume_point(
        csv_path: PathLike, backup_dir: PathLike, fit_state_path: PathLike
) -> ResumePoint:
    """Read the three files a resume starts from and check they agree.

    Interface contract: ``csv_path`` is the run's ``training_log.csv``,
    ``backup_dir`` its ``BackupAndRestore`` directory, ``fit_state_path`` its
    :class:`FitStateSidecar` file. Pure file reading: no TensorFlow call, so it
    may run before the GPU is set up.

    Returns:
        The :class:`ResumePoint`.

    The epoch is read from the latest metadata; when that is missing or torn, from
    the previous copy ``double_checkpoint`` keeps (a kill while the latest was being
    written). :func:`settle_backup` then decides which weights go with it.

    Raises:
        FileNotFoundError: no backup metadata, or no CSV.
        ValueError: no epoch finished, the sidecar has no entry for the epoch
            before ``start`` (so ``best`` and ``wait`` are unknown), or the CSV
            does not hold epochs ``0 .. start - 1`` first (:func:`prior_rows`;
            the message names the ``<stem>.pre_resume_<k>.csv`` copies that do).
    """
    try:
        start = read_backup_epoch(backup_dir)
    except (FileNotFoundError, ValueError) as latest_error:
        try:
            start = read_backup_epoch(backup_dir, previous=True)
        except (FileNotFoundError, ValueError):
            raise latest_error from None
    if start < 1:
        raise ValueError(f"the backup in {backup_dir} records no finished epoch")
    state = read_fit_state(fit_state_path)
    entry = state[_EPOCHS_KEY].get(str(start - 1))
    if not isinstance(entry, dict):
        raise ValueError(
            f"{fit_state_path} holds no entry for epoch {start - 1}, the last epoch the "
            f"backup in {backup_dir} completed, so the early-stopping state is unknown")
    try:
        rows = prior_rows(csv_path, start)
    except ValueError as exc:
        raise ValueError(f"{exc}{_copies_hint(csv_path, start)}") from exc
    return ResumePoint(
        start=start,
        best=entry["best"],
        wait=int(entry["wait"]),
        best_epoch=int(entry["best_epoch"]),
        rows=tuple(rows),
        state=state,
        reduce_lr=entry.get("reduce_lr"),
    )


def _copies_hint(csv_path: PathLike, start_epoch: int) -> str:
    """Name the ``<stem>.pre_resume_<k>.csv`` copies that hold epochs ``0 .. start_epoch - 1``.

    A resume that is killed right after it started has already rewritten
    ``training_log.csv`` (the logger opens it in write mode before the first epoch
    ends), but :func:`copy_log_aside` kept the file as it was. The text says which
    copy to put back; the package never overwrites or deletes a file itself.
    Returns the sentence to append to an error message ('' when no copy exists).
    """
    csv_path = Path(csv_path)
    pattern = f"{csv_path.stem}.pre_resume_*{csv_path.suffix}"
    complete: List[str] = []
    others: List[str] = []
    for copy in sorted(csv_path.parent.glob(pattern)):
        try:
            prior_rows(copy, start_epoch)
        except (OSError, ValueError, KeyError):
            others.append(copy.name)
        else:
            complete.append(copy.name)
    if complete:
        return (
            f"; {', '.join(complete)} holds epochs 0..{start_epoch - 1} (a resume that was killed "
            f"just after it started had already rewritten {csv_path.name}): copy it over "
            f"{csv_path.name} and resume again"
        )
    if others:
        return f"; the copies {', '.join(others)} do not hold them either"
    return ""


def copy_log_aside(csv_path: PathLike) -> Path:
    """Copy ``csv_path`` to ``<stem>.pre_resume_<k>.csv`` for the smallest free ``k``.

    A resume rewrites ``training_log.csv`` (the logger opens it in write mode),
    which drops the killed segment's torn or surplus rows; the copy keeps them.
    The target is created exclusively, so an earlier copy is never overwritten.

    Returns:
        The path of the new copy.

    Raises:
        FileNotFoundError: ``csv_path`` does not exist.
    """
    csv_path = Path(csv_path)
    k = 1
    while True:
        target = csv_path.with_name(f"{csv_path.stem}.pre_resume_{k}{csv_path.suffix}")
        try:
            with open(csv_path, "rb") as source, open(target, "xb") as copy:
                shutil.copyfileobj(source, copy)
            return target
        except FileExistsError:
            k += 1


# ---------------------------------------------------------------------

def backup_iterations(weights_path: PathLike) -> Tuple[int, bool]:
    """Optimizer steps recorded in a ``BackupAndRestore`` weights file, read without TensorFlow.

    ``model.save_weights`` stores the optimizer's variables with ``iterations`` first.
    Under ``mixed_float16`` the optimizer is a ``LossScaleOptimizer`` whose own counter
    stays 0: the steps taken are the inner optimizer's, and a step with non-finite
    gradients is skipped, so that counter may lag the batches seen.

    Returns:
        ``(iterations, loss_scaled)``.

    Raises:
        OSError: The file is missing, truncated or not HDF5 (a torn write).
        KeyError: The file holds no optimizer (a model saved without compile).
    """
    import h5py

    with h5py.File(weights_path, "r") as handle:
        group = handle["optimizer"]
        scaled = "inner_optimizer" in group
        node = group["inner_optimizer"] if scaled else group
        return int(node["vars"]["0"][()]), scaled


def _inspect_copy(
        backup_dir: Path, previous: bool, steps_per_epoch: int
) -> Tuple[Optional[int], Optional[str]]:
    """Return ``(epoch, problem)`` for the latest or the previous copy; ``problem`` None if sound.

    A copy is sound when its metadata holds an epoch, its weights file opens, and the
    optimizer's step count equals ``epoch * steps_per_epoch`` (``<=`` under a loss-scaling
    optimizer, which skips steps with non-finite gradients).
    """
    suffix = PREVIOUS_SUFFIX if previous else ""
    metadata_name = BACKUP_METADATA_NAME + suffix
    weights_name = BACKUP_WEIGHTS_NAME + suffix
    try:
        epoch = read_backup_epoch(backup_dir, previous=previous)
    except FileNotFoundError:
        return None, f"{metadata_name} does not exist"
    except ValueError:
        return None, f"{metadata_name} is torn or holds no epoch"
    try:
        iterations, scaled = backup_iterations(backup_dir / weights_name)
    except FileNotFoundError:
        return epoch, f"{weights_name} does not exist"
    except (OSError, KeyError, ValueError) as exc:
        reason = str(exc).splitlines()[0][:120] if str(exc) else type(exc).__name__
        return epoch, f"{weights_name} cannot be read ({reason})"
    expected = epoch * steps_per_epoch
    if iterations == expected or (scaled and iterations < expected):
        return epoch, None
    return epoch, (
        f"{weights_name} holds an optimizer that took {iterations} steps, but "
        f"{metadata_name} says {epoch} epochs x {steps_per_epoch} steps = {expected}: "
        "the weights and the metadata are from different epochs"
    )


def _copy_atomically(source: Path, target: Path) -> None:
    """Copy ``source`` over ``target`` through a temp file in ``target``'s directory and ``os.replace``."""
    temp = target.with_name(target.name + ".restoring")
    shutil.copyfile(source, temp)
    descriptor = os.open(temp, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temp, target)


def settle_backup(backup_dir: PathLike, steps_per_epoch: int) -> Tuple[str, int]:
    """Make the backup in ``backup_dir`` a pair ``fit`` can restore, or say why it cannot.

    Interface contract: ``steps_per_epoch`` is the run's batches per epoch. Pure file
    work, no TensorFlow. ``BackupAndRestore`` writes the weights, then the metadata, so
    a kill between the two leaves weights one epoch ahead of the metadata (and a kill
    inside the weights write, or a damaged file, leaves a weights file that does not
    open). With ``double_checkpoint=True`` it keeps the previous pair as ``*.bkp``;
    this function takes the latest pair when it is sound, else the previous pair, which
    it copies over the latest atomically (the ``.bkp`` files are never changed, so a kill
    during the copy repeats harmlessly). The caller re-reads the resume point when the
    epoch returned is not the one it read the CSV rows and the sidecar entry for: the
    previous pair is one epoch behind the latest metadata, and the rows of the epoch
    after it are then trained again.

    Returns:
        ``("latest" | "previous", epoch)``: the pair chosen and the epoch its metadata
        records (the epoch ``fit`` will start at).

    Raises:
        ValueError: Neither pair is sound. The message names both weights files and
            what is wrong with each; nothing was changed.
    """
    # DECISION plan-2026-10-30T132432-a865fa08/D-069
    # The pair is checked HERE, before `fit`, with the optimizer's step count. Do NOT
    # leave it to `BackupAndRestore(double_checkpoint=True)` alone: its fallback fires
    # only when the latest weights file cannot be OPENED, and a kill between the weights
    # write and the metadata write leaves a file that opens fine and is one epoch ahead
    # of its metadata, so the resume skipped an epoch of the schedule and exited 0
    # (review-iter-3 concerns 4 and 5).
    backup_dir = Path(backup_dir)
    verdicts = []
    for label, previous in (("latest", False), ("previous", True)):
        epoch, problem = _inspect_copy(backup_dir, previous, steps_per_epoch)
        if problem is None:
            if previous:
                _copy_atomically(
                    backup_dir / (BACKUP_WEIGHTS_NAME + PREVIOUS_SUFFIX),
                    backup_dir / BACKUP_WEIGHTS_NAME,
                )
                _copy_atomically(
                    backup_dir / (BACKUP_METADATA_NAME + PREVIOUS_SUFFIX),
                    backup_dir / BACKUP_METADATA_NAME,
                )
                logger.warning(
                    f"Resume: the latest backup in {backup_dir} is unusable ({verdicts[0][1]}); "
                    f"the previous copy (epoch {epoch}) was put in its place, and the epoch "
                    "after it is trained again"
                )
            return label, epoch
        verdicts.append((label, problem))
    latest, previous_verdict = verdicts
    raise ValueError(
        f"neither copy of the backup in {backup_dir} is usable: "
        f"{BACKUP_WEIGHTS_NAME}: {latest[1]}; "
        f"{BACKUP_WEIGHTS_NAME}{PREVIOUS_SUFFIX}: {previous_verdict[1]}"
    )


# ---------------------------------------------------------------------

def _finite_or_none(value: Any) -> Optional[float]:
    """``float(value)``, or ``None`` for ``None``, NaN and the infinities (JSON holds none)."""
    return float(value) if value is not None and math.isfinite(float(value)) else None


class FitStateSidecar(keras.callbacks.Callback):
    """After each epoch, record ``EarlyStopping``'s ``best`` and ``wait`` on disk.

    Interface contract: ``early_stopping`` is the run's ``EarlyStopping`` (its
    ``on_epoch_end`` has run before this callback's, so this one comes after it
    in the list); ``path`` is the sidecar file. The entry for epoch index ``e``
    is ``{"best": float or None, "wait": int, "best_epoch": int}`` under
    ``epochs["e"]``; the whole file is rewritten atomically
    (:func:`_write_json_atomically`). Place it BEFORE ``BackupAndRestore``:
    the backup then never claims an epoch the sidecar lacks (the converse, an
    entry for an epoch the backup never recorded, is harmless, since a resume
    reads the entry of the backup's last epoch only).

    ``on_train_begin`` loads the file's earlier entries, so a resumed segment
    extends the record instead of replacing it. At each epoch end, entries for
    that epoch and later are dropped before the new one is added: they belong to
    epochs the resume trains again (the first epoch end of a segment is the only
    place that segment's start is known to this callback, since
    ``BackupAndRestore`` sets it later in ``on_train_begin``). A non-finite
    ``best`` is recorded as ``null``. Failure mode: a write error raises
    ``OSError`` out of ``fit``; a resume that cannot be made correct must not be
    silent.

    When ``reduce_lr`` is given the entry also holds ``"reduce_lr": {"best", "wait",
    "cooldown_counter"}`` (a non-finite ``best`` is ``null``); that callback's
    ``on_epoch_end`` has run before this one's, so it comes after it in the list.

    Args:
        early_stopping: The ``keras.callbacks.EarlyStopping`` to read.
        path: Destination ``fit_state.json``.
        reduce_lr: The run's ``ReduceLROnPlateau``, or ``None`` (a scheduled run has none).
    """

    def __init__(
            self,
            early_stopping: "keras.callbacks.EarlyStopping",
            path: PathLike,
            reduce_lr: Optional["keras.callbacks.ReduceLROnPlateau"] = None,
    ) -> None:
        super().__init__()
        self.early_stopping = early_stopping
        self.path = Path(path)
        self.reduce_lr = reduce_lr
        self._epochs: Dict[str, Dict[str, Any]] = {}

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Load the entries an earlier segment wrote."""
        self._epochs = dict(read_fit_state(self.path)[_EPOCHS_KEY])

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Add this epoch's entry and rewrite the file."""
        best = self.early_stopping.best
        self._epochs = {key: value for key, value in self._epochs.items() if int(key) < epoch}
        entry = {
            "best": _finite_or_none(best),
            "wait": int(self.early_stopping.wait),
            "best_epoch": int(self.early_stopping.best_epoch),
        }
        if self.reduce_lr is not None:
            entry["reduce_lr"] = {
                "best": _finite_or_none(self.reduce_lr.best),
                "wait": int(self.reduce_lr.wait),
                "cooldown_counter": int(self.reduce_lr.cooldown_counter),
            }
        self._epochs[str(epoch)] = entry
        state = read_fit_state(self.path)
        state[_EPOCHS_KEY] = self._epochs
        _write_json_atomically(self.path, state)


class RestoreFitState(keras.callbacks.Callback):
    """Set a resumed ``EarlyStopping`` back to where the killed segment left it.

    Interface contract: ``early_stopping`` is the run's ``EarlyStopping``;
    ``best`` and ``wait`` come from the sidecar entry of the epoch before the
    resume point; ``best_weights`` is the list of arrays
    (``model.get_weights()`` order) of the best epoch, read from
    ``best_model.keras``. ``on_train_begin`` writes them onto ``early_stopping``
    and must run AFTER its own ``on_train_begin`` (which resets all of them):
    place it after ``EarlyStopping`` in the callback list.

    Why ``best_weights`` matters: ``EarlyStopping(restore_best_weights=True)``
    restores ``best_weights`` at the end of ``fit`` and, with none recorded,
    records the first resumed epoch's weights. A resumed run that never beats the
    earlier best would then end on a worse epoch's weights, where the later
    steps (best-checkpoint evaluation, conformal, figures) expect the best ones.

    Args:
        early_stopping: The callback to restore.
        best: Best monitored value so far, or ``None`` when none was recorded.
        wait: Epochs since that best.
        best_weights: The best epoch's weights, or ``None`` to leave unset.
        best_epoch: Zero-based index of the best epoch (its log line only).
        reduce_lr: The run's ``ReduceLROnPlateau``, or ``None``. Place the callback
            AFTER it: ``ReduceLROnPlateau.on_train_begin`` resets its state.
        reduce_lr_state: ``{"best", "wait", "cooldown_counter"}`` from the sidecar
            entry (``best`` ``None``: left at the reset value). The learning rate
            itself is not set here: it comes back through the optimizer's weights.
    """

    def __init__(
            self,
            early_stopping: "keras.callbacks.EarlyStopping",
            best: Optional[float],
            wait: int,
            best_weights: Optional[Sequence[Any]],
            best_epoch: int = 0,
            reduce_lr: Optional["keras.callbacks.ReduceLROnPlateau"] = None,
            reduce_lr_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.reduce_lr = reduce_lr
        self.reduce_lr_state = None if reduce_lr_state is None else dict(reduce_lr_state)
        self.early_stopping = early_stopping
        self.best = best
        self.wait = int(wait)
        self.best_weights = None if best_weights is None else list(best_weights)
        self.best_epoch = int(best_epoch)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Overwrite what ``EarlyStopping.on_train_begin`` just reset."""
        # DECISION plan-2026-10-30T132432-a865fa08/D-057
        # Set the four attributes on the live EarlyStopping. Do NOT use
        # `EarlyStopping(baseline=...)` instead: it sets no `wait`, `best` or
        # `best_weights`, so a resumed run that never improves would restore the
        # first resumed epoch's weights, not the best ones.
        stopper = self.early_stopping
        stopper.best = self.best
        stopper.wait = self.wait
        stopper.best_epoch = self.best_epoch
        stopper.best_weights = self.best_weights
        if self.reduce_lr is not None and self.reduce_lr_state is not None:
            # `ReduceLROnPlateau.on_train_begin` reset all three; the rate is in the optimizer.
            plateau, saved = self.reduce_lr, self.reduce_lr_state
            plateau.wait = int(saved["wait"])
            plateau.cooldown_counter = int(saved["cooldown_counter"])
            if saved.get("best") is not None:
                plateau.best = float(saved["best"])
            logger.info(
                f"Resume: ReduceLROnPlateau restored to best={saved.get('best')}, "
                f"wait={plateau.wait}, cooldown={plateau.cooldown_counter}"
            )
        logger.info(
            f"Resume: early stopping restored to best={self.best} (epoch {self.best_epoch + 1}), "
            f"wait={self.wait}, in-memory best weights {'set' if self.best_weights else 'absent'}")
