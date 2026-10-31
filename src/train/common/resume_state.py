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
  weights on the resumed ``EarlyStopping``.
* :func:`read_backup_epoch`, :func:`read_fit_state`, :func:`prior_rows` and
  :func:`merge_history` read the previous segment back; :func:`read_resume_point`
  gathers what a resume needs into one :class:`ResumePoint`, and
  :func:`copy_log_aside` keeps the killed segment's CSV before it is rewritten.

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


def read_backup_epoch(backup_dir: PathLike) -> int:
    """Number of epochs the ``BackupAndRestore`` in ``backup_dir`` completed.

    This is the epoch index a resumed ``fit`` starts at (Keras stores
    ``epoch + 1`` after epoch ``epoch`` ended).

    Raises:
        FileNotFoundError: No ``training_metadata.json`` (no epoch ever ended).
        ValueError: The file holds no integer ``epoch``.
    """
    path = Path(backup_dir) / BACKUP_METADATA_NAME
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
    """

    start: int
    best: Optional[float]
    wait: int
    best_epoch: int
    rows: Tuple[Tuple[int, Dict[str, float]], ...]
    state: Dict[str, Any]


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

    Raises:
        FileNotFoundError: no backup metadata, or no CSV.
        ValueError: no epoch finished, the sidecar has no entry for the epoch
            before ``start`` (so ``best`` and ``wait`` are unknown), or the CSV
            does not hold epochs ``0 .. start - 1`` first (:func:`prior_rows`).
    """
    start = read_backup_epoch(backup_dir)
    if start < 1:
        raise ValueError(f"the backup in {backup_dir} records no finished epoch")
    state = read_fit_state(fit_state_path)
    entry = state[_EPOCHS_KEY].get(str(start - 1))
    if not isinstance(entry, dict):
        raise ValueError(
            f"{fit_state_path} holds no entry for epoch {start - 1}, the last epoch the "
            f"backup in {backup_dir} completed, so the early-stopping state is unknown")
    rows = prior_rows(csv_path, start)
    return ResumePoint(
        start=start,
        best=entry["best"],
        wait=int(entry["wait"]),
        best_epoch=int(entry["best_epoch"]),
        rows=tuple(rows),
        state=state,
    )


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

    Args:
        early_stopping: The ``keras.callbacks.EarlyStopping`` to read.
        path: Destination ``fit_state.json``.
    """

    def __init__(self, early_stopping: "keras.callbacks.EarlyStopping", path: PathLike) -> None:
        super().__init__()
        self.early_stopping = early_stopping
        self.path = Path(path)
        self._epochs: Dict[str, Dict[str, Any]] = {}

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Load the entries an earlier segment wrote."""
        self._epochs = dict(read_fit_state(self.path)[_EPOCHS_KEY])

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Add this epoch's entry and rewrite the file."""
        best = self.early_stopping.best
        self._epochs = {key: value for key, value in self._epochs.items() if int(key) < epoch}
        self._epochs[str(epoch)] = {
            "best": float(best) if best is not None and math.isfinite(float(best)) else None,
            "wait": int(self.early_stopping.wait),
            "best_epoch": int(self.early_stopping.best_epoch),
        }
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
    """

    def __init__(
            self,
            early_stopping: "keras.callbacks.EarlyStopping",
            best: Optional[float],
            wait: int,
            best_weights: Optional[Sequence[Any]],
            best_epoch: int = 0,
    ) -> None:
        super().__init__()
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
        logger.info(
            f"Resume: early stopping restored to best={self.best} (epoch {self.best_epoch + 1}), "
            f"wait={self.wait}, in-memory best weights {'set' if self.best_weights else 'absent'}")
