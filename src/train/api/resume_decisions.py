"""What a resume may do, decided from the files: the refusals, the proofs and the restores.

``TaskTrainer`` runs one sequence; a resume changes it in three places (what is refused before
anything is written, what is proven about the backup before ``fit``, and what is rebuilt from the
files when nothing is left to train). They are functions of the run directory and a few values, with
no trainer in them, so a second task uses them as they are and ``trainer.py`` stays the sequence
and its hooks. The text of every refusal is a contract: ``test_resume_*`` and
``test_a_crashed_reporting_is_resumed`` pin it.

Interface contract
------------------
All functions are plain file work except :func:`restore_finished_fit` and :func:`best_weights_of`,
which load Keras models and therefore run after ``setup_gpu``. A refusal is a
:class:`~train.api.errors.ConfigError` that names the file and the fact, before the run directory is
written to, and says whether ``out`` is still usable.

``resume_point(run_dir, resolved, request)``
    The :class:`~train.common.resume_state.ResumePoint` of a killed run, every refusal applied.
``training_complete_reason(point, resolved)``
    ``None`` while an epoch is left, else the phrase for why none is (D-064): the fit returned, all
    epochs ended, or EarlyStopping had decided to stop. A complete run is resumed to REPORT only.
``check_resume_point(run_dir, resolved, point)`` / ``check_finished_fit(run_dir, point)``
    The refusals of a normal resume, and of a reporting-only one.
``check_final_weights(run_dir, resume, steps_per_epoch)``
    Refuse a reporting-only resume whose last epoch's weights are not proven to be ``latest.weights.h5``.
``check_data_unchanged(run_dir, config, record, data, resume)``
    Refuse data that is not the data the killed run trained on; ``record`` is the task's
    ``task_record`` for the loaded data (``None`` for a task that writes none).
``settle_backup_pair(run_dir, resolved, request, resume, steps_per_epoch)``
    Make the backup a consistent pair; the resume point re-read when the previous pair was chosen.
``best_weights_of(run_dir, resume)`` / ``restore_finished_fit(model, run_dir, resume, best_weights)``
    The best epoch's weights, and the model put where ``fit`` left it, from the files.
``resume_hint(run_dir, resolved, request, epochs_run, steps_per_epoch)``
    The sentence a failed reporting ends with, worded from the resume's own checks.
``older_analysis_note(path, modified_before, analysis)``
    The note for a reporting-only resume whose analyzer wrote nothing.
``data_counts(data)`` / ``data_fingerprint_field(config, data)``
    What the sidecar records about the data, and compares.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); ``keras`` and
``train.common`` names are imported inside the functions that use them.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .args import CheckpointArgs
from .config import StrictConfig
from .errors import ConfigError
from .run_dir import BACKUP_DIR, TASK_FILE, check_model_file, load_resume_point


def resume_point(run_dir: Path, resolved: StrictConfig, request: Any) -> Any:
    """Read where the killed run stopped, or refuse to resume it.

    :param run_dir: the directory :func:`~train.api.run_dir.open_run` accepted for a resume.
    :param resolved: this call's config, every ``"auto"`` filled.
    :param request: the config as the caller built it (``TaskTrainer.config``), which
        ``config_requested.json`` is compared with.
    :return: the :class:`~train.common.resume_state.ResumePoint`.
    :raises ConfigError: see the module contract, "Resume" in ``trainer.py``.
    """
    return check_resume_point(run_dir, resolved, load_resume_point(run_dir, request))


def training_complete_reason(point: Any, resolved: StrictConfig) -> Optional[str]:
    """Say why ``point``'s run has no epoch left to train, or return ``None`` when it has one.

    Interface contract: ``point`` is a resume point (``start``, ``wait``, ``state``),
    ``resolved`` the resolved config (``epochs``, ``checkpoint_args``). Pure, reads no
    file. The reason is a phrase for a log line or a note. A run is complete when its fit
    returned (the sidecar says ``fit_finished``), or every epoch ended, or EarlyStopping
    had already waited out its patience after the last epoch that ended (it decides in
    ``on_epoch_end`` and the process died before ``fit`` returned). A complete run is
    never trained again: its resume is a REPORTING-ONLY one (D-041, D-064). The patience
    rule applies from epoch 2 on: epoch 1 always improves on nothing, so a run killed
    after it had decided nothing.
    """
    if point.state.get("fit_finished"):
        return "its fit returned (fit_state.json says fit_finished)"
    if point.start >= resolved.epochs:
        return f"all {resolved.epochs} of its epochs ended"
    patience = resolved.checkpoint_args.early_stopping_patience
    if point.start > 1 and point.wait >= patience:
        return (
            f"after epoch {point.start} EarlyStopping had waited {point.wait} epochs "
            f"(patience {patience}) and had decided to stop"
        )
    return None


def check_resume_point(run_dir: Path, resolved: StrictConfig, point: Any) -> Any:
    """Apply the refusals of :func:`resume_point` to ``point`` and return it.

    Called again when the backup settles on its previous copy (one epoch earlier).
    """
    from train.common.callbacks import best_checkpoint_path

    checkpoint_cfg: CheckpointArgs = resolved.checkpoint_args
    if not checkpoint_cfg.save_best:
        raise ConfigError(
            f"Cannot resume {run_dir}: checkpoint_args.save_best is false, so the best "
            "weights of the killed segment were never written and a resumed run could "
            "not return to them"
        )
    # DECISION plan-2026-10-31T180033-2faf890a/D-064 (supersedes plan-2026-10-30T132432-a865fa08/D-058)
    # A run with nothing left to train is NOT refused and NOT sent to `fit`: it is a
    # REPORTING-ONLY resume, whether its sidecar says `fit_finished` or the process died
    # between the last epoch and `fit` returning (all epochs ended; EarlyStopping had
    # waited out its patience). Do NOT let it reach `fit`: with `start >= epochs` no epoch
    # ends, `LastEpochWeights` holds nothing and the post-fit steps fail on `None`, and one
    # more epoch that improves would carry on past the stop an uninterrupted run made. Do NOT
    # refuse it either: the training is the expensive part (a refusal sent the review's
    # reproduction back to train again). What the reporting evaluates as the final model is
    # proven by `final_copy_problem`, or the resume is refused.
    if training_complete_reason(point, resolved) is not None:
        return check_finished_fit(run_dir, point)
    if point.best is not None and not Path(best_checkpoint_path(str(run_dir))).is_file():
        raise ConfigError(
            f"Cannot resume {run_dir}: the sidecar records a best value but "
            f"{best_checkpoint_path(str(run_dir))} does not exist"
        )
    if point.best is not None:
        # Plain file I/O, so before the GPU: a best checkpoint cut short by a kill is
        # named here, not met as a Keras traceback after the model is built (D-069).
        check_model_file(Path(best_checkpoint_path(str(run_dir))), f"Cannot resume {run_dir}")
    if "initial" not in point.state:
        raise ConfigError(
            f"Cannot resume {run_dir}: the sidecar holds no epoch-0 evaluation"
        )
    return point


def check_finished_fit(run_dir: Path, point: Any) -> Any:
    """Apply the refusals of a REPORTING-ONLY resume to ``point`` and return it.

    A run with no epoch left (:func:`training_complete_reason`: its sidecar says
    ``fit_finished``, or every epoch ended, or EarlyStopping had decided to stop) and no
    ``results_summary.json`` crashed or was killed while reporting, or between its last
    epoch and ``fit`` returning. Its training is whole, so there is no epoch left to
    refuse for by design, and EarlyStopping having decided to stop is how most runs end.
    What the reporting needs is refused here, before anything is written: the best
    checkpoint (the model after ``fit`` held the best weights), the epoch-0 evaluation,
    and a sidecar and a backup that agree on how many epochs ran. Plain file reading.

    :raises ConfigError: naming the file and the fact, and that the training is kept.
    """
    from train.common.callbacks import best_checkpoint_path

    recorded = sorted(int(key) for key in point.state.get("epochs", {}))
    epochs_run = len(recorded)
    if epochs_run == 0 or recorded != list(range(epochs_run)):
        raise ConfigError(
            f"Cannot resume {run_dir}: the training has no epoch left to run, but "
            f"fit_state.json holds the epochs {recorded}, not 0..n-1, so the number of epochs "
            "that ran is unknown. Choose a new 'out'"
        )
    if point.start != epochs_run:
        raise ConfigError(
            f"Cannot resume {run_dir}: fit_state.json says {epochs_run} epochs ran but the "
            f"backup's latest metadata says {point.start}; the files do not agree on where "
            "the training ended, so the last epoch's weights are not known. Choose a new "
            "'out'"
        )
    best_path = Path(best_checkpoint_path(str(run_dir)))
    if point.best is None or not best_path.is_file():
        raise ConfigError(
            f"Cannot resume {run_dir}: the reporting needs the best weights and "
            f"{best_path} does not exist (no epoch had a finite best). Choose a new 'out'"
        )
    check_model_file(best_path, f"Cannot resume {run_dir}")
    if "initial" not in point.state:
        raise ConfigError(
            f"Cannot resume {run_dir}: the sidecar holds no epoch-0 evaluation"
        )
    return point


def check_final_weights(run_dir: Path, resume: Any, steps_per_epoch: int) -> None:
    """Refuse a reporting-only resume whose last epoch's weights cannot be shown to be right.

    The LAST epoch's weights are not in ``best_model.keras``; the only copy is the epoch-end
    backup ``checkpoints/backup/latest.weights.h5`` (``LastEpochWeights`` was in memory and
    died with the process). It is used only when it is the pair the run's own end wrote:
    metadata ``epoch`` equal to the epochs that ran and ``batch`` 0 (an epoch-end copy), and
    an optimizer step count equal to ``epochs x steps_per_epoch``
    (``resume_state.final_copy_problem``). The previous copy (``.bkp``) is NEVER used here: it is
    one epoch or some batches behind, and ``settle_backup`` (which would put it in place)
    is not called, so nothing in the backup is changed.

    :raises ConfigError: naming ``latest.weights.h5`` and what is wrong with it, and that
        ``best_model.keras`` is untouched.
    """
    from train.common.resume_state import final_copy_problem

    backup = run_dir / BACKUP_DIR
    problem = final_copy_problem(backup, resume.start, steps_per_epoch)
    if problem is not None:
        raise ConfigError(
            f"Cannot resume {run_dir}: its training is complete but the last epoch's weights "
            f"cannot be recovered ({problem}). They exist only in "
            f"{backup / 'latest.weights.h5'}, and the reporting evaluates them as the "
            "final model, so it will not run on any other weights. best_model.keras is "
            "untouched and can be loaded by hand. Choose a new 'out' to train again"
        )


def restore_finished_fit(
    model: Any, run_dir: Path, resume: Any, best_weights: List[Any]
) -> Tuple[Any, Any]:
    """Put the model where ``fit`` left it, from the files, and return ``(history, last)``.

    After ``fit`` the in-memory model holds the BEST weights (EarlyStopping restores them)
    and ``LastEpochWeights`` holds the last epoch's. Here the best weights are
    ``best_model.keras``'s (``best_weights``, read by :func:`best_weights_of`) and the last
    epoch's are ``checkpoints/backup/latest.weights.h5`` (verified by
    :func:`check_final_weights`). The history is ``training_log.csv``'s rows.

    :return: ``(History, LastEpochWeights)``; the model holds the best weights.
    """
    import keras

    from train.common import classification_run
    from train.common.resume_state import BACKUP_WEIGHTS_NAME, rows_as_history

    # DECISION plan-2026-10-31T180033-2faf890a/D-064
    # The optimizer is BUILT before the load, so `load_weights` restores its variables (the
    # moments and the step count) with the model's, and `final_model.keras` carries the
    # optimizer state of the last epoch as a whole run's does. Do NOT drop the build: an
    # unbuilt optimizer skips them and the file held 1 optimizer variable (step count 0)
    # against 161 (step count 36) (review-iter-2 concern 13).
    model.optimizer.build(model.trainable_variables)
    model.load_weights(str(run_dir / BACKUP_DIR / BACKUP_WEIGHTS_NAME))
    last = classification_run.LastEpochWeights()
    last.weights = model.get_weights()
    model.set_weights(best_weights)
    history = keras.callbacks.History()
    history.history = rows_as_history(resume.rows)
    history.epoch = [epoch for epoch, _ in resume.rows]
    return history, last


def resume_hint(
    run_dir: Path, resolved: StrictConfig, request: Any, epochs_run: int, steps_per_epoch: int
) -> str:
    """Return the sentence a failed reporting ends with: can ``resume_interrupted=true`` run it again?

    Interface contract: called from a ``finally`` while the reporting's exception is on its
    way; reads files only. The answer is the resume's own: the checks a resume applies
    before it trains or reports anything (:func:`resume_point`, then
    :func:`check_final_weights`) are applied to the files as they are now, so the hint
    cannot promise a resume that would be refused (``checkpoint_args.save_best`` false, a
    best checkpoint or a last-epoch backup that is missing or torn). ``""`` when the run
    has its summary (a resume of a finished run is refused, and nothing failed to report).
    A refusal is worded as the resume would word it.

    :param epochs_run: the epochs that trained, named in the sentence.
    :param steps_per_epoch: the data's batches per epoch (the proof of the final weights).
    """
    from train.common.run_artifacts import RESULTS_SUMMARY_NAME

    if (run_dir / RESULTS_SUMMARY_NAME).exists():
        return ""
    kept = f" The training is kept ({epochs_run} epochs, best_model.keras, checkpoints/backup)"
    # DECISION plan-2026-10-31T180033-2faf890a/D-064
    # The hint is worded from the resume's OWN checks. Do NOT make it a constant: with
    # `checkpoint_args.save_best` false the old line promised a resume that exited 2
    # ("could not return to them"), and a hint that is not true is worse than none.
    try:
        point = resume_point(run_dir, resolved, request)
        check_final_weights(run_dir, point, steps_per_epoch)
    except ConfigError as refusal:
        return f"{kept}, but resume_interrupted=true would be refused: {refusal}"
    return (
        f"{kept}: run the same request with resume_interrupted=true to run only the "
        "reporting again"
    )


def older_analysis_note(
    path: Path, modified_before: Optional[int], analysis: Dict[str, Any]
) -> Optional[str]:
    """Return a note when a reporting-only resume's ``analyzer.path`` is a file it did not write.

    ``modified_before`` is the file's modification time (ns) before the reporting began,
    ``None`` when it did not exist. The analyzer writes ``analysis_results.json`` before its
    first figure, so a file whose time did not change was not touched by this segment's
    analyzer (it raised before writing, say) and the status read back from it is the earlier
    segment's. ``None`` when the analyzer was skipped, or the file is new or rewritten.
    """
    if not analysis.get("path") or modified_before is None or not path.exists():
        return None
    if path.stat().st_mtime_ns != modified_before:
        return None
    return (
        f"`analyzer.path` names {path}, which this reporting did not write (its modification "
        "time is unchanged: the analyzer of this segment wrote nothing there, see "
        "`failed_analyses` and run.log), so the analyzer's status and numbers in this summary "
        "are those of the earlier segment's file"
    )


def check_data_unchanged(
    run_dir: Path, config: StrictConfig, record: Optional[Dict[str, Any]], data: Any, resume: Any
) -> None:
    """Refuse a resume whose data is not the data the killed run trained on.

    Compares the class names, the geometry, the normalization and the per-channel
    statistics the loader reports now with the run's ``task.json``, and the split
    sizes with the sidecar's ``data_counts``. The loader has already made its one
    statistics pass by this point, so the check costs nothing more.

    :raises ConfigError: naming each difference as ``key: earlier -> now``, and that
        the run cannot be continued on other data.
    """
    from .predictor import read_record, record_differences

    # DECISION plan-2026-10-30T132432-a865fa08/D-069
    # Class names and statistics against `task.json`, split sizes against the sidecar.
    # Do NOT compare the statistics alone: a renamed class leaves them unchanged when
    # its images keep their pixels, and an added image moves them by less than the
    # tolerance in a large set; the names and the sizes catch what they miss. The
    # statistics catch a swap of two class folders (same names, same sizes). A swap
    # that leaves the fit split's statistics equal is caught by the data fingerprint
    # (D-070, D-025): relative path, class and file size of every listed file IN LISTING
    # ORDER (a reordered CSV splits differently), no pixel read; a swap
    # of two same-sized files is caught only by data.fingerprint=content, which hashes bytes.
    lines: List[str] = []
    task_path = run_dir / TASK_FILE
    if record is not None and task_path.is_file():
        lines.extend(record_differences(read_record(task_path), record))
    earlier_counts = resume.state.get("data_counts")
    if not isinstance(earlier_counts, dict):
        raise ConfigError(
            f"Cannot resume {run_dir}: the sidecar holds no record of the data's size, so "
            "it cannot be shown that the data is unchanged. Choose a new 'out'"
        )
    now_counts = data_counts(data)
    for key in sorted(set(earlier_counts) | set(now_counts)):
        if earlier_counts.get(key) != now_counts.get(key):
            lines.append(f"{key}: {earlier_counts.get(key)} -> {now_counts.get(key)}")
    earlier_print = resume.state.get("data_fingerprint")
    now_print = getattr(data, "fingerprint", None)
    now_fields = data_fingerprint_field(config, data)
    earlier_mode = resume.state.get("data_fingerprint_mode")
    now_mode = now_fields.get("data_fingerprint_mode")
    earlier_scheme = resume.state.get("data_fingerprint_scheme")
    now_scheme = now_fields.get("data_fingerprint_scheme")
    if earlier_scheme != now_scheme:
        # DECISION plan-2026-10-31T180033-2faf890a/D-025
        # A sidecar of another scheme is refused, and its digest is NOT compared: the
        # earlier scheme sorted the rows, so it equals nothing the loader computes now and
        # a "fallback to the old comparison" would pass exactly the reordered CSV this
        # exists to refuse. No legacy path (user rule).
        lines.append(
            f"data_fingerprint_scheme: {earlier_scheme or 'none'} -> {now_scheme or 'none'} "
            "(the run recorded its files by a rule that could not tell a reordered CSV, so "
            "it cannot be shown that the split is the same)")
    elif earlier_mode != now_mode:
        lines.append(
            f"data_fingerprint_mode: {earlier_mode or 'none'} -> {now_mode or 'none'} (the "
            "run recorded its data by another rule, so the two digests cannot be compared)")
    if earlier_scheme == now_scheme and earlier_print != now_print:
        lines.append(
            "data_fingerprint: "
            f"{'none' if earlier_print is None else str(earlier_print)[:12]} -> "
            f"{'none' if now_print is None else str(now_print)[:12]} (a file, its class, "
            "its size, the order of the files or, with data.fingerprint=content, its bytes is "
            "not the one the run started with)")
    if lines:
        raise ConfigError(
            f"Cannot resume {run_dir}: the data is not the data the run started with ("
            + "; ".join(lines)
            + "). The earlier epochs trained on the earlier data, and a resume would "
            "standardize, split and label differently; restore the data as it was, or "
            "choose a new 'out'"
        )


def settle_backup_pair(
    run_dir: Path, resolved: StrictConfig, request: Any, resume: Any, steps_per_epoch: int
) -> Any:
    """Make the backup a consistent pair, or refuse; see ``resume_state.settle_backup``.

    :return: ``resume``, or, when the previous copy had to be chosen (so the run goes
        back one epoch), the resume point re-read for that epoch with every refusal of
        :func:`resume_point` applied to it again: the CSV rows, the sidecar entry and
        the best value are those of the epoch ``fit`` will start at.
    :raises ConfigError: when neither the latest nor the previous copy is sound,
        naming both files. Nothing has been changed then.
    """
    from train.common.resume_state import settle_backup

    try:
        label, epoch = settle_backup(run_dir / BACKUP_DIR, steps_per_epoch)
    except ValueError as exc:
        raise ConfigError(
            f"Cannot resume {run_dir}: {exc}. The run cannot be continued from its "
            "backup; choose a new 'out'"
        ) from exc
    if label == "latest" and epoch == resume.start:
        return resume
    # DECISION plan-2026-10-30T132432-a865fa08/D-069
    # When the previous pair is the one that is sound, the run goes back to ITS epoch
    # and the resume point is read AGAIN (CSV rows, sidecar entry, best, wait) for that
    # epoch, with every refusal applied to it. Do NOT keep the point read from the
    # latest metadata: its rows include the epoch `fit` is about to train again, so the
    # CSV would hold that epoch twice (the duplicate row a harness copy of Keras's
    # steps produced in review-iter-3). The point is read again for the previous pair
    # even when its epoch is the same: its `batch` (how much of the epoch the copy had
    # taken, D-063) is the one the resumed epoch is shortened by.
    again = check_resume_point(
        run_dir, resolved, load_resume_point(run_dir, request)
    )
    complete = training_complete_reason(again, resolved)
    if complete is not None:
        # The latest pair said an epoch was left; the pair that is sound says there is none.
        # That is a reporting-only resume, which the run sequence did not choose and must not be
        # entered by the back door: the files disagree on where the training stopped.
        raise ConfigError(
            f"Cannot resume {run_dir}: the latest backup copy is unusable and the previous "
            f"one records epoch {again.start}, where {complete}; the files do not agree on "
            "where the training stopped. Choose a new 'out'"
        )
    return again


def best_weights_of(run_dir: Path, resume: Any) -> Optional[List[Any]]:
    """Return the weights of ``best_model.keras`` (``get_weights`` order), or ``None``.

    ``None`` when the killed segment had no finite best, so no checkpoint was
    written. Loaded AFTER ``setup_gpu``: it creates tensors.
    """
    import keras

    from train.common.callbacks import best_checkpoint_path

    if resume.best is None:
        return None
    path = best_checkpoint_path(str(run_dir))
    try:
        return keras.models.load_model(path, compile=False).get_weights()
    except (OSError, ValueError, KeyError) as exc:
        raise ConfigError(
            f"Cannot resume {run_dir}: {path} cannot be loaded ({str(exc).splitlines()[0][:150]}); "
            "the best weights of the killed segment are unreadable. Choose a new 'out'"
        ) from exc


def data_counts(data: Any) -> Dict[str, int]:
    """The sizes of the three splits a run trained and was selected and tested on (D-069)."""
    return {
        "n_train": int(data.n_train),
        "n_select": int(data.n_select),
        "n_test": int(data.n_test),
    }


def data_fingerprint_field(config: Any, data: Any) -> Dict[str, str]:
    """The sidecar's fingerprint keys of a file-based data record, ``{}`` for one with none (D-070).

    ``data_fingerprint`` is the digest and ``data_fingerprint_mode`` the ``data.fingerprint``
    value that made it (``listing`` or ``content``), so a resume can say which of the two
    changed. ``data_fingerprint_scheme`` is how the rows were ordered and hashed
    (``classification_data.FINGERPRINT_SCHEME``): a sidecar without it, or with another one,
    was written by a fingerprint that could not tell a reordered CSV, and is refused.
    """
    fingerprint = getattr(data, "fingerprint", None)
    if fingerprint is None:
        return {}
    from train.common.classification_data import FINGERPRINT_SCHEME

    return {
        "data_fingerprint": fingerprint,
        "data_fingerprint_mode": getattr(config.data, "fingerprint", None),
        "data_fingerprint_scheme": FINGERPRINT_SCHEME,
    }
