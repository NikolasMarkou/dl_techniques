"""The callback list of a fit, in the order the resume machinery needs.

``TaskTrainer._callbacks`` is one line that calls :func:`build_callbacks`; the list is built here so
the sequence in ``run_sequence`` reads as a sequence and a task that overrides ``_callbacks`` (a test
inserts a scripted monitor or a kill) still calls ``super()`` and edits the list it gets back.

Interface contract
------------------
``build_callbacks(trainer, config, data, run_dir, vis_dir, baseline, experiment_name, last_weights,
resume=None, best_weights=None)``
    :param trainer: the task's :class:`~train.api.trainer.TaskTrainer` (its hooks and ``task``).
    :param config: the RESOLVED config.
    :param data: the loaded data (``steps_per_epoch`` is read).
    :param run_dir: the run directory; ``vis_dir`` the figures' directory, already created while
        ``analysis_args.figures`` is true.
    :param baseline: the epoch-0 metrics (the dashboard's marker).
    :param last_weights: the ``LastEpochWeights`` callback that will hold the last epoch's weights.
    :param resume: the :class:`~train.common.resume_state.ResumePoint` of a resumed run, else ``None``;
        ``best_weights`` the best epoch's weights then.
    :return: ``(epoch_line, callbacks)``: the ``EpochLogLine`` (its ``epoch_times`` is read after the
        fit) and the list, in this order: ``LearningRateLogger`` first; ``create_callbacks``' list
        (EarlyStopping, ModelCheckpoint unless ``save_best`` is false, CSV logger, TerminateOnNaN, ...);
        ``last_weights``; the task's ``extra_callbacks``; ``EpochLogLine``; the task's
        ``figure_callbacks`` and the dashboard while ``analysis_args.figures``; the resume trio last
        (``RestoreFitState`` on a resume only, ``FitStateSidecar``, the backup callback, and
        ``FinishKilledEpoch`` directly before EarlyStopping when ``backup_every_steps`` is set).

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); ``keras`` and
``train.common`` names are imported inside the function.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .args import AnalysisArgs, CheckpointArgs, DebugArgs, LoggerArgs, MetricArgs, OptimArgs
from .config import StrictConfig
from .run_dir import BACKUP_DIR, CHECKPOINTS_DIR


def build_callbacks(
    trainer: Any,
    config: StrictConfig,
    data: Any,
    run_dir: Path,
    vis_dir: Path,
    baseline: Dict[str, float],
    experiment_name: str,
    last_weights: Any,
    resume: Any = None,
    best_weights: Optional[List[Any]] = None,
) -> Tuple[Any, List[Any]]:
    """Return the epoch-line callback and the full callback list, in the D-022 order.

    :param trainer: the :class:`~train.api.trainer.TaskTrainer`: ``task`` names the results
        prefix and ``extra_callbacks``, ``epoch_line_metrics`` and ``figure_callbacks`` are the
        hooks the list is built around.
    :param resume: the :class:`~train.common.resume_state.ResumePoint` of a resumed
        run, else ``None``.
    :param best_weights: the best epoch's weights of a resumed run, else ``None``.
    """
    import keras

    from train.common.callbacks import EpochLogLine, LearningRateLogger, create_callbacks
    from train.common.classification_viz import TrainingDashboardCallback
    from train.common.resume_state import (
        FIT_STATE_NAME,
        FitStateSidecar,
        RestoreFitState,
        StepBackupAndRestore,
        rows_as_history,
    )

    optim_cfg: OptimArgs = config.optim_args
    metric_cfg: MetricArgs = config.metric_args
    checkpoint_cfg: CheckpointArgs = config.checkpoint_args
    logger_cfg: LoggerArgs = config.logger_args
    analysis_cfg: AnalysisArgs = config.analysis_args
    debug_cfg: DebugArgs = config.debug_args
    every = analysis_cfg.epoch_analysis_every
    resumed: Dict[str, Any] = {}
    if resume is not None:
        # The earlier segment's epoch-0 analysis and epoch-1 analysis are on disk; the
        # analyzer would otherwise redo both, on the weights of a model BackupAndRestore
        # has not yet restored, and overwrite them.
        resumed = dict(
            initial_value_threshold=resume.best,
            csv_initial_rows=resume.rows,
            analyzer_on_train_begin=False,
            # A resume inside epoch 0 (a `backup_every_steps` copy) has not yet reached
            # the epoch-1 analysis; any later resume is past it.
            analyzer_first_epoch=resume.start == 0,
        )

    callbacks, _ = create_callbacks(
        model_name=experiment_name,
        results_dir_prefix=trainer.task,
        run_dir=str(run_dir),
        monitor=metric_cfg.watch_metric,
        monitor_mode=metric_cfg.watch_mode,
        patience=checkpoint_cfg.early_stopping_patience,
        use_lr_schedule=optim_cfg.schedule != "constant",
        analyzer_epoch_frequency=max(every, 1),
        include_tensorboard=logger_cfg.tensorboard is not None,
        include_terminate_on_nan=debug_cfg.terminate_on_nan,
        include_analyzer=every > 0,
        analyzer_start_epoch=max(every, 1),
        atomic_best_checkpoint=True,
        **resumed,
    )
    if not checkpoint_cfg.save_best:
        callbacks = [
            callback
            for callback in callbacks
            if not isinstance(callback, keras.callbacks.ModelCheckpoint)
        ]
    # Index 0: `lr` must be in `logs` before the CSV logger reads it, and it is the
    # rate at the START of the epoch (plan-2026-09-19T040641-db6932ec/D-020: the
    # default reads the NEXT epoch's first-step rate under a per-step schedule).
    callbacks.insert(0, LearningRateLogger(at_epoch_start=True))
    callbacks.append(last_weights)
    callbacks.extend(trainer.extra_callbacks(config, data))
    epoch_line = EpochLogLine(trainer.epoch_line_metrics(config))
    # After every callback that edits `logs`, before the dashboard (whose redraw
    # must stay outside this line's clock).
    callbacks.append(epoch_line)
    if analysis_cfg.figures:
        # The task's own per-epoch figures: after every callback that edits `logs`,
        # before the dashboard (`SegmentationGridCallback` in
        # `train.convunext.segmentation_viz` documents the same order).
        callbacks.extend(trainer.figure_callbacks(config, data, vis_dir))
        # DECISION plan-2026-10-31T180033-2faf890a/D-022: the dashboard stays HERE, not
        # behind `figure_callbacks`. It reads only the keys the logs hold (`loss`,
        # `accuracy`, `miou`, `lr`, ...) and draws a panel per key found, so a task
        # without `accuracy` needs nothing; and it needs the baseline, the experiment
        # name and the resumed rows, which the hook's signature does not carry. Do NOT
        # move it into the classification task: a second task would then have to
        # remember it, and its position (last, after `EpochLogLine`) would be each
        # task's to get right.
        # Last: it reads `logs['lr']`, written by LearningRateLogger above.
        callbacks.append(
            TrainingDashboardCallback(
                out_path=vis_dir / "training_dashboard.png",
                baseline_fn=lambda _model: dict(baseline),
                title=f"{experiment_name} (seed {config.seed})",
                initial_history=None if resume is None else rows_as_history(resume.rows),
                initial_baseline=None if resume is None else dict(baseline),
            )
        )
    # The resume trio goes last, in the order `train.common.resume_state` documents:
    # the sidecar after EarlyStopping (it reads its best and wait), the backup after
    # the sidecar, so a kill at any instant leaves a sidecar entry for every epoch the
    # backup claims.
    stopper = next(c for c in callbacks if isinstance(c, keras.callbacks.EarlyStopping))
    plateau = next(
        (c for c in callbacks if isinstance(c, keras.callbacks.ReduceLROnPlateau)), None
    )
    if resume is not None:
        callbacks.append(
            RestoreFitState(
                stopper, resume.best, resume.wait, best_weights, resume.best_epoch,
                reduce_lr=plateau, reduce_lr_state=resume.reduce_lr,
            )
        )
    callbacks.append(
        FitStateSidecar(stopper, run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME, reduce_lr=plateau)
    )
    # DECISION plan-2026-10-30T132432-a865fa08/D-069
    # `double_checkpoint=True`: the backup keeps the previous pair as `*.bkp`. Do NOT
    # drop it to save the second copy's disk space: a kill inside the weights write
    # leaves the only copy torn, and the run can never be resumed (review-iter-3
    # concern 4, measured exposure 0.6 to 3.3 s of every epoch). `settle_backup`
    # chooses between the pair before `fit`; Keras's own fallback (on `OSError` only)
    # stays as the second line.
    # DECISION plan-2026-10-31T180033-2faf890a/D-038
    # `backup_every_steps` swaps in `StepBackupAndRestore`; unset, the stock callback is
    # built exactly as before (the golden). Do NOT pass `save_freq=N` to the stock one:
    # it skips the epoch end, records the wrong epoch after a resume and reruns the whole
    # killed epoch (resume_state.StepBackupAndRestore says what was measured; D-063 for
    # what the resume runs instead).
    if checkpoint_cfg.backup_every_steps is None:
        callbacks.append(
            keras.callbacks.BackupAndRestore(
                str(run_dir / BACKUP_DIR), double_checkpoint=True, delete_checkpoint=False
            )
        )
    else:
        backup = StepBackupAndRestore(
            str(run_dir / BACKUP_DIR),
            save_every_steps=checkpoint_cfg.backup_every_steps,
            steps_per_epoch=data.steps_per_epoch,
        )
        callbacks.append(backup)
        # DECISION plan-2026-10-31T180033-2faf890a/D-063
        # `FinishKilledEpoch` goes directly BEFORE `EarlyStopping` (so after `TerminateOnNaN`,
        # which `create_callbacks` put in front of it). Its batch-end half must see a stop request
        # `TerminateOnNaN` made, and its epoch-end half must give the stop flag back before
        # `EarlyStopping` decides, or a request of EarlyStopping's would be undone or an
        # unrelated one kept. Do NOT append it with the others.
        callbacks.insert(callbacks.index(stopper), backup.finish)
    return epoch_line, callbacks
