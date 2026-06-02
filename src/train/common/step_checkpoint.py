"""Unified step-based checkpoint, CSV-logging, plotting, and analysis callback.

This module hosts a single :class:`StepCheckpointCallback` that is the
behavioral superset of the six near-identical copies that previously lived
inside individual trainers (gpt2, wave_field_llm, cliffordnet NLP / NLP-UNet /
NLP-routing, and CLIP). For datasets where one epoch spans hundreds of
thousands of steps, epoch-level checkpoints are far too infrequent; this
callback saves on a fixed *step* cadence, keeps a rolling window of the most
recent ``.keras`` files, writes a step-level CSV, periodically refreshes
loss/accuracy plots, and (optionally) runs a weight/spectral analysis.

The six original variants diverged along a small set of axes, all of which are
now constructor parameters:

- **Step source** (``step_counter``): vanilla copies kept their own integer
  counter; the routing copy delegated to an external :class:`StepCounter`-like
  object exposing a ``.value`` attribute. Pass that object as ``step_counter``
  and this callback reads from it instead of incrementing internally.
- **GC on save** (``gc_on_save``): the NLP / routing / CLIP copies called
  ``gc.collect()`` after each save to release the transient NumPy copies Keras
  allocates during native ``.keras`` serialization; the vanilla copies did not.
- **Analysis** (``analyze_every_steps``): ``0`` disables analysis entirely (the
  CLIP copy never analyzed); ``> 0`` runs :class:`ModelAnalyzer` synchronously.
- **CSV schema** (``csv_fields``): ``None`` derives field names dynamically from
  the first logged row (NLP / CLIP behavior); a fixed tuple uses
  ``extrasaction="ignore"`` / ``restval=""`` so missing or extra keys never
  raise (routing behavior), and adds ``epoch`` / ``lr`` columns.

``_global_step`` (internal-counter mode) persists across multiple ``fit()``
calls because it is an instance attribute seeded once from ``initial_step`` and
never reset in :meth:`on_train_begin` — this is what makes resume work and what
lets the CLIP trainer drive two stages through one continuous timeline.

This callback is intentionally NOT registered with
``@keras.saving.register_keras_serializable``: Keras callbacks are not part of a
``.keras`` archive, so none of the six originals registered and neither does
this one.
"""

import csv
import gc
import glob
import os
from typing import Any, Dict, Optional, Sequence

import keras

from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig
from dl_techniques.utils.logger import logger

from train.common.step_plots import plot_step_metrics


class StepCheckpointCallback(keras.callbacks.Callback):
    """Save checkpoints and step-level metrics at fixed step intervals.

    Maintains a rolling window of the ``max_checkpoints`` most recent
    ``step_NNNNNNN.keras`` files plus a ``final.keras`` written on
    :meth:`on_train_end`. Writes per-step metrics to ``training_log.csv`` every
    ``log_every_steps`` steps, periodically refreshes step plots, and optionally
    runs a synchronous weight/spectral analysis.

    :param save_dir: Run directory. ``checkpoints/`` and ``training_log.csv``
        are created inside it (and ``step_analysis/`` when analysis is on).
    :param save_every_steps: Checkpoint interval in training steps.
    :param analyze_every_steps: Analysis interval in steps. ``0`` disables
        analysis (no ``step_analysis/`` dir, no analyzer warning).
    :param max_checkpoints: Keep only the N most recent ``step_*.keras`` files.
    :param model_name: Label for analyzer output.
    :param initial_step: Starting step count, for resume. Ignored when an
        external ``step_counter`` is supplied (that object owns the count).
    :param log_every_steps: Step-level CSV logging interval.
    :param plot_every_steps: Plot refresh interval. ``0`` disables periodic
        refresh (a final plot is still emitted on train end).
    :param step_counter: Optional external counter exposing a ``.value`` int
        attribute. When provided, :attr:`global_step` reads from it and this
        callback does NOT increment a counter of its own (the external counter
        is responsible for advancing). When ``None``, an internal counter is
        incremented in :meth:`on_train_batch_end`.
    :param gc_on_save: Call ``gc.collect()`` after each checkpoint save to free
        the transient NumPy buffers Keras allocates during ``.keras`` save.
    :param csv_fields: Optional fixed CSV schema. ``None`` derives field names
        dynamically from the first logged row. A tuple writes a fixed-schema CSV
        (``extrasaction="ignore"``, ``restval=""``) and adds ``epoch`` / ``lr``
        columns to each row.
    """

    def __init__(
        self,
        save_dir: str,
        save_every_steps: int = 25000,
        analyze_every_steps: int = 50000,
        max_checkpoints: int = 3,
        model_name: str = "model",
        initial_step: int = 0,
        log_every_steps: int = 100,
        plot_every_steps: int = 25000,
        step_counter: Optional[Any] = None,
        gc_on_save: bool = False,
        csv_fields: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.save_every_steps = save_every_steps
        self.analyze_every_steps = analyze_every_steps
        self.max_checkpoints = max_checkpoints
        self.model_name = model_name
        self._log_every_steps = log_every_steps
        self._plot_every_steps = plot_every_steps
        self._save_dir = save_dir
        self._gc_on_save = gc_on_save
        self._csv_fields = tuple(csv_fields) if csv_fields is not None else None

        # Step source: external counter (routing) vs internal counter (vanilla).
        # The internal counter is seeded once here and NEVER reset in
        # on_train_begin, so it persists across multiple fit() calls (resume).
        self._counter = step_counter
        self._internal_step = initial_step
        self._current_epoch = 0

        self._ckpt_dir = os.path.join(save_dir, "checkpoints")
        self._analysis_dir = os.path.join(save_dir, "step_analysis")
        os.makedirs(self._ckpt_dir, exist_ok=True)
        if self.analyze_every_steps > 0:
            os.makedirs(self._analysis_dir, exist_ok=True)
            # Preserve the routing variant's synchronous-analyzer warning; only
            # emitted when analysis is actually enabled.
            logger.warning(
                "Step analysis runs synchronously on the training thread; "
                "expect a multi-second stall every "
                f"{self.analyze_every_steps} steps."
            )

        self._csv_path = os.path.join(save_dir, "training_log.csv")
        self._csv_file: Optional[Any] = None
        self._csv_writer: Optional[csv.DictWriter] = None

        self._analysis_config = AnalysisConfig(
            analyze_weights=True,
            analyze_spectral=True,
            analyze_calibration=False,
            analyze_information_flow=False,
            analyze_training_dynamics=False,
            verbose=False,
        )
        logger.info(
            f"StepCheckpointCallback: save every {save_every_steps} steps, "
            f"analyze every {analyze_every_steps} steps "
            f"({'off' if analyze_every_steps == 0 else 'on'}), "
            f"keep max {max_checkpoints} checkpoints, "
            f"log every {log_every_steps} steps, "
            f"plot every {plot_every_steps} steps"
        )

    @property
    def global_step(self) -> int:
        """Current global training step.

        Reads from the external ``step_counter`` when one was supplied,
        otherwise returns the internal counter.
        """
        if self._counter is not None:
            return self._counter.value
        return self._internal_step

    # Back-compat: several call sites and the routing variant referenced
    # ``_global_step``; keep it as an alias of the public property.
    @property
    def _global_step(self) -> int:
        return self.global_step

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._current_epoch = epoch

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        # Advance the internal counter only when we own it.
        if self._counter is None:
            self._internal_step += 1

        step = self.global_step
        # The ``step > 0`` guard (from the routing variant) is universally safe:
        # with an internal counter the first batch lands on step 1, so a
        # ``% N == 0`` gate can never accidentally fire at step 0.
        if step > 0 and step % self._log_every_steps == 0:
            self._log_metrics(logs)
        if step > 0 and step % self.save_every_steps == 0:
            self._save_checkpoint()
        if (
            self.analyze_every_steps > 0
            and step > 0
            and step % self.analyze_every_steps == 0
        ):
            self._run_analysis()
        if (
            self._plot_every_steps > 0
            and step > 0
            and step % self._plot_every_steps == 0
        ):
            self._plot_metrics()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        # Keras puts val_* keys into ``logs`` only at epoch end. For a fixed
        # CSV schema (routing) we flush a row carrying those validation metrics;
        # the dynamic-schema variants did not write an epoch-end row, so we keep
        # that behavior to avoid changing their CSV layout.
        if self._csv_fields is not None:
            self._log_metrics(logs, val_logs=logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        # Both stages of multi-stage training call this; ``final.keras`` always
        # reflects the most recent trained weights.
        self.close()
        os.makedirs(self._ckpt_dir, exist_ok=True)
        path = os.path.join(self._ckpt_dir, "final.keras")
        try:
            self.model.save(path)
            logger.info(f"Final checkpoint saved: {path}")
        except Exception:
            # LESSON: periodic side-effect callbacks must surface failures
            # loudly rather than swallow them silently.
            logger.error(
                f"Failed to write final.keras at {path}", exc_info=True
            )
        self._plot_metrics()

    def close(self) -> None:
        """Idempotent close of the CSV handle.

        Safe to call when the handle was never opened or is already closed
        (used by the CLIP trainer between stages).
        """
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None

    def _current_lr(self) -> float:
        """Best-effort current learning rate (0.0 if unavailable)."""
        opt = getattr(self.model, "optimizer", None)
        if opt is None:
            return 0.0
        lr = opt.learning_rate
        try:
            if callable(lr):
                return float(lr(opt.iterations))
            return float(keras.ops.convert_to_numpy(lr))
        except Exception:
            return 0.0

    def _log_metrics(
        self,
        logs: Optional[Dict[str, Any]],
        val_logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if logs is None and val_logs is None:
            return

        os.makedirs(os.path.dirname(self._csv_path) or ".", exist_ok=True)

        if self._csv_fields is not None:
            self._log_metrics_fixed(logs, val_logs)
        else:
            self._log_metrics_dynamic(logs)

    def _log_metrics_dynamic(self, logs: Optional[Dict[str, Any]]) -> None:
        """Dynamic-schema CSV row (NLP / vanilla / CLIP behavior).

        Coerces each value to ``float`` defensively (folded in from the CLIP
        variant) and pads any field that appeared later in the run.
        """
        if logs is None:
            return
        row: Dict[str, Any] = {"step": self.global_step}
        for k, v in logs.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                continue
        if self._csv_writer is None or not os.path.exists(self._csv_path):
            # Re-open if an external rm removed the file mid-run.
            if self._csv_file is not None:
                try:
                    self._csv_file.close()
                except Exception:
                    pass
            self._csv_file = open(self._csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys()),
            )
            if self._csv_file.tell() == 0:
                self._csv_writer.writeheader()
        # Pad any new keys that appeared mid-run so the writer never raises.
        for k in self._csv_writer.fieldnames:
            row.setdefault(k, None)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _log_metrics_fixed(
        self,
        logs: Optional[Dict[str, Any]],
        val_logs: Optional[Dict[str, Any]],
    ) -> None:
        """Fixed-schema CSV row (routing behavior).

        Uses ``extrasaction="ignore"`` / ``restval=""`` so missing or extra
        keys never raise, and adds ``epoch`` / ``lr`` columns.
        """
        if self._csv_writer is None or not os.path.exists(self._csv_path):
            if self._csv_file is not None:
                try:
                    self._csv_file.close()
                except Exception:
                    pass
            self._csv_file = open(self._csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=list(self._csv_fields),
                extrasaction="ignore",
                restval="",
            )
            if self._csv_file.tell() == 0:
                self._csv_writer.writeheader()
        row: Dict[str, Any] = {
            "step": self.global_step,
            "epoch": self._current_epoch,
            "lr": self._current_lr(),
        }
        if logs:
            row.update(logs)
        if val_logs:
            row.update(
                {k: v for k, v in val_logs.items() if k.startswith("val_")}
            )
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _plot_metrics(self) -> None:
        try:
            plot_step_metrics(self._csv_path, self._save_dir)
        except Exception as e:
            logger.warning(f"Step plot failed at step {self.global_step}: {e}")

    def _save_checkpoint(self) -> None:
        # DECISION plan_2026-05-17_7ed2d007/D-001: re-create dir on every save.
        # A concurrent external rm of the parent (e.g. `git stash -u`) would
        # otherwise crash the next save with FileNotFoundError mid-run.
        os.makedirs(self._ckpt_dir, exist_ok=True)
        path = os.path.join(
            self._ckpt_dir, f"step_{self.global_step:07d}.keras"
        )
        try:
            self.model.save(path)
        except Exception:
            # LESSON: periodic side-effect callbacks must surface failures
            # loudly. Log and skip cleanup so the next save can retry.
            logger.error(
                f"Checkpoint save failed at step {self.global_step} ({path})",
                exc_info=True,
            )
            return
        if self._gc_on_save:
            # Release the transient NumPy copies Keras allocates during native
            # .keras serialization (weights + AdamW m/v slots ~= model size).
            gc.collect()
        logger.info(f"Checkpoint saved: {path} (step {self.global_step:,})")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove oldest checkpoints beyond ``max_checkpoints``."""
        ckpts = sorted(glob.glob(
            os.path.join(self._ckpt_dir, "step_*.keras")
        ))
        while len(ckpts) > self.max_checkpoints:
            old = ckpts.pop(0)
            try:
                os.remove(old)
                logger.info(f"Removed old checkpoint: {old}")
            except OSError as exc:
                logger.warning(f"Could not remove {old}: {exc}")

    def _run_analysis(self) -> None:
        if self.analyze_every_steps <= 0:
            return
        os.makedirs(self._analysis_dir, exist_ok=True)
        step_dir = os.path.join(
            self._analysis_dir, f"step_{self.global_step:07d}"
        )
        try:
            analyzer = ModelAnalyzer(
                models={self.model_name: self.model},
                config=self._analysis_config,
                output_dir=step_dir,
            )
            analyzer.analyze()
            logger.info(f"Step analysis complete: step {self.global_step:,}")
        except Exception as e:
            logger.error(
                f"Step analysis failed at step {self.global_step}: {e}"
            )
