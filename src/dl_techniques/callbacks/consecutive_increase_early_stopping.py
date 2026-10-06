"""Early stopping on three CONSECUTIVE INCREASES, the rule TopoLM was trained under.

Why this exists instead of ``keras.callbacks.EarlyStopping``
-------------------------------------------------------------
The stock callback answers "has the best value been beaten and not recovered
for N evaluations". The rule this implements answers a different question: "did
the monitored value increase on three evaluations in a row?"

Those disagree exactly where it matters. A validation curve that dips to a new
best at evaluation 2 and then rises at 3 and 4 is stopped by this rule and
tolerated by the stock one; a curve that rises, falls back to exactly its
previous value, and rises again is stopped by neither but recovered by neither.
The distinction matters for a run whose monitored value is expected to be noisy
at a fine evaluation cadence, where "consecutive" is the only structure the rule
can rely on.

Restore semantics
-----------------
Weights are snapshotted on every evaluation that did NOT increase, and the
snapshot taken immediately before the final streak is restored. That is the
evaluation with the lowest value of the streak, which is not the best value seen
over the run -- deliberately, and differently from
``restore_best_weights=True``.

Snapshot storage is a choice, not an implementation detail. In-memory
``model.get_weights()`` is the default because it is exact and needs no
filesystem; ``checkpoint_path`` switches to ``model.save_weights`` per snapshot,
which is the only workable option for a model whose weights do not fit in RAM.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM,
      ICLR 2025, Section 3: early stopping after three consecutive increases on
      validation loss. (https://arxiv.org/abs/2410.11516)
"""

import os
from typing import Any, Dict, Optional

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


class ConsecutiveIncreaseEarlyStopping(keras.callbacks.Callback):
    """Stop after ``patience`` evaluations whose monitored value strictly rises.

    The streak is broken -- and a fresh snapshot taken -- by any evaluation whose
    value did not increase, including an exact repeat. A single flat evaluation
    resets the count, which is the rule's whole content: three rises are evidence
    of a trend, and rise-flat-rise is not.

    :param monitor: Metric name to watch. Default ``'val_loss'``.
    :type monitor: str
    :param patience: Consecutive increases tolerated before stopping. Default 3,
        the paper's value.
    :type patience: int
    :param min_delta: Slack below which an increase is not counted. Default
        ``0.0``, so any rise counts; a positive value makes the rule tolerant of
        numerical noise at a fine cadence.
    :type min_delta: float
    :param mode: ``'min'`` or ``'max'``, naming the good direction.
    :type mode: str
    :param restore_weights: Restore the snapshot taken before the final streak.
    :type restore_weights: bool
    :param checkpoint_path: Directory for on-disk snapshots. ``None`` keeps them
        in memory, which is exact and needs no filesystem; a path is required for
        a model whose weights do not fit in RAM.
    :type checkpoint_path: Optional[str]
    :param verbose: Print each decision.
    :type verbose: int
    :raises ValueError: If ``patience`` is below 1, if ``mode`` is unknown, or if
        ``min_delta`` is negative -- naming the offending value.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_weights: bool = True,
        checkpoint_path: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()

        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(
                f"mode must be 'min' or 'max', got {mode!r}"
            )
        if min_delta < 0.0:
            raise ValueError(
                f"min_delta must be >= 0, got {min_delta}"
            )

        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_weights = bool(restore_weights)
        self.checkpoint_path = checkpoint_path
        self.verbose = int(verbose)

        self._previous: Optional[float] = None
        self._streak = 0
        self._best_snapshot: Optional[Any] = None
        self._snapshot_is_disk = False
        self.stopped_at_evaluation: Optional[int] = None

    # -- Keras plumbing -------------------------------------------------

    def set_model(self, model: keras.Model) -> None:
        """Remember the model so a snapshot can be taken at the first evaluation."""
        super().set_model(model)
        if self.checkpoint_path is not None:
            os.makedirs(self.checkpoint_path, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Evaluate one monitored value and act on it."""
        logs = logs or {}
        if self.monitor not in logs:
            logger.debug(
                f"{self.__class__.__name__}: {self.monitor!r} absent from this "
                f"epoch's logs ({sorted(logs)}); streak left untouched"
            )
            return

        value = float(logs[self.monitor])
        increased = self._is_increase(value)

        # Snapshot FIRST, and only when this evaluation did not increase. The
        # order is load-bearing: snapshotting before deciding overwrites the
        # pre-streak weights with the first rising evaluation's, so the restore
        # lands on the value the streak started from rather than the value before
        # it -- which is a different model, silently.
        if not increased:
            self._streak = 0
            self._snapshot()
        else:
            self._streak += 1

        if self.verbose:
            logger.info(
                f"[consecutive-increase] {self.monitor}={value:.6f} "
                f"increased={increased} streak={self._streak}/{self.patience}"
            )

        if self._streak >= self.patience:
            self.stopped_at_evaluation = int(epoch)
            logger.info(
                f"[consecutive-increase] {self.monitor} rose on "
                f"{self.patience} consecutive evaluations; stopping at "
                f"evaluation {epoch}."
            )
            self.model.stop_training = True

    def on_train_end(self, logs: Optional[Dict[str, float]] = None):
        """Restore the snapshot taken before the final streak."""
        if not self.restore_weights or self._best_snapshot is None:
            return
        if self.stopped_at_evaluation is None:
            return

        if self._snapshot_is_disk:
            path = self._best_snapshot
            self.model.load_weights(path)
            logger.info(f"[consecutive-increase] restored weights from {path}")
        else:
            self.model.set_weights(self._best_snapshot)
            logger.info(
                "[consecutive-increase] restored the in-memory weights from "
                "before the final streak"
            )

    # -- internals ------------------------------------------------------

    def _is_increase(self, value: float) -> bool:
        """Whether ``value`` counts as an increase over the previous evaluation."""
        if self._previous is None:
            self._previous = value
            return False
        previous, self._previous = self._previous, value
        if self.mode == "min":
            return value > previous + self.min_delta
        return value < previous - self.min_delta

    def _snapshot(self) -> None:
        """Take the snapshot a future restore will come from.

        Called only on an evaluation that did not increase, which is what makes
        the restore the last GOOD state rather than the best state seen so far.
        """
        if self.checkpoint_path is not None:
            self._snapshot_is_disk = True
            path = os.path.join(
                self.checkpoint_path, "consecutive_increase_best.weights.h5"
            )
            self.model.save_weights(path)
            self._best_snapshot = path
        else:
            self._snapshot_is_disk = False
            self._best_snapshot = self.model.get_weights()

    @property
    def streak(self) -> int:
        """Length of the current run of consecutive increases."""
        return self._streak

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument."""
        # `keras.callbacks.Callback` has no `get_config` of its own, so this
        # starts from an empty dict rather than calling a base method that does
        # not exist.
        config: Dict[str, Any] = {}
        config.update({
            "monitor": self.monitor,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "restore_weights": self.restore_weights,
            "checkpoint_path": self.checkpoint_path,
            "verbose": self.verbose,
        })
        return config