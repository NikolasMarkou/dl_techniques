"""Report the spatial smoothness loss separately from the task loss.

Why the split is necessary
--------------------------
The taps contribute through ``add_loss``, so the training loss Keras reports is
``task + sum(alpha_k * SL_k)``. That number answers "what is the objective",
which is not the same question as "how is the model doing" or "is topography
forming". A run whose task loss is rising can have a *falling* spatial loss --
the regularizer is doing its job -- and only the two numbers read separately tell
you which is happening.

It also exposes a trap this callback is built to make visible: if the backbone's
own losses are not aggregated into the objective (a CLM head constructed without
``aggregate_backbone_losses=True``), the spatial term is computed every step and
discarded. The reported loss then does not contain it, yet ``task_loss`` is still
computed as ``loss - spatial_total``, so ``task_loss`` comes out too low and
``spatial/share`` comes out non-zero (measured: share 0.43 on a head whose
regularizer did nothing). Neither number shows the problem. Two signals do:
``set_model`` warns when the model exposes ``aggregate_backbone_losses is False``,
and a ``spatial/unaccounted`` key is written whenever Keras logs ``compile_loss``
(it does not on Keras 3.8, which is warned about once per fit).

Reading the taps
----------------
Each tap records its last penalty in a non-trainable variable, the same mechanism
BatchNorm uses for moving statistics. That works on every backend and costs one
small read per tap per batch, where a forward pass to recompute the loss would
cost a full graph execution. The read happens every batch, not only at the log
interval, because the running mean below needs every batch's total.

One basis for the split
-----------------------
Keras reports ``logs["loss"]`` at batch end as the running mean over the epoch so
far, not the last batch's loss. ``task_loss`` and ``spatial/share`` therefore use
the running mean of the per-batch spatial totals (reset in ``on_epoch_begin``),
so both sides of the subtraction cover the same batches. The mean is
unweighted: it assumes equal batch sizes, and it assumes ``steps_per_execution=1``
(with more, the callback sees one batch end per execution and its mean covers
fewer batches than Keras'). The per-tap keys and ``spatial/total`` are still the
last batch's own values.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
"""

from typing import Any, Dict, List, Optional

import keras
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.callbacks._shared import walk_layers
from dl_techniques.utils.logger import logger
from dl_techniques.layers.regularization.spatial_smoothness import SpatialSmoothness

# ---------------------------------------------------------------------


class SpatialLossLogger(keras.callbacks.Callback):
    """Split each batch's reported loss into task and spatial components.

    Writes, per batch, on ``logs``:

    - ``spatial/<tap_name>`` -- that tap's penalty for the last batch
    - ``spatial/total`` -- the sum of ``alpha_k * SL_k`` for the last batch
    - ``spatial/unweighted`` -- the sum of the raw penalties, with no ``alpha``
    - ``task_loss`` -- ``loss - mean(spatial totals so far this epoch)``, on the
      same running-mean basis as Keras' ``loss``
    - ``spatial/share`` -- ``mean(spatial totals) / loss``, the fraction of the
      objective the topography term is currently responsible for (NaN when the
      reported loss is exactly 0.0)
    - ``spatial/accounted`` and ``spatial/unaccounted`` -- only when Keras logs
      ``compile_loss``

    Requires ``steps_per_execution=1`` and equal batch sizes (see the module
    docstring). Written keys reach only callbacks listed after this one.

    :param log_every: Emit every N batches. ``1`` is every batch.
    :type log_every: int
    :param prefix: Key prefix under ``logs``. Default ``'spatial'``.
    :type prefix: str
    :param verbose: Log a one-line summary at the configured cadence.
    :type verbose: int
    :raises ValueError: If ``log_every`` is below 1 -- naming the value.
    """

    def __init__(
        self,
        log_every: int = 100,
        prefix: str = "spatial",
        verbose: int = 0,
    ) -> None:
        super().__init__()

        if log_every < 1:
            raise ValueError(
                f"log_every must be >= 1, got {log_every}"
            )

        self.log_every = int(log_every)
        self.prefix = prefix
        self.verbose = int(verbose)
        self._batch = 0
        self._total_sum = 0.0
        self._total_count = 0
        self._warned_no_compile_loss = False
        #: The last set of keys this callback emitted. Keras hands the callback a
        #: ``logs`` dict it then keeps mutating, so a caller that wants to read
        #: the split back afterwards needs its own copy.
        self.last_recorded: Dict[str, float] = {}

    def set_model(self, model: keras.Model) -> None:
        """Discover the taps once, so a per-batch call is a dict lookup.

        Walked through the model's whole layer tree rather than by attribute name:
        the taps sit inside blocks, and a name-based walk has to know the block's
        internal layout and silently misses a tap at any new site. The walk is the
        shared ``walk_layers`` (``_flatten_layers`` with a public fallback), because
        ``Layer.submodules`` does not exist on Keras 3.8 and a bare ``hasattr`` on it
        would have shipped a discovery path that raises on every model.
        """
        super().set_model(model)
        self._taps: List[SpatialSmoothness] = [
            layer
            for layer in walk_layers(model)
            if isinstance(layer, SpatialSmoothness)
        ]
        if not self._taps:
            logger.warning(
                f"{self.__class__.__name__}: no SpatialSmoothness layer found in "
                f"the model; spatial keys will be absent from the logs."
            )
        if getattr(model, "aggregate_backbone_losses", None) is False:
            logger.warning(
                f"{self.__class__.__name__}: the model has "
                f"aggregate_backbone_losses=False, so the spatial penalty is "
                f"computed every step and NOT added to the training loss. "
                f"task_loss and spatial/share would then be wrong; construct "
                f"the head with aggregate_backbone_losses=True."
            )

    def on_train_begin(self, logs: Optional[Dict[str, float]] = None) -> None:
        """Restart the per-run state: cadence counter, last record, warning."""
        # DECISION plan-2026-11-11T131349-13b28b37/D-021: reset per run. Do not
        # keep ``_batch`` across fits: the ``log_every`` phase then depends on
        # how many batches earlier runs saw, so a rerun logs on different
        # batches than a fresh instance. Pinned by
        # test_state_resets_between_fits.py.
        self._batch = 0
        self.last_recorded = {}
        self._warned_no_compile_loss = False

    def on_epoch_begin(self, epoch, logs: Optional[Dict[str, float]] = None) -> None:
        """Restart the running mean: Keras' own loss mean restarts here too."""
        self._total_sum = 0.0
        self._total_count = 0

    def on_train_batch_end(self, batch, logs: Optional[Dict[str, float]] = None):
        """Record the taps' penalties into ``logs`` on the configured cadence."""
        if logs is None:
            return
        self._batch += 1
        if not self._taps:
            return

        # A list, not a dict by name: taps in different blocks share a leaf name.
        penalties = [
            float(ops.convert_to_numpy(tap.last_spatial_loss))
            for tap in self._taps
        ]
        total = sum(
            tap.alpha * penalty for tap, penalty in zip(self._taps, penalties)
        )
        # DECISION plan-2026-11-11T131349-13b28b37/D-019: `logs["loss"]` is a
        # running mean over the epoch, so the spatial total subtracted from it
        # must be a running mean over the same batches, read on EVERY batch (not
        # only the logged ones). Do not subtract this batch's `total`: that
        # mixes a last-batch value with an epoch mean (measured task_loss 18.49
        # against a true 34.53). Pinned by
        # TestSpatialLossLoggerBasis::test_task_loss_subtracts_the_running_mean_of_the_spatial_totals.
        self._total_sum += total
        self._total_count += 1

        if self._batch % self.log_every != 0:
            return
        if "loss" not in logs:
            return

        unweighted = 0.0
        for tap, penalty in zip(self._taps, penalties):
            logs[f"{self.prefix}/{tap.name}"] = penalty
            unweighted += penalty

        reported = float(logs["loss"])
        mean_total = self._total_sum / self._total_count
        share = mean_total / reported if reported != 0.0 else float("nan")
        logs[f"{self.prefix}/total"] = total
        logs[f"{self.prefix}/unweighted"] = unweighted
        logs["task_loss"] = reported - mean_total
        logs[f"{self.prefix}/share"] = share

        # Whether the spatial term actually REACHED the objective is not
        # readable from `loss` alone -- `loss - task` is `spatial_total` here by
        # construction, whatever the training loop did with it. The readable
        # signal is the gap between the reported loss and the compiled one, which
        # Keras records as `compile_loss` before auxiliary terms are folded in.
        # `unaccounted > 0` therefore means the backbone's own losses are being
        # computed and thrown away.
        if "compile_loss" in logs:
            accounted = reported - float(logs["compile_loss"])
            logs[f"{self.prefix}/accounted"] = accounted
            logs[f"{self.prefix}/unaccounted"] = mean_total - accounted
        elif not self._warned_no_compile_loss:
            self._warned_no_compile_loss = True
            logger.warning(
                f"{self.__class__.__name__}: the batch logs carry no "
                f"'compile_loss' key (Keras 3.8 does not write one), so "
                f"{self.prefix}/accounted and {self.prefix}/unaccounted cannot "
                f"be computed and a discarded add_loss is not detectable here."
            )

        self.last_recorded = dict(logs)

        if self.verbose:
            logger.info(
                f"[spatial] batch {self._batch}: loss={reported:.4f} "
                f"task={reported - mean_total:.4f} spatial={mean_total:.4f} "
                f"({100.0 * share:.1f}%) over {len(self._taps)} taps"
            )

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument."""
        # `keras.callbacks.Callback` has no `get_config` of its own, so this
        # starts from an empty dict rather than calling a base method that does
        # not exist.
        config: Dict[str, Any] = {}
        config.update({
            "log_every": self.log_every,
            "prefix": self.prefix,
            "verbose": self.verbose,
        })
        return config

# ------------------------------------------------------------------------------
