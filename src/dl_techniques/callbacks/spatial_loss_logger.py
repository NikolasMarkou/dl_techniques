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
discarded. In that case ``task_loss`` and ``total`` come out equal, and the
``spatial_share`` field is identically ``0.0``.

Reading the taps
----------------
Each tap records its last penalty in a non-trainable variable, the same mechanism
BatchNorm uses for moving statistics. That works on every backend and costs one
small read per tap per log interval, where a forward pass to recompute the loss
would cost a full graph execution.

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

from dl_techniques.utils.logger import logger
from dl_techniques.layers.regularization.spatial_smoothness import SpatialSmoothness

# ---------------------------------------------------------------------


def _walk_layers(model: keras.Model) -> List[keras.layers.Layer]:
    """Every layer in ``model``'s tree, itself included.

    ``Layer._flatten_layers`` is private but has been stable across the Keras 3
    series; the fallback is a depth-first walk over each layer's tracked
    sub-layers, so this keeps working if the private name moves.
    """
    flatten = getattr(model, "_flatten_layers", None)
    if callable(flatten):
        try:
            return list(flatten())
        except Exception:  # noqa: BLE001 - fall through to the manual walk
            logger.debug(
                "_flatten_layers failed; falling back to a manual sub-layer walk",
                exc_info=True,
            )

    found: List[keras.layers.Layer] = []
    seen = set()

    def visit(layer):
        if id(layer) in seen:
            return
        seen.add(id(layer))
        found.append(layer)
        for attribute in vars(layer).values():
            if isinstance(attribute, keras.layers.Layer):
                visit(attribute)
            elif isinstance(attribute, (list, tuple)):
                for entry in attribute:
                    if isinstance(entry, keras.layers.Layer):
                        visit(entry)

    visit(model)
    return found


class SpatialLossLogger(keras.callbacks.Callback):
    """Split each batch's reported loss into task and spatial components.

    Writes, per batch, on ``logs``:

    - ``spatial/<tap_name>`` -- that tap's penalty for the last batch
    - ``spatial/total`` -- the sum of ``alpha_k * SL_k``
    - ``spatial/unweighted`` -- the sum of the raw penalties, with no ``alpha``
    - ``task_loss`` -- ``loss - spatial_total``
    - ``spatial_share`` -- ``spatial_total / loss``, the fraction of the objective
      the topography term is currently responsible for

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
        #: The last set of keys this callback emitted. Keras hands the callback a
        #: ``logs`` dict it then keeps mutating, so a caller that wants to read
        #: the split back afterwards needs its own copy.
        self.last_recorded: Dict[str, float] = {}

    def set_model(self, model: keras.Model) -> None:
        """Discover the taps once, so a per-batch call is a dict lookup.

        Walked through the model's whole layer tree rather than by attribute name:
        the taps sit inside blocks, and a name-based walk has to know the block's
        internal layout and silently misses a tap at any new site. The walk uses
        ``_flatten_layers`` with a public fallback, because ``Layer.submodules``
        does not exist on Keras 3.8 and a bare ``hasattr`` on it would have shipped
        a discovery path that raises on every model.
        """
        super().set_model(model)
        self._taps: List[SpatialSmoothness] = [
            layer
            for layer in _walk_layers(model)
            if isinstance(layer, SpatialSmoothness)
        ]
        if not self._taps:
            logger.warning(
                f"{self.__class__.__name__}: no SpatialSmoothness layer found in "
                f"the model; spatial keys will be absent from the logs."
            )

    def on_train_batch_end(self, batch, logs: Optional[Dict[str, float]] = None):
        """Record the taps' penalties into ``logs`` on the configured cadence."""
        logs = logs if logs is not None else self.logs
        self._batch += 1
        if self._batch % self.log_every != 0:
            return
        if not self._taps or "loss" not in logs:
            return

        total = 0.0
        unweighted = 0.0
        for tap in self._taps:
            penalty = float(ops.convert_to_numpy(tap.last_spatial_loss))
            logs[f"{self.prefix}/{tap.name}"] = penalty
            total += tap.alpha * penalty
            unweighted += penalty

        reported = float(logs["loss"])
        logs[f"{self.prefix}/total"] = total
        logs[f"{self.prefix}/unweighted"] = unweighted
        logs["task_loss"] = reported - total
        logs[f"{self.prefix}/share"] = (
            total / reported if reported != 0.0 else float("nan")
        )

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
            logs[f"{self.prefix}/unaccounted"] = total - accounted

        self.last_recorded = dict(logs)

        if self.verbose:
            logger.info(
                f"[spatial] batch {self._batch}: loss={reported:.4f} "
                f"task={reported - total:.4f} spatial={total:.4f} "
                f"({100.0 * total / reported:.1f}%) over {len(self._taps)} taps"
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