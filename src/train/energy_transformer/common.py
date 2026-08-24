"""Shared plumbing for the two Energy Transformer trainers.

Deliberately SMALL (D-005). It holds only the three things both trainers genuinely share:

1. :func:`build_raw_image_dataset` — a raw-image ``tf.data`` pipeline (imagenette via tfds,
   cifar10 in-memory) with an optional per-sample ``element_map_fn`` hook. The MIM trainer
   passes ``make_masked_patch_map_fn(...)`` through that hook; the classifier passes nothing.
   It is no longer WRITTEN here: it was promoted to :mod:`train.common.datasets` (four
   packages consume it) and is RE-EXPORTED below, object identity intact.
2. :func:`build_optimizer` — the ``learning_rate_schedule_builder`` / ``optimizer_builder``
   block. It is no longer WRITTEN here either: it was promoted to
   :mod:`train.common.optimizer` (four packages consume it) and is RE-EXPORTED below,
   object identity intact.
3. :class:`EnergyTraceCallback` — the out-of-graph energy-descent probe.

There is NO shared config dataclass and NO shared ``train()`` orchestrator: each trainer owns
its own ``TrainingConfig``, ``parse_arguments()`` and ``config_from_args()`` so that an
argparse flag which never reaches the config is a LOCAL, greppable, testable defect rather
than an inherited one.
"""

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

# `build_raw_image_dataset` and its constants were PROMOTED to
# `train.common.datasets` (4 consuming packages); re-exported here so every
# existing `train.energy_transformer.common` import path keeps resolving TO THE
# SAME OBJECT -- see that module's D-006 anchor, and beit's identity test.
from train.common.datasets import (  # noqa: F401  (re-exported, D-006)
    DATASET_NUM_CLASSES,
    ElementMapFn,
    IMAGENETTE_NUM_CLASSES,
    IMAGENETTE_TFDS_NAME,
    SUPPORTED_DATASETS,
    build_raw_image_dataset,
)

# `build_optimizer` was PROMOTED the same way, to `train.common.optimizer` (4
# consuming packages, 9 call sites); re-exported here so every existing
# `train.energy_transformer.common` import path keeps resolving TO THE SAME
# OBJECT -- see that module's D-007 anchor, and beit's/dino's identity tests.
from train.common.optimizer import build_optimizer  # noqa: F401  (re-exported, D-007)

from dl_techniques.utils.logger import logger

# Same idiom as `train/graph_energy_transformer/common.py`: `__all__` is what marks
# the re-exported names above as USED (pyflakes does not honour `# noqa`).
__all__ = [
    "DATASET_NUM_CLASSES",
    "ElementMapFn",
    "IMAGENETTE_NUM_CLASSES",
    "IMAGENETTE_TFDS_NAME",
    "SUPPORTED_DATASETS",
    "build_raw_image_dataset",
    "build_optimizer",
    "EnergyTraceCallback",
]

# ---------------------------------------------------------------------
# energy trace probe
# ---------------------------------------------------------------------

class EnergyTraceCallback(keras.callbacks.Callback):
    """Log the ET block's energy descent trace, OUT OF GRAPH, once per epoch.

    Invariant I5 (H4): the ``(B, T+1)`` energy trace is float32 by design and must NEVER be
    consumed by a graph layer -- under ``mixed_float16`` a default-policy head would autocast
    an O(-1e5) trace down to fp16 and overflow it to inf/nan. The models therefore refuse a
    ``return_energy=True`` backbone outright (D-010), and the trace is read HERE instead: a
    separate PROBE backbone is rebuilt from the live backbone's config with
    ``return_energy=True``, its weights are re-synced from the live model, and it is called on
    one fixed validation batch. The training graph is untouched (``return_energy=True`` costs
    ~1.28x on the graph path; this costs one forward pass per epoch).

    **The weight re-sync is load-bearing and happens EVERY epoch.** A probe whose weights are
    not re-synced logs a stale, plausible, WRONG trace -- a silent lie that looks exactly like
    a real one. The guard is that the epoch-2 trace must DIFFER from the epoch-1 trace.

    Args:
        probe_inputs: One fixed batch of model inputs -- ``(image, input_mask)`` for the MIM
            model, ``image`` for the classifier. Held for the whole run so the traces are
            comparable across epochs.
        csv_path: Where to write the per-epoch trace.
        backbone_attr: Attribute on ``self.model`` holding the live backbone.
    """

    def __init__(
            self,
            probe_inputs: Any,
            csv_path: str,
            backbone_attr: str = "backbone",
    ) -> None:
        super().__init__()
        self.probe_inputs = probe_inputs
        self.csv_path = str(csv_path)
        self.backbone_attr = backbone_attr
        self._header_written = False

    def _build_probe(self, live_backbone: keras.Model) -> keras.Model:
        """Rebuild the backbone from its config with the energy readout enabled."""
        config = dict(live_backbone.get_config())
        config["return_energy"] = True
        probe = live_backbone.__class__.from_config(config)
        probe.build((None,) + tuple(live_backbone.input_shape_config))
        return probe

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        live_backbone = getattr(self.model, self.backbone_attr, None)
        if live_backbone is None:
            raise AttributeError(
                f"EnergyTraceCallback: model {type(self.model).__name__} has no "
                f"'{self.backbone_attr}' attribute; cannot build the energy probe."
            )

        probe = self._build_probe(live_backbone)
        # Re-synced EVERY epoch. Skipping this is the stale-probe failure mode.
        probe.set_weights(live_backbone.get_weights())

        _, energies = probe(self.probe_inputs, training=False)
        # Out of graph, immediately: numpy from here on. Nothing downstream ever sees a tensor.
        trace = np.asarray(keras.ops.convert_to_numpy(energies))
        # The trace's OWN dtype (float32 by the block's design, even under mixed_float16) --
        # logged before any cast, so the log cannot manufacture a dtype the tensor never had.
        trace_dtype = trace.dtype
        per_step = trace.astype(np.float64).mean(axis=0)  # (T+1,)

        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if not self._header_written:
                writer.writerow(["epoch"] + [f"step_{i}" for i in range(per_step.shape[0])])
                self._header_written = True
            writer.writerow([epoch] + [f"{v:.6f}" for v in per_step.tolist()])

        finite = bool(np.all(np.isfinite(per_step)))
        max_rise = float(np.max(np.diff(per_step))) if per_step.shape[0] > 1 else 0.0
        logger.info(
            f"energy trace (epoch {epoch}, dtype={trace_dtype}): "
            f"E_0={per_step[0]:.4f} -> E_T={per_step[-1]:.4f} "
            f"(delta={per_step[-1] - per_step[0]:+.4f}, finite={finite}, max_rise={max_rise:+.3e})"
        )
        if not finite:
            logger.warning("energy trace contains non-finite values -- the descent has diverged")

# ---------------------------------------------------------------------
