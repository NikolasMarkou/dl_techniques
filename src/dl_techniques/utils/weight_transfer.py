"""Generic layer-by-layer weight transfer for Keras 3 models.

This module exists because in Keras 3.8+ the API

    model.load_weights(path.keras, by_name=True, skip_mismatch=True)

raises ``ValueError: Invalid keyword arguments: {'by_name': True}`` — the
``by_name`` path is only valid for legacy ``.h5`` / ``.hdf5`` files.  For our
checkpoints, which are saved via ``model.save(path.keras)``, we need a manual
layer-by-layer transfer that matches weights by layer name, skips
user-specified prefixes (typically task heads), records shape mismatches, and
returns a structured audit report.

Typical use::

    from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

    depth_model = create_cliffordnet_depth(variant="base")
    depth_model.build((None, 256, 256, 3))
    report = load_weights_from_checkpoint(
        target=depth_model,
        ckpt_path="results/coco_pretrain/final_model.keras",
        skip_prefixes=("head_primary_", "head_aux_"),
    )
    logger.info(report.summary_string())
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger


# DECISION D-001: Layer-by-layer set_weights rather than load_weights(by_name=True).
# Keras 3.8's load_weights on a .keras file rejects the by_name kwarg outright.
# Three existing helpers in the repo (cliffordnet/model.py:413, bfunet.py:515,
# convnext_v2.py:400) carry this latent bug. Documented in plans/LESSONS.md.


@dataclass
class TransferReport:
    """Structured audit of a weight-transfer operation.

    :ivar loaded: names of target layers whose weights were overwritten from the source.
    :ivar skipped_by_prefix: target layer names matched by ``skip_prefixes`` and therefore not touched.
    :ivar shape_mismatch: ``(layer_name, target_shapes, source_shapes)`` tuples where names matched but weights did not.
    :ivar missing_in_source: target layer names absent from the source checkpoint (kept at target's init).
    :ivar unused_in_source: source layer names with weights that had no home in the target.
    """

    loaded: List[str] = field(default_factory=list)
    skipped_by_prefix: List[str] = field(default_factory=list)
    shape_mismatch: List[Tuple[str, List[Tuple[int, ...]], List[Tuple[int, ...]]]] = field(
        default_factory=list
    )
    missing_in_source: List[str] = field(default_factory=list)
    unused_in_source: List[str] = field(default_factory=list)

    @property
    def num_loaded(self) -> int:
        return len(self.loaded)

    @property
    def num_shape_mismatch(self) -> int:
        return len(self.shape_mismatch)

    def summary_string(self) -> str:
        """Human-readable single-block summary suitable for logging."""
        lines = [
            "TransferReport:",
            f"  loaded             : {len(self.loaded)} layers",
            f"  skipped_by_prefix  : {len(self.skipped_by_prefix)} layers",
            f"  shape_mismatch     : {len(self.shape_mismatch)} layers",
            f"  missing_in_source  : {len(self.missing_in_source)} layers",
            f"  unused_in_source   : {len(self.unused_in_source)} layers",
        ]
        if self.shape_mismatch:
            lines.append("  shape_mismatch details:")
            for name, tgt, src in self.shape_mismatch[:10]:
                lines.append(f"    - {name}: target={tgt} source={src}")
            if len(self.shape_mismatch) > 10:
                lines.append(f"    ... ({len(self.shape_mismatch) - 10} more)")
        return "\n".join(lines)


def _matches_any_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def _layer_weight_shapes(layer: keras.layers.Layer) -> List[Tuple[int, ...]]:
    return [tuple(w.shape) for w in layer.get_weights()]


def load_weights_from_checkpoint(
    target: keras.Model,
    ckpt_path: str,
    skip_prefixes: Sequence[str] = ("head_",),
    strict: bool = False,
    custom_objects: Optional[dict] = None,
) -> TransferReport:
    """Copy weights layer-by-layer from a saved ``.keras`` checkpoint into *target*.

    For every layer in the source model whose name does not start with any of
    ``skip_prefixes``, look up the same-named layer in *target* and call
    ``target_layer.set_weights(source_layer.get_weights())`` when shapes match.

    :param target: A *built* Keras model receiving weights.  Must already have
        been called once or had ``build(input_shape)`` invoked so its layers
        exist and are weight-shaped.
    :param ckpt_path: Path to a ``.keras`` model file (produced by
        ``keras.Model.save(path)``).  Non-``.keras`` paths raise ``ValueError``
        — use ``keras.models.load_model`` manually for legacy formats.
    :param skip_prefixes: Layer-name prefixes whose source weights should NOT
        be copied into *target*.  Default ``("head_",)`` matches
        :class:`CliffordNetUNet`'s head-layer naming (``head_<name>``,
        ``head_<name>_aux_<k>``).
    :param strict: If ``True``, any ``shape_mismatch`` during transfer raises
        :class:`ValueError`.  When ``False`` (default), mismatches are recorded
        in the report and skipped so other layers can still load.
    :param custom_objects: Forwarded to :func:`keras.models.load_model`.
    :returns: :class:`TransferReport` describing what happened.
    :raises FileNotFoundError: *ckpt_path* does not exist.
    :raises ValueError: *ckpt_path* does not end in ``.keras``; or all source
        layers were filtered / missing in target (no overlap).
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not str(ckpt_path).endswith(".keras"):
        raise ValueError(
            f"load_weights_from_checkpoint only supports .keras files; "
            f"got {ckpt_path!r}. Re-save the source via model.save('path.keras')."
        )
    if not target.built:
        raise ValueError(
            "target model must be built before weight transfer "
            "(call model.build(input_shape) or run a probe forward pass first)."
        )

    logger.info(f"Loading source model from {ckpt_path}")
    source = keras.models.load_model(ckpt_path, custom_objects=custom_objects)

    target_layers = {layer.name: layer for layer in target.layers}
    source_layers = {layer.name: layer for layer in source.layers}

    report = TransferReport()

    # Walk source layers — decide for each whether to transfer, skip, or record.
    for name, src_layer in source_layers.items():
        src_weights = src_layer.get_weights()
        if not src_weights:
            continue  # layer has no state (e.g. pure reshape)

        if _matches_any_prefix(name, skip_prefixes):
            report.skipped_by_prefix.append(name)
            continue

        tgt_layer = target_layers.get(name)
        if tgt_layer is None:
            report.unused_in_source.append(name)
            continue

        tgt_shapes = _layer_weight_shapes(tgt_layer)
        src_shapes = [tuple(w.shape) for w in src_weights]
        if tgt_shapes != src_shapes:
            report.shape_mismatch.append((name, tgt_shapes, src_shapes))
            if strict:
                raise ValueError(
                    f"Shape mismatch on layer {name!r}: target={tgt_shapes} "
                    f"source={src_shapes}"
                )
            continue

        tgt_layer.set_weights(src_weights)
        report.loaded.append(name)

    # Pass 2: detect target layers that have weights but were never loaded.
    loaded_set = set(report.loaded)
    skipped_set = set(report.skipped_by_prefix)
    for name, tgt_layer in target_layers.items():
        if not tgt_layer.get_weights():
            continue  # layer has no state
        if name in loaded_set or name in skipped_set:
            continue
        # Skip prefixes — target head layers are expected to be uninitialized from source
        if _matches_any_prefix(name, skip_prefixes):
            continue
        report.missing_in_source.append(name)

    if (
        not report.loaded
        and not report.skipped_by_prefix
        and not report.shape_mismatch
    ):
        # No overlap at all — almost certainly a user error (wrong checkpoint?).
        # Note: shape_mismatch still counts as overlap — same names, different shapes.
        raise ValueError(
            f"No overlapping layers between target and source checkpoint "
            f"{ckpt_path!r}. Target has {len(target_layers)} layers, source has "
            f"{len(source_layers)}."
        )

    logger.info(report.summary_string())
    return report
