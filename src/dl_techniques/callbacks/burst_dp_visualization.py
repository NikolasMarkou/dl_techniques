"""BurstDP training-time visualization callback.

Saves a fixed-val-batch comparison grid every N optimizer steps and / or
every M epochs. Each row shows one sample with six columns:

    [ref (corrupted)] [aux[0]] [recon pred] [recon target]
    [seg pred (colorized)] [seg target (colorized)]

The callback is intentionally model-agnostic about the rest of the BurstDP
hyperparameters — it just calls ``self.model(inputs, training=False)`` and
expects the standard ``{"recon", "segmentation"}`` output dict.

PNG files land under ``output_dir/viz/`` named ``step_NNNNNNN.png`` or
``epoch_NNNN.png``.

Set ``MPLBACKEND=Agg`` before launching training on a headless host.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger


def _import_matplotlib():
    import matplotlib.pyplot as plt
    return plt


def _build_palette(num_classes: int, seed: int = 0) -> np.ndarray:
    """Return an ``(num_classes, 3)`` uint8 palette.

    Uses matplotlib's ``tab20`` colormap and falls back to a hash-derived
    deterministic RGB for class indices beyond the cmap's length.
    """
    plt = _import_matplotlib()
    cmap = plt.colormaps["tab20"]
    base = (np.array([cmap(i % cmap.N) for i in range(num_classes)])[:, :3] * 255.0).astype(np.uint8)
    if num_classes <= cmap.N:
        return base
    # Beyond tab20 (20 entries) generate deterministic pseudo-random RGBs.
    rng = np.random.default_rng(seed)
    extra = rng.integers(0, 255, size=(num_classes - cmap.N, 3), dtype=np.uint8)
    base[cmap.N:] = extra
    return base


def _to_uint8_rgb(x: np.ndarray) -> np.ndarray:
    """Clip ``[0,1]`` float (any precision) to uint8 RGB image."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _colorize_seg(seg_idx: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """``(H, W)`` int class indices -> ``(H, W, 3)`` uint8."""
    seg_idx = np.clip(seg_idx, 0, palette.shape[0] - 1).astype(np.int64)
    return palette[seg_idx]


class BurstDPVisualizationCallback(keras.callbacks.Callback):
    """Periodic recon + segmentation visualization for BurstDP training.

    Args:
        val_dataset: Anything indexable as ``val_dataset[0] -> (inputs, labels)``
            where ``inputs`` has keys ``ref, aux, aux_mask`` and ``labels``
            has keys ``recon, segmentation`` (numpy arrays). A
            :class:`COCO2017BurstDPLoader` works directly.
        output_dir: Run directory. Creates ``output_dir/viz/``.
        every_steps: Save every N optimizer steps. ``0`` disables the step trigger.
        every_epochs: Save every M epochs. ``0`` disables the epoch trigger.
        num_samples: Rows in the grid. Capped to the fixed batch size.
        seed: Used to seed the palette generation only.
    """

    def __init__(
        self,
        val_dataset: Any,
        output_dir: str | Path,
        every_steps: int = 0,
        every_epochs: int = 1,
        num_samples: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "viz"
        os.makedirs(self.viz_dir, exist_ok=True)
        self.every_steps = int(every_steps)
        self.every_epochs = int(every_epochs)
        self.num_samples = int(num_samples)
        self._step = 0
        self._fixed_inputs: Optional[Dict[str, np.ndarray]] = None
        self._fixed_labels: Optional[Dict[str, np.ndarray]] = None
        self._palette: Optional[np.ndarray] = None
        self._seed = int(seed)
        self._disabled = False

        if self.every_steps <= 0 and self.every_epochs <= 0:
            logger.info("BurstDPVisualizationCallback: both triggers disabled — callback is a no-op.")
            self._disabled = True
            return

        try:
            inputs, labels = val_dataset[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"BurstDPVisualizationCallback: failed to fetch val batch ({exc}); disabling.")
            self._disabled = True
            return

        batch_size = int(inputs["ref"].shape[0])
        if batch_size == 0:
            logger.warning("BurstDPVisualizationCallback: empty val batch; disabling.")
            self._disabled = True
            return
        n = min(self.num_samples, batch_size)
        self.num_samples = n
        self._fixed_inputs = {
            "ref": np.asarray(inputs["ref"][:n], dtype=np.float32).copy(),
            "aux": np.asarray(inputs["aux"][:n], dtype=np.float32).copy(),
            "aux_mask": np.asarray(inputs["aux_mask"][:n], dtype=np.float32).copy(),
        }
        self._fixed_labels = {
            "recon": np.asarray(labels["recon"][:n], dtype=np.float32).copy(),
            "segmentation": np.asarray(labels["segmentation"][:n], dtype=np.int64).copy(),
        }

    # ------------------------------------------------------------------
    # Keras callback hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        if self._disabled or self.every_steps <= 0:
            self._step += 1
            return
        self._step += 1
        if self._step % self.every_steps == 0:
            self._save(tag=f"step_{self._step:07d}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if self._disabled or self.every_epochs <= 0:
            return
        if (epoch + 1) % self.every_epochs == 0:
            self._save(tag=f"epoch_{epoch + 1:04d}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _save(self, tag: str) -> None:
        if self._fixed_inputs is None or self._fixed_labels is None:
            return
        try:
            os.makedirs(self.viz_dir, exist_ok=True)
            outputs = self.model(self._fixed_inputs, training=False)
            recon_pred = np.asarray(keras.ops.convert_to_numpy(outputs["recon"]), dtype=np.float32)
            seg_logits = np.asarray(keras.ops.convert_to_numpy(outputs["segmentation"]), dtype=np.float32)
            seg_pred = np.argmax(seg_logits, axis=-1).astype(np.int64)
            num_classes = int(seg_logits.shape[-1])
            if self._palette is None or self._palette.shape[0] < num_classes:
                self._palette = _build_palette(num_classes, seed=self._seed)
            self._render(
                tag=tag,
                ref=self._fixed_inputs["ref"],
                aux0=self._fixed_inputs["aux"][:, 0],
                recon_pred=recon_pred,
                recon_target=self._fixed_labels["recon"],
                seg_pred=seg_pred,
                seg_target=self._fixed_labels["segmentation"],
            )
        except Exception as exc:  # noqa: BLE001 — render must never crash training
            logger.warning(f"BurstDPVisualizationCallback: render {tag} failed: {exc}")

    def _render(
        self,
        tag: str,
        ref: np.ndarray,
        aux0: np.ndarray,
        recon_pred: np.ndarray,
        recon_target: np.ndarray,
        seg_pred: np.ndarray,
        seg_target: np.ndarray,
    ) -> None:
        plt = _import_matplotlib()
        n = self.num_samples
        col_titles = ["ref", "aux[0]", "recon pred", "recon target", "seg pred", "seg target"]
        fig, axes = plt.subplots(n, 6, figsize=(2.4 * 6, 2.4 * n), squeeze=False)
        for i in range(n):
            tiles = [
                _to_uint8_rgb(ref[i]),
                _to_uint8_rgb(aux0[i]),
                _to_uint8_rgb(recon_pred[i]),
                _to_uint8_rgb(recon_target[i]),
                _colorize_seg(seg_pred[i], self._palette),
                _colorize_seg(seg_target[i], self._palette),
            ]
            for j, tile in enumerate(tiles):
                ax = axes[i, j]
                ax.imshow(tile, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(col_titles[j], fontsize=9)
        fig.suptitle(f"BurstDP @ {tag}", fontsize=11)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        out_path = self.viz_dir / f"{tag}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
