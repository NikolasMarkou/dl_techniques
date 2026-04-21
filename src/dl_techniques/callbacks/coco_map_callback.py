"""COCO mAP evaluation callback for CliffordNet multi-task detection.

On every ``frequency`` validation epoch, runs the model over a local COCO
val loader, decodes YOLOv12 predictions via :mod:`dl_techniques.utils.yolo_decode`,
applies per-class NMS, remaps contiguous class indices back to COCO's
non-contiguous category IDs, and scores via ``pycocotools.cocoeval.COCOeval``.
Writes two metrics into the epoch ``logs`` dict so CSVLogger / TensorBoard /
TrainingCurvesCallback all pick them up:

- ``val_map50``    — mAP at IoU 0.50 (COCOeval stats[1])
- ``val_map5095``  — mean mAP over IoU 0.50..0.95 step 0.05 (COCOeval stats[0])

pycocotools is required (installed during an earlier plan).  The callback is
safe to use even when the model emits zero detections — pycocotools gracefully
produces mAP=0 and we catch its "No annotation to evaluate" edge case.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger
from dl_techniques.utils.yolo_decode import (
    decode_predictions,
    make_anchors_np,
    nms_per_class,
)


try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pycocotools is required for COCOMAPCallback. Install with "
        "`.venv/bin/pip install pycocotools`."
    ) from e


class COCOMAPCallback(keras.callbacks.Callback):
    """Compute COCO mAP at each validation epoch.

    :param val_loader: A :class:`COCO2017MultiTaskLoader` (or compatible) with
        ``emit_boxes=True``.  Must expose ``coco`` (a pycocotools
        :class:`pycocotools.coco.COCO`), ``image_ids`` (ordered list), and
        ``idx_to_cat_id`` (``Dict[int, int]`` mapping model-class-index →
        COCO category id).  ``config.shuffle`` should be ``False`` on val.
    :param image_size: Training-time square image size ``(H == W)``.  Anchors
        are generated at this size; strides divide it evenly.
    :param num_classes: Number of detection classes emitted by the model.
    :param reg_max: DFL regression bins per edge.
    :param strides: Feature-map strides used by the detection head.
    :param detection_head_key: Name of the dict key in the model output that
        holds detection logits.  Defaults to ``"detection"``.
    :param score_threshold: Candidate cutoff before NMS.  Default 0.01.
    :param iou_threshold: NMS IoU threshold.  Default 0.45.
    :param max_per_class: Max detections per class (pre-merge).  Default 100.
    :param max_total: Max detections per image (post-merge).  Default 300.
    :param frequency: Run every N epochs.  Default 1.
    :param max_eval_batches: Optional cap on the number of val batches
        evaluated per epoch (for quick iteration / smoke runs).  ``None`` =
        full val set.
    :param suppress_stdout: If True (default), swallow pycocotools'
        verbose ``summarize()`` prints.
    """

    def __init__(
        self,
        val_loader: Any,
        image_size: int,
        num_classes: int,
        reg_max: int,
        strides: Sequence[int],
        detection_head_key: str = "detection",
        score_threshold: float = 0.01,
        iou_threshold: float = 0.45,
        max_per_class: int = 100,
        max_total: int = 300,
        frequency: int = 1,
        max_eval_batches: Optional[int] = None,
        suppress_stdout: bool = True,
    ) -> None:
        super().__init__()
        self.val_loader = val_loader
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.strides = tuple(int(s) for s in strides)
        self.detection_head_key = detection_head_key
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_per_class = int(max_per_class)
        self.max_total = int(max_total)
        self.frequency = max(1, int(frequency))
        self.max_eval_batches = max_eval_batches
        self.suppress_stdout = suppress_stdout

        # Validate duck-typed loader
        for attr in ("coco", "image_ids", "idx_to_cat_id", "config"):
            if not hasattr(val_loader, attr):
                raise AttributeError(
                    f"val_loader is missing required attribute {attr!r}"
                )
        self.coco: COCO = val_loader.coco
        self.image_ids: List[int] = list(val_loader.image_ids)
        self.idx_to_cat_id: Dict[int, int] = dict(val_loader.idx_to_cat_id)

        # Pre-compute anchors once.
        self.anchors_np, self.strides_np = make_anchors_np(
            (self.image_size, self.image_size), self.strides,
        )

        # Cache original image (H, W) lookup.
        self._img_hw: Dict[int, Tuple[int, int]] = {}
        for img_id in self.image_ids:
            info = self.coco.loadImgs(img_id)[0]
            self._img_hw[img_id] = (int(info["height"]), int(info["width"]))

        logger.info(
            f"COCOMAPCallback: {len(self.image_ids)} val images, "
            f"num_classes={num_classes}, reg_max={reg_max}, strides={self.strides}, "
            f"frequency={self.frequency}"
        )

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is None:
            logs = {}
        if (epoch + 1) % self.frequency != 0:
            return
        try:
            ap50, ap5095 = self._run_eval()
        except Exception as e:
            logger.warning(f"COCOMAPCallback: eval failed: {e}")
            ap50, ap5095 = 0.0, 0.0
        logs["val_map50"] = float(ap50)
        logs["val_map5095"] = float(ap5095)
        logger.info(
            f"Epoch {epoch + 1} — val_map50={ap50:.4f}  val_map5095={ap5095:.4f}"
        )

    # ------------------------------------------------------------------
    # Eval pipeline
    # ------------------------------------------------------------------

    def _run_eval(self) -> Tuple[float, float]:
        """Full pipeline: forward → decode → NMS → pycocotools COCOeval.

        :returns: ``(map_at_50, map_at_50_95)``.
        """
        results: List[Dict[str, Any]] = []
        n_batches = len(self.val_loader)
        if self.max_eval_batches is not None:
            n_batches = min(n_batches, int(self.max_eval_batches))

        evaluated_ids: List[int] = []
        cursor = 0
        for bi in range(n_batches):
            x, _ = self.val_loader[bi]
            preds = self.model(x, training=False)
            det = preds.get(self.detection_head_key)
            if det is None:
                logger.warning(
                    f"COCOMAPCallback: model output missing "
                    f"'{self.detection_head_key}' key — aborting eval."
                )
                return 0.0, 0.0
            det_np = np.asarray(keras.ops.convert_to_numpy(det))

            per_image = decode_predictions(
                det_np,
                self.anchors_np,
                self.strides_np,
                reg_max=self.reg_max,
                num_classes=self.num_classes,
                score_threshold=self.score_threshold,
            )

            for i, (boxes_px, scores, classes) in enumerate(per_image):
                image_id = self.image_ids[cursor + i]
                evaluated_ids.append(int(image_id))
                # Per-class NMS (operates on training-image pixel coords)
                kept_b, kept_s, kept_c = nms_per_class(
                    boxes_px, scores, classes,
                    iou_threshold=self.iou_threshold,
                    max_per_class=self.max_per_class,
                    max_total=self.max_total,
                )
                if kept_b.shape[0] == 0:
                    continue
                # Convert from training-image pixel coords (square, image_size)
                # back to original-image pixel coords.
                orig_h, orig_w = self._img_hw[image_id]
                sx = orig_w / self.image_size
                sy = orig_h / self.image_size
                x1 = kept_b[:, 0] * sx
                y1 = kept_b[:, 1] * sy
                x2 = kept_b[:, 2] * sx
                y2 = kept_b[:, 3] * sy
                w = np.clip(x2 - x1, 0, None)
                h = np.clip(y2 - y1, 0, None)
                # Clip to image bounds to keep pycocotools happy.
                x1 = np.clip(x1, 0, orig_w)
                y1 = np.clip(y1, 0, orig_h)
                w = np.clip(w, 0, orig_w - x1)
                h = np.clip(h, 0, orig_h - y1)

                for j in range(kept_b.shape[0]):
                    cat_id = self.idx_to_cat_id.get(int(kept_c[j]))
                    if cat_id is None:
                        continue
                    if w[j] <= 0 or h[j] <= 0:
                        continue
                    results.append({
                        "image_id": int(image_id),
                        "category_id": int(cat_id),
                        "bbox": [
                            float(x1[j]),
                            float(y1[j]),
                            float(w[j]),
                            float(h[j]),
                        ],
                        "score": float(kept_s[j]),
                    })
            cursor += x.shape[0]

        return self._score_results(results, evaluated_ids)

    def _score_results(
        self,
        results: List[Dict[str, Any]],
        evaluated_ids: List[int],
    ) -> Tuple[float, float]:
        """Run pycocotools COCOeval on a list of detection dicts.

        pycocotools complains when passed an empty detection list; we return
        ``(0, 0)`` in that case.
        """
        if not results:
            return 0.0, 0.0

        stream = io.StringIO() if self.suppress_stdout else None
        ctx = (
            contextlib.redirect_stdout(stream)
            if self.suppress_stdout
            else contextlib.nullcontext()
        )
        with ctx:
            try:
                coco_dt = self.coco.loadRes(results)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"COCOMAPCallback: loadRes failed: {e}")
                return 0.0, 0.0

            coco_eval = COCOeval(self.coco, coco_dt, "bbox")
            # Restrict evaluation to the image ids we ran on — important when
            # max_eval_batches truncates val set.
            if evaluated_ids:
                coco_eval.params.imgIds = sorted(set(evaluated_ids))
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        # COCOeval.stats layout (per pycocotools):
        #   stats[0] = mAP @ IoU 0.50:0.95
        #   stats[1] = mAP @ IoU 0.50
        ap5095 = float(coco_eval.stats[0]) if coco_eval.stats is not None else 0.0
        ap50 = float(coco_eval.stats[1]) if coco_eval.stats is not None else 0.0
        # Negative values (-1) indicate "no GT for this metric" — clamp to 0 for logging.
        if np.isnan(ap50) or ap50 < 0:
            ap50 = 0.0
        if np.isnan(ap5095) or ap5095 < 0:
            ap5095 = 0.0
        return ap50, ap5095
