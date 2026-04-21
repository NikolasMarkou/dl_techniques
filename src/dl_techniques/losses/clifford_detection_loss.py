"""Stride-configurable YOLOv12 object detection loss for CliffordNet U-Net.

Subclasses :class:`YOLOv12ObjectDetectionLoss` to parametrize the feature-map
strides, which the parent hardcodes to ``[8, 16, 32]`` at construct time.
This is the minimal surgery needed to plug the YOLOv12 loss into a U-Net
with non-standard strides (e.g. ``[2, 4, 8]`` for 4-level CliffordNet
variants tapped at decoder levels ``[1, 2, 3]`` — see D-001 of the plan).

All other loss components — CIoU regression, Binary Focal Cross-Entropy,
Distribution Focal Loss, Task-Aligned Assigner, loss weights — are inherited
from the parent unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import keras
from keras import ops

from dl_techniques.losses.yolo12_multitask_loss import YOLOv12ObjectDetectionLoss


# DECISION D-005: Subclass rather than fork. The parent stays intact for
# YOLOv12-model users; we override only _make_anchors.


@keras.saving.register_keras_serializable()
class CliffordDetectionLoss(YOLOv12ObjectDetectionLoss):
    """YOLOv12 object-detection loss with configurable strides.

    :param strides: Feature-map strides for each detection scale.  Must be a
        sequence of 3 positive ints (YOLOv12 head expects exactly 3 scales).
        ``[2, 4, 8]`` matches CliffordNet's 4-level decoder tapped at
        ``[1, 2, 3]``; ``[8, 16, 32]`` reproduces stock YOLOv12.
    :param num_classes: Number of detection classes (80 for COCO).
    :param input_shape: Training image ``(H, W)``.  Anchor grid sizes derive
        from ``input_shape`` ÷ ``strides``.
    :param reg_max: DFL regression bins per box edge (default 16).
    :param box_weight: CIoU weight (default 7.5).
    :param cls_weight: Focal BCE weight (default 0.5).
    :param dfl_weight: DFL weight (default 1.5).
    :param assigner_alpha: TAL alpha (default 0.5).
    :param assigner_beta: TAL beta (default 6.0).
    :param name: Loss name.
    """

    def __init__(
        self,
        strides: Sequence[int] = (8, 16, 32),
        num_classes: int = 80,
        input_shape: Tuple[int, int] = (256, 256),
        reg_max: int = 16,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        assigner_alpha: float = 0.5,
        assigner_beta: float = 6.0,
        name: str = "clifford_detection_loss",
        **kwargs: Any,
    ) -> None:
        strides_tuple = tuple(int(s) for s in strides)
        if len(strides_tuple) != 3:
            raise ValueError(
                f"CliffordDetectionLoss expects exactly 3 strides "
                f"(YOLOv12 head constraint), got {len(strides_tuple)}"
            )
        if any(s <= 0 for s in strides_tuple):
            raise ValueError(f"strides must all be positive, got {strides_tuple}")
        # Must be set BEFORE super().__init__() because the parent calls
        # self._make_anchors() in its __init__ — our override below reads
        # self._strides_config.
        self._strides_config: Tuple[int, ...] = strides_tuple

        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            reg_max=reg_max,
            box_weight=box_weight,
            cls_weight=cls_weight,
            dfl_weight=dfl_weight,
            assigner_alpha=assigner_alpha,
            assigner_beta=assigner_beta,
            name=name,
            **kwargs,
        )

    def _make_anchors(
        self, grid_cell_offset: float = 0.5
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Generate anchors + strides for each feature level using self._strides_config.

        Directly parallels the parent implementation but replaces the
        hardcoded ``[8, 16, 32]`` with ``self._strides_config``.
        """
        H, W = self.input_shape
        anchor_points: List[keras.KerasTensor] = []
        stride_tensor: List[keras.KerasTensor] = []

        for stride in self._strides_config:
            h, w = H // stride, W // stride
            if h <= 0 or w <= 0:
                raise ValueError(
                    f"stride {stride} too large for input_shape {self.input_shape}: "
                    f"resulting feature map ({h}x{w}) is degenerate"
                )
            x_coords = ops.arange(w, dtype="float32") + grid_cell_offset
            y_coords = ops.arange(h, dtype="float32") + grid_cell_offset
            y_grid, x_grid = ops.meshgrid(y_coords, x_coords, indexing="ij")
            xy_grid = ops.stack([x_grid, y_grid], axis=-1)
            xy_grid = ops.reshape(xy_grid, (-1, 2))
            anchor_points.append(xy_grid)
            stride_tensor.append(ops.full((h * w, 1), stride, dtype="float32"))

        return (
            ops.concatenate(anchor_points, 0),
            ops.concatenate(stride_tensor, 0),
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["strides"] = list(self._strides_config)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordDetectionLoss":
        # The parent's __init__ hardcodes reduction="none" and passes it
        # explicitly to super().__init__(), so we must strip any "reduction"
        # key the base get_config emitted to avoid a duplicate-kwarg TypeError.
        config = dict(config)
        config.pop("reduction", None)
        return cls(**config)
