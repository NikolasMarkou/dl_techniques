"""YOLOv12 — public API re-exports.

`feature_extractor` is the standalone backbone emitting the P3/P4/P5 feature
pyramid; `multitask` is the full detection / segmentation / classification model
built on it. The scale table lives on `YOLOv12FeatureExtractor.SCALE_CONFIGS`
(aliased as `MODEL_VARIANTS`) and `YOLOv12MultiTask` validates its `scale`
argument against that same table, so the two never drift.
"""
from .feature_extractor import (
    YOLOv12FeatureExtractor,
    create_yolov12_feature_extractor,
)
from .multitask import YOLOv12MultiTask, create_yolov12_multitask

__all__ = [
    "YOLOv12FeatureExtractor",
    "create_yolov12_feature_extractor",
    "YOLOv12MultiTask",
    "create_yolov12_multitask",
]
