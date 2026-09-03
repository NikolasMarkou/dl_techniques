"""
`YOLOv12MultiTask` wraps `YOLOv12FeatureExtractor` with one head per enabled
task (detection, segmentation, classification) and returns them as a single
Keras functional model.

The three heads share one backbone and neck instead of running as separate
models, so the P3/P4/P5 features are computed once per forward pass and each
head reads off the parts it needs: the detection head regresses boxes with
DFL, the segmentation head decodes a per-pixel mask, and the classification
head pools to a class vector. Detection and segmentation can use different
class counts (for example 80 COCO classes pretraining into a 1-class crack
detector).

With more than one task enabled, `call` returns a dict keyed by task name
(`"detection"`, `"segmentation"`, `"classification"`); with exactly one task
enabled it returns that task's tensor directly, not a one-entry dict, so a
single-task model composes with a plain loss/metric list.

References:
    - Tian et al., 2025. YOLOv12: Attention-Centric Real-Time Object Detectors.
      (https://arxiv.org/abs/2502.12524) -- the detector this multi-task model
      shares a backbone with.
    - Bolya et al., 2019. YOLACT: Real-time Instance Segmentation. ICCV 2019.
      (https://arxiv.org/abs/1904.02689) -- the prototype-mask + per-instance
      coefficient formulation the segmentation head follows.
    - Caruana, 1997. Multitask Learning. Machine Learning 28(1) -- the shared-
      trunk argument this model is an instance of.
    - Kendall et al., 2018. Multi-Task Learning Using Uncertainty to Weigh
      Losses for Scene Geometry and Semantics. CVPR 2018.
      (https://arxiv.org/abs/1705.07115) -- the task-weighting problem the
      per-task loss weights here expose.
    - Lin et al., 2017. Focal Loss for Dense Object Detection. ICCV 2017.
      (https://arxiv.org/abs/1708.02002) and Milletari et al., 2016. V-Net
      (Dice loss). (https://arxiv.org/abs/1606.04797) -- the two terms of
      ``DiceFocalSegmentationLoss``.
"""

import keras
from typing import Optional, Tuple, Dict, Any, List, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.heads.vision.task_types import (
    VisionTaskType,
    TaskConfiguration,
    parse_task_list
)
from dl_techniques.layers.yolo12_heads import (
    YOLOv12DetectionHead,
    YOLOv12SegmentationHead,
    YOLOv12ClassificationHead
)
from .feature_extractor import YOLOv12FeatureExtractor
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.yolo12.multitask")
class YOLOv12MultiTask(keras.Model):
    """A shared YOLOv12 backbone with a detection, segmentation, and/or
    classification head, built as one Keras functional model.

    Architecture:

    .. code-block:: text

        input [B, H, W, 3]
              │
              ▼
        ┌─────────────────────┐
        │ YOLOv12FeatureExtr. │  shared backbone + neck
        └──────────┬──────────┘
                    │ [P3, P4, P5]
          ┌─────────┼──────────────────┐
          ▼          ▼                  ▼
        ┌──────┐  ┌───────────┐   ┌────────────┐
        │detect│  │segment     │   │classify    │  (each optional)
        │ head │  │ head       │   │ head       │
        └──┬───┘  └─────┬──────┘   └─────┬──────┘
           ▼             ▼                ▼
        boxes/cls     per-pixel        class
        (DFL)         mask             logits

        1 task enabled  -> that task's tensor
        >1 task enabled -> {"detection": ..., "segmentation": ..., ...}

    :param num_detection_classes: Number of detection classes. Falls back to `num_classes`.
    :type num_detection_classes: Optional[int]
    :param num_segmentation_classes: Number of segmentation classes. Falls back to `num_classes`.
    :type num_segmentation_classes: Optional[int]
    :param num_classification_classes: Number of classification classes. Falls back to `num_classes`.
    :type num_classification_classes: Optional[int]
    :param num_classes: Class count used for any task whose specific count is not given.
    :type num_classes: int
    :param input_shape: Input image shape ``(height, width, channels)``.
    :type input_shape: Tuple[int, int, int]
    :param scale: Model scale, one of 'n', 's', 'm', 'l', 'x'.
    :type scale: str
    :param reg_max: Maximum value for DFL regression in the detection head.
    :type reg_max: int
    :param task_config: Which tasks to enable — a `TaskConfiguration`, a list of
        `VisionTaskType`, a list of strings, a single `VisionTaskType`, or a single string.
    :type task_config: Union[TaskConfiguration, List[VisionTaskType], List[str], VisionTaskType, str]
    :param segmentation_filters: Decoder filter sizes for the segmentation head.
    :type segmentation_filters: Optional[List[int]]
    :param segmentation_dropout_rate: Dropout rate in the segmentation head.
    :type segmentation_dropout_rate: float
    :param classification_hidden_dims: MLP hidden dims for the classification head.
    :type classification_hidden_dims: Optional[List[int]]
    :param classification_dropout_rate: Dropout rate in the classification head.
    :type classification_dropout_rate: float
    :param kernel_initializer: Weight initializer for all layers.
    :type kernel_initializer: str
    :param name: Model name.
    :type name: Optional[str]

    Example:
        >>> # COCO pretraining
        >>> model = YOLOv12MultiTask(
        ...     num_detection_classes=80,
        ...     num_segmentation_classes=80,
        ...     task_config=[VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION]
        ... )
        >>>
        >>> # Crack detection fine-tuning
        >>> model = YOLOv12MultiTask(
        ...     num_detection_classes=1,
        ...     num_segmentation_classes=1,
        ...     task_config=[VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION]
        ... )
    """

    def __init__(
        self,
        num_detection_classes: Optional[int] = None,
        num_segmentation_classes: Optional[int] = None,
        num_classification_classes: Optional[int] = None,
        num_classes: int = 80,
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        scale: str = "n",
        reg_max: int = 16,
        task_config: Union[
            TaskConfiguration,
            List[VisionTaskType],
            List[str],
            VisionTaskType,
            str
        ] = VisionTaskType.DETECTION,
        segmentation_filters: Optional[List[int]] = None,
        segmentation_dropout_rate: float = 0.1,
        classification_hidden_dims: Optional[List[int]] = None,
        classification_dropout_rate: float = 0.3,
        kernel_initializer: str = "he_normal",
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the multi-task model and build its functional graph.

        :param num_detection_classes: Number of detection classes.
        :type num_detection_classes: Optional[int]
        :param num_segmentation_classes: Number of segmentation classes.
        :type num_segmentation_classes: Optional[int]
        :param num_classification_classes: Number of classification classes.
        :type num_classification_classes: Optional[int]
        :param num_classes: Fallback class count for any task not given its own.
        :type num_classes: int
        :param input_shape: Input image shape ``(height, width, channels)``.
        :type input_shape: Tuple[int, int, int]
        :param scale: Model scale, one of 'n', 's', 'm', 'l', 'x'.
        :type scale: str
        :param reg_max: Maximum value for DFL regression.
        :type reg_max: int
        :param task_config: Which tasks to enable.
        :type task_config: Union[TaskConfiguration, List[VisionTaskType], List[str], VisionTaskType, str]
        :param segmentation_filters: Decoder filter sizes for the segmentation head.
        :type segmentation_filters: Optional[List[int]]
        :param segmentation_dropout_rate: Dropout rate for the segmentation head.
        :type segmentation_dropout_rate: float
        :param classification_hidden_dims: MLP hidden dims for the classification head.
        :type classification_hidden_dims: Optional[List[int]]
        :param classification_dropout_rate: Dropout rate for the classification head.
        :type classification_dropout_rate: float
        :param kernel_initializer: Weight initializer.
        :type kernel_initializer: str
        :param name: Model name.
        :type name: Optional[str]
        :param kwargs: Extra keyword arguments passed to ``keras.Model``.
        """
        # Resolve mutable-default head configs (never share a list across instances).
        if segmentation_filters is None:
            segmentation_filters = [128, 64, 32]
        if classification_hidden_dims is None:
            classification_hidden_dims = [512, 256]

        # Validate inputs.
        valid_scales = set(YOLOv12FeatureExtractor.SCALE_CONFIGS.keys())
        if scale not in valid_scales:
            raise ValueError(
                f"scale must be one of {sorted(valid_scales)}, got {scale!r}"
            )
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        for cls_name, cls_val in (
            ("num_detection_classes", num_detection_classes),
            ("num_segmentation_classes", num_segmentation_classes),
            ("num_classification_classes", num_classification_classes),
        ):
            if cls_val is not None and cls_val <= 0:
                raise ValueError(f"{cls_name} must be positive, got {cls_val}")
        if reg_max <= 0:
            raise ValueError(f"reg_max must be positive, got {reg_max}")

        # Parse task configuration
        self.task_config = parse_task_list(task_config)

        # Configure class counts for each task
        self.num_detection_classes = num_detection_classes if num_detection_classes is not None else num_classes
        self.num_segmentation_classes = num_segmentation_classes if num_segmentation_classes is not None else num_classes
        self.num_classification_classes = num_classification_classes if num_classification_classes is not None else num_classes

        # Store configuration for serialization
        self.num_classes = num_classes  # Keep for backward compatibility
        self.input_shape_config = input_shape
        self.scale = scale
        self.reg_max = reg_max
        self.segmentation_filters = segmentation_filters
        self.segmentation_dropout_rate = segmentation_dropout_rate
        self.classification_hidden_dims = classification_hidden_dims
        self.classification_dropout_rate = classification_dropout_rate
        self.kernel_initializer = kernel_initializer

        if name is None:
            task_names = self.task_config.get_task_names()
            task_str = "_".join([name[:3] for name in task_names])
            name = f"yolov12_multitask_{scale}_{task_str}"

        inputs, outputs = self._build_functional_model()
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        enabled_tasks = self.task_config.get_task_names()
        logger.info(
            f"Created YOLOv12MultiTask-{scale} with enabled tasks: {enabled_tasks}"
        )
        logger.info(f"  Detection classes: {self.num_detection_classes}")
        logger.info(f"  Segmentation classes: {self.num_segmentation_classes}")
        logger.info(f"  Classification classes: {self.num_classification_classes}")

    def _build_functional_model(self) -> Tuple[keras.KerasTensor, Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]]:
        """Build the functional graph: shared backbone plus the enabled heads.

        :return: A tuple ``(inputs, outputs)``, where `outputs` is a single
            tensor for one enabled task or a dict keyed by task name for more.
        :rtype: Tuple[keras.KerasTensor, Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]]
        """
        inputs = keras.Input(shape=self.input_shape_config, name="input_images")

        feature_extractor = YOLOv12FeatureExtractor(
            input_shape=self.input_shape_config,
            scale=self.scale,
            kernel_initializer=self.kernel_initializer,
            name="shared_feature_extractor"
        )

        # Extract multi-scale features
        feature_maps = feature_extractor(inputs)

        # Build task-specific heads and collect outputs
        task_outputs = {}

        if self.task_config.has_detection():
            detection_head = YOLOv12DetectionHead(
                num_classes=self.num_detection_classes,
                reg_max=self.reg_max,
                kernel_initializer=self.kernel_initializer,
                name="detection_head"
            )
            detection_output = detection_head(feature_maps)
            task_outputs[VisionTaskType.DETECTION.value] = detection_output

        if self.task_config.has_segmentation():
            segmentation_head = YOLOv12SegmentationHead(
                num_classes=self.num_segmentation_classes,
                intermediate_filters=self.segmentation_filters,
                dropout_rate=self.segmentation_dropout_rate,
                kernel_initializer=self.kernel_initializer,
                name="segmentation_head"
            )
            segmentation_output = segmentation_head(feature_maps)
            task_outputs[VisionTaskType.SEGMENTATION.value] = segmentation_output

        if self.task_config.has_classification():
            classification_head = YOLOv12ClassificationHead(
                num_classes=self.num_classification_classes,
                hidden_dims=self.classification_hidden_dims,
                dropout_rate=self.classification_dropout_rate,
                kernel_initializer=self.kernel_initializer,
                name="classification_head"
            )
            classification_output = classification_head(feature_maps)
            task_outputs[VisionTaskType.CLASSIFICATION.value] = classification_output

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-048: one enabled task returns its
        # tensor directly, not a one-entry dict — a plain loss/metric list needs this. See decisions.md.
        if len(task_outputs) == 1:
            outputs = next(iter(task_outputs.values()))
        else:
            outputs = task_outputs

        return inputs, outputs

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to reconstruct this model."""
        config = super().get_config()
        config.update({
            "num_detection_classes": self.num_detection_classes,
            "num_segmentation_classes": self.num_segmentation_classes,
            "num_classification_classes": self.num_classification_classes,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape_config,
            "scale": self.scale,
            "reg_max": self.reg_max,
            "task_config": self.task_config.get_task_names(),
            "segmentation_filters": self.segmentation_filters,
            "segmentation_dropout_rate": self.segmentation_dropout_rate,
            "classification_hidden_dims": self.classification_hidden_dims,
            "classification_dropout_rate": self.classification_dropout_rate,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "YOLOv12MultiTask":
        """Reconstruct a model instance from a `get_config` result."""
        return cls(**config)

    def get_feature_extractor(self) -> YOLOv12FeatureExtractor:
        """Return the shared feature extractor embedded in this model's graph.

        :return: The `YOLOv12FeatureExtractor` instance, useful for analysis
            or transfer learning.
        :rtype: YOLOv12FeatureExtractor

        Note:
            If the extractor cannot be found as a layer (should not normally
            happen), a new one is constructed with matching config instead.
        """
        for layer in self.layers:
            if isinstance(layer, YOLOv12FeatureExtractor):
                return layer

        logger.warning("Feature extractor not found as layer, creating new instance")
        return YOLOv12FeatureExtractor(
            input_shape=self.input_shape_config,
            scale=self.scale,
            kernel_initializer=self.kernel_initializer
        )

    def get_enabled_tasks(self) -> List[VisionTaskType]:
        """Return the enabled tasks as `VisionTaskType` enums."""
        return self.task_config.get_enabled_tasks()

    def get_enabled_task_names(self) -> List[str]:
        """Return the enabled tasks as name strings."""
        return self.task_config.get_task_names()

    def has_task(self, task: VisionTaskType) -> bool:
        """Return whether `task` is enabled.

        :param task: Task to check.
        :type task: VisionTaskType
        :return: True if `task` is enabled.
        :rtype: bool
        """
        return task in self.task_config.tasks

    def extract_features(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """Run the shared backbone alone, without any task head.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Whether the call runs in training mode.
        :type training: Optional[bool]
        :return: Feature maps ``[P3, P4, P5]``.
        :rtype: List[keras.KerasTensor]
        """
        feature_extractor = self.get_feature_extractor()
        return feature_extractor(inputs, training=training)

    def get_class_counts(self) -> Dict[str, int]:
        """Return the class count used by each task, as a name-keyed dict."""
        return {
            'detection': self.num_detection_classes,
            'segmentation': self.num_segmentation_classes,
            'classification': self.num_classification_classes
        }

# ---------------------------------------------------------------------

def create_yolov12_multitask(
    num_detection_classes: Optional[int] = None,
    num_segmentation_classes: Optional[int] = None,
    num_classification_classes: Optional[int] = None,
    num_classes: int = 80,
    input_shape: Tuple[int, int, int] = (640, 640, 3),
    scale: str = "n",
    tasks: Union[
        List[VisionTaskType],
        List[str],
        TaskConfiguration,
        VisionTaskType,
        str
    ] = VisionTaskType.DETECTION,
    **kwargs
) -> YOLOv12MultiTask:
    """Create a YOLOv12 multi-task model.

    :param num_detection_classes: Number of detection classes.
    :type num_detection_classes: Optional[int]
    :param num_segmentation_classes: Number of segmentation classes.
    :type num_segmentation_classes: Optional[int]
    :param num_classification_classes: Number of classification classes.
    :type num_classification_classes: Optional[int]
    :param num_classes: Fallback class count for any task not given its own.
    :type num_classes: int
    :param input_shape: Input image shape.
    :type input_shape: Tuple[int, int, int]
    :param scale: Model scale.
    :type scale: str
    :param tasks: Tasks to enable.
    :type tasks: Union[List[VisionTaskType], List[str], TaskConfiguration, VisionTaskType, str]
    :param kwargs: Extra arguments passed to ``YOLOv12MultiTask``.
    :return: A configured multi-task model.
    :rtype: YOLOv12MultiTask

    Example:
        >>> # COCO pretraining - 80 classes for both detection and segmentation
        >>> model = create_yolov12_multitask(
        ...     num_detection_classes=80,
        ...     num_segmentation_classes=80,
        ...     tasks=[VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION],
        ...     scale="s"
        ... )
        >>>
        >>> # Crack detection - binary for both tasks
        >>> model = create_yolov12_multitask(
        ...     num_detection_classes=1,
        ...     num_segmentation_classes=1,
        ...     tasks=[VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION],
        ...     scale="s"
        ... )
        >>>
        >>> # Mixed - many detection classes, binary segmentation
        >>> model = create_yolov12_multitask(
        ...     num_detection_classes=80,
        ...     num_segmentation_classes=1,
        ...     tasks=[VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION],
        ...     scale="s"
        ... )
    """
    model = YOLOv12MultiTask(
        num_detection_classes=num_detection_classes,
        num_segmentation_classes=num_segmentation_classes,
        num_classification_classes=num_classification_classes,
        num_classes=num_classes,
        input_shape=input_shape,
        scale=scale,
        task_config=tasks,
        **kwargs
    )

    task_config = parse_task_list(tasks)
    task_names = task_config.get_task_names()
    class_counts = model.get_class_counts()

    logger.info(f"YOLOv12MultiTask-{scale} created with tasks: {task_names}")
    logger.info(f"Class counts: {class_counts}")
    return model

# ---------------------------------------------------------------------
