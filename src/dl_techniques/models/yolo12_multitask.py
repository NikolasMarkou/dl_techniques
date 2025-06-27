"""
YOLOv12 Multi-Task Model for Object Detection, Segmentation, and Classification.

This module extends the YOLOv12 architecture to support simultaneous object detection,
semantic segmentation, and image classification tasks. The model uses a shared backbone
and neck while providing task-specific heads for each task.

Architecture:
    - Shared Backbone: Feature extraction with ConvNeXt-style blocks
    - Shared Neck: PAN (Path Aggregation Network) with attention mechanisms
    - Detection Head: Multi-scale object detection (existing YOLOv12)
    - Segmentation Head: Progressive upsampling decoder with skip connections
    - Classification Head: Global pooling with attention mechanism

File: src/dl_techniques/models/yolo12_multitask.py
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, List, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.yolo12 import YOLOv12
from dl_techniques.layers.yolo12_heads import (
    YOLOv12DetectionHead,
    YOLOv12SegmentationHead,
    YOLOv12ClassificationHead
)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class YOLOv12MultiTask(keras.Model):
    """
    YOLOv12 Multi-Task Model for simultaneous object detection, segmentation, and classification.

    This model extends the standard YOLOv12 architecture with additional heads for segmentation
    and classification tasks while sharing the backbone and neck features.
    """

    def __init__(
            self,
            num_classes: int = 1,  # For crack detection (binary)
            input_shape: Tuple[int, int, int] = (256, 256, 3),  # Patch size
            scale: str = "n",
            reg_max: int = 16,

            # Task-specific configurations
            enable_detection: bool = True,
            enable_segmentation: bool = True,
            enable_classification: bool = True,

            # Segmentation head config
            segmentation_filters: List[int] = [128, 64, 32],
            segmentation_dropout: float = 0.1,

            # Classification head config
            classification_hidden_dims: List[int] = [512, 256],
            classification_dropout: float = 0.3,

            kernel_initializer: str = "he_normal",
            name: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize YOLOv12 multi-task model.

        Args:
            num_classes: Number of classes (1 for binary crack detection).
            input_shape: Input patch shape.
            scale: Model scale ('n', 's', 'm', 'l', 'x').
            reg_max: Maximum value for DFL regression.
            enable_detection: Whether to enable object detection head.
            enable_segmentation: Whether to enable segmentation head.
            enable_classification: Whether to enable classification head.
            segmentation_filters: Filter sizes for segmentation head.
            segmentation_dropout: Dropout rate for segmentation head.
            classification_hidden_dims: Hidden dimensions for classification head.
            classification_dropout: Dropout rate for classification head.
            kernel_initializer: Weight initializer.
            name: Model name.
        """
        if name is None:
            name = f"yolov12_multitask_{scale}"
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.input_shape_config = input_shape
        self.scale = scale
        self.reg_max = reg_max
        self.enable_detection = enable_detection
        self.enable_segmentation = enable_segmentation
        self.enable_classification = enable_classification
        self.kernel_initializer = kernel_initializer

        # Store configuration for serialization
        self.segmentation_filters = segmentation_filters
        self.segmentation_dropout = segmentation_dropout
        self.classification_hidden_dims = classification_hidden_dims
        self.classification_dropout = classification_dropout

        # Initialize backbone model (YOLOv12 without detection head)
        self.backbone = self._create_backbone()

        # Initialize task-specific heads
        self.detection_head = None
        self.segmentation_head = None
        self.classification_head = None

        if self.enable_detection:
            # Import here to avoid circular imports
            self.detection_head = YOLOv12DetectionHead(
                num_classes=1,  # Only crack class
                reg_max=self.reg_max,
                kernel_initializer=self.kernel_initializer,
                name="detection_head"
            )

        if self.enable_segmentation:
            self.segmentation_head = YOLOv12SegmentationHead(
                num_classes=1,  # Binary segmentation
                intermediate_filters=self.segmentation_filters,
                dropout_rate=self.segmentation_dropout,
                kernel_initializer=self.kernel_initializer,
                name="segmentation_head"
            )

        if self.enable_classification:
            self.classification_head = YOLOv12ClassificationHead(
                num_classes=1,  # Binary classification
                hidden_dims=self.classification_hidden_dims,
                dropout_rate=self.classification_dropout,
                kernel_initializer=self.kernel_initializer,
                name="classification_head"
            )

        logger.info(f"Created YOLOv12MultiTask-{scale} with tasks: "
                    f"detection={enable_detection}, segmentation={enable_segmentation}, "
                    f"classification={enable_classification}")

    def _create_backbone(self) -> keras.Model:
        """Create YOLOv12 backbone (without detection head)."""
        # Create full YOLOv12 model
        full_yolo = YOLOv12(
            num_classes=1,  # Dummy, we'll replace the head
            input_shape=self.input_shape_config,
            scale=self.scale,
            reg_max=self.reg_max,
            kernel_initializer=self.kernel_initializer
        )

        # Extract backbone and neck (everything except detection head)
        inputs = keras.Input(shape=self.input_shape_config)

        # Forward through backbone
        x = full_yolo.stem1(inputs)
        x = full_yolo.stem2(x)
        x = full_yolo.b1(x)

        p3 = full_yolo.down1(x)
        p3 = full_yolo.b2(p3)

        p4 = full_yolo.down2(p3)
        p4 = full_yolo.b3(p4)

        p5 = full_yolo.down3(p4)
        p5 = full_yolo.b4(p5)

        # Forward through neck
        x = full_yolo.up1(p5)
        x = ops.concatenate([x, p4], axis=-1)
        h1 = full_yolo.h1(x)

        x = full_yolo.up2(h1)
        x = ops.concatenate([x, p3], axis=-1)
        h2 = full_yolo.h2(x)

        x = full_yolo.neck_down1(h2)
        x = ops.concatenate([x, h1], axis=-1)
        h3 = full_yolo.h3(x)

        x = full_yolo.neck_down2(h3)
        x = ops.concatenate([x, p5], axis=-1)
        h4 = full_yolo.h4(x)

        # Return the three feature maps [P3, P4, P5]
        outputs = [h2, h3, h4]  # Different scales for multi-task heads

        backbone_model = keras.Model(inputs=inputs, outputs=outputs, name="yolov12_backbone")
        return backbone_model

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Forward pass through multi-task model.

        Args:
            inputs: Input tensor (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            If single task enabled: task output tensor.
            If multiple tasks enabled: dict with task outputs.
        """
        # Forward through shared backbone and neck
        feature_maps = self.backbone(inputs, training=training)  # [P3, P4, P5]

        outputs = {}

        # Forward through task-specific heads
        if self.enable_detection and self.detection_head is not None:
            detection_output = self.detection_head(feature_maps, training=training)
            outputs['detection'] = detection_output

        if self.enable_segmentation and self.segmentation_head is not None:
            segmentation_output = self.segmentation_head(feature_maps, training=training)
            outputs['segmentation'] = segmentation_output

        if self.enable_classification and self.classification_head is not None:
            classification_output = self.classification_head(feature_maps, training=training)
            outputs['classification'] = classification_output

        # Return single output if only one task is enabled
        if len(outputs) == 1:
            return list(outputs.values())[0]

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "input_shape": self.input_shape_config,
            "scale": self.scale,
            "reg_max": self.reg_max,
            "enable_detection": self.enable_detection,
            "enable_segmentation": self.enable_segmentation,
            "enable_classification": self.enable_classification,
            "segmentation_filters": self.segmentation_filters,
            "segmentation_dropout": self.segmentation_dropout,
            "classification_hidden_dims": self.classification_hidden_dims,
            "classification_dropout": self.classification_dropout,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "YOLOv12MultiTask":
        """Create model from configuration."""
        return cls(**config)


def create_yolov12_multitask(
        num_classes: int = 1,
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        scale: str = "n",
        tasks: List[str] = ["detection", "segmentation", "classification"],
        **kwargs
) -> YOLOv12MultiTask:
    """
    Create YOLOv12 multi-task model with specified configuration.

    Args:
        num_classes: Number of classes.
        input_shape: Input shape for patches.
        scale: Model scale.
        tasks: List of tasks to enable.
        **kwargs: Additional arguments.

    Returns:
        YOLOv12MultiTask model instance.
    """
    # Parse task configuration
    enable_detection = "detection" in tasks
    enable_segmentation = "segmentation" in tasks
    enable_classification = "classification" in tasks

    model = YOLOv12MultiTask(
        num_classes=num_classes,
        input_shape=input_shape,
        scale=scale,
        enable_detection=enable_detection,
        enable_segmentation=enable_segmentation,
        enable_classification=enable_classification,
        **kwargs
    )

    logger.info(f"YOLOv12MultiTask-{scale} model created successfully with tasks: {tasks}")
    return model

# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Test multi-task model creation
    model = create_yolov12_multitask(
        scale="n",
        input_shape=(256, 256, 3),
        tasks=["detection", "segmentation", "classification"]
    )

    # Build model
    sample_input = keras.ops.zeros((1, 256, 256, 3))
    outputs = model(sample_input, training=False)

    logger.info("Multi-task model outputs:")
    for task, output in outputs.items():
        logger.info(f"  {task}: {output.shape}")

    # Test single task model
    seg_model = create_yolov12_multitask(
        scale="n",
        input_shape=(256, 256, 3),
        tasks=["segmentation"]
    )

    seg_output = seg_model(sample_input, training=False)
    logger.info(f"Segmentation-only model output: {seg_output.shape}")

# ---------------------------------------------------------------------