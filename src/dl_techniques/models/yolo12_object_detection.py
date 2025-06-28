"""
YOLOv12 Object Detector Implementation for Keras 3.

This module implements the YOLOv12 object detection model by combining the
YOLOv12FeatureExtractor with a detection head. This provides a clean separation
between feature extraction and task-specific processing.

The detector outputs bounding boxes, objectness scores, and class predictions
for multi-scale object detection.

File: src/dl_techniques/models/yolo12_object_detector.py
"""

import keras
from typing import Optional, Tuple, Dict, Any

from dl_techniques.utils.logger import logger
from dl_techniques.models.yolo12_feature_extractor import YOLOv12FeatureExtractor
from dl_techniques.layers.yolo12_heads import YOLOv12DetectionHead


@keras.saving.register_keras_serializable()
class YOLOv12ObjectDetector(keras.Model):
    """
    YOLOv12 Object Detection Model.

    This model combines a YOLOv12FeatureExtractor with a detection head to perform
    multi-scale object detection. The model outputs detection results for three
    different scales, making it suitable for detecting objects of various sizes.

    Args:
        num_classes: Number of object classes to detect.
        input_shape: Input image shape (height, width, channels).
        scale: Model scale configuration ('n', 's', 'm', 'l', 'x').
        reg_max: Maximum value for DFL (Distribution Focal Loss) regression.
        kernel_initializer: Weight initializer for all layers.
        name: Model name.

    Example:
        >>> detector = YOLOv12ObjectDetector(
        ...     num_classes=80,
        ...     input_shape=(640, 640, 3),
        ...     scale="n"
        ... )
        >>> detections = detector(images)
    """

    def __init__(
        self,
        num_classes: int = 80,
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        scale: str = "n",
        reg_max: int = 16,
        kernel_initializer: str = "he_normal",
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize YOLOv12 object detector.

        Args:
            num_classes: Number of object classes to detect.
            input_shape: Input image shape (height, width, channels).
            scale: Model scale ('n', 's', 'm', 'l', 'x').
            reg_max: Maximum value for DFL regression.
            kernel_initializer: Weight initializer.
            name: Model name.
            **kwargs: Additional keyword arguments.
        """
        if name is None:
            name = f"yolov12_object_detector_{scale}"
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.input_shape_config = input_shape
        self.scale = scale
        self.reg_max = reg_max
        self.kernel_initializer = kernel_initializer

        # Store build state for serialization
        self._build_input_shape = None
        self._layers_built = False

        logger.info(f"Created YOLOv12ObjectDetector-{scale} with {num_classes} classes")

    def build(self, input_shape):
        """Build the detector layers."""
        if self._layers_built:
            return

        self._build_input_shape = input_shape
        self._build_layers()
        self._layers_built = True
        super().build(input_shape)

    def _build_layers(self) -> None:
        """Initialize feature extractor and detection head."""
        # Feature extractor (backbone + neck)
        self.feature_extractor = YOLOv12FeatureExtractor(
            input_shape=self.input_shape_config,
            scale=self.scale,
            kernel_initializer=self.kernel_initializer,
            name="feature_extractor"
        )

        # Detection head
        self.detection_head = YOLOv12DetectionHead(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            kernel_initializer=self.kernel_initializer,
            name="detection_head"
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through object detector.

        Args:
            inputs: Input tensor (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Detection outputs from the detection head. The exact format depends
            on the YOLOv12DetectionHead implementation.
        """
        # Extract multi-scale features
        feature_maps = self.feature_extractor(inputs, training=training)

        # Apply detection head
        detections = self.detection_head(feature_maps, training=training)

        return detections

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute output shape for detection results.

        Args:
            input_shape: Input tensor shape.

        Returns:
            Output shape tuple.
        """
        # Get feature map shapes from extractor
        feature_shapes = self.feature_extractor.compute_output_shape(input_shape)

        # Detection head processes these features - exact output shape depends on head implementation
        # This is a placeholder - you'll need to implement based on your detection head
        batch_size = input_shape[0]

        # Typical YOLO output shape calculation would go here
        # For now, return a placeholder
        return (batch_size, -1, self.num_classes + 5)  # [batch, detections, classes + bbox + obj]

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "input_shape": self.input_shape_config,
            "scale": self.scale,
            "reg_max": self.reg_max,
            "kernel_initializer": self.kernel_initializer,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build model from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "YOLOv12ObjectDetector":
        """Create model from configuration."""
        return cls(**config)

    def get_feature_extractor(self) -> YOLOv12FeatureExtractor:
        """
        Get the feature extractor component.

        Returns:
            The YOLOv12FeatureExtractor instance used by this detector.

        Example:
            >>> detector = YOLOv12ObjectDetector(num_classes=80)
            >>> extractor = detector.get_feature_extractor()
            >>> features = extractor(images)  # Direct feature extraction
        """
        return self.feature_extractor

    def get_detection_head(self) -> YOLOv12DetectionHead:
        """
        Get the detection head component.

        Returns:
            The YOLOv12DetectionHead instance used by this detector.

        Example:
            >>> detector = YOLOv12ObjectDetector(num_classes=80)
            >>> head = detector.get_detection_head()
        """
        return self.detection_head

    def extract_features(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> list:
        """
        Extract features without applying detection head.

        This method is useful for feature visualization, transfer learning,
        or using the features for other downstream tasks.

        Args:
            inputs: Input tensor (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            List of feature maps [P3, P4, P5] from the feature extractor.

        Example:
            >>> detector = YOLOv12ObjectDetector(num_classes=80)
            >>> features = detector.extract_features(images)
            >>> print([f.shape for f in features])  # Feature map shapes
        """
        return self.feature_extractor(inputs, training=training)


def create_yolov12_object_detector(
    num_classes: int = 80,
    input_shape: Tuple[int, int, int] = (640, 640, 3),
    scale: str = "n",
    **kwargs
) -> YOLOv12ObjectDetector:
    """
    Create a YOLOv12 object detector with specified configuration.

    Args:
        num_classes: Number of object classes to detect.
        input_shape: Input image shape.
        scale: Model scale.
        **kwargs: Additional arguments for YOLOv12ObjectDetector.

    Returns:
        YOLOv12ObjectDetector model instance.

    Example:
        >>> detector = create_yolov12_object_detector(
        ...     num_classes=20,
        ...     input_shape=(416, 416, 3),
        ...     scale="s"
        ... )
        >>> detections = detector(images)
    """
    detector = YOLOv12ObjectDetector(
        num_classes=num_classes,
        input_shape=input_shape,
        scale=scale,
        **kwargs
    )

    logger.info(f"YOLOv12ObjectDetector-{scale} created successfully")
    return detector


# Convenience functions for different scales
def create_yolov12_nano(num_classes: int = 80, **kwargs) -> YOLOv12ObjectDetector:
    """Create YOLOv12-nano object detector."""
    return create_yolov12_object_detector(num_classes=num_classes, scale="n", **kwargs)


def create_yolov12_small(num_classes: int = 80, **kwargs) -> YOLOv12ObjectDetector:
    """Create YOLOv12-small object detector."""
    return create_yolov12_object_detector(num_classes=num_classes, scale="s", **kwargs)


def create_yolov12_medium(num_classes: int = 80, **kwargs) -> YOLOv12ObjectDetector:
    """Create YOLOv12-medium object detector."""
    return create_yolov12_object_detector(num_classes=num_classes, scale="m", **kwargs)


def create_yolov12_large(num_classes: int = 80, **kwargs) -> YOLOv12ObjectDetector:
    """Create YOLOv12-large object detector."""
    return create_yolov12_object_detector(num_classes=num_classes, scale="l", **kwargs)


def create_yolov12_xlarge(num_classes: int = 80, **kwargs) -> YOLOv12ObjectDetector:
    """Create YOLOv12-extra-large object detector."""
    return create_yolov12_object_detector(num_classes=num_classes, scale="x", **kwargs)