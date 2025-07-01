"""
YOLOv12 Multi-Task Learning Model Implementation

This module implements a YOLOv12-based multi-task learning model that can simultaneously
perform object detection, instance segmentation, and image classification using a shared
feature extraction backbone. The implementation uses Keras Functional API with named
outputs for clean, dictionary-based results in multi-task scenarios.

FIXED VERSION: Now supports separate class counts for detection and segmentation tasks.
This allows COCO pretraining with 80-class segmentation and crack detection fine-tuning
with binary segmentation.

Architecture Overview
--------------------
The model follows a multitask learning architecture with three main components:

1. **Shared Feature Extractor**: A YOLOv12 backbone and neck that extracts multi-scale
   features (P3, P4, P5) from input images. This shared component enables efficient
   computation and knowledge transfer between tasks.

2. **Task-Specific Heads**: Specialized heads for each computer vision task:
   - Detection Head: Performs object detection with bounding box regression and
     classification using DFL (Distribution Focal Loss) regression
   - Segmentation Head: Generates pixel-level segmentation masks using a decoder
     architecture with configurable filter sizes and dropout
   - Classification Head: Performs global image classification using an MLP with
     configurable hidden dimensions and dropout

3. **Flexible Task Configuration**: Tasks can be enabled/disabled individually using
   TaskType enums, allowing for various combinations from single-task to full multi-task
   learning scenarios.

Key Features
-----------
- **Multi-Scale Feature Extraction**: Leverages YOLOv12's FPN-style neck for rich
  multi-scale feature representations
- **Named Outputs**: Returns clean dictionary-based outputs for multi-task scenarios
  (e.g., {"detection": tensor, "segmentation": tensor}) or single tensor for single tasks
- **Configurable Architecture**: Supports multiple YOLOv12 scales ('n', 's', 'm', 'l', 'x')
  and flexible head configurations
- **Separate Class Configuration**: Detection and segmentation can have different numbers
  of classes (e.g., 80 classes for COCO detection, 80 classes for COCO segmentation,
  but 1 class for crack detection, 1 class for crack segmentation)
- **Serialization Support**: Full Keras serialization compatibility with get_config()
  and from_config() methods
- **Factory Functions**: Pre-configured convenience functions for common task combinations

Usage Examples
--------------
COCO Pretraining (80 detection classes, 80 segmentation classes):
    >>> model = YOLOv12MultiTask(
    ...     num_detection_classes=80,
    ...     num_segmentation_classes=80,
    ...     task_config=[TaskType.DETECTION, TaskType.SEGMENTATION],
    ...     scale='s'
    ... )

Crack Detection Fine-tuning (1 detection class, 1 segmentation class):
    >>> model = YOLOv12MultiTask(
    ...     num_detection_classes=1,
    ...     num_segmentation_classes=1,
    ...     task_config=[TaskType.DETECTION, TaskType.SEGMENTATION],
    ...     scale='s'
    ... )

Mixed Configuration (80 detection classes, 1 segmentation class):
    >>> model = YOLOv12MultiTask(
    ...     num_detection_classes=80,
    ...     num_segmentation_classes=1,
    ...     task_config=[TaskType.DETECTION, TaskType.SEGMENTATION],
    ...     scale='s'
    ... )

Task Configuration
-----------------
Tasks can be specified in multiple flexible ways:
- Single TaskType enum: TaskType.DETECTION
- List of TaskType enums: [TaskType.DETECTION, TaskType.SEGMENTATION]
- String representations: "detection" or ["detection", "segmentation"]
- TaskConfiguration objects for advanced configuration
- Predefined CommonTaskConfigurations for common combinations

"""

import keras
from typing import Optional, Tuple, Dict, Any, List, Union

from dl_techniques.utils.logger import logger
from dl_techniques.utils.vision_task_types import (
    TaskType,
    TaskConfiguration,
    parse_task_list
)
from dl_techniques.models.yolo12_feature_extractor import YOLOv12FeatureExtractor
from dl_techniques.layers.yolo12_heads import (
    YOLOv12DetectionHead,
    YOLOv12SegmentationHead,
    YOLOv12ClassificationHead
)


@keras.saving.register_keras_serializable()
class YOLOv12MultiTask(keras.Model):
    """
    YOLOv12 Multi-Task Learning Model using Named Outputs (Functional API).

    This model combines a shared YOLOv12FeatureExtractor with multiple task-specific
    heads to perform simultaneous object detection, segmentation, and classification.
    Uses Keras Functional API with named outputs for clean dictionary-based results.

    FIXED VERSION: Now supports separate class counts for detection and segmentation.

    Args:
        num_detection_classes: Number of classes for detection task.
        num_segmentation_classes: Number of classes for segmentation task.
        num_classification_classes: Number of classes for classification task.
        num_classes: Backward compatibility - used for all tasks if specific counts not provided.
        input_shape: Input image shape (height, width, channels).
        scale: Model scale configuration ('n', 's', 'm', 'l', 'x').
        reg_max: Maximum value for DFL regression in detection.
        task_config: TaskConfiguration instance or list of TaskType enums.
        segmentation_filters: Filter sizes for segmentation decoder.
        segmentation_dropout: Dropout rate for segmentation head.
        classification_hidden_dims: Hidden dimensions for classification head.
        classification_dropout: Dropout rate for classification head.
        kernel_initializer: Weight initializer for all layers.
        name: Model name.

    Example:
        >>> # COCO pretraining
        >>> model = YOLOv12MultiTask(
        ...     num_detection_classes=80,
        ...     num_segmentation_classes=80,
        ...     task_config=[TaskType.DETECTION, TaskType.SEGMENTATION]
        ... )
        >>>
        >>> # Crack detection fine-tuning
        >>> model = YOLOv12MultiTask(
        ...     num_detection_classes=1,
        ...     num_segmentation_classes=1,
        ...     task_config=[TaskType.DETECTION, TaskType.SEGMENTATION]
        ... )
    """

    def __init__(
        self,
        # Class configuration - now separate for each task
        num_detection_classes: Optional[int] = None,
        num_segmentation_classes: Optional[int] = None,
        num_classification_classes: Optional[int] = None,
        num_classes: int = 80,  # Backward compatibility fallback
        # Model configuration
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        scale: str = "n",
        reg_max: int = 16,
        # Task configuration using enums
        task_config: Union[
            TaskConfiguration,
            List[TaskType],
            List[str],
            TaskType,
            str
        ] = TaskType.DETECTION,
        # Segmentation head configuration
        segmentation_filters: List[int] = [128, 64, 32],
        segmentation_dropout: float = 0.1,
        # Classification head configuration
        classification_hidden_dims: List[int] = [512, 256],
        classification_dropout: float = 0.3,
        # Common configuration
        kernel_initializer: str = "he_normal",
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize YOLOv12 multi-task model using Functional API.

        Args:
            num_detection_classes: Number of classes for detection task.
            num_segmentation_classes: Number of classes for segmentation task.
            num_classification_classes: Number of classes for classification task.
            num_classes: Fallback for backward compatibility.
            input_shape: Input image shape (height, width, channels).
            scale: Model scale ('n', 's', 'm', 'l', 'x').
            reg_max: Maximum value for DFL regression.
            task_config: Task configuration - can be TaskConfiguration, list of TaskType enums,
                        list of strings, single TaskType, or single string.
            segmentation_filters: Filter sizes for segmentation decoder.
            segmentation_dropout: Dropout rate for segmentation.
            classification_hidden_dims: Hidden dims for classification MLP.
            classification_dropout: Dropout rate for classification.
            kernel_initializer: Weight initializer.
            name: Model name.
            **kwargs: Additional keyword arguments.
        """
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
        self.segmentation_dropout = segmentation_dropout
        self.classification_hidden_dims = classification_hidden_dims
        self.classification_dropout = classification_dropout
        self.kernel_initializer = kernel_initializer

        if name is None:
            task_names = self.task_config.get_task_names()
            task_str = "_".join([name[:3] for name in task_names])  # Short names
            name = f"yolov12_multitask_{scale}_{task_str}"

        # Build the model using Functional API
        inputs, outputs = self._build_functional_model()

        # Initialize the Model with named outputs
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        enabled_tasks = self.task_config.get_task_names()
        logger.info(
            f"Created YOLOv12MultiTask-{scale} with enabled tasks: {enabled_tasks}"
        )
        logger.info(f"  Detection classes: {self.num_detection_classes}")
        logger.info(f"  Segmentation classes: {self.num_segmentation_classes}")
        logger.info(f"  Classification classes: {self.num_classification_classes}")

    def _build_functional_model(self) -> Tuple[keras.KerasTensor, Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]]:
        """
        Build the multi-task model using Functional API.

        Returns:
            Tuple of (inputs, outputs) where outputs is either a single tensor
            for single-task models or a dictionary for multi-task models.
        """
        # Define inputs
        inputs = keras.Input(shape=self.input_shape_config, name="input_images")

        # Shared feature extractor (backbone + neck)
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
                num_classes=self.num_detection_classes,  # Use detection-specific class count
                reg_max=self.reg_max,
                kernel_initializer=self.kernel_initializer,
                name="detection_head"
            )
            detection_output = detection_head(feature_maps)
            task_outputs[TaskType.DETECTION.value] = detection_output

        if self.task_config.has_segmentation():
            segmentation_head = YOLOv12SegmentationHead(
                num_classes=self.num_segmentation_classes,  # Use segmentation-specific class count
                intermediate_filters=self.segmentation_filters,
                dropout_rate=self.segmentation_dropout,
                kernel_initializer=self.kernel_initializer,
                name="segmentation_head"
            )
            segmentation_output = segmentation_head(feature_maps)
            task_outputs[TaskType.SEGMENTATION.value] = segmentation_output

        if self.task_config.has_classification():
            classification_head = YOLOv12ClassificationHead(
                num_classes=self.num_classification_classes,  # Use classification-specific class count
                hidden_dims=self.classification_hidden_dims,
                dropout_rate=self.classification_dropout,
                kernel_initializer=self.kernel_initializer,
                name="classification_head"
            )
            classification_output = classification_head(feature_maps)
            task_outputs[TaskType.CLASSIFICATION.value] = classification_output

        outputs = task_outputs

        return inputs, outputs

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            # Separate class counts
            "num_detection_classes": self.num_detection_classes,
            "num_segmentation_classes": self.num_segmentation_classes,
            "num_classification_classes": self.num_classification_classes,
            # Backward compatibility
            "num_classes": self.num_classes,
            # Other config
            "input_shape": self.input_shape_config,
            "scale": self.scale,
            "reg_max": self.reg_max,
            # Serialize task config as task names list for simplicity
            "task_config": self.task_config.get_task_names(),
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

    def get_feature_extractor(self) -> YOLOv12FeatureExtractor:
        """
        Get the shared feature extractor.

        Returns:
            The YOLOv12FeatureExtractor instance.

        Note:
            Since this is a Functional API model, the feature extractor
            is embedded within the model graph. This method helps access
            it for analysis or transfer learning.
        """
        # Find the feature extractor layer in the model
        for layer in self.layers:
            if isinstance(layer, YOLOv12FeatureExtractor):
                return layer

        # If not found as a layer, create a new one with same config
        logger.warning("Feature extractor not found as layer, creating new instance")
        return YOLOv12FeatureExtractor(
            input_shape=self.input_shape_config,
            scale=self.scale,
            kernel_initializer=self.kernel_initializer
        )

    def get_enabled_tasks(self) -> List[TaskType]:
        """
        Get list of enabled tasks.

        Returns:
            List of enabled TaskType enums.
        """
        return self.task_config.get_enabled_tasks()

    def get_enabled_task_names(self) -> List[str]:
        """
        Get list of enabled task names as strings.

        Returns:
            List of enabled task names.
        """
        return self.task_config.get_task_names()

    def has_task(self, task: TaskType) -> bool:
        """
        Check if a specific task is enabled.

        Args:
            task: TaskType enum to check.

        Returns:
            True if the task is enabled, False otherwise.
        """
        return task in self.task_config.tasks

    def extract_features(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """
        Extract shared features without applying task heads.

        Args:
            inputs: Input tensor.
            training: Whether in training mode.

        Returns:
            List of feature maps [P3, P4, P5].

        Note:
            This creates a separate feature extraction call since the
            feature extractor is embedded in the functional model.
        """
        feature_extractor = self.get_feature_extractor()
        return feature_extractor(inputs, training=training)

    def get_class_counts(self) -> Dict[str, int]:
        """
        Get the number of classes for each task.

        Returns:
            Dictionary mapping task names to class counts.
        """
        return {
            'detection': self.num_detection_classes,
            'segmentation': self.num_segmentation_classes,
            'classification': self.num_classification_classes
        }


def create_yolov12_multitask(
    num_detection_classes: Optional[int] = None,
    num_segmentation_classes: Optional[int] = None,
    num_classification_classes: Optional[int] = None,
    num_classes: int = 80,  # Backward compatibility
    input_shape: Tuple[int, int, int] = (640, 640, 3),
    scale: str = "n",
    tasks: Union[
        List[TaskType],
        List[str],
        TaskConfiguration,
        TaskType,
        str
    ] = TaskType.DETECTION,
    **kwargs
) -> YOLOv12MultiTask:
    """
    Create YOLOv12 multi-task model with specified tasks and class counts.

    Args:
        num_detection_classes: Number of detection classes.
        num_segmentation_classes: Number of segmentation classes.
        num_classification_classes: Number of classification classes.
        num_classes: Fallback class count for backward compatibility.
        input_shape: Input image shape.
        scale: Model scale.
        tasks: Tasks to enable - can be TaskConfiguration, list of TaskType enums,
               list of strings, single TaskType, or single string.
        **kwargs: Additional arguments.

    Returns:
        YOLOv12MultiTask model instance.

    Example:
        >>> # COCO pretraining - 80 classes for both detection and segmentation
        >>> model = create_yolov12_multitask(
        ...     num_detection_classes=80,
        ...     num_segmentation_classes=80,
        ...     tasks=[TaskType.DETECTION, TaskType.SEGMENTATION],
        ...     scale="s"
        ... )
        >>>
        >>> # Crack detection - binary for both tasks
        >>> model = create_yolov12_multitask(
        ...     num_detection_classes=1,
        ...     num_segmentation_classes=1,
        ...     tasks=[TaskType.DETECTION, TaskType.SEGMENTATION],
        ...     scale="s"
        ... )
        >>>
        >>> # Mixed - many detection classes, binary segmentation
        >>> model = create_yolov12_multitask(
        ...     num_detection_classes=80,
        ...     num_segmentation_classes=1,
        ...     tasks=[TaskType.DETECTION, TaskType.SEGMENTATION],
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