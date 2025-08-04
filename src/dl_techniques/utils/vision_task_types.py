from enum import Enum, unique
from typing import List, Set, Dict, Optional


# ---------------------------------------------------------------------

@unique
class TaskType(Enum):
    """
    Enumeration of supported computer vision tasks for multi-task models.

    Each task represents a different computer vision capability that can be
    enabled in the multi-task architecture. Tasks are organized into categories
    for better understanding and compatibility checking.

    Core Detection & Segmentation:
        DETECTION: Object detection with bounding box regression and classification.
        SEGMENTATION: Pixel-level semantic segmentation.
        INSTANCE_SEGMENTATION: Segmenting individual object instances.
        PANOPTIC_SEGMENTATION: Combined semantic and instance segmentation.
        CLASSIFICATION: Global image-level classification.

    Geometric Understanding:
        DEPTH_ESTIMATION: Predicting depth maps from RGB images.
        SURFACE_NORMALS: Estimating surface normal vectors.
        STEREO_MATCHING: Depth estimation from stereo image pairs.

    Motion & Temporal:
        OPTICAL_FLOW: Estimating motion between consecutive frames.
        MOTION_SEGMENTATION: Segmenting moving objects in video.

    Structural Analysis:
        POSE_ESTIMATION: Estimating object or human poses.
        KEYPOINT_DETECTION: Detecting and localizing keypoints.
        EDGE_DETECTION: Detecting edges and boundaries.
        LINE_DETECTION: Detecting line segments and geometric structures.

    Attention & Saliency:
        SALIENCY_DETECTION: Identifying salient regions in images.
        ATTENTION_PREDICTION: Predicting human visual attention maps.

    Image Enhancement & Restoration:
        DENOISING: Removing noise from images.
        SUPER_RESOLUTION: Upscaling images with enhanced detail.
        INPAINTING: Filling in missing or corrupted parts of images.
        DEHAZE: Removing haze, fog, or atmospheric effects.
        SHADOW_REMOVAL: Detecting and removing shadows.
        REFLECTION_REMOVAL: Removing reflections from images.

    Color & Style:
        COLORIZATION: Adding color to grayscale images.
        STYLE_TRANSFER: Transferring artistic styles between images.
        WHITE_BALANCE: Correcting color temperature and white balance.

    Advanced Segmentation:
        MATTING: Extracting foreground objects with soft boundaries.
        HAIR_SEGMENTATION: Specialized segmentation for hair regions.
        SKY_SEGMENTATION: Specialized segmentation for sky regions.

    Medical & Scientific:
        MEDICAL_SEGMENTATION: Medical image segmentation tasks.
        CELL_COUNTING: Counting cells or objects in microscopy images.

    Document & Text:
        TEXT_DETECTION: Detecting text regions in natural images.
        DOCUMENT_LAYOUT: Analyzing document structure and layout.

    3D Understanding:
        DEPTH_COMPLETION: Completing sparse depth maps.
        SURFACE_RECONSTRUCTION: Reconstructing 3D surfaces from images.
        CAMERA_POSE: Estimating camera pose and orientation.

    Quality Assessment:
        IMAGE_QUALITY: Assessing image quality metrics.
        AESTHETIC_SCORING: Predicting aesthetic quality scores.
    """

    # Core Detection & Segmentation
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    PANOPTIC_SEGMENTATION = "panoptic_segmentation"
    CLASSIFICATION = "classification"

    # Geometric Understanding
    DEPTH_ESTIMATION = "depth_estimation"
    SURFACE_NORMALS = "surface_normals"
    STEREO_MATCHING = "stereo_matching"

    # Motion & Temporal
    OPTICAL_FLOW = "optical_flow"
    MOTION_SEGMENTATION = "motion_segmentation"

    # Structural Analysis
    POSE_ESTIMATION = "pose_estimation"
    KEYPOINT_DETECTION = "keypoint_detection"
    EDGE_DETECTION = "edge_detection"
    LINE_DETECTION = "line_detection"

    # Attention & Saliency
    SALIENCY_DETECTION = "saliency_detection"
    ATTENTION_PREDICTION = "attention_prediction"

    # Image Enhancement & Restoration
    DENOISING = "denoising"
    SUPER_RESOLUTION = "super_resolution"
    INPAINTING = "inpainting"
    DEHAZE = "dehaze"
    SHADOW_REMOVAL = "shadow_removal"
    REFLECTION_REMOVAL = "reflection_removal"

    # Color & Style
    COLORIZATION = "colorization"
    STYLE_TRANSFER = "style_transfer"
    WHITE_BALANCE = "white_balance"

    # Advanced Segmentation
    MATTING = "matting"
    HAIR_SEGMENTATION = "hair_segmentation"
    SKY_SEGMENTATION = "sky_segmentation"

    # Medical & Scientific
    MEDICAL_SEGMENTATION = "medical_segmentation"
    CELL_COUNTING = "cell_counting"

    # Document & Text
    TEXT_DETECTION = "text_detection"
    DOCUMENT_LAYOUT = "document_layout"

    # 3D Understanding
    DEPTH_COMPLETION = "depth_completion"
    SURFACE_RECONSTRUCTION = "surface_reconstruction"
    CAMERA_POSE = "camera_pose"

    # Quality Assessment
    IMAGE_QUALITY = "image_quality"
    AESTHETIC_SCORING = "aesthetic_scoring"

    @classmethod
    def all_tasks(cls) -> List["TaskType"]:
        """
        Get all available task types.

        Returns:
            List of all TaskType enum values.

        Example:
            >>> all_tasks = TaskType.all_tasks()
            >>> print(len(all_tasks))
            42
        """
        return list(cls)

    @classmethod
    def get_task_categories(cls) -> Dict[str, List["TaskType"]]:
        """
        Get tasks organized by categories.

        Returns:
            Dictionary mapping category names to lists of tasks.

        Example:
            >>> categories = TaskType.get_task_categories()
            >>> core_tasks = categories["Core Detection & Segmentation"]
            >>> assert TaskType.DETECTION in core_tasks
        """
        return {
            "Core Detection & Segmentation": [
                cls.DETECTION,
                cls.SEGMENTATION,
                cls.INSTANCE_SEGMENTATION,
                cls.PANOPTIC_SEGMENTATION,
                cls.CLASSIFICATION,
            ],
            "Geometric Understanding": [
                cls.DEPTH_ESTIMATION,
                cls.SURFACE_NORMALS,
                cls.STEREO_MATCHING,
            ],
            "Motion & Temporal": [
                cls.OPTICAL_FLOW,
                cls.MOTION_SEGMENTATION,
            ],
            "Structural Analysis": [
                cls.POSE_ESTIMATION,
                cls.KEYPOINT_DETECTION,
                cls.EDGE_DETECTION,
                cls.LINE_DETECTION,
            ],
            "Attention & Saliency": [
                cls.SALIENCY_DETECTION,
                cls.ATTENTION_PREDICTION,
            ],
            "Image Enhancement & Restoration": [
                cls.DENOISING,
                cls.SUPER_RESOLUTION,
                cls.INPAINTING,
                cls.DEHAZE,
                cls.SHADOW_REMOVAL,
                cls.REFLECTION_REMOVAL,
            ],
            "Color & Style": [
                cls.COLORIZATION,
                cls.STYLE_TRANSFER,
                cls.WHITE_BALANCE,
            ],
            "Advanced Segmentation": [
                cls.MATTING,
                cls.HAIR_SEGMENTATION,
                cls.SKY_SEGMENTATION,
            ],
            "Medical & Scientific": [
                cls.MEDICAL_SEGMENTATION,
                cls.CELL_COUNTING,
            ],
            "Document & Text": [
                cls.TEXT_DETECTION,
                cls.DOCUMENT_LAYOUT,
            ],
            "3D Understanding": [
                cls.DEPTH_COMPLETION,
                cls.SURFACE_RECONSTRUCTION,
                cls.CAMERA_POSE,
            ],
            "Quality Assessment": [
                cls.IMAGE_QUALITY,
                cls.AESTHETIC_SCORING,
            ],
        }

    @classmethod
    def get_compatible_tasks(cls, task: "TaskType") -> List["TaskType"]:
        """
        Get tasks that are commonly combined with the given task.

        Args:
            task: The reference task.

        Returns:
            List of tasks that work well together with the reference task.

        Example:
            >>> compatible = TaskType.get_compatible_tasks(TaskType.DETECTION)
            >>> assert TaskType.INSTANCE_SEGMENTATION in compatible
        """
        compatibility_map = {
            # Core tasks work well together
            cls.DETECTION: [
                cls.CLASSIFICATION, cls.INSTANCE_SEGMENTATION, cls.SEGMENTATION,
                cls.KEYPOINT_DETECTION, cls.POSE_ESTIMATION
            ],
            cls.SEGMENTATION: [
                cls.DETECTION, cls.DEPTH_ESTIMATION, cls.SURFACE_NORMALS,
                cls.EDGE_DETECTION, cls.SALIENCY_DETECTION
            ],
            cls.INSTANCE_SEGMENTATION: [
                cls.DETECTION, cls.PANOPTIC_SEGMENTATION, cls.SEGMENTATION,
                cls.MATTING, cls.POSE_ESTIMATION
            ],

            # Geometric tasks complement each other
            cls.DEPTH_ESTIMATION: [
                cls.SURFACE_NORMALS, cls.SEGMENTATION, cls.EDGE_DETECTION,
                cls.STEREO_MATCHING, cls.DEPTH_COMPLETION
            ],
            cls.SURFACE_NORMALS: [
                cls.DEPTH_ESTIMATION, cls.EDGE_DETECTION, cls.SEGMENTATION
            ],

            # Motion tasks
            cls.OPTICAL_FLOW: [
                cls.MOTION_SEGMENTATION, cls.DETECTION, cls.SEGMENTATION
            ],

            # Enhancement tasks can be combined
            cls.DENOISING: [
                cls.SUPER_RESOLUTION, cls.DEHAZE, cls.SHADOW_REMOVAL
            ],
            cls.SUPER_RESOLUTION: [
                cls.DENOISING, cls.INPAINTING, cls.DEHAZE
            ],

            # Specialized segmentation
            cls.MATTING: [
                cls.INSTANCE_SEGMENTATION, cls.SEGMENTATION, cls.HAIR_SEGMENTATION
            ],

            # Structural analysis
            cls.KEYPOINT_DETECTION: [
                cls.POSE_ESTIMATION, cls.DETECTION, cls.EDGE_DETECTION
            ],
            cls.POSE_ESTIMATION: [
                cls.KEYPOINT_DETECTION, cls.DETECTION, cls.INSTANCE_SEGMENTATION
            ],
        }

        return compatibility_map.get(task, [])

    @classmethod
    def get_output_types(cls, task: "TaskType") -> Dict[str, str]:
        """
        Get the expected output types for a given task.

        Args:
            task: The task to get output types for.

        Returns:
            Dictionary mapping output names to their types/shapes.

        Example:
            >>> outputs = TaskType.get_output_types(TaskType.DETECTION)
            >>> assert "bboxes" in outputs
            >>> assert "classes" in outputs
        """
        output_types = {
            cls.DETECTION: {
                "bboxes": "float32[N, 4]",
                "classes": "int32[N]",
                "scores": "float32[N]"
            },
            cls.SEGMENTATION: {
                "masks": "int32[H, W]",
                "logits": "float32[H, W, C]"
            },
            cls.INSTANCE_SEGMENTATION: {
                "instance_masks": "int32[H, W]",
                "instance_ids": "int32[N]",
                "bboxes": "float32[N, 4]",
                "classes": "int32[N]"
            },
            cls.CLASSIFICATION: {
                "logits": "float32[C]",
                "probabilities": "float32[C]"
            },
            cls.DEPTH_ESTIMATION: {
                "depth": "float32[H, W]",
                "confidence": "float32[H, W]"
            },
            cls.SURFACE_NORMALS: {
                "normals": "float32[H, W, 3]",
                "confidence": "float32[H, W]"
            },
            cls.OPTICAL_FLOW: {
                "flow": "float32[H, W, 2]",
                "occlusion": "float32[H, W]"
            },
            cls.KEYPOINT_DETECTION: {
                "keypoints": "float32[N, K, 2]",
                "visibility": "float32[N, K]",
                "scores": "float32[N]"
            },
            cls.POSE_ESTIMATION: {
                "poses": "float32[N, P]",
                "confidence": "float32[N]"
            },
            cls.EDGE_DETECTION: {
                "edges": "float32[H, W]",
                "gradients": "float32[H, W, 2]"
            },
            cls.SALIENCY_DETECTION: {
                "saliency": "float32[H, W]",
                "attention": "float32[H, W]"
            },
            cls.DENOISING: {
                "denoised": "float32[H, W, C]",
                "noise_estimate": "float32[H, W, C]"
            },
            cls.SUPER_RESOLUTION: {
                "high_res": "float32[H*S, W*S, C]",
                "upscale_factor": "int32"
            },
            cls.MATTING: {
                "alpha": "float32[H, W]",
                "foreground": "float32[H, W, C]",
                "background": "float32[H, W, C]"
            },
            cls.COLORIZATION: {
                "colored": "float32[H, W, 3]",
                "confidence": "float32[H, W]"
            },
        }

        return output_types.get(task, {"output": "float32[...]"})

    @classmethod
    def from_string(cls, task_str: str) -> "TaskType":
        """
        Create TaskType from string value.

        Args:
            task_str: String representation of the task.

        Returns:
            TaskType enum value.

        Raises:
            ValueError: If task_str is not a valid task type.

        Example:
            >>> task = TaskType.from_string("depth_estimation")
            >>> assert task == TaskType.DEPTH_ESTIMATION
        """
        task_str = task_str.lower().strip()
        for task in cls:
            if task.value == task_str:
                return task

        valid_tasks = [task.value for task in cls]
        raise ValueError(
            f"Invalid task type: '{task_str}'. "
            f"Valid options are: {valid_tasks}"
        )

    @classmethod
    def from_strings(cls, task_strs: List[str]) -> List["TaskType"]:
        """
        Create list of TaskTypes from list of strings.

        Args:
            task_strs: List of string representations of tasks.

        Returns:
            List of TaskType enum values.

        Raises:
            ValueError: If any task_str is not a valid task type.

        Example:
            >>> tasks = TaskType.from_strings(["detection", "depth_estimation"])
            >>> assert TaskType.DETECTION in tasks
            >>> assert TaskType.DEPTH_ESTIMATION in tasks
        """
        return [cls.from_string(task_str) for task_str in task_strs]

    @classmethod
    def to_strings(cls, tasks: List["TaskType"]) -> List[str]:
        """
        Convert list of TaskTypes to list of strings.

        Args:
            tasks: List of TaskType enum values.

        Returns:
            List of string representations.

        Example:
            >>> tasks = [TaskType.DETECTION, TaskType.SURFACE_NORMALS]
            >>> strings = TaskType.to_strings(tasks)
            >>> assert strings == ["detection", "surface_normals"]
        """
        return [task.value for task in tasks]

    def get_category(self) -> str:
        """
        Get the category this task belongs to.

        Returns:
            Category name as string.

        Example:
            >>> category = TaskType.DEPTH_ESTIMATION.get_category()
            >>> assert category == "Geometric Understanding"
        """
        categories = self.get_task_categories()
        for category_name, task_list in categories.items():
            if self in task_list:
                return category_name
        return "Uncategorized"

    def is_compatible_with(self, other: "TaskType") -> bool:
        """
        Check if this task is compatible with another task.

        Args:
            other: Another TaskType to check compatibility with.

        Returns:
            True if tasks are compatible, False otherwise.

        Example:
            >>> is_compat = TaskType.DETECTION.is_compatible_with(TaskType.SEGMENTATION)
            >>> assert is_compat == True
        """
        compatible_tasks = self.get_compatible_tasks(self)
        return other in compatible_tasks

    def __str__(self) -> str:
        """String representation of the task type."""
        return self.value

    def __repr__(self) -> str:
        """Detailed string representation of the task type."""
        return f"TaskType.{self.name}"


# ---------------------------------------------------------------------

class TaskConfiguration:
    """
    Configuration helper for managing task combinations in multi-task models.

    This class provides utilities for validating and managing task configurations,
    ensuring that valid combinations are used and providing helpful error messages.

    Args:
        tasks: List of TaskType enum values to enable.
        validate_compatibility: Whether to check task compatibility.

    Example:
        >>> config = TaskConfiguration([TaskType.DETECTION, TaskType.SEGMENTATION])
        >>> assert config.has_detection()
        >>> assert config.has_segmentation()
        >>> assert not config.has_classification()
    """

    def __init__(self, tasks: List[TaskType], validate_compatibility: bool = True):
        """
        Initialize task configuration.

        Args:
            tasks: List of TaskType enum values to enable.
            validate_compatibility: Whether to validate task compatibility.

        Raises:
            ValueError: If tasks list is empty, contains duplicates, or
                       contains incompatible tasks (when validation enabled).
        """
        if not tasks:
            raise ValueError("At least one task must be specified")

        # Check for duplicates
        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate tasks found in configuration")

        self._tasks: Set[TaskType] = set(tasks)

        if validate_compatibility and len(tasks) > 1:
            self._validate_task_compatibility()

    def _validate_task_compatibility(self) -> None:
        """Validate that all tasks in the configuration are compatible."""
        task_list = list(self._tasks)

        # Check for obviously incompatible combinations
        incompatible_pairs = [
            (TaskType.COLORIZATION, TaskType.DENOISING),  # Different input requirements
            (TaskType.STEREO_MATCHING, TaskType.OPTICAL_FLOW),  # Different input types
        ]

        for task1, task2 in incompatible_pairs:
            if task1 in self._tasks and task2 in self._tasks:
                raise ValueError(f"Tasks {task1} and {task2} are incompatible")

    @property
    def tasks(self) -> Set[TaskType]:
        """Get the set of enabled tasks."""
        return self._tasks.copy()

    def has_task(self, task: TaskType) -> bool:
        """Check if a specific task is enabled."""
        return task in self._tasks

    # Core task checks
    def has_detection(self) -> bool:
        """Check if detection task is enabled."""
        return TaskType.DETECTION in self._tasks

    def has_segmentation(self) -> bool:
        """Check if segmentation task is enabled."""
        return TaskType.SEGMENTATION in self._tasks

    def has_classification(self) -> bool:
        """Check if classification task is enabled."""
        return TaskType.CLASSIFICATION in self._tasks

    # Geometric task checks
    def has_depth_estimation(self) -> bool:
        """Check if depth estimation task is enabled."""
        return TaskType.DEPTH_ESTIMATION in self._tasks

    def has_surface_normals(self) -> bool:
        """Check if surface normals estimation task is enabled."""
        return TaskType.SURFACE_NORMALS in self._tasks

    # Instance segmentation checks
    def has_instance_segmentation(self) -> bool:
        """Check if instance segmentation task is enabled."""
        return TaskType.INSTANCE_SEGMENTATION in self._tasks

    def has_panoptic_segmentation(self) -> bool:
        """Check if panoptic segmentation task is enabled."""
        return TaskType.PANOPTIC_SEGMENTATION in self._tasks

    # Enhancement task checks
    def has_denoising(self) -> bool:
        """Check if denoising task is enabled."""
        return TaskType.DENOISING in self._tasks

    def has_super_resolution(self) -> bool:
        """Check if super resolution task is enabled."""
        return TaskType.SUPER_RESOLUTION in self._tasks

    def is_single_task(self) -> bool:
        """Check if only one task is enabled."""
        return len(self._tasks) == 1

    def is_multi_task(self) -> bool:
        """Check if multiple tasks are enabled."""
        return len(self._tasks) > 1

    def get_enabled_tasks(self) -> List[TaskType]:
        """Get list of enabled tasks in a consistent order."""
        # Return in a consistent order for reproducibility
        all_tasks = TaskType.all_tasks()
        return [task for task in all_tasks if task in self._tasks]

    def get_task_names(self) -> List[str]:
        """Get list of enabled task names as strings."""
        return TaskType.to_strings(self.get_enabled_tasks())

    def get_tasks_by_category(self) -> Dict[str, List[TaskType]]:
        """
        Get enabled tasks organized by category.

        Returns:
            Dictionary mapping category names to lists of enabled tasks.
        """
        categories = TaskType.get_task_categories()
        result = {}

        for category_name, category_tasks in categories.items():
            enabled_in_category = [task for task in category_tasks if task in self._tasks]
            if enabled_in_category:
                result[category_name] = enabled_in_category

        return result

    def get_output_specifications(self) -> Dict[TaskType, Dict[str, str]]:
        """
        Get output specifications for all enabled tasks.

        Returns:
            Dictionary mapping tasks to their output specifications.
        """
        return {task: TaskType.get_output_types(task) for task in self._tasks}

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dictionary with task names and their enabled status.
        """
        result = {}
        for task in TaskType.all_tasks():
            result[f"enable_{task.value}"] = task in self._tasks
        return result

    @classmethod
    def from_dict(cls, config_dict: dict, validate_compatibility: bool = True) -> "TaskConfiguration":
        """
        Create TaskConfiguration from dictionary.

        Args:
            config_dict: Dictionary with boolean flags for tasks.
            validate_compatibility: Whether to validate task compatibility.

        Returns:
            TaskConfiguration instance.

        Example:
            >>> config_dict = {
            ...     "enable_detection": True,
            ...     "enable_depth_estimation": True,
            ...     "enable_classification": False
            ... }
            >>> config = TaskConfiguration.from_dict(config_dict)
        """
        tasks = []

        for task in TaskType.all_tasks():
            key = f"enable_{task.value}"
            if config_dict.get(key, False):
                tasks.append(task)

        return cls(tasks, validate_compatibility=validate_compatibility)

    @classmethod
    def from_strings(cls, task_strings: List[str], validate_compatibility: bool = True) -> "TaskConfiguration":
        """
        Create TaskConfiguration from list of task name strings.

        Args:
            task_strings: List of task names as strings.
            validate_compatibility: Whether to validate task compatibility.

        Returns:
            TaskConfiguration instance.

        Example:
            >>> config = TaskConfiguration.from_strings(["detection", "depth_estimation"])
        """
        tasks = TaskType.from_strings(task_strings)
        return cls(tasks, validate_compatibility=validate_compatibility)

    def __str__(self) -> str:
        """String representation of the configuration."""
        task_names = self.get_task_names()
        return f"TaskConfiguration({', '.join(task_names)})"

    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        tasks_repr = [repr(task) for task in self.get_enabled_tasks()]
        return f"TaskConfiguration([{', '.join(tasks_repr)}])"

    def __eq__(self, other) -> bool:
        """Check equality with another TaskConfiguration."""
        if not isinstance(other, TaskConfiguration):
            return False
        return self._tasks == other._tasks

    def __hash__(self) -> int:
        """Hash function for TaskConfiguration."""
        return hash(frozenset(self._tasks))


# ---------------------------------------------------------------------

# Predefined common task configurations
class CommonTaskConfigurations:
    """
    Predefined common task configurations for convenience.

    This class provides commonly used task combinations as class properties,
    making it easy to create models with standard configurations.
    """

    # Single task configurations - Core tasks
    DETECTION_ONLY = TaskConfiguration([TaskType.DETECTION])
    SEGMENTATION_ONLY = TaskConfiguration([TaskType.SEGMENTATION])
    CLASSIFICATION_ONLY = TaskConfiguration([TaskType.CLASSIFICATION])
    DEPTH_ONLY = TaskConfiguration([TaskType.DEPTH_ESTIMATION])
    SURFACE_NORMALS_ONLY = TaskConfiguration([TaskType.SURFACE_NORMALS])

    # Single task configurations - Specialized
    INSTANCE_SEGMENTATION_ONLY = TaskConfiguration([TaskType.INSTANCE_SEGMENTATION])
    PANOPTIC_SEGMENTATION_ONLY = TaskConfiguration([TaskType.PANOPTIC_SEGMENTATION])
    DENOISING_ONLY = TaskConfiguration([TaskType.DENOISING])
    SUPER_RESOLUTION_ONLY = TaskConfiguration([TaskType.SUPER_RESOLUTION])
    KEYPOINT_DETECTION_ONLY = TaskConfiguration([TaskType.KEYPOINT_DETECTION])

    # Two-task combinations - Core
    DETECTION_SEGMENTATION = TaskConfiguration([TaskType.DETECTION, TaskType.SEGMENTATION])
    DETECTION_CLASSIFICATION = TaskConfiguration([TaskType.DETECTION, TaskType.CLASSIFICATION])
    SEGMENTATION_CLASSIFICATION = TaskConfiguration([TaskType.SEGMENTATION, TaskType.CLASSIFICATION])

    # Two-task combinations - Geometric
    DEPTH_NORMALS = TaskConfiguration([TaskType.DEPTH_ESTIMATION, TaskType.SURFACE_NORMALS])
    SEGMENTATION_DEPTH = TaskConfiguration([TaskType.SEGMENTATION, TaskType.DEPTH_ESTIMATION])
    DETECTION_DEPTH = TaskConfiguration([TaskType.DETECTION, TaskType.DEPTH_ESTIMATION])

    # Two-task combinations - Instance segmentation
    DETECTION_INSTANCE_SEG = TaskConfiguration([TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION])
    SEGMENTATION_INSTANCE_SEG = TaskConfiguration([TaskType.SEGMENTATION, TaskType.INSTANCE_SEGMENTATION])

    # Three-task combinations
    DETECTION_SEGMENTATION_DEPTH = TaskConfiguration([
        TaskType.DETECTION, TaskType.SEGMENTATION, TaskType.DEPTH_ESTIMATION
    ])
    DETECTION_SEGMENTATION_CLASSIFICATION = TaskConfiguration([
        TaskType.DETECTION, TaskType.SEGMENTATION, TaskType.CLASSIFICATION
    ])
    GEOMETRIC_UNDERSTANDING = TaskConfiguration([
        TaskType.DEPTH_ESTIMATION, TaskType.SURFACE_NORMALS, TaskType.EDGE_DETECTION
    ])

    # Panoptic understanding (full scene parsing)
    PANOPTIC_UNDERSTANDING = TaskConfiguration([
        TaskType.DETECTION, TaskType.SEGMENTATION, TaskType.INSTANCE_SEGMENTATION, TaskType.DEPTH_ESTIMATION
    ])

    # Enhancement pipeline
    IMAGE_ENHANCEMENT = TaskConfiguration([
        TaskType.DENOISING, TaskType.SUPER_RESOLUTION, TaskType.DEHAZE
    ])

    # Pose and structure
    POSE_AND_STRUCTURE = TaskConfiguration([
        TaskType.POSE_ESTIMATION, TaskType.KEYPOINT_DETECTION, TaskType.EDGE_DETECTION
    ])

    # All core tasks
    ALL_CORE_TASKS = TaskConfiguration([
        TaskType.DETECTION, TaskType.SEGMENTATION, TaskType.CLASSIFICATION,
        TaskType.DEPTH_ESTIMATION, TaskType.SURFACE_NORMALS
    ])

    # All tasks (be careful with this one!)
    ALL_TASKS = TaskConfiguration(TaskType.all_tasks(), validate_compatibility=False)

    @classmethod
    def get_all_configurations(cls) -> List[TaskConfiguration]:
        """
        Get all predefined configurations.

        Returns:
            List of all predefined TaskConfiguration instances.
        """
        return [
            # Single tasks - Core
            cls.DETECTION_ONLY,
            cls.SEGMENTATION_ONLY,
            cls.CLASSIFICATION_ONLY,
            cls.DEPTH_ONLY,
            cls.SURFACE_NORMALS_ONLY,

            # Single tasks - Specialized
            cls.INSTANCE_SEGMENTATION_ONLY,
            cls.PANOPTIC_SEGMENTATION_ONLY,
            cls.DENOISING_ONLY,
            cls.SUPER_RESOLUTION_ONLY,
            cls.KEYPOINT_DETECTION_ONLY,

            # Two-task combinations
            cls.DETECTION_SEGMENTATION,
            cls.DETECTION_CLASSIFICATION,
            cls.SEGMENTATION_CLASSIFICATION,
            cls.DEPTH_NORMALS,
            cls.SEGMENTATION_DEPTH,
            cls.DETECTION_DEPTH,
            cls.DETECTION_INSTANCE_SEG,
            cls.SEGMENTATION_INSTANCE_SEG,

            # Three-task combinations
            cls.DETECTION_SEGMENTATION_DEPTH,
            cls.DETECTION_SEGMENTATION_CLASSIFICATION,
            cls.GEOMETRIC_UNDERSTANDING,
            cls.PANOPTIC_UNDERSTANDING,
            cls.IMAGE_ENHANCEMENT,
            cls.POSE_AND_STRUCTURE,

            # Comprehensive
            cls.ALL_CORE_TASKS,
        ]

    @classmethod
    def get_configurations_by_complexity(cls) -> Dict[str, List[TaskConfiguration]]:
        """
        Get configurations organized by complexity level.

        Returns:
            Dictionary mapping complexity levels to configuration lists.
        """
        return {
            "Single Task": [
                cls.DETECTION_ONLY, cls.SEGMENTATION_ONLY, cls.CLASSIFICATION_ONLY,
                cls.DEPTH_ONLY, cls.SURFACE_NORMALS_ONLY, cls.INSTANCE_SEGMENTATION_ONLY,
                cls.DENOISING_ONLY, cls.SUPER_RESOLUTION_ONLY, cls.KEYPOINT_DETECTION_ONLY
            ],
            "Two Tasks": [
                cls.DETECTION_SEGMENTATION, cls.DETECTION_CLASSIFICATION,
                cls.SEGMENTATION_CLASSIFICATION, cls.DEPTH_NORMALS,
                cls.SEGMENTATION_DEPTH, cls.DETECTION_DEPTH,
                cls.DETECTION_INSTANCE_SEG, cls.SEGMENTATION_INSTANCE_SEG
            ],
            "Three Tasks": [
                cls.DETECTION_SEGMENTATION_DEPTH, cls.DETECTION_SEGMENTATION_CLASSIFICATION,
                cls.GEOMETRIC_UNDERSTANDING, cls.IMAGE_ENHANCEMENT, cls.POSE_AND_STRUCTURE
            ],
            "Complex Multi-Task": [
                cls.PANOPTIC_UNDERSTANDING, cls.ALL_CORE_TASKS
            ]
        }


# ---------------------------------------------------------------------

# Utility functions for backward compatibility and convenience
def parse_task_list(tasks, validate_compatibility: bool = True) -> TaskConfiguration:
    """
    Parse various task input formats into TaskConfiguration.

    Args:
        tasks: Can be:
            - List of TaskType enums
            - List of strings
            - TaskConfiguration instance
            - Single TaskType enum
            - Single string
        validate_compatibility: Whether to validate task compatibility.

    Returns:
        TaskConfiguration instance.

    Example:
        >>> config1 = parse_task_list(["detection", "depth_estimation"])
        >>> config2 = parse_task_list([TaskType.DETECTION, TaskType.DEPTH_ESTIMATION])
        >>> assert config1 == config2
    """
    if isinstance(tasks, TaskConfiguration):
        return tasks
    elif isinstance(tasks, TaskType):
        return TaskConfiguration([tasks], validate_compatibility=validate_compatibility)
    elif isinstance(tasks, str):
        return TaskConfiguration([TaskType.from_string(tasks)], validate_compatibility=validate_compatibility)
    elif isinstance(tasks, (list, tuple)):
        if not tasks:
            raise ValueError("Task list cannot be empty")

        # Check if first element is string or TaskType
        if isinstance(tasks[0], str):
            return TaskConfiguration.from_strings(list(tasks), validate_compatibility=validate_compatibility)
        elif isinstance(tasks[0], TaskType):
            return TaskConfiguration(list(tasks), validate_compatibility=validate_compatibility)
        else:
            raise ValueError(f"Invalid task type in list: {type(tasks[0])}")
    else:
        raise ValueError(f"Invalid tasks format: {type(tasks)}")


def get_task_suggestions(base_task: TaskType, max_suggestions: int = 5) -> List[TaskType]:
    """
    Get task suggestions that work well with a base task.

    Args:
        base_task: The base task to find compatible tasks for.
        max_suggestions: Maximum number of suggestions to return.

    Returns:
        List of compatible TaskType suggestions.

    Example:
        >>> suggestions = get_task_suggestions(TaskType.DETECTION)
        >>> assert TaskType.SEGMENTATION in suggestions
    """
    compatible_tasks = TaskType.get_compatible_tasks(base_task)
    return compatible_tasks[:max_suggestions]


def validate_task_combination(tasks: List[TaskType]) -> tuple[bool, Optional[str]]:
    """
    Validate if a combination of tasks is reasonable.

    Args:
        tasks: List of tasks to validate.

    Returns:
        Tuple of (is_valid, error_message).

    Example:
        >>> is_valid, error = validate_task_combination([TaskType.DETECTION, TaskType.SEGMENTATION])
        >>> assert is_valid == True
        >>> assert error is None
    """
    try:
        TaskConfiguration(tasks, validate_compatibility=True)
        return True, None
    except ValueError as e:
        return False, str(e)

# ---------------------------------------------------------------------