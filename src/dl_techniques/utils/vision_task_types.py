"""
Task Type Enumerations.

This module defines enumerations for different computer vision tasks
supported by the YOLOv12 multitask architecture. Using enumerations
provides type safety and prevents string-based errors.
"""

from enum import Enum, unique
from typing import List, Set


# ---------------------------------------------------------------------

@unique
class TaskType(Enum):
    """
    Enumeration of supported computer vision tasks in YOLOv12 multi-task models.

    Each task represents a different computer vision capability that can be
    enabled in the multi-task architecture.

    Values:
        DETECTION: Object detection with bounding box regression and classification.
        SEGMENTATION: Pixel-level semantic segmentation.
        CLASSIFICATION: Global image-level classification.
    """

    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"

    @classmethod
    def all_tasks(cls) -> List["TaskType"]:
        """
        Get all available task types.

        Returns:
            List of all TaskType enum values.

        Example:
            >>> all_tasks = TaskType.all_tasks()
            >>> print([task.value for task in all_tasks])
            ['detection', 'segmentation', 'classification']
        """
        return list(cls)

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
            >>> task = TaskType.from_string("detection")
            >>> assert task == TaskType.DETECTION
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
            >>> tasks = TaskType.from_strings(["detection", "segmentation"])
            >>> assert TaskType.DETECTION in tasks
            >>> assert TaskType.SEGMENTATION in tasks
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
            >>> tasks = [TaskType.DETECTION, TaskType.SEGMENTATION]
            >>> strings = TaskType.to_strings(tasks)
            >>> assert strings == ["detection", "segmentation"]
        """
        return [task.value for task in tasks]

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

    Example:
        >>> config = TaskConfiguration([TaskType.DETECTION, TaskType.SEGMENTATION])
        >>> assert config.has_detection()
        >>> assert config.has_segmentation()
        >>> assert not config.has_classification()
    """

    def __init__(self, tasks: List[TaskType]):
        """
        Initialize task configuration.

        Args:
            tasks: List of TaskType enum values to enable.

        Raises:
            ValueError: If tasks list is empty or contains duplicates.
        """
        if not tasks:
            raise ValueError("At least one task must be specified")

        # Check for duplicates
        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate tasks found in configuration")

        self._tasks: Set[TaskType] = set(tasks)

    @property
    def tasks(self) -> Set[TaskType]:
        """Get the set of enabled tasks."""
        return self._tasks.copy()

    def has_detection(self) -> bool:
        """Check if detection task is enabled."""
        return TaskType.DETECTION in self._tasks

    def has_segmentation(self) -> bool:
        """Check if segmentation task is enabled."""
        return TaskType.SEGMENTATION in self._tasks

    def has_classification(self) -> bool:
        """Check if classification task is enabled."""
        return TaskType.CLASSIFICATION in self._tasks

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

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dictionary with boolean flags for each task.
        """
        return {
            "enable_detection": self.has_detection(),
            "enable_segmentation": self.has_segmentation(),
            "enable_classification": self.has_classification(),
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TaskConfiguration":
        """
        Create TaskConfiguration from dictionary.

        Args:
            config_dict: Dictionary with boolean flags for tasks.

        Returns:
            TaskConfiguration instance.

        Example:
            >>> config_dict = {
            ...     "enable_detection": True,
            ...     "enable_segmentation": False,
            ...     "enable_classification": True
            ... }
            >>> config = TaskConfiguration.from_dict(config_dict)
        """
        tasks = []

        if config_dict.get("enable_detection", False):
            tasks.append(TaskType.DETECTION)
        if config_dict.get("enable_segmentation", False):
            tasks.append(TaskType.SEGMENTATION)
        if config_dict.get("enable_classification", False):
            tasks.append(TaskType.CLASSIFICATION)

        return cls(tasks)

    @classmethod
    def from_strings(cls, task_strings: List[str]) -> "TaskConfiguration":
        """
        Create TaskConfiguration from list of task name strings.

        Args:
            task_strings: List of task names as strings.

        Returns:
            TaskConfiguration instance.

        Example:
            >>> config = TaskConfiguration.from_strings(["detection", "segmentation"])
        """
        tasks = TaskType.from_strings(task_strings)
        return cls(tasks)

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

    # Single task configurations
    DETECTION_ONLY = TaskConfiguration([TaskType.DETECTION])
    SEGMENTATION_ONLY = TaskConfiguration([TaskType.SEGMENTATION])
    CLASSIFICATION_ONLY = TaskConfiguration([TaskType.CLASSIFICATION])

    # Two-task combinations
    DETECTION_SEGMENTATION = TaskConfiguration([TaskType.DETECTION, TaskType.SEGMENTATION])
    DETECTION_CLASSIFICATION = TaskConfiguration([TaskType.DETECTION, TaskType.CLASSIFICATION])
    SEGMENTATION_CLASSIFICATION = TaskConfiguration([TaskType.SEGMENTATION, TaskType.CLASSIFICATION])

    # All tasks
    ALL_TASKS = TaskConfiguration([TaskType.DETECTION, TaskType.SEGMENTATION, TaskType.CLASSIFICATION])

    @classmethod
    def get_all_configurations(cls) -> List[TaskConfiguration]:
        """
        Get all predefined configurations.

        Returns:
            List of all predefined TaskConfiguration instances.
        """
        return [
            cls.DETECTION_ONLY,
            cls.SEGMENTATION_ONLY,
            cls.CLASSIFICATION_ONLY,
            cls.DETECTION_SEGMENTATION,
            cls.DETECTION_CLASSIFICATION,
            cls.SEGMENTATION_CLASSIFICATION,
            cls.ALL_TASKS,
        ]


# ---------------------------------------------------------------------

# Utility functions for backward compatibility
def parse_task_list(tasks) -> TaskConfiguration:
    """
    Parse various task input formats into TaskConfiguration.

    Args:
        tasks: Can be:
            - List of TaskType enums
            - List of strings
            - TaskConfiguration instance
            - Single TaskType enum
            - Single string

    Returns:
        TaskConfiguration instance.

    Example:
        >>> config1 = parse_task_list(["detection", "segmentation"])
        >>> config2 = parse_task_list([TaskType.DETECTION, TaskType.SEGMENTATION])
        >>> assert config1 == config2
    """
    if isinstance(tasks, TaskConfiguration):
        return tasks
    elif isinstance(tasks, TaskType):
        return TaskConfiguration([tasks])
    elif isinstance(tasks, str):
        return TaskConfiguration([TaskType.from_string(tasks)])
    elif isinstance(tasks, (list, tuple)):
        if not tasks:
            raise ValueError("Task list cannot be empty")

        # Check if first element is string or TaskType
        if isinstance(tasks[0], str):
            return TaskConfiguration.from_strings(list(tasks))
        elif isinstance(tasks[0], TaskType):
            return TaskConfiguration(list(tasks))
        else:
            raise ValueError(f"Invalid task type in list: {type(tasks[0])}")
    else:
        raise ValueError(f"Invalid tasks format: {type(tasks)}")

# ---------------------------------------------------------------------
