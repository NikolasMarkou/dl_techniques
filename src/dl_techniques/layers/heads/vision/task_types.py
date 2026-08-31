"""Task types and configuration objects for the vision heads.

This module says what a vision task is. It builds no layer. Three classes and
three functions live here:

* :class:`VisionTaskType` -- an enum of 37 task names, with lookup helpers for
  categories, compatible tasks and output shapes.
* :class:`TaskConfiguration` -- a validated set of tasks, plus queries for
  what that set contains.
* :class:`CommonTaskConfigurations` -- 26 ready-made task sets.
* :func:`parse_task_list`, :func:`get_task_suggestions` and
  :func:`validate_task_combination` -- module-level helpers.

``vision/factory.py`` reads these objects and picks the head class to build.
Only 10 of the 37 task names have a head; the other 27 raise ``ValueError``.
The table in :class:`VisionTaskType` lists both groups.

``TaskType`` is a module-level alias for :class:`VisionTaskType`, kept so
older callers keep working. It is the same object, not a copy.
"""

from enum import Enum, unique
from typing import List, Set, Dict, Optional


# ---------------------------------------------------------------------

@unique
class VisionTaskType(Enum):
    """
    The 37 computer vision tasks a head can be asked to serve.

    Each member carries a lowercase string value. That string is the
    serialization form, and :meth:`from_string` parses it back.

    **Head Dispatch:**

    ``vision/factory.py``'s ``create_vision_head`` decides which head class
    runs a task. Ten members reach a head. Five get a class of their own.
    Five reuse another task's class, two of them with a different output
    channel count. Denoising and super-resolution route through
    ``create_enhancement_head``, which returns an ``EnhancementHead``. The
    remaining 27 members have no head and raise ``ValueError``.

    .. code-block:: text

        Own head class
        --------------------------------------------
        DETECTION              DetectionHead
        SEGMENTATION           SegmentationHead
        INSTANCE_SEGMENTATION  InstanceSegmentationHead
        CLASSIFICATION         ClassificationHead
        DEPTH_ESTIMATION       DepthEstimationHead

        Reuses another task's head class
        --------------------------------------------
        SURFACE_NORMALS     DepthEstimationHead(output_channels=3)
        OPTICAL_FLOW        DepthEstimationHead(output_channels=2)
        KEYPOINT_DETECTION  DetectionHead
        DENOISING           EnhancementHead
        SUPER_RESOLUTION    EnhancementHead(scale_factor=2)

        No head -- raises ValueError("Unsupported task type")
        --------------------------------------------
        PANOPTIC_SEGMENTATION, STEREO_MATCHING,
        MOTION_SEGMENTATION, POSE_ESTIMATION, EDGE_DETECTION,
        LINE_DETECTION, SALIENCY_DETECTION, ATTENTION_PREDICTION,
        INPAINTING, DEHAZE, SHADOW_REMOVAL, REFLECTION_REMOVAL,
        COLORIZATION, STYLE_TRANSFER, WHITE_BALANCE, MATTING,
        HAIR_SEGMENTATION, SKY_SEGMENTATION, MEDICAL_SEGMENTATION,
        CELL_COUNTING, TEXT_DETECTION, DOCUMENT_LAYOUT,
        DEPTH_COMPLETION, SURFACE_RECONSTRUCTION, CAMERA_POSE,
        IMAGE_QUALITY, AESTHETIC_SCORING

    **Categories:**

    :meth:`get_task_categories` files every member into one of twelve named
    categories, listed here with the number of members in each. The filing is
    descriptive. It does not decide which head class runs a task, and the
    factory never reads it.

    .. code-block:: text

        Core Detection & Segmentation     5
        Geometric Understanding           3
        Motion & Temporal                 2
        Structural Analysis               4
        Attention & Saliency              2
        Image Enhancement & Restoration   6
        Color & Style                     3
        Advanced Segmentation             3
        Medical & Scientific              2
        Document & Text                   2
        3D Understanding                  3
        Quality Assessment                2
                                         --
        total                             37

    Note:
        The dispatch table restates a mapping that lives in
        ``create_vision_head``. That function is the arbiter. If the two
        disagree, the function is right.

    Example:
        >>> task = VisionTaskType.from_string("depth_estimation")
        >>> str(task)
        'depth_estimation'
        >>> task.get_category()
        'Geometric Understanding'
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
    def all_tasks(cls) -> List["VisionTaskType"]:
        """
        List every task member in declaration order.

        The order is stable, so callers can use it to sort task sets
        reproducibly. :meth:`TaskConfiguration.get_enabled_tasks` does.

        :return: All 37 VisionTaskType members, in declaration order.
        :rtype: List[VisionTaskType]
        """
        return list(cls)

    @classmethod
    def get_task_categories(cls) -> Dict[str, List["VisionTaskType"]]:
        """
        Group every task member under one of twelve category names.

        Every member appears exactly once, so the twelve lists partition the
        enum. The grouping is for display and for
        :meth:`TaskConfiguration.get_tasks_by_category`. No factory reads it.

        :return: Category name mapped to the tasks filed under it.
        :rtype: Dict[str, List[VisionTaskType]]
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
    def get_compatible_tasks(cls, task: "VisionTaskType") -> List["VisionTaskType"]:
        """
        List the tasks commonly trained alongside the given task.

        The suggestion map covers 11 of the 37 members. Anything else returns
        an empty list, which means "no suggestion recorded", not "nothing is
        compatible". The map is also not symmetric: a task can appear in
        another's list without that other appearing in its own.

        :param task: The reference task.
        :type task: VisionTaskType
        :return: Tasks recorded as working well with the reference task, or
            an empty list when the reference task has no entry.
        :rtype: List[VisionTaskType]
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
    def get_output_types(cls, task: "VisionTaskType") -> Dict[str, str]:
        """
        Describe the tensors a task is expected to produce.

        The shapes are documentation strings, not runtime shapes. Nothing
        checks a head against them. 15 of the 37 members have an entry; the
        rest fall back to the single generic key ``{"output":
        "float32[...]"}``.

        :param task: The task to describe.
        :type task: VisionTaskType
        :return: Output name mapped to a shape string.
        :rtype: Dict[str, str]
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
    def from_string(cls, task_str: str) -> "VisionTaskType":
        """
        Parse a task name back into its enum member.

        The input is lowercased and stripped first, so ``" Detection "``
        resolves. An unknown name raises rather than falling back.

        :param task_str: String representation of the task.
        :type task_str: str
        :return: VisionTaskType enum value.
        :rtype: VisionTaskType
        :raises ValueError: If task_str is not a valid task type.
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
    def from_strings(cls, task_strs: List[str]) -> List["VisionTaskType"]:
        """
        Parse a list of task names into enum members.

        Calls :meth:`from_string` per element, so the first unknown name
        raises and no partial list is returned.

        :param task_strs: List of string representations of tasks.
        :type task_strs: List[str]
        :return: List of VisionTaskType enum values.
        :rtype: List[VisionTaskType]
        :raises ValueError: If any task_str is not a valid task type.
        """
        return [cls.from_string(task_str) for task_str in task_strs]

    @classmethod
    def to_strings(cls, tasks: List["VisionTaskType"]) -> List[str]:
        """
        Convert enum members back to their string values.

        This is the inverse of :meth:`from_strings` and preserves order.

        :param tasks: List of VisionTaskType enum values.
        :type tasks: List[VisionTaskType]
        :return: The ``value`` of each member, in the order given.
        :rtype: List[str]
        """
        return [task.value for task in tasks]

    def get_category(self) -> str:
        """
        Name the category this task is filed under.

        Every member is filed, so ``"Uncategorized"`` is a guard against a
        member being added to the enum without being added to
        :meth:`get_task_categories`.

        :return: The category name, or ``"Uncategorized"`` if the member is
            missing from the category map.
        :rtype: str
        """
        categories = self.get_task_categories()
        for category_name, task_list in categories.items():
            if self in task_list:
                return category_name
        return "Uncategorized"

    def is_compatible_with(self, other: "VisionTaskType") -> bool:
        """
        Report whether another task is in this task's suggestion list.

        This reads :meth:`get_compatible_tasks`, which covers 11 of the 37
        members. For the other 26 the answer is always False, because their
        suggestion list is empty. False here means "not recorded as a good
        pair", not "rejected". :class:`TaskConfiguration` does not call this
        method; its own check uses a separate two-pair reject list.

        :param other: Another VisionTaskType to check against.
        :type other: VisionTaskType
        :return: True when ``other`` appears in this task's suggestion list.
        :rtype: bool
        """
        compatible_tasks = self.get_compatible_tasks(self)
        return other in compatible_tasks

    def __str__(self) -> str:
        """
        Return the member's string value.

        This is the serialization form :meth:`from_string` parses.

        :return: The lowercase task name, for example ``'depth_estimation'``.
        :rtype: str
        """
        return self.value

    def __repr__(self) -> str:
        """
        Return the qualified member name.

        :return: A string of the form ``VisionTaskType.DEPTH_ESTIMATION``.
        :rtype: str
        """
        return f"VisionTaskType.{self.name}"


# ---------------------------------------------------------------------

class TaskConfiguration:
    """
    A validated set of tasks for a multi-task vision model.

    Construction checks the list, then stores it as a set. The query methods
    answer what that set contains: a named task, one of nine specific tasks,
    a single task or several, and the set grouped by category.

    Only two task pairs are rejected as incompatible. The check is a guard
    against two known-bad combinations, not a full compatibility model, and
    it never consults :meth:`VisionTaskType.get_compatible_tasks`.

    **Architecture Overview:**

    .. code-block:: text

        tasks: List[VisionTaskType]
                  │
                  ▼
        ┌───────────────────────┐         ┌────────────┐
        │ empty or duplicates?  │─ yes ──►│ ValueError │
        └──────────┬────────────┘         └────────────┘
                   │ no
                   ▼
        ┌───────────────────────┐         ┌────────────┐
        │ known-bad pair, and   │─ yes ──►│ ValueError │
        │ validation enabled?   │         └────────────┘
        └──────────┬────────────┘
                   │ no
                   ▼
             self._tasks: Set[VisionTaskType]
                   │
                   ├─► has_task() and the 9 has_*() predicates
                   ├─► is_single_task() / is_multi_task()
                   ├─► get_enabled_tasks() / get_task_names()
                   ├─► get_tasks_by_category()
                   ├─► get_output_specifications()
                   └─► to_dict()

    :param tasks: List of VisionTaskType enum values to enable.
    :type tasks: List[VisionTaskType]
    :param validate_compatibility: Whether to check task compatibility. The
        check only runs when more than one task is given.
    :type validate_compatibility: bool
    :raises ValueError: If tasks list is empty, contains duplicates, or
        contains an incompatible pair (when validation is enabled).
    """

    def __init__(self, tasks: List[VisionTaskType], validate_compatibility: bool = True):
        """
        Initialize task configuration.

        :param tasks: List of VisionTaskType enum values to enable.
        :type tasks: List[VisionTaskType]
        :param validate_compatibility: Whether to validate task compatibility.
        :type validate_compatibility: bool
        :raises ValueError: If tasks list is empty, contains duplicates, or
            contains incompatible tasks (when validation enabled).
        """
        if not tasks:
            raise ValueError("At least one task must be specified")

        # Check for duplicates
        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate tasks found in configuration")

        self._tasks: Set[VisionTaskType] = set(tasks)

        if validate_compatibility and len(tasks) > 1:
            self._validate_task_compatibility()

    def _validate_task_compatibility(self) -> None:
        """
        Reject the two task pairs known not to work together.

        Colorization and denoising want different inputs, as do stereo
        matching and optical flow. Every other combination passes.

        :return: Nothing.
        :rtype: None
        :raises ValueError: If both members of a rejected pair are enabled.
        """
        task_list = list(self._tasks)

        # Check for obviously incompatible combinations
        incompatible_pairs = [
            (VisionTaskType.COLORIZATION, VisionTaskType.DENOISING),
            (VisionTaskType.STEREO_MATCHING, VisionTaskType.OPTICAL_FLOW),
        ]

        for task1, task2 in incompatible_pairs:
            if task1 in self._tasks and task2 in self._tasks:
                raise ValueError(f"Tasks {task1} and {task2} are incompatible")

    @property
    def tasks(self) -> Set[VisionTaskType]:
        """
        The set of enabled tasks.

        A copy is returned, so mutating it does not change the
        configuration.

        :return: A copy of the enabled task set.
        :rtype: Set[VisionTaskType]
        """
        return self._tasks.copy()

    def has_task(self, task: VisionTaskType) -> bool:
        """
        Report whether one named task is enabled.

        The nine ``has_*`` methods below are fixed-task shorthands for this.

        :param task: The task to look for.
        :type task: VisionTaskType
        :return: True when the task is in this configuration.
        :rtype: bool
        """
        return task in self._tasks

    # Core task checks
    def has_detection(self) -> bool:
        """
        Report whether object detection is enabled.

        :return: True when ``VisionTaskType.DETECTION`` is in the set.
        :rtype: bool
        """
        return VisionTaskType.DETECTION in self._tasks

    def has_segmentation(self) -> bool:
        """
        Report whether semantic segmentation is enabled.

        :return: True when ``VisionTaskType.SEGMENTATION`` is in the set.
        :rtype: bool
        """
        return VisionTaskType.SEGMENTATION in self._tasks

    def has_classification(self) -> bool:
        """
        Report whether image-level classification is enabled.

        :return: True when ``VisionTaskType.CLASSIFICATION`` is in the set.
        :rtype: bool
        """
        return VisionTaskType.CLASSIFICATION in self._tasks

    # Geometric task checks
    def has_depth_estimation(self) -> bool:
        """
        Report whether depth estimation is enabled.

        :return: True when ``VisionTaskType.DEPTH_ESTIMATION`` is in the set.
        :rtype: bool
        """
        return VisionTaskType.DEPTH_ESTIMATION in self._tasks

    def has_surface_normals(self) -> bool:
        """
        Report whether surface normal estimation is enabled.

        :return: True when ``VisionTaskType.SURFACE_NORMALS`` is in the set.
        :rtype: bool
        """
        return VisionTaskType.SURFACE_NORMALS in self._tasks

    # Instance segmentation checks
    def has_instance_segmentation(self) -> bool:
        """
        Report whether instance segmentation is enabled.

        :return: True when ``VisionTaskType.INSTANCE_SEGMENTATION`` is in the
            set.
        :rtype: bool
        """
        return VisionTaskType.INSTANCE_SEGMENTATION in self._tasks

    def has_panoptic_segmentation(self) -> bool:
        """
        Report whether panoptic segmentation is enabled.

        This task has no head. ``create_vision_head`` raises ``ValueError``
        for it, so a True answer here does not mean a head can be built.

        :return: True when ``VisionTaskType.PANOPTIC_SEGMENTATION`` is in the
            set.
        :rtype: bool
        """
        return VisionTaskType.PANOPTIC_SEGMENTATION in self._tasks

    # Enhancement task checks
    def has_denoising(self) -> bool:
        """
        Report whether denoising is enabled.

        :return: True when ``VisionTaskType.DENOISING`` is in the set.
        :rtype: bool
        """
        return VisionTaskType.DENOISING in self._tasks

    def has_super_resolution(self) -> bool:
        """
        Report whether super-resolution is enabled.

        :return: True when ``VisionTaskType.SUPER_RESOLUTION`` is in the set.
        :rtype: bool
        """
        return VisionTaskType.SUPER_RESOLUTION in self._tasks

    def is_single_task(self) -> bool:
        """
        Report whether exactly one task is enabled.

        :return: True when the set holds one task.
        :rtype: bool
        """
        return len(self._tasks) == 1

    def is_multi_task(self) -> bool:
        """
        Report whether more than one task is enabled.

        :return: True when the set holds two or more tasks.
        :rtype: bool
        """
        return len(self._tasks) > 1

    def get_enabled_tasks(self) -> List[VisionTaskType]:
        """
        List the enabled tasks in enum declaration order.

        The set itself has no order. Sorting by
        :meth:`VisionTaskType.all_tasks` makes the result reproducible across
        runs, which matters for naming multi-task outputs.

        :return: The enabled tasks, in declaration order.
        :rtype: List[VisionTaskType]
        """
        all_tasks = VisionTaskType.all_tasks()
        return [task for task in all_tasks if task in self._tasks]

    def get_task_names(self) -> List[str]:
        """
        List the enabled task names as strings.

        Same order as :meth:`get_enabled_tasks`.

        :return: The ``value`` of each enabled task.
        :rtype: List[str]
        """
        return VisionTaskType.to_strings(self.get_enabled_tasks())

    def get_tasks_by_category(self) -> Dict[str, List[VisionTaskType]]:
        """
        Group the enabled tasks by category.

        Categories with no enabled task are left out, so the result has at
        most twelve keys and never an empty list.

        :return: Category name mapped to the enabled tasks filed under it.
        :rtype: Dict[str, List[VisionTaskType]]
        """
        categories = VisionTaskType.get_task_categories()
        result = {}

        for category_name, category_tasks in categories.items():
            enabled_in_category = [task for task in category_tasks if task in self._tasks]
            if enabled_in_category:
                result[category_name] = enabled_in_category

        return result

    def get_output_specifications(self) -> Dict[VisionTaskType, Dict[str, str]]:
        """
        Describe the outputs of every enabled task.

        Each value comes from :meth:`VisionTaskType.get_output_types`, so it
        is documentation, not a runtime shape check.

        :return: Task mapped to its output name/shape strings.
        :rtype: Dict[VisionTaskType, Dict[str, str]]
        """
        return {task: VisionTaskType.get_output_types(task) for task in self._tasks}

    def to_dict(self) -> dict:
        """
        Convert the configuration to a flat boolean dictionary.

        All 37 tasks get a key, named ``enable_<task value>``, whether
        enabled or not. :meth:`from_dict` reads this form back.

        :return: 37 ``enable_*`` keys mapped to True or False.
        :rtype: dict
        """
        result = {}
        for task in VisionTaskType.all_tasks():
            result[f"enable_{task.value}"] = task in self._tasks
        return result

    @classmethod
    def from_dict(cls, config_dict: dict, validate_compatibility: bool = True) -> "TaskConfiguration":
        """
        Build a configuration from a boolean dictionary.

        Reads the ``enable_<task value>`` keys :meth:`to_dict` writes. A
        missing key counts as False. A dictionary with no key set to True
        produces an empty task list, and construction then raises.

        :param config_dict: Dictionary with boolean flags for tasks.
        :type config_dict: dict
        :param validate_compatibility: Whether to validate task compatibility.
        :type validate_compatibility: bool
        :return: TaskConfiguration instance.
        :rtype: TaskConfiguration
        :raises ValueError: If no task flag is True, or the enabled tasks
            contain an incompatible pair.
        """
        tasks = []

        for task in VisionTaskType.all_tasks():
            key = f"enable_{task.value}"
            if config_dict.get(key, False):
                tasks.append(task)

        return cls(tasks, validate_compatibility=validate_compatibility)

    @classmethod
    def from_strings(cls, task_strings: List[str], validate_compatibility: bool = True) -> "TaskConfiguration":
        """
        Build a configuration from a list of task name strings.

        Parses through :meth:`VisionTaskType.from_strings`, so an unknown
        name raises before the configuration is built.

        :param task_strings: List of task names as strings.
        :type task_strings: List[str]
        :param validate_compatibility: Whether to validate task compatibility.
        :type validate_compatibility: bool
        :return: TaskConfiguration instance.
        :rtype: TaskConfiguration
        :raises ValueError: If a name is unknown, the list is empty, holds
            duplicates, or holds an incompatible pair.
        """
        tasks = VisionTaskType.from_strings(task_strings)
        return cls(tasks, validate_compatibility=validate_compatibility)

    def __str__(self) -> str:
        """
        Return a short readable summary listing the task values.

        :return: A string such as ``TaskConfiguration(detection,
            segmentation)``.
        :rtype: str
        """
        task_names = self.get_task_names()
        return f"TaskConfiguration({', '.join(task_names)})"

    def __repr__(self) -> str:
        """
        Return a summary naming the enum members.

        :return: A string such as
            ``TaskConfiguration([VisionTaskType.DETECTION])``.
        :rtype: str
        """
        tasks_repr = [repr(task) for task in self.get_enabled_tasks()]
        return f"TaskConfiguration([{', '.join(tasks_repr)}])"

    def __eq__(self, other) -> bool:
        """
        Compare two configurations by their task sets.

        Order does not matter and neither does
        ``validate_compatibility``: two configurations holding the same tasks
        are equal.

        :param other: The object to compare against.
        :type other: object
        :return: True when ``other`` is a TaskConfiguration with the same
            task set.
        :rtype: bool
        """
        if not isinstance(other, TaskConfiguration):
            return False
        return self._tasks == other._tasks

    def __hash__(self) -> int:
        """
        Hash the task set, so configurations can be dictionary keys.

        Consistent with :meth:`__eq__`: equal configurations hash equal.

        :return: The hash of the frozen task set.
        :rtype: int
        """
        return hash(frozenset(self._tasks))


# ---------------------------------------------------------------------

# Predefined common task configurations
class CommonTaskConfigurations:
    """
    26 ready-made :class:`TaskConfiguration` presets.

    Every attribute is a class-level instance, built once at import time.
    They are shared, so treat them as read-only; :attr:`TaskConfiguration.tasks`
    already returns a copy.

    **Presets:**

    The table lists each preset's task count, how many of those tasks
    ``create_vision_head`` can actually build a head for, and how many
    categories the tasks span. ``Heads`` below ``N`` means the preset names a
    task with no head.

    .. code-block:: text

        Preset                                  N  Heads  Cats
        -------------------------------------  --  -----  ----
        DETECTION_ONLY                          1      1     1
        SEGMENTATION_ONLY                       1      1     1
        CLASSIFICATION_ONLY                     1      1     1
        DEPTH_ONLY                              1      1     1
        SURFACE_NORMALS_ONLY                    1      1     1
        INSTANCE_SEGMENTATION_ONLY              1      1     1
        PANOPTIC_SEGMENTATION_ONLY              1      0     1
        DENOISING_ONLY                          1      1     1
        SUPER_RESOLUTION_ONLY                   1      1     1
        KEYPOINT_DETECTION_ONLY                 1      1     1
        DETECTION_SEGMENTATION                  2      2     1
        DETECTION_CLASSIFICATION                2      2     1
        SEGMENTATION_CLASSIFICATION             2      2     1
        DEPTH_NORMALS                           2      2     1
        SEGMENTATION_DEPTH                      2      2     2
        DETECTION_DEPTH                         2      2     2
        DETECTION_INSTANCE_SEG                  2      2     1
        SEGMENTATION_INSTANCE_SEG               2      2     1
        DETECTION_SEGMENTATION_DEPTH            3      3     2
        DETECTION_SEGMENTATION_CLASSIFICATION   3      3     1
        GEOMETRIC_UNDERSTANDING                 3      2     2
        PANOPTIC_UNDERSTANDING                  4      4     2
        IMAGE_ENHANCEMENT                       3      2     1
        POSE_AND_STRUCTURE                      3      1     1
        ALL_CORE_TASKS                          5      5     2
        ALL_TASKS                              37     10    12

        N      = len(get_enabled_tasks())
        Heads  = tasks create_vision_head dispatches
        Cats   = len(get_tasks_by_category())

    ``ALL_TASKS`` turns compatibility validation off, because the full enum
    contains both rejected pairs. Besides it, four presets name a task with
    no head:
    ``PANOPTIC_SEGMENTATION_ONLY``, ``GEOMETRIC_UNDERSTANDING``,
    ``IMAGE_ENHANCEMENT`` and ``POSE_AND_STRUCTURE``.

    Note:
        The two listing methods do not cover all 26.
        :meth:`get_all_configurations` returns 25, leaving out ``ALL_TASKS``.
        :meth:`get_configurations_by_complexity` returns 24, leaving out
        ``ALL_TASKS`` and ``PANOPTIC_SEGMENTATION_ONLY``.

    Example:
        >>> cfg = CommonTaskConfigurations.DETECTION_SEGMENTATION_DEPTH
        >>> len(cfg.tasks)
        3
        >>> cfg.has_depth_estimation()
        True
    """

    # Single task configurations - Core tasks
    DETECTION_ONLY = TaskConfiguration([VisionTaskType.DETECTION])
    SEGMENTATION_ONLY = TaskConfiguration([VisionTaskType.SEGMENTATION])
    CLASSIFICATION_ONLY = TaskConfiguration([VisionTaskType.CLASSIFICATION])
    DEPTH_ONLY = TaskConfiguration([VisionTaskType.DEPTH_ESTIMATION])
    SURFACE_NORMALS_ONLY = TaskConfiguration([VisionTaskType.SURFACE_NORMALS])

    # Single task configurations - Specialized
    INSTANCE_SEGMENTATION_ONLY = TaskConfiguration([VisionTaskType.INSTANCE_SEGMENTATION])
    PANOPTIC_SEGMENTATION_ONLY = TaskConfiguration([VisionTaskType.PANOPTIC_SEGMENTATION])
    DENOISING_ONLY = TaskConfiguration([VisionTaskType.DENOISING])
    SUPER_RESOLUTION_ONLY = TaskConfiguration([VisionTaskType.SUPER_RESOLUTION])
    KEYPOINT_DETECTION_ONLY = TaskConfiguration([VisionTaskType.KEYPOINT_DETECTION])

    # Two-task combinations - Core
    DETECTION_SEGMENTATION = TaskConfiguration([VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION])
    DETECTION_CLASSIFICATION = TaskConfiguration([VisionTaskType.DETECTION, VisionTaskType.CLASSIFICATION])
    SEGMENTATION_CLASSIFICATION = TaskConfiguration([VisionTaskType.SEGMENTATION, VisionTaskType.CLASSIFICATION])

    # Two-task combinations - Geometric
    DEPTH_NORMALS = TaskConfiguration([VisionTaskType.DEPTH_ESTIMATION, VisionTaskType.SURFACE_NORMALS])
    SEGMENTATION_DEPTH = TaskConfiguration([VisionTaskType.SEGMENTATION, VisionTaskType.DEPTH_ESTIMATION])
    DETECTION_DEPTH = TaskConfiguration([VisionTaskType.DETECTION, VisionTaskType.DEPTH_ESTIMATION])

    # Two-task combinations - Instance segmentation
    DETECTION_INSTANCE_SEG = TaskConfiguration([VisionTaskType.DETECTION, VisionTaskType.INSTANCE_SEGMENTATION])
    SEGMENTATION_INSTANCE_SEG = TaskConfiguration([VisionTaskType.SEGMENTATION, VisionTaskType.INSTANCE_SEGMENTATION])

    # Three-task combinations
    DETECTION_SEGMENTATION_DEPTH = TaskConfiguration([
        VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION, VisionTaskType.DEPTH_ESTIMATION
    ])
    DETECTION_SEGMENTATION_CLASSIFICATION = TaskConfiguration([
        VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION, VisionTaskType.CLASSIFICATION
    ])
    GEOMETRIC_UNDERSTANDING = TaskConfiguration([
        VisionTaskType.DEPTH_ESTIMATION, VisionTaskType.SURFACE_NORMALS, VisionTaskType.EDGE_DETECTION
    ])

    # Panoptic understanding (full scene parsing)
    PANOPTIC_UNDERSTANDING = TaskConfiguration([
        VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION, VisionTaskType.INSTANCE_SEGMENTATION, VisionTaskType.DEPTH_ESTIMATION
    ])

    # Enhancement pipeline
    IMAGE_ENHANCEMENT = TaskConfiguration([
        VisionTaskType.DENOISING, VisionTaskType.SUPER_RESOLUTION, VisionTaskType.DEHAZE
    ])

    # Pose and structure
    POSE_AND_STRUCTURE = TaskConfiguration([
        VisionTaskType.POSE_ESTIMATION, VisionTaskType.KEYPOINT_DETECTION, VisionTaskType.EDGE_DETECTION
    ])

    # All core tasks
    ALL_CORE_TASKS = TaskConfiguration([
        VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION, VisionTaskType.CLASSIFICATION,
        VisionTaskType.DEPTH_ESTIMATION, VisionTaskType.SURFACE_NORMALS
    ])

    # All tasks (be careful with this one!)
    ALL_TASKS = TaskConfiguration(VisionTaskType.all_tasks(), validate_compatibility=False)

    @classmethod
    def get_all_configurations(cls) -> List[TaskConfiguration]:
        """
        List the presets, except ``ALL_TASKS``.

        25 of the 26 presets are returned. ``ALL_TASKS`` is left out because
        it enables every task and is not a configuration to iterate over.
        The instances are the shared class attributes, not copies.

        :return: 25 predefined TaskConfiguration instances.
        :rtype: List[TaskConfiguration]
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
        Group presets under four complexity labels.

        The labels are "Single Task" (9), "Two Tasks" (8), "Three Tasks" (5)
        and "Complex Multi-Task" (2), so 24 of the 26 presets appear.
        ``ALL_TASKS`` and ``PANOPTIC_SEGMENTATION_ONLY`` are not listed.
        The labels describe task count, not head count.

        :return: Complexity label mapped to its preset instances.
        :rtype: Dict[str, List[TaskConfiguration]]
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
    Normalize any accepted task spelling into a TaskConfiguration.

    Five input forms are accepted: an existing configuration, a single enum
    member, a single task name, a list of enum members, and a list of names.
    An existing configuration is returned unchanged, so
    ``validate_compatibility`` is ignored in that case.

    A mixed list is not supported. Only the first element decides how the
    whole list is read. A list starting with an enum but holding a string is
    stored as given, raising nothing here and failing wherever the string is
    later used as a task.

    :param tasks: Task specification in any supported format.
    :type tasks: Union[TaskConfiguration, VisionTaskType, str,
        List[VisionTaskType], List[str]]
    :param validate_compatibility: Whether to validate task compatibility.
    :type validate_compatibility: bool
    :return: TaskConfiguration instance.
    :rtype: TaskConfiguration
    :raises ValueError: If the task format is invalid or the task list is empty.
    """
    if isinstance(tasks, TaskConfiguration):
        return tasks
    elif isinstance(tasks, VisionTaskType):
        return TaskConfiguration([tasks], validate_compatibility=validate_compatibility)
    elif isinstance(tasks, str):
        return TaskConfiguration([VisionTaskType.from_string(tasks)], validate_compatibility=validate_compatibility)
    elif isinstance(tasks, (list, tuple)):
        if not tasks:
            raise ValueError("Task list cannot be empty")

        # Check if first element is string or VisionTaskType
        if isinstance(tasks[0], str):
            return TaskConfiguration.from_strings(list(tasks), validate_compatibility=validate_compatibility)
        elif isinstance(tasks[0], VisionTaskType):
            return TaskConfiguration(list(tasks), validate_compatibility=validate_compatibility)
        else:
            raise ValueError(f"Invalid task type in list: {type(tasks[0])}")
    else:
        raise ValueError(f"Invalid tasks format: {type(tasks)}")


def get_task_suggestions(base_task: VisionTaskType, max_suggestions: int = 5) -> List[VisionTaskType]:
    """
    Suggest tasks that are commonly trained with a base task.

    A thin cap over :meth:`VisionTaskType.get_compatible_tasks`. Only 11 of
    the 37 members have suggestions, so an empty result is common and means
    "none recorded".

    :param base_task: The base task to find compatible tasks for.
    :type base_task: VisionTaskType
    :param max_suggestions: Maximum number of suggestions to return.
    :type max_suggestions: int
    :return: Up to ``max_suggestions`` compatible tasks, possibly empty.
    :rtype: List[VisionTaskType]
    """
    compatible_tasks = VisionTaskType.get_compatible_tasks(base_task)
    return compatible_tasks[:max_suggestions]


def validate_task_combination(tasks: List[VisionTaskType]) -> tuple[bool, Optional[str]]:
    """
    Check a task list without raising.

    Builds a :class:`TaskConfiguration` with validation on and converts the
    ``ValueError`` into a return value. The message is the exception's text.
    Empty lists and duplicates are rejected here too, not just incompatible
    pairs.

    :param tasks: List of tasks to validate.
    :type tasks: List[VisionTaskType]
    :return: ``(True, None)`` when the list is accepted, otherwise
        ``(False, message)``.
    :rtype: tuple[bool, Optional[str]]
    """
    try:
        TaskConfiguration(tasks, validate_compatibility=True)
        return True, None
    except ValueError as e:
        return False, str(e)


# ---------------------------------------------------------------------

# DECISION plan_2026-06-08_8b32ca51/D-003: `TaskType` is a backward-compat alias
# for the renamed `VisionTaskType`. Do NOT delete it as "dead code": an
# out-of-tree caller may still import `TaskType` from here (the 4 in-tree callers
# were migrated). It is the SAME object, so `isinstance` and `is` checks hold
# across both names. The owning plan is gone; this comment is the only record.
TaskType = VisionTaskType

# ---------------------------------------------------------------------