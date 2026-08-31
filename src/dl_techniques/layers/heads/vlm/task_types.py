"""Task types and configuration objects for the VLM heads.

This module says what a vision-language task is. It builds no layer. Three
classes live here:

* :class:`VLMTaskType` -- an enum of 47 task names, with a category listing
  and string parsing.
* :class:`VLMTaskConfig` -- a dataclass holding the hyperparameters of one
  VLM task head.
* :class:`VLMTaskConfiguration` -- a validated set of tasks, plus a query for
  whether that set needs text generation.

``vlm/factory.py`` reads these objects and picks the head class to build.
Only 4 of the 47 task names reach a real head. Two more are mapped to
``BaseVLMHead`` and then rejected by the same function, because
``BaseVLMHead`` defines no ``call()``. The other 41 raise ``ValueError``.
The table in :class:`VLMTaskType` lists all three groups.
"""

from enum import Enum, unique
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set


# ---------------------------------------------------------------------


@unique
class VLMTaskType(Enum):
    """
    The 47 vision-language tasks a head can be asked to serve.

    Each member carries a lowercase string value. That string is the
    serialization form, and :meth:`from_string` parses it back.

    **Head dispatch:**

    ``vlm/factory.py``'s ``get_head_class`` decides which head class runs a
    task. Four members reach a real head. Two are mapped to ``BaseVLMHead``
    in that function's own table and are then rejected by it, because
    ``BaseVLMHead`` defines no ``call()``. The remaining 41 are absent from
    the table. Both failing groups raise ``ValueError`` when the head is
    created, not when it is called. The three groups sum to 47.

    .. code-block:: text

        Has a head class
        -------------------------  ----------------------
        IMAGE_CAPTIONING           ImageCaptioningHead
        VISUAL_QUESTION_ANSWERING  VQAHead
        VISUAL_GROUNDING           VisualGroundingHead
        IMAGE_TEXT_MATCHING        ImageTextMatchingHead

        Mapped to BaseVLMHead, then rejected -- 2
        --------------------------------------------
        DENSE_CAPTIONING, VISUAL_DIALOGUE

        Not in the table -- raises ValueError -- 41
        --------------------------------------------
        VISUAL_STORYTELLING, IMAGE_PARAGRAPH_CAPTIONING,
        VISUAL_REASONING, VISUAL_COMMONSENSE_REASONING,
        CHART_QUESTION_ANSWERING, DIAGRAM_UNDERSTANDING,
        REFERRING_EXPRESSION_COMPREHENSION,
        REFERRING_EXPRESSION_GENERATION, PHRASE_GROUNDING,
        IMAGE_RETRIEVAL, TEXT_RETRIEVAL, CROSS_MODAL_RETRIEVAL,
        EMBODIED_QUESTION_ANSWERING, VISUAL_CHAT,
        OPTICAL_CHARACTER_RECOGNITION, SCENE_TEXT_RECOGNITION,
        DOCUMENT_UNDERSTANDING, TABLE_UNDERSTANDING,
        FORM_UNDERSTANDING, VISUAL_ENTAILMENT, VISUAL_INFERENCE,
        FACT_VERIFICATION, TEXT_TO_IMAGE_GENERATION,
        IMAGE_EDITING_INSTRUCTION, VISUAL_INSTRUCTION_FOLLOWING,
        IMAGE_MANIPULATION_GUIDANCE,
        MULTIMODAL_SENTIMENT_ANALYSIS,
        MULTIMODAL_EMOTION_RECOGNITION, MEME_UNDERSTANDING,
        VISUAL_METAPHOR_UNDERSTANDING, VIDEO_CAPTIONING,
        VIDEO_QUESTION_ANSWERING, VIDEO_SUMMARIZATION,
        TEMPORAL_GROUNDING, ACTION_RECOGNITION, MEDICAL_VQA,
        MEDICAL_REPORT_GENERATION,
        SCIENTIFIC_FIGURE_UNDERSTANDING,
        VISUAL_INSTRUCTION_GENERATION,
        EDUCATIONAL_CONTENT_UNDERSTANDING, DIAGRAM_TO_TEXT

    **Categories:**

    :meth:`get_task_categories` files every member into one of twelve named
    categories, listed here with the number of members in each. The filing
    is descriptive. It does not decide which head class runs a task, and the
    factory never reads it.

    .. code-block:: text

        Image Understanding & Description   4
        Visual Question Answering           5
        Visual Grounding & Localization     4
        Image-Text Matching & Retrieval     4
        Visual Dialogue & Interaction       3
        OCR & Document Understanding        5
        Visual Entailment & Inference       3
        Multi-modal Generation              4
        Multi-modal Classification          4
        Video Understanding                 5
        Medical & Scientific                3
        Educational & Instructional         3
                                           --
        total                              47

    Note:
        The dispatch table restates a mapping that lives in
        ``get_head_class``. That function is the arbiter. If the two
        disagree, the function is right.

    Example:
        >>> task = VLMTaskType.from_string("visual_grounding")
        >>> str(task)
        'visual_grounding'
        >>> repr(task)
        'VLMTaskType.VISUAL_GROUNDING'
    """

    # Image Understanding & Description
    IMAGE_CAPTIONING = "image_captioning"
    DENSE_CAPTIONING = "dense_captioning"
    VISUAL_STORYTELLING = "visual_storytelling"
    IMAGE_PARAGRAPH_CAPTIONING = "image_paragraph_captioning"

    # Visual Question Answering
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    VISUAL_REASONING = "visual_reasoning"
    VISUAL_COMMONSENSE_REASONING = "visual_commonsense_reasoning"
    CHART_QUESTION_ANSWERING = "chart_question_answering"
    DIAGRAM_UNDERSTANDING = "diagram_understanding"

    # Visual Grounding & Localization
    VISUAL_GROUNDING = "visual_grounding"
    REFERRING_EXPRESSION_COMPREHENSION = "referring_expression_comprehension"
    REFERRING_EXPRESSION_GENERATION = "referring_expression_generation"
    PHRASE_GROUNDING = "phrase_grounding"

    # Image-Text Matching & Retrieval
    IMAGE_TEXT_MATCHING = "image_text_matching"
    IMAGE_RETRIEVAL = "image_retrieval"
    TEXT_RETRIEVAL = "text_retrieval"
    CROSS_MODAL_RETRIEVAL = "cross_modal_retrieval"

    # Visual Dialogue & Interaction
    VISUAL_DIALOGUE = "visual_dialogue"
    EMBODIED_QUESTION_ANSWERING = "embodied_question_answering"
    VISUAL_CHAT = "visual_chat"

    # OCR & Document Understanding
    OPTICAL_CHARACTER_RECOGNITION = "optical_character_recognition"
    SCENE_TEXT_RECOGNITION = "scene_text_recognition"
    DOCUMENT_UNDERSTANDING = "document_understanding"
    TABLE_UNDERSTANDING = "table_understanding"
    FORM_UNDERSTANDING = "form_understanding"

    # Visual Entailment & Inference
    VISUAL_ENTAILMENT = "visual_entailment"
    VISUAL_INFERENCE = "visual_inference"
    FACT_VERIFICATION = "fact_verification"

    # Multi-modal Generation
    TEXT_TO_IMAGE_GENERATION = "text_to_image_generation"
    IMAGE_EDITING_INSTRUCTION = "image_editing_instruction"
    VISUAL_INSTRUCTION_FOLLOWING = "visual_instruction_following"
    IMAGE_MANIPULATION_GUIDANCE = "image_manipulation_guidance"

    # Multi-modal Classification
    MULTIMODAL_SENTIMENT_ANALYSIS = "multimodal_sentiment_analysis"
    MULTIMODAL_EMOTION_RECOGNITION = "multimodal_emotion_recognition"
    MEME_UNDERSTANDING = "meme_understanding"
    VISUAL_METAPHOR_UNDERSTANDING = "visual_metaphor_understanding"

    # Video Understanding
    VIDEO_CAPTIONING = "video_captioning"
    VIDEO_QUESTION_ANSWERING = "video_question_answering"
    VIDEO_SUMMARIZATION = "video_summarization"
    TEMPORAL_GROUNDING = "temporal_grounding"
    ACTION_RECOGNITION = "action_recognition"

    # Medical & Scientific
    MEDICAL_VQA = "medical_vqa"
    MEDICAL_REPORT_GENERATION = "medical_report_generation"
    SCIENTIFIC_FIGURE_UNDERSTANDING = "scientific_figure_understanding"

    # Educational & Instructional
    VISUAL_INSTRUCTION_GENERATION = "visual_instruction_generation"
    EDUCATIONAL_CONTENT_UNDERSTANDING = "educational_content_understanding"
    DIAGRAM_TO_TEXT = "diagram_to_text"

    @classmethod
    def all_tasks(cls) -> List["VLMTaskType"]:
        """
        List every task type, in declaration order.

        :return: All 47 members of the enum.
        :rtype: List[VLMTaskType]
        """
        return list(cls)

    @classmethod
    def get_task_categories(cls) -> Dict[str, List["VLMTaskType"]]:
        """
        Group the members into twelve named categories.

        The grouping is descriptive only. ``vlm/factory.py`` never reads it.
        Every member is filed exactly once, so the twelve lists hold 47
        members in total. The per-category counts are in the class
        docstring.

        :return: Category name mapped to the members filed under it.
        :rtype: Dict[str, List[VLMTaskType]]
        """
        return {
            "Image Understanding & Description": [
                cls.IMAGE_CAPTIONING,
                cls.DENSE_CAPTIONING,
                cls.VISUAL_STORYTELLING,
                cls.IMAGE_PARAGRAPH_CAPTIONING,
            ],
            "Visual Question Answering": [
                cls.VISUAL_QUESTION_ANSWERING,
                cls.VISUAL_REASONING,
                cls.VISUAL_COMMONSENSE_REASONING,
                cls.CHART_QUESTION_ANSWERING,
                cls.DIAGRAM_UNDERSTANDING,
            ],
            "Visual Grounding & Localization": [
                cls.VISUAL_GROUNDING,
                cls.REFERRING_EXPRESSION_COMPREHENSION,
                cls.REFERRING_EXPRESSION_GENERATION,
                cls.PHRASE_GROUNDING,
            ],
            "Image-Text Matching & Retrieval": [
                cls.IMAGE_TEXT_MATCHING,
                cls.IMAGE_RETRIEVAL,
                cls.TEXT_RETRIEVAL,
                cls.CROSS_MODAL_RETRIEVAL,
            ],
            "Visual Dialogue & Interaction": [
                cls.VISUAL_DIALOGUE,
                cls.EMBODIED_QUESTION_ANSWERING,
                cls.VISUAL_CHAT,
            ],
            "OCR & Document Understanding": [
                cls.OPTICAL_CHARACTER_RECOGNITION,
                cls.SCENE_TEXT_RECOGNITION,
                cls.DOCUMENT_UNDERSTANDING,
                cls.TABLE_UNDERSTANDING,
                cls.FORM_UNDERSTANDING,
            ],
            "Visual Entailment & Inference": [
                cls.VISUAL_ENTAILMENT,
                cls.VISUAL_INFERENCE,
                cls.FACT_VERIFICATION,
            ],
            "Multi-modal Generation": [
                cls.TEXT_TO_IMAGE_GENERATION,
                cls.IMAGE_EDITING_INSTRUCTION,
                cls.VISUAL_INSTRUCTION_FOLLOWING,
                cls.IMAGE_MANIPULATION_GUIDANCE,
            ],
            "Multi-modal Classification": [
                cls.MULTIMODAL_SENTIMENT_ANALYSIS,
                cls.MULTIMODAL_EMOTION_RECOGNITION,
                cls.MEME_UNDERSTANDING,
                cls.VISUAL_METAPHOR_UNDERSTANDING,
            ],
            "Video Understanding": [
                cls.VIDEO_CAPTIONING,
                cls.VIDEO_QUESTION_ANSWERING,
                cls.VIDEO_SUMMARIZATION,
                cls.TEMPORAL_GROUNDING,
                cls.ACTION_RECOGNITION,
            ],
            "Medical & Scientific": [
                cls.MEDICAL_VQA,
                cls.MEDICAL_REPORT_GENERATION,
                cls.SCIENTIFIC_FIGURE_UNDERSTANDING,
            ],
            "Educational & Instructional": [
                cls.VISUAL_INSTRUCTION_GENERATION,
                cls.EDUCATIONAL_CONTENT_UNDERSTANDING,
                cls.DIAGRAM_TO_TEXT,
            ],
        }

    @classmethod
    def from_string(cls, task_str: str) -> "VLMTaskType":
        """
        Parse a task type from its string value.

        The input is lowercased and stripped first, so ``" Visual_Grounding "``
        resolves to :attr:`VISUAL_GROUNDING`.

        :param task_str: A task string value, such as ``"image_captioning"``.
        :type task_str: str
        :return: The member carrying that value.
        :rtype: VLMTaskType
        :raises ValueError: If no member carries that value. The message
            lists all 47 valid strings.
        """
        task_str = task_str.lower().strip()
        for task in cls:
            if task.value == task_str:
                return task

        valid_tasks = [task.value for task in cls]
        raise ValueError(
            f"Invalid task type: '{task_str}'. " f"Valid options are: {valid_tasks}"
        )

    def __str__(self) -> str:
        """
        Return the task's string value.

        This is the serialization form :meth:`from_string` reads back.

        :return: The member's lowercase value, such as ``"visual_grounding"``.
        :rtype: str
        """
        return self.value

    def __repr__(self) -> str:
        """
        Return the qualified member name.

        :return: The member as ``VLMTaskType.NAME``.
        :rtype: str
        """
        return f"VLMTaskType.{self.name}"


# ---------------------------------------------------------------------


@dataclass
class VLMTaskConfig:
    """
    The hyperparameters of one VLM task head.

    A dataclass. It holds vocabulary, dimension, fusion, pooling and
    generation settings, and nothing else. ``vlm/factory.py``'s
    ``create_vlm_head`` takes one of these and builds the matching head.
    Only ``name`` and ``task_type`` are required.

    :meth:`__post_init__` fills in three fields that are left at ``None``.
    Note that ``hidden_size`` and ``fusion_hidden_size`` default to 768, not
    to ``None``, so a caller has to pass ``None`` by hand to reach those two
    fixups. ``num_classes`` does default to ``None`` and is filled for three
    task types.

    **Defaults:**

    .. code-block:: text

        Field                    Default
        -----------------------  -----------
        name                     required
        task_type                required
        vocab_size               50000
        max_text_length          512
        hidden_size              768
        vision_hidden_size       768
        text_hidden_size         768
        fusion_hidden_size       768
        dropout_rate             0.1
        num_classes              None
        use_cross_attention      True
        fusion_type              "attention"
        pooling_type             "avg"
        use_task_specific_heads  True
        temperature              1.0
        beam_size                1
        loss_weight              1.0

    Note:
        This class does not check that ``task_type`` has a head. That check
        happens later, in ``get_head_class``. Building a config for one of
        the 43 unsupported task types succeeds; creating its head raises.

    Example:
        >>> config = VLMTaskConfig(
        ...     name="caption",
        ...     task_type=VLMTaskType.IMAGE_CAPTIONING,
        ... )
        >>> config.fusion_hidden_size
        768

    :param name: Unique identifier for the task. Multi-task heads key their
        sub-heads on it.
    :type name: str
    :param task_type: Which VLM task this head serves.
    :type task_type: VLMTaskType
    :param vocab_size: Size of the text vocabulary.
    :type vocab_size: int
    :param max_text_length: Maximum text sequence length in tokens.
    :type max_text_length: int
    :param hidden_size: Hidden dimension of the task head. ``None`` means
        take the larger of the two encoder dimensions.
    :type hidden_size: Optional[int]
    :param vision_hidden_size: Hidden dimension of the vision encoder.
    :type vision_hidden_size: Optional[int]
    :param text_hidden_size: Hidden dimension of the text encoder.
    :type text_hidden_size: Optional[int]
    :param fusion_hidden_size: Hidden dimension of the fusion stage.
        ``None`` means derive it from ``fusion_type``.
    :type fusion_hidden_size: Optional[int]
    :param dropout_rate: Dropout rate used inside the head.
    :type dropout_rate: float
    :param num_classes: Number of output classes. Only classification tasks
        use it. ``None`` means unset, and is filled for three task types.
    :type num_classes: Optional[int]
    :param use_cross_attention: Whether the head attends across modalities.
    :type use_cross_attention: bool
    :param fusion_type: How the vision and text streams are combined.
    :type fusion_type: Literal["concat", "add", "multiply", "attention"]
    :param pooling_type: How vision features are pooled to one vector.
    :type pooling_type: Literal["avg", "max", "cls"]
    :param use_task_specific_heads: Whether to add task-specific output
        projections.
    :type use_task_specific_heads: bool
    :param temperature: Sampling temperature for generation tasks.
    :type temperature: float
    :param beam_size: Beam width for beam search. 1 means greedy decoding.
    :type beam_size: int
    :param loss_weight: Weight of this task's loss in multi-task training.
    :type loss_weight: float
    """

    name: str
    task_type: VLMTaskType
    vocab_size: int = 50000
    max_text_length: int = 512
    hidden_size: Optional[int] = 768
    vision_hidden_size: Optional[int] = 768
    text_hidden_size: Optional[int] = 768
    fusion_hidden_size: Optional[int] = 768
    dropout_rate: float = 0.1
    num_classes: Optional[int] = None
    use_cross_attention: bool = True
    fusion_type: Literal["concat", "add", "multiply", "attention"] = "attention"
    pooling_type: Literal["avg", "max", "cls"] = "avg"
    use_task_specific_heads: bool = True
    temperature: float = 1.0
    beam_size: int = 1
    loss_weight: float = 1.0

    def __post_init__(self):
        """
        Fill in the fields that were left at ``None``.

        ``hidden_size`` becomes the larger of ``vision_hidden_size`` and
        ``text_hidden_size``. ``fusion_hidden_size`` becomes their sum when
        ``fusion_type`` is ``"concat"``, because concatenation stacks the two
        streams instead of mixing them. For every other fusion type it becomes
        ``hidden_size``. ``num_classes`` is filled for three task
        types only: 3 for ``MULTIMODAL_SENTIMENT_ANALYSIS``, 7 for
        ``MULTIMODAL_EMOTION_RECOGNITION``, 3 for ``VISUAL_ENTAILMENT``.
        Every other task keeps ``num_classes`` as it was.

        The ``2`` fallback in the ``default_classes.get`` call is never
        reached. The surrounding test already restricts the task type to the
        same three keys the dict holds.
        """
        if self.hidden_size is None:
            self.hidden_size = max(self.vision_hidden_size, self.text_hidden_size)

        if self.fusion_hidden_size is None:
            if self.fusion_type == "concat":
                self.fusion_hidden_size = (
                    self.vision_hidden_size + self.text_hidden_size
                )
            else:
                self.fusion_hidden_size = self.hidden_size

        classification_tasks = [
            VLMTaskType.MULTIMODAL_SENTIMENT_ANALYSIS,
            VLMTaskType.MULTIMODAL_EMOTION_RECOGNITION,
            VLMTaskType.VISUAL_ENTAILMENT,
        ]
        if self.task_type in classification_tasks and self.num_classes is None:
            default_classes = {
                VLMTaskType.MULTIMODAL_SENTIMENT_ANALYSIS: 3,
                VLMTaskType.MULTIMODAL_EMOTION_RECOGNITION: 7,
                VLMTaskType.VISUAL_ENTAILMENT: 3,
            }
            self.num_classes = default_classes.get(self.task_type, 2)


# ---------------------------------------------------------------------


class VLMTaskConfiguration:
    """
    A validated set of VLM tasks for a multi-task model.

    The constructor takes a list, rejects an empty one and rejects
    duplicates, then stores the tasks as a set. Order is not kept. Use
    :meth:`has_task` to test membership and :meth:`requires_generation` to
    ask whether the set needs a text decoder.

    This class holds task types only. It carries no per-task hyperparameters
    and never touches a head class. So it neither knows nor checks which of the
    47 task types have a head.

    **Generation tasks:**

    :meth:`requires_generation` returns ``True`` when the set intersects
    these seven members. The list is hardcoded in that method.

    .. code-block:: text

        IMAGE_CAPTIONING
        DENSE_CAPTIONING
        VISUAL_STORYTELLING
        VISUAL_DIALOGUE
        REFERRING_EXPRESSION_GENERATION
        VIDEO_CAPTIONING
        MEDICAL_REPORT_GENERATION

    Note:
        Five of those seven have no head. Only ``IMAGE_CAPTIONING`` reaches
        a real head class, and ``DENSE_CAPTIONING`` and ``VISUAL_DIALOGUE``
        are rejected placeholders. So ``requires_generation()`` can return
        ``True`` for a set no factory can build.

    Example:
        >>> config = VLMTaskConfiguration([VLMTaskType.IMAGE_CAPTIONING])
        >>> config.requires_generation()
        True

    :ivar tasks: The enabled tasks. The property returns a copy, so mutating
        the result does not change the configuration.
    :vartype tasks: Set[VLMTaskType]

    :param tasks: The task types to enable. Must be non-empty and free of
        duplicates.
    :type tasks: List[VLMTaskType]
    :raises ValueError: If the list is empty or contains a duplicate.
    """

    def __init__(self, tasks: List[VLMTaskType]):
        """
        Validate the task list and store it as a set.

        :param tasks: The task types to enable. Must be non-empty and free
            of duplicates.
        :type tasks: List[VLMTaskType]
        :raises ValueError: If the list is empty or contains a duplicate.
        """
        if not tasks:
            raise ValueError("At least one task must be specified")
        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate tasks found in configuration")
        self._tasks: Set[VLMTaskType] = set(tasks)

    @property
    def tasks(self) -> Set[VLMTaskType]:
        """
        The enabled tasks.

        A fresh copy is returned each time, so mutating it does not change
        the configuration.

        :return: The enabled task types.
        :rtype: Set[VLMTaskType]
        """
        return self._tasks.copy()

    def has_task(self, task: VLMTaskType) -> bool:
        """
        Test whether one task is enabled.

        :param task: The task type to look for.
        :type task: VLMTaskType
        :return: ``True`` if the task is in this configuration.
        :rtype: bool
        """
        return task in self._tasks

    def requires_generation(self) -> bool:
        """
        Test whether any enabled task needs a text decoder.

        The seven generation tasks are listed in the class docstring.

        :return: ``True`` if the enabled set contains at least one of them.
        :rtype: bool
        """
        generation_tasks = {
            VLMTaskType.IMAGE_CAPTIONING,
            VLMTaskType.DENSE_CAPTIONING,
            VLMTaskType.VISUAL_STORYTELLING,
            VLMTaskType.VISUAL_DIALOGUE,
            VLMTaskType.REFERRING_EXPRESSION_GENERATION,
            VLMTaskType.VIDEO_CAPTIONING,
            VLMTaskType.MEDICAL_REPORT_GENERATION,
        }
        return bool(self._tasks & generation_tasks)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a plain dictionary.

        The ``tasks`` list holds string values, not enum members, and its
        order follows set iteration order rather than the order the tasks
        were passed in. ``requires_generation`` is stored alongside them.
        There is no matching ``from_dict``.

        :return: A dict with keys ``"tasks"`` and ``"requires_generation"``.
        :rtype: Dict[str, Any]
        """
        return {
            "tasks": [task.value for task in self._tasks],
            "requires_generation": self.requires_generation(),
        }