"""
VLM Task Types and Configuration

Comprehensive task type definitions and configuration helpers for Visual Language Models.
Designed to work with multi-modal foundation models that process both vision and text.
"""

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, List, Literal, Optional, Set


# ---------------------------------------------------------------------


@unique
class VLMTaskType(Enum):
    """
    Enumeration of supported VLM tasks for multi-modal models.

    Each task represents a different visual-language capability that bridges
    vision and language understanding. Tasks are organized into categories
    based on their primary function.
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
        """Get all available task types."""
        return list(cls)

    @classmethod
    def get_task_categories(cls) -> Dict[str, List["VLMTaskType"]]:
        """Get tasks organized by categories."""
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
        """Create VLMTaskType from string value."""
        task_str = task_str.lower().strip()
        for task in cls:
            if task.value == task_str:
                return task

        valid_tasks = [task.value for task in cls]
        raise ValueError(
            f"Invalid task type: '{task_str}'. " f"Valid options are: {valid_tasks}"
        )

    def __str__(self) -> str:
        """String representation of the task type."""
        return self.value

    def __repr__(self) -> str:
        """Detailed string representation of the task type."""
        return f"VLMTaskType.{self.name}"


# ---------------------------------------------------------------------


@dataclass
class VLMTaskConfig:
    """
    Configuration for a specific VLM task.

    Args:
        name: Unique identifier for the task
        task_type: Type of VLM task
        vocab_size: Size of text vocabulary
        max_text_length: Maximum text sequence length
        hidden_size: Hidden dimension for task head
        vision_hidden_size: Vision encoder hidden dimension
        text_hidden_size: Text encoder hidden dimension
        fusion_hidden_size: Multi-modal fusion hidden dimension
        dropout_rate: Dropout rate for regularization
        num_classes: Number of classes (for classification tasks)
        use_cross_attention: Whether to use cross-modal attention
        fusion_type: Type of multi-modal fusion ('concat', 'add', 'multiply', 'attention')
        pooling_type: Type of pooling for vision features ('avg', 'max', 'cls')
        use_task_specific_heads: Whether to use task-specific output heads
        temperature: Temperature for generation tasks
        beam_size: Beam size for beam search
        loss_weight: Weight for this task's loss in multi-task training
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
        """Post-initialization validation and setup."""
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
    Configuration helper for managing task combinations in VLM multi-task models.
    """

    def __init__(self, tasks: List[VLMTaskType]):
        """Initialize task configuration."""
        if not tasks:
            raise ValueError("At least one task must be specified")
        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate tasks found in configuration")
        self._tasks: Set[VLMTaskType] = set(tasks)

    @property
    def tasks(self) -> Set[VLMTaskType]:
        """Get the set of enabled tasks."""
        return self._tasks.copy()

    def has_task(self, task: VLMTaskType) -> bool:
        """Check if a specific task is enabled."""
        return task in self._tasks

    def requires_generation(self) -> bool:
        """Check if any task requires text generation capabilities."""
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
        """Convert configuration to dictionary for serialization."""
        return {
            "tasks": [task.value for task in self._tasks],
            "requires_generation": self.requires_generation(),
        }