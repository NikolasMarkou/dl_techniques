"""
VLM Task Types and Configuration

Comprehensive task type definitions and configuration helpers for Visual Language Models.
Designed to work with multi-modal foundation models that process both vision and text.
"""

from enum import Enum, unique
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Any, Literal


# ---------------------------------------------------------------------

@unique
class VLMTaskType(Enum):
    """
    Enumeration of supported VLM tasks for multi-modal models.

    Each task represents a different visual-language capability that bridges
    vision and language understanding. Tasks are organized into categories
    based on their primary function.

    Image Understanding & Description:
        IMAGE_CAPTIONING: Generate natural language descriptions of images
        DENSE_CAPTIONING: Generate region-specific captions for image areas
        VISUAL_STORYTELLING: Generate narrative stories from image sequences
        IMAGE_PARAGRAPH_CAPTIONING: Generate detailed paragraph descriptions

    Visual Question Answering:
        VISUAL_QUESTION_ANSWERING: Answer questions about image content
        VISUAL_REASONING: Complex reasoning over visual information
        VISUAL_COMMONSENSE_REASONING: Reasoning with commonsense knowledge
        CHART_QUESTION_ANSWERING: Answer questions about charts/graphs
        DIAGRAM_UNDERSTANDING: Understanding and reasoning about diagrams

    Visual Grounding & Localization:
        VISUAL_GROUNDING: Localize image regions from text descriptions
        REFERRING_EXPRESSION_COMPREHENSION: Ground referring expressions
        REFERRING_EXPRESSION_GENERATION: Generate descriptions for regions
        PHRASE_GROUNDING: Ground multiple phrases in images

    Image-Text Matching & Retrieval:
        IMAGE_TEXT_MATCHING: Determine image-text correspondence
        IMAGE_RETRIEVAL: Retrieve images based on text queries
        TEXT_RETRIEVAL: Retrieve text based on image queries
        CROSS_MODAL_RETRIEVAL: Bi-directional retrieval

    Visual Dialogue & Interaction:
        VISUAL_DIALOGUE: Multi-turn conversations about images
        EMBODIED_QUESTION_ANSWERING: QA in embodied/3D environments
        VISUAL_CHAT: Open-ended visual conversations

    OCR & Document Understanding:
        OPTICAL_CHARACTER_RECOGNITION: Extract text from images
        SCENE_TEXT_RECOGNITION: Recognize text in natural scenes
        DOCUMENT_UNDERSTANDING: Understand document structure and content
        TABLE_UNDERSTANDING: Extract and understand tabular data
        FORM_UNDERSTANDING: Process forms and structured documents

    Visual Entailment & Inference:
        VISUAL_ENTAILMENT: Determine if text follows from image
        VISUAL_INFERENCE: Make inferences from visual content
        FACT_VERIFICATION: Verify facts against visual evidence

    Multi-modal Generation:
        TEXT_TO_IMAGE_GENERATION: Generate images from text descriptions
        IMAGE_EDITING_INSTRUCTION: Generate editing instructions
        VISUAL_INSTRUCTION_FOLLOWING: Follow visual instructions
        IMAGE_MANIPULATION_GUIDANCE: Guide image manipulation tasks

    Multi-modal Classification:
        MULTIMODAL_SENTIMENT_ANALYSIS: Analyze sentiment from image-text
        MULTIMODAL_EMOTION_RECOGNITION: Recognize emotions from multimodal input
        MEME_UNDERSTANDING: Understand and classify memes
        VISUAL_METAPHOR_UNDERSTANDING: Understand visual metaphors

    Video Understanding:
        VIDEO_CAPTIONING: Generate captions for video content
        VIDEO_QUESTION_ANSWERING: Answer questions about videos
        VIDEO_SUMMARIZATION: Summarize video content
        TEMPORAL_GROUNDING: Localize moments in videos from text
        ACTION_RECOGNITION: Recognize and describe actions in videos

    Medical & Scientific:
        MEDICAL_VQA: Medical visual question answering
        MEDICAL_REPORT_GENERATION: Generate medical reports from images
        SCIENTIFIC_FIGURE_UNDERSTANDING: Understand scientific figures

    Educational & Instructional:
        VISUAL_INSTRUCTION_GENERATION: Generate instructions from visual demos
        EDUCATIONAL_CONTENT_UNDERSTANDING: Understand educational materials
        DIAGRAM_TO_TEXT: Convert diagrams to text descriptions
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
    def get_compatible_tasks(cls, task: "VLMTaskType") -> List["VLMTaskType"]:
        """Get tasks that are commonly combined with the given task."""
        compatibility_map = {
            cls.IMAGE_CAPTIONING: [
                cls.DENSE_CAPTIONING,
                cls.VISUAL_QUESTION_ANSWERING,
                cls.IMAGE_TEXT_MATCHING,
                cls.VISUAL_GROUNDING,
            ],
            cls.VISUAL_QUESTION_ANSWERING: [
                cls.IMAGE_CAPTIONING,
                cls.VISUAL_REASONING,
                cls.VISUAL_GROUNDING,
                cls.VISUAL_DIALOGUE,
            ],
            cls.VISUAL_GROUNDING: [
                cls.REFERRING_EXPRESSION_COMPREHENSION,
                cls.REFERRING_EXPRESSION_GENERATION,
                cls.DENSE_CAPTIONING,
                cls.PHRASE_GROUNDING,
            ],
            cls.IMAGE_TEXT_MATCHING: [
                cls.IMAGE_RETRIEVAL,
                cls.TEXT_RETRIEVAL,
                cls.CROSS_MODAL_RETRIEVAL,
                cls.IMAGE_CAPTIONING,
            ],
            cls.VISUAL_DIALOGUE: [
                cls.VISUAL_QUESTION_ANSWERING,
                cls.VISUAL_CHAT,
                cls.VISUAL_REASONING,
            ],
            cls.DOCUMENT_UNDERSTANDING: [
                cls.OPTICAL_CHARACTER_RECOGNITION,
                cls.TABLE_UNDERSTANDING,
                cls.FORM_UNDERSTANDING,
                cls.SCENE_TEXT_RECOGNITION,
            ],
            cls.VIDEO_CAPTIONING: [
                cls.VIDEO_QUESTION_ANSWERING,
                cls.VIDEO_SUMMARIZATION,
                cls.ACTION_RECOGNITION,
                cls.TEMPORAL_GROUNDING,
            ],
        }

        return compatibility_map.get(task, [])

    @classmethod
    def get_output_types(cls, task: "VLMTaskType") -> Dict[str, str]:
        """Get the expected output types for a given task."""
        output_types = {
            cls.IMAGE_CAPTIONING: {
                "caption": "str",
                "confidence": "float32",
            },
            cls.DENSE_CAPTIONING: {
                "regions": "float32[N, 4]",
                "captions": "List[str]",
                "confidence": "float32[N]",
            },
            cls.VISUAL_QUESTION_ANSWERING: {
                "answer": "str",
                "confidence": "float32",
                "answer_type": "str",  # yes/no, number, other
            },
            cls.VISUAL_GROUNDING: {
                "bbox": "float32[4]",  # [x1, y1, x2, y2]
                "confidence": "float32",
                "phrase": "str",
            },
            cls.IMAGE_TEXT_MATCHING: {
                "score": "float32",
                "match": "bool",
            },
            cls.VISUAL_DIALOGUE: {
                "response": "str",
                "history": "List[Dict[str, str]]",
            },
            cls.OPTICAL_CHARACTER_RECOGNITION: {
                "text": "str",
                "bboxes": "float32[N, 4]",
                "confidence": "float32[N]",
            },
            cls.VISUAL_ENTAILMENT: {
                "entailment": "str",  # entailment, neutral, contradiction
                "confidence": "float32",
            },
            cls.VIDEO_CAPTIONING: {
                "caption": "str",
                "timestamps": "float32[2]",  # [start, end]
            },
            cls.MULTIMODAL_SENTIMENT_ANALYSIS: {
                "sentiment": "str",  # positive, negative, neutral
                "confidence": "float32",
                "aspects": "List[Dict[str, Any]]",
            },
        }

        return output_types.get(task, {"output": "Any"})

    @classmethod
    def get_input_requirements(cls, task: "VLMTaskType") -> Dict[str, Any]:
        """Get input requirements for a task."""
        requirements = {
            cls.IMAGE_CAPTIONING: {
                "image": "required",
                "text": "optional",  # For guided captioning
                "max_length": 100,
            },
            cls.VISUAL_QUESTION_ANSWERING: {
                "image": "required",
                "question": "required",
                "context": "optional",
                "max_answer_length": 50,
            },
            cls.VISUAL_GROUNDING: {
                "image": "required",
                "text": "required",  # Referring expression
                "return_all_matches": False,
            },
            cls.IMAGE_TEXT_MATCHING: {
                "image": "required",
                "text": "required",
                "use_cross_attention": True,
            },
            cls.VISUAL_DIALOGUE: {
                "image": "required",
                "dialogue_history": "required",
                "current_utterance": "required",
                "max_turns": 10,
            },
            cls.VIDEO_CAPTIONING: {
                "video_frames": "required",
                "frame_rate": "optional",
                "max_frames": 32,
            },
            cls.DOCUMENT_UNDERSTANDING: {
                "image": "required",
                "layout_info": "optional",
                "ocr_results": "optional",
            },
        }

        return requirements.get(task, {"image": "required", "text": "optional"})

    @classmethod
    def from_string(cls, task_str: str) -> "VLMTaskType":
        """Create VLMTaskType from string value."""
        task_str = task_str.lower().strip()
        for task in cls:
            if task.value == task_str:
                return task

        valid_tasks = [task.value for task in cls]
        raise ValueError(
            f"Invalid task type: '{task_str}'. "
            f"Valid options are: {valid_tasks}"
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
    fusion_type: Literal['concat', 'add', 'multiply', 'attention'] = 'attention'
    pooling_type: Literal['avg', 'max', 'cls'] = 'avg'
    use_task_specific_heads: bool = True
    temperature: float = 1.0
    beam_size: int = 1
    loss_weight: float = 1.0

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Set default hidden sizes if not specified
        if self.hidden_size is None:
            self.hidden_size = max(self.vision_hidden_size, self.text_hidden_size)

        if self.fusion_hidden_size is None:
            if self.fusion_type == 'concat':
                self.fusion_hidden_size = self.vision_hidden_size + self.text_hidden_size
            else:
                self.fusion_hidden_size = self.hidden_size

        # Validate task-specific requirements
        classification_tasks = [
            VLMTaskType.MULTIMODAL_SENTIMENT_ANALYSIS,
            VLMTaskType.MULTIMODAL_EMOTION_RECOGNITION,
            VLMTaskType.VISUAL_ENTAILMENT,
        ]

        if self.task_type in classification_tasks and self.num_classes is None:
            # Set default number of classes
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

    def __init__(
            self,
            tasks: List[VLMTaskType],
            validate_compatibility: bool = True
    ):
        """Initialize task configuration."""
        if not tasks:
            raise ValueError("At least one task must be specified")

        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate tasks found in configuration")

        self._tasks: Set[VLMTaskType] = set(tasks)

        if validate_compatibility and len(tasks) > 1:
            self._validate_task_compatibility()

    def _validate_task_compatibility(self) -> None:
        """Validate that all tasks in the configuration are compatible."""
        task_list = list(self._tasks)

        # Check for incompatible combinations
        incompatible_pairs = [
            # Different modality requirements
            (VLMTaskType.OPTICAL_CHARACTER_RECOGNITION, VLMTaskType.VIDEO_CAPTIONING),
            # Conflicting objectives
            (VLMTaskType.TEXT_TO_IMAGE_GENERATION, VLMTaskType.IMAGE_CAPTIONING),
        ]

        for task1, task2 in incompatible_pairs:
            if task1 in self._tasks and task2 in self._tasks:
                raise ValueError(f"Tasks {task1} and {task2} are incompatible")

    @property
    def tasks(self) -> Set[VLMTaskType]:
        """Get the set of enabled tasks."""
        return self._tasks.copy()

    def has_task(self, task: VLMTaskType) -> bool:
        """Check if a specific task is enabled."""
        return task in self._tasks

    def requires_visual_grounding(self) -> bool:
        """Check if any task requires visual grounding capabilities."""
        grounding_tasks = {
            VLMTaskType.VISUAL_GROUNDING,
            VLMTaskType.REFERRING_EXPRESSION_COMPREHENSION,
            VLMTaskType.REFERRING_EXPRESSION_GENERATION,
            VLMTaskType.PHRASE_GROUNDING,
            VLMTaskType.DENSE_CAPTIONING,
        }
        return bool(self._tasks & grounding_tasks)

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

    def requires_video_processing(self) -> bool:
        """Check if any task requires video processing capabilities."""
        video_tasks = {
            VLMTaskType.VIDEO_CAPTIONING,
            VLMTaskType.VIDEO_QUESTION_ANSWERING,
            VLMTaskType.VIDEO_SUMMARIZATION,
            VLMTaskType.TEMPORAL_GROUNDING,
            VLMTaskType.ACTION_RECOGNITION,
        }
        return bool(self._tasks & video_tasks)

    def requires_ocr(self) -> bool:
        """Check if any task requires OCR capabilities."""
        ocr_tasks = {
            VLMTaskType.OPTICAL_CHARACTER_RECOGNITION,
            VLMTaskType.SCENE_TEXT_RECOGNITION,
            VLMTaskType.DOCUMENT_UNDERSTANDING,
            VLMTaskType.TABLE_UNDERSTANDING,
            VLMTaskType.FORM_UNDERSTANDING,
        }
        return bool(self._tasks & ocr_tasks)

    def get_max_sequence_length(self) -> int:
        """Get the maximum sequence length required across all tasks."""
        max_length = 512  # Default

        if self.requires_generation():
            max_length = max(max_length, 1024)

        if VLMTaskType.VISUAL_STORYTELLING in self._tasks:
            max_length = max(max_length, 2048)

        if VLMTaskType.MEDICAL_REPORT_GENERATION in self._tasks:
            max_length = max(max_length, 4096)

        return max_length

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "tasks": [task.value for task in self._tasks],
            "requires_visual_grounding": self.requires_visual_grounding(),
            "requires_generation": self.requires_generation(),
            "requires_video_processing": self.requires_video_processing(),
            "requires_ocr": self.requires_ocr(),
            "max_sequence_length": self.get_max_sequence_length(),
        }


# ---------------------------------------------------------------------

class CommonVLMTaskConfigurations:
    """Predefined common VLM task configurations."""

    # Single task configurations
    CAPTIONING_ONLY = VLMTaskConfiguration([VLMTaskType.IMAGE_CAPTIONING])
    VQA_ONLY = VLMTaskConfiguration([VLMTaskType.VISUAL_QUESTION_ANSWERING])
    GROUNDING_ONLY = VLMTaskConfiguration([VLMTaskType.VISUAL_GROUNDING])
    MATCHING_ONLY = VLMTaskConfiguration([VLMTaskType.IMAGE_TEXT_MATCHING])
    OCR_ONLY = VLMTaskConfiguration([VLMTaskType.OPTICAL_CHARACTER_RECOGNITION])

    # Vision-language understanding
    VISION_LANGUAGE_UNDERSTANDING = VLMTaskConfiguration([
        VLMTaskType.IMAGE_CAPTIONING,
        VLMTaskType.VISUAL_QUESTION_ANSWERING,
        VLMTaskType.IMAGE_TEXT_MATCHING,
    ])

    # Visual grounding suite
    VISUAL_GROUNDING_SUITE = VLMTaskConfiguration([
        VLMTaskType.VISUAL_GROUNDING,
        VLMTaskType.REFERRING_EXPRESSION_COMPREHENSION,
        VLMTaskType.REFERRING_EXPRESSION_GENERATION,
        VLMTaskType.PHRASE_GROUNDING,
    ])

    # Dense understanding
    DENSE_UNDERSTANDING = VLMTaskConfiguration([
        VLMTaskType.DENSE_CAPTIONING,
        VLMTaskType.VISUAL_GROUNDING,
        VLMTaskType.IMAGE_CAPTIONING,
    ])

    # Document AI
    DOCUMENT_AI = VLMTaskConfiguration([
        VLMTaskType.OPTICAL_CHARACTER_RECOGNITION,
        VLMTaskType.DOCUMENT_UNDERSTANDING,
        VLMTaskType.TABLE_UNDERSTANDING,
        VLMTaskType.FORM_UNDERSTANDING,
    ])

    # Video understanding
    VIDEO_UNDERSTANDING = VLMTaskConfiguration([
        VLMTaskType.VIDEO_CAPTIONING,
        VLMTaskType.VIDEO_QUESTION_ANSWERING,
        VLMTaskType.ACTION_RECOGNITION,
    ])

    # Visual dialogue
    VISUAL_DIALOGUE_SUITE = VLMTaskConfiguration([
        VLMTaskType.VISUAL_DIALOGUE,
        VLMTaskType.VISUAL_QUESTION_ANSWERING,
        VLMTaskType.VISUAL_CHAT,
    ])

    # Medical VLM
    MEDICAL_VLM = VLMTaskConfiguration([
        VLMTaskType.MEDICAL_VQA,
        VLMTaskType.MEDICAL_REPORT_GENERATION,
        VLMTaskType.IMAGE_CAPTIONING,
    ])

    # Comprehensive VLM
    COMPREHENSIVE_VLM = VLMTaskConfiguration([
        VLMTaskType.IMAGE_CAPTIONING,
        VLMTaskType.VISUAL_QUESTION_ANSWERING,
        VLMTaskType.VISUAL_GROUNDING,
        VLMTaskType.IMAGE_TEXT_MATCHING,
        VLMTaskType.VISUAL_REASONING,
    ])