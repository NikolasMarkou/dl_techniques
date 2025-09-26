"""
NLP Task Types and Configuration

Comprehensive task type definitions and configuration helpers for NLP multi-task models.
Designed to work with any NLP foundation model (BERT, GPT, T5, etc.).
"""

from enum import Enum, unique
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Any

# ---------------------------------------------------------------------

@unique
class NLPTaskType(Enum):
    """
    Enumeration of supported NLP tasks for multi-task models.

    Each task represents a different NLP capability that can be
    enabled in the multi-task architecture. Tasks are organized into
    categories for better understanding and compatibility checking.

    Token-Level Tasks:
        TOKEN_CLASSIFICATION: General token classification (NER, POS, etc.)
        NAMED_ENTITY_RECOGNITION: Identifying and classifying named entities
        PART_OF_SPEECH_TAGGING: Grammatical tagging of tokens
        DEPENDENCY_PARSING: Syntactic dependency relationships
        SEMANTIC_ROLE_LABELING: Identifying semantic roles
        WORD_SENSE_DISAMBIGUATION: Determining word meanings in context

    Sequence-Level Tasks:
        TEXT_CLASSIFICATION: Document/sentence classification
        SENTIMENT_ANALYSIS: Sentiment polarity detection
        EMOTION_DETECTION: Fine-grained emotion classification
        INTENT_CLASSIFICATION: Intent detection for dialogue
        TOPIC_CLASSIFICATION: Topic categorization
        SPAM_DETECTION: Spam/ham classification

    Span-Level Tasks:
        QUESTION_ANSWERING: Extractive QA with span selection
        SPAN_EXTRACTION: General span extraction
        COREFERENCE_RESOLUTION: Identifying coreferent mentions
        EVENT_EXTRACTION: Extracting events and arguments
        RELATION_EXTRACTION: Extracting relations between entities

    Sentence-Pair Tasks:
        TEXT_SIMILARITY: Semantic similarity scoring
        NATURAL_LANGUAGE_INFERENCE: Textual entailment
        PARAPHRASE_DETECTION: Identifying paraphrases
        DUPLICATE_DETECTION: Finding duplicate texts

    Generation Tasks:
        TEXT_GENERATION: Autoregressive text generation
        MASKED_LANGUAGE_MODELING: Predicting masked tokens
        TEXT_SUMMARIZATION: Abstractive/extractive summarization
        MACHINE_TRANSLATION: Language translation
        TEXT_COMPLETION: Completing partial text
        DIALOGUE_GENERATION: Conversational response generation

    Regression Tasks:
        TEXT_REGRESSION: Continuous value prediction
        READABILITY_SCORING: Text complexity assessment
        QUALITY_SCORING: Text quality evaluation

    Structured Tasks:
        MULTIPLE_CHOICE: Multiple choice question answering
        RANKING: Text ranking and reranking
        SEQUENCE_LABELING: General sequence labeling
        TEXT_MATCHING: Matching texts to references

    Information Extraction:
        KEY_PHRASE_EXTRACTION: Extracting important phrases
        FACT_EXTRACTION: Extracting factual claims
        OPINION_EXTRACTION: Extracting opinions and aspects
    """

    # Token-Level Tasks
    TOKEN_CLASSIFICATION = "token_classification"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    PART_OF_SPEECH_TAGGING = "part_of_speech_tagging"
    DEPENDENCY_PARSING = "dependency_parsing"
    SEMANTIC_ROLE_LABELING = "semantic_role_labeling"
    WORD_SENSE_DISAMBIGUATION = "word_sense_disambiguation"

    # Sequence-Level Tasks
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    EMOTION_DETECTION = "emotion_detection"
    INTENT_CLASSIFICATION = "intent_classification"
    TOPIC_CLASSIFICATION = "topic_classification"
    SPAM_DETECTION = "spam_detection"

    # Span-Level Tasks
    QUESTION_ANSWERING = "question_answering"
    SPAN_EXTRACTION = "span_extraction"
    COREFERENCE_RESOLUTION = "coreference_resolution"
    EVENT_EXTRACTION = "event_extraction"
    RELATION_EXTRACTION = "relation_extraction"

    # Sentence-Pair Tasks
    TEXT_SIMILARITY = "text_similarity"
    NATURAL_LANGUAGE_INFERENCE = "natural_language_inference"
    PARAPHRASE_DETECTION = "paraphrase_detection"
    DUPLICATE_DETECTION = "duplicate_detection"

    # Generation Tasks
    TEXT_GENERATION = "text_generation"
    MASKED_LANGUAGE_MODELING = "masked_language_modeling"
    TEXT_SUMMARIZATION = "text_summarization"
    MACHINE_TRANSLATION = "machine_translation"
    TEXT_COMPLETION = "text_completion"
    DIALOGUE_GENERATION = "dialogue_generation"

    # Regression Tasks
    TEXT_REGRESSION = "text_regression"
    READABILITY_SCORING = "readability_scoring"
    QUALITY_SCORING = "quality_scoring"

    # Structured Tasks
    MULTIPLE_CHOICE = "multiple_choice"
    RANKING = "ranking"
    SEQUENCE_LABELING = "sequence_labeling"
    TEXT_MATCHING = "text_matching"

    # Information Extraction
    KEY_PHRASE_EXTRACTION = "key_phrase_extraction"
    FACT_EXTRACTION = "fact_extraction"
    OPINION_EXTRACTION = "opinion_extraction"

    @classmethod
    def all_tasks(cls) -> List["NLPTaskType"]:
        """Get all available task types."""
        return list(cls)

    @classmethod
    def get_task_categories(cls) -> Dict[str, List["NLPTaskType"]]:
        """Get tasks organized by categories."""
        return {
            "Token-Level Tasks": [
                cls.TOKEN_CLASSIFICATION,
                cls.NAMED_ENTITY_RECOGNITION,
                cls.PART_OF_SPEECH_TAGGING,
                cls.DEPENDENCY_PARSING,
                cls.SEMANTIC_ROLE_LABELING,
                cls.WORD_SENSE_DISAMBIGUATION,
            ],
            "Sequence-Level Tasks": [
                cls.TEXT_CLASSIFICATION,
                cls.SENTIMENT_ANALYSIS,
                cls.EMOTION_DETECTION,
                cls.INTENT_CLASSIFICATION,
                cls.TOPIC_CLASSIFICATION,
                cls.SPAM_DETECTION,
            ],
            "Span-Level Tasks": [
                cls.QUESTION_ANSWERING,
                cls.SPAN_EXTRACTION,
                cls.COREFERENCE_RESOLUTION,
                cls.EVENT_EXTRACTION,
                cls.RELATION_EXTRACTION,
            ],
            "Sentence-Pair Tasks": [
                cls.TEXT_SIMILARITY,
                cls.NATURAL_LANGUAGE_INFERENCE,
                cls.PARAPHRASE_DETECTION,
                cls.DUPLICATE_DETECTION,
            ],
            "Generation Tasks": [
                cls.TEXT_GENERATION,
                cls.MASKED_LANGUAGE_MODELING,
                cls.TEXT_SUMMARIZATION,
                cls.MACHINE_TRANSLATION,
                cls.TEXT_COMPLETION,
                cls.DIALOGUE_GENERATION,
            ],
            "Regression Tasks": [
                cls.TEXT_REGRESSION,
                cls.READABILITY_SCORING,
                cls.QUALITY_SCORING,
            ],
            "Structured Tasks": [
                cls.MULTIPLE_CHOICE,
                cls.RANKING,
                cls.SEQUENCE_LABELING,
                cls.TEXT_MATCHING,
            ],
            "Information Extraction": [
                cls.KEY_PHRASE_EXTRACTION,
                cls.FACT_EXTRACTION,
                cls.OPINION_EXTRACTION,
            ],
        }

    @classmethod
    def get_compatible_tasks(cls, task: "NLPTaskType") -> List["NLPTaskType"]:
        """Get tasks that are commonly combined with the given task."""
        compatibility_map = {
            cls.NAMED_ENTITY_RECOGNITION: [
                cls.PART_OF_SPEECH_TAGGING,
                cls.DEPENDENCY_PARSING,
                cls.RELATION_EXTRACTION,
                cls.EVENT_EXTRACTION,
                cls.COREFERENCE_RESOLUTION,
            ],
            cls.SENTIMENT_ANALYSIS: [
                cls.EMOTION_DETECTION,
                cls.OPINION_EXTRACTION,
                cls.TEXT_CLASSIFICATION,
                cls.QUALITY_SCORING,
            ],
            cls.QUESTION_ANSWERING: [
                cls.SPAN_EXTRACTION,
                cls.NATURAL_LANGUAGE_INFERENCE,
                cls.TEXT_MATCHING,
                cls.MULTIPLE_CHOICE,
            ],
            cls.TEXT_CLASSIFICATION: [
                cls.SENTIMENT_ANALYSIS,
                cls.TOPIC_CLASSIFICATION,
                cls.INTENT_CLASSIFICATION,
                cls.SPAM_DETECTION,
            ],
            cls.PART_OF_SPEECH_TAGGING: [
                cls.NAMED_ENTITY_RECOGNITION,
                cls.DEPENDENCY_PARSING,
                cls.SEMANTIC_ROLE_LABELING,
            ],
            cls.TEXT_GENERATION: [
                cls.TEXT_COMPLETION,
                cls.DIALOGUE_GENERATION,
                cls.TEXT_SUMMARIZATION,
            ],
            cls.NATURAL_LANGUAGE_INFERENCE: [
                cls.TEXT_SIMILARITY,
                cls.PARAPHRASE_DETECTION,
                cls.QUESTION_ANSWERING,
            ],
        }

        return compatibility_map.get(task, [])

    @classmethod
    def get_output_types(cls, task: "NLPTaskType") -> Dict[str, str]:
        """Get the expected output types for a given task."""
        output_types = {
            cls.TOKEN_CLASSIFICATION: {
                "logits": "float32[B, L, C]",
                "labels": "int32[B, L]",
            },
            cls.TEXT_CLASSIFICATION: {
                "logits": "float32[B, C]",
                "probabilities": "float32[B, C]",
            },
            cls.QUESTION_ANSWERING: {
                "start_logits": "float32[B, L]",
                "end_logits": "float32[B, L]",
                "answer_spans": "int32[B, 2]",
            },
            cls.TEXT_SIMILARITY: {
                "similarity_score": "float32[B]",
                "embeddings": "float32[B, D]",
            },
            cls.TEXT_GENERATION: {
                "logits": "float32[B, L, V]",
                "generated_ids": "int32[B, L]",
            },
            cls.NATURAL_LANGUAGE_INFERENCE: {
                "logits": "float32[B, 3]",  # entailment, neutral, contradiction
                "probabilities": "float32[B, 3]",
            },
            cls.SPAN_EXTRACTION: {
                "start_logits": "float32[B, L]",
                "end_logits": "float32[B, L]",
                "span_labels": "float32[B, L, L]",
            },
            cls.TEXT_REGRESSION: {
                "value": "float32[B]",
                "confidence": "float32[B]",
            },
            cls.MULTIPLE_CHOICE: {
                "logits": "float32[B, N]",  # N choices
                "probabilities": "float32[B, N]",
            },
            cls.RANKING: {
                "scores": "float32[B, N]",
                "rankings": "int32[B, N]",
            },
        }

        return output_types.get(task, {"output": "float32[...]"})

    @classmethod
    def get_input_requirements(cls, task: "NLPTaskType") -> Dict[str, Any]:
        """Get input requirements for a task."""
        requirements = {
            cls.TEXT_CLASSIFICATION: {
                "input_type": "single_sequence",
                "max_length": 512,
                "special_tokens": ["[CLS]", "[SEP]"],
            },
            cls.NATURAL_LANGUAGE_INFERENCE: {
                "input_type": "sequence_pair",
                "max_length": 512,
                "special_tokens": ["[CLS]", "[SEP]"],
            },
            cls.TOKEN_CLASSIFICATION: {
                "input_type": "single_sequence",
                "max_length": 512,
                "preserve_tokenization": True,
            },
            cls.QUESTION_ANSWERING: {
                "input_type": "sequence_pair",
                "max_length": 512,
                "special_tokens": ["[CLS]", "[SEP]", "[QUESTION]"],
            },
            cls.TEXT_GENERATION: {
                "input_type": "single_sequence",
                "max_length": 1024,
                "autoregressive": True,
            },
            cls.MULTIPLE_CHOICE: {
                "input_type": "multiple_sequences",
                "max_length": 512,
                "num_choices": "variable",
            },
        }

        return requirements.get(task, {"input_type": "single_sequence"})

    @classmethod
    def from_string(cls, task_str: str) -> "NLPTaskType":
        """Create NLPTaskType from string value."""
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
        return f"NLPTaskType.{self.name}"


# ---------------------------------------------------------------------

@dataclass
class NLPTaskConfig:
    """
    Configuration for a specific NLP task.

    Args:
        name: Unique identifier for the task
        task_type: Type of NLP task
        num_classes: Number of output classes (for classification tasks)
        num_labels: Alternative to num_classes for compatibility
        max_length: Maximum sequence length
        dropout_rate: Dropout rate for task-specific head
        hidden_size: Hidden dimension for task head
        loss_weight: Weight for this task's loss in multi-task training
        label_smoothing: Label smoothing parameter
        use_crf: Whether to use CRF for sequence labeling
        use_attention_pooling: Use attention-based pooling
        vocabulary_size: Size of vocabulary (for generation tasks)
        beam_size: Beam size for generation tasks
        temperature: Temperature for generation sampling
    """
    name: str
    task_type: NLPTaskType
    num_classes: Optional[int] = None
    num_labels: Optional[int] = None  # Alternative naming
    max_length: int = 512
    dropout_rate: float = 0.1
    hidden_size: Optional[int] = None
    loss_weight: float = 1.0
    label_smoothing: float = 0.0
    use_crf: bool = False
    use_attention_pooling: bool = False
    vocabulary_size: Optional[int] = None
    beam_size: int = 1
    temperature: float = 1.0

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Handle num_classes/num_labels ambiguity
        if self.num_labels is not None and self.num_classes is None:
            self.num_classes = self.num_labels
        elif self.num_classes is not None:
            self.num_labels = self.num_classes

        # Validate required fields for classification tasks
        classification_tasks = [
            NLPTaskType.TEXT_CLASSIFICATION,
            NLPTaskType.TOKEN_CLASSIFICATION,
            NLPTaskType.SENTIMENT_ANALYSIS,
            NLPTaskType.NAMED_ENTITY_RECOGNITION,
            NLPTaskType.PART_OF_SPEECH_TAGGING,
        ]

        if self.task_type in classification_tasks and self.num_classes is None:
            raise ValueError(f"{self.task_type} requires num_classes to be specified")

        # Set vocabulary size for generation tasks
        generation_tasks = [
            NLPTaskType.TEXT_GENERATION,
            NLPTaskType.MASKED_LANGUAGE_MODELING,
            NLPTaskType.TEXT_SUMMARIZATION,
            NLPTaskType.MACHINE_TRANSLATION,
        ]

        if self.task_type in generation_tasks and self.vocabulary_size is None:
            self.vocabulary_size = 32000  # Default vocab size


# ---------------------------------------------------------------------

class NLPTaskConfiguration:
    """
    Configuration helper for managing task combinations in NLP multi-task models.
    """

    def __init__(
            self,
            tasks: List[NLPTaskType],
            validate_compatibility: bool = True
    ):
        """Initialize task configuration."""
        if not tasks:
            raise ValueError("At least one task must be specified")

        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate tasks found in configuration")

        self._tasks: Set[NLPTaskType] = set(tasks)

        if validate_compatibility and len(tasks) > 1:
            self._validate_task_compatibility()

    def _validate_task_compatibility(self) -> None:
        """Validate that all tasks in the configuration are compatible."""
        task_list = list(self._tasks)

        # Check for incompatible combinations
        incompatible_pairs = [
            # Generation tasks generally don't mix well with classification
            (NLPTaskType.TEXT_GENERATION, NLPTaskType.TOKEN_CLASSIFICATION),
            # Different tokenization requirements
            (NLPTaskType.DEPENDENCY_PARSING, NLPTaskType.TEXT_GENERATION),
        ]

        for task1, task2 in incompatible_pairs:
            if task1 in self._tasks and task2 in self._tasks:
                raise ValueError(f"Tasks {task1} and {task2} are incompatible")

    @property
    def tasks(self) -> Set[NLPTaskType]:
        """Get the set of enabled tasks."""
        return self._tasks.copy()

    def has_task(self, task: NLPTaskType) -> bool:
        """Check if a specific task is enabled."""
        return task in self._tasks

    def requires_token_level(self) -> bool:
        """Check if any task requires token-level processing."""
        token_tasks = {
            NLPTaskType.TOKEN_CLASSIFICATION,
            NLPTaskType.NAMED_ENTITY_RECOGNITION,
            NLPTaskType.PART_OF_SPEECH_TAGGING,
            NLPTaskType.DEPENDENCY_PARSING,
            NLPTaskType.SEMANTIC_ROLE_LABELING,
        }
        return bool(self._tasks & token_tasks)

    def requires_sequence_pair(self) -> bool:
        """Check if any task requires sequence pair inputs."""
        pair_tasks = {
            NLPTaskType.NATURAL_LANGUAGE_INFERENCE,
            NLPTaskType.TEXT_SIMILARITY,
            NLPTaskType.PARAPHRASE_DETECTION,
            NLPTaskType.QUESTION_ANSWERING,
        }
        return bool(self._tasks & pair_tasks)

    def requires_generation(self) -> bool:
        """Check if any task requires generation capabilities."""
        generation_tasks = {
            NLPTaskType.TEXT_GENERATION,
            NLPTaskType.TEXT_SUMMARIZATION,
            NLPTaskType.MACHINE_TRANSLATION,
            NLPTaskType.DIALOGUE_GENERATION,
        }
        return bool(self._tasks & generation_tasks)

    def get_max_sequence_length(self) -> int:
        """Get the maximum sequence length required across all tasks."""
        max_length = 512  # Default

        if self.requires_generation():
            max_length = max(max_length, 1024)

        if NLPTaskType.TEXT_SUMMARIZATION in self._tasks:
            max_length = max(max_length, 2048)

        return max_length

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "tasks": [task.value for task in self._tasks],
            "requires_token_level": self.requires_token_level(),
            "requires_sequence_pair": self.requires_sequence_pair(),
            "requires_generation": self.requires_generation(),
            "max_sequence_length": self.get_max_sequence_length(),
        }


# ---------------------------------------------------------------------

class CommonNLPTaskConfigurations:
    """Predefined common NLP task configurations."""

    # Single task configurations
    TEXT_CLASSIFICATION_ONLY = NLPTaskConfiguration([NLPTaskType.TEXT_CLASSIFICATION])
    NER_ONLY = NLPTaskConfiguration([NLPTaskType.NAMED_ENTITY_RECOGNITION])
    SENTIMENT_ONLY = NLPTaskConfiguration([NLPTaskType.SENTIMENT_ANALYSIS])
    QA_ONLY = NLPTaskConfiguration([NLPTaskType.QUESTION_ANSWERING])
    GENERATION_ONLY = NLPTaskConfiguration([NLPTaskType.TEXT_GENERATION])

    # Token-level combinations
    TOKEN_TASKS = NLPTaskConfiguration([
        NLPTaskType.NAMED_ENTITY_RECOGNITION,
        NLPTaskType.PART_OF_SPEECH_TAGGING,
        NLPTaskType.DEPENDENCY_PARSING,
    ])

    # Classification combinations
    CLASSIFICATION_SUITE = NLPTaskConfiguration([
        NLPTaskType.TEXT_CLASSIFICATION,
        NLPTaskType.SENTIMENT_ANALYSIS,
        NLPTaskType.EMOTION_DETECTION,
    ])

    # Information extraction
    INFORMATION_EXTRACTION = NLPTaskConfiguration([
        NLPTaskType.NAMED_ENTITY_RECOGNITION,
        NLPTaskType.RELATION_EXTRACTION,
        NLPTaskType.EVENT_EXTRACTION,
    ])

    # Question answering and comprehension
    COMPREHENSION_SUITE = NLPTaskConfiguration([
        NLPTaskType.QUESTION_ANSWERING,
        NLPTaskType.NATURAL_LANGUAGE_INFERENCE,
        NLPTaskType.MULTIPLE_CHOICE,
    ])

    # Similarity and matching
    SIMILARITY_SUITE = NLPTaskConfiguration([
        NLPTaskType.TEXT_SIMILARITY,
        NLPTaskType.PARAPHRASE_DETECTION,
        NLPTaskType.DUPLICATE_DETECTION,
    ])

    # GLUE-like benchmark tasks
    GLUE_TASKS = NLPTaskConfiguration([
        NLPTaskType.TEXT_CLASSIFICATION,
        NLPTaskType.SENTIMENT_ANALYSIS,
        NLPTaskType.TEXT_SIMILARITY,
        NLPTaskType.NATURAL_LANGUAGE_INFERENCE,
        NLPTaskType.PARAPHRASE_DETECTION,
    ])