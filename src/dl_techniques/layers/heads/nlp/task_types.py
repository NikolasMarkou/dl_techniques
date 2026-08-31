"""Task types and configuration objects for the NLP heads.

This module says what an NLP task is. It builds no layer. Four objects live
here:

* :class:`NLPTaskType` -- an enum of 37 task names, with lookup helpers for
  categories, compatible tasks, output shapes and input requirements.
* :class:`NLPTaskConfig` -- a dataclass holding the hyperparameters of one
  task head.
* :class:`NLPTaskConfiguration` -- a validated set of tasks, plus queries for
  the capabilities that set needs.
* :class:`CommonNLPTaskConfigurations` -- eleven ready-made task sets.

``nlp/factory.py`` reads these objects and picks the head class to construct.
The types name tasks, not models, so heads on BERT, GPT and T5 all take the
same config.
"""

from enum import Enum, unique
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Any

# ---------------------------------------------------------------------

@unique
class NLPTaskType(Enum):
    """
    The 37 NLP tasks a head can be asked to serve.

    Each member carries a lowercase string value. That string is the
    serialization form, and :meth:`from_string` parses it back.

    The members are filed into eight named categories by
    :meth:`get_task_categories`: token-level, sequence-level, span-level,
    sentence-pair, generation, regression, structured, and information
    extraction. That filing is descriptive only. It does not decide which
    head class runs a task.

    **Head Routing:**

    ``nlp/factory.py``'s ``get_head_class`` decides that, and the table below
    is its mapping. Thirteen members have no head: asking for one raises
    ``ValueError`` rather than falling back to a default.

    .. code-block:: text

        Task member                 Head class
        --------------------------  ------------------------
        TOKEN_CLASSIFICATION        TokenClassificationHead
        NAMED_ENTITY_RECOGNITION    TokenClassificationHead
        PART_OF_SPEECH_TAGGING      TokenClassificationHead
        DEPENDENCY_PARSING          none -- raises ValueError
        SEMANTIC_ROLE_LABELING      none -- raises ValueError
        WORD_SENSE_DISAMBIGUATION   none -- raises ValueError
        TEXT_CLASSIFICATION         TextClassificationHead
        SENTIMENT_ANALYSIS          TextClassificationHead
        EMOTION_DETECTION           TextClassificationHead
        INTENT_CLASSIFICATION       TextClassificationHead
        TOPIC_CLASSIFICATION        TextClassificationHead
        SPAM_DETECTION              TextClassificationHead
        QUESTION_ANSWERING          QuestionAnsweringHead
        SPAN_EXTRACTION             QuestionAnsweringHead
        COREFERENCE_RESOLUTION      none -- raises ValueError
        EVENT_EXTRACTION            none -- raises ValueError
        RELATION_EXTRACTION         none -- raises ValueError
        TEXT_SIMILARITY             TextSimilarityHead
        NATURAL_LANGUAGE_INFERENCE  TextClassificationHead
        PARAPHRASE_DETECTION        TextSimilarityHead
        DUPLICATE_DETECTION         TextSimilarityHead
        TEXT_GENERATION             TextGenerationHead
        MASKED_LANGUAGE_MODELING    TextGenerationHead
        TEXT_SUMMARIZATION          TextGenerationHead
        MACHINE_TRANSLATION         none -- raises ValueError
        TEXT_COMPLETION             TextGenerationHead
        DIALOGUE_GENERATION         none -- raises ValueError
        TEXT_REGRESSION             TextClassificationHead
        READABILITY_SCORING         TextClassificationHead
        QUALITY_SCORING             TextClassificationHead
        MULTIPLE_CHOICE             MultipleChoiceHead
        RANKING                     none -- raises ValueError
        SEQUENCE_LABELING           TokenClassificationHead
        TEXT_MATCHING               none -- raises ValueError
        KEY_PHRASE_EXTRACTION       none -- raises ValueError
        FACT_EXTRACTION             none -- raises ValueError
        OPINION_EXTRACTION          none -- raises ValueError

    Note:
        The table restates a mapping that lives in ``get_head_class``. That
        function is the arbiter. If the two disagree, the function is right.

    Example:
        >>> task = NLPTaskType.from_string("sentiment_analysis")
        >>> str(task)
        'sentiment_analysis'
        >>> NLPTaskType.get_output_types(task)
        {'output': 'float32[...]'}
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
        """
        List every task type in declaration order.

        :return: All 37 enum members.
        :rtype: List[NLPTaskType]
        """
        return list(cls)

    @classmethod
    def get_task_categories(cls) -> Dict[str, List["NLPTaskType"]]:
        """
        Group the task types into their eight named categories.

        The grouping is documentation. It has no effect on which head class
        serves a task.

        :return: Category name mapped to the members filed under it.
        :rtype: Dict[str, List[NLPTaskType]]
        """
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
        """
        List the tasks commonly trained alongside a given task.

        Seven tasks have an entry. Any other task returns an empty list, which
        means "no suggestion", not "nothing is compatible".

        :param task: The task to look up.
        :type task: NLPTaskType
        :return: Suggested companion tasks, empty when none are recorded.
        :rtype: List[NLPTaskType]
        """
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
        """
        Describe the tensors a task's head is expected to emit.

        Shapes are written as strings, with ``B`` batch, ``L`` sequence
        length, ``C`` classes, ``D`` embedding width, ``V`` vocabulary and
        ``N`` choices. Ten tasks have an entry.

        :param task: The task to look up.
        :type task: NLPTaskType
        :return: Output name mapped to its shape string. Tasks with no entry
            get ``{"output": "float32[...]"}``.
        :rtype: Dict[str, str]
        """
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
                # The 3 classes are entailment, neutral, contradiction.
                "logits": "float32[B, 3]",
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
                "logits": "float32[B, N]",
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
        """
        Describe how a task's inputs must be prepared.

        Six tasks have an entry. Each records an ``input_type``, and may add a
        ``max_length``, the special tokens the tokenizer must emit, or a flag
        such as ``autoregressive``.

        :param task: The task to look up.
        :type task: NLPTaskType
        :return: Requirement name mapped to its value. Tasks with no entry get
            ``{"input_type": "single_sequence"}``.
        :rtype: Dict[str, Any]
        """
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
        """
        Parse a task type from its string value.

        The input is lowercased and stripped before matching, so
        ``" Sentiment_Analysis "`` resolves. Matching is against the enum
        VALUE, not the member name.

        :param task_str: A task value such as ``"sentiment_analysis"``.
        :type task_str: str
        :return: The matching member.
        :rtype: NLPTaskType
        :raises ValueError: If no member carries that value. The message lists
            every valid value.
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

    def __str__(self) -> str:
        """
        Return the enum value, which is the serialization form.

        :return: The lowercase task string, for example
            ``"sentiment_analysis"``.
        :rtype: str
        """
        return self.value

    def __repr__(self) -> str:
        """
        Return the qualified member name for debugging.

        :return: A string such as ``"NLPTaskType.SENTIMENT_ANALYSIS"``.
        :rtype: str
        """
        return f"NLPTaskType.{self.name}"


# ---------------------------------------------------------------------

@dataclass
class NLPTaskConfig:
    """
    Hyperparameters for one NLP task head.

    One instance describes one task. ``nlp/factory.py`` reads it to build the
    head. Fields that do not apply to the task type are ignored by the head.

    :meth:`__post_init__` runs three fixups after construction, shown below.

    **Architecture Overview:**

    .. code-block:: text

        NLPTaskConfig(name, task_type, ...)
                    │
                    ▼
        ┌──────────────────────────────┐
        │ mirror num_labels/num_classes│
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐        ┌────────────┐
        │ classification task without  │─ yes ─►│ ValueError │
        │ num_classes?                 │        └────────────┘
        └──────────────┬───────────────┘
                       │ no
                       ▼
        ┌──────────────────────────────┐
        │ generation task and          │
        │ vocabulary_size is None?     │─ yes ─► set it to 32000
        └──────────────┬───────────────┘
                       │ no
                       ▼
                  ready to use

    :param name: Unique identifier for the task.
    :type name: str
    :param task_type: Type of NLP task.
    :type task_type: NLPTaskType
    :param num_classes: Number of output classes (for classification tasks).
    :type num_classes: Optional[int]
    :param num_labels: Alternative to num_classes for compatibility.
    :type num_labels: Optional[int]
    :param max_length: Maximum sequence length.
    :type max_length: int
    :param dropout_rate: Dropout rate for task-specific head.
    :type dropout_rate: float
    :param hidden_size: Hidden dimension for task head.
    :type hidden_size: Optional[int]
    :param loss_weight: Weight for this task's loss in multi-task training.
    :type loss_weight: float
    :param label_smoothing: Label smoothing parameter.
    :type label_smoothing: float
    :param use_crf: Whether to use CRF for sequence labeling.
    :type use_crf: bool
    :param use_attention_pooling: Use attention-based pooling.
    :type use_attention_pooling: bool
    :param vocabulary_size: Size of vocabulary (for generation tasks).
    :type vocabulary_size: Optional[int]
    :param beam_size: Beam size for generation tasks.
    :type beam_size: int
    :param temperature: Temperature for generation sampling.
    :type temperature: float
    """
    name: str
    task_type: NLPTaskType
    num_classes: Optional[int] = None
    # Alternative spelling of num_classes; __post_init__ keeps the two equal.
    num_labels: Optional[int] = None
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
        """
        Reconcile the label fields, then validate and fill defaults.

        ``num_classes`` and ``num_labels`` end up equal when either was given.
        Classification tasks must carry a class count. Generation tasks get a
        default vocabulary size when none was supplied.

        :return: Nothing. The instance is modified in place.
        :rtype: None
        :raises ValueError: If the task type is a classification task and
            ``num_classes`` is still None after the label fields are
            reconciled.
        """
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

        # Set vocabulary size for generation tasks.
        #
        # DECISION plan-2026-08-23T203721-009b7ccf/D-022
        # TEXT_COMPLETION is listed here: without it `create_bert_with_head`
        # died with `ValueError: vocabulary_size must be specified for
        # generation tasks`. Do NOT drop MACHINE_TRANSLATION to match
        # `get_head_class`: callers skipping it need it. See decisions.md D-022.
        generation_tasks = [
            NLPTaskType.TEXT_GENERATION,
            NLPTaskType.MASKED_LANGUAGE_MODELING,
            NLPTaskType.TEXT_SUMMARIZATION,
            NLPTaskType.TEXT_COMPLETION,
            NLPTaskType.MACHINE_TRANSLATION,
        ]

        if self.task_type in generation_tasks and self.vocabulary_size is None:
            self.vocabulary_size = 32000


# ---------------------------------------------------------------------

class NLPTaskConfiguration:
    """
    A validated set of tasks for a multi-task NLP model.

    Construction checks the list, then stores it as a set. The query methods
    answer what that set needs: token-level outputs, paired inputs,
    generation, and a sequence length wide enough for every member.

    Only two task pairs are rejected as incompatible. The check is a guard
    against two known-bad combinations, not a full compatibility model.

    **Architecture Overview:**

    .. code-block:: text

        tasks: List[NLPTaskType]
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
             self._tasks: Set[NLPTaskType]
                   │
                   ├─► requires_token_level()
                   ├─► requires_sequence_pair()
                   ├─► requires_generation()
                   ├─► get_max_sequence_length()
                   └─► to_dict()

    :param tasks: List of NLPTaskType enum values to enable.
    :type tasks: List[NLPTaskType]
    :param validate_compatibility: Whether to validate task compatibility.
    :type validate_compatibility: bool
    :raises ValueError: If tasks list is empty, contains duplicates, or
        contains incompatible tasks (when validation enabled).
    """

    def __init__(
            self,
            tasks: List[NLPTaskType],
            validate_compatibility: bool = True
    ):
        """
        Initialize task configuration.

        :param tasks: List of NLPTaskType enum values to enable.
        :type tasks: List[NLPTaskType]
        :param validate_compatibility: Whether to validate task compatibility.
        :type validate_compatibility: bool
        :raises ValueError: If tasks list is empty, contains duplicates, or
            contains incompatible tasks (when validation enabled).
        """
        if not tasks:
            raise ValueError("At least one task must be specified")

        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate tasks found in configuration")

        self._tasks: Set[NLPTaskType] = set(tasks)

        if validate_compatibility and len(tasks) > 1:
            self._validate_task_compatibility()

    def _validate_task_compatibility(self) -> None:
        """
        Reject the two task pairs known not to work together.

        :return: Nothing.
        :rtype: None
        :raises ValueError: If both members of a rejected pair are enabled.
        """
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
        """
        Return the enabled tasks.

        The returned set is a copy, so mutating it does not change this
        configuration.

        :return: The enabled task types.
        :rtype: Set[NLPTaskType]
        """
        return self._tasks.copy()

    def has_task(self, task: NLPTaskType) -> bool:
        """
        Report whether one task is enabled.

        :param task: The task to look for.
        :type task: NLPTaskType
        :return: True when the task is in this configuration.
        :rtype: bool
        """
        return task in self._tasks

    def requires_token_level(self) -> bool:
        """
        Report whether any enabled task needs per-token outputs.

        Five task types count: token classification, named entity
        recognition, part-of-speech tagging, dependency parsing and semantic
        role labeling.

        :return: True when at least one of those is enabled.
        :rtype: bool
        """
        token_tasks = {
            NLPTaskType.TOKEN_CLASSIFICATION,
            NLPTaskType.NAMED_ENTITY_RECOGNITION,
            NLPTaskType.PART_OF_SPEECH_TAGGING,
            NLPTaskType.DEPENDENCY_PARSING,
            NLPTaskType.SEMANTIC_ROLE_LABELING,
        }
        return bool(self._tasks & token_tasks)

    def requires_sequence_pair(self) -> bool:
        """
        Report whether any enabled task takes two sequences as input.

        Four task types count: natural language inference, text similarity,
        paraphrase detection and question answering.

        :return: True when at least one of those is enabled.
        :rtype: bool
        """
        pair_tasks = {
            NLPTaskType.NATURAL_LANGUAGE_INFERENCE,
            NLPTaskType.TEXT_SIMILARITY,
            NLPTaskType.PARAPHRASE_DETECTION,
            NLPTaskType.QUESTION_ANSWERING,
        }
        return bool(self._tasks & pair_tasks)

    def requires_generation(self) -> bool:
        """
        Report whether any enabled task produces text.

        Five task types count: text generation, summarization, completion,
        machine translation and dialogue generation. Masked language
        modeling is NOT one of them here, though
        :meth:`NLPTaskConfig.__post_init__` does treat it as generation when
        it fills in a default vocabulary size.

        :return: True when at least one of those is enabled.
        :rtype: bool
        """
        generation_tasks = {
            NLPTaskType.TEXT_GENERATION,
            NLPTaskType.TEXT_SUMMARIZATION,
            NLPTaskType.TEXT_COMPLETION,
            NLPTaskType.MACHINE_TRANSLATION,
            NLPTaskType.DIALOGUE_GENERATION,
        }
        return bool(self._tasks & generation_tasks)

    def get_max_sequence_length(self) -> int:
        """
        Return a sequence length wide enough for every enabled task.

        The floor is 512. Any generation task raises it to 1024, and text
        summarization raises it to 2048.

        :return: The maximum required sequence length in tokens.
        :rtype: int
        """
        # Floor, raised below when a task needs more.
        max_length = 512

        if self.requires_generation():
            max_length = max(max_length, 1024)

        if NLPTaskType.TEXT_SUMMARIZATION in self._tasks:
            max_length = max(max_length, 2048)

        return max_length

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the configuration to plain Python types.

        Tasks become their string values. The four derived capability
        answers are stored alongside them, so a reader does not have to
        recompute them.

        :return: A dictionary with keys ``tasks``, ``requires_token_level``,
            ``requires_sequence_pair``, ``requires_generation`` and
            ``max_sequence_length``.
        :rtype: Dict[str, Any]
        """
        return {
            "tasks": [task.value for task in self._tasks],
            "requires_token_level": self.requires_token_level(),
            "requires_sequence_pair": self.requires_sequence_pair(),
            "requires_generation": self.requires_generation(),
            "max_sequence_length": self.get_max_sequence_length(),
        }


# ---------------------------------------------------------------------

class CommonNLPTaskConfigurations:
    """
    Eleven ready-made :class:`NLPTaskConfiguration` presets.

    Every attribute is a class-level instance, built once at import time.
    They are shared, so treat them as read-only; :attr:`tasks` already
    returns a copy.

    **Presets:**

    The table lists each preset's task count and the four capability answers
    its configuration reports.

    .. code-block:: text

        Preset                     N  Token  Pair  Gen  MaxLen
        ------------------------  --  -----  ----  ---  ------
        TEXT_CLASSIFICATION_ONLY   1  no     no    no      512
        NER_ONLY                   1  yes    no    no      512
        SENTIMENT_ONLY             1  no     no    no      512
        QA_ONLY                    1  no     yes   no      512
        GENERATION_ONLY            1  no     no    yes    1024
        TOKEN_TASKS                3  yes    no    no      512
        CLASSIFICATION_SUITE       3  no     no    no      512
        INFORMATION_EXTRACTION     3  yes    no    no      512
        COMPREHENSION_SUITE        3  no     yes   no      512
        SIMILARITY_SUITE           3  no     yes   no      512
        GLUE_TASKS                 5  no     yes   no      512

        N      = len(tasks)
        Token  = requires_token_level()
        Pair   = requires_sequence_pair()
        Gen    = requires_generation()
        MaxLen = get_max_sequence_length()

    Example:
        >>> cfg = CommonNLPTaskConfigurations.GLUE_TASKS
        >>> len(cfg.tasks)
        5
        >>> cfg.requires_sequence_pair()
        True
    """

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