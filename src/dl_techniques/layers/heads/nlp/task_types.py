"""Task types and configuration objects for the NLP heads.

This module says what an NLP task is. It builds no layer. Five objects live
here:

* :class:`NLPTaskType` -- an enum of 37 task names, with lookup helpers for
  categories, compatible tasks, output shapes and input requirements.
* :data:`NLP_HEAD_DISPATCH` -- the one table from task type to the name of
  the head class that serves it (24 rows; the other 13 task types have none).
* :class:`NLPTaskConfig` -- a dataclass holding the hyperparameters of one
  task head, validated against the head its task type is routed to.
* :class:`NLPTaskConfiguration` -- a validated set of tasks, plus queries for
  the capabilities that set needs.
* :class:`CommonNLPTaskConfigurations` -- eleven ready-made task sets.

``nlp/factory.py`` reads these objects and builds the head the table names.
The types name tasks, not models, so heads on BERT, GPT and T5 all take the
same config.
"""

import dataclasses
import numbers
from enum import Enum, unique
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

from ..validation import positive_int

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

    :data:`NLP_HEAD_DISPATCH`, below this class, decides that: 24 members are
    served by one of six head classes and the other 13 have no head, so
    asking for one raises ``ValueError`` rather than falling back to a
    default. The table is not restated here.

    Example:
        >>> task = NLPTaskType.from_string("sentiment_analysis")
        >>> str(task)
        'sentiment_analysis'
        >>> NLPTaskType.get_output_types(task)
        {'logits': 'float32[B, C]', 'probabilities': 'float32[B, C]'}
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
        Name the tensors the head serving ``task`` returns.

        The keys are exactly the keys of that head's output dict for one
        sequence input, at its defaults; shapes are strings, with ``B`` batch,
        ``L`` sequence length, ``C`` classes, ``H`` the head's ``hidden_size``,
        ``V`` vocabulary and ``N`` choices. Two heads have a second form,
        named in the docstring of each: ``TextSimilarityHead`` given a pair
        returns ``'similarity_score'`` ``[B]``, plus ``'embeddings_1'`` and
        ``'embeddings_2'`` with ``output_embeddings``; a
        ``TokenClassificationHead`` with ``use_crf`` returns ``'logits'``
        only.

        :param task: The task to look up.
        :type task: NLPTaskType
        :return: Output name mapped to its shape string (a new dict).
        :rtype: Dict[str, str]
        :raises ValueError: If no head serves ``task`` (13 task types).
        """
        head = NLP_HEAD_DISPATCH.get(task)
        if head is None:
            raise ValueError(
                f"No NLP head serves {task.value!r}, so it has no output types. "
                f"Served task types: {sorted(t.value for t in NLP_HEAD_DISPATCH)}."
            )
        return dict(_OUTPUT_TYPES[head])

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

# DECISION plan-2026-10-14T062616-81938e81/D-064
# The one table from task type to NLP head. get_head_class, NLPTaskConfig's
# checks, get_output_types and model_head's generation set all read it. It
# names head classes instead of holding them because nlp/factory.py imports
# this module. Do NOT restate any part of it elsewhere, and do NOT add a row
# for a task type no head serves (13 have none; they raise). See decisions.md
# D-064 (and D-052, D-059, the vision and VLM tables).
NLP_HEAD_DISPATCH: Dict[NLPTaskType, str] = {
    NLPTaskType.TEXT_CLASSIFICATION: "TextClassificationHead",
    NLPTaskType.SENTIMENT_ANALYSIS: "TextClassificationHead",
    NLPTaskType.EMOTION_DETECTION: "TextClassificationHead",
    NLPTaskType.INTENT_CLASSIFICATION: "TextClassificationHead",
    NLPTaskType.TOPIC_CLASSIFICATION: "TextClassificationHead",
    NLPTaskType.SPAM_DETECTION: "TextClassificationHead",
    NLPTaskType.NATURAL_LANGUAGE_INFERENCE: "TextClassificationHead",
    # Regression: a classification head with num_classes=1; read 'logits'.
    NLPTaskType.TEXT_REGRESSION: "TextClassificationHead",
    NLPTaskType.READABILITY_SCORING: "TextClassificationHead",
    NLPTaskType.QUALITY_SCORING: "TextClassificationHead",
    NLPTaskType.TOKEN_CLASSIFICATION: "TokenClassificationHead",
    NLPTaskType.NAMED_ENTITY_RECOGNITION: "TokenClassificationHead",
    NLPTaskType.PART_OF_SPEECH_TAGGING: "TokenClassificationHead",
    NLPTaskType.SEQUENCE_LABELING: "TokenClassificationHead",
    NLPTaskType.QUESTION_ANSWERING: "QuestionAnsweringHead",
    NLPTaskType.SPAN_EXTRACTION: "QuestionAnsweringHead",
    NLPTaskType.TEXT_SIMILARITY: "TextSimilarityHead",
    NLPTaskType.PARAPHRASE_DETECTION: "TextSimilarityHead",
    NLPTaskType.DUPLICATE_DETECTION: "TextSimilarityHead",
    NLPTaskType.TEXT_GENERATION: "TextGenerationHead",
    NLPTaskType.MASKED_LANGUAGE_MODELING: "TextGenerationHead",
    NLPTaskType.TEXT_SUMMARIZATION: "TextGenerationHead",
    NLPTaskType.TEXT_COMPLETION: "TextGenerationHead",
    NLPTaskType.MULTIPLE_CHOICE: "MultipleChoiceHead",
}

#: The heads that cannot be built without ``num_classes``.
_HEADS_NEEDING_NUM_CLASSES = frozenset({"TextClassificationHead", "TokenClassificationHead"})

#: The vocabulary a generation task config gets when it states none.
DEFAULT_VOCABULARY_SIZE = 32000

#: The keys each head returns, with shape strings: ``B`` batch, ``L``
#: sequence length, ``C`` classes, ``H`` the head's ``hidden_size``, ``V``
#: vocabulary, ``N`` choices.
_OUTPUT_TYPES: Dict[str, Dict[str, str]] = {
    "TextClassificationHead": {
        "logits": "float32[B, C]",
        "probabilities": "float32[B, C]",
    },
    "TokenClassificationHead": {
        "logits": "float32[B, L, C]",
        "predictions": "int32[B, L]",
    },
    "QuestionAnsweringHead": {
        "start_logits": "float32[B, L]",
        "end_logits": "float32[B, L]",
    },
    "TextSimilarityHead": {
        "embeddings": "float32[B, H]",
    },
    "TextGenerationHead": {
        "logits": "float32[B, L, V]",
    },
    "MultipleChoiceHead": {
        "logits": "float32[B, N]",
        "probabilities": "float32[B, N]",
    },
}


@dataclass
class NLPTaskConfig:
    """
    Hyperparameters for one NLP task head.

    One instance describes one task. ``nlp/factory.py`` reads it to build the
    head that :data:`NLP_HEAD_DISPATCH` names for ``task_type``. Every field is
    read by some head; a field that has no reader does not exist here.

    :meth:`__post_init__` parses and validates every field, then applies the
    one requirement of the head the task type is routed to, shown below.

    **Architecture Overview:**

    .. code-block:: text

        NLPTaskConfig(name, task_type, ...)
                    │
                    ▼
        ┌──────────────────────────────┐        ┌────────────────────┐
        │ name, task_type (a string is │─ bad ─►│ ValueError or      │
        │ parsed), counts, dropout,    │        │ TypeError, naming  │
        │ use_crf valid?               │        │ the field          │
        └──────────────┬───────────────┘        └────────────────────┘
                       │ yes
                       ▼
        NLP_HEAD_DISPATCH[task_type]
                       │
          ┌────────────┼──────────────────┬──────────────┐
          ▼            ▼                  ▼              ▼
        Text/Token   TextGeneration    any other      no head
        Classif.     Head              head           (13 types)
          │            │                  │              │
          ▼            ▼                  ▼              ▼
        num_classes  vocabulary_size   nothing        nothing
        required,    None -> 32000     required       filled
        else
        ValueError

    :param name: Unique identifier for the task. A non-empty string.
    :type name: str
    :param task_type: Type of NLP task, as a member or its string value.
    :type task_type: NLPTaskType
    :param num_classes: Number of output classes. Required by the task types
        routed to ``TextClassificationHead`` or ``TokenClassificationHead``; a
        positive integer wherever it is set.
    :type num_classes: Optional[int]
    :param dropout_rate: Dropout rate inside the head, in ``[0, 1)``.
    :type dropout_rate: float
    :param hidden_size: Working width of the head. ``None`` means the head's
        ``intermediate_size``, else its ``input_dim``.
    :type hidden_size: Optional[int]
    :param use_crf: Read by ``TokenClassificationHead``: when set the head
        emits ``'logits'`` only, for a CRF the caller adds. No CRF layer is
        built in the head.
    :type use_crf: bool
    :param vocabulary_size: Size of the vocabulary a generation head scores.
        Filled with 32000 for a generation task type when unset.
    :type vocabulary_size: Optional[int]
    """
    name: str
    task_type: NLPTaskType
    num_classes: Optional[int] = None
    dropout_rate: float = 0.1
    hidden_size: Optional[int] = None
    use_crf: bool = False
    vocabulary_size: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Parse and validate every field, then apply the routed head's needs.

        :return: Nothing. The instance is normalized in place: a string
            ``task_type`` becomes its member, a numpy count a plain ``int``.
        :rtype: None
        :raises TypeError: If ``task_type`` is neither an :class:`NLPTaskType`
            nor a string.
        :raises ValueError: If ``name`` is not a non-empty string, a string
            ``task_type`` names no task, a set count is not a positive integer,
            ``dropout_rate`` is outside ``[0, 1)``, ``use_crf`` is not a bool,
            or the routed head needs ``num_classes`` and it is unset.
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"NLPTaskConfig: name must be a non-empty string, got {self.name!r}."
            )
        if isinstance(self.task_type, str):
            self.task_type = NLPTaskType.from_string(self.task_type)
        elif not isinstance(self.task_type, NLPTaskType):
            raise TypeError(
                "NLPTaskConfig: task_type must be an NLPTaskType or its string "
                f"value, got {type(self.task_type).__name__} {self.task_type!r}."
            )
        for field_name in ("num_classes", "hidden_size", "vocabulary_size"):
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name,
                        positive_int(value, field_name, "NLPTaskConfig"))
        if (isinstance(self.dropout_rate, bool)
                or not isinstance(self.dropout_rate, numbers.Real)
                or not 0.0 <= self.dropout_rate < 1.0):
            raise ValueError(
                "NLPTaskConfig: dropout_rate must be a number in [0, 1), got "
                f"{self.dropout_rate!r}."
            )
        self.dropout_rate = float(self.dropout_rate)
        if not isinstance(self.use_crf, bool):
            raise ValueError(
                f"NLPTaskConfig: use_crf must be a bool, got {self.use_crf!r}."
            )

        # DECISION plan-2026-10-14T062616-81938e81/D-064
        # The class-count requirement and the vocabulary default come from
        # NLP_HEAD_DISPATCH. Do NOT keep a list of task types here: two hand-kept
        # lists (5 classification, 5 generation) let nine routed task types
        # build a config their head refused, and gave MACHINE_TRANSLATION, which
        # has no head, a vocabulary. Supersedes the generation list of
        # plan-2026-08-23T203721-009b7ccf/D-022. See decisions.md D-064.
        head = NLP_HEAD_DISPATCH.get(self.task_type)
        if head in _HEADS_NEEDING_NUM_CLASSES and self.num_classes is None:
            raise ValueError(
                f"NLPTaskConfig: {self.task_type.value!r} is served by {head}, "
                "which requires num_classes (a positive integer); got None."
            )
        if head == "TextGenerationHead" and self.vocabulary_size is None:
            self.vocabulary_size = DEFAULT_VOCABULARY_SIZE

    def to_dict(self) -> Dict[str, Any]:
        """
        Return every field as plain JSON data.

        Interface contract, shared by every NLP head's ``get_config``,
        ``MultiTaskNLPHead.get_config`` and ``model_head.task_config_to_dict``:
        a new dict holding each dataclass field, with ``task_type`` as its
        string value. ``NLPTaskConfig(**config.to_dict()) == config``.

        :return: The fields, JSON-serializable.
        :rtype: Dict[str, Any]
        """
        fields = dataclasses.asdict(self)
        fields["task_type"] = self.task_type.value
        return fields

    @classmethod
    def coerce(cls, value: Union["NLPTaskConfig", Dict[str, Any]]) -> "NLPTaskConfig":
        """
        Return a new config built from an instance or a dict of its fields.

        Interface contract, shared by the NLP heads' constructors and the two
        factory functions: an instance is copied (the caller's object is never
        held, so editing it later cannot change a built head or its
        ``get_config``), and a dict is passed to the constructor, which parses
        a string ``task_type``. The dict is not modified.

        :param value: A config, or a dict of its fields.
        :type value: Union[NLPTaskConfig, Dict[str, Any]]
        :return: A new instance.
        :rtype: NLPTaskConfig
        :raises TypeError: If ``value`` is neither, or the dict holds a field
            the dataclass does not have.
        """
        if isinstance(value, cls):
            return dataclasses.replace(value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            "task_config must be an NLPTaskConfig or a dict of its fields, "
            f"got {type(value).__name__}."
        )


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