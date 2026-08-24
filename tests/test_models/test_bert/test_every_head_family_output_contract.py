"""The output contract of EVERY reachable NLP head family, through the factory.

Step 4 of this plan changed what ``create_bert_with_head`` returns: a head dict
with a single informative key is unwrapped to a bare tensor. Only three task
types were covered by any test at the time (``SENTIMENT_ANALYSIS``,
``NAMED_ENTITY_RECOGNITION``, ``QUESTION_ANSWERING``), so the change silently
moved the contract of a fourth family and nothing noticed.

The family that moved is ``TextSimilarityHead``. Fed the
``{hidden_states, attention_mask}`` dict this factory always builds, it takes
its SINGLE-sequence branch and returns ``{'embeddings': ...}`` -- one key, no
``logits`` -- so the unwrap collapses it to a bare ``(None, embedding_dim)``
tensor. Before step 4 it was ``{'embeddings': tensor}``. The ``D-018`` anchor
asserted the opposite ("TextSimilarityHead emits three independent tensors;
those keep their dicts"); it named the head's PAIR branch, which this factory
cannot reach. The anchor and the docstring were corrected rather than the code:
a one-key dict carries exactly one informative tensor, and leaving it wrapped
would make the model non-compilable with plain ``metrics=[...]`` for no gain.

This module pins all 24 reachable families at once, so the next head added to
``layers/heads/nlp/factory.py`` cannot change any of them unnoticed.

Why each arm is not satisfied by construction:

``test_the_output_contract_of_every_reachable_task_type``
    The table below is a MEASURED expectation, one row per task type, naming
    the container (bare tensor vs dict) AND the shape. A change that keeps the
    container but moves the shape -- e.g. a similarity head that started
    emitting ``logits`` instead of ``embeddings`` -- still fails.
``test_the_reachable_set_is_exactly_what_the_table_covers``
    The anti-rot arm. Adding a row to ``get_head_class``'s ``head_mapping``
    without adding it here fails, so the table cannot silently stop being a
    census.
``test_the_two_dict_families_are_exactly_the_span_heads``
    States the RULE, not just the rows: exactly two families keep a dict, and
    they are the two span heads. If a future edit restricted the unwrap to
    logits-bearing dicts, the three similarity families would rejoin the dict
    set and this arm would fail with them named.
``test_every_generation_task_gets_a_vocabulary_size``
    A defect this module found, not the review. ``NLPTaskConfig.__post_init__``
    keeps its OWN list of "generation tasks" that need a default
    ``vocabulary_size``, and it disagreed with ``get_head_class``: it carried
    ``MACHINE_TRANSLATION`` (which ``get_head_class`` refuses outright) and
    omitted ``TEXT_COMPLETION`` (which it routes to ``TextGenerationHead``). So
    ``create_bert_with_head(task_type=TEXT_COMPLETION)`` raised
    ``ValueError: vocabulary_size must be specified for generation tasks``
    while its three siblings built fine. This arm CHECKS the lockstep instead
    of trusting it.
``test_a_bare_tensor_model_compiles_and_predicts``
    The reason the contract matters at all: the bare-tensor form is what makes
    ``metrics=['accuracy']`` compile and ``predict(x).shape`` read.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.heads.nlp.factory import (
    QuestionAnsweringHead,
    TextGenerationHead,
    get_head_class,
)
from dl_techniques.models.language.bert import create_bert_with_head

VARIANT = "tiny"
NUM_CLASSES = 3

# MEASURED on GPU 1, one row per reachable NLPTaskType. `T` marks a bare
# tensor, `D` a dict. The shapes are the KerasTensor shapes of `model.output`
# for `create_bert_with_head(bert_variant='tiny', num_classes=3)`.
#
# `None` in a shape is a dynamic axis (batch, or sequence length). The
# similarity rows' 256 is the head's default embedding dim; the generation
# rows' 32000 is `NLPTaskConfig`'s default vocabulary size.
_QA_DICT = {"start_logits": (None, None), "end_logits": (None, None)}

# Sentinel for a task type that is reachable through `get_head_class` but that
# `create_bert_with_head` deliberately REFUSES to build.
RAISES = "RAISES"

EXPECTED_CONTRACT = {
    # TokenClassificationHead -- one logit vector per token
    NLPTaskType.TOKEN_CLASSIFICATION: (None, None, NUM_CLASSES),
    NLPTaskType.NAMED_ENTITY_RECOGNITION: (None, None, NUM_CLASSES),
    NLPTaskType.PART_OF_SPEECH_TAGGING: (None, None, NUM_CLASSES),
    NLPTaskType.SEQUENCE_LABELING: (None, None, NUM_CLASSES),
    # TextClassificationHead -- one logit vector per sequence
    NLPTaskType.TEXT_CLASSIFICATION: (None, NUM_CLASSES),
    NLPTaskType.SENTIMENT_ANALYSIS: (None, NUM_CLASSES),
    NLPTaskType.EMOTION_DETECTION: (None, NUM_CLASSES),
    NLPTaskType.INTENT_CLASSIFICATION: (None, NUM_CLASSES),
    NLPTaskType.TOPIC_CLASSIFICATION: (None, NUM_CLASSES),
    NLPTaskType.SPAM_DETECTION: (None, NUM_CLASSES),
    NLPTaskType.NATURAL_LANGUAGE_INFERENCE: (None, NUM_CLASSES),
    # Regression-flavoured task types route to the SAME classification head and
    # honour num_classes; they do NOT force a scalar output.
    NLPTaskType.TEXT_REGRESSION: (None, NUM_CLASSES),
    NLPTaskType.READABILITY_SCORING: (None, NUM_CLASSES),
    NLPTaskType.QUALITY_SCORING: (None, NUM_CLASSES),
    # QuestionAnsweringHead -- the ONLY families that keep a dict
    NLPTaskType.QUESTION_ANSWERING: _QA_DICT,
    NLPTaskType.SPAN_EXTRACTION: _QA_DICT,
    # TextSimilarityHead -- BARE EMBEDDINGS, not logits, not a dict. This is
    # the contract step 4 moved and the D-018 anchor described backwards.
    NLPTaskType.TEXT_SIMILARITY: (None, 256),
    NLPTaskType.PARAPHRASE_DETECTION: (None, 256),
    NLPTaskType.DUPLICATE_DETECTION: (None, 256),
    # TextGenerationHead -- vocabulary logits per position
    NLPTaskType.TEXT_GENERATION: (None, None, 32000),
    NLPTaskType.MASKED_LANGUAGE_MODELING: (None, None, 32000),
    NLPTaskType.TEXT_SUMMARIZATION: (None, None, 32000),
    # TEXT_COMPLETION used to RAISE here; see the D-022 anchor in task_types.py
    NLPTaskType.TEXT_COMPLETION: (None, None, 32000),
    # MultipleChoiceHead -- REFUSED by the factory (step 13.3, D-024). It is
    # reachable through get_head_class but cannot be built here, because this
    # factory can only feed it a token sequence. `RAISES` is the sentinel; the
    # message itself is pinned by
    # tests/test_models/test_bert/test_sequence_length_and_multiple_choice.py.
    NLPTaskType.MULTIPLE_CHOICE: RAISES,
}


def _reachable_task_types():
    """Every ``NLPTaskType`` for which ``get_head_class`` returns a head."""
    reachable = []
    for task_type in NLPTaskType:
        try:
            get_head_class(task_type)
        except ValueError:
            continue
        reachable.append(task_type)
    return reachable


REACHABLE = _reachable_task_types()


def _build(task_type: NLPTaskType) -> keras.Model:
    return create_bert_with_head(
        bert_variant=VARIANT,
        task_config=NLPTaskConfig(
            name=task_type.name.lower(),
            task_type=task_type,
            num_classes=NUM_CLASSES,
        ),
    )


def _describe(output) -> object:
    if isinstance(output, dict):
        return {key: tuple(value.shape) for key, value in output.items()}
    return tuple(output.shape)


@pytest.mark.parametrize("task_type", REACHABLE, ids=lambda t: t.name)
def test_the_output_contract_of_every_reachable_task_type(task_type):
    """Each family's ``model.output`` container AND shape are pinned."""
    assert task_type in EXPECTED_CONTRACT, (
        f"{task_type.name} is reachable through get_head_class but has no row "
        "in EXPECTED_CONTRACT; add one with its MEASURED contract"
    )
    expected = EXPECTED_CONTRACT[task_type]
    if expected is RAISES:
        with pytest.raises(ValueError):
            _build(task_type)
        return

    model = _build(task_type)
    actual = _describe(model.output)

    assert actual == expected, (
        f"{task_type.name} ({get_head_class(task_type).__name__}) output "
        f"contract moved: expected {expected}, measured {actual}"
    )


def test_the_reachable_set_is_exactly_what_the_table_covers():
    """The table must stay a census, not a sample."""
    uncovered = sorted(t.name for t in REACHABLE if t not in EXPECTED_CONTRACT)
    assert not uncovered, (
        f"reachable task type(s) {uncovered} have no pinned contract"
    )
    stale = sorted(
        t.name for t in EXPECTED_CONTRACT if t not in REACHABLE
    )
    assert not stale, (
        f"EXPECTED_CONTRACT pins {stale}, which get_head_class no longer routes"
    )


def test_the_two_dict_families_are_exactly_the_span_heads():
    """State the RULE: only genuinely multi-tensor heads keep a dict."""
    # Derived from LIVE builds, not from EXPECTED_CONTRACT. Reading the table
    # would make this a table-consistency check that a code change cannot move
    # -- measured: under an injection that restricted the unwrap to
    # logits-bearing dicts, the table-reading version stayed GREEN while three
    # families had rejoined the dict set.
    dict_families = sorted(
        t.name
        for t in REACHABLE
        if EXPECTED_CONTRACT[t] is not RAISES and isinstance(_build(t).output, dict)
    )
    assert dict_families == ["QUESTION_ANSWERING", "SPAN_EXTRACTION"], (
        f"the set of dict-output families changed to {dict_families}. If the "
        "similarity families are back, the unwrap was restricted to "
        "logits-bearing dicts -- that was considered and rejected; see the "
        "D-018 anchor in models/language/bert/model.py"
    )
    for name in dict_families:
        task_type = NLPTaskType[name]
        assert get_head_class(task_type) is QuestionAnsweringHead


def test_every_generation_task_gets_a_vocabulary_size():
    """``NLPTaskConfig``'s generation list must cover every generation head.

    Two hand-maintained lists used to disagree, and the disagreement was a
    hard raise for one public task type. This arm checks the lockstep in the
    direction that matters: every type routed to ``TextGenerationHead`` must
    come out of ``__post_init__`` with a vocabulary.
    """
    missing = []
    for task_type in REACHABLE:
        if get_head_class(task_type) is not TextGenerationHead:
            continue
        config = NLPTaskConfig(
            name=task_type.name.lower(),
            task_type=task_type,
            num_classes=NUM_CLASSES,
        )
        if config.vocabulary_size is None:
            missing.append(task_type.name)

    assert not missing, (
        f"{missing} route to TextGenerationHead but NLPTaskConfig leaves "
        "vocabulary_size=None, so create_bert_with_head raises "
        "'vocabulary_size must be specified for generation tasks' for them. "
        "Add them to generation_tasks in layers/heads/nlp/task_types.py "
        "(see the D-022 anchor there)."
    )


def test_a_bare_tensor_model_compiles_and_predicts():
    """The point of the contract: a bare tensor is compilable and indexable."""
    model = _build(NLPTaskType.SENTIMENT_ANALYSIS)
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    batch, seq_len = 2, 6
    ids = np.random.RandomState(0).randint(0, 100, (batch, seq_len)).astype("int32")
    inputs = {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "token_type_ids": np.zeros_like(ids),
    }
    predictions = model.predict(inputs, verbose=0)
    assert predictions.shape == (batch, NUM_CLASSES), predictions.shape


def test_the_similarity_families_are_bare_embeddings_not_logits():
    """The specific contract the D-018 anchor described backwards.

    Kept as its own named arm rather than left implicit in the table, because
    this is the one family whose contract step 4 actually moved, and a reader
    chasing the anchor's old claim should land on an assertion that states the
    measured truth.
    """
    for task_type in (
        NLPTaskType.TEXT_SIMILARITY,
        NLPTaskType.PARAPHRASE_DETECTION,
        NLPTaskType.DUPLICATE_DETECTION,
    ):
        output = _build(task_type).output
        assert not isinstance(output, dict), (
            f"{task_type.name} returned a dict {list(output)}; the "
            "single-key unwrap no longer covers {'embeddings'}"
        )
        assert len(output.shape) == 2 and output.shape[-1] == 256, (
            f"{task_type.name} output shape {tuple(output.shape)} is not the "
            "measured (None, embedding_dim) embeddings tensor"
        )
