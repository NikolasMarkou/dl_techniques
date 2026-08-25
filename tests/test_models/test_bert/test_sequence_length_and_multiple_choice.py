"""Two silent-wrong-answer paths in BERT, both now loud.

(a) **Sequences longer than ``max_position_embeddings``.** BERT's position
    embeddings are a learned table of ``max_position_embeddings`` rows. Nothing
    checked the input length, so an over-long sequence produced an
    out-of-range position gather -- and the outcome depended on the DEVICE.
    MEASURED at ``max_position_embeddings=16``, ``seq_len=17``: on GPU 1 the
    forward returned a finite ``(2, 3)`` with no warning (the gather reads
    zeros); on CPU the same call raised ``InvalidArgumentError`` from
    ``Embedding.call()``. A silent wrong answer on the device this repo trains
    on is the worst of the three outcomes.

    The check lives in ``BERT.call``, and that placement was re-derived rather
    than assumed. Keras 3 executes a Functional graph by calling each
    operation's ``call()`` with REAL tensors, so
    ``create_bert_with_head(...).predict(x)`` reaches ``BERT.call`` with a
    fully static ``TensorShape([2, 17])`` even though the graph was BUILT over
    ``keras.Input(shape=(None,))``. One check therefore covers both entry
    paths; a check in the factory would have covered neither, because the
    factory only ever sees the symbolic ``None`` sequence axis.

(b) **``MULTIPLE_CHOICE`` through ``create_bert_with_head``.** It built without
    error and produced a semantically wrong graph: ``MultipleChoiceHead``
    scores a ``(batch, num_choices, hidden)`` tensor, the factory can only feed
    it the token sequence ``(batch, seq_len, hidden)``, so ``logits`` came out
    ``(None, None)`` -- one score per TOKEN -- and ``num_classes`` was ignored.
    Step 4's bare-tensor contract made that output look MORE legitimate. The
    factory now refuses, matching the "NO SILENT FALLBACK" precedent
    ``get_head_class`` already set in the same subsystem.

Why each arm is not satisfied by construction:

``test_over_long_sequence_raises_on_the_bare_encoder``
``test_over_long_sequence_raises_through_the_head_factory``
    The two entry paths, separately. The factory arm is the one that was
    silently finite on GPU before this change, so it is the arm that matters;
    the bare-encoder arm proves the check is not factory-specific.
``test_at_and_below_the_limit_are_unaffected``
    The anti-vacuity arm, and it is doing real work: a check written with
    ``>=`` instead of ``>`` would pass both raise arms above while breaking
    every model run at exactly its declared maximum length. It also pins that
    the accepted forward is FINITE, not merely non-raising.
``test_the_message_names_both_numbers``
    An actionable message is the point. A bare ``ValueError`` would satisfy the
    two raise arms.
``test_a_dynamic_sequence_axis_returns_silently``
    The check must return silently when the length is not statically known.
    This arm calls the validator DIRECTLY, because the obvious version ("the
    factory still builds") was measured VACUOUS -- see the arm's own docstring.
``test_multiple_choice_is_refused_with_an_actionable_message``
    Pins (b), message included.
``test_the_refusal_is_specific_to_multiple_choice``
    Anti-vacuity for (b): the sibling task types must still build. A refusal
    that caught everything would pass the arm above.
``test_the_multiple_choice_head_itself_still_works_on_a_rank_3_input``
    The refusal is about THIS FACTORY, not about the head. If someone
    "simplified" this by deleting or deprecating the head, this arm fails.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    NLPTaskType,
    create_nlp_head,
)
from dl_techniques.models.language.bert import BERT, create_bert_with_head

MAX_POS = 16
NUM_CLASSES = 3
BATCH = 2


def _ids(seq_len: int) -> np.ndarray:
    rng = np.random.RandomState(7)
    return rng.randint(0, 100, (BATCH, seq_len)).astype("int32")


def _feed(seq_len: int) -> dict:
    ids = _ids(seq_len)
    return {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "token_type_ids": np.zeros_like(ids),
    }


def _encoder() -> BERT:
    return BERT.from_variant("tiny", max_position_embeddings=MAX_POS)


def _head_model(task_type=NLPTaskType.SENTIMENT_ANALYSIS) -> keras.Model:
    return create_bert_with_head(
        bert_variant="tiny",
        task_config=NLPTaskConfig(
            name=task_type.name.lower(),
            task_type=task_type,
            num_classes=NUM_CLASSES,
        ),
        bert_config_overrides={"max_position_embeddings": MAX_POS},
    )


# ---------------------------------------------------------------------------
# (a) sequence length
# ---------------------------------------------------------------------------


def test_over_long_sequence_raises_on_the_bare_encoder():
    with pytest.raises(ValueError, match="max_position_embeddings"):
        _encoder()({"input_ids": _ids(MAX_POS + 1)})


def test_over_long_sequence_raises_through_the_head_factory():
    """The path that was SILENTLY FINITE on GPU before this change."""
    model = _head_model()
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model.predict(_feed(MAX_POS + 1), verbose=0)


# seq_len=1 is deliberately NOT here: a length-1 sequence makes the attention
# softmax run over an axis of size 1, and Keras emits a UserWarning that this
# suite promotes to an error. That is unrelated to the length guard.
@pytest.mark.parametrize("seq_len", [2, MAX_POS - 1, MAX_POS])
def test_at_and_below_the_limit_are_unaffected(seq_len):
    """Anti-vacuity: a ``>=`` off-by-one would break the declared maximum."""
    encoder_out = _encoder()({"input_ids": _ids(seq_len)})
    hidden = keras.ops.convert_to_numpy(encoder_out["last_hidden_state"])
    assert hidden.shape[:2] == (BATCH, seq_len), hidden.shape
    assert np.isfinite(hidden).all()

    predictions = _head_model().predict(_feed(seq_len), verbose=0)
    assert predictions.shape == (BATCH, NUM_CLASSES), predictions.shape
    assert np.isfinite(predictions).all()


def test_the_message_names_both_numbers():
    with pytest.raises(ValueError) as excinfo:
        _encoder()({"input_ids": _ids(MAX_POS + 3)})
    message = str(excinfo.value)
    assert f"seq_len={MAX_POS + 3}" in message, message
    assert f"max_position_embeddings={MAX_POS}" in message, message


def test_a_dynamic_sequence_axis_returns_silently():
    """A length that is not statically known has nothing to check.

    This arm calls the validator DIRECTLY, and that is a correction to how it
    was first written. The obvious version -- "``_head_model()`` still
    builds" -- is VACUOUS, measured: an injection that raised on a ``None``
    length left the whole module at ``13 passed``, because Keras builds the
    Functional graph through ``compute_output_spec`` and never reaches
    ``BERT.call`` symbolically at all (also measured: a probe patched onto
    ``BERT.call`` records ZERO invocations during ``create_bert_with_head``,
    then exactly one, with a fully static ``TensorShape([2, 17])``, during
    ``predict``). So the dynamic branch is only reachable directly, and only a
    direct call can guard it.
    """
    encoder = _encoder()
    symbolic = keras.Input(shape=(None,), dtype="int32")
    assert symbolic.shape[-1] is None, symbolic.shape

    # Must return None, not raise. `int(None)` would be a TypeError.
    assert encoder._validate_sequence_length(symbolic) is None

    # And the factory, which only ever sees this shape, still builds.
    model = _head_model()
    assert model.inputs[0].shape[1] is None, model.inputs[0].shape
    assert model.output.shape[-1] == NUM_CLASSES


# ---------------------------------------------------------------------------
# (b) MULTIPLE_CHOICE
# ---------------------------------------------------------------------------


def test_multiple_choice_is_refused_with_an_actionable_message():
    with pytest.raises(ValueError) as excinfo:
        _head_model(NLPTaskType.MULTIPLE_CHOICE)
    message = str(excinfo.value)
    assert "num_choices" in message, message
    assert "create_nlp_head" in message, message


@pytest.mark.parametrize(
    "task_type",
    [
        NLPTaskType.SENTIMENT_ANALYSIS,
        NLPTaskType.NAMED_ENTITY_RECOGNITION,
        NLPTaskType.QUESTION_ANSWERING,
        NLPTaskType.TEXT_SIMILARITY,
    ],
    ids=lambda t: t.name,
)
def test_the_refusal_is_specific_to_multiple_choice(task_type):
    """Anti-vacuity: the refusal must not catch its siblings."""
    assert _head_model(task_type) is not None


def test_the_multiple_choice_head_itself_still_works_on_a_rank_3_input():
    """The head is fine; only this factory's wiring for it was not.

    Pins the alternative the refusal message tells the caller to use, so the
    message cannot become a lie.
    """
    num_choices, hidden = 4, 32
    head = create_nlp_head(
        task_config=NLPTaskConfig(
            name="multiple_choice",
            task_type=NLPTaskType.MULTIPLE_CHOICE,
            num_classes=num_choices,
        ),
        input_dim=hidden,
    )
    choices = keras.ops.convert_to_tensor(
        np.random.RandomState(3).randn(BATCH, num_choices, hidden).astype("float32")
    )
    logits = head({"hidden_states": choices})["logits"]
    assert tuple(logits.shape) == (BATCH, num_choices), tuple(logits.shape)
