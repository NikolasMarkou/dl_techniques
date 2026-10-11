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
    than assumed. The head now lives in the model (``include_top=True``,
    plan-2026-10-11T004932-b83935e1), and its ``predict(x)`` reaches
    ``BERT.call`` with the batch's fully static ``TensorShape([2, 17])`` on a
    fresh model, while a symbolic call over ``keras.Input(shape=(None,))``
    never reaches ``BERT.call`` at all (Keras answers it from
    ``compute_output_shape``; re-measured on the new API). The functional head
    factory this replaced behaved the same way. One check therefore covers
    both entry paths.

(b) **``MULTIPLE_CHOICE`` with a head.** It built without error and produced
    a semantically wrong graph: ``MultipleChoiceHead`` scores a
    ``(batch, num_choices, hidden)`` tensor, the model can only feed it the
    token sequence ``(batch, seq_len, hidden)``, so ``logits`` came out
    ``(None, None)`` -- one score per TOKEN -- and ``num_classes`` was ignored.
    Step 4's bare-tensor contract made that output look MORE legitimate. The
    head builder (``build_model_task_head``) now refuses, matching the "NO
    SILENT FALLBACK" precedent ``get_head_class`` already set in the same
    subsystem.

Why each arm is not satisfied by construction:

``test_over_long_sequence_raises_on_the_bare_encoder``
``test_over_long_sequence_raises_through_the_head_factory``
    The two entry paths, separately. The headed arm is the one that was
    silently finite on GPU before this change, so it is the arm that matters;
    the bare-encoder arm proves the check is not head-specific.
``test_at_and_below_the_limit_are_unaffected``
    The anti-vacuity arm, and it is doing real work: a check written with
    ``>=`` instead of ``>`` would pass both raise arms above while breaking
    every model run at exactly its declared maximum length. It also pins that
    the accepted forward is FINITE, not merely non-raising.
``test_the_message_names_both_numbers``
    An actionable message is the point. A bare ``ValueError`` would satisfy the
    two raise arms.
``test_a_relaxed_retrace_still_refuses_an_over_long_batch``
    A step function retraced with a ``None`` sequence axis skips the static
    check; a run-time check (D-021 of plan-2026-10-11T004932-b83935e1) must
    fire there, with the same message.
``test_a_refused_first_predict_leaves_the_model_usable``
    The refusal must not poison Keras's cached step functions or the
    variables; later legal ``predict``/``fit`` calls must work.
``test_a_dynamic_sequence_axis_returns_silently``
    The check must return silently when the length is not statically known.
    This arm calls the validator DIRECTLY, because the obvious version ("the
    head model still builds") was measured VACUOUS -- see the arm's own
    docstring. It also pins that one headed model serves two different
    sequence lengths, i.e. nothing fixes the length at build.
``test_multiple_choice_is_refused_with_an_actionable_message``
    Pins (b), message included.
``test_the_refusal_is_specific_to_multiple_choice``
    Anti-vacuity for (b): the sibling task types must still build. A refusal
    that caught everything would pass the arm above.
``test_the_multiple_choice_head_itself_still_works_on_a_rank_3_input``
    The refusal is about feeding the head a token sequence, not about the
    head. If someone
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
from dl_techniques.models.language.bert import BERT, create_bert

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
    return create_bert(
        "tiny",
        include_top=True,
        task_config=NLPTaskConfig(
            name=task_type.name.lower(),
            task_type=task_type,
            num_classes=NUM_CLASSES,
        ),
        max_position_embeddings=MAX_POS,
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


def test_a_relaxed_retrace_still_refuses_an_over_long_batch():
    """Reviewer W6 of plan-2026-10-11T004932-b83935e1, D-021.

    After ``predict`` has seen two lengths, TensorFlow retraces the step
    function with the sequence axis relaxed to ``None`` and never re-enters
    Python for a new length, so the static check is skipped. Before D-021 the
    over-long batch then failed in the position-embedding gather on CPU
    (``InvalidArgumentError`` naming ``GatherV2``) and, on GPU, returned
    zeros-backed garbage. Now a run-time check fails first, with the D-023
    message. A Python ``ValueError`` cannot cross a graph boundary, so in that
    graph the type is TensorFlow's ``InvalidArgumentError``.
    """
    import tensorflow as tf

    model = _head_model()
    for seq_len in (5, MAX_POS - 4):
        model.predict(_feed(seq_len), verbose=0)
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError),
        match=f"exceeds max_position_embeddings={MAX_POS}",
    ):
        model.predict(_feed(MAX_POS + 1), verbose=0)
    # And the model still serves a legal length afterwards.
    predictions = model.predict(_feed(MAX_POS), verbose=0)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("headed", [True, False], ids=["headed", "headless"])
def test_a_refused_first_predict_leaves_the_model_usable(headed):
    """An over-long FIRST ``predict`` on a fresh model must not break the model.

    Measured before D-021: the refusal fired inside the first trace of
    Keras's cached step function, after ``build`` had created the variables
    in that trace, and every later ``predict`` or ``fit`` failed ("Structures
    don't have the same nested structure", then reshape errors on empty
    variables). Now ``build`` refuses before any variable exists and the
    cached step functions are dropped.
    """
    model = _head_model() if headed else _encoder()
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model.predict(_feed(MAX_POS + 1), verbose=0)

    for seq_len in (5, MAX_POS - 4):
        out = model.predict(_feed(seq_len), verbose=0)
        hidden = out if headed else out["last_hidden_state"]
        assert hidden.shape[:2] == ((BATCH, NUM_CLASSES) if headed else (BATCH, seq_len))
        assert np.isfinite(hidden).all()

    if headed:
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        history = model.fit(_feed(8), np.zeros((BATCH,), "int32"), verbose=0)
        assert np.isfinite(history.history["loss"]).all()


def test_a_dynamic_sequence_axis_returns_silently(monkeypatch):
    """A length that is not statically known has nothing to check.

    This arm calls the validator DIRECTLY, and that is a correction to how it
    was first written. The obvious version -- "the head model still builds" --
    is VACUOUS, measured: an injection that raised on a ``None`` length left
    the whole module at ``13 passed``, because Keras answers a symbolic call
    through ``compute_output_spec`` and never reaches ``BERT.call``
    symbolically at all. Re-measured on the in-model head (2026-10-11): a
    probe patched onto ``BERT.call`` records ZERO invocations during a
    symbolic call over ``keras.Input(shape=(None,))``, then one with a static
    ``TensorShape([2, 5])`` at the first ``predict``. The zero is asserted
    below, so the reason for the direct call is pinned, not just told. So the
    dynamic branch is only reachable directly, and only a direct call can
    guard it.

    What the old factory arm protected -- a head model over a dynamic
    sequence axis builds, and the length is not fixed at build -- is asserted
    on the new API: the symbolic call gives ``(None, NUM_CLASSES)``, and one
    model predicts at two different lengths.
    """
    encoder = _encoder()
    symbolic = keras.Input(shape=(None,), dtype="int32")
    assert symbolic.shape[-1] is None, symbolic.shape

    # Must return the input unchanged, not raise. `int(None)` would be a TypeError.
    assert encoder._validate_sequence_length(symbolic) is symbolic

    model = _head_model()
    calls = []
    real_call = BERT.call

    def counting_call(self, *args, **kwargs):
        calls.append(1)
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(BERT, "call", counting_call)
    output = model({
        name: keras.Input(shape=(None,), dtype="int32", name=name)
        for name in ("input_ids", "attention_mask", "token_type_ids")
    })
    assert not calls, "a symbolic call now reaches BERT.call; see docstring"
    assert tuple(output.shape) == (None, NUM_CLASSES), output.shape
    monkeypatch.setattr(BERT, "call", real_call)

    for seq_len in (5, MAX_POS - 4):
        predictions = model.predict(_feed(seq_len), verbose=0)
        assert predictions.shape == (BATCH, NUM_CLASSES), predictions.shape


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
    """The head is fine; only a model's token-sequence wiring for it was not.

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
