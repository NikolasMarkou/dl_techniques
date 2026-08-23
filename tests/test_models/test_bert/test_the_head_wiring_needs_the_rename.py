"""Hand-wiring a BERT encoder to an NLP head requires an explicit key RENAME.

Rationale
---------
`BERT` emits ``{"last_hidden_state", "attention_mask"}``. Every head in
``layers/heads/nlp/factory.py`` reads ``inputs['hidden_states']`` (``:468`` for
the classification head, ``:592`` for the token head). The names do not match,
and the mismatch does NOT surface where you would look for it: the heads'
``build()`` and ``compute_output_shape()`` use ``.get(..., default)``, so
``keras.Model(...)`` construction succeeds and the graph looks fine. It raises
only when ``call()`` actually runs.

BERT README §9 Pattern 2 walked the reader straight into that. It fed
``bert_outputs`` -- the encoder's dict -- directly into the heads. The step that
fixed the README is the reason this file exists: a README fix asserts the trap;
this file measures it.

MEASURED, ``BERT.from_variant("tiny", pretrained=False)`` + a
``TextClassificationHead`` and a ``TokenClassificationHead`` over a ``(2, 16)``
batch:

    WITH the rename      keras.Model(...).predict(...) -> {'sentiment': (2, 2),
                                                           'ner': (2, 16, 9)}
    WITHOUT the rename   keras.Model(...) construction SUCCEEDS
                         .predict(...) raises KeyError

and the raise text names both the missing key and the keys that WERE offered::

    KeyError: Exception encountered when calling TextClassificationHead.call().
    hidden_states
    Arguments received by TextClassificationHead.call():
      * inputs={'last_hidden_state': 'tf.Tensor(shape=(2, 16, 256), ...)',
                'attention_mask': 'tf.Tensor(shape=(2, 16), dtype=int32)'}
      * training=False

The fix is the rename at the call site, mirroring what
``create_bert_with_head`` already does internally (``bert.py:1205-1208``). It is
NOT a ``last_hidden_state`` alias on ``BaseNLPHead``: that class is the base of
every NLP head in the repo, and the README is the thing that was wrong. See
``decisions.md`` D-004 (plan-2026-08-23T203721-009b7ccf).

RED proof
---------
This file's own second test IS the RED arm -- it pins the failure the first test
would hit if the rename were dropped, so the pair cannot both pass against a
model that has stopped reading its inputs at all. Confirmed by deleting the
rename from the first test's graph (feeding ``encoder_outputs`` straight in):
1 failed, 2 passed, with the ACTUAL text

    KeyError: "Exception encountered when calling TextClassificationHead.call()
    ...\\n\\x1b[1mhidden_states\\x1b[0m\\n\\nArguments received by
    TextClassificationHead.call():\\n  * inputs={'last_hidden_state': ...

and confirmed in the other direction by giving ``BaseNLPHead.call`` the alias
the decision forbids (``inputs.get('hidden_states', inputs.get(
'last_hidden_state'))``): the RED arm goes GREEN-side-up as
``Failed: DID NOT RAISE <class 'KeyError'>``, 1 failed / 2 passed. So each arm
convicts a different mutation and neither carries the other.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.bert import BERT
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    NLPTaskType,
    TextClassificationHead,
    TokenClassificationHead,
)

# ---------------------------------------------------------------------

BATCH = 2
SEQ_LEN = 16
NUM_SENTIMENT_CLASSES = 2
NUM_NER_CLASSES = 9


def _inputs() -> dict:
    """``name=`` is load-bearing, not decoration.

    An unnamed ``keras.Input`` inside a dict is auto-named ``keras_tensor_N``,
    which then becomes the data key the model demands at ``fit``/``predict``
    time. Keras emits a UserWarning about it; the README's original snippet had
    all three unnamed, which is a second sign it was never executed.
    """
    return {
        "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(
            shape=(None,), dtype="int32", name="attention_mask"
        ),
        "token_type_ids": keras.Input(
            shape=(None,), dtype="int32", name="token_type_ids"
        ),
    }


def _heads(hidden_size: int):
    sentiment = TextClassificationHead(
        task_config=NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=NUM_SENTIMENT_CLASSES,
        ),
        input_dim=hidden_size,
        name="sentiment",
    )
    ner = TokenClassificationHead(
        task_config=NLPTaskConfig(
            name="ner",
            task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
            num_classes=NUM_NER_CLASSES,
        ),
        input_dim=hidden_size,
        name="ner",
    )
    return sentiment, ner


def _batch() -> dict:
    ids = np.random.RandomState(0).randint(
        0, 100, (BATCH, SEQ_LEN)
    ).astype("int32")
    return {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "token_type_ids": np.zeros_like(ids),
    }


def _multi_task_model(*, rename: bool) -> keras.Model:
    """The README §9 Pattern 2 graph, with the rename as the single variable."""
    encoder = BERT.from_variant("tiny", pretrained=False)
    encoder.trainable = True

    inputs = _inputs()
    encoder_outputs = encoder(inputs)

    if rename:
        head_inputs = {
            "hidden_states": encoder_outputs["last_hidden_state"],
            "attention_mask": encoder_outputs["attention_mask"],
        }
    else:
        head_inputs = encoder_outputs

    sentiment_head, ner_head = _heads(encoder.hidden_size)
    return keras.Model(
        inputs=inputs,
        outputs={
            "sentiment": sentiment_head(head_inputs)["logits"],
            "ner": ner_head(head_inputs)["logits"],
        },
    )


def test_the_corrected_pattern_2_graph_runs_end_to_end():
    """The README snippet, executed -- not merely read."""
    predictions = _multi_task_model(rename=True).predict(_batch(), verbose=0)

    assert set(predictions) == {"sentiment", "ner"}
    assert predictions["sentiment"].shape == (BATCH, NUM_SENTIMENT_CLASSES)
    assert predictions["ner"].shape == (BATCH, SEQ_LEN, NUM_NER_CLASSES)
    for name, value in predictions.items():
        assert np.all(np.isfinite(value)), f"{name} is not finite"


def test_building_the_graph_without_the_rename_succeeds_and_hides_the_defect():
    """The trap's MECHANISM: `.get(..., default)` defers the failure.

    If construction raised, the README snippet could never have shipped
    looking correct. This arm is what makes the deferral itself a pinned
    property rather than an explanation in a comment.
    """
    model = _multi_task_model(rename=False)
    assert isinstance(model, keras.Model)
    # MEASURED: the dict output survives construction with BOTH keys, and the
    # per-head shapes are already resolved -- `compute_output_shape()` ran on
    # the wrong dict and returned a plausible answer instead of raising.
    assert set(model.output) == {"sentiment", "ner"}
    assert model.output["sentiment"].shape == (None, NUM_SENTIMENT_CLASSES)
    assert model.output["ner"].shape == (None, None, NUM_NER_CLASSES)


def test_forward_without_the_rename_raises_keyerror_on_hidden_states():
    """The RED arm. It fires at `predict`, several frames from the mistake."""
    model = _multi_task_model(rename=False)

    with pytest.raises(KeyError, match="hidden_states"):
        model.predict(_batch(), verbose=0)
