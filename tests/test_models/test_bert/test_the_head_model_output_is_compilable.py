"""``create_bert_with_head`` must return a model that can actually be compiled.

F-03/F-04. Before this guard, the factory handed ``keras.Model`` the task
head's raw output dict ``{'logits', 'probabilities'}``, where ``probabilities``
is *exactly* ``softmax(logits)``. A dict output with a derived duplicate in it
made the repo's flagship NLP factory impossible to compile the documented way.
MEASURED on this repo before the fix:

- ``model.predict(x).shape`` -> ``AttributeError: 'dict' object has no
  attribute 'shape'``
- ``loss='sparse_categorical_crossentropy'`` -> ``KeyError: The path:
  ('logits',) ...``
- ``metrics=['accuracy']``, ``metrics={'logits': ...}`` and the dict+list forms
  ALL raise.

Only ``loss={'logits': ...}`` with NO metrics compiled, i.e. there was no way
to attach a metric to this model at all.

The fix drops ``probabilities`` when ``logits`` is present and, if exactly one
key survives, uses that bare tensor as the model output. Nothing is lost --
arm (c) MEASURES that the dropped key was a pure derivative. A head emitting
genuinely independent tensors keeps its dict, which arm (d) pins.

RED PROOFS -- ACTUAL observed text, not predicted.

Injection 1, the collapse reverted (``model_outputs = task_outputs`` verbatim,
i.e. the pre-fix code) -> **3 failed, 1 passed**:

- ``test_the_head_model_predicts_a_bare_tensor``::
  ``AssertionError: predict() returned a dict; the documented contract is a
  bare tensor whose .shape reads.`` -- NOTE the predicted text for this arm was
  ``AttributeError: 'dict' object has no attribute 'shape'`` and it is WRONG:
  this test asserts the type BEFORE reading ``.shape``, deliberately, so the
  arm convicts by its own name instead of by a framework crash.
- ``test_the_head_model_compiles_with_a_string_loss_and_a_metric``::
  ``KeyError: "The path: ('logits',) in the `loss` argument, can't be found in
  either the model's output (`y_pred`) or in the labels (`y_true`)."`` from
  ``keras/src/trainers/compile_utils.py:553``. It fires at ``fit``, not at
  ``compile`` -- Keras defers loss-structure resolution to the first batch.
- ``test_the_dropped_probabilities_key_was_a_pure_derivative``::
  ``AssertionError: the factory output must be the bare logits tensor, got a
  dict with keys ['logits', 'probabilities']``
- arm (d) still PASSED, as it must: reverting the collapse cannot break the
  dict-preserving case.

Injection 2, the collapse made unconditional (``model_outputs =
next(iter(task_outputs.values()))`` with no key filter and no length guard)
-> **1 failed, 3 passed**:

- ``test_a_genuinely_multi_output_head_keeps_its_dict``::
  ``AssertionError: QuestionAnsweringHead emits two independent tensors and
  must keep a dict output; got <class 'tensorflow.python.framework.ops.EagerTensor'>``

Injection 3, the derived-key set narrowed to ``("probabilities",)`` -- i.e.
the ONE key the plan predicted, before the AST walk found the second
-> **1 failed, 4 passed**:

- ``test_the_token_head_predictions_key_was_also_a_pure_derivative``::
  ``AssertionError: the NER model must expose the bare logits tensor, got a
  dict with keys ['logits', 'predictions']``
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.heads.nlp.factory import BaseNLPHead
from dl_techniques.models.bert.bert import BERT, create_bert_with_head

BATCH_SIZE = 4
SEQ_LENGTH = 16
NUM_CLASSES = 3

# `tiny` is still 4 layers of 128; these overrides keep every arm under a
# second without changing anything the contract depends on.
SMALL = {"hidden_size": 64, "intermediate_size": 128, "num_layers": 2, "num_heads": 2}


def _inputs(seed: int = 0) -> dict:
    """The three inputs the factory's Functional wrapper requires."""
    rng = np.random.default_rng(seed)
    return {
        "input_ids": rng.integers(0, 1000, size=(BATCH_SIZE, SEQ_LENGTH)).astype("int32"),
        "attention_mask": np.ones((BATCH_SIZE, SEQ_LENGTH), dtype="int32"),
        "token_type_ids": np.zeros((BATCH_SIZE, SEQ_LENGTH), dtype="int32"),
    }


@pytest.fixture
def classification_model() -> keras.Model:
    task_config = NLPTaskConfig(
        name="sentiment",
        task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        num_classes=NUM_CLASSES,
    )
    return create_bert_with_head(
        "tiny", task_config, bert_config_overrides=SMALL
    )


def test_the_head_model_predicts_a_bare_tensor(classification_model) -> None:
    """(a) ``model.predict(x).shape`` must read -- it raised on a dict."""
    predictions = classification_model.predict(_inputs(), verbose=0)

    assert not isinstance(predictions, dict), (
        f"predict() returned a {type(predictions).__name__}; the documented "
        f"contract is a bare tensor whose .shape reads."
    )
    assert predictions.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"expected {(BATCH_SIZE, NUM_CLASSES)}, got {predictions.shape}"
    )
    assert np.all(np.isfinite(predictions))


def test_the_head_model_compiles_with_a_string_loss_and_a_metric(
    classification_model,
) -> None:
    """(b) the documented compile + one fit step, and an accuracy key lands.

    ``metrics=['accuracy']`` is the assertion that could not be satisfied AT
    ALL before the fix -- every metrics spelling raised -- so this arm is the
    one that closes F-03 rather than merely restating F-04.
    """
    classification_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    labels = np.array([0, 1, 2, 0], dtype="int32")
    history = classification_model.fit(
        _inputs(), labels, epochs=1, batch_size=BATCH_SIZE, verbose=0
    )

    accuracy_keys = [k for k in history.history if "accuracy" in k]
    assert accuracy_keys, (
        f"one fit step produced no accuracy metric; history carries "
        f"{sorted(history.history)}"
    )
    assert np.isfinite(history.history["loss"][0])


def test_the_dropped_probabilities_key_was_a_pure_derivative(
    classification_model,
) -> None:
    """(c) ``softmax(model(x))`` reproduces the head's own ``probabilities``.

    The reference is not a recollection of the pre-change value: the head layer
    and the encoder are pulled back out of the built model and re-run, which
    reconstructs the EXACT dict the factory used to expose. If any head's
    ``probabilities`` were not ``softmax(logits)``, this arm falsifies the
    whole approach (plan assumption A3) instead of silently losing information.
    """
    inputs = _inputs(seed=7)

    head_layers = [
        layer for layer in classification_model.layers
        if isinstance(layer, BaseNLPHead)
    ]
    encoders = [
        layer for layer in classification_model.layers if isinstance(layer, BERT)
    ]
    assert len(head_layers) == 1 and len(encoders) == 1, (
        f"expected exactly one head and one encoder inside the factory model; "
        f"found {len(head_layers)} head(s) and {len(encoders)} encoder(s)"
    )

    encoder_outputs = encoders[0](inputs, training=False)
    reference = head_layers[0](
        {
            "hidden_states": encoder_outputs["last_hidden_state"],
            "attention_mask": encoder_outputs["attention_mask"],
        },
        training=False,
    )
    # The head itself is untouched by this fix -- only the factory's choice of
    # model outputs changed. Pin that, or the arm below could pass against a
    # head that stopped emitting probabilities entirely.
    assert set(reference) == {"logits", "probabilities"}, (
        f"the classification head no longer emits both keys: {sorted(reference)}"
    )

    factory_output = classification_model(inputs, training=False)
    assert not isinstance(factory_output, dict), (
        f"the factory output must be the bare logits tensor, got a dict with "
        f"keys {sorted(factory_output)}"
    )

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(reference["logits"]),
        keras.ops.convert_to_numpy(factory_output),
        atol=1e-6,
        rtol=0,
        err_msg="the surviving output is not the head's logits tensor",
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(
            keras.ops.softmax(factory_output, axis=-1)
        ),
        keras.ops.convert_to_numpy(reference["probabilities"]),
        atol=1e-6,
        rtol=0,
        err_msg=(
            "softmax(logits) does not reproduce the head's own 'probabilities' "
            "-- dropping that key LOSES information and this change is unsound"
        ),
    )


def test_the_token_head_predictions_key_was_also_a_pure_derivative() -> None:
    """(e) ``TokenClassificationHead`` puts ``argmax(logits)`` beside ``logits``.

    This arm exists because the plan predicted ONE derived key and the tree has
    TWO. ``TokenClassificationHead.call`` (``factory.py:615-621``) emits
    ``{'logits', 'predictions'}`` where ``predictions = ops.argmax(logits,
    axis=-1)`` -- an int32 tensor no loss can consume -- so the NER model was a
    dict for the same non-reason. The derived set was RE-DERIVED by an AST walk
    over every ``call()`` in that module, not assumed: exactly
    ``probabilities`` (``TextClassificationHead``, ``MultipleChoiceHead``) and
    ``predictions`` (``TokenClassificationHead``) are pure functions of
    ``logits``; ``QuestionAnsweringHead`` and ``TextSimilarityHead`` emit
    genuinely independent tensors and are untouched.
    """
    task_config = NLPTaskConfig(
        name="ner",
        task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
        num_classes=9,
    )
    model = create_bert_with_head(
        "tiny", task_config, bert_config_overrides=SMALL
    )
    inputs = _inputs(seed=11)

    logits = model(inputs, training=False)
    assert not isinstance(logits, dict), (
        f"the NER model must expose the bare logits tensor, got a dict with "
        f"keys {sorted(logits)}"
    )
    assert logits.shape == (BATCH_SIZE, SEQ_LENGTH, 9), logits.shape

    head_layer = [
        layer for layer in model.layers if isinstance(layer, BaseNLPHead)
    ][0]
    encoder = [layer for layer in model.layers if isinstance(layer, BERT)][0]
    encoder_outputs = encoder(inputs, training=False)
    reference = head_layer(
        {
            "hidden_states": encoder_outputs["last_hidden_state"],
            "attention_mask": encoder_outputs["attention_mask"],
        },
        training=False,
    )
    assert set(reference) == {"logits", "predictions"}, sorted(reference)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(
            keras.ops.argmax(logits, axis=-1)
        ).astype("int64"),
        keras.ops.convert_to_numpy(reference["predictions"]).astype("int64"),
        err_msg=(
            "argmax(logits) does not reproduce the head's own 'predictions' "
            "-- dropping that key LOSES information"
        ),
    )

    # The whole point: with the dict gone, the token-level objective compiles.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    labels = np.zeros((BATCH_SIZE, SEQ_LENGTH), dtype="int32")
    history = model.fit(
        inputs, labels, epochs=1, batch_size=BATCH_SIZE, verbose=0
    )
    assert [k for k in history.history if "accuracy" in k], sorted(history.history)


def test_a_genuinely_multi_output_head_keeps_its_dict() -> None:
    """(d) ``QuestionAnsweringHead`` emits two independent tensors -> dict."""
    task_config = NLPTaskConfig(
        name="qa",
        task_type=NLPTaskType.QUESTION_ANSWERING,
        num_classes=2,
    )
    model = create_bert_with_head(
        "tiny", task_config, bert_config_overrides=SMALL
    )

    outputs = model(_inputs(), training=False)

    assert isinstance(outputs, dict), (
        f"QuestionAnsweringHead emits two independent tensors and must keep a "
        f"dict output; got {type(outputs)}"
    )
    assert set(outputs) == {"start_logits", "end_logits"}, sorted(outputs)
    for key in ("start_logits", "end_logits"):
        assert outputs[key].shape == (BATCH_SIZE, SEQ_LENGTH), (
            f"{key}: {outputs[key].shape}"
        )


def test_all_three_inputs_are_required_but_the_encoder_needs_only_one() -> None:
    """(e) F-11 / D-009: the factory's INPUT contract, and why it is stricter.

    Not a vacuous pin. It is two-sided: the factory must REJECT the two-key
    dict, and the bare encoder must ACCEPT the one-key dict. A change that
    relaxed the factory reds the first half; a change that tightened `BERT`
    reds the second. A one-sided "it raises" assertion would be satisfied by
    any breakage at all, which is why the exact missing-key name is matched.

    MEASURED (this is the ACTUAL text, not a prediction)::

        ValueError: Missing data for input "token_type_ids". You passed a data
        dictionary with keys ['input_ids', 'attention_mask']. Expected the
        following keys: ['attention_mask', 'input_ids', 'token_type_ids']

    RED proofs, one per half, ACTUAL text:

    * **Delete the `token_type_ids` `keras.Input`** from the factory's
      Functional wrapper -- literally the relaxation D-009 refuses. This arm
      alone: ``Failed: DID NOT RAISE <class 'ValueError'>``. (Run over the
      whole file it is 6 failed / 0 passed: every other arm feeds all three
      keys, so the relaxation breaks them too. That is why the isolating
      figure above is quoted from the single-node run.)
    * **Make `BERT.call` require `token_type_ids`** (``raise ValueError`` when
      it is None): 1 failed, 5 passed -- ONLY this arm, by its second half::

          ValueError: Exception encountered when calling BERT.call().
          token_type_ids is required
    """
    task_config = NLPTaskConfig(
        name="sentiment",
        task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        num_classes=NUM_CLASSES,
    )
    model = create_bert_with_head(
        "tiny", task_config, bert_config_overrides=SMALL
    )

    full = _inputs()
    assert set(full) == {"input_ids", "attention_mask", "token_type_ids"}

    without_segments = {
        key: value for key, value in full.items() if key != "token_type_ids"
    }
    with pytest.raises(ValueError, match=r"token_type_ids"):
        model.predict(without_segments, verbose=0)

    # The paired half. `BERT` alone forwards on `input_ids` ALONE -- so the
    # factory's strictness is the wrapper's, not the model's, and D-009's
    # "document, do not relax" only makes sense if this half holds.
    encoder = BERT.from_variant("tiny", **SMALL)
    outputs = encoder({"input_ids": full["input_ids"]}, training=False)
    assert set(outputs) == {"last_hidden_state", "attention_mask"}
    assert outputs["last_hidden_state"].shape == (
        BATCH_SIZE, SEQ_LENGTH, SMALL["hidden_size"],
    )
