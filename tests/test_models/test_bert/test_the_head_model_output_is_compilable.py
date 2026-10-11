"""A headed BERT must be a model that can actually be compiled.

Since plan-2026-10-11T004932-b83935e1 the head lives in the model,
``create_bert(..., include_top=True, task_config=...)``, and the output rule
below is ``layers/heads/nlp/model_head.py::finalize_head_outputs``. The history
that follows is the functional head factory's, which this replaced; the
injections named there now target ``finalize_head_outputs``.

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
from dl_techniques.models.language.bert import BERT, create_bert

BATCH_SIZE = 4
SEQ_LENGTH = 16
NUM_CLASSES = 3

# `tiny` is still 4 layers of 128; these overrides keep every arm under a
# second without changing anything the contract depends on.
SMALL = {"hidden_size": 64, "intermediate_size": 128, "num_layers": 2, "num_heads": 2}


def _inputs(seed: int = 0) -> dict:
    """All three inputs; only ``input_ids`` is required (see the last arm)."""
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
    return create_bert("tiny", include_top=True, task_config=task_config, **SMALL)


def _head_and_reference(model: BERT, inputs: dict) -> dict:
    """The head's RAW output dict, re-run outside the model on the same weights.

    The encoder is read through a headless twin built from the model's own
    config and given the model's encoder weights layer by layer; its output is
    fed to ``model.task_head`` exactly as ``BERT.call`` feeds it (mask cast to
    the hidden-state dtype).
    """
    config = model.get_config()
    config.update(include_top=False, task_config=None, head_config_overrides=None)
    config.pop("name", None)
    encoder = BERT.from_config(config)
    encoder(inputs, training=False)
    for layer in encoder.layers:
        layer.set_weights(model.get_layer(layer.name).get_weights())
    encoder_outputs = encoder(inputs, training=False)
    hidden = encoder_outputs["last_hidden_state"]
    assert isinstance(model.task_head, BaseNLPHead)
    return model.task_head(
        {
            "hidden_states": hidden,
            "attention_mask": keras.ops.cast(
                encoder_outputs["attention_mask"], hidden.dtype
            ),
        },
        training=False,
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
    and the encoder weights are pulled back out of the built model and re-run,
    which reconstructs the EXACT dict the head emits. If any head's
    ``probabilities`` were not ``softmax(logits)``, this arm falsifies the
    whole approach (plan assumption A3) instead of silently losing information.
    """
    inputs = _inputs(seed=7)

    head_layers = [
        layer for layer in classification_model.layers
        if isinstance(layer, BaseNLPHead)
    ]
    assert len(head_layers) == 1, (
        f"expected exactly one head inside the model; found {len(head_layers)}"
    )

    factory_output = classification_model(inputs, training=False)
    reference = _head_and_reference(classification_model, inputs)
    # The head itself is untouched by this fix -- only the factory's choice of
    # model outputs changed. Pin that, or the arm below could pass against a
    # head that stopped emitting probabilities entirely.
    assert set(reference) == {"logits", "probabilities"}, (
        f"the classification head no longer emits both keys: {sorted(reference)}"
    )

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
    model = create_bert("tiny", include_top=True, task_config=task_config, **SMALL)
    inputs = _inputs(seed=11)

    logits = model(inputs, training=False)
    assert not isinstance(logits, dict), (
        f"the NER model must expose the bare logits tensor, got a dict with "
        f"keys {sorted(logits)}"
    )
    assert logits.shape == (BATCH_SIZE, SEQ_LENGTH, 9), logits.shape

    reference = _head_and_reference(model, inputs)
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
    model = create_bert("tiny", include_top=True, task_config=task_config, **SMALL)

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


def test_only_input_ids_is_required_like_the_encoder() -> None:
    """(f) The head model's INPUT contract is now the encoder's own.

    This arm used to pin the opposite (F-11 / D-009 of
    plan-2026-08-23T203721-009b7ccf): the functional head factory declared one
    ``keras.Input`` per key, so a data dict without ``token_type_ids`` raised
    ``ValueError: Missing data for input "token_type_ids"`` while the bare
    encoder accepted ``input_ids`` alone. D-009 refused to relax the FACTORY's
    signature as a breaking change. That factory is gone
    (plan-2026-10-11T004932-b83935e1): the head runs inside ``BERT.call``, so a
    headed model takes exactly the inputs the encoder takes. What the arm
    protected -- that the input contract of the head model is stated and
    two-sided -- is kept:

    * the headed model ACCEPTS ``{"input_ids"}`` alone, through ``predict``;
    * and gives the same logits as the full three-key dict with the documented
      defaults spelled out (all-ones mask, all-zeros segments), so the two
      optional inputs default to what the encoder defaults them to. MEASURED
      (2026-10-11, CPU): max|delta| 1.19e-07 between two ``predict`` traces,
      hence ``atol=1e-6``.

    RED proofs, run 2026-10-11 (CPU), ACTUAL outcomes: making ``BERT.call``
    raise when ``token_type_ids`` is None -> 1 failed, ``token_type_ids is
    required``; resolving a missing mask to ZEROS before the head -> 1 failed,
    max abs difference 0.1295 in the ``assert_allclose``. The second needs the
    mean pooling below: under the head's default ``'cls'`` pooling the same
    mutation stayed GREEN.
    """
    task_config = NLPTaskConfig(
        name="sentiment",
        task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        num_classes=NUM_CLASSES,
    )
    # Mean pooling, so the head READS the mask: under the default 'cls'
    # pooling a wrong mask default is invisible (measured: the zeros-mask
    # mutation below stayed green under 'cls').
    model = create_bert(
        "tiny", include_top=True, task_config=task_config,
        head_config_overrides={"pooling_type": "mean"}, **SMALL,
    )

    full = _inputs()
    assert set(full) == {"input_ids", "attention_mask", "token_type_ids"}
    assert np.all(full["attention_mask"] == 1)
    assert np.all(full["token_type_ids"] == 0)

    ids_only = model.predict({"input_ids": full["input_ids"]}, verbose=0)
    assert ids_only.shape == (BATCH_SIZE, NUM_CLASSES), ids_only.shape

    np.testing.assert_allclose(
        ids_only,
        model.predict(full, verbose=0),
        atol=1e-6,
        rtol=0,
        err_msg="the omitted inputs do not default to all-ones / all-zeros",
    )

    # The encoder half, unchanged: `BERT` alone forwards on `input_ids` ALONE.
    encoder = BERT.from_variant("tiny", **SMALL)
    outputs = encoder({"input_ids": full["input_ids"]}, training=False)
    assert set(outputs) == {"last_hidden_state", "attention_mask"}
    assert outputs["last_hidden_state"].shape == (
        BATCH_SIZE, SEQ_LENGTH, SMALL["hidden_size"],
    )
