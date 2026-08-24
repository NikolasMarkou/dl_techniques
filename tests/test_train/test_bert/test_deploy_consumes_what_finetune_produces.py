"""``train.bert.deploy`` must receive a bare logits tensor, not a dict.

``deploy.predict_sentiment`` does::

    logits = model.predict(inputs, verbose=0)
    probabilities = tf.nn.softmax(logits, axis=-1)[0]

That is only correct if the model it loaded emits a BARE tensor. Nothing in the
tree checked it; ``src/train/bert/deploy.py`` had no test coverage of any kind.

**A correction to the review that asked for this arm.** It said deploy.py
"would have raised before step 4 and is correct after" -- i.e. that step 4
accidentally repaired it. MEASURED: that is FALSE, because deploy.py never
consumes a ``create_bert_with_head`` model at all. Its producer is
``train.bert.finetune.create_sentiment_model``, which builds its own Functional
graph and indexes ``head({...})['logits']`` itself, so its output has ALWAYS
been a bare tensor. deploy.py was never broken and step 4 did not touch its
path. What step 4 DID change is that a ``create_bert_with_head`` model is now
substitutable there; that is pinned as a separate arm.

Why each arm is not satisfied by construction:

``test_finetune_produces_a_bare_logits_tensor``
    Runs the REAL ``create_sentiment_model`` (with the encoder load stubbed to
    a live ``BERT.from_variant('tiny')``), so a future refactor that returned
    the head's dict straight through is caught at the producer.
``test_deploy_predict_sentiment_consumes_it``
    Drives the REAL ``predict_sentiment`` end to end against that model. This
    is the arm that would have gone RED had the producer emitted a dict.
``test_predict_sentiment_raises_on_a_dict_output_model``
    The anti-vacuity arm. It proves the two arms above are not passing for
    free: fed a dict-output model, ``predict_sentiment`` fails. Without this,
    "it returned a label" says nothing about the container.
``test_a_factory_head_model_is_substitutable``
    The claim step 4 actually earned: ``create_bert_with_head`` now yields a
    bare tensor, so it can be dropped into this consumer unchanged.

Nothing here writes to disk and repo-root ``results/`` is never touched: the
encoder load is stubbed and no model is saved.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.models.bert.model import BERT, create_bert_with_head

import train.bert.finetune as finetune
from train.bert.deploy import predict_sentiment

SEQ_LEN = 8
NUM_CLASSES = 2


class _StubPreprocessor:
    """Stands in for ``TiktokenPreprocessor``.

    ``predict_sentiment`` only uses ``encode(text, return_tensors='tf')``, and
    only for its return value, so a fixed three-key batch is a faithful stand-in
    and keeps the arm off the tokenizer's own surface.
    """

    def encode(self, text: str, return_tensors: str = "tf"):
        ids = np.arange(1, SEQ_LEN + 1, dtype="int32")[None, :]
        return {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "token_type_ids": np.zeros_like(ids),
        }


@pytest.fixture(scope="module")
def finetune_model(module_monkeypatch):
    """The REAL ``create_sentiment_model``, with only the disk load stubbed."""
    encoder = BERT.from_variant("tiny")
    module_monkeypatch.setattr(
        finetune.keras.models, "load_model", lambda *a, **k: encoder
    )
    # NOTE: FinetuneConfig is a PLAIN class, not a dataclass -- it takes no
    # constructor arguments, so the field is set after construction.
    config = finetune.FinetuneConfig()
    config.num_classes = NUM_CLASSES
    model, _encoder = finetune.create_sentiment_model(config)
    return model


@pytest.fixture(scope="module")
def module_monkeypatch():
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    yield patcher
    patcher.undo()


def test_finetune_produces_a_bare_logits_tensor(finetune_model):
    """The producer half of the contract."""
    assert not isinstance(finetune_model.output, dict), (
        f"create_sentiment_model returned a dict {list(finetune_model.output)}; "
        "deploy.predict_sentiment softmaxes its predict() result directly and "
        "cannot consume that"
    )
    assert tuple(finetune_model.output.shape) == (None, NUM_CLASSES), (
        tuple(finetune_model.output.shape)
    )


def test_deploy_predict_sentiment_consumes_it(finetune_model):
    """The consumer half, driven through the real function."""
    label, confidence = predict_sentiment(
        "the film was surprisingly good", finetune_model, _StubPreprocessor()
    )
    assert label in ("Negative", "Positive"), label
    assert 0.0 <= confidence <= 1.0, confidence


def test_predict_sentiment_raises_on_a_dict_output_model():
    """Anti-vacuity: the two arms above are NOT passing for free."""
    inputs = {
        name: keras.Input(shape=(None,), dtype="int32", name=name)
        for name in ("input_ids", "attention_mask", "token_type_ids")
    }
    pooled = keras.layers.GlobalAveragePooling1D()(
        keras.ops.expand_dims(keras.ops.cast(inputs["input_ids"], "float32"), -1)
    )
    logits = keras.layers.Dense(NUM_CLASSES)(pooled)
    dict_model = keras.Model(inputs=inputs, outputs={"logits": logits})

    # ACTUAL measured text, not predicted:
    #   ValueError: Attempt to convert a value ({'logits': array([[...]],
    #   dtype=float32)}) with an unsupported type (<class 'dict'>) to a Tensor.
    with pytest.raises(ValueError, match="unsupported type"):
        predict_sentiment("anything at all", dict_model, _StubPreprocessor())


def test_a_factory_head_model_is_substitutable():
    """What step 4 actually earned for this consumer."""
    model = create_bert_with_head(
        bert_variant="tiny",
        task_config=NLPTaskConfig(
            name="sentiment_analysis",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=NUM_CLASSES,
        ),
    )
    assert not isinstance(model.output, dict), list(model.output)

    label, confidence = predict_sentiment(
        "the film was surprisingly good", model, _StubPreprocessor()
    )
    assert label in ("Negative", "Positive"), label
    assert 0.0 <= confidence <= 1.0, confidence
