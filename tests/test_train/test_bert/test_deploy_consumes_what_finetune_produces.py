"""``train.bert.deploy`` must receive a bare logits tensor, not a dict.

``deploy.predict_sentiment`` does::

    logits = model.predict(inputs, verbose=0)
    probabilities = tf.nn.softmax(logits, axis=-1)[0]

That is only correct if the model it loaded emits a BARE tensor. Nothing in the
tree checked it; ``src/train/bert/deploy.py`` had no test coverage of any kind.

**What the producer is now.** ``train.bert.finetune.create_sentiment_model``
rebuilds the BERT class itself with ``include_top=True`` (plan-2026-10-11T170958-dc4d2202
D-010), so the artifact ``finetune`` saves reloads as a ``BERT`` whose head
layer is ``task_head``. It used to be a Functional graph with a head layer named
``sentiment_analysis_head``. ``deploy.py`` never looks at layer names or the
model class, so BOTH artifacts must keep working: an old Functional artifact is
a user's checkpoint on disk, and ``test_an_old_functional_artifact_still_works``
pins it. The earlier correction stands: deploy.py was never broken by the
head-factory removal, it only became able to take a head-factory model.

Why each arm is not satisfied by construction:

``test_finetune_produces_a_bare_logits_tensor``
    Runs the REAL ``create_sentiment_model`` over a tiny encoder archive saved
    by this module, so a future refactor that returned the head's dict straight
    through is caught at the producer.
``test_deploy_predict_sentiment_consumes_it``
    Drives the REAL ``predict_sentiment`` end to end against that model. This
    is the arm that would have gone RED had the producer emitted a dict.
``test_deploy_consumes_the_saved_artifact``
    The artifact as ``deploy.load_model_and_preprocessor`` meets it: saved by
    ``model.save`` and reloaded by ``keras.models.load_model``.
``test_an_old_functional_artifact_still_works``
    The previous artifact shape (Functional, head layer ``sentiment_analysis_head``)
    still flows through the unchanged consumer.
``test_predict_sentiment_raises_on_a_dict_output_model``
    The anti-vacuity arm. It proves the two arms above are not passing for
    free: fed a dict-output model, ``predict_sentiment`` fails. Without this,
    "it returned a label" says nothing about the container.
``test_a_factory_head_model_is_substitutable``
    The claim step 4 actually earned, kept on the new API: a headed BERT
    yields a bare tensor, so it can be dropped into this consumer unchanged.

Everything this module writes goes under pytest's ``tmp_path``; repo-root
``results/`` is never touched.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType, create_nlp_head
from dl_techniques.models.language.bert.model import BERT, create_bert

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
def encoder_path(tmp_path_factory):
    """A tiny headless BERT archive, the thing ``pretrain`` hands to ``finetune``."""
    encoder = BERT.from_variant("tiny", vocab_size=64, max_position_embeddings=SEQ_LEN)
    encoder(_StubPreprocessor().encode("probe"), training=False)
    path = str(tmp_path_factory.mktemp("encoder") / "encoder.keras")
    encoder.save(path)
    return path


@pytest.fixture(scope="module")
def finetune_model(encoder_path):
    """The REAL ``create_sentiment_model`` over a real archive; nothing stubbed."""
    # NOTE: FinetuneConfig is a PLAIN class, not a dataclass -- it takes no
    # constructor arguments, so the field is set after construction.
    config = finetune.FinetuneConfig()
    config.pretrained_encoder_path = encoder_path
    config.num_classes = NUM_CLASSES
    return finetune.create_sentiment_model(config)


def test_finetune_produces_a_bare_logits_tensor(finetune_model):
    """The producer half of the contract."""
    output = finetune_model(_StubPreprocessor().encode("probe"), training=False)
    assert not isinstance(output, dict), (
        f"create_sentiment_model returned a dict {list(output)}; "
        "deploy.predict_sentiment softmaxes its predict() result directly and "
        "cannot consume that"
    )
    assert tuple(output.shape) == (1, NUM_CLASSES), tuple(output.shape)


def test_deploy_predict_sentiment_consumes_it(finetune_model):
    """The consumer half, driven through the real function."""
    label, confidence = predict_sentiment(
        "the film was surprisingly good", finetune_model, _StubPreprocessor()
    )
    assert label in ("Negative", "Positive"), label
    assert 0.0 <= confidence <= 1.0, confidence


def test_deploy_consumes_the_saved_artifact(finetune_model, tmp_path):
    """Save and reload exactly as ``finetune`` and ``deploy`` do."""
    path = str(tmp_path / "bert_sentiment_final_best.keras")
    finetune_model.save(path)
    reloaded = keras.models.load_model(path)
    assert type(reloaded) is BERT and reloaded.include_top is True
    label, confidence = predict_sentiment(
        "the film was surprisingly good", reloaded, _StubPreprocessor()
    )
    assert label in ("Negative", "Positive"), label
    assert 0.0 <= confidence <= 1.0, confidence


def test_an_old_functional_artifact_still_works(encoder_path, tmp_path):
    """The artifact shape ``finetune`` wrote before ``include_top``.

    A Functional graph over three named Inputs, the encoder as a layer and a
    ``sentiment_analysis_head``: still on disk in users' ``results/`` and still
    read by the unchanged consumer, through a save and a reload.
    """
    encoder = keras.models.load_model(encoder_path)
    head = create_nlp_head(
        NLPTaskConfig(
            name="sentiment_analysis",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=NUM_CLASSES,
        ),
        input_dim=encoder.hidden_size,
    )
    inputs = {
        name: keras.Input(shape=(None,), dtype="int32", name=name)
        for name in ("input_ids", "attention_mask", "token_type_ids")
    }
    hidden = encoder(inputs)["last_hidden_state"][:, 0, :]
    old = keras.Model(inputs, head({"hidden_states": hidden})["logits"])
    path = str(tmp_path / "old_final.keras")
    old.save(path)

    reloaded = keras.models.load_model(path)
    assert isinstance(reloaded, keras.Model) and not isinstance(reloaded, BERT)
    assert any(l.name.startswith("sentiment_analysis_head") for l in reloaded.layers)
    label, confidence = predict_sentiment(
        "the film was surprisingly good", reloaded, _StubPreprocessor()
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
    model = create_bert(
        "tiny",
        include_top=True,
        task_config=NLPTaskConfig(
            name="sentiment_analysis",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=NUM_CLASSES,
        ),
    )
    output = model(_StubPreprocessor().encode("probe"), training=False)
    assert not isinstance(output, dict), list(output)
    assert tuple(output.shape) == (1, NUM_CLASSES), tuple(output.shape)

    label, confidence = predict_sentiment(
        "the film was surprisingly good", model, _StubPreprocessor()
    )
    assert label in ("Negative", "Positive"), label
    assert 0.0 <= confidence <= 1.0, confidence
