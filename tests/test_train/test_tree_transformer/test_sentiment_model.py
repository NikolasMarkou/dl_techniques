"""``train.tree_transformer.finetune.create_sentiment_model``, built for the first time.

Nothing in ``tests/`` ever called it: ``test_encoder_handoff.py`` pins only the
path and filename contract. The model it builds changed shape with
plan-2026-10-11T170958-dc4d2202 D-010 (the encoder class rebuilt with
``include_top=True`` instead of a Functional graph around a loaded encoder), so
the facts that change are pinned here:

* the parameter count: 1,042,050 for the tiny encoder below. The old Functional
  model had 1,108,098 because it kept the encoder's ``lm_head_projection`` (66,048
  weights, a headless-only output layer) next to a head that never read it;
* the layer names (``task_head`` last, no ``lm_head_projection``);
* the backbone arrays equal the saved encoder's, by value;
* the head reads position 0 (the CLS slice), with a non-vacuity precondition;
* a missing encoder file raises (no random-encoder fallback).

The encoder is built and saved here (tiny variant, small vocabulary); no
dataset is touched and everything is written under ``tmp_path``.
"""

import keras
import numpy as np
import pytest

import train.tree_transformer.finetune as finetune
from dl_techniques.models.language.tree_transformer import TreeTransformer

VOCAB = 512
LENGTH = 16


def _inputs(batch: int = 2):
    ascending = np.arange(1, LENGTH + 1, dtype="int32")
    ids = np.stack(
        [ascending if row % 2 == 0 else ascending[::-1] for row in range(batch)]
    )
    return {
        "input_ids": keras.ops.convert_to_tensor(ids),
        "attention_mask": keras.ops.ones((batch, LENGTH), dtype="int32"),
        "token_type_ids": keras.ops.zeros((batch, LENGTH), dtype="int32"),
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    encoder = TreeTransformer.from_variant("tiny", vocab_size=VOCAB)
    encoder(_inputs(1), training=False)
    path = str(tmp_path_factory.mktemp("tree_encoder") / "tree_encoder.keras")
    encoder.save(path)
    config = finetune.FinetuneConfig()
    config.pretrained_encoder_path = path
    return finetune.create_sentiment_model(config), encoder


def test_the_model_is_the_encoder_class_with_its_own_head(built):
    model, encoder = built
    assert type(model) is TreeTransformer and model.include_top is True
    assert [l.name for l in model.layers][-1] == "task_head"
    assert "lm_head_projection" not in {l.name for l in model.layers}
    assert "lm_head_projection" in {l.name for l in encoder.layers}


def test_the_parameter_count(built):
    model, encoder = built
    assert model.count_params() == 1_042_050
    assert model.task_head.count_params() == 17_282
    lm_head_projection = next(l for l in encoder.layers if l.name == "lm_head_projection")
    assert (
        model.count_params()
        == encoder.count_params() - lm_head_projection.count_params()
        + model.task_head.count_params()
    )


def test_the_output_is_a_bare_logits_tensor_of_num_classes(built):
    model, _ = built
    output = model(_inputs(), training=False)
    assert not isinstance(output, dict), list(output)
    assert tuple(output.shape) == (2, finetune.FinetuneConfig.num_classes)


def test_the_backbone_is_the_pretrained_one(built):
    model, encoder = built
    saved = {l.name: l for l in encoder.layers}
    compared = 0
    for layer in model.layers:
        if layer.name == "task_head":
            continue
        for got, want in zip(layer.weights, saved[layer.name].weights):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want), err_msg=got.path)
            compared += 1
    assert compared > 0


def test_the_head_reads_the_CLS_POSITION(built):
    model, encoder = built
    inputs = _inputs()
    hidden = encoder(inputs, training=False)["last_hidden_state"]
    head = model.task_head
    at_cls = np.asarray(head({"hidden_states": hidden[:, 0, :]}, training=False)["logits"])
    at_second = np.asarray(head({"hidden_states": hidden[:, 1, :]}, training=False)["logits"])
    assert not np.allclose(at_cls, at_second, atol=1e-6), (
        "probe is vacuous: positions 0 and 1 give the same logits"
    )
    np.testing.assert_allclose(
        np.asarray(model(inputs, training=False)), at_cls, rtol=1e-5, atol=1e-6
    )


def test_a_missing_encoder_file_raises(tmp_path):
    config = finetune.FinetuneConfig()
    config.pretrained_encoder_path = str(tmp_path / "absent.keras")
    with pytest.raises(FileNotFoundError):
        finetune.create_sentiment_model(config)
