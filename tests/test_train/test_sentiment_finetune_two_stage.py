"""The two-stage schedule of the bert, fnet and tree_transformer fine-tuning scripts.

Stage 1 trains the head alone, stage 2 everything (plan-2026-10-11T170958-dc4d2202,
D-010: ``set_backbone_trainable`` replaced ``<encoder>.trainable = ...``, because
the encoder is no longer a layer inside the model). This drives the REAL
``finetune_sentiment_model`` of each script over a tiny encoder and a tiny
in-memory dataset (tokenizer, TFDS and the callback factory are replaced; nothing
is downloaded), and records what is trainable at the moment each stage is
compiled -- the moment that decides what ``fit`` updates.

* stage 1: the trainable variables ARE the ``task_head`` variables;
* stage 2: every variable that is trainable in a fresh model is trainable again.

All files go under ``tmp_path``; the working directory is moved there too, so a
relative ``results/`` path inside a script cannot reach the repository.
"""

from typing import List, Set

import keras
import numpy as np
import pytest
import tensorflow as tf

import train.bert.finetune as bert_finetune
import train.fnet.finetune as fnet_finetune
import train.tree_transformer.finetune as tree_finetune
from dl_techniques.models.language.bert import BERT
from dl_techniques.models.language.fnet import FNet
from dl_techniques.models.language.tree_transformer import TreeTransformer

VOCAB = 64
LENGTH = 8

SUBJECTS = {
    "bert": (bert_finetune, BERT, {"vocab_size": VOCAB, "max_position_embeddings": LENGTH}),
    "fnet": (fnet_finetune, FNet, {"vocab_size": VOCAB, "max_position_embeddings": LENGTH}),
    "tree": (tree_finetune, TreeTransformer, {"vocab_size": VOCAB}),
}


def _dataset() -> tf.data.Dataset:
    rng = np.random.RandomState(0)
    features = {
        "input_ids": rng.randint(1, VOCAB, size=(8, LENGTH)).astype("int32"),
        "attention_mask": np.ones((8, LENGTH), "int32"),
        "token_type_ids": np.zeros((8, LENGTH), "int32"),
    }
    labels = np.array([0, 1] * 4, "int32")
    return tf.data.Dataset.from_tensor_slices((features, labels)).batch(4)


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_stage_one_trains_the_head_and_stage_two_trains_everything(
    name, tmp_path, monkeypatch
):
    module, cls, geometry = SUBJECTS[name]
    monkeypatch.chdir(tmp_path)

    encoder = cls.from_variant("tiny", **geometry)
    encoder(next(iter(_dataset()))[0], training=False)
    encoder_path = str(tmp_path / "encoder.keras")
    encoder.save(encoder_path)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(module, "create_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(module, "load_text_dataset", lambda *a, **k: None)
    monkeypatch.setattr(module, "preprocess_classification_dataset", lambda *a, **k: _dataset())
    monkeypatch.setattr(module, "create_nlp_callbacks", lambda **k: ([], str(run_dir)))
    monkeypatch.setattr(module, "set_seeds", lambda *a, **k: None)

    compiled: List[Set[int]] = []
    heads: List[Set[int]] = []
    real_compile = module.compile_model

    def recording_compile(model, config, learning_rate):
        compiled.append({id(v) for v in model.trainable_weights})
        heads.append({id(v) for v in model.task_head.trainable_weights})
        return real_compile(model, config, learning_rate)

    monkeypatch.setattr(module, "compile_model", recording_compile)

    config = module.FinetuneConfig()
    config.pretrained_encoder_path = encoder_path
    config.save_dir = str(tmp_path / "save")
    config.run_two_stage_finetuning = True
    config.stage1_epochs = 1
    config.stage2_epochs = 1
    config.max_seq_length = LENGTH

    model = module.finetune_sentiment_model(config)[0]

    assert len(compiled) == 2, "expected one compile per stage"
    stage1, stage2 = compiled
    assert heads[0] and stage1 == heads[0], "stage 1 must train exactly the task_head"
    fresh = module.create_sentiment_model(config)
    assert stage2 == {id(v) for v in model.trainable_weights}
    assert len(stage2) == len(fresh.trainable_weights) > len(stage1), (
        "stage 2 left part of the model frozen"
    )
