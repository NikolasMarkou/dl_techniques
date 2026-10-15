"""The model ``src/train/bert/wikipedia/finetune.py`` fine-tunes, and its failure mode.

Until plan-2026-10-11T170958-dc4d2202 (D-010) ``main()`` wrapped the encoder load in
``except Exception`` and, on ANY failure (a wrong path, a corrupt archive, an
incompatible checkpoint), logged a warning and trained a RANDOM ``BERT.from_variant("base")``
instead: a "pretrained-encoder fine-tune" that silently starts from scratch and reports
a number that means nothing. The model is now built by ``create_sst2_model`` through
``train.common.nlp.build_headed_model_from_encoder`` (``include_top=True``), and a
failed load raises.

``test_a_missing_encoder_raises_out_of_main``
    Drives the REAL ``main()`` up to the model with the data stubbed. If a fallback
    came back inside ``main`` or ``create_sst2_model`` the error would be a different
    one (or none), so the exception TYPE is the assertion, and ``BERT.from_variant``
    is replaced by a sentinel that must stay uncalled.
``test_the_model_is_a_headed_bert_with_the_pretrained_backbone``
    Builds the model from a real tiny archive and compares the arrays by value.

No network, no dataset, nothing under the repository: the archive is written to ``tmp_path``.
"""

from typing import Any

import keras
import numpy as np
import pytest
import tensorflow_datasets as tfds

import train.common
import train.bert.wikipedia.finetune as finetune
from dl_techniques.models.language.bert import BERT

VOCAB = 64
LENGTH = 8


class _FromVariantMustNotBeCalled:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError("BERT.from_variant was called: a random encoder fallback is back")


def _tiny_archive(tmp_path) -> tuple:
    encoder = BERT.from_variant("tiny", vocab_size=VOCAB, max_position_embeddings=LENGTH)
    encoder(
        {
            "input_ids": np.ones((1, LENGTH), "int32"),
            "attention_mask": np.ones((1, LENGTH), "int32"),
            "token_type_ids": np.zeros((1, LENGTH), "int32"),
        },
        training=False,
    )
    path = str(tmp_path / "encoder.keras")
    encoder.save(path)
    return encoder, path


def test_the_model_is_a_headed_bert_with_the_pretrained_backbone(tmp_path):
    encoder, path = _tiny_archive(tmp_path)
    config = finetune.FinetuneConfig()
    config.pretrained_encoder_path = path

    model = finetune.create_sst2_model(config)

    assert type(model) is BERT and model.include_top is True
    assert model.task_head.name == "task_head"
    assert model.task_head.count_params() == 67_330
    saved = {l.name: l for l in encoder.layers}
    compared = 0
    for layer in model.layers:
        if layer is model.task_head:
            continue
        for got, want in zip(layer.weights, saved[layer.name].weights):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want), err_msg=got.path)
            compared += 1
    assert compared == len(encoder.weights)


def test_a_missing_encoder_raises_out_of_main(tmp_path, monkeypatch):
    sentinel = _FromVariantMustNotBeCalled()
    monkeypatch.setattr(BERT, "from_variant", sentinel)
    monkeypatch.setattr(train.common, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(tfds, "load", lambda *a, **k: ({"train": None, "validation": None}, None))
    monkeypatch.setattr(finetune, "create_tokenizer", lambda config: None)
    monkeypatch.setattr(finetune, "preprocess_glue", lambda *a, **k: None)
    monkeypatch.setattr(finetune.FinetuneConfig, "save_dir", str(tmp_path / "run"))
    monkeypatch.setattr(
        finetune.FinetuneConfig, "pretrained_encoder_path", str(tmp_path / "absent.keras")
    )

    with pytest.raises(FileNotFoundError):
        finetune.main([])

    assert sentinel.calls == 0


def test_a_corrupt_encoder_archive_raises_too(tmp_path):
    path = tmp_path / "corrupt.keras"
    path.write_bytes(b"this is not a zip archive")
    config = finetune.FinetuneConfig()
    config.pretrained_encoder_path = str(path)
    with pytest.raises(ValueError, match="not a readable"):
        finetune.create_sst2_model(config)
