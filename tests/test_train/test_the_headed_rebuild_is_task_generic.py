"""The fine-tuning rebuild of ``train.common.nlp`` serves any task, not only sentiment.

Guard for plan-2026-10-14T062616-81938e81 step 11.3 (the user's heads review,
``research/reviews/20261015_heads_review.md`` § 4: "trainer
``train/common/nlp.py:695`` hard-codes ``SENTIMENT_ANALYSIS`` rebuild +
``TASK_HEAD_NAME`` attr"). Measured at ``846589b7f``:

* ``build_sentiment_model_from_encoder(model_cls, path, num_classes,
  dropout_rate)`` built an ``NLPTaskConfig`` of its own, always
  ``SENTIMENT_ANALYSIS``; a token-classification or QA fine-tune on the same
  encoder had no rebuild route. Reproduced. ``build_headed_model_from_encoder``
  takes the task config; the four sentiment scripts pass
  ``sentiment_task_config(num_classes, dropout_rate)``.
* ``set_backbone_trainable`` found the head with
  ``getattr(model, TASK_HEAD_NAME)``: the ATTRIBUTE ``task_head``, which only
  happens to equal the LAYER name ``build_model_task_head`` guarantees. A model
  holding its ``'task_head'`` layer under another attribute was refused.
  Reproduced. The head is found by layer name now.

Everything is written under pytest's ``tmp_path``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.models.language.bert import BERT
from train.common.nlp import (
    build_headed_model_from_encoder,
    sentiment_task_config,
    set_backbone_trainable,
)
from tests.layer_naming import PublicAutoName

VOCAB = 97
LENGTH = 12


def _inputs(batch=2):
    ids = np.random.RandomState(0).randint(1, VOCAB, size=(batch, LENGTH)).astype("int32")
    return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


@pytest.fixture(scope="module")
def encoder_path(tmp_path_factory):
    encoder = BERT.from_variant("tiny", vocab_size=VOCAB, max_position_embeddings=LENGTH)
    encoder(_inputs(1), training=False)
    path = str(tmp_path_factory.mktemp("encoder") / "bert.keras")
    encoder.save(path)
    return path


def test_a_token_classification_head_can_be_rebuilt(encoder_path) -> None:
    ner = NLPTaskConfig(name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
                        num_classes=5)
    model = build_headed_model_from_encoder(BERT, encoder_path, ner)
    assert type(model.task_head).__name__ == "TokenClassificationHead"
    assert tuple(model(_inputs(), training=False).shape) == (2, LENGTH, 5)


def test_a_question_answering_head_can_be_rebuilt_from_a_dict(encoder_path) -> None:
    model = build_headed_model_from_encoder(
        BERT, encoder_path, {"name": "qa", "task_type": "question_answering"})
    assert sorted(model(_inputs(), training=False)) == ["end_logits", "start_logits"]


def test_the_sentiment_config_is_the_one_the_scripts_used() -> None:
    cfg = sentiment_task_config(3, dropout_rate=0.2)
    assert (cfg.name, cfg.task_type, cfg.num_classes, cfg.dropout_rate) == (
        "sentiment_analysis", NLPTaskType.SENTIMENT_ANALYSIS, 3, 0.2)


class _HeadUnderAnotherAttribute(PublicAutoName, keras.Model):
    def __init__(self):
        super().__init__()
        self.body = keras.layers.Dense(4, name="body")
        self.classifier_head = keras.layers.Dense(2, name="task_head")

    def call(self, x):
        return self.classifier_head(self.body(x))


def test_the_head_is_found_by_layer_name_not_attribute() -> None:
    model = _HeadUnderAnotherAttribute()
    model(np.zeros((1, 3), "float32"))
    set_backbone_trainable(model, False)
    assert [l.name for l in model.layers if l.trainable] == ["task_head"]
    set_backbone_trainable(model, True)
    assert all(l.trainable for l in model.layers)
