"""Every trainable weight of both Qwen3 embedding/reranker trainers receives
a nonzero, finite gradient from its REAL compiled loss.

Reuses `tests/test_models/gradient_flow_oracle.py::assert_gradients_reach_
every_trainable_weight` (found by grep before writing a second copy -- see
`src/train/CLAUDE.md` DRY discipline and
`plans/plan-2026-09-13T073704-245ab5d5/decisions.md`), matching the existing
`tests/test_train/test_omnipoint/test_train_omnipoint_cli.py` precedent for
importing this oracle from `tests/test_train/`.

Both models are tiny (a few hidden units, one layer) purely for speed --
this test measures WHETHER gradients flow, not learning quality.
"""

from __future__ import annotations

import keras
import numpy as np

from dl_techniques.losses.infonce_loss import SymmetricInfoNCELoss
from tests.test_models.gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
)
from train.qwen3_embeddings.common import (
    TrainingConfig,
    build_embedding_datasets,
    build_embedding_model,
    build_optimizer,
    build_reranker_datasets,
    build_reranker_model,
    embedding_steps_per_epoch,
    reranker_steps_per_epoch,
)

_TINY = dict(
    hidden_size=16, num_layers=1, num_heads=2, intermediate_size=32,
    query_maxlen=8, doc_maxlen=16, reranker_maxlen=24,
    batch_size=4, num_train_groups=16, num_val_groups=8, nway=4,
)


def test_embedding_tower_gradients_reach_every_trainable_weight() -> None:
    config = TrainingConfig(**_TINY)
    train_ds, _val_ds, query_tok, _doc_tok = build_embedding_datasets(config)
    vocab_size = query_tok.vocab_size

    optimizer = build_optimizer(config, embedding_steps_per_epoch(config))
    model = build_embedding_model(config, vocab_size=vocab_size, optimizer=optimizer)

    inputs, _labels = next(iter(train_ds))
    model(inputs, training=False)  # force build -- a subclassed Model is unbuilt until first call

    def loss_fn(outputs):
        batch = keras.ops.shape(outputs)[0]
        return SymmetricInfoNCELoss(temperature=config.infonce_temperature)(
            keras.ops.zeros((batch,)), outputs
        )

    report = assert_gradients_reach_every_trainable_weight(model, inputs, loss_fn=loss_fn)
    assert len(report) > 0


def test_reranker_gradients_reach_every_trainable_weight() -> None:
    config = TrainingConfig(**_TINY)
    train_ds, _val_ds, tokenizer, yes_id, no_id = build_reranker_datasets(config)

    optimizer = build_optimizer(config, reranker_steps_per_epoch(config))
    model = build_reranker_model(
        config, vocab_size=tokenizer.vocab_size, yes_token_id=yes_id,
        no_token_id=no_id, optimizer=optimizer,
    )

    inputs, labels = next(iter(train_ds))
    model(inputs, training=False)  # force build -- a subclassed Model is unbuilt until first call
    labels_f32 = keras.ops.cast(labels, "float32")

    def loss_fn(outputs):
        return keras.ops.mean(keras.losses.binary_crossentropy(labels_f32, outputs))

    report = assert_gradients_reach_every_trainable_weight(model, inputs, loss_fn=loss_fn)
    assert len(report) > 0


def test_embedding_tower_gradient_report_has_no_none_or_nan_entries() -> None:
    """The oracle's own report, inspected directly (not just its assertion),
    so a regression that widens `expect_zero` silently cannot hide a
    disconnected weight."""
    config = TrainingConfig(**_TINY)
    train_ds, _val_ds, query_tok, _doc_tok = build_embedding_datasets(config)

    optimizer = build_optimizer(config, embedding_steps_per_epoch(config))
    model = build_embedding_model(config, vocab_size=query_tok.vocab_size, optimizer=optimizer)

    inputs, _labels = next(iter(train_ds))
    model(inputs, training=False)  # force build -- a subclassed Model is unbuilt until first call

    def loss_fn(outputs):
        batch = keras.ops.shape(outputs)[0]
        return SymmetricInfoNCELoss(temperature=config.infonce_temperature)(
            keras.ops.zeros((batch,)), outputs
        )

    from tests.test_models.gradient_flow_oracle import gradient_report

    report = gradient_report(model, inputs, loss_fn=loss_fn)
    assert report, "gradient_report returned an empty report"
    for path, value in report.items():
        assert value is not None, f"{path}: gradient is None (disconnected)"
        assert not np.isnan(value), f"{path}: gradient is NaN"
        assert value > 0.0, f"{path}: gradient is exactly zero (dead weight)"
