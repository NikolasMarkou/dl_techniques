"""
A compiled metric that collides with a tracker name is dropped LOUDLY -- D-131.

`MaskedLanguageModel.metrics` appends compiled metrics to its two internal
trackers, deduped BY NAME. That dedup is correct and must stay: `train_step`
returns ``{m.name: m.result() for m in self.metrics}``, so a second metric named
"accuracy" would not add a row, it would overwrite the tracker's under the same
key. What was wrong is that the drop was SILENT, and the dropped metric is not
inert.

Measured 2026-08-21 (GPU:1 / RTX 4070, one epoch, 8 rows of 16 tokens, seed 0),
against ``create_mlm_training_model`` as it shipped:

    reported history["accuracy"] (tracker) : 0.055556
    dropped compiled "accuracy"            : 0.015625

Two different numbers, one of them invisible. So the fixes are (a) the factory
no longer compiles the colliding metric at all, and (b) a caller who compiles
one gets a warning naming it.

RED PROOF, AND A COUPLING THE FIRST ATTEMPT HID
-----------------------------------------------
Each defect was injected ALONE, and each is caught by exactly one test:

    restore ``metrics=[SparseCategoricalAccuracy(name="accuracy")]`` in the
    factory  ->  ``test_factory_compiles_no_clashing_metric`` FAILS
    delete the ``logger.warning`` block from ``MaskedLanguageModel.metrics``
             ->  ``test_clash_warns`` FAILS

Injecting BOTH at once left ``test_factory_compiles_no_clashing_metric`` GREEN,
because it asserts the ABSENCE of the clash warning and the warning machinery
was gone too. That is the "the RED-proof passes identically with and without the
defect" shape, and it is recorded rather than papered over: this test's teeth
depend on the warning in ``mlm.py`` existing. If that warning is ever removed,
this test degrades to the ``model.metrics == ["loss", "accuracy"]`` line, which
is true either way. Both mutations verified 2026-08-21.
"""

import numpy as np
import keras
import pytest

from dl_techniques.models.masked_language_model import (
    CausalLanguageModel,
    MaskedLanguageModel,
    create_mlm_training_model,
)


class StubEncoder(keras.Model):
    """Minimal encoder honouring the `hidden_size` / `last_hidden_state` contract."""

    def __init__(self, vocab: int = 64, hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden
        self.emb = keras.layers.Embedding(vocab, hidden)
        self.proj = keras.layers.Dense(hidden)

    def call(self, inputs, training=None):
        ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
        return {"last_hidden_state": self.proj(self.emb(ids))}


def _built_encoder() -> StubEncoder:
    enc = StubEncoder()
    enc({"input_ids": np.zeros((1, 4), "int32")})
    return enc


def _fit(model: MaskedLanguageModel):
    ids = np.random.RandomState(0).randint(2, 64, size=(8, 16))
    return model.fit(
        {"input_ids": ids, "attention_mask": np.ones_like(ids)},
        epochs=1,
        batch_size=4,
        verbose=0,
    )


class TestMetricNameClash:
    def test_factory_compiles_no_clashing_metric(self, caplog):
        model = create_mlm_training_model(
            encoder=_built_encoder(), vocab_size=64, mask_token_id=1
        )
        with caplog.at_level("WARNING"):
            _fit(model)
        # Asserted via the WARNING, not via `_compile_metrics`: that attribute is
        # `None` when nothing was compiled, so a `== []` check on it reads
        # AttributeError rather than a clean pass and cannot express the claim.
        assert not any("NOT be reported" in r.message for r in caplog.records), \
            [r.message for r in caplog.records]
        assert [m.name for m in model.metrics] == ["loss", "accuracy"]

    def test_clash_warns(self, caplog):
        model = MaskedLanguageModel(
            encoder=_built_encoder(), vocab_size=64, mask_token_id=1
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        with caplog.at_level("WARNING"):
            _fit(model)
        assert any("accuracy" in r.message and "NOT be reported" in r.message
                   for r in caplog.records), [r.message for r in caplog.records]

    def test_non_clashing_metric_is_reported(self):
        """The control: the dedup drops only the colliding NAME, nothing else."""
        model = MaskedLanguageModel(
            encoder=_built_encoder(), vocab_size=64, mask_token_id=1
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="top1")],
        )
        assert "top1" in _fit(model).history


def test_causal_language_model_is_exported():
    """It lives in `clm.py` and was reachable only by full module path."""
    assert CausalLanguageModel is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
