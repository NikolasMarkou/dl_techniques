"""
`visualize_mlm_predictions` had ZERO test coverage and four raw-`tf` calls -- D-132.

The narrow raw-TF migration inherited by step 27 named nine sites in this
package. Re-derived at HEAD, `mlm.py`'s `cast`/`maximum`/`reduce_sum`/
`reduce_mean` and `memory_bank/wave_field_memory_llm.py`'s stray `tf.constant`
were ALREADY gone (a comment at `wave_field_memory_llm.py:756` records the
removal). Four were live, all in this function: `tf.cast`, `tf.shape`,
`tf.minimum`, `tf.where`.

The function is called from `src/train/common/nlp.py:636` and nothing tested it,
so the migration would have been a blind edit to live trainer surface. This file
is the guard that made it observable, not a bonus.

What it pins is the function's one real behaviour: the "filled" sequence takes
the model's PREDICTION at masked positions and the LABEL everywhere else. RED
proof: swapping the two branches of `keras.ops.where` fails
`test_filled_takes_prediction_only_at_masked_positions` (verified 2026-08-21);
a shape-only or smoke-only test would pass against that swap.
"""

import numpy as np
import keras
import pytest

from dl_techniques.models.masked_language_model import (
    MaskedLanguageModel,
    visualize_mlm_predictions,
)

VOCAB, HIDDEN, SEQ = 64, 32, 12


class StubEncoder(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = HIDDEN
        self.emb = keras.layers.Embedding(VOCAB, HIDDEN)

    def call(self, inputs, training=None):
        ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
        return {"last_hidden_state": self.emb(ids)}


class StubTokenizer:
    """Records every id sequence handed to `decode`, in call order."""

    def __init__(self):
        self.seen = []

    def decode(self, ids, **kwargs):
        self.seen.append(np.asarray(ids).tolist())
        return " ".join(str(int(i)) for i in np.asarray(ids).ravel())


@pytest.fixture(scope="module")
def model():
    keras.utils.set_random_seed(11)
    enc = StubEncoder()
    enc({"input_ids": np.zeros((1, 4), "int32")})
    return MaskedLanguageModel(encoder=enc, vocab_size=VOCAB, mask_token_id=1)


def test_runs_and_decodes_three_sequences_per_sample(model):
    tok = StubTokenizer()
    ids = np.random.RandomState(0).randint(2, VOCAB, size=(3, SEQ))
    visualize_mlm_predictions(
        mlm_model=model,
        inputs={"input_ids": ids, "attention_mask": np.ones_like(ids)},
        tokenizer=tok,
        num_samples=2,
    )
    # original, masked, filled -- per sample.
    assert len(tok.seen) == 3 * 2


def test_num_samples_is_clamped_to_the_batch(model):
    tok = StubTokenizer()
    ids = np.random.RandomState(1).randint(2, VOCAB, size=(2, SEQ))
    visualize_mlm_predictions(
        mlm_model=model,
        inputs={"input_ids": ids, "attention_mask": np.ones_like(ids)},
        tokenizer=tok,
        num_samples=99,
    )
    assert len(tok.seen) == 3 * 2


def test_filled_takes_prediction_only_at_masked_positions(model, monkeypatch):
    """The assertion with teeth: which branch of `where` goes where."""
    ids = np.random.RandomState(2).randint(2, VOCAB, size=(1, SEQ))
    masked_positions = np.zeros((1, SEQ), dtype=bool)
    masked_positions[0, [2, 5, 9]] = True
    masked_inputs = {
        "input_ids": np.where(masked_positions, 1, ids).astype("int32"),
        "attention_mask": np.ones_like(ids),
    }
    monkeypatch.setattr(
        model, "_mask_tokens", lambda _inp: (masked_inputs, ids, masked_positions)
    )

    tok = StubTokenizer()
    visualize_mlm_predictions(
        mlm_model=model, inputs=masked_inputs, tokenizer=tok, num_samples=1
    )
    original, _masked, filled = tok.seen
    assert original == ids[0].tolist()

    predicted = np.argmax(
        np.asarray(model(masked_inputs, training=False)), axis=-1
    )[0]
    expected = np.where(masked_positions[0], predicted, ids[0]).astype(int)
    assert filled == expected.tolist()
    # And the branches are actually distinguishable at those three positions,
    # or the assertion above would hold under a swapped `where` too.
    assert any(predicted[p] != ids[0][p] for p in (2, 5, 9))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
