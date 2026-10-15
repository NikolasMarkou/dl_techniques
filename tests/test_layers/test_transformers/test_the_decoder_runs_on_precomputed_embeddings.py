"""`TextDecoder` runs on embeddings a caller computed, and `call` is exactly
`embed_tokens` then `decode_embeddings`.

NanoVLM (plan-2026-10-14T062616-81938e81 step 9.1, D-040) puts projected
image tokens BEFORE the text's word embeddings and runs one causal decoder
over both, the reference nanoVLM layout. `TextDecoder.call` took only token
IDs, so it was split into its two halves rather than copied into a second
transformer. The split must leave every other consumer (GPT-2) bit-identical,
which `call` == `decode_embeddings(embed_tokens(ids))` pins for every
embedding and positional type, eager and through a traced `predict`.

The same step found `positional_type='rope'` (NanoVLM's 'modern' row) building
no positional encoding at all; any value outside `'learned'`/`'sincos'` is now
refused.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.transformers.text_decoder import TextDecoder

VOCAB = 50
DIM = 16
SEQ = 9

CONFIGS = {
    "learned": dict(positional_type="learned"),
    "sincos": dict(positional_type="sincos"),
    "factorized": dict(embedding_type="factorized", normalization_type="rms_norm",
                       normalization_position="pre", ffn_type="swiglu"),
}


def _decoder(**kwargs) -> TextDecoder:
    decoder = TextDecoder(vocab_size=VOCAB, embed_dim=DIM, depth=2, num_heads=2,
                          max_seq_len=16, dropout_rate=0.0,
                          attention_dropout_rate=0.0, **kwargs)
    decoder.build((None, 16))
    return decoder


def _ids_and_mask():
    rng = np.random.default_rng(0)
    ids = rng.integers(1, VOCAB, (2, SEQ)).astype("int32")
    mask = np.ones_like(ids)
    mask[1, 6:] = 0
    return ids, mask


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_call_is_embed_tokens_then_decode_embeddings(name):
    decoder = _decoder(**CONFIGS[name])
    ids, mask = _ids_and_mask()
    whole = keras.ops.convert_to_numpy(decoder(ids, attention_mask=mask, training=False))
    halves = keras.ops.convert_to_numpy(decoder.decode_embeddings(
        decoder.embed_tokens(ids), attention_mask=mask, training=False))
    np.testing.assert_array_equal(whole, halves)

    # embed_tokens is the word table alone: no position, norm or dropout.
    embedded = keras.ops.convert_to_numpy(decoder.embed_tokens(ids))
    if name == "factorized":
        table = keras.ops.convert_to_numpy(decoder.factorized_embed_layer.embeddings)
        kernel = keras.ops.convert_to_numpy(decoder.embed_projection_layer.kernel)
        np.testing.assert_allclose(embedded, table[ids] @ kernel, atol=1e-6, rtol=0)
    else:
        table = keras.ops.convert_to_numpy(decoder.word_embeddings.embeddings)
        np.testing.assert_array_equal(embedded, table[ids])


class WholeCallProbe(keras.Model):
    """Runs the decoder through `__call__`."""

    def __init__(self, decoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder

    def call(self, inputs):
        ids, mask = inputs
        return self.decoder(ids, attention_mask=mask)


class SplitCallProbe(WholeCallProbe):
    """Runs the same decoder through its two halves, as NanoVLM does."""

    def call(self, inputs):
        ids, mask = inputs
        return self.decoder.decode_embeddings(
            self.decoder.embed_tokens(ids), attention_mask=mask)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_split_is_also_identical_through_a_traced_predict(name):
    """`predict` traces `call` into a graph; the halves must survive that too
    (they run inside NanoVLM's traced `call` under `fit`)."""
    decoder = _decoder(**CONFIGS[name])
    ids, mask = _ids_and_mask()
    whole = WholeCallProbe(decoder).predict([ids, mask], verbose=0)
    split = SplitCallProbe(decoder).predict([ids, mask], verbose=0)
    np.testing.assert_array_equal(whole, split)


def test_a_prefix_of_foreign_embeddings_is_decoded_causally():
    """Vectors put before the text are positions 0..P-1 of the same causal
    stack: text cannot move them, and they move every text position."""
    decoder = _decoder()
    ids, _ = _ids_and_mask()
    rng = np.random.default_rng(1)
    prefix = rng.normal(size=(2, 4, DIM)).astype("float32")
    other_prefix = rng.normal(size=(2, 4, DIM)).astype("float32")
    edited = ids.copy()
    edited[:, 0] = ids[:, 0] % (VOCAB - 1) + 1

    def run(p, tokens):
        joint = keras.ops.concatenate([p, decoder.embed_tokens(tokens)], axis=1)
        return keras.ops.convert_to_numpy(decoder.decode_embeddings(joint, training=False))

    base = run(prefix, ids)
    np.testing.assert_array_equal(base[:, :4], run(prefix, edited)[:, :4])
    moved = np.max(np.abs(base[:, 4:] - run(other_prefix, ids)[:, 4:]), axis=-1)
    assert np.all(moved > 0.0), moved


@pytest.mark.parametrize("positional_type", ["rope", "none", "shared"])
def test_an_unknown_positional_type_is_refused(positional_type):
    with pytest.raises(ValueError, match="positional_type must be one of"):
        TextDecoder(vocab_size=VOCAB, embed_dim=DIM, depth=1, num_heads=2,
                    positional_type=positional_type)
