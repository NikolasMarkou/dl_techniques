"""Tests for the BPE tokenizer layers.

Covers ``BPETokenizer`` (string -> token-ID preprocessing layer) and
``TokenEmbedding`` (token-ID -> dense vectors), plus the ``train_bpe`` helper.
Includes construction (incl. ``ValueError`` paths), tokenization behaviour, a
forward pass, ``compute_output_shape`` agreement, and full ``.keras``
serialization round-trips.
"""

import os
import json
import keras
import numpy as np
import pytest

from dl_techniques.layers.tokenizers.bpe import (
    train_bpe,
    create_bpe_pipeline,
    BPETokenizer,
    TokenEmbedding,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def trained_vocab():
    """Train a tiny BPE vocab on a small corpus."""
    texts = [
        "the quick brown fox",
        "the lazy dog sleeps",
        "quick brown foxes jump",
        "the dog and the fox",
    ]
    vocab_dict, merges = train_bpe(texts, vocab_size=80, min_frequency=1)
    return vocab_dict, merges


# ---------------------------------------------------------------------
# train_bpe
# ---------------------------------------------------------------------

def test_train_bpe_returns_vocab_and_merges():
    vocab_dict, merges = train_bpe(["hello world", "hello there"],
                                   vocab_size=60, min_frequency=1)
    assert isinstance(vocab_dict, dict)
    assert isinstance(merges, list)
    assert len(vocab_dict) > 0


# ---------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------

class TestBPETokenizer:

    def test_construction(self, trained_vocab):
        vocab_dict, merges = trained_vocab
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges, max_length=16)
        assert tok.max_length == 16
        # Special tokens always present.
        for special in (tok.pad_token, tok.unk_token, tok.eos_token):
            assert special in tok.vocab_dict

    def test_construction_defaults_empty(self):
        tok = BPETokenizer()
        # The three special tokens still get registered.
        assert tok.vocab_size == 3

    def test_invalid_max_length(self):
        with pytest.raises(ValueError):
            BPETokenizer(max_length=0)
        with pytest.raises(ValueError):
            BPETokenizer(max_length=-5)

    def test_tokenize_texts_shape(self, trained_vocab):
        vocab_dict, merges = trained_vocab
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges, max_length=16)
        ids = tok.tokenize_texts(["the quick fox", "lazy dog"])
        assert len(ids) == 2
        assert all(len(seq) == 16 for seq in ids)

    def test_call_raises_not_implemented(self, trained_vocab):
        vocab_dict, merges = trained_vocab
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges)
        with pytest.raises(NotImplementedError):
            tok(keras.ops.zeros((1, 4)))

    def test_decode_round_trip(self, trained_vocab):
        vocab_dict, merges = trained_vocab
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges, max_length=32)
        ids = tok.tokenize_texts(["the quick brown fox"])[0]
        decoded = tok.decode_tokens(ids)
        assert "quick" in decoded

    def test_compute_output_shape(self, trained_vocab):
        vocab_dict, merges = trained_vocab
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges, max_length=16)
        assert tok.compute_output_shape((None, 1)) == (None, 16)

    def test_get_config_round_trip(self, trained_vocab):
        vocab_dict, merges = trained_vocab
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges, max_length=16)
        config = tok.get_config()
        rebuilt = BPETokenizer.from_config(config)
        assert rebuilt.max_length == 16
        # Tokenization is deterministic across the config round-trip.
        assert (tok.tokenize_texts(["the quick fox"])
                == rebuilt.tokenize_texts(["the quick fox"]))


class TestBPETokenizerContracts:
    """Behaviour the tokenizer, the trainer and the pipeline must agree on."""

    CORPUS = ["hello, world", "hello, world", "hello there, world"]

    def _trained(self, **kwargs):
        vocab_dict, merges = train_bpe(
            self.CORPUS, vocab_size=80, min_frequency=1, **kwargs
        )
        return vocab_dict, merges

    def test_json_round_trip_keeps_merges_usable(self, trained_vocab):
        vocab_dict, merges = trained_vocab
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges, max_length=16)
        config = json.loads(json.dumps(tok.get_config()))
        rebuilt = BPETokenizer.from_config(config)
        assert all(isinstance(pair, tuple) for pair in rebuilt.merges)
        assert (tok.tokenize_texts(["the quick fox"])
                == rebuilt.tokenize_texts(["the quick fox"]))

    def test_pipeline_embedding_covers_every_id_the_tokenizer_emits(self):
        tok, emb = create_bpe_pipeline(
            self.CORPUS, vocab_size=80, embedding_dim=8, max_length=12,
            min_frequency=1,
        )
        ids = np.array(tok.tokenize_texts(["hello, world", "hello"]))
        assert emb.vocab_size == tok.vocab_size
        assert ids.max() < emb.vocab_size
        out = keras.ops.convert_to_numpy(emb(ids))
        assert out.shape == (2, 12, 8)

    def test_encoding_splits_punctuation_the_way_training_did(self):
        vocab_dict, merges = self._trained()
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges, max_length=16)
        ids = tok.tokenize_texts(["hello, world"])[0]
        # "hello" is trained as a word of its own, so "hello," must not
        # absorb the comma into the word.
        assert vocab_dict["hello</w>"] in ids
        assert tok.vocab_dict[tok.unk_token] not in ids

    def test_handle_punctuation_false_splits_on_whitespace_only(self):
        vocab_dict, merges = self._trained(handle_punctuation=False)
        tok = BPETokenizer(
            vocab_dict=vocab_dict, merges=merges, max_length=16,
            handle_punctuation=False,
        )
        ids = tok.tokenize_texts(["hello, world"])[0]
        assert vocab_dict["hello,</w>"] in ids

    def test_handle_punctuation_is_serialized(self):
        tok = BPETokenizer(handle_punctuation=False)
        assert tok.get_config()["handle_punctuation"] is False
        assert BPETokenizer.from_config(tok.get_config()).handle_punctuation is False

    def test_long_text_still_ends_in_eos(self, trained_vocab):
        vocab_dict, merges = trained_vocab
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges, max_length=8)
        eos = tok.vocab_dict[tok.eos_token]
        ids = tok.tokenize_texts(["the quick brown fox " * 6])[0]
        assert len(ids) == 8
        assert ids[-1] == eos

    def test_short_text_has_eos_then_padding(self, trained_vocab):
        vocab_dict, merges = trained_vocab
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges, max_length=16)
        eos = tok.vocab_dict[tok.eos_token]
        pad = tok.vocab_dict[tok.pad_token]
        ids = tok.tokenize_texts(["the fox"])[0]
        first_pad = ids.index(pad)
        assert ids[first_pad - 1] == eos
        assert set(ids[first_pad:]) == {pad}

    def test_max_length_one_keeps_only_eos(self, trained_vocab):
        vocab_dict, merges = trained_vocab
        tok = BPETokenizer(vocab_dict=vocab_dict, merges=merges, max_length=1)
        assert tok.tokenize_texts(["the quick fox"])[0] == [
            tok.vocab_dict[tok.eos_token]
        ]


# ---------------------------------------------------------------------
# TokenEmbedding
# ---------------------------------------------------------------------

class TestTokenEmbedding:

    def test_construction(self):
        emb = TokenEmbedding(vocab_size=100, embedding_dim=32)
        assert emb.vocab_size == 100
        assert emb.embedding_dim == 32

    def test_invalid_vocab_size(self):
        with pytest.raises(ValueError):
            TokenEmbedding(vocab_size=0, embedding_dim=32)

    def test_invalid_embedding_dim(self):
        with pytest.raises(ValueError):
            TokenEmbedding(vocab_size=100, embedding_dim=-1)

    def test_forward_pass(self):
        emb = TokenEmbedding(vocab_size=100, embedding_dim=32)
        tokens = keras.ops.convert_to_tensor(
            np.random.randint(0, 100, size=(4, 16)).astype("int32")
        )
        out = emb(tokens)
        assert tuple(out.shape) == (4, 16, 32)

    def test_compute_output_shape_matches_call(self):
        emb = TokenEmbedding(vocab_size=100, embedding_dim=32)
        tokens = keras.ops.convert_to_tensor(
            np.random.randint(0, 100, size=(4, 16)).astype("int32")
        )
        out = emb(tokens)
        computed = emb.compute_output_shape((4, 16))
        assert tuple(out.shape) == tuple(computed)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(16,), dtype="int32")
        out = TokenEmbedding(vocab_size=100, embedding_dim=32, name="tok_emb")(inp)
        model = keras.Model(inp, out)

        tokens = np.random.randint(0, 100, size=(4, 16)).astype("int32")
        y_before = model(tokens)

        path = os.path.join(tmp_path, "tok_emb.keras")
        model.save(path)
        reloaded = keras.models.load_model(
            path, custom_objects={"TokenEmbedding": TokenEmbedding}
        )
        y_after = reloaded(tokens)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_before),
            keras.ops.convert_to_numpy(y_after),
            rtol=1e-6, atol=1e-6,
        )

    def test_get_config(self):
        emb = TokenEmbedding(vocab_size=50, embedding_dim=8, mask_zero=False)
        config = emb.get_config()
        assert config["vocab_size"] == 50
        assert config["embedding_dim"] == 8
        assert config["mask_zero"] is False
