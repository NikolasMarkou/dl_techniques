"""Tests for the BPE tokenizer layers.

Covers ``BPETokenizer`` (string -> token-ID preprocessing layer) and
``TokenEmbedding`` (token-ID -> dense vectors), plus the ``train_bpe`` helper.
Includes construction (incl. ``ValueError`` paths), tokenization behaviour, a
forward pass, ``compute_output_shape`` agreement, and full ``.keras``
serialization round-trips.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.tokenizers.bpe import (
    train_bpe,
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
