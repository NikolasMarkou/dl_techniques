"""Tests for ``ByteTokenizer`` (``dl_techniques.layers.blt.byte_tokenizer``)."""

from dl_techniques.layers.blt.byte_tokenizer import ByteTokenizer

B, SEQ = 2, 10


class TestByteTokenizer:

    def test_text_round_trip(self):
        tok = ByteTokenizer()
        ids = tok.text_to_bytes("Hello", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
        assert tok.tokens_to_text(ids) == "Hello"

    def test_compute_output_shape(self):
        assert ByteTokenizer().compute_output_shape((B, SEQ)) == (B, None)

    def test_get_config_round_trip(self):
        tok = ByteTokenizer(vocab_size=300, byte_offset=5)
        rebuilt = ByteTokenizer.from_config(tok.get_config())
        assert rebuilt.vocab_size == 300 and rebuilt.byte_offset == 5
