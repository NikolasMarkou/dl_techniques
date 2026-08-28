"""Stage 1 must feed packed, padding-free batches.

This is the mitigation the whole study rests on. One arm's block cannot honour
a padding mask, so if padding ever reappears in stage 1 the two arms are no
longer being compared on the same footing and every stage-1 number becomes
uninterpretable. That failure would be completely silent -- the shapes are
identical either way -- so it is pinned here.
"""

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.tokenizers.ascii_char import (
    CLS_ID,
    MASK_ID,
    PAD_ID,
    SEP_ID,
    VOCAB_SIZE,
)
from train.embeddings_experimental.data import (
    build_packed_mlm_dataset,
    packed_character_windows,
)

SEQ_LEN = 32


@pytest.fixture
def text_dataset():
    """A handful of documents of deliberately mismatched lengths."""
    texts = [
        "the quick brown fox jumps over the lazy dog. " * 3,
        "short one.",
        "a" * 200,
        "another document with a different length entirely, for good measure.",
    ]
    return tf.data.Dataset.from_tensor_slices(texts)


class TestPackedWindows:
    """The generator itself."""

    def test_every_window_is_exactly_seq_len(self, text_dataset):
        windows = list(
            packed_character_windows(
                lambda: iter(text_dataset.as_numpy_iterator()), SEQ_LEN
            )
        )
        assert windows
        assert all(w.shape == (SEQ_LEN,) for w in windows)

    def test_no_window_contains_a_pad_id(self, text_dataset):
        """The load-bearing assertion."""
        windows = list(
            packed_character_windows(
                lambda: iter(text_dataset.as_numpy_iterator()), SEQ_LEN
            )
        )
        offenders = [i for i, w in enumerate(windows) if PAD_ID in w.tolist()]
        assert offenders == [], (
            f"windows {offenders} contain padding; stage 1 must be packed, "
            "or the Clifford arm's numbers stop being comparable"
        )

    def test_ids_are_in_range(self, text_dataset):
        for window in packed_character_windows(
            lambda: iter(text_dataset.as_numpy_iterator()), SEQ_LEN
        ):
            assert window.min() >= 0
            assert window.max() < VOCAB_SIZE

    def test_documents_are_separated(self, text_dataset):
        """A separator must appear, or documents run together invisibly."""
        joined = np.concatenate(
            list(
                packed_character_windows(
                    lambda: iter(text_dataset.as_numpy_iterator()), SEQ_LEN
                )
            )
        )
        assert SEP_ID in joined.tolist()

    def test_a_trailing_partial_window_is_discarded(self):
        """Emitting a short final window would reintroduce padding."""
        ds = tf.data.Dataset.from_tensor_slices(["abc"])
        windows = list(
            packed_character_windows(lambda: iter(ds.as_numpy_iterator()), SEQ_LEN)
        )
        assert windows == []

    def test_empty_documents_are_skipped_without_raising(self):
        ds = tf.data.Dataset.from_tensor_slices(["", "x" * 100, ""])
        windows = list(
            packed_character_windows(lambda: iter(ds.as_numpy_iterator()), SEQ_LEN)
        )
        assert all(w.shape == (SEQ_LEN,) for w in windows)


class TestPackedDataset:
    """The tf.data pipeline the trainer actually consumes."""

    def test_batches_carry_the_two_expected_keys(self, text_dataset):
        batch = next(
            iter(
                build_packed_mlm_dataset(
                    text_dataset, seq_len=SEQ_LEN, batch_size=2, training=False
                )
            )
        )
        assert set(batch) == {"input_ids", "attention_mask"}

    def test_the_attention_mask_is_all_ones(self, text_dataset):
        """Packed means every position is real."""
        for batch in build_packed_mlm_dataset(
            text_dataset, seq_len=SEQ_LEN, batch_size=2, training=False
        ):
            mask = batch["attention_mask"].numpy()
            assert mask.min() == 1, "a zero in the mask means padding crept in"

    def test_no_batch_contains_a_pad_id(self, text_dataset):
        for batch in build_packed_mlm_dataset(
            text_dataset, seq_len=SEQ_LEN, batch_size=2, training=False
        ):
            assert PAD_ID not in batch["input_ids"].numpy().tolist()

    def test_shapes(self, text_dataset):
        batch = next(
            iter(
                build_packed_mlm_dataset(
                    text_dataset, seq_len=SEQ_LEN, batch_size=2, training=False
                )
            )
        )
        assert batch["input_ids"].shape == (2, SEQ_LEN)
        assert batch["attention_mask"].shape == (2, SEQ_LEN)

    def test_partial_batches_are_dropped(self, text_dataset):
        """A ragged final batch would change the contrastive batch size."""
        for batch in build_packed_mlm_dataset(
            text_dataset, seq_len=SEQ_LEN, batch_size=3, training=False
        ):
            assert batch["input_ids"].shape[0] == 3
