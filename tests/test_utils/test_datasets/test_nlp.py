"""Tests for ``dl_techniques.datasets.nlp``.

Covers ``_hf_to_tf_dataset`` sharding semantics (D-002) and the
``min_article_length`` default change (D-003). The Wikipedia loader itself
is not exercised — we mock the HF dataset with an in-memory
``datasets.Dataset`` to keep the tests offline.
"""

from __future__ import annotations

import datasets
import pytest


@pytest.fixture
def small_hf_dataset():
    """In-memory HF dataset with deterministic content."""
    return datasets.Dataset.from_dict(
        {"text": [f"doc-{i:03d}" for i in range(32)]},
    )


class TestHfToTfDatasetSharding:
    """Behavioural tests for ``_hf_to_tf_dataset``'s ``num_shards`` parameter."""

    def test_single_shard_preserves_order(self, small_hf_dataset):
        from dl_techniques.datasets.nlp import _hf_to_tf_dataset

        ds = _hf_to_tf_dataset(small_hf_dataset, shuffle=False, num_shards=1)
        out = [t.numpy().decode() for t in ds]
        assert out == [f"doc-{i:03d}" for i in range(32)]

    def test_multi_shard_yields_all_elements(self, small_hf_dataset):
        from dl_techniques.datasets.nlp import _hf_to_tf_dataset

        ds = _hf_to_tf_dataset(
            small_hf_dataset, shuffle=True, seed=7, num_shards=4,
        )
        out = sorted(t.numpy().decode() for t in ds)
        assert out == sorted(f"doc-{i:03d}" for i in range(32))

    def test_same_seed_deterministic(self, small_hf_dataset):
        from dl_techniques.datasets.nlp import _hf_to_tf_dataset

        a = list(_hf_to_tf_dataset(
            small_hf_dataset, shuffle=True, seed=11, num_shards=4,
        ).as_numpy_iterator())
        b = list(_hf_to_tf_dataset(
            small_hf_dataset, shuffle=True, seed=11, num_shards=4,
        ).as_numpy_iterator())
        assert a == b

    def test_different_seed_reorders(self, small_hf_dataset):
        from dl_techniques.datasets.nlp import _hf_to_tf_dataset

        a = list(_hf_to_tf_dataset(
            small_hf_dataset, shuffle=True, seed=11, num_shards=4,
        ).as_numpy_iterator())
        b = list(_hf_to_tf_dataset(
            small_hf_dataset, shuffle=True, seed=99, num_shards=4,
        ).as_numpy_iterator())
        # Both contain the same elements but in different orders.
        assert sorted(a) == sorted(b)
        assert a != b

    def test_iterator_restart_under_multi_shard(self, small_hf_dataset):
        """tf.data.Dataset.from_generator rebuilds the generator on restart;
        verify the multi-shard pipeline yields a complete, full-cardinality
        sequence on each restart."""
        from dl_techniques.datasets.nlp import _hf_to_tf_dataset

        ds = _hf_to_tf_dataset(
            small_hf_dataset, shuffle=True, seed=11, num_shards=4,
        )
        first = list(ds.as_numpy_iterator())
        second = list(ds.as_numpy_iterator())
        assert sorted(first) == sorted(second) == sorted(
            (f"doc-{i:03d}".encode() for i in range(32))
        )


class TestLoadWikipediaDefaults:
    """Pin the new default for D-003 so a future regression doesn't silently
    re-instate the 500-char filter."""

    def test_min_article_length_default_is_zero(self):
        import inspect

        from dl_techniques.datasets.nlp import load_wikipedia_train_val

        sig = inspect.signature(load_wikipedia_train_val)
        assert sig.parameters["min_article_length"].default == 0
