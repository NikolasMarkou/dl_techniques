"""Tests for the ReID data module (no Market1501 download needed)."""

import numpy as np
import pytest

from dl_techniques.datasets.vision.reid import (
    parse_market1501_filename,
    reindex_person_ids,
    create_id_validation_split,
    synthetic_reid_generator,
    cmc_and_map,
    pk_batch_generator,
)


class TestMarket1501Parsing:
    def test_valid_filename(self):
        assert parse_market1501_filename("0002_c1s1_000451_03.jpg") == (2, 1)

    def test_double_extension(self):
        assert parse_market1501_filename("0002_c1s1_000451_03.jpg.jpg") == (2, 1)

    def test_invalid_names_rejected(self):
        assert parse_market1501_filename("Thumbs.db") is None
        assert parse_market1501_filename("0002_c1s1_000451_03.png") is None
        assert parse_market1501_filename("garbage.jpg") is None

    def test_reindex_contiguous(self):
        contiguous, mapping = reindex_person_ids([7, 3, 7, 12])
        assert contiguous == [1, 0, 1, 2]
        assert mapping == {3: 0, 7: 1, 12: 2}

    def test_validation_split_disjoint_ids(self):
        ids = np.array([i // 4 for i in range(40)])
        train_idx, val_idx = create_id_validation_split(ids, fraction=0.2, seed=1234)
        assert len(set(ids[train_idx]) & set(ids[val_idx])) == 0
        assert len(val_idx) == 8  # 2 of 10 identities


class TestSyntheticReid:
    def test_deterministic_shapes(self):
        first = list(synthetic_reid_generator(4, 3, seed=5))
        second = list(synthetic_reid_generator(4, 3, seed=5))
        assert len(first) == 12
        for (img_a, id_a), (img_b, id_b) in zip(first, second):
            assert img_a.shape == (128, 64, 3)
            assert id_a == id_b
            np.testing.assert_array_equal(img_a, img_b)

    def test_identities_separable_by_appearance(self):
        # Same-identity cosine distance must beat cross-identity distance:
        # otherwise no metric can learn the split and the smoke run is vacuous.
        samples = list(synthetic_reid_generator(6, 4, seed=0))
        by_id = {}
        for image, identity in samples:
            by_id.setdefault(identity, []).append(image.reshape(-1))
        ids = sorted(by_id)
        same, cross = [], []
        for i, a in enumerate(ids):
            same.append(1.0 - np.dot(by_id[a][0], by_id[a][1]) / (
                np.linalg.norm(by_id[a][0]) * np.linalg.norm(by_id[a][1])))
            b = ids[(i + 1) % len(ids)]
            cross.append(1.0 - np.dot(by_id[a][0], by_id[b][0]) / (
                np.linalg.norm(by_id[a][0]) * np.linalg.norm(by_id[b][0])))
        assert max(same) < min(cross)

    def test_pk_batches_cover_identities(self):
        images = [img for img, _ in synthetic_reid_generator(6, 4, seed=1)]
        ids = [i for _, i in synthetic_reid_generator(6, 4, seed=1)]
        gen = pk_batch_generator(images, ids, p_ids=3, k_shots=2, seed=0, augment=False)
        batch, labels = next(gen)
        assert batch.shape == (6, 128, 64, 3)
        assert sorted(set(labels.tolist())) == sorted(set(labels.tolist()))
        assert len(set(labels.tolist())) == 3
        with pytest.raises(ValueError):
            next(pk_batch_generator(images, ids, p_ids=99, k_shots=2))


class TestCmcMap:
    def test_perfect_ranking(self):
        rng = np.random.default_rng(0)
        gallery = rng.random((10, 8)).astype(np.float64)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        query = gallery[:4].copy()
        out = cmc_and_map(query, np.arange(4), gallery, np.arange(10))
        assert out["cmc@1"] == 1.0
        assert out["mAP"] == 1.0

    def test_keys_and_range(self):
        rng = np.random.default_rng(2)
        q = rng.random((6, 8))
        g = rng.random((20, 8))
        out = cmc_and_map(q, np.arange(6) % 3, g, np.arange(20) % 5,
                          topk=(1, 3))
        assert set(out) == {"cmc@1", "cmc@3", "mAP"}
        assert all(0.0 <= v <= 1.0 for v in out.values())
