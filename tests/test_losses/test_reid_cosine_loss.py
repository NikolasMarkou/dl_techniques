"""Tests for the ReID metric-learning losses."""

import keras
import numpy as np
import pytest

from dl_techniques.losses import SoftmarginTripletLoss, magnet_loss_fn


def _unit_cluster(n_per_id, n_ids, dim=16, noise=0.02, seed=0):
    # Identity centers are orthonormal basis vectors, so clusters are ~90
    # degrees apart ON the sphere (a coordinate-separation trick collapses
    # under L2 normalization -- every center lands on the same pole).
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for i in range(n_ids):
        center = np.zeros(dim, dtype=np.float32)
        center[i % dim] = 1.0
        feats.append(
            center + rng.normal(0, noise, (n_per_id, dim)).astype(np.float32)
        )
        labels += [i] * n_per_id
    feats = np.concatenate(feats, axis=0)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return feats, np.array(labels, dtype=np.int32)


class TestSoftmarginTripletLoss:
    def test_per_sample_shape(self):
        feats, labels = _unit_cluster(4, 3)
        out = SoftmarginTripletLoss().call(labels, feats)
        assert tuple(out.shape) == (12,)

    def test_well_separated_clusters_near_floor(self):
        # On the unit sphere the diameter is 2, so even perfect clusters sit
        # at softplus(-(sqrt(2) - spread)) ~= 0.24, not 0. The claim is the
        # floor: tight orthonormal clusters beat overlapping ones.
        feats, labels = _unit_cluster(4, 3)
        rng = np.random.default_rng(4)
        overlap = rng.normal(0, 1, (12, 16)).astype(np.float32)
        overlap /= np.linalg.norm(overlap, axis=1, keepdims=True)
        overlap_labels = np.array([0] * 6 + [1] * 6, dtype=np.int32)
        good = float(np.asarray(SoftmarginTripletLoss().call(labels, feats)).mean())
        bad = float(
            np.asarray(SoftmarginTripletLoss().call(overlap_labels, overlap)).mean()
        )
        assert good < 0.3
        assert good < bad

    def test_collapsed_identities_positive_loss(self):
        rng = np.random.default_rng(4)
        feats = rng.normal(0, 1, (8, 16)).astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        labels = np.array([0] * 4 + [1] * 4, dtype=np.int32)
        out = np.asarray(SoftmarginTripletLoss().call(labels, feats))
        assert bool(np.all(np.isfinite(out)))
        assert float(out.mean()) > 0.0

    def test_single_identity_is_finite(self):
        feats, labels = _unit_cluster(4, 1)
        out = np.asarray(SoftmarginTripletLoss().call(labels, feats))
        assert bool(np.all(np.isfinite(out)))

    def test_config_round_trip(self):
        rebuilt = SoftmarginTripletLoss.from_config(SoftmarginTripletLoss().get_config())
        assert isinstance(rebuilt, SoftmarginTripletLoss)


class TestMagnetLossFn:
    def _raw_clusters(self, std=0.05, sep=20.0, seed=0):
        rng = np.random.default_rng(seed)
        feats, labels = [], []
        for i in range(4):
            center = np.zeros(16, dtype=np.float32)
            center[0] = i * sep
            feats.append(center + rng.normal(0, std, (8, 16)).astype(np.float32))
            labels += [i] * 8
        return np.concatenate(feats, axis=0), np.array(labels)

    def test_compact_clusters_small_loss(self):
        feats, labels = self._raw_clusters(std=0.05)
        loss, means, var = magnet_loss_fn(feats, labels)
        assert np.isfinite(loss)
        assert loss < 1e-3
        assert means.shape == (4, 16)
        # Means recover the true centers.
        np.testing.assert_allclose(
            means[:, 0], [0.0, 20.0, 40.0, 60.0], atol=0.2, rtol=0
        )
        assert var >= 0.0

    def test_overlapping_clusters_larger_loss(self):
        overlap_loss, _, _ = magnet_loss_fn(*self._raw_clusters(std=3.0, sep=2.0))
        compact_loss, _, _ = magnet_loss_fn(*self._raw_clusters(std=0.05))
        assert np.isfinite(overlap_loss)
        assert overlap_loss > compact_loss
