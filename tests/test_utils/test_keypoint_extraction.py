"""Contract for ``dl_techniques.utils.keypoint_extraction``.

Oracles: the numpy ``grid_logits_to_heatmap`` and ``simple_nms`` of
``train.superpoint.homographic_adaptation`` (heatmap and NMS), and an explicit
numpy loop (bilinear sampling, align-corners, pixel centers at integers).
The ``TestMutantsAreRed`` class proves the NMS and coordinate checks can fail.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.utils import keypoint_extraction as ke
from train.superpoint.homographic_adaptation import (
    grid_logits_to_heatmap,
    simple_nms,
)

pytestmark = pytest.mark.usefixtures("tf32_disabled")


@pytest.fixture(scope="module")
def tf32_disabled():
    prev = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    try:
        yield False
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(prev)
        assert tf.config.experimental.tensor_float_32_execution_enabled() == prev


def _np(x):
    return keras.ops.convert_to_numpy(x)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _nms_oracle_mask(hm, radius):
    """Oracle NMS mask via simple_nms with threshold -1 (no thresholding)."""
    pts = simple_nms(hm, radius, -1.0)
    m = np.zeros(hm.shape, bool)
    for x, y in pts.astype(int):
        m[y, x] = True
    return m


def _nms_matches(fn, seeds=range(4), radii=(0, 1, 2, 4)):
    for s in seeds:
        hm = _rng(s).random((2, 23, 31)).astype("float32")
        for r in radii:
            got = _np(fn(hm, r))
            for b in range(2):
                if not np.array_equal(got[b], _nms_oracle_mask(hm[b], r)):
                    return False
    return True


def _bilinear_oracle(desc, kp):
    B, H, W, D = desc.shape
    out = np.zeros((B, kp.shape[1], D), np.float64)
    for b in range(B):
        for n in range(kp.shape[1]):
            x = min(max(float(kp[b, n, 0]), 0.0), W - 1)
            y = min(max(float(kp[b, n, 1]), 0.0), H - 1)
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
            wx, wy = x - x0, y - y0
            v = (
                desc[b, y0, x0] * (1 - wy) * (1 - wx)
                + desc[b, y0, x1] * (1 - wy) * wx
                + desc[b, y1, x0] * wy * (1 - wx)
                + desc[b, y1, x1] * wy * wx
            )
            out[b, n] = v / max(np.linalg.norm(v), 1e-12)
    return out


class TestHeatmap:
    def test_matches_numpy_oracle(self):
        logits = _rng(1).normal(size=(3, 5, 7, 65)).astype("float32") * 3
        got = _np(ke.superpoint_heatmap(logits))
        assert got.shape == (3, 40, 56) and got.dtype == np.float32
        for b in range(3):
            np.testing.assert_allclose(
                got[b], grid_logits_to_heatmap(logits[b]), atol=1e-6, rtol=0
            )

    def test_bad_channels_raise(self):
        with pytest.raises(ValueError):
            ke.superpoint_heatmap(np.zeros((1, 2, 2, 64), "float32"))


class TestNms:
    def test_matches_numpy_oracle(self):
        assert _nms_matches(ke.heatmap_nms)

    def test_ties_all_survive(self):
        hm = np.ones((1, 6, 6), "float32")
        assert _np(ke.heatmap_nms(hm, 2)).all()

    def test_negative_radius(self):
        with pytest.raises(ValueError):
            ke.heatmap_nms(np.zeros((1, 3, 3), "float32"), -1)


class TestSelect:
    def test_ordering_padding_and_coordinates(self):
        hm = np.zeros((1, 16, 20), "float32")
        hm[0, 3, 5] = 0.9   # (x=5, y=3)
        hm[0, 10, 17] = 0.5
        hm[0, 12, 2] = 0.7
        kp, sc, m = map(_np, ke.select_keypoints(hm, 6, 0.1, 2, 0))
        assert kp.shape == (1, 6, 2) and sc.shape == (1, 6) and m.shape == (1, 6)
        np.testing.assert_array_equal(m[0], [1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(kp[0, :3], [[5, 3], [2, 12], [17, 10]])
        np.testing.assert_allclose(sc[0, :3], [0.9, 0.7, 0.5])
        assert not kp[0, 3:].any() and not sc[0, 3:].any()

    def test_fewer_than_k_and_empty(self):
        hm = np.zeros((2, 8, 8), "float32")
        hm[0, 4, 4] = 0.3
        kp, sc, m = map(_np, ke.select_keypoints(hm, 5, 0.0, 1, 0))
        assert m.sum(1).tolist() == [1, 0]
        assert np.isfinite(kp).all() and np.isfinite(sc).all()
        assert not kp[1].any()

    def test_k_larger_than_pixels(self):
        hm = _rng(2).random((1, 3, 3)).astype("float32")
        kp, sc, m = map(_np, ke.select_keypoints(hm, 20, 0.0, 0, 0))
        assert kp.shape == (1, 20, 2) and m.sum() == 9
        assert not kp[0, 9:].any() and not m[0, 9:].any()

    def test_threshold_is_strict(self):
        hm = np.zeros((1, 5, 5), "float32")
        hm[0, 2, 2] = 0.5
        assert _np(ke.select_keypoints(hm, 3, 0.5, 1, 0)[2]).sum() == 0
        assert _np(ke.select_keypoints(hm, 3, 0.49, 1, 0)[2]).sum() == 1

    def test_border(self):
        hm = np.zeros((1, 12, 12), "float32")
        hm[0, 1, 6] = 0.9   # y=1 inside border 2
        hm[0, 6, 6] = 0.5
        hm[0, 6, 9] = 0.4   # x=9 = W-3, kept
        hm[0, 6, 10] = 0.8  # x=10 = W-2, removed
        _, _, m = ke.select_keypoints(hm, 6, 0.0, 0, 2)
        kp = _np(ke.select_keypoints(hm, 6, 0.0, 0, 2)[0])
        assert _np(m).sum() == 2
        np.testing.assert_array_equal(kp[0, :2], [[6, 6], [9, 6]])

    def test_tie_break_is_row_major(self):
        hm = np.zeros((1, 8, 8), "float32")
        for (y, x) in [(5, 1), (2, 6), (2, 3)]:
            hm[0, y, x] = 0.5
        kp = _np(ke.select_keypoints(hm, 3, 0.0, 0, 0)[0])
        np.testing.assert_array_equal(kp[0], [[3, 2], [6, 2], [1, 5]])

    def test_matches_oracle_selection_set(self):
        hm = _rng(5).random((2, 30, 40)).astype("float32") ** 3
        kp, sc, m = map(_np, ke.select_keypoints(hm, 400, 0.2, 3, 0))
        for b in range(2):
            ref = simple_nms(hm[b], 3, 0.2)
            got = kp[b][m[b]]
            assert {tuple(p) for p in got.tolist()} == {tuple(p) for p in ref.tolist()}
            assert (np.diff(sc[b][m[b]]) <= 0).all()


class TestSampling:
    @pytest.mark.parametrize(
        "pts",
        [
            [[0, 0], [3, 4], [11, 7], [5, 2]],               # integers, corners
            [[0.5, 0.5], [3.5, 4.5], [10.5, 6.5], [2.25, 1.75]],  # fractional
            [[-1.0, -2.0], [11.4, 7.9], [100.0, 100.0], [0.2, 6.9]],  # beyond border
        ],
    )
    def test_matches_numpy_loop(self, pts):
        desc = _rng(3).normal(size=(2, 8, 12, 6)).astype("float32")
        kp = np.broadcast_to(np.array(pts, "float32"), (2, 4, 2)).copy()
        got = _np(ke.sample_descriptors(desc, kp))
        np.testing.assert_allclose(got, _bilinear_oracle(desc, kp), atol=1e-5, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(got, axis=-1), 1.0, atol=1e-5)

    def test_integer_coordinate_returns_that_pixel(self):
        desc = _rng(4).normal(size=(1, 5, 6, 4)).astype("float32")
        got = _np(ke.sample_descriptors(desc, np.array([[[2, 3]]], "float32")))[0, 0]
        ref = desc[0, 3, 2] / np.linalg.norm(desc[0, 3, 2])
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0)

    def test_zero_descriptor_stays_finite(self):
        got = _np(ke.sample_descriptors(np.zeros((1, 4, 4, 3), "float32"),
                                        np.zeros((1, 2, 2), "float32")))
        assert np.isfinite(got).all() and not got.any()


def _outputs(b=2, h=32, w=48, d=16, seed=7):
    r = _rng(seed)
    logits = r.normal(size=(b, h // 8, w // 8, 65)).astype("float32") * 2
    desc = r.normal(size=(b, h, w, d)).astype("float32")
    desc /= np.linalg.norm(desc, axis=-1, keepdims=True)
    return {"keypoints": logits, "descriptors": desc}


class TestDecode:
    def test_batch_with_different_counts(self):
        out = _outputs()
        # sparsify image 1 so it has fewer real points than image 0
        out["keypoints"][1] -= 0
        out["keypoints"][1][..., :64] -= 8.0
        res = {k: _np(v) for k, v in ke.decode_superpoint(
            out, max_keypoints=64, threshold=0.02, nms_radius=2, border=2).items()}
        n = res["mask"].sum(1)
        assert n[0] != n[1] and n.max() > 0
        assert res["keypoints"].shape == (2, 64, 2)
        assert res["descriptors"].shape == (2, 64, 16)
        norms = np.linalg.norm(res["descriptors"], axis=-1)
        np.testing.assert_allclose(norms[res["mask"]], 1.0, atol=1e-5)
        assert not res["descriptors"][~res["mask"]].any()
        assert not res["keypoints"][~res["mask"]].any()
        # real points are in-bounds and respect the border
        kp = res["keypoints"][res["mask"]]
        assert (kp[:, 0] >= 2).all() and (kp[:, 0] <= 48 - 3).all()
        assert (kp[:, 1] >= 2).all() and (kp[:, 1] <= 32 - 3).all()

    def test_equals_oracle_pipeline(self):
        out = _outputs(b=1)
        res = {k: _np(v) for k, v in ke.decode_superpoint(
            out, 500, 0.01, 3, 0).items()}
        hm = grid_logits_to_heatmap(out["keypoints"][0])
        ref = simple_nms(hm, 3, 0.01)
        got = res["keypoints"][0][res["mask"][0]]
        assert {tuple(p) for p in got.tolist()} == {tuple(p) for p in ref.tolist()}
        np.testing.assert_allclose(
            res["descriptors"][0][res["mask"][0]],
            _bilinear_oracle(out["descriptors"], res["keypoints"])[0][res["mask"][0]],
            atol=1e-5, rtol=0)

    def test_graph_mode_equals_eager(self):
        out = _outputs()
        eager = ke.decode_superpoint(out, 40, 0.01, 2, 2)

        @tf.function(input_signature=[
            tf.TensorSpec((None, 4, 6, 65)), tf.TensorSpec((None, 32, 48, 16))])
        def run(a, d):
            return ke.decode_superpoint({"keypoints": a, "descriptors": d}, 40, 0.01, 2, 2)

        graph = run(tf.constant(out["keypoints"]), tf.constant(out["descriptors"]))
        for k in eager:
            np.testing.assert_allclose(_np(graph[k]), _np(eager[k]), atol=1e-6, rtol=0)

    @pytest.mark.parametrize("policy", ["mixed_float16", "mixed_bfloat16"])
    def test_policy_inputs_give_float32(self, policy):
        prev = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            out = _outputs()
            low = "float16" if "float16" in policy else "bfloat16"
            cast = {k: tf.cast(v, low) for k, v in out.items()}
            res = ke.decode_superpoint(cast, 32, 0.01, 2, 2)
            for k in ("keypoints", "scores", "descriptors"):
                assert _np(res[k]).dtype == np.float32
                assert np.isfinite(_np(res[k])).all()
            assert _np(res["mask"]).dtype == np.bool_
        finally:
            keras.mixed_precision.set_global_policy(prev)


class TestMutantsAreRed:
    def test_nms_without_suppression_is_red(self):
        assert not _nms_matches(lambda hm, r: np.ones(hm.shape, bool))

    def test_nms_wrong_window_is_red(self):
        assert not _nms_matches(lambda hm, r: ke.heatmap_nms(hm, r + 1))

    def test_xy_swapped_is_red(self):
        desc = _rng(3).normal(size=(1, 8, 12, 6)).astype("float32")
        kp = np.array([[[3, 4], [10, 2]]], "float32")
        ref = _bilinear_oracle(desc, kp)
        good = _np(ke.sample_descriptors(desc, kp))
        bad = _np(ke.sample_descriptors(desc, kp[..., ::-1].copy()))
        assert np.allclose(good, ref, atol=1e-5)
        assert not np.allclose(bad, ref, atol=1e-5)

    def test_selection_xy_swap_is_red(self):
        hm = np.zeros((1, 16, 20), "float32")
        hm[0, 3, 5] = 0.9
        kp = _np(ke.select_keypoints(hm, 2, 0.1, 2, 0)[0])[0, 0]
        assert kp.tolist() == [5.0, 3.0] and kp[::-1].tolist() != kp.tolist()
