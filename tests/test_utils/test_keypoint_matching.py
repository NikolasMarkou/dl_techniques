"""Contract for ``dl_techniques.utils.keypoint_matching``.

Oracle: an explicit python-loop float64 transcription of the rules in the module
docstring. Hand-built cases pin the outcome independently of the oracle. The
``TestMutantsAreRed`` class proves the mutual-NN and inverse-direction checks can fail.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.utils import keypoint_matching as km

pytestmark = pytest.mark.usefixtures("tf32_disabled")

SIZE = (200.0, 160.0)


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


def _run(kp0, kp1, m0, m1, H, s0=SIZE, s1=SIZE, **kw):
    B = kp0.shape[0]
    sz = lambda s: np.tile(np.asarray(s, "float32")[None], (B, 1))
    out = km.homography_matches(kp0, kp1, m0, m1, H, sz(s0), sz(s1), **kw)
    return _np(out["matches0"]), _np(out["matches1"])


def _oracle(kp0, kp1, m0, m1, H, s0, s1, pos=3.0, neg=3.0):
    """Loop oracle for one image; returns (matches0, matches1)."""
    M, N = len(kp0), len(kp1)
    H = H.astype(np.float64)
    det = np.linalg.det(H)
    ok = abs(det) >= 1e-6 * np.linalg.norm(H) ** 3
    Hi = np.linalg.inv(H) if ok else np.eye(3)

    def proj(p, T, size):
        v = T @ np.array([p[0], p[1], 1.0])
        if not v[2] > 1e-6:
            return None
        q = v[:2] / v[2]
        if not (0 <= q[0] < size[0] and 0 <= q[1] < size[1]):
            return None
        return q

    p0 = [proj(kp0[i], H, s1) if (m0[i] and ok) else None for i in range(M)]
    p1 = [proj(kp1[j], Hi, s0) if (m1[j] and ok) else None for j in range(N)]
    D = np.full((M, N), np.inf)
    for i in range(M):
        for j in range(N):
            if p0[i] is not None and p1[j] is not None:
                D[i, j] = max(
                    np.linalg.norm(p0[i] - kp1[j]), np.linalg.norm(kp0[i] - p1[j])
                )

    def side(D, mk):
        n = D.shape[0]
        res = np.full(n, -2)
        for i in range(n):
            if not mk[i]:
                continue
            row = D[i]
            j = int(np.argmin(row))
            if row[j] <= pos and int(np.argmin(D[:, j])) == i:
                res[i] = j
            elif row.min() > neg:
                res[i] = -1
        return res

    return side(D, m0), side(D.T, m1)


def _random_case(seed, B=3, M=24, N=20):
    rng = np.random.default_rng(seed)
    Hs = []
    for _ in range(B):
        H = np.eye(3) + rng.normal(scale=0.04, size=(3, 3))
        H[0, 2] += rng.uniform(-15, 15)
        H[1, 2] += rng.uniform(-15, 15)
        H[2, :2] *= 1e-3
        H[2, 2] = 1.0
        Hs.append(H)
    H = np.stack(Hs)
    kp0 = rng.uniform([0, 0], SIZE, size=(B, M, 2))
    kp1 = rng.uniform([-5, -5], [SIZE[0] + 5, SIZE[1] + 5], size=(B, N, 2))
    for b in range(B):  # plant true correspondences with jitter, some duplicates
        for k in range(10):
            v = H[b] @ np.append(kp0[b, k], 1.0)
            kp1[b, k] = v[:2] / v[2] + rng.normal(scale=1.2, size=2)
        kp1[b, 11] = kp1[b, 10] + rng.normal(scale=0.5, size=2)
    m0 = rng.random((B, M)) > 0.2
    m1 = rng.random((B, N)) > 0.2
    return (
        kp0.astype("float32"),
        kp1.astype("float32"),
        m0,
        m1,
        H.astype("float32"),
    )


class TestOracle:
    @pytest.mark.parametrize("seed", range(6))
    def test_matches_loop_oracle(self, seed):
        kp0, kp1, m0, m1, H = _random_case(seed)
        g0, g1 = _run(kp0, kp1, m0, m1, H)
        assert g0.dtype == np.int32 and g1.dtype == np.int32
        assert (g0 >= 0).sum() > 0
        for b in range(len(H)):
            e0, e1 = _oracle(kp0[b], kp1[b], m0[b], m1[b], H[b], SIZE, SIZE)
            np.testing.assert_array_equal(g0[b], e0)
            np.testing.assert_array_equal(g1[b], e1)

    @pytest.mark.parametrize("neg", [3.0, 6.0])
    def test_oracle_with_wider_band(self, neg):
        kp0, kp1, m0, m1, H = _random_case(11)
        g0, g1 = _run(kp0, kp1, m0, m1, H, neg_threshold=neg)
        for b in range(len(H)):
            e0, e1 = _oracle(kp0[b], kp1[b], m0[b], m1[b], H[b], SIZE, SIZE, neg=neg)
            np.testing.assert_array_equal(g0[b], e0)
            np.testing.assert_array_equal(g1[b], e1)

    @pytest.mark.parametrize("policy", ["mixed_float16", "float32"])
    def test_policy_inputs_give_int32(self, policy):
        kp0, kp1, m0, m1, H = _random_case(2)
        old = keras.config.dtype_policy()
        keras.config.set_dtype_policy(policy)
        try:
            dt = "float16" if policy == "mixed_float16" else "float32"
            g0, g1 = _run(
                kp0.astype(dt), kp1.astype(dt), m0.astype(dt), m1.astype(dt), H.astype(dt)
            )
        finally:
            keras.config.set_dtype_policy(old)
        assert g0.dtype == np.int32 and g1.dtype == np.int32
        assert (g0 >= 0).sum() > 0


class TestHandBuilt:
    def test_identity_with_duplicates(self):
        kp = np.array([[[10, 10], [10, 10], [50, 60], [90, 20]]], "float32")
        m = np.ones((1, 4), bool)
        H = np.eye(3, dtype="float32")[None]
        g0, g1 = _run(kp, kp, m, m, H)
        # first copy wins the mutual-NN contest, the duplicate is ignored
        np.testing.assert_array_equal(g0[0], [0, -2, 2, 3])
        np.testing.assert_array_equal(g1[0], [0, -2, 2, 3])

    def test_translation_known_outcome(self):
        kp0 = np.array([[[10, 10], [100, 50], [30, 120], [150, 150]]], "float32")
        H = np.array([[[1, 0, 20], [0, 1, 5], [0, 0, 1]]], "float32")
        kp1 = np.array([[[30.5, 15], [120, 55.5], [190, 130], [5, 5]]], "float32")
        m = np.ones((1, 4), bool)
        g0, g1 = _run(kp0, kp1, m, m, H)
        # kp0[2] -> (50,125) and kp0[3] -> (170,155): in image, no kp1 near: dustbin
        np.testing.assert_array_equal(g0[0], [0, 1, -1, -1])
        np.testing.assert_array_equal(g1[0], [0, 1, -1, -1])

    def test_warp_outside_other_image_is_dustbin(self):
        kp0 = np.array([[[190, 10], [20, 20]]], "float32")
        H = np.array([[[1, 0, 30], [0, 1, 0], [0, 0, 1]]], "float32")
        kp1 = np.array([[[220, 10], [50, 20]]], "float32")  # first is outside image 1
        m = np.ones((1, 2), bool)
        g0, g1 = _run(kp0, kp1, m, m, H)
        # kp0[0] -> (220,10) outside; kp1[0] is outside image 1 itself, both dustbin
        np.testing.assert_array_equal(g0[0], [-1, 1])
        np.testing.assert_array_equal(g1[0], [-1, 1])

    def test_ambiguity_band_is_ignored(self):
        kp0 = np.array([[[50, 50], [120, 80]]], "float32")
        kp1 = np.array([[[52.5, 50], [120, 80]]], "float32")
        H = np.eye(3, dtype="float32")[None]
        m = np.ones((1, 2), bool)
        # dist 2.5: within pos (3) so positive by default
        g0, _ = _run(kp0, kp1, m, m, H)
        np.testing.assert_array_equal(g0[0], [0, 1])
        # tighter pos with a wider band: 2.5 in (1, 3] -> ignored, not dustbin
        g0, g1 = _run(kp0, kp1, m, m, H, pos_threshold=1.0, neg_threshold=3.0)
        np.testing.assert_array_equal(g0[0], [-2, 1])
        np.testing.assert_array_equal(g1[0], [-2, 1])
        # beyond the outer threshold: dustbin
        g0, _ = _run(kp0, kp1, m, m, H, pos_threshold=1.0, neg_threshold=2.0)
        np.testing.assert_array_equal(g0[0], [-1, 1])

    def test_mutual_nn_failure_gives_one_positive(self):
        kp0 = np.array([[[50, 50], [51, 50]]], "float32")  # both want kp1[0]
        kp1 = np.array([[[50.2, 50]]], "float32")
        H = np.eye(3, dtype="float32")[None]
        g0, g1 = _run(kp0, kp1, np.ones((1, 2), bool), np.ones((1, 1), bool), H)
        np.testing.assert_array_equal(g0[0], [0, -2])
        np.testing.assert_array_equal(g1[0], [0])
        assert (g0 >= 0).sum() == 1

    def test_padded_slots_are_ignored_and_never_match(self):
        kp0 = np.array([[[50, 50], [50, 50]]], "float32")
        kp1 = np.array([[[50, 50], [50, 50]]], "float32")
        H = np.eye(3, dtype="float32")[None]
        g0, g1 = _run(
            kp0, kp1, np.array([[1, 0]], bool), np.array([[0, 1]], bool), H
        )
        # real kp0[0] pairs with real kp1[1]; coincident padded slots never match
        np.testing.assert_array_equal(g0[0], [1, -2])
        np.testing.assert_array_equal(g1[0], [-2, 0])

    def test_all_masked_side(self):
        kp0 = np.array([[[50, 50], [60, 60]]], "float32")
        kp1 = np.array([[[50, 50]]], "float32")
        H = np.eye(3, dtype="float32")[None]
        g0, g1 = _run(kp0, kp1, np.zeros((1, 2), bool), np.ones((1, 1), bool), H)
        np.testing.assert_array_equal(g0[0], [-2, -2])
        np.testing.assert_array_equal(g1[0], [-1])
        g0, g1 = _run(kp0, kp1, np.zeros((1, 2), bool), np.zeros((1, 1), bool), H)
        assert (g0 == -2).all() and (g1 == -2).all()

    def test_threshold_order_raises(self):
        kp = np.zeros((1, 1, 2), "float32")
        with pytest.raises(ValueError, match="neg_threshold"):
            _run(kp, kp, np.ones((1, 1)), np.ones((1, 1)), np.eye(3)[None].astype("float32"),
                 pos_threshold=3.0, neg_threshold=1.0)


class TestProperties:
    @pytest.mark.parametrize("seed", range(5))
    def test_bidirectional_consistency(self, seed):
        kp0, kp1, m0, m1, H = _random_case(seed + 20)
        g0, g1 = _run(kp0, kp1, m0, m1, H)
        for b in range(len(H)):
            for i, j in enumerate(g0[b]):
                if j >= 0:
                    assert g1[b, j] == i
            for j, i in enumerate(g1[b]):
                if i >= 0:
                    assert g0[b, i] == j

    @pytest.mark.parametrize("H", [np.zeros((3, 3)), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
                                   np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]]),
                                   np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1e-12]])])
    def test_near_singular_no_nan_all_dustbin(self, H):
        kp0, kp1, m0, m1, _ = _random_case(3, B=1)
        g0, g1 = _run(kp0, kp1, m0, m1, H[None].astype("float32"))
        assert (g0[m0] == -1).all() and (g1[m1] == -1).all()
        assert (g0[~m0] == -2).all() and (g1[~m1] == -2).all()

    def test_nan_homography_does_not_raise(self):
        kp0, kp1, m0, m1, _ = _random_case(3, B=1)
        H = np.full((1, 3, 3), np.nan, "float32")
        g0, g1 = _run(kp0, kp1, m0, m1, H)
        assert (g0[m0] == -1).all()

    def test_graph_mode_equals_eager(self):
        kp0, kp1, m0, m1, H = _random_case(4)
        sz = np.tile(np.asarray(SIZE, "float32")[None], (3, 1))
        eager = km.homography_matches(kp0, kp1, m0, m1, H, sz, sz)

        @tf.function
        def fn(a, b, c, d, h, s):
            return km.homography_matches(a, b, c, d, h, s, s)

        graph = fn(kp0, kp1, m0, m1, H, sz)
        for k in ("matches0", "matches1"):
            np.testing.assert_array_equal(_np(eager[k]), _np(graph[k]))

    def test_label_statistics(self):
        m = np.array([[0, -1, -2, -2]], "int32")
        mask = np.array([[1, 1, 1, 0]], bool)
        s = {k: float(_np(v)) for k, v in km.label_statistics(m, mask).items()}
        assert s == pytest.approx({"positive": 1 / 3, "dustbin": 1 / 3, "ignored": 1 / 3})
        z = km.label_statistics(m, np.zeros((1, 4), bool))
        assert float(_np(z["positive"])) == 0.0


class TestMutantsAreRed:
    """Each mutant must break at least one guard above (shown to fail)."""

    @staticmethod
    def _agrees_with_oracle(seeds=range(4)):
        for s in seeds:
            kp0, kp1, m0, m1, H = _random_case(s)
            g0, g1 = _run(kp0, kp1, m0, m1, H)
            for b in range(len(H)):
                e0, e1 = _oracle(kp0[b], kp1[b], m0[b], m1[b], H[b], SIZE, SIZE)
                if not (np.array_equal(g0[b], e0) and np.array_equal(g1[b], e1)):
                    return False
        return True

    def test_baseline_green(self):
        assert self._agrees_with_oracle()

    def test_no_mutual_check_is_red(self, monkeypatch):
        real = km._mutual_nn

        def mutant(dist, thr):
            nn0, nn1, min0, min1, _, _ = real(dist, thr)
            return nn0, nn1, min0, min1, min0 <= thr, min1 <= thr

        monkeypatch.setattr(km, "_mutual_nn", mutant)
        assert not self._agrees_with_oracle()
        kp0 = np.array([[[50, 50], [51, 50]]], "float32")
        kp1 = np.array([[[50.2, 50]]], "float32")
        g0, _ = _run(kp0, kp1, np.ones((1, 2), bool), np.ones((1, 1), bool),
                     np.eye(3, dtype="float32")[None])
        assert (g0 >= 0).sum() == 2  # the one-positive guard would fail

    def test_forward_h_instead_of_inverse_is_red(self, monkeypatch):
        real = km.invert_3x3
        monkeypatch.setattr(km, "invert_3x3", lambda h: (km.ops.cast(h, "float32"), real(h)[1]))
        assert not self._agrees_with_oracle()

    def test_outside_image_not_excluded_is_red(self, monkeypatch):
        real = km._project

        def mutant(points, h, size):
            xy, _ = real(points, h, size * 0 + 1e9)
            return xy, km.ops.ones_like(points[..., 0], dtype="bool")

        monkeypatch.setattr(km, "_project", mutant)
        assert not self._agrees_with_oracle()
