"""Contract for ``dl_techniques.utils.keypoint_matching``.

Oracle: a float64 numpy transcription of glue-factory's
``gt_matches_from_homography`` (not of this repo's rule), plus a frozen-output
test against that function run in torch. Hand-built cases pin the outcome
independently of the oracle. The
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


def _gf_gt_matches(kp0, kp1, H, pos_th, neg_th, visible0=None, visible1=None):
    """Numpy float64 transcription of glue-factory ``gt_matches_from_homography``.

    Source: cvg/glue-factory ``gluefactory/geometry/gt_generation.py`` (main,
    last touched 2025-07-10, commit bd356aa), line by line: squared distances,
    ``dist = max(dist0, dist1)``, positive = mutual argmin and ``dist < pos^2``,
    ``negative0`` from the FORWARD distances ``dist0`` only and ``negative1``
    from the BACKWARD distances ``dist1`` only, ``-1`` unmatched, ``-2`` ignore.
    It is NOT a transcription of this repo's rule.

    The optional ``visible*`` masks add the repo's documented extension (the
    same device glue-factory uses in its pose/depth variant): pairs with an
    invisible member are infinite in ``dist``, and an invisible keypoint is
    unmatched.
    """
    Hi = np.linalg.inv(H)

    def warp(p, T):
        v = np.concatenate([p, np.ones((len(p), 1))], -1) @ T.T
        return v[:, :2] / v[:, 2:3]

    kp0_1, kp1_0 = warp(kp0, H), warp(kp1, Hi)
    dist0 = ((kp0_1[:, None] - kp1[None]) ** 2).sum(-1)
    dist1 = ((kp0[:, None] - kp1_0[None]) ** 2).sum(-1)
    dist = np.maximum(dist0, dist1)
    vis0 = np.ones(len(kp0), bool) if visible0 is None else visible0
    vis1 = np.ones(len(kp1), bool) if visible1 is None else visible1
    dist = np.where(vis0[:, None] & vis1[None, :], dist, np.inf)

    min0 = dist.argmin(-1)
    min1 = dist.argmin(-2)
    ismin0 = np.zeros(dist.shape, bool)
    ismin1 = np.zeros(dist.shape, bool)
    ismin0[np.arange(len(kp0)), min0] = True
    ismin1[min1, np.arange(len(kp1))] = True
    positive = ismin0 & ismin1 & (dist < pos_th**2)
    negative0 = (dist0.min(-1) > neg_th**2) | ~vis0
    negative1 = (dist1.min(-2) > neg_th**2) | ~vis1
    m0 = np.where(positive.any(-1), min0, -2)
    m1 = np.where(positive.any(-2), min1, -2)
    m0 = np.where(negative0, -1, m0)
    m1 = np.where(negative1, -1, m1)
    return m0, m1


COND_LIMIT = 1e6


def _is_usable(H, s0, s1):
    """Independent usability rule for a homography (not copied from the code).

    Derivation: float32 carries about 6e-8 relative precision, and inverting a
    matrix of condition number ``c`` loses a factor ``c`` of it, so at
    ``c = 1e6`` about 6% of the inverse is noise and a 3 px label threshold is
    meaningless; above that the homography is unusable. The condition number
    must be taken in UNIT coordinates, ``Hn = diag(1/w1, 1/h1, 1) H diag(w0, h0, 1)``,
    because pixel units mix the translation entries (hundreds) with the
    projective entries (1e-3) and make every large-image homography look
    ill-conditioned. numpy's SVD-based ``cond`` is the instrument.
    """
    Hn = np.diag([1.0 / s1[0], 1.0 / s1[1], 1.0]) @ H @ np.diag([s0[0], s0[1], 1.0])
    if not np.all(np.isfinite(Hn)):
        return False
    return bool(np.linalg.matrix_rank(Hn) == 3 and np.linalg.cond(Hn) < COND_LIMIT)


def _oracle(kp0, kp1, m0, m1, H, s0, s1, pos=3.0, neg=3.0):
    """Oracle for one image; returns (matches0, matches1) over the padded arrays.

    Runs :func:`_gf_gt_matches` on the real keypoints only (the reference is
    called on unpadded sets) and scatters back; padded slots are -2. A
    non-invertible homography gives all-dustbin, the repo's documented rule.
    """
    M, N = len(kp0), len(kp1)
    H = H.astype(np.float64)
    out0, out1 = np.full(M, -2), np.full(N, -2)
    r0, r1 = np.flatnonzero(m0), np.flatnonzero(m1)
    ok = _is_usable(H, s0, s1)
    if not ok:
        out0[r0], out1[r1] = -1, -1
        return out0, out1
    if len(r0) == 0 or len(r1) == 0:
        out0[r0], out1[r1] = -1, -1
        return out0, out1
    Hi = np.linalg.inv(H)

    def inside(p, T, size):
        q = np.concatenate([p, np.ones((len(p), 1))], -1) @ T.T
        w = q[:, 2]
        xy = q[:, :2] / w[:, None]
        return (w > 1e-6) & (xy[:, 0] >= 0) & (xy[:, 0] < size[0]) & (xy[:, 1] >= 0) & (xy[:, 1] < size[1])

    k0 = kp0[r0].astype(np.float64)
    k1 = kp1[r1].astype(np.float64)
    g0, g1 = _gf_gt_matches(
        k0, k1, H, pos, neg, visible0=inside(k0, H, s1), visible1=inside(k1, Hi, s0)
    )
    out0[r0] = np.where(g0 >= 0, r1[np.maximum(g0, 0)], g0)
    out1[r1] = np.where(g1 >= 0, r0[np.maximum(g1, 0)], g1)
    return out0, out1


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
        # kp0[0] -> (220,10) is outside image 1: dustbin. kp1[0] lies outside its
        # own image (detectors never produce that) but warps back inside image 0
        # onto kp0[0] (backward error 0): one-sided rule, so it is ignored, and
        # the pair with the invisible kp0[0] is never positive.
        np.testing.assert_array_equal(g0[0], [-1, 1])
        np.testing.assert_array_equal(g1[0], [-2, 1])

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


class TestOneSidedDustbin:
    """D-016: dustbin is one-sided per image, as in glue-factory."""

    @staticmethod
    def _case():
        # H halves coordinates: forward error is measured in image 1, backward in
        # image 0, so one pair can be forward-near and backward-far.
        H = np.array([[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]]], "float32")
        kp0 = np.array([[[20, 20]]], "float32")  # -> (10, 10) in image 1
        kp1 = np.array([[[12, 10]]], "float32")  # forward error 2, backward 4
        m = np.ones((1, 1), bool)
        return kp0, kp1, m, H

    def test_forward_near_backward_far(self):
        kp0, kp1, m, H = self._case()
        g0, g1 = _run(kp0, kp1, m, m, H, s0=(200.0, 160.0), s1=(100.0, 80.0))
        # pair distance max(2, 4) = 4 > 3: not positive. Image 0 has a candidate
        # within 3 px FORWARD so it is ignored; image 1 has none BACKWARD: dustbin.
        np.testing.assert_array_equal(g0[0], [-2])
        np.testing.assert_array_equal(g1[0], [-1])
        e0, e1 = _oracle(kp0[0], kp1[0], m[0], m[0], H[0], (200.0, 160.0), (100.0, 80.0))
        np.testing.assert_array_equal(g0[0], e0)
        np.testing.assert_array_equal(g1[0], e1)

    def test_backward_near_forward_far(self):
        # swap roles: H doubles coordinates
        H = np.array([[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1]]], "float32")
        kp0 = np.array([[[10, 10]]], "float32")  # -> (20, 20)
        kp1 = np.array([[[24, 20]]], "float32")  # forward error 4, backward 2
        m = np.ones((1, 1), bool)
        g0, g1 = _run(kp0, kp1, m, m, H, s0=(100.0, 80.0), s1=(200.0, 160.0))
        np.testing.assert_array_equal(g0[0], [-1])
        np.testing.assert_array_equal(g1[0], [-2])

    @pytest.mark.parametrize("seed", range(6))
    def test_oracle_agrees_where_rules_differ(self, seed):
        kp0, kp1, m0, m1, H = _random_case(seed + 40)
        g0, g1 = _run(kp0, kp1, m0, m1, H)
        for b in range(len(H)):
            e0, e1 = _oracle(kp0[b], kp1[b], m0[b], m1[b], H[b], SIZE, SIZE)
            np.testing.assert_array_equal(g0[b], e0)
            np.testing.assert_array_equal(g1[b], e1)


class TestSingularityScale:
    """D-020: conditioning is judged in unit coordinates, so image size is irrelevant.

    The pixel-unit rule ``|det H| >= 1e-6 ||H||_F^3`` flagged 7.75% of the real
    generator's pairs at 240 px and 50% at 480 px as singular (every keypoint
    dustbin, no positives).
    """

    @staticmethod
    def _pairs(tmp_path, size, count=64):
        import os

        from train.lightglue.data import list_images, make_pair_dataset

        rng = np.random.RandomState(1)
        for i in range(count):
            img = rng.randint(30, 200, size=(90, 120, 3)).astype(np.uint8)
            (tmp_path / f"img{i}.png").write_bytes(tf.io.encode_png(tf.constant(img)).numpy())
        ds = make_pair_dataset(list_images(str(tmp_path)), size, count, seed=0, shuffle=False,
                               photometric_jitter=False, drop_remainder=False)
        return next(iter(ds))["H0to1"].numpy()

    @staticmethod
    def _positives(H, size):
        rng = np.random.default_rng(0)
        k0 = rng.uniform(0.15 * size, 0.85 * size, (len(H), 64, 2))
        q = np.concatenate([k0, np.ones((len(H), 64, 1))], -1) @ H.transpose(0, 2, 1).astype("float64")
        k1 = q[..., :2] / q[..., 2:]
        ones = np.ones((len(H), 64), bool)
        sz = (float(size), float(size))
        g0, _ = _run(k0.astype("float32"), k1.astype("float32"), ones, ones, H, s0=sz, s1=sz)
        inside = ((k1 >= 0) & (k1 < size)).all(-1)
        return (g0 >= 0).sum(-1), inside.sum(-1)

    @pytest.mark.parametrize("size", [240, 480])
    def test_generator_homographies_keep_their_positives(self, tmp_path, size):
        H = self._pairs(tmp_path, size)
        pixel_rule = np.abs(np.linalg.det(H.astype("float64"))) < 1e-6 * np.linalg.norm(H.astype("float64"), axis=(1, 2)) ** 3
        # the old rule fires on these generator pairs at this size (control: the test can fail)
        assert pixel_rule.sum() >= 1
        pos, inside = self._positives(H, size)
        assert (inside > 20).all()
        assert (pos >= 0.9 * inside).all(), (pos, inside)

    @pytest.mark.parametrize("size", [240, 480])
    def test_known_ordinary_homography_is_not_singular(self, size):
        # det ~1.06, an ordinary mild warp with a translation of about 0.3 of the image
        t = 0.3 * size
        H = np.array([[[1.03, 0.05, t], [-0.04, 1.02, -t * 0.5], [1e-5, 0.0, 1.0]]], "float32")
        _, ok = km.invert_3x3(H, np.full((1, 2), size, "float32"), np.full((1, 2), size, "float32"))
        assert bool(_np(ok)[0])
        assert _is_usable(H[0].astype("float64"), (size, size), (size, size))

    def test_inverse_is_consistent_with_the_normalised_test(self):
        size = np.full((1, 2), 480, "float32")
        H = np.array([[[0.9, 0.1, 150], [-0.1, 1.1, 60], [2e-5, 1e-5, 1.0]]], "float32")
        inv, ok = km.invert_3x3(H, size, size)
        assert bool(_np(ok)[0])
        np.testing.assert_allclose(_np(inv)[0] @ H[0], np.eye(3), atol=1e-4)

    def test_pixel_unit_rule_is_red(self, monkeypatch):
        """Reinjecting the old pixel-unit test must fail the regression above."""
        import inspect

        src = inspect.getsource(km.invert_3x3)
        assert "hn = h" in src
        new = src.replace("normalise = size0 is not None and size1 is not None", "normalise = False")
        assert new != src
        ns = dict(vars(km))
        exec(new, ns)
        monkeypatch.setattr(km, "invert_3x3", ns["invert_3x3"])
        H = np.array([[[1.0, 0.0, 300.0], [0.0, 1.0, 200.0], [0.0, 0.0, 1.0]]], "float32")
        s = np.full((1, 2), 480, "float32")
        _, ok = km.invert_3x3(H, s, s)
        assert not bool(_np(ok)[0])  # identity-like pair declared singular by the old rule


class TestFrozenGlueFactory:
    """Exact label equality with glue-factory's torch function.

    Fixture ``data/glue_factory_homography_labels.npz`` (24 pairs, 128 px):
    keypoints are 128-slot padded SuperPoint detections (a 60-step smoke
    SuperPoint, NMS 4, border 4, threshold 0.005) on COCO val2017 pairs from
    ``train.lightglue.data.make_pair_dataset`` (seed 0, no jitter) as it was BEFORE
    the border-free rewrite of D-017 (the labels do not depend on the generator);
    ``H`` is the forward homography. ``g0``/``g1`` are the outputs of cvg/glue-factory
    ``gt_matches_from_homography(kp0, kp1, H, pos_th=3, neg_th=3)`` run in
    float64 on CPU torch 2.14.1 on the REAL keypoints of each pair (padded
    slots set to -2), source commit bd356aa (2025-07-10), 2026-10-02. No torch
    is needed to run this test. The image-bounds extension of this repo changes
    no label on these pairs.
    """

    @pytest.fixture(scope="class")
    def fx(self):
        import os

        path = os.path.join(
            os.path.dirname(__file__), "data", "glue_factory_homography_labels.npz"
        )
        return np.load(path)

    def test_labels_equal_glue_factory(self, fx):
        size = float(fx["size"])
        g0, g1 = _run(
            fx["k0"], fx["k1"], fx["m0"], fx["m1"], fx["H"].astype("float32"),
            s0=(size, size), s1=(size, size),
        )
        np.testing.assert_array_equal(g0, fx["g0"])
        np.testing.assert_array_equal(g1, fx["g1"])
        assert (fx["g0"] >= 0).sum() > 300 and (fx["g0"] == -1).sum() > 0
        assert (fx["g0"] == -2).sum() > 50  # the ignored class is exercised


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

    def test_two_sided_max_dustbin_is_red(self, monkeypatch):
        """Reinjecting the D-011 rule (dustbin from the max distance) must fail."""
        import inspect

        src = inspect.getsource(km.homography_matches)
        assert "fwd_min" in src and "bwd_min" in src
        new_src = src.replace("fwd_min > neg_threshold", "min0 > neg_threshold").replace(
            "bwd_min > neg_threshold", "min1 > neg_threshold"
        )
        assert new_src != src
        ns = dict(vars(km))
        exec(new_src, ns)
        monkeypatch.setattr(km, "homography_matches", ns["homography_matches"])
        kp0, kp1, m, H = TestOneSidedDustbin._case()
        g0, g1 = _run(kp0, kp1, m, m, H, s0=(200.0, 160.0), s1=(100.0, 80.0))
        assert not (g0[0, 0] == -2 and g1[0, 0] == -1)
        assert not self._agrees_with_oracle()

    def test_forward_h_instead_of_inverse_is_red(self, monkeypatch):
        real = km.invert_3x3
        monkeypatch.setattr(km, "invert_3x3", lambda h, *a: (keras.ops.cast(h, "float32"), real(h, *a)[1]))
        assert not self._agrees_with_oracle()

    def test_outside_image_not_excluded_is_red(self, monkeypatch):
        real = km._project

        def mutant(points, h, size):
            xy, _ = real(points, h, size * 0 + 1e9)
            return xy, keras.ops.ones_like(points[..., 0], dtype="bool")

        monkeypatch.setattr(km, "_project", mutant)
        assert not self._agrees_with_oracle()
