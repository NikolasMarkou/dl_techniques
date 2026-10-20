"""The numpy transcription of the official PFT-SR forward checks itself.

``pft_reference_numpy.py`` is the oracle for the equation-level guards in this
directory. These arms pin the facts it rests on, each against an independent
statement of the same fact, so a transcription slip in the oracle shows up
here rather than as a guard that agrees with a wrong port:

- the two sparse gathers equal the dense products when the index is every key;
- the official top-k path keeps exactly ``k`` keys per query and drops the rest;
- the shifted-window mask is zero inside a window that no region boundary
  crosses and -100 across a boundary;
- the relative position index covers every one of the ``(2ws-1)^2`` table rows;
- the mirror pad is ``np.pad(mode='symmetric')`` while the pad fits the image;
- window partition and reverse are inverse.
"""

import numpy as np

from . import pft_reference_numpy as ref


def test_the_sparse_gathers_equal_the_dense_products_on_a_full_index():
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((3, 5, 4)) for _ in range(3))
    full = np.broadcast_to(np.arange(5), (3, 5, 5)).copy()
    np.testing.assert_allclose(ref.smm_qmk(q, k, full), q @ k.transpose(0, 2, 1),
                               rtol=0, atol=1e-12)
    a = rng.random((3, 5, 5))
    np.testing.assert_allclose(ref.smm_amv(a, v, full), a @ v, rtol=0, atol=1e-12)


def test_the_sparse_gathers_read_only_the_indexed_keys():
    """Anti-vacuity for the arm above: a permuted, truncated index reads the
    named keys and no others."""
    rng = np.random.default_rng(1)
    q, k = rng.standard_normal((1, 2, 3)), rng.standard_normal((1, 4, 3))
    idx = np.array([[[3, 1], [0, 2]]])
    out = ref.smm_qmk(q, k, idx)
    assert abs(out[0, 0, 0] - np.dot(q[0, 0], k[0, 3])) < 1e-12
    assert abs(out[0, 1, 1] - np.dot(q[0, 1], k[0, 2])) < 1e-12
    assert abs(out[0, 0, 0] - np.dot(q[0, 0], k[0, 1])) > 1e-3


def test_topk_keeps_k_entries_and_the_largest():
    a = np.random.default_rng(2).random((2, 3, 7))
    vals, idx = ref.topk(a, 3)
    assert idx.shape == (2, 3, 3)
    assert np.all(np.sort(vals, -1) == np.sort(a, -1)[..., -3:])


def test_the_shift_mask_is_zero_inside_one_region_and_minus_100_across():
    m = ref.calculate_mask(8, 12, 4)
    assert m.shape == (6, 16, 16)
    assert np.all(m[0] == 0.0), "the top-left window lies inside one region"
    assert set(np.unique(m[-1])) == {0.0, -100.0}, "the bottom-right window spans 4 regions"
    assert np.all(np.diagonal(m, axis1=1, axis2=2) == 0.0)


def test_the_relative_position_index_covers_the_table():
    for ws in (2, 4, 8):
        rpi = ref.relative_position_index(ws)
        assert rpi.shape == (ws * ws, ws * ws)
        assert set(rpi.reshape(-1)) == set(range((2 * ws - 1) ** 2))
        assert np.all(np.diagonal(rpi) == (ws - 1) * (2 * ws - 1) + (ws - 1))


def test_the_mirror_pad_is_symmetric_padding():
    x = np.random.default_rng(3).random((1, 7, 10, 3))
    got = ref.mirror_pad(x, 4)
    assert got.shape == (1, 8, 12, 3)
    np.testing.assert_array_equal(
        got, np.pad(x, ((0, 0), (0, 1), (0, 2), (0, 0)), mode="symmetric"))


def test_window_partition_and_reverse_are_inverse():
    x = np.random.default_rng(4).random((2, 8, 12, 5))
    w = ref.window_partition(x, 4)
    assert w.shape == (12, 4, 4, 5)
    np.testing.assert_array_equal(ref.window_reverse(w, 4, 8, 12), x)


def test_the_framework_still_rounds_the_two_constants_the_oracle_rounds():
    """If TensorFlow ever computes ``gelu``'s ``sqrt(2)`` or ``LeakyRelu``'s
    slope in float64, the oracle's emulation of the float32 rounding is stale
    and this arm says so (the guards would then fail at ~1e-9)."""
    import keras
    x = np.linspace(-4.0, 4.0, 4001)
    got = np.asarray(keras.ops.gelu(keras.ops.convert_to_tensor(x, "float64"),
                                    approximate=False))
    np.testing.assert_allclose(got, ref.gelu(x), rtol=0, atol=1e-15)
    assert np.abs(got - ref.gelu(x, sqrt2=np.sqrt(2.0))).max() > 1e-10
    neg = np.asarray(keras.layers.LeakyReLU(negative_slope=0.01, dtype="float64")(
        np.array([-1.0])))
    assert neg[0] == -ref.tf_slope(0.01) and neg[0] != -0.01
