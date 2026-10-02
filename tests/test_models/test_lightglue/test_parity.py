"""Full-model parity of ``LightGlue.call`` against the explicit-loop numpy oracle.

The oracle (``tests/lightglue_reference_numpy.py``) transcribes the official reference
forward with early exit and pruning off. It runs on the REAL keypoints only (the reference
removes padding before the assignment), so the padded comparisons below read the model's
real rows and columns plus its dustbin row and column. Weights are random torch-layout
arrays loaded through the test-only transpose mapping in ``weight_loading.py``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.keypoints.lightglue import model as lightglue_model
from dl_techniques.models.vision.keypoints.lightglue.model import LightGlue
from tests import lightglue_reference_numpy as ref

from .weight_loading import load_torch_weights, oracle_forward, random_inputs

# TF32 matmul would turn the float32 comparison into a ~1e-3 one; run in the TF32-off regime.
pytestmark = pytest.mark.usefixtures("tf32_disabled")

_EPS32 = float(np.finfo(np.float32).eps)
D, H, L = 32, 2, 2
M = N = 16
THRESHOLD = 0.0


def _tol(expected):
    """float32 bound for a chain of ``L`` layers, derived from the arithmetic.

    Per layer: self block (about 8 chained rounded stages), cross block (about 8) and the
    assignment (about 4), so about 20 stages; ``L`` layers chain the descriptors. Each stage
    contributes at most a few eps32 of relative error (fan-in 16 to 64, LayerNorm can double
    it). The bound is ``4 * stages * L * eps32 * (1 + max|expected|)``; the block tests
    use 64 eps32 for one block (8 stages), this is the same per-stage rate.
    """
    return 4 * 20 * L * _EPS32 * (1.0 + float(np.abs(expected).max()))


def _build(seed=0, input_dim=D, add_scale_ori=False, **kw):
    rng = np.random.RandomState(seed)
    w = ref.random_weights(rng, D, H, L, m=4 if add_scale_ori else 2, input_dim=input_dim)
    model = LightGlue(input_dim=input_dim, descriptor_dim=D, num_layers=L, num_heads=H,
                      filter_threshold=THRESHOLD, add_scale_ori=add_scale_ori, **kw)
    model.build(None)
    load_torch_weights(model, w)
    return model, w


def _real_index(real, total):
    return np.array(list(range(real)) + [total])


class TestUnpaddedParity:

    @pytest.mark.parametrize("input_dim", [D, 24])
    def test_log_assignments_confidences_and_matches(self, input_dim):
        model, w = _build(input_dim=input_dim)
        data = random_inputs(np.random.RandomState(1), 2, M, N, input_dim)
        out = model(data)
        oracle = oracle_forward(w, data, L, H)

        got_la = keras.ops.convert_to_numpy(out["log_assignments"])
        exp_la = np.stack([np.stack(o["log_assignments"]) for o in oracle])
        assert got_la.shape == exp_la.shape == (2, L, M + 1, N + 1)
        np.testing.assert_allclose(got_la, exp_la, rtol=0, atol=_tol(exp_la))

        for side, key in (("0", "confidences0"), ("1", "confidences1")):
            got = keras.ops.convert_to_numpy(out["token_confidences" + side])
            exp = np.stack([np.stack(o[key]) for o in oracle])
            np.testing.assert_allclose(got, exp, rtol=0, atol=_tol(exp))

        m0 = keras.ops.convert_to_numpy(out["matches0"])
        m1 = keras.ops.convert_to_numpy(out["matches1"])
        for b, o in enumerate(oracle):
            em0, em1, _, _ = ref.filter_matches(o["log_assignments"][-1], THRESHOLD)
            np.testing.assert_array_equal(m0[b], em0)
            np.testing.assert_array_equal(m1[b], em1)
        assert (m0 >= 0).sum() > 0, "vacuous: no match to compare"

    def test_scale_and_orientation_features(self):
        model, w = _build(add_scale_ori=True)
        data = random_inputs(np.random.RandomState(2), 2, M, N, D, extra=True)
        out = model(data)
        oracle = oracle_forward(w, data, L, H, extra=True)
        got = keras.ops.convert_to_numpy(out["log_assignments"])
        exp = np.stack([np.stack(o["log_assignments"]) for o in oracle])
        np.testing.assert_allclose(got, exp, rtol=0, atol=_tol(exp))


class TestPaddedParity:

    def test_padded_batch_matches_oracle_on_real_keypoints(self):
        model, w = _build()
        k0, k1 = 11, 13
        rng = np.random.RandomState(3)
        data = random_inputs(rng, 2, M, N, D)
        mask0 = np.zeros((2, M), "float32")
        mask1 = np.zeros((2, N), "float32")
        mask0[:, :k0] = 1
        mask1[:, :k1] = 1
        out = model({**data, "mask0": mask0, "mask1": mask1})
        real = {"keypoints0": data["keypoints0"][:, :k0], "keypoints1": data["keypoints1"][:, :k1],
                "descriptors0": data["descriptors0"][:, :k0], "descriptors1": data["descriptors1"][:, :k1],
                "image_size0": data["image_size0"], "image_size1": data["image_size1"]}
        oracle = oracle_forward(w, real, L, H)

        got = keras.ops.convert_to_numpy(out["log_assignments"])
        exp = np.stack([np.stack(o["log_assignments"]) for o in oracle])
        rows, cols = _real_index(k0, M), _real_index(k1, N)
        sel = got[:, :, rows][:, :, :, cols]
        assert sel.shape == exp.shape
        np.testing.assert_allclose(sel, exp, rtol=0, atol=_tol(exp))
        # padded rows and columns carry no mass
        assert np.all(got[:, :, k0:M, :] == 0) and np.all(got[:, :, :, k1:N] == 0)

        m0 = keras.ops.convert_to_numpy(out["matches0"])
        m1 = keras.ops.convert_to_numpy(out["matches1"])
        for b, o in enumerate(oracle):
            em0, em1, _, _ = ref.filter_matches(o["log_assignments"][-1], THRESHOLD)
            np.testing.assert_array_equal(m0[b, :k0], em0)
            np.testing.assert_array_equal(m1[b, :k1], em1)
            assert np.all(m0[b, k0:] == -1) and np.all(m1[b, k1:] == -1)
        assert (m0 >= 0).sum() > 0

    def test_padding_content_and_amount_do_not_change_real_outputs(self):
        """End-to-end padding invariance: junk in the padded slots, and more padding."""
        model, _ = _build(seed=4)
        k0, k1 = 9, 12
        rng = np.random.RandomState(5)
        base = random_inputs(rng, 2, M, N, D)
        mask0 = np.zeros((2, M), "float32")
        mask1 = np.zeros((2, N), "float32")
        mask0[:, :k0] = 1
        mask1[:, :k1] = 1

        junk = {k: v.copy() for k, v in base.items()}
        junk["descriptors0"][:, k0:] = rng.normal(size=junk["descriptors0"][:, k0:].shape) * 50
        junk["descriptors1"][:, k1:] = rng.normal(size=junk["descriptors1"][:, k1:].shape) * 50
        junk["keypoints0"][:, k0:] = rng.uniform(-500, 500, junk["keypoints0"][:, k0:].shape)
        junk["keypoints1"][:, k1:] = rng.uniform(-500, 500, junk["keypoints1"][:, k1:].shape)

        wide = {k: v for k, v in base.items()}
        extra0, extra1 = 8, 5
        wide["keypoints0"] = np.pad(base["keypoints0"], ((0, 0), (0, extra0), (0, 0)))
        wide["descriptors0"] = np.pad(base["descriptors0"], ((0, 0), (0, extra0), (0, 0)))
        wide["keypoints1"] = np.pad(base["keypoints1"], ((0, 0), (0, extra1), (0, 0)))
        wide["descriptors1"] = np.pad(base["descriptors1"], ((0, 0), (0, extra1), (0, 0)))
        wmask0 = np.pad(mask0, ((0, 0), (0, extra0)))
        wmask1 = np.pad(mask1, ((0, 0), (0, extra1)))

        ref_out = model({**base, "mask0": mask0, "mask1": mask1})
        junk_out = model({**junk, "mask0": mask0, "mask1": mask1})
        wide_out = model({**wide, "mask0": wmask0, "mask1": wmask1})

        a = keras.ops.convert_to_numpy(ref_out["log_assignments"])
        b = keras.ops.convert_to_numpy(junk_out["log_assignments"])
        c = keras.ops.convert_to_numpy(wide_out["log_assignments"])
        tol = _tol(a)
        np.testing.assert_allclose(b, a, rtol=0, atol=tol)
        rows, cols = _real_index(k0, M), _real_index(k1, N)
        wrows, wcols = _real_index(k0, M + extra0), _real_index(k1, N + extra1)
        # more padding: compare the real rows/columns plus the dustbin
        np.testing.assert_allclose(
            c[:, :, wrows][:, :, :, wcols], a[:, :, rows][:, :, :, cols], rtol=0, atol=tol)
        for key in ("token_confidences0", "token_confidences1"):
            n = k0 if key.endswith("0") else k1
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(junk_out[key])[:, :, :n],
                keras.ops.convert_to_numpy(ref_out[key])[:, :, :n], rtol=0, atol=tol)
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(junk_out["matches0"])[:, :k0],
            keras.ops.convert_to_numpy(ref_out["matches0"])[:, :k0])

    def test_one_sided_mask_equals_full_mask_with_ones(self):
        model, _ = _build(seed=6)
        data = random_inputs(np.random.RandomState(7), 2, M, N, D)
        mask0 = np.ones((2, M), "float32")
        mask0[:, 12:] = 0
        a = model({**data, "mask0": mask0})
        b = model({**data, "mask0": mask0, "mask1": np.ones((2, N), "float32")})
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(a["log_assignments"]),
                                      keras.ops.convert_to_numpy(b["log_assignments"]))


# ---------------------------------------------------------------------
# mutation RED proofs: the parity checks above can fail
# ---------------------------------------------------------------------


def _unpadded_parity_holds(model, w):
    data = random_inputs(np.random.RandomState(1), 2, M, N, D)
    got = keras.ops.convert_to_numpy(model(data)["log_assignments"])
    exp = np.stack([np.stack(o["log_assignments"]) for o in oracle_forward(w, data, L, H)])
    np.testing.assert_allclose(got, exp, rtol=0, atol=_tol(exp))


def _padded_parity_holds(model, w):
    k0, k1 = 11, 13
    data = random_inputs(np.random.RandomState(3), 2, M, N, D)
    mask0 = np.zeros((2, M), "float32")
    mask1 = np.zeros((2, N), "float32")
    mask0[:, :k0] = 1
    mask1[:, :k1] = 1
    got = keras.ops.convert_to_numpy(model({**data, "mask0": mask0, "mask1": mask1})["log_assignments"])
    real = {**data, "keypoints0": data["keypoints0"][:, :k0], "keypoints1": data["keypoints1"][:, :k1],
            "descriptors0": data["descriptors0"][:, :k0], "descriptors1": data["descriptors1"][:, :k1]}
    exp = np.stack([np.stack(o["log_assignments"]) for o in oracle_forward(w, real, L, H)])
    sel = got[:, :, _real_index(k0, M)][:, :, :, _real_index(k1, N)]
    np.testing.assert_allclose(sel, exp, rtol=0, atol=_tol(exp))


class TestParityMutations:
    """Each mutant breaks one convention; the control shows the same helper passes unmutated."""

    def test_controls_pass(self):
        model, w = _build()
        _unpadded_parity_holds(model, w)
        _padded_parity_holds(model, w)

    def test_per_axis_normalisation_is_caught(self, monkeypatch):
        """Dividing each axis by its own half-size instead of max(size)/2 (images are not square)."""
        def per_axis(keypoints, image_size):
            size = keras.ops.cast(image_size, "float32")[:, None, :]
            return (keras.ops.cast(keypoints, "float32") - size / 2.0) / (size / 2.0)

        model, w = _build()
        monkeypatch.setattr(lightglue_model, "normalize_keypoints", per_axis)
        with pytest.raises(AssertionError):
            _unpadded_parity_holds(model, w)

    def test_image_one_using_image_zero_positions_is_caught(self, monkeypatch):
        original = LightGlue._positions

        def wrong(self, inputs, side):
            return original(self, inputs, "0")

        model, w = _build()
        monkeypatch.setattr(LightGlue, "_positions", wrong)
        with pytest.raises(AssertionError):
            _unpadded_parity_holds(model, w)

    def test_ignoring_the_masks_is_caught(self):
        model, w = _build()
        original = model.call

        def unmasked(inputs, training=None):
            return original({k: v for k, v in inputs.items() if not k.startswith("mask")})

        model.call = unmasked
        try:
            with pytest.raises(AssertionError):
                _padded_parity_holds(model, w)
        finally:
            del model.call
