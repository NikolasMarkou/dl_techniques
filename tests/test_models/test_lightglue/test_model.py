"""Test suite for the LightGlue model (model.py), static masked path.

Covers instantiation and early validation, the config round-trip, forward-pass structure
(shapes, dtypes, keys, graph versus eager), ``.keras`` save/load, ``compute_output_shape``
on a fresh instance, gradient flow, and the public package API. Value-level parity against
the numpy oracle is in ``test_parity.py``; the shared-oracle adoption is in
``test_oracle_adoption.py``.
"""

import os

import keras
import numpy as np
import pytest
import tensorflow as tf

import dl_techniques.models.vision.keypoints.lightglue as lightglue_pkg
from dl_techniques.models.vision.keypoints.lightglue.model import (
    LightGlue,
    create_lightglue,
    normalize_keypoints,
)

from .weight_loading import random_inputs

D, H, L = 16, 2, 3
M, N = 12, 9
OUT_KEYS = {
    "log_assignments", "token_confidences0", "token_confidences1",
    "token_logits0", "token_logits1",
    "matches0", "matches1", "matching_scores0", "matching_scores1",
}


def _model(**overrides) -> LightGlue:
    kwargs = dict(input_dim=D, descriptor_dim=D, num_layers=L, num_heads=H)
    kwargs.update(overrides)
    return LightGlue(**kwargs)


def _inputs(batch=2, m=M, n=N, input_dim=D, seed=0, extra=False):
    return random_inputs(np.random.RandomState(seed), batch, m, n, input_dim, extra=extra)


@pytest.fixture
def model() -> LightGlue:
    return _model()


class TestInstantiation:

    def test_defaults_are_the_published_size(self):
        m = LightGlue()
        assert (m.input_dim, m.descriptor_dim, m.num_layers, m.num_heads) == (256, 256, 9, 4)
        assert (m.filter_threshold, m.depth_confidence, m.width_confidence) == (0.1, 0.95, 0.99)
        assert m.input_proj is None, "input_dim == descriptor_dim must use the identity"
        assert len(m.self_blocks) == len(m.cross_blocks) == len(m.assignments) == 9
        assert len(m.confidences) == 8

    def test_input_projection_exists_only_when_widths_differ(self):
        assert _model(input_dim=24).input_proj is not None
        assert _model(input_dim=D).input_proj is None

    def test_factory(self):
        m = create_lightglue(input_dim=D, descriptor_dim=D, num_layers=2, num_heads=H)
        assert isinstance(m, LightGlue) and m.num_layers == 2

    def test_factory_forwards_keyword_arguments(self):
        m = create_lightglue(input_dim=D, descriptor_dim=D, num_layers=2, num_heads=H,
                             filter_threshold=0.3, add_scale_ori=True)
        assert m.filter_threshold == 0.3 and m.add_scale_ori and m.num_pos_features == 4

    @pytest.mark.parametrize("overrides", [
        {"input_dim": 0},
        {"descriptor_dim": 0},
        {"num_layers": 0},
        {"num_heads": 0},
        {"descriptor_dim": 18, "num_heads": 4},        # not divisible
        {"descriptor_dim": 12, "num_heads": 4},        # head_dim 3 is odd
        {"filter_threshold": 1.5},
        {"filter_threshold": -0.1},
        {"depth_confidence": 2.0},
        {"width_confidence": -0.5},
        {"gamma": 0.0},
    ])
    def test_invalid_arguments_raise_value_error(self, overrides):
        with pytest.raises(ValueError):
            _model(**overrides)

    @pytest.mark.parametrize("name", ["depth_confidence", "width_confidence"])
    def test_minus_one_disables_and_is_stored(self, name):
        m = _model(**{name: -1})
        assert getattr(m, name) == -1

    def test_descriptor_width_mismatch_raises_at_build(self):
        m = _model()
        data = _inputs(input_dim=D + 2)
        with pytest.raises(ValueError, match="input_dim"):
            m(data)


class TestConfig:

    def test_config_contains_all_ctor_params(self, model):
        cfg = model.get_config()
        for key in ("input_dim", "descriptor_dim", "num_layers", "num_heads", "filter_threshold",
                    "depth_confidence", "width_confidence", "add_scale_ori", "gamma"):
            assert key in cfg, f"missing ctor param {key!r} in get_config()"

    def test_from_config_reconstructs(self):
        m = _model(input_dim=24, filter_threshold=0.2, depth_confidence=-1, width_confidence=0.5,
                   add_scale_ori=True, gamma=2.0)
        m2 = LightGlue.from_config(m.get_config())
        for key in ("input_dim", "descriptor_dim", "num_layers", "num_heads", "filter_threshold",
                    "depth_confidence", "width_confidence", "add_scale_ori", "gamma"):
            assert getattr(m2, key) == getattr(m, key), key

    def test_autocast_is_off_and_survives_the_round_trip(self, model):
        assert model.autocast is False
        assert LightGlue.from_config(model.get_config()).autocast is False


class TestForward:

    def test_output_structure_and_shapes(self, model):
        out = model(_inputs())
        assert isinstance(out, dict) and set(out) == OUT_KEYS
        assert tuple(out["log_assignments"].shape) == (2, L, M + 1, N + 1)
        assert tuple(out["token_confidences0"].shape) == (2, L - 1, M)
        assert tuple(out["token_confidences1"].shape) == (2, L - 1, N)
        assert tuple(out["token_logits0"].shape) == (2, L - 1, M)
        assert tuple(out["token_logits1"].shape) == (2, L - 1, N)
        assert tuple(out["matches0"].shape) == (2, M) and tuple(out["matches1"].shape) == (2, N)
        assert tuple(out["matching_scores0"].shape) == (2, M)
        assert tuple(out["matching_scores1"].shape) == (2, N)

    def test_confidences_are_the_sigmoid_of_the_logits(self, model):
        out = model(_inputs())
        for side in ("0", "1"):
            z = keras.ops.convert_to_numpy(out["token_logits" + side]).astype("float64")
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(out["token_confidences" + side]),
                1.0 / (1.0 + np.exp(-z)), rtol=0, atol=1e-6)
            assert np.abs(z).max() > 0.0, "vacuous: all-zero logits"

    def test_dtypes(self, model):
        out = model(_inputs())
        assert out["log_assignments"].dtype == tf.float32
        assert out["matches0"].dtype == tf.int32 and out["matches1"].dtype == tf.int32

    def test_values_are_finite_and_in_range(self, model):
        out = {k: keras.ops.convert_to_numpy(v) for k, v in model(_inputs()).items()}
        for key, value in out.items():
            assert np.isfinite(value).all(), key
        assert (out["log_assignments"] <= 1e-5).all(), "log probabilities cannot exceed 0"
        for key in ("token_confidences0", "token_confidences1"):
            assert ((out[key] >= 0) & (out[key] <= 1)).all()
        assert ((out["matches0"] >= -1) & (out["matches0"] < N)).all()
        assert ((out["matches1"] >= -1) & (out["matches1"] < M)).all()

    def test_matches_are_mutual(self):
        m = _model(filter_threshold=0.0)
        out = {k: keras.ops.convert_to_numpy(v) for k, v in m(_inputs()).items()}
        m0, m1 = out["matches0"], out["matches1"]
        assert (m0 >= 0).any(), "vacuous: no matches"
        for b in range(m0.shape[0]):
            for i, j in enumerate(m0[b]):
                if j >= 0:
                    assert m1[b, j] == i

    def test_training_flag_is_inert(self, model):
        data = _inputs()
        a = keras.ops.convert_to_numpy(model(data, training=True)["log_assignments"])
        b = keras.ops.convert_to_numpy(model(data, training=False)["log_assignments"])
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("batch, m, n", [(1, 5, 5), (3, 7, 4), (2, 4, 11)])
    def test_variable_batch_and_point_counts(self, model, batch, m, n):
        out = model(_inputs(batch, m, n, seed=batch))
        assert tuple(out["log_assignments"].shape) == (batch, L, m + 1, n + 1)
        assert tuple(out["matches0"].shape) == (batch, m)

    def test_single_layer_has_empty_confidences(self):
        m = _model(num_layers=1)
        out = m(_inputs())
        assert tuple(out["token_confidences0"].shape) == (2, 0, M)
        assert tuple(out["token_confidences1"].shape) == (2, 0, N)
        assert tuple(out["log_assignments"].shape) == (2, 1, M + 1, N + 1)

    def test_scale_and_orientation_inputs(self):
        m = _model(add_scale_ori=True)
        out = m(_inputs(extra=True))
        assert tuple(out["log_assignments"].shape) == (2, L, M + 1, N + 1)
        assert tuple(m.posenc.kernel.shape) == (4, D // H // 2)
        with pytest.raises(KeyError):
            _model(add_scale_ori=True)(_inputs(extra=False))

    def test_graph_mode_matches_eager(self, model):
        data = {k: tf.constant(v) for k, v in _inputs().items()}
        eager = model(data)
        graph = tf.function(lambda x: model(x))(data)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(graph["log_assignments"]),
            keras.ops.convert_to_numpy(eager["log_assignments"]), rtol=0, atol=2e-3)
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(graph["matches0"]),
            keras.ops.convert_to_numpy(eager["matches0"]))

    def test_graph_mode_with_dynamic_point_counts(self, model):
        spec = {
            "keypoints0": tf.TensorSpec((None, None, 2)), "keypoints1": tf.TensorSpec((None, None, 2)),
            "descriptors0": tf.TensorSpec((None, None, D)), "descriptors1": tf.TensorSpec((None, None, D)),
            "image_size0": tf.TensorSpec((None, 2)), "image_size1": tf.TensorSpec((None, 2)),
            "mask0": tf.TensorSpec((None, None)), "mask1": tf.TensorSpec((None, None)),
        }
        fn = tf.function(lambda x: model(x), input_signature=[spec])
        for batch, m, n in ((2, 6, 5), (1, 9, 3)):
            data = _inputs(batch, m, n)
            data["mask0"] = np.ones((batch, m), "float32")
            data["mask1"] = np.ones((batch, n), "float32")
            out = fn(data)
            assert tuple(out["log_assignments"].shape) == (batch, L, m + 1, n + 1)

    def test_masks_may_be_bool_or_float(self, model):
        data = _inputs()
        mask0 = np.arange(M)[None, :] < 8
        mask1 = np.arange(N)[None, :] < 6
        mask0, mask1 = np.repeat(mask0, 2, 0), np.repeat(mask1, 2, 0)
        a = model({**data, "mask0": mask0, "mask1": mask1})
        b = model({**data, "mask0": mask0.astype("float32"), "mask1": mask1.astype("float32")})
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(a["log_assignments"]), keras.ops.convert_to_numpy(b["log_assignments"]))

    def test_all_ones_mask_equals_no_mask(self, model):
        data = _inputs()
        a = model(data)
        b = model({**data, "mask0": np.ones((2, M), "float32"), "mask1": np.ones((2, N), "float32")})
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(a["log_assignments"]),
            keras.ops.convert_to_numpy(b["log_assignments"]), rtol=0, atol=2e-3)

    def test_inference_only_confidences_are_stored_not_used(self):
        """call() ignores depth/width confidence: they exist for the adaptive path."""
        data = _inputs()
        keras.utils.set_random_seed(0)
        a = _model(depth_confidence=0.95, width_confidence=0.99)
        a(data)
        keras.utils.set_random_seed(0)
        b = _model(depth_confidence=-1, width_confidence=-1)
        b(data)
        b.set_weights(a.get_weights())
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(a(data)["log_assignments"]),
            keras.ops.convert_to_numpy(b(data)["log_assignments"]))


class TestNormalizeKeypoints:

    def test_formula_on_a_non_square_image(self):
        kp = np.array([[[0.0, 0.0], [96.0, 64.0], [48.0, 32.0]]], "float32")
        size = np.array([[96.0, 64.0]], "float32")
        out = keras.ops.convert_to_numpy(normalize_keypoints(kp, size))
        np.testing.assert_allclose(out[0], [[-1.0, -2 / 3], [1.0, 2 / 3], [0.0, 0.0]], rtol=0, atol=1e-6)

    def test_per_image_sizes(self):
        kp = np.full((2, 1, 2), 50.0, "float32")
        size = np.array([[100.0, 100.0], [200.0, 100.0]], "float32")
        out = keras.ops.convert_to_numpy(normalize_keypoints(kp, size))
        np.testing.assert_allclose(out[0, 0], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(out[1, 0], [-0.5, 0.0], atol=1e-6)

    @pytest.mark.parametrize("dtype", ["float16", "float32", "int32"])
    def test_result_is_float32_for_narrower_input_dtypes(self, dtype):
        out = normalize_keypoints(np.ones((1, 2, 2), dtype), np.full((1, 2), 64, dtype))
        assert out.dtype == tf.float32

    def test_result_stays_float64_for_float64_input(self):
        out = normalize_keypoints(np.ones((1, 2, 2), "float64"), np.full((1, 2), 64, "float64"))
        assert out.dtype == tf.float64


class TestSaveLoad:

    def test_keras_roundtrip(self, tmp_path):
        m = _model(input_dim=24, add_scale_ori=True)
        data = _inputs(input_dim=24, extra=True)
        out1 = m(data)
        path = os.path.join(str(tmp_path), "lightglue.keras")
        m.save(path)
        m2 = keras.models.load_model(path)
        assert isinstance(m2, LightGlue)
        out2 = m2(data)
        for key in OUT_KEYS:
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(out1[key]), keras.ops.convert_to_numpy(out2[key]),
                rtol=0, atol=1e-5, err_msg=key)

    def test_roundtrip_restores_every_weight(self, tmp_path):
        m = _model()
        m(_inputs())
        path = os.path.join(str(tmp_path), "w.keras")
        m.save(path)
        m2 = keras.models.load_model(path)
        assert len(m2.weights) == len(m.weights)
        for a, b in zip(m.weights, m2.weights):
            np.testing.assert_array_equal(keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b))


class TestComputeOutputShape:

    def _shapes(self, batch, m, n):
        return {"keypoints0": (batch, m, 2), "keypoints1": (batch, n, 2)}

    def test_dict_shape_on_fresh_instance(self):
        shapes = _model().compute_output_shape(self._shapes(None, None, 7))
        assert set(shapes) == OUT_KEYS
        assert shapes["log_assignments"] == (None, L, None, 8)
        assert shapes["token_confidences0"] == (None, L - 1, None)
        assert shapes["token_confidences1"] == (None, L - 1, 7)
        assert shapes["token_logits0"] == (None, L - 1, None)
        assert shapes["token_logits1"] == (None, L - 1, 7)
        assert shapes["matches1"] == (None, 7)

    def test_matches_actual_call(self):
        m = _model()
        shapes = m.compute_output_shape(self._shapes(2, M, N))
        out = m(_inputs())
        assert set(shapes) == set(out)
        for key in out:
            assert shapes[key] == tuple(out[key].shape), key


class TestGradients:

    def test_all_trainable_vars_get_grads(self, model):
        data = {k: tf.constant(v) for k, v in _inputs().items()}
        with tf.GradientTape() as tape:
            out = model(data, training=True)
            loss = keras.ops.mean(out["log_assignments"]) + keras.ops.mean(out["token_confidences0"])
        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) == len(model.trainable_variables) > 0
        none_vars = [v.path for v, g in zip(model.trainable_variables, grads) if g is None]
        # token_confidences1 is not in this loss, but the confidence head is shared by both
        # images, so every weight is still connected; log_assignments needs no confidence.
        assert not none_vars, f"None gradients for: {none_vars}"


class TestPublicAPI:

    def test_all_exact_membership(self):
        assert set(lightglue_pkg.__all__) == {"LightGlue", "create_lightglue"}

    def test_symbols_importable(self):
        assert lightglue_pkg.LightGlue is LightGlue
        assert lightglue_pkg.create_lightglue is create_lightglue

    def test_registered_name_has_the_family_directories_stripped(self):
        assert keras.saving.get_registered_name(LightGlue) == "dl_techniques.models.lightglue.model>LightGlue"
