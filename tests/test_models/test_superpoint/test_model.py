"""
Test suite for the SuperPoint model (model.py).

Covers instantiation/variants, config round-trip (incl. object regularizer),
forward-pass shapes + descriptor L2-norm, ``.keras`` save/load, dict
``compute_output_shape`` on a fresh instance, gradient flow over both heads,
and the public package API.
"""

import os

import keras
import numpy as np
import pytest
import tensorflow as tf

import dl_techniques.models.superpoint as superpoint_pkg
from dl_techniques.models.superpoint.model import SuperPoint, create_superpoint


# ---------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------

# Small spatial config: 64x64x1 keeps H/8 == 8 (meaningful 8x8 detector grid)
# while staying CPU/GPU cheap.
_INPUT_SHAPE = (64, 64, 1)
_GRID = _INPUT_SHAPE[0] // 8  # 8
_DESC_DIM = 256
_DETECTOR_CH = 65


def _tiny_model(**overrides) -> SuperPoint:
    """Smallest tiny-variant SuperPoint for fast tests."""
    return SuperPoint.from_variant("tiny", input_shape=_INPUT_SHAPE, **overrides)


def _sample_input(batch: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(
        batch, _INPUT_SHAPE[0], _INPUT_SHAPE[1], _INPUT_SHAPE[2]
    ).astype("float32")


@pytest.fixture
def model() -> SuperPoint:
    return _tiny_model()


@pytest.fixture
def sample_input() -> np.ndarray:
    return _sample_input(batch=2, seed=0)


# ---------------------------------------------------------------------
# Instantiation / variants
# ---------------------------------------------------------------------

class TestInstantiation:

    @pytest.mark.parametrize("variant", ["tiny", "base", "large"])
    def test_construct_variant(self, variant):
        m = SuperPoint.from_variant(variant, input_shape=_INPUT_SHAPE)
        assert isinstance(m, SuperPoint)
        assert m.dims == SuperPoint.MODEL_VARIANTS[variant]["dims"]
        assert m.depths == SuperPoint.MODEL_VARIANTS[variant]["depths"]

    def test_create_superpoint_factory(self):
        m = create_superpoint("tiny", input_shape=_INPUT_SHAPE)
        assert isinstance(m, SuperPoint)

    def test_bogus_variant_raises(self):
        with pytest.raises(ValueError):
            create_superpoint("bogus", input_shape=_INPUT_SHAPE)

    def test_bogus_variant_from_variant_raises(self):
        with pytest.raises(ValueError):
            SuperPoint.from_variant("bogus", input_shape=_INPUT_SHAPE)

    def test_mismatched_depths_dims_raises(self):
        with pytest.raises(ValueError):
            SuperPoint(depths=[3, 3], dims=[96, 192, 384], input_shape=_INPUT_SHAPE)


# ---------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------

class TestConfig:

    def test_config_contains_all_ctor_params(self):
        m = _tiny_model()
        cfg = m.get_config()
        for key in (
            "depths", "dims", "input_shape", "descriptor_dim",
            "drop_path_rate", "kernel_size", "activation", "use_bias",
            "kernel_regularizer",
        ):
            assert key in cfg, f"missing ctor param '{key}' in get_config()"

    def test_from_config_reconstructs(self):
        m = _tiny_model()
        cfg = m.get_config()
        m2 = SuperPoint.from_config(m.get_config())
        assert m2.depths == m.depths
        assert m2.dims == m.dims
        assert tuple(m2._input_shape) == tuple(m._input_shape)
        assert m2.descriptor_dim == m.descriptor_dim
        assert m2.drop_path_rate == m.drop_path_rate
        assert m2.kernel_size == m.kernel_size
        assert m2.activation == m.activation
        assert m2.use_bias == m.use_bias

    def test_object_regularizer_survives_roundtrip(self):
        reg = keras.regularizers.L2(1e-4)
        m = SuperPoint.from_variant(
            "tiny", input_shape=_INPUT_SHAPE, kernel_regularizer=reg
        )
        cfg = m.get_config()
        # Serialized form must be a dict (not a live object) in the config.
        assert isinstance(cfg["kernel_regularizer"], dict)

        m2 = SuperPoint.from_config(m.get_config())
        assert isinstance(m2.kernel_regularizer, keras.regularizers.L2)
        np.testing.assert_allclose(
            float(m2.kernel_regularizer.l2), float(reg.l2), atol=1e-8
        )


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------

class TestForward:

    def test_output_dict_keys(self, model, sample_input):
        out = model(sample_input)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"keypoints", "descriptors"}

    def test_output_shapes(self, model, sample_input):
        out = model(sample_input)
        assert tuple(out["keypoints"].shape) == (2, _GRID, _GRID, _DETECTOR_CH)
        assert tuple(out["descriptors"].shape) == (
            2, _INPUT_SHAPE[0], _INPUT_SHAPE[1], _DESC_DIM
        )

    def test_descriptors_unit_l2(self, model, sample_input):
        out = model(sample_input)
        desc = keras.ops.convert_to_numpy(out["descriptors"])
        norms = np.linalg.norm(desc, axis=-1)
        np.testing.assert_allclose(
            norms, np.ones_like(norms), atol=1e-5,
            err_msg="descriptors are not unit-L2 along the channel axis",
        )

    def test_training_true_and_false_run(self, model, sample_input):
        out_train = model(sample_input, training=True)
        out_eval = model(sample_input, training=False)
        assert set(out_train.keys()) == {"keypoints", "descriptors"}
        assert set(out_eval.keys()) == {"keypoints", "descriptors"}

    @pytest.mark.parametrize("batch", [1, 3])
    def test_variable_batch_size(self, model, batch):
        x = _sample_input(batch=batch, seed=batch)
        out = model(x)
        assert tuple(out["keypoints"].shape) == (batch, _GRID, _GRID, _DETECTOR_CH)
        assert tuple(out["descriptors"].shape) == (
            batch, _INPUT_SHAPE[0], _INPUT_SHAPE[1], _DESC_DIM
        )


# ---------------------------------------------------------------------
# .keras save/load round-trip
# ---------------------------------------------------------------------

class TestSaveLoad:

    def test_keras_roundtrip_both_heads(self, tmp_path):
        m = _tiny_model()
        x = _sample_input(batch=2, seed=11)
        out1 = m(x)
        kp1 = keras.ops.convert_to_numpy(out1["keypoints"])
        desc1 = keras.ops.convert_to_numpy(out1["descriptors"])

        path = os.path.join(str(tmp_path), "superpoint.keras")
        m.save(path)
        m2 = keras.models.load_model(path)

        out2 = m2(x)
        kp2 = keras.ops.convert_to_numpy(out2["keypoints"])
        desc2 = keras.ops.convert_to_numpy(out2["descriptors"])

        np.testing.assert_allclose(
            kp1, kp2, atol=1e-4,
            err_msg="keypoints head differs after .keras round-trip",
        )
        np.testing.assert_allclose(
            desc1, desc2, atol=1e-4,
            err_msg="descriptors head differs after .keras round-trip",
        )


# ---------------------------------------------------------------------
# compute_output_shape (fresh, unbuilt instance)
# ---------------------------------------------------------------------

class TestComputeOutputShape:

    def test_dict_shape_on_fresh_instance(self):
        # Fresh instance, no forward pass / build yet.
        m = _tiny_model()
        shapes = m.compute_output_shape((None, _INPUT_SHAPE[0], _INPUT_SHAPE[1], 1))
        assert set(shapes.keys()) == {"keypoints", "descriptors"}
        assert shapes["keypoints"] == (None, _GRID, _GRID, _DETECTOR_CH)
        assert shapes["descriptors"] == (
            None, _INPUT_SHAPE[0], _INPUT_SHAPE[1], _DESC_DIM
        )

    def test_matches_actual_call(self):
        m = _tiny_model()
        shapes = m.compute_output_shape((2, _INPUT_SHAPE[0], _INPUT_SHAPE[1], 1))
        out = m(_sample_input(batch=2, seed=3))
        assert shapes["keypoints"] == tuple(out["keypoints"].shape)
        assert shapes["descriptors"] == tuple(out["descriptors"].shape)


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------

class TestGradients:

    def test_all_trainable_vars_get_grads(self, model, sample_input):
        x = tf.convert_to_tensor(sample_input)
        with tf.GradientTape() as tape:
            out = model(x, training=True)
            loss = (
                keras.ops.mean(out["keypoints"])
                + keras.ops.mean(out["descriptors"])
            )
        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) == len(model.trainable_variables)
        assert len(grads) > 0
        none_vars = [
            v.name for v, g in zip(model.trainable_variables, grads) if g is None
        ]
        assert not none_vars, f"None gradients for: {none_vars}"


# ---------------------------------------------------------------------
# Variants (build + forward; descriptor/detector channel invariants)
# ---------------------------------------------------------------------

class TestVariants:

    @pytest.mark.parametrize("variant", ["tiny", "base", "large"])
    def test_variant_builds_and_forwards(self, variant):
        m = SuperPoint.from_variant(variant, input_shape=_INPUT_SHAPE)
        out = m(_sample_input(batch=2, seed=5))
        assert tuple(out["keypoints"].shape) == (2, _GRID, _GRID, _DETECTOR_CH)
        # descriptor channel dim fixed at 256 across all variants.
        assert out["descriptors"].shape[-1] == _DESC_DIM
        assert out["keypoints"].shape[-1] == _DETECTOR_CH


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

class TestPublicAPI:

    def test_all_exact_membership(self):
        assert set(superpoint_pkg.__all__) == {"SuperPoint", "create_superpoint"}

    def test_symbols_importable(self):
        assert superpoint_pkg.SuperPoint is SuperPoint
        assert superpoint_pkg.create_superpoint is create_superpoint
