"""Tests for KANInitializer (Rigas et al. 2026 variance-controlled KAN init).

Covers all 9 Success Criteria of plan_2026-06-12_6cc7c378:
    1. Construction / validation
    2. Shape dispatch (2D residual / 3D spline) + finite
    3. N-consistency with KANLinear (N = grid_size + spline_order)
    4. Seed reproducibility
    5. Three schemes + power_law magnitude ordering (residual std > spline std)
    6. get_config / from_config round-trip
    7. KANLinear integration (forward + gradient flow)
    8. .keras model save/load round-trip
    9. Backward-compat (default base_scaler all-ones; factory registers param)
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.initializers import KANInitializer, create_kan_initializers
from dl_techniques.layers.ffn.kan_linear import KANLinear
from dl_techniques.layers.ffn.factory import FFN_REGISTRY


class TestKANInitializerConstruction:
    """SC1: construction defaults / custom params / validation."""

    def test_initialization_defaults(self):
        init = KANInitializer()
        assert init.scheme == "power_law"
        assert init.target == "residual"
        assert init.grid_size == 5
        assert init.spline_order == 3
        assert init.alpha == 0.25
        assert init.beta == 1.75

    def test_initialization_custom(self):
        init = KANInitializer(
            scheme="glorot_inspired",
            target="spline",
            grid_size=8,
            spline_order=4,
            alpha=0.3,
            beta=2.0,
            baseline_noise=0.05,
            seed=42,
        )
        assert init.scheme == "glorot_inspired"
        assert init.target == "spline"
        assert init.grid_size == 8
        assert init.spline_order == 4
        assert init.alpha == 0.3
        assert init.beta == 2.0
        assert init.baseline_noise == 0.05
        assert init.seed == 42

    def test_invalid_scheme(self):
        with pytest.raises(ValueError):
            KANInitializer(scheme="not_a_scheme")

    def test_invalid_target(self):
        with pytest.raises(ValueError):
            KANInitializer(target="not_a_target")

    def test_invalid_grid_size(self):
        with pytest.raises(ValueError):
            KANInitializer(grid_size=0)

    def test_invalid_spline_order(self):
        with pytest.raises(ValueError):
            KANInitializer(spline_order=-1)


class TestKANInitializerShapeDispatch:
    """SC2: shape dispatch + finite output + dimensionality guards."""

    def test_shape_residual_2d(self):
        init = KANInitializer(target="residual", seed=0)
        w = init((4, 8))
        assert tuple(w.shape) == (4, 8)
        w_np = ops.convert_to_numpy(w)
        assert not np.any(np.isnan(w_np))
        assert not np.any(np.isinf(w_np))
        assert not ops.any(ops.isnan(w))
        assert not ops.any(ops.isinf(w))

    def test_shape_spline_3d(self):
        init = KANInitializer(target="spline", seed=0)
        w = init((4, 8, 8))
        assert tuple(w.shape) == (4, 8, 8)
        assert not ops.any(ops.isnan(w))
        assert not ops.any(ops.isinf(w))

    def test_residual_rejects_3d(self):
        init = KANInitializer(target="residual", seed=0)
        with pytest.raises(ValueError):
            init((4, 8, 8))

    def test_spline_rejects_2d(self):
        init = KANInitializer(target="spline", seed=0)
        with pytest.raises(ValueError):
            init((4, 8))


class TestKANInitializerNConsistency:
    """SC3: N == grid_size + spline_order, matched to KANLinear.spline_weight."""

    def test_n_matches_kan_linear(self):
        grid_size, spline_order = 5, 3
        expected_n = grid_size + spline_order  # == 8

        # A real KANLinear: its spline_weight last-dim must equal N.
        layer = KANLinear(
            features=8, grid_size=grid_size, spline_order=spline_order
        )
        layer.build((None, 4))
        spline_shape = tuple(layer.spline_weight.shape)
        assert spline_shape[-1] == expected_n
        assert spline_shape == (4, 8, expected_n)

        # The spline-target initializer, called on that exact shape, returns a
        # matching shape (N derived directly from shape[-1] == grid_size+order).
        _, spline_init = create_kan_initializers(
            grid_size=grid_size, spline_order=spline_order, seed=0
        )
        out = spline_init(spline_shape)
        assert tuple(out.shape) == spline_shape
        assert tuple(out.shape)[-1] == grid_size + spline_order


class TestKANInitializerSeed:
    """SC4: same seed -> identical; different seed -> different (both targets)."""

    @pytest.mark.parametrize(
        "target,shape",
        [("residual", (32, 16)), ("spline", (32, 16, 8))],
    )
    def test_seed_reproducibility(self, target, shape):
        a = ops.convert_to_numpy(KANInitializer(target=target, seed=7)(shape))
        b = ops.convert_to_numpy(KANInitializer(target=target, seed=7)(shape))
        assert np.allclose(a, b, atol=1e-7)

    @pytest.mark.parametrize(
        "target,shape",
        [("residual", (32, 16)), ("spline", (32, 16, 8))],
    )
    def test_different_seeds_differ(self, target, shape):
        a = ops.convert_to_numpy(KANInitializer(target=target, seed=1)(shape))
        b = ops.convert_to_numpy(KANInitializer(target=target, seed=2)(shape))
        assert not np.allclose(a, b)


class TestKANInitializerSchemes:
    """SC5: all schemes finite; power_law residual std > spline std."""

    @pytest.mark.parametrize(
        "scheme", ["power_law", "glorot_inspired", "baseline"]
    )
    def test_all_schemes_finite(self, scheme):
        res_w = ops.convert_to_numpy(
            KANInitializer(scheme=scheme, target="residual", seed=0)((64, 64))
        )
        spl_w = ops.convert_to_numpy(
            KANInitializer(scheme=scheme, target="spline", seed=0)((64, 64, 8))
        )
        assert np.all(np.isfinite(res_w))
        assert np.all(np.isfinite(spl_w))

    def test_power_law_magnitude_ordering(self):
        # beta (1.75) > alpha (0.25) -> residual std > spline std.
        res_w = ops.convert_to_numpy(
            KANInitializer(scheme="power_law", target="residual", seed=0)(
                (64, 64)
            )
        )
        spl_w = ops.convert_to_numpy(
            KANInitializer(scheme="power_law", target="spline", seed=0)(
                (64, 64, 8)
            )
        )
        assert np.std(res_w) > np.std(spl_w)


class TestKANInitializerSerialization:
    """SC6: get_config returns all 8 keys; from_config reproduces output."""

    def test_serialization_roundtrip(self):
        init = KANInitializer(
            scheme="glorot_inspired",
            target="spline",
            grid_size=6,
            spline_order=2,
            alpha=0.3,
            beta=1.9,
            baseline_noise=0.2,
            seed=11,
        )
        cfg = init.get_config()
        expected_keys = {
            "scheme",
            "target",
            "grid_size",
            "spline_order",
            "alpha",
            "beta",
            "baseline_noise",
            "seed",
        }
        assert expected_keys.issubset(set(cfg.keys()))

        restored = KANInitializer.from_config(cfg)
        assert restored.scheme == init.scheme
        assert restored.target == init.target
        assert restored.grid_size == init.grid_size
        assert restored.spline_order == init.spline_order
        assert restored.alpha == init.alpha
        assert restored.beta == init.beta
        assert restored.baseline_noise == init.baseline_noise
        assert restored.seed == init.seed

        np.testing.assert_allclose(
            ops.convert_to_numpy(init((16, 8, 8))),
            ops.convert_to_numpy(restored((16, 8, 8))),
            atol=1e-7,
        )

    def test_keras_serialize_deserialize(self):
        init = KANInitializer(scheme="power_law", target="residual", seed=3)
        restored = keras.initializers.deserialize(
            keras.initializers.serialize(init)
        )
        assert isinstance(restored, KANInitializer)
        np.testing.assert_allclose(
            ops.convert_to_numpy(init((8, 8))),
            ops.convert_to_numpy(restored((8, 8))),
            atol=1e-7,
        )


class TestKANInitializerKANLinearIntegration:
    """SC7: KANLinear wired with both initializers; forward + gradients."""

    def test_kan_linear_integration_forward_grad(self):
        res_init, spline_init = create_kan_initializers(
            grid_size=5, spline_order=3, scheme="power_law", seed=0
        )
        layer = KANLinear(
            features=8,
            grid_size=5,
            spline_order=3,
            kernel_initializer=spline_init,
            base_scaler_initializer=res_init,
        )

        x = keras.Variable(keras.random.normal((2, 4)))
        with tf.GradientTape() as tape:
            out = layer(x)
            loss = ops.mean(ops.square(out))

        out_np = ops.convert_to_numpy(out)
        assert np.all(np.isfinite(out_np))

        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == 3
        assert all(g is not None for g in grads)
        for g in grads:
            g_np = ops.convert_to_numpy(g)
            assert not np.any(np.isnan(g_np)), "gradient contains NaN"
            assert np.any(g_np != 0.0), "gradient is all-zero"


class TestKANInitializerSaveLoad:
    """SC8: .keras model save/load round-trip + initializer identity."""

    def test_model_save_load_with_kan_initializer(self, tmp_path):
        res_init, spline_init = create_kan_initializers(
            grid_size=5, spline_order=3, scheme="power_law", seed=5
        )
        inputs = keras.Input(shape=(4,))
        outputs = KANLinear(
            features=8,
            grid_size=5,
            spline_order=3,
            kernel_initializer=spline_init,
            base_scaler_initializer=res_init,
        )(inputs)
        model = keras.Model(inputs, outputs)

        x = np.random.randn(3, 4).astype("float32")
        before = ops.convert_to_numpy(model(x))

        path = os.path.join(str(tmp_path), "kan_model.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = ops.convert_to_numpy(loaded(x))

        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)

        loaded_layer = None
        for lyr in loaded.layers:
            if isinstance(lyr, KANLinear):
                loaded_layer = lyr
                break
        assert loaded_layer is not None

        assert isinstance(loaded_layer.base_scaler_initializer, KANInitializer)
        assert isinstance(loaded_layer.kernel_initializer, KANInitializer)
        assert loaded_layer.base_scaler_initializer.seed == 5
        assert loaded_layer.kernel_initializer.seed == 5
        assert loaded_layer.base_scaler_initializer.target == "residual"
        assert loaded_layer.kernel_initializer.target == "spline"


class TestKANInitializerBackwardCompat:
    """SC9: default KANLinear base_scaler all-ones; factory registers param."""

    def test_default_base_scaler_is_ones(self):
        layer = KANLinear(features=8)
        layer.build((None, 4))
        assert bool(ops.all(layer.base_scaler == 1.0))

    def test_factory_registers_base_scaler_initializer(self):
        assert "base_scaler_initializer" in FFN_REGISTRY["kan"]["optional_params"]


if __name__ == "__main__":
    pytest.main([__file__, "-vvv"])
