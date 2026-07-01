"""
Test suite for the BiasFreeBatchNorm layer (layers/norms/bias_free_batch_norm.py).

BiasFreeBatchNorm is a variance-only, fixed-statistic normalization designed to
preserve degree-1 homogeneity ``f(alpha * x) = alpha * f(x)`` at INFERENCE
(``training=False``). It creates NO ``moving_mean`` and NO ``beta`` — only an
EMA-tracked non-trainable ``running_var`` and an optional trainable ``gamma``.

Coverage (plan_2026-07-01_8054f023 Step 10a / SC1a / SC3):
- construction + output shape + weight inventory (no moving_mean, no beta);
- ``get_config`` round-trip;
- INFERENCE homogeneity: after populating ``running_var`` with a few
  ``training=True`` calls, ``f(alpha x) ~= alpha f(x)`` (rel err < 1e-5) for
  ``alpha`` in {0.5, 2.0, 5.0}, and ``f(0) ~= 0`` (no additive offset);
- training-mode is scale-INVARIANT (degree-0) — documented, not a defect;
- ``.keras`` save/load round-trip through a tiny functional model.

All homogeneity / round-trip checks use ``training=False``. Run CPU-only:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_layers/test_norms/test_bias_free_batch_norm.py -q
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.norms.bias_free_batch_norm import BiasFreeBatchNorm


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _populate_running_var(layer: BiasFreeBatchNorm, shape, n_steps: int = 20,
                          seed: int = 0) -> None:
    """Drive a few ``training=True`` calls so ``running_var`` diverges from 1.0.

    The EMA update ``running_var <- momentum*running_var + (1-momentum)*batch_var``
    only moves the statistic under ``training=True``. We feed non-unit-variance
    batches so the fixed inference statistic becomes data-meaningful (and not the
    trivial init value of 1.0).
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        # variance clearly != 1 so the EMA visibly moves running_var.
        x = (rng.standard_normal(size=shape) * 3.0).astype("float32")
        layer(x, training=True)


class TestBiasFreeBatchNormConstruction:
    """Construction, shape, config round-trip, and weight inventory."""

    def test_default_construction(self) -> None:
        layer = BiasFreeBatchNorm()
        assert layer.axis == -1
        assert layer.epsilon == 1e-6
        assert layer.momentum == 0.99
        assert layer.use_scale is True
        assert not layer.built
        assert layer.running_var is None
        assert layer.gamma is None

    def test_forward_pass_and_shape(self) -> None:
        layer = BiasFreeBatchNorm()
        x = keras.random.normal((4, 8, 8, 16), seed=42)
        y = layer(x, training=False)
        assert layer.built
        assert y.shape == x.shape

    def test_weight_inventory_no_mean_no_beta(self) -> None:
        """Variance-only: exactly running_var (non-trainable) + gamma (trainable);
        NO moving_mean, NO beta."""
        layer = BiasFreeBatchNorm()
        _ = layer(keras.random.normal((2, 4, 4, 8), seed=1), training=False)

        weight_names = {w.name for w in layer.weights}
        assert any("running_var" in n for n in weight_names)
        assert not any("moving_mean" in n or "mean" in n for n in weight_names), (
            f"BiasFreeBatchNorm must NOT create a mean weight; found {weight_names}"
        )
        assert not any("beta" in n for n in weight_names), (
            f"BiasFreeBatchNorm must NOT create a beta weight; found {weight_names}"
        )
        # running_var is non-trainable, gamma is trainable.
        assert layer.running_var.trainable is False
        assert layer.gamma is not None and layer.gamma.trainable is True

    def test_use_scale_false_has_no_gamma(self) -> None:
        layer = BiasFreeBatchNorm(use_scale=False)
        _ = layer(keras.random.normal((2, 4, 4, 8), seed=1), training=False)
        assert layer.gamma is None
        assert len(layer.trainable_weights) == 0

    def test_config_round_trip(self) -> None:
        layer = BiasFreeBatchNorm(axis=-1, epsilon=1e-5, momentum=0.95, use_scale=False)
        config = layer.get_config()
        for key in ("axis", "epsilon", "momentum", "use_scale"):
            assert key in config, f"missing {key} in get_config()"

        rebuilt = BiasFreeBatchNorm.from_config(config)
        assert rebuilt.axis == layer.axis
        assert rebuilt.epsilon == layer.epsilon
        assert rebuilt.momentum == layer.momentum
        assert rebuilt.use_scale == layer.use_scale

    def test_invalid_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be positive"):
            BiasFreeBatchNorm(epsilon=0.0)
        with pytest.raises(ValueError, match="momentum must be in"):
            BiasFreeBatchNorm(momentum=1.5)


class TestBiasFreeBatchNormHomogeneity:
    """INFERENCE-time degree-1 homogeneity: f(alpha x) = alpha f(x)."""

    SHAPE = (4, 8, 8, 16)

    def _built_layer(self) -> BiasFreeBatchNorm:
        layer = BiasFreeBatchNorm()
        _ = layer(keras.random.normal(self.SHAPE, seed=7), training=False)
        # Populate running_var to a non-trivial (data-meaningful) value, and set a
        # non-unit gamma so the test would also catch a gamma mishandling.
        _populate_running_var(layer, self.SHAPE, n_steps=25, seed=3)
        layer.gamma.assign(keras.ops.convert_to_tensor(
            np.linspace(0.5, 2.0, self.SHAPE[-1]).astype("float32")
        ))
        return layer

    @pytest.mark.parametrize("alpha", [0.5, 2.0, 5.0])
    def test_inference_homogeneous(self, alpha: float) -> None:
        layer = self._built_layer()
        rng = np.random.default_rng(11)
        x = (rng.standard_normal(size=self.SHAPE) * 2.0).astype("float32")

        fx = np.asarray(keras.ops.convert_to_numpy(layer(x, training=False)))
        f_ax = np.asarray(keras.ops.convert_to_numpy(layer(alpha * x, training=False)))

        denom = max(float(np.max(np.abs(alpha * fx))), 1e-8)
        rel = float(np.max(np.abs(f_ax - alpha * fx)) / denom)
        assert rel < 1e-5, (
            f"BiasFreeBatchNorm not degree-1 homogeneous at inference for "
            f"alpha={alpha}: rel_err={rel:.3e}"
        )

    def test_zeros_map_to_zeros(self) -> None:
        """No additive offset: f(0) == 0 (variance-only, no beta/mean)."""
        layer = self._built_layer()
        zeros = np.zeros(self.SHAPE, dtype="float32")
        fz = np.asarray(keras.ops.convert_to_numpy(layer(zeros, training=False)))
        assert np.max(np.abs(fz)) < 1e-6, (
            f"f(zeros) should be ~0 for a variance-only (no-mean, no-beta) norm; "
            f"max|f(0)|={np.max(np.abs(fz)):.3e}"
        )

    def test_training_mode_is_scale_invariant(self) -> None:
        """Documenting: in TRAINING mode the layer uses the per-batch variance, so
        it is scale-INVARIANT (degree-0), NOT degree-1. This is architecturally
        unavoidable for the BatchNorm family and is why homogeneity is probed with
        training=False. f_train(alpha x) ~= f_train(x), not alpha*f_train(x)."""
        layer = BiasFreeBatchNorm(use_scale=False)
        _ = layer(keras.random.normal(self.SHAPE, seed=5), training=False)
        rng = np.random.default_rng(13)
        x = (rng.standard_normal(size=self.SHAPE) * 2.0).astype("float32")
        alpha = 3.0

        fx = np.asarray(keras.ops.convert_to_numpy(layer(x, training=True)))
        f_ax = np.asarray(keras.ops.convert_to_numpy(layer(alpha * x, training=True)))
        # scale-invariant: alpha cancels in alpha x / sqrt(alpha^2 var).
        assert np.allclose(fx, f_ax, atol=1e-4), (
            "training-mode BiasFreeBatchNorm should be scale-INVARIANT (degree-0)"
        )


class TestBiasFreeBatchNormSerialization:
    """.keras save/load round-trip through a tiny functional model (SC3)."""

    def test_keras_round_trip(self) -> None:
        shape = (4, 8, 8, 16)
        inputs = keras.Input(shape=shape[1:])
        outputs = BiasFreeBatchNorm(epsilon=1e-6, momentum=0.9)(inputs)
        model = keras.Model(inputs, outputs)

        # Populate running_var so the saved statistic is non-trivial.
        rng = np.random.default_rng(2)
        for _ in range(15):
            model(( rng.standard_normal(shape) * 2.5).astype("float32"), training=True)

        x = (rng.standard_normal(shape) * 2.0).astype("float32")
        y_before = np.asarray(keras.ops.convert_to_numpy(model(x, training=False)))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bfbn.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            y_after = np.asarray(keras.ops.convert_to_numpy(loaded(x, training=False)))

        np.testing.assert_allclose(
            y_before, y_after, atol=1e-5,
            err_msg="BiasFreeBatchNorm outputs differ after .keras round-trip",
        )


if __name__ == "__main__":
    pytest.main([__file__])
