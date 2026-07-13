"""
Test suite for the EnergyLayerNorm layer (layers/norms/energy_layer_norm.py).

EnergyLayerNorm is the Energy Transformer's layer norm (arXiv:2302.07253 eq. 1-2). Its
defining property — and the reason stock ``keras.layers.LayerNormalization`` cannot be
reused — is the parameterization:

    gamma is a SCALAR   (shape == ())
    delta is a VECTOR   (shape == (D,))

``test_build_creates_weights`` is the TRIPWIRE for that property (plan
plan_2026-07-13_57c9833e, success criterion S1): a reviewer "fixing" gamma into a
per-feature vector must make it RED.

Coverage: instantiation, invalid config, forward pass, weight shapes, normalization
correctness vs a numpy reference, .keras serialization round-trip, get_config
completeness, from_config, compute_output_shape (incl. PRE-build), variable batch size,
gradient flow, and norms-factory integration (S5).

Run:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \\
        tests/test_layers/test_norms/test_energy_layer_norm.py -q
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

# LESSONS [I:4]: TF32 truncates matmul mantissas on Ampere+ and makes an exact numeric
# comparison a coin-flip. The Jacobian-symmetry check below is exactly that kind of test.
tf.config.experimental.enable_tensor_float_32_execution(False)

from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
from dl_techniques.layers.norms.energy_layer_norm import EnergyLayerNorm, _GAMMA_FLOOR
from dl_techniques.layers.norms.factory import create_normalization_layer


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def input_shape():
    """(batch, tokens, embed_dim)."""
    return (4, 12, 32)


@pytest.fixture
def sample_input(input_shape):
    rng = np.random.default_rng(42)
    # Non-zero mean and non-unit variance so centering/scaling are both exercised.
    return (rng.standard_normal(size=input_shape) * 2.5 + 1.5).astype("float32")


@pytest.fixture
def layer():
    return EnergyLayerNorm()


# ---------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------

class TestEnergyLayerNorm:
    """Full layer contract for EnergyLayerNorm."""

    def test_instantiation(self):
        default = EnergyLayerNorm()
        assert default.epsilon == 1e-5
        assert isinstance(default.gamma_initializer, keras.initializers.Ones)
        assert isinstance(default.delta_initializer, keras.initializers.Zeros)
        assert default.gamma is None
        assert default.delta is None
        assert not default.built

        custom = EnergyLayerNorm(
            epsilon=1e-3,
            gamma_initializer="zeros",
            delta_initializer="ones",
        )
        assert custom.epsilon == 1e-3
        assert isinstance(custom.gamma_initializer, keras.initializers.Zeros)
        assert isinstance(custom.delta_initializer, keras.initializers.Ones)

    @pytest.mark.parametrize("bad_epsilon", [0.0, -1e-5, -1.0])
    def test_invalid_config(self, bad_epsilon):
        with pytest.raises(ValueError, match="epsilon must be a positive number"):
            EnergyLayerNorm(epsilon=bad_epsilon)

    def test_forward_pass(self, layer, sample_input, input_shape):
        y = layer(sample_input, training=False)
        assert tuple(y.shape) == input_shape
        y_np = keras.ops.convert_to_numpy(y)
        assert np.all(np.isfinite(y_np))
        assert not np.any(np.isnan(y_np))

    # -----------------------------------------------------------------
    # S1 — THE TRIPWIRE: scalar gamma, vector delta.
    # -----------------------------------------------------------------

    def test_build_creates_weights(self, layer, sample_input, input_shape):
        """gamma MUST be a scalar and delta MUST be a (D,) vector (S1).

        This is the entire reason the class exists. Stock LayerNormalization has a
        VECTOR gamma; a vector gamma here breaks the identity g = dL/dx and destroys
        the Energy Transformer's descent guarantee.
        """
        layer.build(input_shape)
        d = input_shape[-1]

        assert tuple(layer.gamma.shape) == (), (
            f"gamma MUST be a SCALAR (shape ()), got {tuple(layer.gamma.shape)}. "
            "A per-feature gamma is NOT the ET layer norm."
        )
        assert tuple(layer.delta.shape) == (d,), (
            f"delta MUST be a VECTOR of shape ({d},), got {tuple(layer.delta.shape)}"
        )
        assert len(layer.trainable_weights) == 2, (
            f"expected exactly 2 trainable weights (gamma, delta), "
            f"got {[w.name for w in layer.trainable_weights]}"
        )
        assert len(layer.non_trainable_weights) == 0

    # -----------------------------------------------------------------
    # Numerical correctness
    # -----------------------------------------------------------------

    def test_normalization_correctness(self, sample_input):
        """With gamma=1, delta=0 the output is zero-mean / unit-variance on axis -1,
        and matches a hand-computed numpy reference."""
        eps = 1e-5
        layer = EnergyLayerNorm(epsilon=eps)
        y = keras.ops.convert_to_numpy(layer(sample_input, training=False))

        # Statistics along the last axis.
        np.testing.assert_allclose(
            y.mean(axis=-1), np.zeros(y.shape[:-1]), atol=1e-5,
            err_msg="output should be zero-mean along the last axis",
        )
        np.testing.assert_allclose(
            y.var(axis=-1), np.ones(y.shape[:-1]), atol=1e-4,
            err_msg="output should be unit-variance along the last axis",
        )

        # Hand-computed reference: gamma*(x - xbar)/sqrt(var + eps) + delta.
        x = sample_input.astype("float64")
        xbar = x.mean(axis=-1, keepdims=True)
        centered = x - xbar
        var = (centered ** 2).mean(axis=-1, keepdims=True)
        ref = 1.0 * centered / np.sqrt(var + eps) + 0.0
        np.testing.assert_allclose(y, ref.astype("float32"), rtol=1e-5, atol=1e-5)

    def test_normalization_correctness_nondefault_weights(self, sample_input):
        """Reference check with a non-unit gamma and a non-zero delta, so a dropped
        gamma or delta cannot pass by accident."""
        eps = 1e-5
        d = sample_input.shape[-1]
        layer = EnergyLayerNorm(epsilon=eps)
        layer.build(sample_input.shape)

        gamma_val = 1.7
        delta_val = np.linspace(-1.0, 1.0, d).astype("float32")
        layer.gamma.assign(keras.ops.convert_to_tensor(gamma_val, dtype="float32"))
        layer.delta.assign(keras.ops.convert_to_tensor(delta_val))

        y = keras.ops.convert_to_numpy(layer(sample_input, training=False))

        x = sample_input.astype("float64")
        xbar = x.mean(axis=-1, keepdims=True)
        centered = x - xbar
        var = (centered ** 2).mean(axis=-1, keepdims=True)
        ref = gamma_val * centered / np.sqrt(var + eps) + delta_val.astype("float64")
        np.testing.assert_allclose(y, ref.astype("float32"), rtol=1e-5, atol=1e-5)

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def test_serialization_cycle(self, sample_input, input_shape):
        inputs = keras.Input(shape=input_shape[1:])
        outputs = EnergyLayerNorm(epsilon=1e-4, delta_initializer="ones")(inputs)
        model = keras.Model(inputs, outputs)

        y_before = keras.ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "energy_layer_norm.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            y_after = keras.ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            y_before, y_after, rtol=1e-6, atol=1e-6,
            err_msg="EnergyLayerNorm outputs differ after a .keras round-trip",
        )

    def test_get_config_complete(self):
        layer = EnergyLayerNorm(
            epsilon=1e-4,
            gamma_initializer="ones",
            delta_initializer="zeros",
        )
        config = layer.get_config()
        for key in ("epsilon", "gamma_initializer", "delta_initializer"):
            assert key in config, f"{key} missing from get_config()"
        assert config["epsilon"] == 1e-4

    def test_from_config_reconstruction(self, sample_input):
        original = EnergyLayerNorm(epsilon=1e-4, delta_initializer="ones")
        rebuilt = EnergyLayerNorm.from_config(original.get_config())

        assert rebuilt.epsilon == original.epsilon
        assert isinstance(rebuilt.delta_initializer, keras.initializers.Ones)

        # Fresh weights (ones-gamma / ones-delta) => identical forward output.
        y0 = keras.ops.convert_to_numpy(original(sample_input, training=False))
        y1 = keras.ops.convert_to_numpy(rebuilt(sample_input, training=False))
        np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)

    # -----------------------------------------------------------------
    # Shapes
    # -----------------------------------------------------------------

    def test_compute_output_shape(self, layer, input_shape):
        layer.build(input_shape)
        assert layer.compute_output_shape(input_shape) == input_shape

    def test_compute_output_shape_before_build(self, input_shape):
        """compute_output_shape must work on an UNBUILT layer (no weight shapes read)."""
        unbuilt = EnergyLayerNorm()
        assert not unbuilt.built
        assert unbuilt.compute_output_shape(input_shape) == input_shape
        assert unbuilt.compute_output_shape((None, 7, 16)) == (None, 7, 16)
        assert not unbuilt.built

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_variable_batch_size(self, batch_size):
        layer = EnergyLayerNorm()
        rng = np.random.default_rng(batch_size)
        x = rng.standard_normal(size=(batch_size, 6, 16)).astype("float32")
        y = layer(x, training=False)
        assert tuple(y.shape) == (batch_size, 6, 16)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))

    # -----------------------------------------------------------------
    # Gradients
    # -----------------------------------------------------------------

    def test_gradient_flow(self, sample_input, input_shape):
        import tensorflow as tf

        layer = EnergyLayerNorm()
        layer.build(input_shape)
        x = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(y))

        grads = tape.gradient(loss, layer.trainable_weights)
        assert len(grads) == 2
        for g, w in zip(grads, layer.trainable_weights):
            assert g is not None, f"no gradient for {w.name}"
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g))), (
                f"non-finite gradient for {w.name}"
            )

    # -----------------------------------------------------------------
    # Factory (S5)
    # -----------------------------------------------------------------

    def test_factory_integration(self, sample_input, input_shape):
        layer = create_normalization_layer("energy_layer_norm", name="eln")
        assert isinstance(layer, EnergyLayerNorm)

        y = layer(sample_input, training=False)
        assert tuple(y.shape) == input_shape
        assert tuple(layer.gamma.shape) == ()
        assert tuple(layer.delta.shape) == (input_shape[-1],)


class TestGammaPositivityConstraint:
    """S15: `gamma > 0` is ENFORCED, not merely documented.

    `gamma > 0` is the PRECONDITION for the PSD Hessian of the Lagrangian — i.e. the
    precondition for the ENTIRE energy-descent guarantee. Iteration 1 stated it as an
    invariant and enforced it NOWHERE: an adversarial reviewer set `gamma = -1.0` and the
    block silently performed energy ASCENT (max diff(E) = +1.3e4), with no error and no
    failing test.
    """

    def test_constraint_is_present_by_default(self, sample_input):
        layer = EnergyLayerNorm()
        layer(sample_input, training=False)

        assert layer.gamma.constraint is not None, (
            "gamma has NO constraint — a trained gamma < 0 silently inverts the descent"
        )
        assert isinstance(layer.gamma_constraint, ValueRangeConstraint)
        assert layer.gamma_constraint.min_value == pytest.approx(_GAMMA_FLOOR)

    def test_constraint_is_ACTIVE_after_an_optimizer_step(self, sample_input):
        """Not just attached — it must actually PROJECT gamma back above the floor."""
        layer = EnergyLayerNorm()
        layer(sample_input, training=False)

        # Drive gamma hard negative, then let the optimizer's constraint hook fire.
        layer.gamma.assign(-5.0)
        assert float(keras.ops.convert_to_numpy(layer.gamma)) == -5.0

        optimizer = keras.optimizers.SGD(learning_rate=0.0)   # lr=0: ONLY the constraint acts
        optimizer.build(layer.trainable_variables)
        zeros = [keras.ops.zeros_like(v) for v in layer.trainable_variables]
        optimizer.apply_gradients(zip(zeros, layer.trainable_variables))

        gamma = float(keras.ops.convert_to_numpy(layer.gamma))
        assert gamma >= _GAMMA_FLOOR, (
            f"gamma = {gamma} is below the floor {_GAMMA_FLOOR} AFTER an optimizer step — "
            "the constraint is attached but INERT"
        )

    def test_constraint_can_be_disabled_DELIBERATELY(self, sample_input):
        """Overridable — but only by explicitly passing None, never silently."""
        layer = EnergyLayerNorm(gamma_constraint=None)
        layer(sample_input, training=False)
        assert layer.gamma.constraint is None

    def test_constraint_round_trips_through_get_config(self, sample_input, input_shape):
        """If the constraint were dropped from get_config, a RELOADED model would train
        itself into energy ascent — the exact defect the constraint exists to stop."""
        original = EnergyLayerNorm()
        original(sample_input, training=False)

        restored = EnergyLayerNorm.from_config(original.get_config())
        restored(sample_input, training=False)

        assert isinstance(restored.gamma_constraint, ValueRangeConstraint)
        assert restored.gamma.constraint is not None
        assert restored.gamma_constraint.min_value == pytest.approx(_GAMMA_FLOOR)

        # An explicit None must round-trip as None, NOT silently re-acquire the default.
        unconstrained = EnergyLayerNorm(gamma_constraint=None)
        unconstrained(sample_input, training=False)
        assert (
            EnergyLayerNorm.from_config(unconstrained.get_config()).gamma_constraint is None
        )


class TestJacobianSymmetry:
    """S16: the SCALAR-gamma property, guarded BEHAVIORALLY (not by a shape check).

    `g = dL/dx` for `L = D*gamma*sqrt(var+eps) + sum_j delta_j x_j`. The Jacobian
    `J = dg/dx` is therefore the HESSIAN of a scalar L, and a Hessian is NECESSARILY
    SYMMETRIC. A per-feature VECTOR gamma breaks that symmetry — so it is not the gradient
    of ANY scalar potential and the descent guarantee evaporates.

    Iteration 1 guarded the scalar-gamma property with a SHAPE assertion only. A reviewer
    patched in a vector gamma and S7 still descended: nothing BEHAVIORAL protected the
    property that justifies this class existing. This test is that guard.

    `tf.GradientTape().jacobian` is legitimate HERE — autodiff is forbidden only in `src/`.
    """

    @staticmethod
    def _jacobian(layer, x_np):
        """J[i, j] = d g_i / d x_j for a SINGLE token vector."""
        x = tf.Variable(x_np.reshape(1, 1, -1), dtype=tf.float32)
        with tf.GradientTape() as tape:
            g = layer(x, training=False)
            g_flat = tf.reshape(g, [-1])                       # (D,)
        j = tape.jacobian(g_flat, x)                           # (D, 1, 1, D)
        return np.asarray(tf.reshape(j, (x_np.size, x_np.size)))

    def test_jacobian_is_symmetric(self):
        rng = np.random.default_rng(17)
        x_np = (rng.standard_normal(16) * 2.0 + 0.5).astype("float32")

        layer = EnergyLayerNorm()
        layer.build((1, 1, 16))
        layer.gamma.assign(1.7)                                # a SCALAR, as designed
        layer.delta.assign(rng.standard_normal(16).astype("float32"))

        j = self._jacobian(layer, x_np)
        asym = float(np.abs(j - j.T).max())

        # Anti-vacuity: a zero Jacobian would be symmetric for free.
        assert np.abs(j).max() > 1e-2, f"Jacobian is ~0 (max |J| = {np.abs(j).max():.3e})"
        assert asym < 1e-5, (
            f"dg/dx is NOT symmetric (max |J - J^T| = {asym:.3e}). It must be: it is the "
            "HESSIAN of the Lagrangian L. An asymmetric J means `g` is not the gradient of "
            "ANY scalar potential — the ET descent guarantee is FALSE. Did gamma become a "
            "per-feature VECTOR?"
        )


class TestDtypePolicies:
    """S13: EnergyLayerNorm is FINITE under every global dtype policy."""

    def test_no_nan_under_mixed_precision(self, dtype_policy, sample_input, input_shape):
        layer = EnergyLayerNorm()
        out = keras.ops.convert_to_numpy(layer(sample_input, training=False))

        assert out.shape == input_shape
        assert np.isnan(out).sum() == 0, f"{np.isnan(out).sum()}/{out.size} NaN"
        assert np.isinf(out).sum() == 0, f"{np.isinf(out).sum()}/{out.size} Inf"
        assert np.all(np.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__])
