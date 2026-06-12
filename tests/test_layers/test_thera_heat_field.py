"""Scoped tests for THERA ``ThermalActivation`` + ``HeatField`` (plan step 3)."""

import os
import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.thera_heat_field import (
    ThermalActivation,
    HeatField,
    DEFAULT_K_INIT,
    DEFAULT_COMPONENTS_INIT_SCALE,
)


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------

B, HQ, WQ, HIDDEN, OUT = 2, 8, 8, 32, 3


@pytest.fixture
def field_inputs():
    rng = np.random.default_rng(0)
    rel_coords = rng.standard_normal((B, HQ, WQ, 2)).astype("float32") * 0.1
    phi_phase = rng.standard_normal((B, HQ, WQ, HIDDEN)).astype("float32")
    phi_kernel = rng.standard_normal((B, HQ, WQ, HIDDEN, OUT)).astype("float32")
    t = np.full((B, 1), 1.0, dtype="float32")
    return rel_coords, phi_phase, phi_kernel, t


@pytest.fixture
def heat_field():
    return HeatField(hidden_dim=HIDDEN, out_dim=OUT)


# ---------------------------------------------------------------------
# 1. forward shape
# ---------------------------------------------------------------------

class TestForwardShape:
    def test_output_shape_and_finite(self, heat_field, field_inputs):
        rel_coords, phi_phase, phi_kernel, t = field_inputs
        out = heat_field(rel_coords, phi_phase, phi_kernel, t)
        assert tuple(out.shape) == (B, HQ, WQ, OUT)
        out_np = ops.convert_to_numpy(out)
        assert np.all(np.isfinite(out_np))

    def test_thermal_activation_shape(self):
        act = ThermalActivation(w0=1.0)
        x = ops.convert_to_tensor(np.random.randn(B, HQ, WQ, HIDDEN).astype("float32"))
        phase = ops.convert_to_tensor(np.random.randn(B, HQ, WQ, HIDDEN).astype("float32"))
        norm = ops.convert_to_tensor(np.abs(np.random.randn(HIDDEN)).astype("float32"))
        out = act(x, t=1.0, norm=norm, k=0.5, phase=phase)
        assert tuple(out.shape) == (B, HQ, WQ, HIDDEN)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))


# ---------------------------------------------------------------------
# 2. t-sweep finiteness + envelope behavior
# ---------------------------------------------------------------------

class TestTimeSweep:
    def test_finite_across_t(self, heat_field, field_inputs):
        rel_coords, phi_phase, phi_kernel, _ = field_inputs
        for tval in (0.0, 1e-3, 1.0, 100.0):
            t = np.full((B, 1), tval, dtype="float32")
            out = heat_field(rel_coords, phi_phase, phi_kernel, t)
            assert np.all(np.isfinite(ops.convert_to_numpy(out))), f"t={tval}"

    def test_envelope_unity_at_t0(self, heat_field, field_inputs):
        # At t=0 the heat envelope exp(0) == 1, so the field == the no-decay
        # SIREN field. Compare HeatField output at t=0 against a hand-rolled
        # no-envelope evaluation.
        rel_coords, phi_phase, phi_kernel, _ = field_inputs
        t0 = np.zeros((B, 1), dtype="float32")
        out = ops.convert_to_numpy(heat_field(rel_coords, phi_phase, phi_kernel, t0))

        comps = ops.convert_to_numpy(heat_field.components)  # (2, hidden)
        x = np.einsum("...c,ck->...k", rel_coords, comps)
        thermal = np.sin(heat_field.w0 * x + phi_phase)  # envelope == 1 at t=0
        ref = np.einsum("...k,...ko->...o", thermal, phi_kernel)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_envelope_monotone_shrink(self, heat_field, field_inputs):
        # As t grows the envelope -> 0, so the field magnitude shrinks
        # monotonically; at very large t it is ~0.
        rel_coords, phi_phase, phi_kernel, _ = field_inputs
        max_abs = []
        for tval in (0.0, 0.1, 1.0, 10.0, 1000.0):
            t = np.full((B, 1), tval, dtype="float32")
            out = ops.convert_to_numpy(heat_field(rel_coords, phi_phase, phi_kernel, t))
            max_abs.append(np.max(np.abs(out)))
        for a, b in zip(max_abs, max_abs[1:]):
            assert b <= a + 1e-6, f"non-monotone: {max_abs}"
        assert max_abs[-1] < 1e-3, f"large-t not ~0: {max_abs[-1]}"


# ---------------------------------------------------------------------
# 3. differentiability w.r.t. rel_coords (step-9 Jacobian-through-field path)
# ---------------------------------------------------------------------

class TestDifferentiability:
    def test_grad_wrt_rel_coords(self, heat_field, field_inputs):
        rel_coords, phi_phase, phi_kernel, t = field_inputs
        rc = tf.Variable(rel_coords)
        pp = tf.constant(phi_phase)
        pk = tf.constant(phi_kernel)
        tt = tf.constant(t)
        with tf.GradientTape() as tape:
            tape.watch(rc)
            out = heat_field(rc, pp, pk, tt)
            loss = tf.reduce_sum(out)
        grad = tape.gradient(loss, rc)
        assert grad is not None
        g = grad.numpy()
        assert np.all(np.isfinite(g))
        assert np.any(np.abs(g) > 0)


# ---------------------------------------------------------------------
# 4. .keras round-trip through a tiny model
# ---------------------------------------------------------------------

class TestKerasRoundTrip:
    def test_save_load(self, field_inputs, tmp_path):
        rel_coords, phi_phase, phi_kernel, t = field_inputs

        rc_in = keras.Input(shape=(HQ, WQ, 2), name="rel_coords")
        pp_in = keras.Input(shape=(HQ, WQ, HIDDEN), name="phi_phase")
        pk_in = keras.Input(shape=(HQ, WQ, HIDDEN, OUT), name="phi_kernel")
        t_in = keras.Input(shape=(1,), name="t")
        field = HeatField(hidden_dim=HIDDEN, out_dim=OUT, name="hf")
        out = field(rc_in, pp_in, pk_in, t_in)
        model = keras.Model([rc_in, pp_in, pk_in, t_in], out)

        ref = model.predict([rel_coords, phi_phase, phi_kernel, t], verbose=0)

        path = os.path.join(tmp_path, "hf.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        got = reloaded.predict([rel_coords, phi_phase, phi_kernel, t], verbose=0)

        np.testing.assert_allclose(got, ref, atol=1e-5)

        # weights survived
        rel_field = reloaded.get_layer("hf")
        np.testing.assert_allclose(
            ops.convert_to_numpy(rel_field.components),
            ops.convert_to_numpy(field.components),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            float(ops.convert_to_numpy(rel_field.k)),
            float(ops.convert_to_numpy(field.k)),
            atol=1e-6,
        )


# ---------------------------------------------------------------------
# 5. get_config / from_config round-trip
# ---------------------------------------------------------------------

class TestConfigRoundTrip:
    def test_thermal_activation_config(self):
        act = ThermalActivation(w0=2.5)
        cfg = act.get_config()
        rebuilt = ThermalActivation.from_config(cfg)
        assert rebuilt.w0 == act.w0 == 2.5

    def test_heat_field_config(self):
        hf = HeatField(
            hidden_dim=64, out_dim=5, w0=1.5, c=4.0,
            k_init=0.07, components_init_scale=8.0,
        )
        cfg = hf.get_config()
        rebuilt = HeatField.from_config(cfg)
        assert rebuilt.hidden_dim == 64
        assert rebuilt.out_dim == 5
        assert rebuilt.w0 == 1.5
        assert rebuilt.c == 4.0
        assert rebuilt.k_init == 0.07
        assert rebuilt.components_init_scale == 8.0

    def test_heat_field_defaults(self):
        hf = HeatField(hidden_dim=16, out_dim=3)
        assert hf.k_init == pytest.approx(DEFAULT_K_INIT)
        assert hf.components_init_scale == DEFAULT_COMPONENTS_INIT_SCALE


# ---------------------------------------------------------------------
# 6. components weight shape + LinearUpInitializer distribution
# ---------------------------------------------------------------------

class TestComponentsWeight:
    def test_shape_and_disk_distribution(self):
        scale = 4.0
        hf = HeatField(hidden_dim=256, out_dim=3, components_init_scale=scale)
        # trigger build
        rel_coords = np.zeros((1, 2, 2, 2), dtype="float32")
        phi_phase = np.zeros((1, 2, 2, 256), dtype="float32")
        phi_kernel = np.zeros((1, 2, 2, 256, 3), dtype="float32")
        t = np.zeros((1, 1), dtype="float32")
        hf(rel_coords, phi_phase, phi_kernel, t)

        comps = ops.convert_to_numpy(hf.components)
        assert comps.shape == (2, 256)
        # Each column is a 2D frequency vector with norm in [0, pi*scale].
        col_norms = np.linalg.norm(comps, axis=0)
        max_r = np.pi * scale
        assert np.all(col_norms >= -1e-6)
        assert np.all(col_norms <= max_r + 1e-4)
        # Uniform-on-disk => mean radius well inside the disk (not clustered at edge).
        assert col_norms.mean() < max_r

    def test_k_initial_value(self):
        hf = HeatField(hidden_dim=16, out_dim=3, k_init=0.123)
        hf(
            np.zeros((1, 2, 2, 2), dtype="float32"),
            np.zeros((1, 2, 2, 16), dtype="float32"),
            np.zeros((1, 2, 2, 16, 3), dtype="float32"),
            np.zeros((1, 1), dtype="float32"),
        )
        assert float(ops.convert_to_numpy(hf.k)) == pytest.approx(0.123)
