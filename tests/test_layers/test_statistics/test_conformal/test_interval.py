"""Test suite for :class:`ConformalInterval`.

Covers:
- construction (registered, non-trainable, zero trainable weights after build);
- forward numeric (clip + interval math) and ``compute_output_shape`` arity;
- ``calibrate()``-then-forward;
- CRUX: a NON-ZERO calibrated ``q`` surviving a real ``.keras``
  ``model.save`` / ``load_model`` round-trip (weights + forward intervals);
- config-only (``from_config``) round-trip recovering ``q`` (D-004);
- ``return_mu=False`` 2-tuple arity.
"""

import keras
import numpy as np
import pytest

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.statistics.conformal import ConformalInterval

# ---------------------------------------------------------------------

ATOL = 1e-6
CAL_Q = 0.0488  # a representative non-zero conformal radius (crux value)


class TestConformalInterval:
    """Unit tests for the fixed-weights conformal interval layer."""

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        """Seed RNGs for reproducibility."""
        keras.utils.set_random_seed(1234)
        np.random.seed(1234)

    @pytest.fixture
    def mu_outside(self) -> np.ndarray:
        """A point-estimate tensor straddling BOTH bounds of the ``[0, 1]`` domain.

        The denoiser domain is strictly positive and NOT symmetric about zero, so
        out-of-domain must be probed on both sides independently (below ``0.0``
        AND above ``1.0``) — a single ``abs(mu) > 0.5``-style probe would be
        wrong.
        """
        return np.array(
            [[-0.4, -0.05, 0.0, 0.3, 0.75, 1.4]],
            dtype="float32",
        )

    # -- construction -------------------------------------------------

    def test_construction(self, registration_contract) -> None:
        """Layer registers, is non-trainable, and has no trainable weights."""
        # The literal `"Custom>ConformalInterval"` this line used to spell
        # is now only the LEGACY alias; the shared predicate checks it and the
        # package-qualified key the layer actually registers under today.
        registration_contract(ConformalInterval)

        layer = ConformalInterval()
        assert layer.trainable is False
        assert layer.q_hat is None  # not built yet

        # Build via a forward call.
        layer(np.zeros((1, 4), dtype="float32"))
        assert layer.q_hat is not None
        assert len(layer.trainable_weights) == 0
        # The non-trainable conformal_q weight exists.
        assert len(layer.non_trainable_weights) == 1
        assert layer.non_trainable_weights[0].name == "conformal_q_hat"

    # -- forward numeric ----------------------------------------------

    def test_default_domain_is_unit_interval(self) -> None:
        """The constructor defaults pin the ``[0, 1]`` denoiser domain."""
        layer = ConformalInterval()
        assert layer.domain_min == 0.0
        assert layer.domain_max == 1.0

    def test_forward_numeric(self, mu_outside: np.ndarray) -> None:
        """``call`` clips ``mu`` and returns ``(mu_c, mu_c-q, mu_c+q)``."""
        layer = ConformalInterval(q_hat_init=0.1)
        mu_c, lower, upper = layer(mu_outside)

        mu_c = keras.ops.convert_to_numpy(mu_c)
        lower = keras.ops.convert_to_numpy(lower)
        upper = keras.ops.convert_to_numpy(upper)

        expected_mu_c = np.clip(mu_outside, 0.0, 1.0)
        np.testing.assert_allclose(mu_c, expected_mu_c, atol=ATOL)
        np.testing.assert_allclose(lower, expected_mu_c - 0.1, atol=ATOL)
        np.testing.assert_allclose(upper, expected_mu_c + 0.1, atol=ATOL)

        # BOTH clip bounds genuinely fired (two-sided out-of-domain inputs).
        assert np.any(mu_outside < 0.0) and np.any(mu_outside > 1.0)

    def test_forward_compute_output_shape(self) -> None:
        """``compute_output_shape`` arity matches the returned tuple (3-tuple)."""
        layer = ConformalInterval()
        in_shape = (None, 8, 8, 3)
        out_shape = layer.compute_output_shape(in_shape)
        assert isinstance(out_shape, tuple)
        assert len(out_shape) == 3
        for s in out_shape:
            assert s == in_shape

    # -- calibrate ----------------------------------------------------

    def test_calibrate_then_forward(self, mu_outside: np.ndarray) -> None:
        """A fresh forward reflects the calibrated ``q``."""
        layer = ConformalInterval()
        layer(mu_outside)  # build
        layer.set_q_hat(CAL_Q)

        assert float(keras.ops.convert_to_numpy(layer.q_hat)) == pytest.approx(CAL_Q, abs=ATOL)

        _, lower, upper = layer(mu_outside)
        mu_c = np.clip(mu_outside, 0.0, 1.0)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(lower), mu_c - CAL_Q, atol=ATOL
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(upper), mu_c + CAL_Q, atol=ATOL
        )

    # -- CRUX: .keras round-trip --------------------------------------

    def test_round_trip_keras(self, tmp_path) -> None:
        """CRUX: a NON-ZERO calibrated ``q`` survives a ``.keras`` round-trip."""
        inp = keras.Input(shape=(8, 8, 3))
        out = ConformalInterval()(inp)
        model = keras.Model(inp, out)

        # Locate the layer instance and calibrate to a non-zero value.
        layer = next(
            l for l in model.layers if isinstance(l, ConformalInterval)
        )
        layer.set_q_hat(CAL_Q)

        # Pre-save forward. The probe straddles BOTH bounds of the [0, 1] domain
        # so the round-trip is compared on a forward where both clips fired.
        x = np.random.uniform(-0.5, 1.5, size=(2, 8, 8, 3)).astype("float32")
        pre = [keras.ops.convert_to_numpy(t) for t in model(x)]

        save_path = tmp_path / "m.keras"
        model.save(save_path)
        reloaded = keras.models.load_model(
            save_path,
            custom_objects={"ConformalInterval": ConformalInterval},
        )

        rel_layer = next(
            l for l in reloaded.layers if isinstance(l, ConformalInterval)
        )
        reloaded_q = float(keras.ops.convert_to_numpy(rel_layer.q_hat))
        assert reloaded_q == pytest.approx(CAL_Q, abs=ATOL), (
            f"CRUX FAIL: reloaded q={reloaded_q} != {CAL_Q}"
        )

        post = [keras.ops.convert_to_numpy(t) for t in reloaded(x)]
        assert len(pre) == len(post) == 3
        for a, b in zip(pre, post):
            np.testing.assert_allclose(a, b, atol=ATOL)

    # -- config-path round-trip (D-004) -------------------------------

    def test_config_round_trip(self, mu_outside: np.ndarray) -> None:
        """``from_config(get_config())`` recovers ``q`` off the config path."""
        layer = ConformalInterval()
        layer(mu_outside)  # build
        layer.set_q_hat(CAL_Q)

        cfg = layer.get_config()
        assert cfg["q_hat_init"] == pytest.approx(CAL_Q, abs=ATOL)
        # The domain travels WITH the config (which is why a model saved before
        # the [-0.5,+0.5] -> [0,1] migration reloads with the OLD bounds; that is
        # intended and deliberately un-shimmed).
        assert cfg["domain_min"] == 0.0 and cfg["domain_max"] == 1.0

        layer2 = ConformalInterval.from_config(cfg)
        layer2(mu_outside)  # build layer2 so its weight is created from q_hat_init
        assert float(keras.ops.convert_to_numpy(layer2.q_hat)) == pytest.approx(
            CAL_Q, abs=ATOL
        )

    # -- return_mu=False ----------------------------------------------

    def test_return_mu_false(self, mu_outside: np.ndarray) -> None:
        """``return_mu=False`` returns a 2-tuple of correct shapes."""
        layer = ConformalInterval(q_hat_init=0.1, return_mu=False)
        out = layer(mu_outside)
        assert isinstance(out, tuple) and len(out) == 2

        lower, upper = out
        mu_c = np.clip(mu_outside, 0.0, 1.0)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(lower), mu_c - 0.1, atol=ATOL
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(upper), mu_c + 0.1, atol=ATOL
        )

        out_shape = layer.compute_output_shape((None, 6))
        assert isinstance(out_shape, tuple) and len(out_shape) == 2


# ---------------------------------------------------------------------
# v2 guide section 16.1 / 16.3 arms: validation, dtype, XLA, shapes, values on reload
# ---------------------------------------------------------------------

import tensorflow as tf  # noqa: E402


class TestV2Arms:
    """Arguments are validated, the radius lives in the VARIABLE dtype, and the layer
    behaves under XLA, an unknown batch, a half-precision policy and a values-level reload."""

    def test_arguments_are_validated_naming_the_value(self) -> None:
        with pytest.raises(ValueError, match="q_hat_init must be non-negative, got -0.1"):
            ConformalInterval(q_hat_init=-0.1)
        with pytest.raises(ValueError, match="q_hat_init must be non-negative, got nan"):
            ConformalInterval(q_hat_init=float("nan"))
        with pytest.raises(ValueError, match="domain_min must be below domain_max, got domain_min=1.0, domain_max=0.0"):
            ConformalInterval(domain_min=1.0, domain_max=0.0)

    def test_calibrate_refuses_a_negative_or_nan_radius_but_takes_infinity(self) -> None:
        layer = ConformalInterval()
        layer(np.zeros((1, 4), dtype="float32"))
        with pytest.raises(ValueError, match="q_hat must be non-negative, got -0.2"):
            layer.set_q_hat(-0.2)
        with pytest.raises(ValueError, match="q_hat must be non-negative, got nan"):
            layer.set_q_hat(float("nan"))
        layer.set_q_hat(float("inf"))
        mu_c, lower, upper = layer(np.array([[0.5]], dtype="float32"))
        assert np.isneginf(keras.ops.convert_to_numpy(lower)).all()
        assert np.isposinf(keras.ops.convert_to_numpy(upper)).all()

    def test_the_radius_is_stored_in_the_variable_dtype(self, dtype_policy) -> None:
        layer = ConformalInterval()
        layer(np.zeros((2, 3), dtype="float32"))
        want = "float64" if dtype_policy == "float64" else "float32"
        assert layer.q_hat.dtype == want
        layer.set_q_hat(0.0488)
        # a float16 variable would hold 0.04879760742...; the variable keeps float32
        assert float(keras.ops.convert_to_numpy(layer.q_hat)) == pytest.approx(0.0488, abs=1e-7)

    def test_forward_arms_are_finite_and_widen_by_the_radius(self, dtype_policy) -> None:
        layer = ConformalInterval()
        layer(np.zeros((2, 3), dtype="float32"))
        layer.set_q_hat(0.25)
        mu = np.array([[0.1, 0.5, 0.9]], dtype="float32")
        mu_c, lower, upper = [keras.ops.convert_to_numpy(t).astype("float64") for t in layer(mu)]
        assert np.isfinite(lower).all() and np.isfinite(upper).all()
        tol = 1e-3 if dtype_policy == "mixed_float16" else 1e-6
        np.testing.assert_allclose(upper - mu_c, 0.25, rtol=0, atol=tol)
        np.testing.assert_allclose(mu_c - lower, 0.25, rtol=0, atol=tol)

    def test_xla_matches_eager(self) -> None:
        layer = ConformalInterval()
        layer(np.zeros((1, 6), dtype="float32"))
        layer.set_q_hat(CAL_Q)
        mu = np.random.default_rng(1).uniform(-0.5, 1.5, size=(4, 6)).astype("float32")
        eager = [keras.ops.convert_to_numpy(t) for t in layer(mu)]
        fn = tf.function(lambda t: layer(t), jit_compile=True)
        for a, b in zip(eager, fn(tf.constant(mu))):
            np.testing.assert_allclose(a, b.numpy(), rtol=0, atol=1e-6)

    def test_a_trace_with_an_unknown_batch_matches_the_static_one(self) -> None:
        layer = ConformalInterval()
        layer(np.zeros((1, 6), dtype="float32"))
        layer.set_q_hat(CAL_Q)
        mu = np.random.default_rng(2).uniform(-0.5, 1.5, size=(5, 6)).astype("float32")
        static = keras.ops.convert_to_numpy(layer(mu)[1])
        fn = tf.function(lambda t: layer(t)[1], input_signature=[tf.TensorSpec([None, 6], tf.float32)])
        np.testing.assert_array_equal(fn(tf.constant(mu)).numpy(), static)

    def test_weights_are_identical_before_the_reloaded_model_is_called(self, tmp_path) -> None:
        inp = keras.Input(shape=(6,))
        model = keras.Model(inp, ConformalInterval()(inp))
        layer = next(l for l in model.layers if isinstance(l, ConformalInterval))
        layer.set_q_hat(CAL_Q)
        path = tmp_path / "w.keras"
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"ConformalInterval": ConformalInterval}
        )
        rel = next(l for l in loaded.layers if isinstance(l, ConformalInterval))
        assert [w.path.split("/")[-1] for w in rel.weights] == ["conformal_q_hat"]
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(layer.q_hat), keras.ops.convert_to_numpy(rel.q_hat)
        )

    def test_every_constructor_argument_is_in_the_config(self) -> None:
        cfg = ConformalInterval().get_config()
        for key in ("q_hat_init", "domain_min", "domain_max", "return_mu"):
            assert key in cfg
