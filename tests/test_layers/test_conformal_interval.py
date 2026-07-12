"""Test suite for :class:`ConformalIntervalLayer`.

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

from dl_techniques.layers.conformal_interval import ConformalIntervalLayer

# ---------------------------------------------------------------------

ATOL = 1e-6
CAL_Q = 0.0488  # a representative non-zero conformal radius (crux value)


class TestConformalIntervalLayer:
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

    def test_construction(self) -> None:
        """Layer registers, is non-trainable, and has no trainable weights."""
        # Registered in the Keras serializable registry (default package
        # prefix "Custom" for a bare @register_keras_serializable()).
        assert (
            keras.saving.get_registered_object(
                "Custom>ConformalIntervalLayer"
            )
            is ConformalIntervalLayer
        )

        layer = ConformalIntervalLayer()
        assert layer.trainable is False
        assert layer.q is None  # not built yet

        # Build via a forward call.
        layer(np.zeros((1, 4), dtype="float32"))
        assert layer.q is not None
        assert len(layer.trainable_weights) == 0
        # The non-trainable conformal_q weight exists.
        assert len(layer.non_trainable_weights) == 1
        assert layer.non_trainable_weights[0].name == "conformal_q"

    # -- forward numeric ----------------------------------------------

    def test_default_domain_is_unit_interval(self) -> None:
        """The constructor defaults pin the ``[0, 1]`` denoiser domain."""
        layer = ConformalIntervalLayer()
        assert layer.domain_min == 0.0
        assert layer.domain_max == 1.0

    def test_forward_numeric(self, mu_outside: np.ndarray) -> None:
        """``call`` clips ``mu`` and returns ``(mu_c, mu_c-q, mu_c+q)``."""
        layer = ConformalIntervalLayer(q_init=0.1)
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
        layer = ConformalIntervalLayer()
        in_shape = (None, 8, 8, 3)
        out_shape = layer.compute_output_shape(in_shape)
        assert isinstance(out_shape, tuple)
        assert len(out_shape) == 3
        for s in out_shape:
            assert s == in_shape

    # -- calibrate ----------------------------------------------------

    def test_calibrate_then_forward(self, mu_outside: np.ndarray) -> None:
        """A fresh forward reflects the calibrated ``q``."""
        layer = ConformalIntervalLayer()
        layer(mu_outside)  # build
        layer.calibrate(CAL_Q)

        assert float(keras.ops.convert_to_numpy(layer.q)) == pytest.approx(CAL_Q, abs=ATOL)

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
        out = ConformalIntervalLayer()(inp)
        model = keras.Model(inp, out)

        # Locate the layer instance and calibrate to a non-zero value.
        layer = next(
            l for l in model.layers if isinstance(l, ConformalIntervalLayer)
        )
        layer.calibrate(CAL_Q)

        # Pre-save forward. The probe straddles BOTH bounds of the [0, 1] domain
        # so the round-trip is compared on a forward where both clips fired.
        x = np.random.uniform(-0.5, 1.5, size=(2, 8, 8, 3)).astype("float32")
        pre = [keras.ops.convert_to_numpy(t) for t in model(x)]

        save_path = tmp_path / "m.keras"
        model.save(save_path)
        reloaded = keras.models.load_model(
            save_path,
            custom_objects={"ConformalIntervalLayer": ConformalIntervalLayer},
        )

        rel_layer = next(
            l for l in reloaded.layers if isinstance(l, ConformalIntervalLayer)
        )
        reloaded_q = float(keras.ops.convert_to_numpy(rel_layer.q))
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
        layer = ConformalIntervalLayer()
        layer(mu_outside)  # build
        layer.calibrate(CAL_Q)

        cfg = layer.get_config()
        assert cfg["q_init"] == pytest.approx(CAL_Q, abs=ATOL)
        # The domain travels WITH the config (which is why a model saved before
        # the [-0.5,+0.5] -> [0,1] migration reloads with the OLD bounds; that is
        # intended and deliberately un-shimmed).
        assert cfg["domain_min"] == 0.0 and cfg["domain_max"] == 1.0

        layer2 = ConformalIntervalLayer.from_config(cfg)
        layer2(mu_outside)  # build layer2 so its weight is created from q_init
        assert float(keras.ops.convert_to_numpy(layer2.q)) == pytest.approx(
            CAL_Q, abs=ATOL
        )

    # -- return_mu=False ----------------------------------------------

    def test_return_mu_false(self, mu_outside: np.ndarray) -> None:
        """``return_mu=False`` returns a 2-tuple of correct shapes."""
        layer = ConformalIntervalLayer(q_init=0.1, return_mu=False)
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
