"""Tests for FlowMatchingVelocityLoss (rectified-flow velocity MSE)."""

import keras
import numpy as np
import pytest

from dl_techniques.losses import FlowMatchingVelocityLoss


class TestFlowMatchingVelocityLoss:
    """Unit tests for the flow-matching velocity loss."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(1234)

    # ------------------------------------------------------------------ #
    # Correctness
    # ------------------------------------------------------------------ #
    def test_matches_numpy_mse(self, rng) -> None:
        """Loss equals numpy mean((y_pred - y_true)^2) under default reduction."""
        y_true = rng.normal(size=(8, 16)).astype("float32")
        y_pred = rng.normal(size=(8, 16)).astype("float32")

        loss = FlowMatchingVelocityLoss()
        got = float(loss(y_true, y_pred))

        # default reduction sum_over_batch_size == global mean over all elements
        expected = float(np.mean((y_pred - y_true) ** 2))
        assert got == pytest.approx(expected, abs=1e-6)

    def test_velocity_target_exact_prediction_is_zero(self, rng) -> None:
        """Predicting the exact velocity target x1-x0 gives ~0 loss."""
        x0 = rng.normal(size=(4, 32)).astype("float32")  # data
        x1 = rng.normal(size=(4, 32)).astype("float32")  # noise
        v_target = x1 - x0

        loss = FlowMatchingVelocityLoss()
        got = float(loss(v_target, v_target))
        assert got == pytest.approx(0.0, abs=1e-6)

    def test_multidim_feature_reduction(self, rng) -> None:
        """Works with rank-3 (B, L, C) velocity tensors."""
        y_true = rng.normal(size=(3, 5, 7)).astype("float32")
        y_pred = rng.normal(size=(3, 5, 7)).astype("float32")

        loss = FlowMatchingVelocityLoss()
        got = float(loss(y_true, y_pred))
        expected = float(np.mean((y_pred - y_true) ** 2))
        assert got == pytest.approx(expected, abs=1e-6)

    # ------------------------------------------------------------------ #
    # loss_weight
    # ------------------------------------------------------------------ #
    def test_loss_weight_scales(self, rng) -> None:
        """loss_weight scalar linearly scales the loss."""
        y_true = rng.normal(size=(6, 10)).astype("float32")
        y_pred = rng.normal(size=(6, 10)).astype("float32")

        base = float(FlowMatchingVelocityLoss()(y_true, y_pred))
        scaled = float(FlowMatchingVelocityLoss(loss_weight=3.0)(y_true, y_pred))
        assert scaled == pytest.approx(3.0 * base, abs=1e-6)

    # ------------------------------------------------------------------ #
    # Registration / export / serialization
    # ------------------------------------------------------------------ #
    def test_importable_from_package(self) -> None:
        """Class is importable from the losses package __init__."""
        from dl_techniques.losses import FlowMatchingVelocityLoss as Imported
        assert Imported is FlowMatchingVelocityLoss

    def test_keras_registered(self) -> None:
        """Class is registered with Keras serialization registry."""
        obj = keras.saving.get_registered_object(
            "dl_techniques.losses>FlowMatchingVelocityLoss"
        )
        assert obj is FlowMatchingVelocityLoss

    def test_get_config_round_trip(self) -> None:
        """from_config(get_config()) reproduces loss_weight/name/reduction."""
        loss = FlowMatchingVelocityLoss(
            loss_weight=2.5,
            time_weighting=False,
            name="my_flow_loss",
            reduction="sum",
        )
        config = loss.get_config()
        restored = FlowMatchingVelocityLoss.from_config(config)

        assert restored.loss_weight == 2.5
        assert restored.name == "my_flow_loss"
        assert restored.reduction == "sum"
        assert restored.time_weighting is False

    def test_round_trip_numerically_equivalent(self, rng) -> None:
        """Restored loss produces identical values."""
        y_true = rng.normal(size=(5, 12)).astype("float32")
        y_pred = rng.normal(size=(5, 12)).astype("float32")

        loss = FlowMatchingVelocityLoss(loss_weight=1.7)
        restored = FlowMatchingVelocityLoss.from_config(loss.get_config())
        assert float(restored(y_true, y_pred)) == pytest.approx(
            float(loss(y_true, y_pred)), abs=1e-6
        )

    def test_time_weighting_is_noop(self, rng) -> None:
        """time_weighting=True does not change the numerical loss."""
        y_true = rng.normal(size=(4, 8)).astype("float32")
        y_pred = rng.normal(size=(4, 8)).astype("float32")

        off = float(FlowMatchingVelocityLoss(time_weighting=False)(y_true, y_pred))
        on = float(FlowMatchingVelocityLoss(time_weighting=True)(y_true, y_pred))
        assert on == pytest.approx(off, abs=1e-6)

    # ------------------------------------------------------------------ #
    # Smoke: compile + train_on_batch
    # ------------------------------------------------------------------ #
    def test_compile_train_on_batch_finite(self, rng) -> None:
        """A tiny model compiles with the loss and trains one finite step."""
        model = keras.Sequential([
            keras.layers.Input(shape=(16,)),
            keras.layers.Dense(16),
        ])
        model.compile(optimizer="adam", loss=FlowMatchingVelocityLoss())

        x = rng.normal(size=(8, 16)).astype("float32")
        y = rng.normal(size=(8, 16)).astype("float32")  # velocity target
        history = model.train_on_batch(x, y, return_dict=True)
        assert np.isfinite(history["loss"])
