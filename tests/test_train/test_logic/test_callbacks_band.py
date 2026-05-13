"""Unit tests for train.logic.callbacks_band.StopOnAccuracyBand.

Pure synthetic-history tests — no Keras training is required. We drive the
callback's ``on_epoch_end`` directly with a tiny mock model that exposes a
``stop_training`` attribute (matching the keras.callbacks.Callback contract).

Plan: plan_2026-05-13_798d3a60.
"""

import pytest

from train.logic.callbacks_band import StopOnAccuracyBand


class _MockModel:
    """Minimal keras-model substitute: only ``stop_training`` is touched."""

    def __init__(self) -> None:
        self.stop_training = False


def _run_history(callback: StopOnAccuracyBand, history):
    """Drive `callback` through a list of {monitor_key: value} dicts."""
    model = _MockModel()
    callback.set_model(model)
    for epoch, logs in enumerate(history):
        if model.stop_training:
            break
        callback.on_epoch_end(epoch, logs)
    return model


class TestBandSemantics:
    def test_fires_on_entry_to_band(self):
        """Monitor crosses into [0.70, 0.95] at epoch 3 -> fires there."""
        cb = StopOnAccuracyBand("val_accuracy", 0.70, 0.95)
        history = [
            {"val_accuracy": 0.55},  # below band
            {"val_accuracy": 0.60},
            {"val_accuracy": 0.65},
            {"val_accuracy": 0.72},  # ENTRY
            {"val_accuracy": 0.85},  # would also be in band but we stopped
        ]
        model = _run_history(cb, history)
        assert cb.fired is True
        assert cb.band_epoch == 3
        assert cb.band_value == pytest.approx(0.72)
        assert model.stop_training is True

    def test_does_not_fire_on_overshoot(self):
        """Monitor jumps from below band to above band — no entry, no fire."""
        cb = StopOnAccuracyBand("val_accuracy", 0.70, 0.95)
        history = [
            {"val_accuracy": 0.30},
            {"val_accuracy": 0.40},
            {"val_accuracy": 0.50},
            {"val_accuracy": 0.99},  # above band, no entry
            {"val_accuracy": 1.00},
        ]
        model = _run_history(cb, history)
        assert cb.fired is False
        assert cb.band_epoch is None
        assert cb.band_value is None
        assert model.stop_training is False

    def test_inclusive_boundaries(self):
        """Exact boundary values (low and high) count as in-band."""
        cb_low = StopOnAccuracyBand("val_accuracy", 0.70, 0.95)
        m_low = _run_history(cb_low, [{"val_accuracy": 0.70}])
        assert cb_low.fired is True
        assert m_low.stop_training is True

        cb_high = StopOnAccuracyBand("val_accuracy", 0.70, 0.95)
        m_high = _run_history(cb_high, [{"val_accuracy": 0.95}])
        assert cb_high.fired is True
        assert m_high.stop_training is True


class TestEdgeCases:
    def test_missing_monitor_is_noop(self):
        """When the monitor key is absent, callback is a no-op (warns once)."""
        cb = StopOnAccuracyBand("val_accuracy", 0.70, 0.95)
        model = _run_history(cb, [{"loss": 0.5}, {"loss": 0.4}])
        assert cb.fired is False
        assert model.stop_training is False

    def test_fires_only_once(self):
        """After firing, subsequent on_epoch_end calls are ignored."""
        cb = StopOnAccuracyBand("val_accuracy", 0.70, 0.95)
        model = _MockModel()
        cb.set_model(model)
        cb.on_epoch_end(0, {"val_accuracy": 0.80})
        assert cb.fired is True
        first_epoch = cb.band_epoch
        # Force a 2nd call (simulating a misbehaving training loop) and
        # ensure the recorded band epoch does not move.
        cb.on_epoch_end(1, {"val_accuracy": 0.85})
        assert cb.band_epoch == first_epoch

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            StopOnAccuracyBand("val_accuracy", 0.70, 0.95, mode="window")

    def test_invalid_band_rejected(self):
        with pytest.raises(ValueError, match="low"):
            StopOnAccuracyBand("val_accuracy", 0.95, 0.70)
