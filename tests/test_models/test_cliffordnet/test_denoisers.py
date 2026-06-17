"""
Test suite for the CliffordNet denoiser family:
- CliffordNetDenoiser (denoiser.py)
- CliffordNetConditionalDenoiser (conditional_denoiser.py)
- CliffordNetConfidenceDenoiser (confidence_denoiser.py)

Covers construction (incl. ValueError validation), a forward pass, and the M2
full .keras save -> load -> identical-output round-trip. All three are exercised
in their unconditional (single-tensor input) mode for compactness.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.cliffordnet.denoiser import CliffordNetDenoiser
from dl_techniques.models.cliffordnet.conditional_denoiser import (
    CliffordNetConditionalDenoiser,
)
from dl_techniques.models.cliffordnet.confidence_denoiser import (
    CliffordNetConfidenceDenoiser,
)

IN_CH = 1
SPATIAL = 32


def _image(batch=2, channels=IN_CH):
    return np.random.default_rng(0).random(
        (batch, SPATIAL, SPATIAL, channels)).astype("float32")


def _round_trip_identical(model, x, tmp_path, name):
    before = keras.ops.convert_to_numpy(model(x, training=False))
    path = os.path.join(str(tmp_path), f"{name}.keras")
    model.save(path)
    reloaded = keras.models.load_model(path)
    after = keras.ops.convert_to_numpy(reloaded(x, training=False))
    # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
    np.testing.assert_allclose(before, after, atol=1e-4,
                               err_msg=f"{name} differs after .keras round-trip")


# ---------------------------------------------------------------------
# CliffordNetDenoiser
# ---------------------------------------------------------------------

class TestCliffordNetDenoiser:

    def _model(self):
        return CliffordNetDenoiser(channels=8, depth=2, in_channels=IN_CH)

    def test_forward_residual_shape(self):
        model = self._model()
        x = _image()
        y = model(x, training=False)
        assert y.shape == (2, SPATIAL, SPATIAL, IN_CH)

    def test_invalid_channels(self):
        with pytest.raises(ValueError):
            CliffordNetDenoiser(channels=0)

    def test_invalid_depth(self):
        with pytest.raises(ValueError):
            CliffordNetDenoiser(depth=0)

    def test_keras_round_trip(self, tmp_path):
        _round_trip_identical(self._model(), _image(), tmp_path, "clifford_denoiser")


# ---------------------------------------------------------------------
# CliffordNetConditionalDenoiser (unconditional mode)
# ---------------------------------------------------------------------

class TestCliffordNetConditionalDenoiser:

    def _model(self):
        return CliffordNetConditionalDenoiser(
            in_channels=IN_CH,
            level_channels=[8, 16, 32],
            level_blocks=[1, 1, 1],
        )

    def test_forward_shape(self):
        model = self._model()
        x = _image()
        y = model(x, training=False)
        assert y.shape == (2, SPATIAL, SPATIAL, IN_CH)

    def test_invalid_in_channels(self):
        with pytest.raises(ValueError):
            CliffordNetConditionalDenoiser(in_channels=0)

    def test_keras_round_trip(self, tmp_path):
        _round_trip_identical(
            self._model(), _image(), tmp_path, "clifford_conditional_denoiser")


# ---------------------------------------------------------------------
# CliffordNetConfidenceDenoiser (unconditional mode)
# ---------------------------------------------------------------------

class TestCliffordNetConfidenceDenoiser:

    def _model(self):
        return CliffordNetConfidenceDenoiser(
            in_channels=IN_CH,
            level_channels=[8, 16, 32],
            level_blocks=[1, 1, 1],
        )

    def test_forward_outputs_uncertainty(self):
        model = self._model()
        x = _image()
        y = model(x, training=False)
        # Gaussian uncertainty -> [mu, log_var] => 2*in_channels output channels
        assert y.shape[:-1] == (2, SPATIAL, SPATIAL)
        assert y.shape[-1] == 2 * IN_CH

    def test_invalid_in_channels(self):
        with pytest.raises(ValueError):
            CliffordNetConfidenceDenoiser(in_channels=0)

    def test_keras_round_trip(self, tmp_path):
        _round_trip_identical(
            self._model(), _image(), tmp_path, "clifford_confidence_denoiser")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
