"""
Test suite for the normalization factory (`norms/factory.py`).

Covers:
- construct-all: every NormalizationType key builds a usable keras Layer.
- F1 (plan_2026-06-15_2485b951): validate_normalization_config accepts the
  band/GRN initializer+regularizer params (previously false-rejected).
- create_normalization_from_config round-trip.
- known validation failures still raise.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.norms.factory import (
    create_normalization_layer,
    create_normalization_from_config,
    validate_normalization_config,
    get_normalization_info,
)

ALL_TYPES = [
    "layer_norm", "batch_norm", "rms_norm", "zero_centered_rms_norm",
    "zero_centered_band_rms_norm", "zero_centered_adaptive_band_rms_norm",
    "band_rms", "adaptive_band_rms", "band_logit_norm", "global_response_norm",
    "logit_norm", "max_logit_norm", "decoupled_max_logit", "dml_plus_focal",
    "dml_plus_center", "dynamic_tanh", "energy_layer_norm",
]


class TestFactoryConstruction:
    @pytest.mark.parametrize("norm_type", ALL_TYPES)
    def test_construct_all(self, norm_type):
        layer = create_normalization_layer(norm_type, name=f"n_{norm_type}")
        assert isinstance(layer, keras.layers.Layer)

    def test_info_covers_all_types(self):
        info = get_normalization_info()
        # every factory-dispatchable type except the two Keras built-ins+aliases
        # is described; all 16 keys should be present.
        for t in ALL_TYPES:
            assert t in info, f"{t} missing from get_normalization_info()"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_normalization_layer("does_not_exist")


class TestF1ValidateWhitelist:
    """Pins F1: regularizer/initializer params must be accepted by validate."""

    def test_band_rms_accepts_regularizer(self):
        assert validate_normalization_config(
            "band_rms",
            max_band_width=0.1,
            band_initializer="zeros",
            band_regularizer=keras.regularizers.L2(1e-5),
        )

    def test_adaptive_band_rms_accepts_regularizer(self):
        assert validate_normalization_config(
            "adaptive_band_rms",
            band_regularizer=keras.regularizers.L1(1e-4),
        )

    def test_grn_accepts_regularizers(self):
        assert validate_normalization_config(
            "global_response_norm",
            gamma_regularizer=keras.regularizers.L2(1e-5),
            beta_regularizer=None,
            activity_regularizer=None,
        )

    def test_invalid_param_still_rejected(self):
        with pytest.raises(ValueError):
            validate_normalization_config("dynamic_tanh", epsilon=1e-6)


class TestFromConfig:
    def test_from_config_roundtrip(self):
        config = {
            "type": "zero_centered_band_rms_norm",
            "max_band_width": 0.08,
            "epsilon": 1e-6,
            "axis": -1,
        }
        layer = create_normalization_from_config(config)
        x = ops.convert_to_tensor(np.random.randn(4, 16).astype("float32"))
        assert tuple(layer(x).shape) == (4, 16)

    def test_missing_type_raises(self):
        with pytest.raises(KeyError):
            create_normalization_from_config({"max_band_width": 0.1})
