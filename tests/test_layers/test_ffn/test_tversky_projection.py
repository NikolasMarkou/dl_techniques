"""
Tests for TverskyProjectionLayer (relocated to layers/ffn/).

Covers:
- Initialization (valid + invalid)
- Forward pass output shape on rank-2 input
- Serialization round-trip via get_config / from_config
- .keras save/load round-trip wrapped in a keras.Model
- Factory wiring (create_ffn_layer('tversky', ...))
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.ffn import (
    TverskyProjectionLayer,
    create_ffn_layer,
    validate_ffn_config,
    get_ffn_info,
)


class TestTverskyProjectionLayer:
    """Test suite for the relocated TverskyProjectionLayer."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        return {
            'units': 10,
            'num_features': 12,
        }

    @pytest.fixture
    def sample_input(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((4, 8)).astype('float32')

    # ---------- Initialization ----------

    def test_init_valid(self, basic_config: Dict[str, Any]) -> None:
        layer = TverskyProjectionLayer(**basic_config)
        assert layer.units == 10
        assert layer.num_features == 12
        assert layer.intersection_reduction == 'product'
        assert layer.difference_reduction == 'subtractmatch'

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {'units': 0, 'num_features': 4},
            {'units': -1, 'num_features': 4},
            {'units': 4, 'num_features': 0},
            {'units': 4, 'num_features': -1},
        ],
    )
    def test_init_invalid_dims(self, bad_kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            TverskyProjectionLayer(**bad_kwargs)

    def test_init_invalid_reductions(self) -> None:
        # The layer itself accepts the strings but `call()` raises NotImplementedError
        # for unknown reductions. The factory-level validate_ffn_config is the
        # authoritative gatekeeper — covered in test_factory.py.
        with pytest.raises(ValueError):
            validate_ffn_config(
                'tversky', units=4, num_features=4, intersection_reduction='nope'
            )
        with pytest.raises(ValueError):
            validate_ffn_config(
                'tversky', units=4, num_features=4, difference_reduction='nope'
            )

    # ---------- Forward pass ----------

    def test_forward_shape_rank2(
        self, basic_config: Dict[str, Any], sample_input: np.ndarray
    ) -> None:
        layer = TverskyProjectionLayer(**basic_config)
        x = ops.convert_to_tensor(sample_input)
        y = layer(x)
        assert tuple(y.shape) == (4, 10)

    def test_compute_output_shape(self, basic_config: Dict[str, Any]) -> None:
        layer = TverskyProjectionLayer(**basic_config)
        out_shape = layer.compute_output_shape((None, 8))
        assert tuple(out_shape) == (None, 10)

    # ---------- Serialization ----------

    def test_get_config_round_trip(self, basic_config: Dict[str, Any]) -> None:
        layer = TverskyProjectionLayer(**basic_config, name='tv_test')
        config = layer.get_config()
        rebuilt = TverskyProjectionLayer.from_config(config)
        assert rebuilt.units == layer.units
        assert rebuilt.num_features == layer.num_features
        assert rebuilt.intersection_reduction == layer.intersection_reduction
        assert rebuilt.difference_reduction == layer.difference_reduction

    def test_keras_save_load_round_trip(
        self, basic_config: Dict[str, Any], sample_input: np.ndarray
    ) -> None:
        inputs = keras.Input(shape=(8,), dtype='float32')
        outputs = TverskyProjectionLayer(**basic_config, name='tv_layer')(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        x = ops.convert_to_tensor(sample_input)
        y_before = model(x)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'tv_model.keras')
            model.save(path)
            reloaded = keras.models.load_model(path)
            y_after = reloaded(x)

        np.testing.assert_allclose(
            ops.convert_to_numpy(y_before),
            ops.convert_to_numpy(y_after),
            atol=1e-6,
        )

    # ---------- Factory wiring ----------

    def test_factory_creates_tversky(self) -> None:
        layer = create_ffn_layer('tversky', units=10, num_features=12)
        assert isinstance(layer, TverskyProjectionLayer)
        assert layer.units == 10
        assert layer.num_features == 12

    def test_factory_info_exposes_tversky(self) -> None:
        info = get_ffn_info()
        assert 'tversky' in info
        entry = info['tversky']
        assert 'units' in entry['required_params']
        assert 'num_features' in entry['required_params']
