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

import inspect
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
from dl_techniques.layers.ffn import factory as factory_module
from dl_techniques.layers.ffn import tversky_projection as layer_module
from dl_techniques.layers.ffn.tversky_projection import (
    VALID_DIFFERENCE_REDUCTIONS,
    VALID_INTERSECTION_REDUCTIONS,
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
        # Both guards read the SAME two frozensets, which live in
        # tversky_projection.py; the factory imports them. The layer's own
        # __init__ guard is covered by TestTverskyReductionValidation below.
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


class TestTverskyReductionValidation:
    """Guards for C-10: the reduction names are validated at construction.

    At BASE both names were stored raw. A typo constructed, survived
    ``get_config()``, and surfaced only inside ``call()`` as
    ``NotImplementedError`` -- not the ``ValueError`` this package uses for
    configuration errors, and for a compiled model not until the graph trace.
    """

    @pytest.mark.parametrize(
        "bad_kwargs, needle",
        [
            ({'intersection_reduction': 'prodcut'}, 'intersection_reduction'),
            ({'intersection_reduction': 'maximum'}, 'intersection_reduction'),
            ({'difference_reduction': 'subtract_match'}, 'difference_reduction'),
            ({'difference_reduction': 'ignore-match'}, 'difference_reduction'),
        ],
    )
    def test_bad_reduction_raises_value_error_from_init(
        self, bad_kwargs: Dict[str, Any], needle: str
    ) -> None:
        """ValueError from __init__, NOT NotImplementedError from call()."""
        with pytest.raises(ValueError) as exc:
            TverskyProjectionLayer(units=4, num_features=8, **bad_kwargs)
        assert needle in str(exc.value)
        assert list(bad_kwargs.values())[0] in str(exc.value)

    def test_construction_is_what_fails_not_the_forward_pass(self) -> None:
        """Pins the TIMING, not only the exception type.

        A guard that only asserted ``pytest.raises(ValueError)`` around a
        construct-then-call block would pass at BASE too, since
        ``NotImplementedError`` would be raised by the call. This one proves
        no layer object is ever produced.
        """
        made = []
        try:
            made.append(
                TverskyProjectionLayer(
                    units=4, num_features=8, intersection_reduction='prodcut'
                )
            )
        except ValueError:
            pass
        assert made == [], "a layer with an invalid reduction was constructed"

    def test_every_valid_name_still_constructs_and_runs(self) -> None:
        """Anti-vacuity: the guard must not reject the supported values."""
        x = keras.random.normal((2, 6))
        for ir in sorted(VALID_INTERSECTION_REDUCTIONS):
            for dr in sorted(VALID_DIFFERENCE_REDUCTIONS):
                layer = TverskyProjectionLayer(
                    units=4,
                    num_features=8,
                    intersection_reduction=ir,
                    difference_reduction=dr,
                )
                y = layer(x)
                assert y.shape == (2, 4)
                assert not keras.ops.any(keras.ops.isnan(y))

    def test_the_factory_reads_the_layers_frozensets_not_a_copy(self) -> None:
        """SC10, static half: one object, two readers.

        The executed half of SC10 -- editing the frozenset and watching BOTH
        guards change behaviour -- is recorded in this plan's report; this
        assertion pins the identity the executed proof depends on, so a
        future re-inlining of a literal set into ``factory.py`` fails here.
        """
        assert factory_module.VALID_INTERSECTION_REDUCTIONS is (
            layer_module.VALID_INTERSECTION_REDUCTIONS
        )
        assert factory_module.VALID_DIFFERENCE_REDUCTIONS is (
            layer_module.VALID_DIFFERENCE_REDUCTIONS
        )
        src = inspect.getsource(factory_module.validate_ffn_config)
        assert "{'product', 'min', 'mean'}" not in src, (
            "factory.py has re-grown its own copy of the intersection set"
        )
        assert "{'ignorematch', 'subtractmatch'}" not in src, (
            "factory.py has re-grown its own copy of the difference set"
        )

    def test_both_guards_reject_the_same_name(self) -> None:
        """The layer guard and the factory guard agree, member for member."""
        for bad in ['prodcut', 'maximum', '', 'PRODUCT']:
            with pytest.raises(ValueError):
                TverskyProjectionLayer(
                    units=4, num_features=8, intersection_reduction=bad
                )
            with pytest.raises(ValueError):
                validate_ffn_config(
                    'tversky', units=4, num_features=4,
                    intersection_reduction=bad,
                )
