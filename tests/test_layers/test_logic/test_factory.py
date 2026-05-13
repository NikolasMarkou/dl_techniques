"""Tests for `dl_techniques.layers.logic.factory`."""
import os
import tempfile
import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.layers.logic import (
    CircuitDepthLayer,
    LearnableArithmeticOperator,
    LearnableLogicOperator,
    LearnableNeuralCircuit,
    LOGIC_REGISTRY,
    create_logic_from_config,
    create_logic_layer,
    get_logic_info,
    validate_logic_config,
)


class TestRegistry:
    """Static registry contents."""

    def test_registry_keys(self):
        assert set(LOGIC_REGISTRY.keys()) == {
            "arithmetic",
            "logic",
            "circuit_depth",
            "neural_circuit",
        }

    def test_registry_classes(self):
        assert LOGIC_REGISTRY["arithmetic"]["class"] is LearnableArithmeticOperator
        assert LOGIC_REGISTRY["logic"]["class"] is LearnableLogicOperator
        assert LOGIC_REGISTRY["circuit_depth"]["class"] is CircuitDepthLayer
        assert LOGIC_REGISTRY["neural_circuit"]["class"] is LearnableNeuralCircuit

    def test_registry_entries_complete(self):
        for layer_type, info in LOGIC_REGISTRY.items():
            assert "class" in info
            assert "description" in info
            assert "required_params" in info
            assert "optional_params" in info
            assert "use_case" in info
            assert isinstance(info["required_params"], list)
            assert isinstance(info["optional_params"], dict)

    def test_get_logic_info_returns_copy(self):
        info = get_logic_info()
        info["arithmetic"]["description"] = "MUTATED"
        # Original registry untouched
        assert LOGIC_REGISTRY["arithmetic"]["description"] != "MUTATED"


class TestPublicImport:
    """Smoke test for package-level imports."""

    def test_top_level_imports(self):
        # SC4 verbatim
        from dl_techniques.layers.logic import (  # noqa: F401
            LearnableNeuralCircuit,
            LearnableLogicOperator,
            LearnableArithmeticOperator,
            CircuitDepthLayer,
            create_logic_layer,
        )


class TestCreate:
    """Factory construction per layer type."""

    @pytest.mark.parametrize(
        "layer_type,expected_class",
        [
            ("arithmetic", LearnableArithmeticOperator),
            ("logic", LearnableLogicOperator),
            ("circuit_depth", CircuitDepthLayer),
            ("neural_circuit", LearnableNeuralCircuit),
        ],
    )
    def test_create_each_type_default(self, layer_type, expected_class):
        layer = create_logic_layer(layer_type)
        assert isinstance(layer, expected_class)
        assert not layer.built

    def test_kwargs_forwarded(self):
        layer = create_logic_layer(
            "circuit_depth",
            num_logic_ops=3,
            num_arithmetic_ops=5,
            use_residual=False,
            name="custom_cd",
        )
        assert layer.num_logic_ops == 3
        assert layer.num_arithmetic_ops == 5
        assert layer.use_residual is False
        assert layer.name == "custom_cd"

    def test_unknown_params_filtered(self):
        # Bogus kwarg must not raise — factory filters unknowns.
        layer = create_logic_layer(
            "arithmetic",
            operation_types=["add", "multiply"],
            this_is_not_a_real_param=42,
        )
        assert layer.operation_types == ["add", "multiply"]

    def test_invalid_layer_type_raises(self):
        with pytest.raises(ValueError, match="Unknown logic layer type"):
            create_logic_layer("not_a_real_type")

    def test_validate_logic_config_negative_int(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_logic_config("circuit_depth", num_logic_ops=0)

    def test_validate_logic_config_negative_float(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_logic_config("arithmetic", temperature_init=-0.5)

    def test_create_from_config(self):
        cfg = {"type": "neural_circuit", "circuit_depth": 2, "use_residual": True}
        layer = create_logic_from_config(cfg)
        assert isinstance(layer, LearnableNeuralCircuit)
        assert layer.circuit_depth == 2
        assert layer.use_residual is True

    def test_create_from_config_missing_type(self):
        with pytest.raises(ValueError, match="must include a 'type' key"):
            create_logic_from_config({"circuit_depth": 4})

    def test_create_from_config_not_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            create_logic_from_config("not_a_dict")  # type: ignore[arg-type]


class TestForwardRoundTrip:
    """End-to-end: factory build → call → save → reload."""

    def _make_model(self, layer):
        # Use rank-3 input — exercises rank-relaxation paths too.
        inp = keras.Input(shape=(4, 8))
        out = layer(inp)
        return keras.Model(inp, out)

    @pytest.mark.parametrize(
        "layer_type,kwargs",
        [
            ("arithmetic", {"operation_types": ["add", "multiply"]}),
            ("logic", {"operation_types": ["and", "or"], "allow_unary_degenerate": True}),
            ("circuit_depth", {"num_logic_ops": 2, "num_arithmetic_ops": 2}),
            ("neural_circuit", {"circuit_depth": 2}),
        ],
    )
    def test_factory_layer_round_trip(self, layer_type, kwargs):
        layer = create_logic_layer(layer_type, **kwargs)
        model = self._make_model(layer)

        x = ops.convert_to_tensor(np.random.normal(0, 1, (2, 4, 8)).astype(np.float32))
        y1 = model(x)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y2 = reloaded(x)

        np.testing.assert_allclose(ops.convert_to_numpy(y1), ops.convert_to_numpy(y2), atol=1e-6)


# ---------------------------------------------------------------------------
# Regression tests added in plan_2026-05-13_e52a5ac8
# ---------------------------------------------------------------------------

import pytest as _pytest

class TestPlanE52a5ac8Factory:
    def test_validate_rejects_bool_as_int(self):
        """H10: bool is a Python int subclass; validator must reject it."""
        from dl_techniques.layers.logic.factory import validate_logic_config
        with _pytest.raises(ValueError, match="must be a positive integer"):
            validate_logic_config("circuit_depth", num_logic_ops=True)
        with _pytest.raises(ValueError, match="must be a positive integer"):
            validate_logic_config("neural_circuit", circuit_depth=False)

    def test_get_logic_info_returns_deepcopy(self):
        """H9: mutating returned dict must not affect the registry."""
        from dl_techniques.layers.logic.factory import get_logic_info
        info1 = get_logic_info()
        # Mutate nested
        info1["arithmetic"]["optional_params"]["use_temperature"] = "MUTATED"
        info1["arithmetic"]["description"] = "MUTATED"
        # Re-read fresh
        info2 = get_logic_info()
        assert info2["arithmetic"]["optional_params"]["use_temperature"] is True
        assert info2["arithmetic"]["description"] != "MUTATED"

    def test_factory_passes_apply_sigmoid(self):
        """C3 factory wiring: create_logic_layer must accept apply_sigmoid."""
        from dl_techniques.layers.logic.factory import create_logic_layer
        layer = create_logic_layer("logic", apply_sigmoid=False)
        assert layer.apply_sigmoid is False
        layer_default = create_logic_layer("logic")
        assert layer_default.apply_sigmoid is True
