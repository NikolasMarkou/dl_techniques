"""Tests for the SSM factory: registry construction and strict kwargs."""

import pytest

from dl_techniques.layers.ssm import (
    SSM_REGISTRY,
    create_ssm_layer,
    create_ssm_from_config,
    list_ssm_types,
)


def test_registry_keys_are_pinned() -> None:
    assert sorted(SSM_REGISTRY.keys()) == ["context_mamba", "selective_ssm"]
    assert sorted(list_ssm_types()) == ["context_mamba", "selective_ssm"]


def test_create_selective_ssm_minimal() -> None:
    layer = create_ssm_layer("selective_ssm", d_model=16)
    assert layer.d_model == 16
    assert layer.d_state == 16


def test_create_context_mamba_minimal() -> None:
    layer = create_ssm_layer("context_mamba", d_model=16)
    assert layer.d_model == 16
    assert layer.normalization_type == "layer_norm"


def test_unknown_type_raises() -> None:
    with pytest.raises(ValueError):
        create_ssm_layer("definitely_not_ssm", d_model=16)  # type: ignore[arg-type]


def test_undeclared_key_raises_naming_key() -> None:
    with pytest.raises(ValueError, match="unsupported parameter"):
        create_ssm_layer("selective_ssm", d_model=16, bogus_key=1)


def test_missing_required_raises() -> None:
    with pytest.raises(ValueError):
        create_ssm_layer("selective_ssm")  # type: ignore[call-arg]


def test_from_config_round_trip() -> None:
    layer = create_ssm_from_config(
        {"type": "selective_ssm", "d_model": 16, "d_state": 8}
    )
    assert layer.d_state == 8
    layer2 = create_ssm_from_config({"type": "context_mamba", "d_model": 16})
    assert layer2.d_model == 16
