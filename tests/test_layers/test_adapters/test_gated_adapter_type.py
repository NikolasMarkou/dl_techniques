"""`GatedAdapter.adapter_type` has one member and the layer says so."""

import pytest

from dl_techniques.layers.adapters import GatedAdapter


def test_the_only_adapter_type_builds_and_round_trips():
    adapter = GatedAdapter(input_dim=8, output_dim=8, rank=2, alpha=4.0)
    assert adapter.adapter_type == "local_support"
    assert GatedAdapter.from_config(adapter.get_config()).adapter_type == "local_support"


@pytest.mark.parametrize("bad", ["foo", "lora", "", None])
def test_an_unknown_adapter_type_is_refused(bad):
    """It used to be stored and serialized without ever being read, so a typo
    built a perfectly good local-support adapter."""
    with pytest.raises(ValueError, match="adapter_type"):
        GatedAdapter(input_dim=8, output_dim=8, rank=2, alpha=4.0, adapter_type=bad)
