"""``get_sequence_pooling_info`` hands out copies the caller cannot leak back.

Guard for plan-2026-10-14T062616-81938e81 step 5.4 (D-203). Measured at
69b68fe5a: the per-entry ``.copy()`` was shallow, so
``info['attention']['optional_params']['hidden_dim'] = 999`` changed the
registry (read back 999) and every later factory default.
"""

from dl_techniques.layers.sequence_pooling import (
    SEQUENCE_POOLING_REGISTRY,
    create_sequence_pooling_layer,
    get_sequence_pooling_info,
)


def test_mutating_the_info_leaves_the_registry_and_the_factory_alone():
    original = SEQUENCE_POOLING_REGISTRY["attention"]["optional_params"]["hidden_dim"]
    info = get_sequence_pooling_info()
    info["attention"]["optional_params"]["hidden_dim"] = 999
    info["sequence"]["required_params"].append("bogus")
    assert SEQUENCE_POOLING_REGISTRY["attention"]["optional_params"]["hidden_dim"] == original
    assert SEQUENCE_POOLING_REGISTRY["sequence"]["required_params"] == []
    assert create_sequence_pooling_layer("attention").hidden_dim == original


def test_the_info_keeps_the_layer_classes_themselves():
    info = get_sequence_pooling_info()
    for name, entry in SEQUENCE_POOLING_REGISTRY.items():
        assert info[name]["class"] is entry["class"]
