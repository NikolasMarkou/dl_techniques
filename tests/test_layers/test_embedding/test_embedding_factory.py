"""Tests for the embedding factory: covers all registered keys, parameter
passthrough, validation, and config-driven construction."""

import math
import pytest
import keras

from dl_techniques.layers.embedding import (
    create_embedding_layer,
    create_embedding_from_config,
    validate_embedding_config,
)
from dl_techniques.layers.embedding.factory import (
    EMBEDDING_REGISTRY,
    get_embedding_info,
)

# Minimal valid construction kwargs for every registered key.
VALID_CFGS = {
    "patch_1d": dict(patch_size=4, embed_dim=16),
    "patch_2d": dict(patch_size=4, embed_dim=16),
    "positional_learned": dict(max_seq_len=32, dim=16),
    "rope": dict(head_dim=16, max_seq_len=32),
    "dual_rope": dict(head_dim=16, max_seq_len=32),
    "continuous_rope": dict(dim=64, ndim=3),
    "continuous_sincos": dict(dim=64, ndim=3),
    "bert_embeddings": dict(vocab_size=50, hidden_size=16,
                            max_position_embeddings=32, type_vocab_size=2),
    "modern_bert_embeddings": dict(vocab_size=50, hidden_size=16, type_vocab_size=2),
    "albert_factorized": dict(vocab_size=50, bottleneck_dim=8, output_dim=16),
    "positional_sine_2d": dict(num_pos_feats=8),
    "scalar_sinusoidal": dict(dim=64),
    "mrope_ideogram4": dict(head_dim=64, rope_theta=10000.0, mrope_section=(2, 3, 3)),
}


class TestEmbeddingFactory:

    def test_all_registry_keys_have_test_config(self):
        # Guards against adding a registry key without test coverage here.
        assert set(VALID_CFGS) == set(EMBEDDING_REGISTRY)

    @pytest.mark.parametrize("key", list(VALID_CFGS))
    def test_construct_all_keys(self, key):
        layer = create_embedding_layer(key, **VALID_CFGS[key])
        assert isinstance(layer, keras.layers.Layer)
        assert type(layer).__name__ == EMBEDDING_REGISTRY[key]["class"].__name__

    def test_name_passthrough(self):
        layer = create_embedding_layer("rope", head_dim=16, max_seq_len=32, name="my_rope")
        assert layer.name == "my_rope"

    def test_optional_param_passthrough(self):
        # modern_bert dropout_rate is an optional factory param (ctor has no default).
        l = create_embedding_layer("modern_bert_embeddings", vocab_size=50,
                                   hidden_size=16, type_vocab_size=2, dropout_rate=0.3)
        assert l.dropout_rate == 0.3
        # albert bottleneck honored.
        l2 = create_embedding_layer("albert_factorized", vocab_size=50,
                                    bottleneck_dim=8, output_dim=16)
        assert l2.bottleneck_dim == 8
        # positional_sine_2d temperature override.
        l3 = create_embedding_layer("positional_sine_2d", num_pos_feats=16, temperature=5000.0)
        assert l3.temperature == 5000.0

    def test_from_config(self):
        layer = create_embedding_from_config(
            {"type": "positional_sine_2d", "num_pos_feats": 16, "temperature": 5000.0})
        assert layer.num_pos_feats == 16 and layer.temperature == 5000.0

    def test_get_embedding_info_covers_registry(self):
        info = get_embedding_info()
        assert set(info) == set(EMBEDDING_REGISTRY)
        for entry in info.values():
            assert "required_params" in entry and "optional_params" in entry

    # ---- validation errors ------------------------------------------

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_embedding_layer("does_not_exist", dim=8)

    def test_missing_required_raises(self):
        with pytest.raises(ValueError):
            validate_embedding_config("albert_factorized", vocab_size=50)  # missing bottleneck/output

    def test_bad_value_raises(self):
        with pytest.raises(ValueError):
            validate_embedding_config("positional_sine_2d", temperature=-1.0)
        with pytest.raises(ValueError):
            validate_embedding_config("rope", head_dim=16, max_seq_len=32, rope_theta=-1.0)
        with pytest.raises(ValueError):
            validate_embedding_config("albert_factorized", vocab_size=50,
                                      bottleneck_dim=0, output_dim=16)
