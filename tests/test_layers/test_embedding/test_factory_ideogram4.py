"""Factory-registration tests for the Ideogram4 embedding layers.

Covers the two newly-registered embedding types in
``dl_techniques.layers.embedding.factory``:

- ``'scalar_sinusoidal'`` -> ``ScalarSinusoidalEmbedding``
- ``'mrope_ideogram4'``   -> ``Ideogram4MRoPE``

Also spot-checks that an existing registry type still constructs (no
regression) and that the factory's required-param validation fires.
"""

import pytest

from dl_techniques.layers.embedding.factory import (
    create_embedding_layer,
    EMBEDDING_REGISTRY,
)
from dl_techniques.layers.embedding.multi_axis_rope import Ideogram4MRoPE
from dl_techniques.layers.embedding.scalar_sinusoidal_embedding import (
    ScalarSinusoidalEmbedding,
)
from dl_techniques.layers.embedding.rotary_position_embedding import (
    RotaryPositionEmbedding,
)


class TestIdeogram4EmbeddingFactory:
    def test_scalar_sinusoidal_constructs(self):
        layer = create_embedding_layer('scalar_sinusoidal', dim=32)
        assert isinstance(layer, ScalarSinusoidalEmbedding)
        assert layer.dim == 32

    def test_scalar_sinusoidal_forwards_input_range(self):
        layer = create_embedding_layer(
            'scalar_sinusoidal', dim=16, input_range=(-1.0, 1.0)
        )
        assert isinstance(layer, ScalarSinusoidalEmbedding)
        assert layer.range_min == -1.0
        assert layer.range_max == 1.0

    def test_mrope_ideogram4_constructs(self):
        layer = create_embedding_layer(
            'mrope_ideogram4',
            head_dim=32,
            rope_theta=10000,
            mrope_section=(4, 3, 3),
        )
        assert isinstance(layer, Ideogram4MRoPE)

    def test_missing_required_param_raises(self):
        # 'dim' is required for scalar_sinusoidal; omitting it must raise.
        with pytest.raises(ValueError):
            create_embedding_layer('scalar_sinusoidal')

    def test_mrope_missing_required_param_raises(self):
        with pytest.raises(ValueError):
            create_embedding_layer('mrope_ideogram4', head_dim=32, rope_theta=10000)

    def test_new_keys_registered(self):
        assert 'scalar_sinusoidal' in EMBEDDING_REGISTRY
        assert 'mrope_ideogram4' in EMBEDDING_REGISTRY

    def test_existing_type_still_builds(self):
        # Regression guard: an already-registered type still constructs.
        layer = create_embedding_layer('rope', head_dim=64, max_seq_len=128)
        assert isinstance(layer, RotaryPositionEmbedding)
