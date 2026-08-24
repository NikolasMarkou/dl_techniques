"""BERT's ``layer_norm_eps`` must reach every in-block norm, not just embeddings.

Pre-fix (measured 2026-08-19 at ``num_layers=2``): the embedding norm ran at
BERT's own ``1e-12`` while all four encoder-block norms ran at
``create_normalization_layer``'s ``1e-6`` default -- a 1e6x split inside one
model, invisible to every shape/count assertion. See decisions.md D-007
(``plan-2026-08-19-a616f581``).
"""

import numpy as np
import pytest

from dl_techniques.models.bert.bert import BERT
from tests.norm_epsilon_oracle import (
    assert_epsilon_tracks_the_knob,
    assert_every_block_norm_uses,
    collect_norm_epsilons,
)

NUM_LAYERS = 2


def _build(layer_norm_eps: float) -> BERT:
    model = BERT(
        vocab_size=64,
        hidden_size=32,
        num_layers=NUM_LAYERS,
        num_heads=4,
        intermediate_size=64,
        layer_norm_eps=layer_norm_eps,
    )
    model(np.ones((1, 8), dtype="int32"))
    return model


class TestBertBlockNormEpsilon:
    def test_every_encoder_block_norm_uses_the_models_epsilon(self):
        model = _build(1e-12)
        assert_every_block_norm_uses(
            model.encoder_layers,
            expected=1e-12,
            expected_count=2 * NUM_LAYERS,
        )

    def test_the_embedding_norm_agrees_with_the_block_norms(self):
        """The defect was a SPLIT, so both halves are pinned to one value."""
        model = _build(1e-12)
        assert model.embeddings.layer_norm.epsilon == pytest.approx(1e-12, rel=0)
        block = collect_norm_epsilons(model.encoder_layers)
        assert {e for _, _, e in block} == {1e-12}

    def test_the_epsilon_tracks_the_knob(self):
        """Liveness: without this arm, a hardcoded 1e-12 would also pass."""
        assert_epsilon_tracks_the_knob(
            build=_build,
            blocks_of=lambda m: m.encoder_layers,
            first=1e-12,
            second=1e-3,
            expected_count=2 * NUM_LAYERS,
        )
