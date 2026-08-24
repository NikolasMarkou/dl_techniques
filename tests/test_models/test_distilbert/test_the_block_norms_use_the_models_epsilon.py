"""DistilBERT's ``layer_norm_eps`` must reach every in-block norm.

Pre-fix (measured 2026-08-19 at ``num_layers=2``): 4 of 4 block norms ran at
``create_normalization_layer``'s ``1e-6`` default instead of DistilBERT's
``1e-12``. See decisions.md D-007 (``plan-2026-08-19-a616f581``).
"""

import numpy as np

from dl_techniques.models.distilbert.model import DistilBERT
from tests.norm_epsilon_oracle import (
    assert_epsilon_tracks_the_knob,
    assert_every_block_norm_uses,
)

NUM_LAYERS = 2


def _build(layer_norm_eps: float) -> DistilBERT:
    model = DistilBERT(
        vocab_size=64,
        hidden_size=32,
        num_layers=NUM_LAYERS,
        num_heads=4,
        intermediate_size=64,
        layer_norm_eps=layer_norm_eps,
    )
    model(np.ones((1, 8), dtype="int32"))
    return model


class TestDistilBertBlockNormEpsilon:
    def test_every_encoder_block_norm_uses_the_models_epsilon(self):
        model = _build(1e-12)
        assert_every_block_norm_uses(
            model.encoder_layers, expected=1e-12, expected_count=2 * NUM_LAYERS
        )

    def test_the_epsilon_tracks_the_knob(self):
        assert_epsilon_tracks_the_knob(
            build=_build,
            blocks_of=lambda m: m.encoder_layers,
            first=1e-12,
            second=1e-3,
            expected_count=2 * NUM_LAYERS,
        )
