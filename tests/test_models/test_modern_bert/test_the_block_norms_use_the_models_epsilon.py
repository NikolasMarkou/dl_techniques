"""ModernBERT's ``layer_norm_eps`` must reach every in-block norm.

Pre-fix (measured 2026-08-19 at ``num_layers=2``): ``final_layer_norm`` ran at
ModernBERT's ``1e-12`` while all 4 encoder-block norms ran at
``create_normalization_layer``'s ``1e-6`` default. See decisions.md D-007
(``plan-2026-08-19-a616f581``).
"""

import numpy as np
import pytest

from dl_techniques.models.modern_bert.model import ModernBERT
from tests.norm_epsilon_oracle import (
    assert_epsilon_tracks_the_knob,
    assert_every_block_norm_uses,
)

NUM_LAYERS = 2


def _build(layer_norm_eps: float) -> ModernBERT:
    model = ModernBERT(
        vocab_size=64,
        hidden_size=32,
        num_layers=NUM_LAYERS,
        num_heads=4,
        intermediate_size=64,
        layer_norm_eps=layer_norm_eps,
    )
    model(np.ones((1, 8), dtype="int32"))
    return model


class TestModernBertBlockNormEpsilon:
    def test_every_encoder_block_norm_uses_the_models_epsilon(self):
        model = _build(1e-12)
        assert_every_block_norm_uses(
            model.encoder_layers, expected=1e-12, expected_count=2 * NUM_LAYERS
        )

    def test_the_final_norm_agrees_with_the_block_norms(self):
        """The defect was a SPLIT between the final norm and the block norms."""
        model = _build(1e-12)
        assert model.final_norm.epsilon == pytest.approx(1e-12, rel=0)

    def test_the_epsilon_tracks_the_knob(self):
        assert_epsilon_tracks_the_knob(
            build=_build,
            blocks_of=lambda m: m.encoder_layers,
            first=1e-12,
            second=1e-3,
            expected_count=2 * NUM_LAYERS,
        )
