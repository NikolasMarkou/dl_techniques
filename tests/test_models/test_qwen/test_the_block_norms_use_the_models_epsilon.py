"""Qwen3's ``norm_eps`` must reach every in-block norm (F-10).

The trap that made this defect survive: ``norm_eps``'s DEFAULT is ``1e-6``,
which is EXACTLY ``create_normalization_layer``'s own default, so at default
construction the broken model reads correct (measured pre-fix: 4 of 4 block
norms at 1e-06, "matching" the knob by coincidence). Only a non-default knob
exposes it -- pre-fix at ``norm_eps=1e-3``: 0 of 4. Every assertion here
therefore uses a non-default value. See decisions.md D-007
(``plan-2026-08-19-a616f581``).
"""

import numpy as np
import pytest

from dl_techniques.models.qwen.qwen3 import Qwen3
from tests.norm_epsilon_oracle import (
    assert_epsilon_tracks_the_knob,
    assert_every_block_norm_uses,
)

NUM_LAYERS = 2
FACTORY_DEFAULT_EPSILON = 1e-6


def _build(norm_eps: float) -> Qwen3:
    model = Qwen3(
        vocab_size=64,
        hidden_size=32,
        num_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_seq_len=16,
        norm_eps=norm_eps,
    )
    model(np.ones((1, 8), dtype="int32"))
    return model


class TestQwen3BlockNormEpsilon:
    def test_every_block_norm_uses_the_models_epsilon(self):
        """Uses 1e-3, NOT the 1e-6 default -- see this module's docstring."""
        model = _build(1e-3)
        assert_every_block_norm_uses(
            model.blocks, expected=1e-3, expected_count=2 * NUM_LAYERS
        )
        assert 1e-3 != FACTORY_DEFAULT_EPSILON, (
            "this guard is only discriminating at a NON-default knob"
        )

    def test_the_final_norm_agrees_with_the_block_norms(self):
        model = _build(1e-3)
        assert model.final_norm.epsilon == pytest.approx(1e-3, rel=0)

    def test_the_epsilon_tracks_the_knob(self):
        assert_epsilon_tracks_the_knob(
            build=_build,
            blocks_of=lambda m: m.blocks,
            first=1e-3,
            second=1e-8,
            expected_count=2 * NUM_LAYERS,
        )
