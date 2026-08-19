"""``GroupAttention``'s norm must run at the enclosing block's ``layer_norm_eps``.

This half of D-007 is a PRODUCT gap, not a dropped keyword: before 2026-08-19
``GroupAttention.__init__`` declared no epsilon parameter at all, so
``TreeTransformerBlock`` -- which stores ``layer_norm_eps`` and passes it to its
own ``norm1``/``norm2`` -- had no channel through which to reach it, and
``self.norm`` silently took ``create_normalization_layer``'s ``1e-6`` default.
No call-site guard could ever have seen this; the parameter had to exist first.
See decisions.md D-007 (``plan-2026-08-19-a616f581``).
"""

import numpy as np
import pytest

from dl_techniques.models.tree_transformer.components import (
    GroupAttention,
    TreeTransformerBlock,
)
from dl_techniques.models.tree_transformer.model import TreeTransformer
from tests.norm_epsilon_oracle import (
    assert_epsilon_tracks_the_knob,
    assert_every_block_norm_uses,
)

NUM_LAYERS = 2
FACTORY_DEFAULT_EPSILON = 1e-6


def _block(layer_norm_eps: float) -> TreeTransformerBlock:
    return TreeTransformerBlock(
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        layer_norm_eps=layer_norm_eps,
    )


def _model(layer_norm_eps: float) -> TreeTransformer:
    model = TreeTransformer(
        vocab_size=64,
        hidden_size=32,
        num_layers=NUM_LAYERS,
        num_heads=4,
        intermediate_size=64,
        max_len=16,
        layer_norm_eps=layer_norm_eps,
    )
    model(np.ones((1, 8), dtype="int32"))
    return model


class TestGroupAttentionEpsilon:
    def test_the_group_attention_norm_uses_the_blocks_epsilon(self):
        block = _block(1e-12)
        assert block.group_attn.norm.epsilon == pytest.approx(1e-12, rel=0)
        assert block.group_attn.norm.epsilon != FACTORY_DEFAULT_EPSILON

    def test_every_norm_in_the_block_agrees(self):
        """norm1, norm2 AND the group-attention norm -- the defect was a split."""
        block = _block(1e-12)
        assert_every_block_norm_uses([block], expected=1e-12)

    def test_the_assembled_model_wires_it_through(self):
        """A block-level fix is worthless if the model does not pass the knob."""
        model = _model(1e-12)
        assert_every_block_norm_uses(model.blocks, expected=1e-12)

    def test_the_epsilon_tracks_the_knob(self):
        assert_epsilon_tracks_the_knob(
            build=_block,
            blocks_of=lambda b: [b],
            first=1e-12,
            second=1e-3,
        )

    def test_the_epsilon_survives_a_config_round_trip(self):
        """An unserialized new parameter silently reverts on load."""
        original = GroupAttention(hidden_size=32, layer_norm_eps=1e-3)
        config = original.get_config()
        assert config["layer_norm_eps"] == pytest.approx(1e-3, rel=0)
        restored = GroupAttention.from_config(config)
        assert restored.norm.epsilon == pytest.approx(1e-3, rel=0)
