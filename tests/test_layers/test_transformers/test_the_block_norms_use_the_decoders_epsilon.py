"""``TextDecoder``'s ``layer_norm_eps`` must reach every in-block norm.

Pre-fix (measured 2026-08-19 at ``depth=2``): ``embed_norm`` and ``final_norm``
ran at the decoder's own ``1e-12`` while all 4 block norms ran at
``create_normalization_layer``'s ``1e-6`` default. See decisions.md D-007
(``plan-2026-08-19-a616f581``).
"""

import numpy as np
import pytest

from dl_techniques.layers.transformers.text_decoder import TextDecoder
from tests.norm_epsilon_oracle import (
    assert_epsilon_tracks_the_knob,
    assert_every_block_norm_uses,
)

DEPTH = 2


def _build(layer_norm_eps: float) -> TextDecoder:
    layer = TextDecoder(
        vocab_size=64,
        embed_dim=32,
        depth=DEPTH,
        num_heads=4,
        max_seq_len=16,
        layer_norm_eps=layer_norm_eps,
    )
    layer(np.ones((1, 8), dtype="int32"))
    return layer


class TestTextDecoderBlockNormEpsilon:
    def test_every_decoder_block_norm_uses_the_decoders_epsilon(self):
        decoder = _build(1e-12)
        assert_every_block_norm_uses(
            decoder.decoder_layers, expected=1e-12, expected_count=2 * DEPTH
        )

    def test_the_embed_and_final_norms_agree_with_the_block_norms(self):
        decoder = _build(1e-12)
        assert decoder.embed_norm.epsilon == pytest.approx(1e-12, rel=0)
        assert decoder.final_norm.epsilon == pytest.approx(1e-12, rel=0)

    def test_the_epsilon_tracks_the_knob(self):
        assert_epsilon_tracks_the_knob(
            build=_build,
            blocks_of=lambda d: d.decoder_layers,
            first=1e-12,
            second=1e-3,
            expected_count=2 * DEPTH,
        )
