"""ModernBERT's ``layer_norm_eps`` must reach every in-block norm.

Pre-fix (measured 2026-08-19 at ``num_layers=2``): ``final_layer_norm`` ran at
ModernBERT's ``1e-12`` while all 4 encoder-block norms ran at
``create_normalization_layer``'s ``1e-6`` default. See decisions.md D-007
(``plan-2026-08-19-a616f581``).

``local_attention_window_size=4`` is passed explicitly, matching every other
module in this directory. See the ``WINDOW`` constant below for why.
"""

import numpy as np
import pytest

from dl_techniques.models.modern_bert.model import ModernBERT
from tests.norm_epsilon_oracle import (
    assert_epsilon_tracks_the_knob,
    assert_every_block_norm_uses,
)

NUM_LAYERS = 2

# DECISION plan-2026-08-19T163559-499b6f0e/D-009
# `WINDOW` must stay small. Omitting it takes `ModernBERT.__init__`'s shipped
# default of 128, and the local layers are grid-window attention: they fold the
# sequence into a synthetic square grid and pad it to `window_size**2` slots
# INDEPENDENT of the input length. At 128 this file's `(1, 8)` input reached
# `SingleWindowAttention.call` as `(1, 16384, 32)` and the relative-position-bias
# `take` asked for a `[268435456, 4]` float32 tensor (~4.3 GB), one of several:
# `ResourceExhaustedError` on BOTH the 4070 (12 GB) and the 4090 (24 GB), i.e.
# deterministic, not contention. Measured 2026-08-19 at HEAD: 3 failed in 55.16s.
#
# WHAT NOT TO DO, and why:
#   * Do NOT restore the shipped 128 "so the test exercises the real config".
#     The property under test is a CONSTRUCTION-TIME attribute of the block
#     norms; `tests/norm_epsilon_oracle.py` filters sub-layers by
#     `"norm" in type(sub).__name__.lower()` and never touches attention. The
#     forward call exists only to trigger the lazy build. The window size is
#     incidental to the assertion, and shrinking it keeps the local `'window'`
#     branch on the same code path, just at a tractable size. RED-PROVEN at
#     WINDOW=4, in two injections, because the defect was a SPLIT and no single
#     injection can redden both halves:
#       (1) deleting the `attention_norm_args`/`ffn_norm_args` wiring at
#           `models/modern_bert/model.py:574-575` (the literal pre-fix state)
#           reddens `test_every_encoder_block_norm_uses_the_models_epsilon` and
#           `test_the_epsilon_tracks_the_knob` -- "4 of 4 in-block norms do not
#           use the model's own epsilon 1e-12", all four reading 1e-06.
#       (2) `test_the_final_norm_agrees_with_the_block_norms` asserts on the
#           half that was always CORRECT, so (1) leaves it green by design;
#           hardcoding `final_norm`'s epsilon to 1e-6 reddens it alone
#           ("assert 1e-06 == 1e-12"). Both injections were reverted.
#     So all 3 node ids are proven capable of failing at the smaller size.
#   * Do NOT `xfail`/`skipif` this on device memory instead. That was
#     considered and REJECTED: it would park a green-by-omission guard on the
#     one plan that owns the RED, and the OOM is not the property's cost.
#   * Do NOT read this as the ModernBERT `window_size` defect (D-027/D-019,
#     owned elsewhere). That item shrinks the PER-VARIANT sizes in
#     `MODEL_VARIANTS`; this file uses no variant, so that fix would leave
#     these three node ids RED. See decisions.md D-009.
WINDOW = 4


def _build(layer_norm_eps: float) -> ModernBERT:
    model = ModernBERT(
        vocab_size=64,
        hidden_size=32,
        num_layers=NUM_LAYERS,
        num_heads=4,
        intermediate_size=64,
        layer_norm_eps=layer_norm_eps,
        local_attention_window_size=WINDOW,
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
