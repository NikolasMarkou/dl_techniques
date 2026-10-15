"""A NanoVLM text position's logits must not depend on LATER text tokens.

NanoVLM trains as a causal language model: `NanoVLMLoss` scores text position
`t` against token `t + 1`. The text decoder is causal, but the default
`'cross_attention'` fusion lets every modality attend to every other one, in
every block. From the second block on, a text position attends to vision
tokens that, one block earlier, attended to the WHOLE text sequence, so the
future leaks into the past and next-token training can copy instead of
predict.

MEASURED 2026-10-15 (plan-2026-10-14T062616-81938e81 step 9), eval mode,
dropout 0: changing text token 6 moves the logits at text positions 0..5 by
up to 8.4e-3 with 2 fusion blocks, and by exactly 0.0 with 1 block. Every
`MODEL_VARIANTS` row uses 3 to 8 blocks. Pinned as a strict xfail: the fix is
a fusion design change (vision must not attend to text in a causal model),
which step 9 reports and does not make.
"""

import numpy as np
import pytest

from dl_techniques.models.vision_language.nano_vlm import NanoVLM

VOCAB = 32
TEXT_LEN = 8
CHANGED = 6


def _model(num_fusion_layers: int) -> NanoVLM:
    return NanoVLM.from_variant(
        "mini", vocab_size=VOCAB, output_dropout_rate=0.0,
        vision_config={"img_size": 32, "embed_dim": 16, "depth": 1, "num_heads": 2},
        text_config={"embed_dim": 16, "depth": 1, "num_heads": 2, "max_seq_len": 16},
        fusion_config={"attention_config": {"num_heads": 2},
                       "num_fusion_layers": num_fusion_layers},
    )


def _earlier_positions_moved(num_fusion_layers: int) -> float:
    model = _model(num_fusion_layers)
    rng = np.random.default_rng(0)
    image = rng.uniform(-1.0, 1.0, (1, 32, 32, 3)).astype("float32")
    tokens = rng.integers(1, VOCAB, (1, TEXT_LEN)).astype("int32")
    changed = tokens.copy()
    changed[0, CHANGED] = tokens[0, CHANGED] % (VOCAB - 1) + 1
    assert changed[0, CHANGED] != tokens[0, CHANGED]

    a = np.asarray(model({"images": image, "text_tokens": tokens}, training=False))
    b = np.asarray(model({"images": image, "text_tokens": changed}, training=False))
    vision_len = a.shape[1] - TEXT_LEN
    text_a = a[:, vision_len:vision_len + CHANGED]
    text_b = b[:, vision_len:vision_len + CHANGED]
    # The changed position itself must move, or the probe proves nothing.
    assert np.any(a[:, vision_len + CHANGED] != b[:, vision_len + CHANGED])
    return float(np.max(np.abs(text_a - text_b)))


def test_one_fusion_block_is_causal():
    """The control: with one block no text position sees a later token."""
    assert _earlier_positions_moved(num_fusion_layers=1) == 0.0


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "MEASURED 2026-10-15: with 2 cross_attention fusion blocks a later text "
    "token moves earlier text positions' logits (8.4e-3) through the vision "
    "stream; open fusion design defect, reported in step 9"))
def test_two_fusion_blocks_are_causal():
    assert _earlier_positions_moved(num_fusion_layers=2) == 0.0
