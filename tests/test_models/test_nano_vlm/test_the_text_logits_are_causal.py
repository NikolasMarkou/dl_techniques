"""A NanoVLM position's logits must not depend on LATER text tokens.

NanoVLM trains as a causal language model: `NanoVLMLoss` scores text position
`t` against token `t + 1`. Until plan-2026-10-14T062616-81938e81 step 9.1
(D-040) the text tower fed a symmetric `MultiModalFusion('cross_attention')`
in which vision tokens attended to the whole text, and from the second fusion
block on a text position read those vision tokens back, so a later token
moved earlier text logits (MEASURED 8.4e-3 at 2 blocks; the rows used 3 to
8). That was pinned here as a strict xfail.

The model is now the reference nanoVLM layout: projected vision tokens are a
prefix of ONE causal decoder over `[vision; text]`. Both directions are
asserted (GUIDE.md section 5): a later token moves nothing before it, by
exactly 0.0, at every changed position; and it does move its own and later
positions, and the image moves every text logit, so the probe cannot pass
by being blind. The tiny model has 3 decoder blocks (the leak needed 2), and
the 'mini' row is checked at its own depth (6) and 224px.
"""

import numpy as np
import pytest

from dl_techniques.models.vision_language.nano_vlm import NanoVLM

VOCAB = 32
TEXT_LEN = 8

#: 32px images in 8px patches: a 4x4 grid, pixel-shuffled 2x2 to 2x2 = 4
#: tokens, plus the CLS token.
TINY = dict(
    vision_config={"img_size": 32, "patch_size": 8, "embed_dim": 16, "depth": 1,
                   "num_heads": 2},
    text_config={"embed_dim": 16, "depth": 3, "num_heads": 2, "max_seq_len": 16},
)


def _tiny() -> NanoVLM:
    return NanoVLM.from_variant("mini", vocab_size=VOCAB, **TINY)


def _mini() -> NanoVLM:
    return NanoVLM.from_variant("mini", vocab_size=VOCAB)


def _inputs(img_size: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    image = rng.uniform(-1.0, 1.0, (1, img_size, img_size, 3)).astype("float32")
    tokens = rng.integers(1, VOCAB, (1, TEXT_LEN)).astype("int32")
    return image, tokens


def _logits(model, image, tokens):
    return np.asarray(model({"images": image, "text_tokens": tokens}, training=False))


def _assert_causal_at_every_position(model, img_size):
    image, tokens = _inputs(img_size)
    sv = model.num_vision_tokens
    base = _logits(model, image, tokens)
    assert base.shape == (1, sv + TEXT_LEN, VOCAB)

    for changed in range(TEXT_LEN):
        edited = tokens.copy()
        edited[0, changed] = tokens[0, changed] % (VOCAB - 1) + 1
        assert edited[0, changed] != tokens[0, changed]
        moved = _logits(model, image, edited)

        before = np.max(np.abs(moved[:, :sv + changed] - base[:, :sv + changed]),
                        initial=0.0)
        assert before == 0.0, (
            f"changing text token {changed} moved an earlier position's logits "
            f"by {before}")
        # The positive arm: the changed position itself, and every later one.
        after = np.max(np.abs(moved[:, sv + changed:] - base[:, sv + changed:]), axis=-1)
        assert np.all(after > 0.0), (changed, after)


def _assert_the_image_reaches_every_text_position(model, img_size):
    image, tokens = _inputs(img_size)
    other_image, _ = _inputs(img_size, seed=1)
    sv = model.num_vision_tokens
    a = _logits(model, image, tokens)[:, sv:]
    b = _logits(model, other_image, tokens)[:, sv:]
    assert np.all(np.max(np.abs(a - b), axis=-1) > 0.0)


def test_a_three_block_model_is_causal_at_every_text_position():
    model = _tiny()
    assert len(model.text_decoder.decoder_layers) == 3
    assert model.num_vision_tokens == 5
    _assert_causal_at_every_position(model, img_size=32)


def test_the_image_reaches_every_text_position():
    _assert_the_image_reaches_every_text_position(_tiny(), img_size=32)


@pytest.mark.parametrize("probe", ["causal", "image_reaches_text"])
def test_the_mini_row_is_causal_at_its_own_depth(probe):
    """The row itself, 224px and 6 decoder blocks; only the vocabulary is small."""
    model = _mini()
    assert len(model.text_decoder.decoder_layers) == 6
    assert model.num_vision_tokens == 50
    if probe == "causal":
        _assert_causal_at_every_position(model, img_size=224)
    else:
        _assert_the_image_reaches_every_text_position(model, img_size=224)
