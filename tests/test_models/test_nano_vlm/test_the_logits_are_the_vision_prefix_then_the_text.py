"""NanoVLM's logits are `num_vision_tokens` vision positions, then the text.

Until plan-2026-10-14T062616-81938e81 step 9.1 (D-040) the output length
depended on which of 8 `MultiModalFusion` strategies was chosen (vision plus
text, text alone, or no sequence axis at all), and `generate()` refused seven
of them. The model now has one layout, the reference nanoVLM's: the pixel-
shuffled, projected vision tokens, then the text, through one causal decoder.
Pinned here against the actual forward output:

* the vision prefix length is `(grid / s)^2 + 1` (the CLS token survives the
  pixel shuffle), and `compute_output_shape`, the eager call and a symbolic
  `model(keras.Input(...))` agree on `Sv + T`;
* the text padding mask is spliced in AFTER an all-ones vision mask: an
  all-ones text mask changes nothing, and a padded text token is invisible
  to every later position;
* the text length limit is `text_config['max_seq_len']`, not reduced by the
  vision prefix, and a longer text is refused by name;
* `generate()` extends the prompt and refuses to run past the limit.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.nano_vlm import NanoVLM

VOCAB = 32
BATCH = 2
IMG = 32
PATCH = 8
GRID = IMG // PATCH
MAX_TEXT = 12


def _model(scale_factor: int = 2, **kwargs) -> NanoVLM:
    return NanoVLM.from_variant(
        "mini", vocab_size=VOCAB, output_dropout_rate=0.0,
        vision_config={"img_size": IMG, "patch_size": PATCH, "embed_dim": 16,
                       "depth": 1, "num_heads": 2},
        text_config={"embed_dim": 24, "depth": 2, "num_heads": 2,
                     "max_seq_len": MAX_TEXT},
        projection_config={"scale_factor": scale_factor},
        **kwargs,
    )


def _batch(text_len: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-1.0, 1.0, (BATCH, IMG, IMG, 3)).astype("float32"),
        rng.integers(1, VOCAB, (BATCH, text_len)).astype("int32"),
    )


@pytest.mark.parametrize("scale_factor", [1, 2])
def test_the_vision_prefix_length_follows_the_pixel_shuffle(scale_factor):
    model = _model(scale_factor)
    expected = (GRID // scale_factor) ** 2 + 1
    assert model.num_vision_tokens == expected
    images, text = _batch(text_len=7)
    logits = model({"images": images, "text_tokens": text}, training=False)
    assert tuple(logits.shape) == (BATCH, expected + 7, VOCAB)


def test_the_vision_and_text_widths_may_differ():
    """The projection maps the vision width (16) to the text width (24); the
    old fusion needed them equal."""
    model = _model()
    assert model.vision_config["embed_dim"] != model.text_config["embed_dim"]
    assert model.modality_projection.output_dim == model.text_config["embed_dim"]


def test_the_claim_the_eager_call_and_a_symbolic_call_agree():
    model = _model()
    images, text = _batch(text_len=7)
    eager = tuple(model({"images": images, "text_tokens": text}, training=False).shape)
    claimed = tuple(model.compute_output_shape({
        "images": (BATCH, IMG, IMG, 3), "text_tokens": (BATCH, 7)}))
    symbolic = tuple(model({
        "images": keras.Input((IMG, IMG, 3), batch_size=BATCH),
        "text_tokens": keras.Input((7,), batch_size=BATCH, dtype="int32"),
    }).shape)
    assert eager == claimed == symbolic == (BATCH, model.num_vision_tokens + 7, VOCAB)


def test_an_unknown_batch_size_is_carried_through():
    """The F-9 B10 row: the batch axis is never fixed by the model (the
    pre-D-040 fusion broadcast a batch-1 tensor); a symbolic ``None`` batch
    and the claimed shape both keep it ``None``."""
    model = _model()
    claimed = tuple(model.compute_output_shape({
        "images": (None, IMG, IMG, 3), "text_tokens": (None, 7)}))
    symbolic = tuple(model({
        "images": keras.Input((IMG, IMG, 3)),
        "text_tokens": keras.Input((7,), dtype="int32"),
    }).shape)
    assert claimed == symbolic == (None, model.num_vision_tokens + 7, VOCAB)
    for batch in (1, 3):
        images, text = _batch(text_len=7)
        images = np.repeat(images[:1], batch, axis=0)
        text = np.repeat(text[:1], batch, axis=0)
        out = model({"images": images, "text_tokens": text}, training=False)
        assert tuple(out.shape) == (batch, model.num_vision_tokens + 7, VOCAB)


def test_an_all_ones_text_mask_changes_nothing():
    """The vision prefix is spliced in as attended; a zero vision mask or a
    mask aligned to the wrong positions would move these logits."""
    model = _model()
    images, text = _batch(text_len=8)
    plain = np.asarray(model({"images": images, "text_tokens": text}, training=False))
    masked = np.asarray(model({"images": images, "text_tokens": text,
                               "attention_mask": np.ones_like(text)}, training=False))
    np.testing.assert_array_equal(plain, masked)


def test_a_padded_text_token_is_invisible_to_later_positions():
    model = _model()
    images, text = _batch(text_len=8)
    sv = model.num_vision_tokens
    padded = 3
    mask = np.ones_like(text)
    mask[:, padded] = 0
    edited = text.copy()
    edited[:, padded] = text[:, padded] % (VOCAB - 1) + 1

    def logits(tokens, attention_mask):
        inputs = {"images": images, "text_tokens": tokens}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        return np.asarray(model(inputs, training=False))

    later = slice(sv + padded + 1, None)
    np.testing.assert_array_equal(logits(text, mask)[:, later], logits(edited, mask)[:, later])
    # Control: without the mask the same edit does reach the later positions.
    assert np.all(np.max(np.abs(logits(text, None)[:, later]
                                - logits(edited, None)[:, later]), axis=-1) > 0.0)


def test_the_text_limit_is_the_text_config_max_seq_len():
    model = _model()
    assert model.text_config["max_seq_len"] == MAX_TEXT
    assert model.text_decoder.max_seq_len == model.num_vision_tokens + MAX_TEXT
    images, text = _batch(text_len=MAX_TEXT)
    model({"images": images, "text_tokens": text}, training=False)

    longer = _model()
    with pytest.raises(ValueError, match=rf"text_tokens has {MAX_TEXT + 1} positions"):
        longer.build({"images": (None, IMG, IMG, 3), "text_tokens": (None, MAX_TEXT + 1)})


def test_generate_extends_the_prompt_and_refuses_to_run_past_the_limit():
    model = _model()
    images, prompt = _batch(text_len=3)
    generated = keras.ops.convert_to_numpy(
        model.generate(images, prompt, max_length=4, top_k=0, eos_token_id=-1))
    assert generated.shape == (BATCH, 7)
    np.testing.assert_array_equal(generated[:, :3], prompt)

    # Greedy: the first new token is the argmax of the last text position.
    logits = np.asarray(model({"images": images, "text_tokens": prompt}, training=False))
    np.testing.assert_array_equal(generated[:, 3], np.argmax(logits[:, -1], axis=-1))

    with pytest.raises(ValueError, match="exceeds text_config"):
        model.generate(images, prompt, max_length=MAX_TEXT - 2)
