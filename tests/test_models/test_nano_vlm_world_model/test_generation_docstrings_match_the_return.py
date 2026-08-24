"""F-25: the two generation docstrings described a return the code never produced.

Both entry points on :class:`ScoreBasedNanoVLM` documented a richer object than they
hand back, and the gap is one decoder each:

* ``generate_from_text`` said ``Generated images [batch, H, W, C]``. The reverse
  diffusion runs in the VISION ENCODER'S feature space and the method's own inline
  comment reads ``Decode latents to images (this would need a decoder) / For now,
  return the latent representation``. The return is rank-3 encoder latents.
* ``generate_from_image`` said ``Generated text embeddings [batch, max_length,
  text_dim]``. Its last two statements apply ``text_decoder_head`` and
  ``ops.argmax(..., axis=-1)``, so the ``text_dim`` axis is already collapsed and the
  return is rank-2 INTEGER token ids.

The shape of ``generate_from_text``'s return was in fact already asserted by
``test_generate_from_text.py`` -- ``(2, 5, EMBED_DIM)``, plainly rank 3 -- for two
whole plan steps, next to a docstring promising ``[batch, H, W, C]``. A behavioural
assertion alone therefore cannot see this defect class, which is why each test below
pairs the MEASURED return with a check on the prose that describes it.

RED-proof: restoring either original ``Returns:`` line fails the corresponding
``*_docstring`` test while every behavioural assertion here stays green -- and
changing the CODE to actually emit images/embeddings fails the behavioural test while
the docstring test stays green. The two arms are independently RED.
"""

import inspect
import re

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.nano_vlm_world_model.model import ScoreBasedNanoVLM

IMG_SIZE = 32
PATCH = 16
EMBED_DIM = 64
VOCAB = 64
MAX_LEN = 8
BATCH = 2
# 32/16 = 2x2 patches, + 1 CLS token.
N_TOKENS = (IMG_SIZE // PATCH) ** 2 + 1


def _model(generation_mode):
    """Seeded tiny model; no stochastic rates are configurable on this path."""
    keras.utils.set_random_seed(1234)
    return ScoreBasedNanoVLM(
        vision_config={
            'img_size': IMG_SIZE, 'patch_size': PATCH, 'embed_dim': EMBED_DIM,
            'depth': 2, 'num_heads': 4, 'output_mode': 'none',
        },
        text_config={
            'vocab_size': VOCAB, 'embed_dim': EMBED_DIM,
            'depth': 2, 'num_heads': 4, 'max_seq_len': 32,
        },
        diffusion_config={'num_timesteps': 100, 'beta_schedule': 'cosine'},
        vocab_size=VOCAB,
        generation_mode=generation_mode,
    )


def _returns_clause(fn):
    """The text of the ``Returns:`` block of ``fn``'s docstring, lowercased."""
    doc = inspect.getdoc(fn)
    assert doc is not None, f"{fn.__qualname__} has no docstring"
    m = re.search(r"^Returns:\n(.*?)(?:\n\S|\Z)", doc, re.S | re.M)
    assert m is not None, f"{fn.__qualname__} has no `Returns:` block"
    return " ".join(m.group(1).split()).lower()


class TestGenerateFromTextReturnsLatentsNotImages:

    def test_the_return_is_rank_3_encoder_latents(self):
        """MEASURED (CPU/GPU-agnostic, shape only): (2, 5, 64) -- rank 3, not rank 4."""
        model = _model('text_to_image')
        text = keras.random.normal((BATCH, 8, EMBED_DIM), seed=7)
        out = model.generate_from_text(text, num_inference_steps=2)
        shape = tuple(ops.shape(out))

        assert len(shape) == 3, (
            f"generate_from_text returned rank {len(shape)} {shape}; a pixel grid "
            f"[batch, H, W, C] would be rank 4. No latent->pixel decoder exists."
        )
        assert shape == (BATCH, N_TOKENS, EMBED_DIM), shape
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))

    def test_the_docstring_does_not_promise_images(self):
        """The prose arm. RED against the pre-fix `Generated images [batch, H, W, C]`."""
        clause = _returns_clause(ScoreBasedNanoVLM.generate_from_text)

        assert "image" not in clause, (
            f"generate_from_text's `Returns:` still promises images: {clause!r}. "
            f"It returns rank-3 vision-encoder latents; there is no decoder."
        )
        assert "h, w, c" not in clause, clause
        assert "latent" in clause, clause


class TestGenerateFromImageReturnsTokenIdsNotEmbeddings:

    def test_the_return_is_rank_2_integer_token_ids(self):
        """MEASURED: (2, 8) int -- `text_decoder_head` + argmax already ran."""
        model = _model('image_to_text')
        vision = keras.random.normal((BATCH, N_TOKENS, EMBED_DIM), seed=7)
        out = model.generate_from_image(
            vision, num_inference_steps=2, max_length=MAX_LEN
        )
        shape = tuple(ops.shape(out))

        assert len(shape) == 2, (
            f"generate_from_image returned rank {len(shape)} {shape}; a text "
            f"embedding [batch, max_length, text_dim] would be rank 3."
        )
        assert shape == (BATCH, MAX_LEN), shape

        arr = ops.convert_to_numpy(out)
        assert np.issubdtype(arr.dtype, np.integer), (
            f"dtype {arr.dtype} is not integer; argmax token ids are expected"
        )
        assert arr.min() >= 0 and arr.max() < VOCAB, (arr.min(), arr.max())

    def test_the_docstring_does_not_promise_embeddings(self):
        """RED against the pre-fix `Generated text embeddings [batch, max_length, text_dim]`."""
        clause = _returns_clause(ScoreBasedNanoVLM.generate_from_image)

        assert "embedding" not in clause, (
            f"generate_from_image's `Returns:` still promises embeddings: "
            f"{clause!r}. It returns argmax token ids; the text_dim axis is gone."
        )
        assert "text_dim" not in clause, clause
        assert "token" in clause, clause
