"""CliffordCLIP's two final projections carry no bias, as OpenAI CLIP's do not.

OpenAI CLIP (``clip/model.py``) projects each tower with a bare matrix:
``x = x @ self.proj`` (``VisionTransformer.forward``) and
``x = x[...] @ self.text_projection`` (``CLIP.encode_text``). Until
plan-2026-10-14T062616-81938e81 step 9.2 both CliffordCLIP projections were
``Dense(embed_dim, use_bias=self.use_bias)``, so the default model (``use_bias=True``)
added a learned offset to every embedding before the L2 normalization.

``use_bias`` still governs every other Dense in the model; only these two are
fixed bias-free. The parameter delta is a hand sum against the count measured
at fa7499206 (before the change): exactly ``2 * embed_dim``.
"""

import numpy as np
import pytest

from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP

#: ``nano``, vocab 1000, 64 px, patch 4, context 16, ``use_bias=True``:
#: ``count_params()`` measured at fa7499206, with both projection biases.
COUNT_WITH_PROJECTION_BIASES = 7_360_833
#: ``nano``'s ``embed_dim``, its last vision stage width and its text width.
EMBED_DIM, VISION_WIDTH, TEXT_WIDTH = 256, 256, 128


def _nano(**overrides) -> CliffordCLIP:
    m = CliffordCLIP.from_variant(
        "nano", vocab_size=1000, image_size=64, context_length=16,
        vision_patch_size=4, **overrides)
    m({"image": np.zeros((1, 64, 64, 3), "float32"),
       "text": np.ones((1, 16), "int32")})
    return m


def _projection_weights(model: CliffordCLIP):
    return {
        w.path.split("/", 1)[1]: tuple(w.shape)
        for w in model.weights
        if w.path.split("/")[1] in ("vision_projection", "text_projection")
    }


@pytest.mark.parametrize("use_bias", [True, False])
def test_each_projection_is_one_bias_free_kernel(use_bias):
    assert _projection_weights(_nano(use_bias=use_bias)) == {
        "vision_projection/kernel": (VISION_WIDTH, EMBED_DIM),
        "text_projection/kernel": (TEXT_WIDTH, EMBED_DIM),
    }


def test_use_bias_still_reaches_the_other_dense_layers():
    """The fix is the two projections, not the model's ``use_bias``."""
    biased = [w.path for w in _nano(use_bias=True).weights if w.path.endswith("/bias")]
    assert biased, "use_bias=True built no bias anywhere"


def test_the_parameter_count_drops_by_exactly_two_embed_dims():
    assert _nano(use_bias=True).count_params() == (
        COUNT_WITH_PROJECTION_BIASES - 2 * EMBED_DIM)
