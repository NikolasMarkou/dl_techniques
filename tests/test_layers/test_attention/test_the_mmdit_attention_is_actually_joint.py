"""Guards: MMDiT joint attention must be joint, and must carry its scale.

Why this file exists
--------------------
Two mutations of ``mmdit_joint_attention.py`` pass all 22 of its existing tests,
both proven non-inert first:

1. Inserting a block-diagonal additive mask that removes ALL cross-modal
   attention. The layer then degenerates into two independent self-attention
   blocks -- exactly what its own docstring says the concatenation exists to
   avoid ("This concat IS the 'joint' in joint attention: after it there is a
   single attention problem, so image tokens can attend to text tokens and vice
   versa"). Nothing in the suite checked that the two streams interact.
2. Dropping the ``1/sqrt(head_dim)`` pre-softmax scale. No test checked the logit
   magnitude against any oracle.

Both guards below use oracles derived WITHOUT the implementation: a perturbation
probe for the first, a hand-computed softmax with every projection forced to the
identity for the second.

A third guard (plan-2026-10-14-81938e81 step 10.1, D-043) covers the final-block
layout. With ``context_pre_only=True`` the layer no longer builds a text query;
a numpy reference that still computes one, attends over all ``N`` query rows and
keeps the image rows (the layout before 10.1, and the diffusers one) must agree
with it on the image output.

TF32
----
The hand-computed oracle compares a GPU matmul against a numpy one at
``atol=1e-5``. On Ampere+ hardware TF32 tensor-core matmul carries roughly 1e-3
relative precision, and the test measured a 2.2e-3 mismatch on an RTX 4070 while
passing on CPU. This module therefore opts into the repo's existing
``tf32_disabled`` fixture, the same way ``test_linear_attention.py`` does, rather
than loosening the bound -- the discriminating signal against the scale-deleted
mutant is 0.385, so a loosened bound would still work, but a tight oracle that is
honest about its precision regime is worth more than a slack one.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.mmdit_joint_attention import (
    MMDiTJointAttention,
)

pytestmark = pytest.mark.usefixtures("tf32_disabled")

DIM = 32
HEADS = 4
IMAGE_TOKENS = 6
TEXT_TOKENS = 4


@pytest.fixture(name="streams")
def _streams():
    image = np.random.RandomState(0).randn(1, IMAGE_TOKENS, DIM).astype("float32")
    text = np.random.RandomState(1).randn(1, TEXT_TOKENS, DIM).astype("float32")
    return image, text


def test_perturbing_text_moves_the_image_output(streams):
    """The 'joint' property, one direction."""
    image, text = streams
    keras.utils.set_random_seed(0)
    layer = MMDiTJointAttention(dim=DIM, num_heads=HEADS)

    image_out, _ = layer([image, text])

    perturbed = text.copy()
    perturbed[0, 0, :] += 5.0
    image_out_after, _ = layer([image, perturbed])

    delta = np.abs(
        np.asarray(image_out_after) - np.asarray(image_out)
    ).max()
    assert delta > 1e-6, (
        "the IMAGE output did not move when a TEXT token was perturbed, so the "
        "two streams are not attending to each other and this layer is two "
        "independent self-attention blocks"
    )


def test_perturbing_image_moves_the_text_output(streams):
    """The 'joint' property, the other direction.

    Both directions are asserted because a mask that is block-diagonal in only
    one corner would satisfy a single-direction check.
    """
    image, text = streams
    keras.utils.set_random_seed(0)
    layer = MMDiTJointAttention(dim=DIM, num_heads=HEADS)

    _, text_out = layer([image, text])

    perturbed = image.copy()
    perturbed[0, 0, :] += 5.0
    _, text_out_after = layer([perturbed, text])

    delta = np.abs(np.asarray(text_out_after) - np.asarray(text_out)).max()
    assert delta > 1e-6, (
        "the TEXT output did not move when an IMAGE token was perturbed"
    )


def test_the_prescale_matches_a_hand_computed_oracle():
    """Pin the pre-softmax scale to exactly ``1/sqrt(head_dim)``.

    Hand-computable case: 1 head, ``head_dim == dim == 4``, 2 image + 2 text
    tokens, every projection forced to the identity so the layer reduces to a
    plain scaled-dot-product softmax over the concatenated stream. The expected
    values come from numpy, never from the layer.

    Why this can fail if the implementation is wrong: with the scale deleted the
    output matches the UNSCALED softmax instead, 0.385 away from this oracle.
    """
    dim, heads = 4, 1
    keras.utils.set_random_seed(0)
    layer = MMDiTJointAttention(
        dim=dim, num_heads=heads, qk_norm=False, use_bias=False
    )
    image = np.random.RandomState(11).randn(1, 2, dim).astype("float32")
    text = np.random.RandomState(12).randn(1, 2, dim).astype("float32")
    layer([image, text])  # build

    identity = np.eye(dim, dtype="float32")
    for projection in (
        layer.to_q,
        layer.to_k,
        layer.to_v,
        layer.to_out,
        layer.add_q_proj,
        layer.add_k_proj,
        layer.add_v_proj,
        layer.to_add_out,
    ):
        projection.set_weights([identity])

    image_out, text_out = layer([image, text])
    actual = np.concatenate(
        [np.asarray(image_out)[0], np.asarray(text_out)[0]], axis=0
    )

    tokens = np.concatenate([image[0], text[0]], axis=0)
    logits = (tokens @ tokens.T) * (dim ** -0.5)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    expected = weights @ tokens

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)


def _rms(x, scale, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * scale


def test_the_final_block_image_output_matches_the_full_joint_reference():
    """``context_pre_only`` drops the text query rows, not the image output.

    The reference is the OLD final-block computation written out in numpy:
    queries for ALL ``N_img + N_txt`` tokens, the text ones through their own
    projection and QK-norm (``ref_add_q`` / ``ref_norm_added_q``, which the new
    layer does not have), one joint softmax over all keys, then the image rows
    sliced out and projected. Every weight the two layouts share is copied
    into the layer.

    Why this can fail: an image query that stopped seeing the text keys and
    values, a wrong slice, or a lost QK-norm each move the image output by far
    more than ``1e-5``.
    """
    dim, heads, eps = 16, 2, 1e-6
    head_dim = dim // heads
    n_img, n_txt, batch = 5, 3, 2
    rs = np.random.RandomState(21)
    image = rs.randn(batch, n_img, dim).astype("float32")
    text = rs.randn(batch, n_txt, dim).astype("float32")

    layer = MMDiTJointAttention(dim=dim, num_heads=heads, context_pre_only=True)
    layer([image, text])  # build
    assert layer.add_q_proj is None and layer.norm_added_q is None

    def dense_weights():
        return [
            (0.3 * rs.randn(dim, dim)).astype("float32"),
            (0.3 * rs.randn(dim)).astype("float32"),
        ]

    def norm_scale():
        return (1.0 + 0.3 * rs.randn(head_dim)).astype("float32")

    w = {name: dense_weights() for name in (
        "to_q", "to_k", "to_v", "to_out", "add_k_proj", "add_v_proj",
        "ref_add_q",
    )}
    s = {name: norm_scale() for name in (
        "norm_q", "norm_k", "norm_added_k", "ref_norm_added_q",
    )}
    for name in ("to_q", "to_k", "to_v", "to_out", "add_k_proj", "add_v_proj"):
        getattr(layer, name).set_weights(w[name])
    for name in ("norm_q", "norm_k", "norm_added_k"):
        getattr(layer, name).scale.assign(s[name])

    def proj(x, name):
        kernel, bias = w[name]
        y = x.astype("float64") @ kernel + bias
        return y.reshape(batch, -1, heads, head_dim).transpose(0, 2, 1, 3)

    q = np.concatenate([
        _rms(proj(image, "to_q"), s["norm_q"], eps),
        _rms(proj(text, "ref_add_q"), s["ref_norm_added_q"], eps),
    ], axis=2)
    k = np.concatenate([
        _rms(proj(image, "to_k"), s["norm_k"], eps),
        _rms(proj(text, "add_k_proj"), s["norm_added_k"], eps),
    ], axis=2)
    v = np.concatenate([proj(image, "to_v"), proj(text, "add_v_proj")], axis=2)

    logits = (q @ k.transpose(0, 1, 3, 2)) * head_dim ** -0.5
    logits = logits - logits.max(axis=-1, keepdims=True)
    attn = np.exp(logits)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    joint = (attn @ v).transpose(0, 2, 1, 3).reshape(batch, n_img + n_txt, dim)
    kernel, bias = w["to_out"]
    expected = joint[:, :n_img, :] @ kernel + bias

    actual = np.asarray(layer([image, text]))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
