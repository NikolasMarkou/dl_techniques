"""Guard: the ``1/sqrt(head_dim)`` pre-softmax scale must reach the logits.

Why this file exists
--------------------
Deleting the scale multiply is caught by ZERO tests in five separate files, every
deletion proven non-inert:

    multi_head_cross_attention.py   0 / 93   (widest blast radius -- `multi_head`
                                              and `perceiver` both delegate here)
    single_window_attention.py      0 / 63
    ideogram4_attention.py          0 / 13
    lighthouse_attention.py         0 / 12
    progressive_focused_attention.py 0 / 19

Without the scale, attention logits grow with ``head_dim``, softmax saturates and
gradients vanish -- a model that trains badly rather than one that crashes, which
is why no shape, dtype, finiteness or round-trip test notices.

What is asserted
----------------
Two things per layer, and the second is the one that matters:

1. the stored scale EQUALS ``head_dim ** -0.5``; and
2. perturbing that stored value at runtime CHANGES the output.

(2) is what makes this a guard rather than a restatement of the constructor. If
the multiply is deleted, the stored attribute becomes decorative and perturbing it
is inert, so the delta collapses to exactly 0.0 and the test goes red. (1) alone
would pass happily against a layer that computes the value and then ignores it.

Measured deltas on HEAD for a x2 perturbation:

    multi_head_cross      0.698
    single_window         0.378
    ideogram4             0.567
    lighthouse            1.335
    progressive_focused   0.530
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.ideogram4_attention import Ideogram4Attention
from dl_techniques.layers.attention.lighthouse_attention import (
    LighthouseAttention,
)
from dl_techniques.layers.attention.multi_head_cross_attention import (
    MultiHeadCrossAttention,
)
from dl_techniques.layers.attention.progressive_focused_attention import (
    ProgressiveFocusedAttention,
)
from dl_techniques.layers.attention.single_window_attention import (
    SingleWindowAttention,
)

# The hand-computed oracle below compares a GPU matmul against numpy at
# atol=1e-5, which Ampere TF32 (~1e-3 relative) breaks: it passes on CPU and
# fails on an RTX 4070. Same opt-in `test_linear_attention.py` uses.
pytestmark = pytest.mark.usefixtures("tf32_disabled")

DIM = 32
HEADS = 4
HEAD_DIM = DIM // HEADS
TOKENS = 16


def _sequence():
    return np.random.default_rng(0).normal(size=(2, TOKENS, DIM)).astype("float32")


def _feature_map():
    return np.random.default_rng(0).normal(size=(1, 8, 8, DIM)).astype("float32")


def _rope_tables():
    """Hand-built cos/sin tables; `Ideogram4Attention.call` requires them."""
    position = np.arange(TOKENS)[None, :, None]
    inv_freq = 1.0 / (10000 ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))[None, None, :]
    angle = np.repeat((position * inv_freq).astype("float32"), 2, axis=-1)
    return np.cos(angle).astype("float32"), np.sin(angle).astype("float32")


def _unwrap(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _multi_head_cross():
    return MultiHeadCrossAttention(dim=DIM, num_heads=HEADS), (
        lambda layer: layer(_sequence(), training=False)
    ), "scale"


def _single_window():
    return SingleWindowAttention(dim=DIM, window_size=4, num_heads=HEADS), (
        lambda layer: layer(_sequence(), training=False)
    ), "scale"


def _ideogram4():
    segments = np.zeros((2, TOKENS), dtype="int32")
    segments[:, TOKENS // 2:] = 1
    cos, sin = _rope_tables()
    return Ideogram4Attention(hidden_size=DIM, num_heads=HEADS), (
        lambda layer: layer(
            _sequence(), segment_ids=segments, cos=cos, sin=sin, training=False
        )
    ), "_inv_sqrt_dim"


def _lighthouse():
    return LighthouseAttention(dim=DIM, num_heads=HEADS), (
        lambda layer: layer(_sequence(), training=False)
    ), "_scale"


def _progressive_focused():
    return ProgressiveFocusedAttention(
        dim=DIM, num_heads=HEADS, window_size=4
    ), (lambda layer: _unwrap(layer(_feature_map(), training=False))), "_scale"


LAYERS = {
    "multi_head_cross": _multi_head_cross,
    "single_window": _single_window,
    "ideogram4": _ideogram4,
    "lighthouse": _lighthouse,
    "progressive_focused": _progressive_focused,
}


@pytest.mark.parametrize("name", sorted(LAYERS))
def test_the_stored_scale_is_the_inverse_square_root_of_head_dim(name):
    """(1) The value itself."""
    keras.utils.set_random_seed(0)
    layer, call, attribute = LAYERS[name]()
    call(layer)  # build
    assert float(getattr(layer, attribute)) == pytest.approx(
        HEAD_DIM ** -0.5, rel=1e-6
    ), f"{name}: stored scale is not head_dim ** -0.5"


def test_the_scale_divides_rather_than_multiplies_would_be_caught():
    """Direction, not just presence.

    The perturbation test below proves the stored scale REACHES the logits. It
    does not prove the logits are scaled the right WAY: replacing `q * scale`
    with `q / scale` is a factor-of-head_dim error -- exactly the "logits grow
    with head_dim, softmax saturates" failure this file exists to prevent -- and
    an adversarial review measured it leaving 73 tests green.

    So one hand-computed VALUE oracle is pinned here, on
    `MultiHeadCrossAttention` because `multi_head` and `perceiver` both delegate
    to it. Every projection is forced to the identity, reducing the layer to
    plain scaled dot-product attention whose expected values come from numpy.
    """
    dim, heads = 4, 1
    head_dim = dim // heads
    keras.utils.set_random_seed(0)
    layer = MultiHeadCrossAttention(dim=dim, num_heads=heads, use_bias=False)
    tokens = np.random.RandomState(5).randn(1, 3, dim).astype("float32")
    layer(tokens, training=False)  # build

    # `kv_dense` is FUSED: one (dim, 2*dim) kernel producing K and V, so the
    # identity for it is [I | I] rather than a square I.
    identity = np.eye(dim, dtype="float32")
    layer.q_dense.set_weights([identity])
    layer.kv_dense.set_weights([np.concatenate([identity, identity], axis=1)])
    layer.proj_dense.set_weights([identity])

    actual = np.asarray(layer(tokens, training=False))

    logits = (tokens[0] @ tokens[0].T) * (head_dim ** -0.5)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    expected = probs @ tokens[0]

    np.testing.assert_allclose(
        actual[0], expected, atol=1e-5, rtol=0,
        err_msg=(
            "the pre-softmax logits are not scaled by exactly head_dim ** -0.5; "
            "a dropped scale, an inverted one (`/` instead of `*`) or a wrong "
            "exponent all land here"
        ),
    )


@pytest.mark.parametrize("name", sorted(LAYERS))
def test_perturbing_the_scale_changes_the_output(name):
    """(2) The value must actually reach the logits.

    Why this can fail if the implementation is wrong: with the multiply deleted
    the stored attribute is decorative, so doubling it moves nothing and this
    delta is exactly 0.0.
    """
    keras.utils.set_random_seed(0)
    layer, call, attribute = LAYERS[name]()
    before = np.asarray(call(layer))

    setattr(layer, attribute, getattr(layer, attribute) * 2.0)
    after = np.asarray(call(layer))

    delta = np.abs(before - after).max()
    assert delta > 1e-4, (
        f"{name}: doubling the stored attention scale moved the output by only "
        f"{delta}. The scale is computed and stored but never multiplied into "
        "the logits, so it is decorative."
    )
