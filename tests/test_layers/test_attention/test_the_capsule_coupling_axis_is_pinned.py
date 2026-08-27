"""Guard: pin which axis the dynamic-routing coupling coefficients normalize over.

Why this test exists
--------------------
``CapsuleRoutingSelfAttention`` cites Sabour et al. 2017, which normalizes
``c_ij = softmax_j(b_ij)`` over the OUTPUT capsules (``sum_j c_ij == 1`` per input
capsule). This implementation normalizes over the INPUT capsule axis instead --
the transpose. Flipping ``_site_config(-2)`` to ``(-1)`` passed the whole 65-test
suite in BOTH directions, so nothing pinned which convention shipped.

The deviation is deliberate and documented (decisions.md D-008): the paper's axis
makes ``_horizontal_routing``'s ``num_output_capsules = 1`` branch a size-1
softmax no-op and produces a reproducible NaN under ``mixed_float16``. This guard
exists so the choice stays a choice -- a future edit that flips the axis has to
turn this red rather than sail through.

Why NON-SQUARE: with ``num_input_capsules == num_output_capsules`` the coupling
tensor is square, and a normalization over either axis produces a tensor of the
same shape whose row and column sums are easy to confuse. A non-square
configuration makes the two axes structurally distinguishable, so a transposed
implementation cannot satisfy this test by coincidence.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.capsule_routing_attention import (
    CapsuleRoutingSelfAttention,
)

HEADS = 4
KEY_DIM = 8
SEQ = 8
DIM = 32


def _capture_all_routing_weights(layer, tokens):
    """Run the layer and return EVERY coupling tensor it produced.

    All of them, not just the first: `_horizontal_routing`'s positional branch
    calls `_dynamic_routing` with `num_output_capsules == 1`, which is the only
    NON-SQUARE coupling this layer builds and therefore the only one that can
    tell the two axes apart structurally.
    """
    captured = []
    original = layer._dynamic_routing

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(np.asarray(result[1]))
        return result

    layer._dynamic_routing = spy
    layer(tokens, training=False)
    assert captured, "_dynamic_routing was never reached"
    return captured


def _capture_routing_weights(layer, tokens):
    """The first coupling tensor, for the square-configuration assertions."""
    return _capture_all_routing_weights(layer, tokens)[0]


@pytest.fixture(name="tokens")
def _tokens():
    return np.random.default_rng(0).normal(size=(2, SEQ, DIM)).astype("float32")


@pytest.fixture(name="layer")
def _layer():
    keras.utils.set_random_seed(0)
    return CapsuleRoutingSelfAttention(num_heads=HEADS, key_dim=KEY_DIM)


def test_the_coupling_sums_to_one_over_the_input_capsule_axis(layer, tokens):
    """The shipped convention, pinned.

    Why this can fail if the implementation is wrong: flipping the site to
    `axis=-1` makes these sums stop being 1 and makes the axis-(-1) sums become 1
    instead, which the companion assertion below catches from the other side.
    """
    weights = _capture_routing_weights(layer, tokens)
    over_inputs = weights.sum(axis=-2)
    np.testing.assert_allclose(over_inputs, np.ones_like(over_inputs), atol=1e-5)


def test_the_coupling_does_not_sum_to_one_over_the_output_capsule_axis(
    layer, tokens
):
    """The other side of the same pin.

    Stated as an explicit NOT: it is what distinguishes this implementation from
    the cited paper, and it is the assertion a paper-matching rewrite must break.
    """
    weights = _capture_routing_weights(layer, tokens)
    over_outputs = weights.sum(axis=-1)
    assert not np.allclose(over_outputs, np.ones_like(over_outputs), atol=1e-3), (
        "the coupling coefficients now sum to 1 over the OUTPUT capsule axis, "
        "i.e. the normalization axis was flipped to Sabour et al.'s convention. "
        "That is a real behaviour change and it reintroduces a mixed_float16 NaN "
        "in _horizontal_routing's num_output_capsules == 1 branch -- see "
        "decisions.md D-008 before changing this."
    )


def test_the_pin_holds_on_the_non_square_coupling(layer, tokens):
    """A square coupling cannot tell the two axes apart; the positional one can.

    The per-head couplings are square by construction (`num_heads x num_heads`),
    so they cannot discriminate a transpose. `_horizontal_routing`'s positional
    branch builds the one non-square coupling in the layer
    (`num_output_capsules == 1`), and that is precisely the branch a flip to
    `axis=-1` would turn into a size-1 no-op. This asserts it exists and is
    normalized on the shipped axis -- never skipping, because a guard that can
    only skip is not a guard.
    """
    all_weights = _capture_all_routing_weights(layer, tokens)
    non_square = [w for w in all_weights if w.shape[-1] != w.shape[-2]]
    assert non_square, (
        "no non-square coupling was produced, so _horizontal_routing's "
        f"positional branch was not reached; shapes seen: "
        f"{[w.shape for w in all_weights]}"
    )
    for weights in non_square:
        over_inputs = weights.sum(axis=-2)
        np.testing.assert_allclose(
            over_inputs, np.ones_like(over_inputs), atol=1e-5
        )


def test_the_coupling_is_actually_data_dependent(layer, tokens):
    """Routing-by-agreement must respond to the input, not be uniform.

    Zeroing the agreement update leaves the coefficients uniform forever and
    passes the whole existing suite; a uniform tensor would also satisfy the
    sums-to-one assertions above, so it is excluded here explicitly.
    """
    weights = _capture_routing_weights(layer, tokens)
    spread = float(weights.max() - weights.min())
    assert spread > 1e-3, (
        f"coupling coefficients span only {spread}: routing-by-agreement is not "
        "responding to the data, so the iterative update is inert"
    )
