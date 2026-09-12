"""Sentence-named guard: a shared mem-block's weights are identical -- not just
equal-valued -- at every depth position it is invoked from.

Zamba2's defining mechanic is that ONE physical
:class:`Zamba2SharedAttentionBlock` instance is built once per mem-block slot
and then called at several different depth positions in the decoder stack
(plan.md invariant 1). Keras shares weights by construction when the same
Python layer object is called more than once, but a regression that
accidentally builds a FRESH block per call site (the historically-invisible
"looks identical, isn't shared" failure -- v2 guide Section 12.7, named in
this plan's Pre-Mortem) would still pass a plain value-equality check right
after initialization, since two freshly-initialized layers can coincidentally
start from very different but individually-finite values and a shape-only
assertion would miss it entirely. This guard asserts ``is``-identity on the
underlying ``Variable`` objects, which a copy can never satisfy.
"""

import keras

from dl_techniques.models.language.zamba2.layers import Zamba2SharedAttentionBlock


def test_shared_weights_identical_across_depth() -> None:
    """The same block instance, called at two different depths, shares the
    identical (``is``-identity) weight ``Variable`` objects at both call
    sites -- proving genuine weight sharing, not look-alike duplication."""
    block = Zamba2SharedAttentionBlock(d_model=32, num_heads=4, max_seq_len=64)

    hidden_state_depth_2 = keras.random.normal(shape=(2, 10, 32))
    original_embedding = keras.random.normal(shape=(2, 10, 32))
    hidden_state_depth_5 = keras.random.normal(shape=(2, 10, 32))

    # First occurrence: depth position 2 in some hypothetical layer_mapping.
    _ = block(hidden_state_depth_2, original_embedding)
    weights_at_depth_2 = list(block.weights)
    assert len(weights_at_depth_2) > 0, "Block must hold weights after its first call"

    # Second occurrence: depth position 5, same physical instance.
    _ = block(hidden_state_depth_5, original_embedding)
    weights_at_depth_5 = list(block.weights)

    assert len(weights_at_depth_2) == len(weights_at_depth_5)
    for w_depth_2, w_depth_5 in zip(weights_at_depth_2, weights_at_depth_5):
        assert w_depth_2 is w_depth_5, (
            f"Weight {w_depth_2.name!r} differs between depth 2 and depth 5 call "
            "sites -- the mem-block is not actually shared across depth."
        )

    # Negative twin: two INDEPENDENTLY constructed blocks must NOT share
    # weight objects, closing the mutation-family the positive assertion
    # alone would miss (a guard that always passes regardless of sharing).
    other_block = Zamba2SharedAttentionBlock(d_model=32, num_heads=4, max_seq_len=64)
    _ = other_block(hidden_state_depth_2, original_embedding)
    other_weights = list(other_block.weights)

    assert len(other_weights) == len(weights_at_depth_2)
    for w_shared, w_independent in zip(weights_at_depth_2, other_weights):
        assert w_shared is not w_independent, (
            "Two independently constructed blocks must not share weight "
            "Variable objects -- this would mean ALL blocks are accidentally "
            "aliased to one another, not just occurrences of the SAME block."
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
