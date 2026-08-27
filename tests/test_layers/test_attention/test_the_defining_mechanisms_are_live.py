"""Guards for layer mechanisms that no test could see break.

Why this file exists
--------------------
Each layer below had its DEFINING property deleted -- the thing it exists to do --
and its own suite stayed green, every deletion proven non-inert first:

    multi_head_latent_attention  widen away the KV bottleneck      0 / 86
    group_query_attention        collapse KV grouping to MQA       0 / 57
    beit_attention               all heads share head 0's bias     0 / 99
    hopfield_attention           disable the iterative loop-back   0 / 76
    perceiver_attention          sever the data array              0 / 26
    non_local_attention          force self-only attention         0 / 16

The suites are not thin -- ``gated_attention`` has 121 tests and
``multi_head_cross_attention`` 93. They assert shapes, configs, dtypes,
finiteness and round-trips, every one of which a gutted layer still satisfies.
What is missing is an oracle for the MECHANISM, so that is what each guard here
pins, stated as a property derivable without reading the implementation.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.beit_attention import BeitAttention
from dl_techniques.layers.attention.group_query_attention import (
    GroupedQueryAttention,
)
from dl_techniques.layers.attention.hopfield_attention import HopfieldAttention
from dl_techniques.layers.attention.multi_head_latent_attention import (
    MultiHeadLatentAttention,
)
from dl_techniques.layers.attention.non_local_attention import NonLocalAttention
from dl_techniques.layers.attention.perceiver_attention import (
    PerceiverAttention,
)

pytestmark = pytest.mark.usefixtures("tf32_disabled")

DIM = 64
TOKENS = 32


def _sequence(dim=DIM, tokens=TOKENS, seed=0):
    return (
        np.random.default_rng(seed)
        .normal(size=(2, tokens, dim))
        .astype("float32")
    )


# --------------------------------------------------------------------------
# MultiHeadLatentAttention -- the KV bottleneck must actually be low-rank
# --------------------------------------------------------------------------
def test_the_kv_path_rank_is_bounded_by_the_latent_dim():
    """MLA exists to force keys/values through a low-rank bottleneck.

    Widening `kv_down_proj` to full width keeps every shape, config key and
    round-trip intact -- the tensors still line up -- so the suite cannot tell.
    The property that actually distinguishes them is the RANK of the composed
    down-then-up map, which must not exceed `kv_latent_dim`.

    Why this can fail if the implementation is wrong: with the bottleneck widened
    the measured rank rises above the latent dim (8 -> 9 and beyond).
    """
    latent = 8
    keras.utils.set_random_seed(0)
    layer = MultiHeadLatentAttention(
        dim=DIM, num_heads=4, kv_latent_dim=latent
    )
    layer(_sequence(), training=False)

    weights = {w.path.split("/")[-2]: np.asarray(w) for w in layer.weights}
    composed = weights["kv_down_proj"] @ weights["kv_up_proj"]

    rank = int(np.linalg.matrix_rank(composed, tol=1e-4))
    assert rank <= latent, (
        f"the composed KV map has rank {rank} > kv_latent_dim={latent}: keys and "
        "values are no longer passing through the low-rank bottleneck that is "
        "this layer's entire reason to exist"
    )
    assert rank == latent, (
        f"rank collapsed to {rank}, below kv_latent_dim={latent}; the bottleneck "
        "is narrower than configured"
    )


# --------------------------------------------------------------------------
# GroupedQueryAttention -- KV heads must be SHARED, in contiguous blocks
# --------------------------------------------------------------------------
def test_kv_heads_are_shared_by_contiguous_query_groups():
    """GQA exists so several query heads share one KV head.

    Two failure modes look identical in shape: collapsing to a single KV head
    (MQA), and repeating with `tile` instead of `repeat`, which interleaves the
    groups so query head i is paired with the wrong KV head. Both pass all 57
    existing tests.

    The projection widths pin the sharing ratio; a per-KV-head perturbation pins
    the contiguous grouping, since `repeat` gives [kv0, kv0, kv1, kv1] and `tile`
    gives [kv0, kv1, kv0, kv1].
    """
    heads, kv_heads = 8, 2
    keras.utils.set_random_seed(0)
    layer = GroupedQueryAttention(
        dim=DIM, num_heads=heads, num_kv_heads=kv_heads
    )
    layer(_sequence(), training=False)

    weights = {w.path.split("/")[-2]: np.asarray(w) for w in layer.weights}
    head_dim = DIM // heads
    assert weights["w_q"].shape[1] == heads * head_dim
    assert weights["w_k"].shape[1] == kv_heads * head_dim, (
        f"K projects to {weights['w_k'].shape[1]} units; with {kv_heads} KV "
        f"heads of width {head_dim} it must be {kv_heads * head_dim}. A collapse "
        "to one KV head (MQA) or an expansion to full MHA both show up here."
    )
    assert weights["w_v"].shape[1] == kv_heads * head_dim
    assert weights["w_k"].shape[1] < weights["w_q"].shape[1], (
        "K is as wide as Q, so nothing is being shared and this is not GQA"
    )

    # The shape assertions above catch a COLLAPSE (to MQA) or an EXPANSION (to
    # full MHA), because both change a projection width. They CANNOT catch the
    # ORDERING defect -- `repeat` gives [kv0, kv0, kv1, kv1] while `tile` gives
    # [kv0, kv1, kv0, kv1] -- because both produce identically shaped tensors.
    #
    # Measured: swapping to `tile` moves the output by 1.4769 and left all 63
    # tests in this file and the GQA suite GREEN. This guard shipped with a
    # docstring promising a perturbation its body did not contain; an adversarial
    # review caught it. The ordering is now pinned against an EXTERNAL oracle.
    ordering_layer, tokens, weights_by_name = _gqa_probe()
    actual = np.asarray(ordering_layer(tokens, training=False))

    matched = {
        order: np.abs(actual - _gqa_reference(tokens, weights_by_name, order)).max()
        for order in ("repeat", "tile")
    }
    assert matched["repeat"] < 1e-5, (
        f"the layer does not match a contiguous (`repeat`) grouping: "
        f"max|delta| = {matched['repeat']}"
    )
    assert matched["tile"] > 1e-2, (
        f"the layer matches an INTERLEAVED (`tile`) grouping as closely as a "
        f"contiguous one ({matched['tile']}), so query head i is paired with the "
        "wrong KV head"
    )


def _gqa_probe():
    """A RoPE-free GQA layer plus its inputs, for the ordering oracle.

    `rope_percentage=0.0` disables the rotary embedding so the oracle below can
    be plain scaled dot-product attention; the grouping order is what is under
    test, and RoPE is orthogonal to it.
    """
    keras.utils.set_random_seed(0)
    layer = GroupedQueryAttention(
        dim=DIM, num_heads=8, num_kv_heads=2, rope_percentage=0.0
    )
    tokens = np.random.default_rng(0).normal(size=(1, 16, DIM)).astype("float32")
    layer(tokens, training=False)
    weights = {w.path.split("/")[-2]: np.asarray(w) for w in layer.weights}
    return layer, tokens, weights


def _gqa_reference(tokens, weights, order):
    """Scaled dot-product attention with the KV heads grouped `order`-wise.

    Built from the layer's own projection weights but never calling the layer's
    attention math, so it is external to the code path under test.
    """
    heads, kv_heads = 8, 2
    head_dim = DIM // heads
    groups = heads // kv_heads
    batch, length = tokens.shape[0], tokens.shape[1]

    def _heads(matrix, count):
        return (tokens @ matrix).reshape(
            batch, length, count, head_dim
        ).transpose(0, 2, 1, 3)

    query = _heads(weights["w_q"], heads)
    key = _heads(weights["w_k"], kv_heads)
    value = _heads(weights["w_v"], kv_heads)

    if order == "repeat":
        key, value = (np.repeat(t, groups, axis=1) for t in (key, value))
    else:
        key, value = (np.tile(t, (1, groups, 1, 1)) for t in (key, value))

    scores = (query @ key.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights_ = np.exp(scores)
    attended = (weights_ / weights_.sum(axis=-1, keepdims=True)) @ value
    merged = attended.transpose(0, 2, 1, 3).reshape(batch, length, DIM)
    return merged @ weights["w_o"]


def test_the_relative_position_bias_differs_across_heads():
    """Collapsing every head's bias onto head 0's passes all 99 tests.

    Every liveness and orientation probe in that suite uses a single head or a
    head-agnostic reduction, so one head's bias overwriting another is invisible.
    The table is `(num_relative_distance, num_heads)`; distinct columns are the
    property.
    """
    heads = 4
    tokens = _sequence(tokens=4 * 4 + 1)

    def _with_table(columns):
        keras.utils.set_random_seed(0)
        layer = BeitAttention(dim=DIM, num_heads=heads, window_size=(4, 4))
        layer(tokens, training=False)
        table = next(
            w
            for w in layer.weights
            if len(w.shape) == 2 and w.shape[-1] == heads
        )
        table.assign(keras.ops.convert_to_tensor(columns))
        return np.asarray(layer(tokens, training=False))


    # Build two bias tables: one with a DISTINCT column per head, one where every
    # head is given head 0's column. The table is zero-initialised, so comparing
    # the raw columns at init proves nothing -- that is the documented
    # false-inert trap. Only a behavioural comparison discriminates.
    keras.utils.set_random_seed(0)
    probe = BeitAttention(dim=DIM, num_heads=heads, window_size=(4, 4))
    probe(tokens, training=False)
    shape = next(
        w.shape for w in probe.weights if len(w.shape) == 2 and w.shape[-1] == heads
    )
    rng = np.random.default_rng(3)
    per_head = rng.normal(size=tuple(shape)).astype("float32")
    collapsed = np.repeat(per_head[:, :1], heads, axis=1)

    distinct_out = _with_table(per_head)
    collapsed_out = _with_table(collapsed)

    assert np.abs(distinct_out - collapsed_out).max() > 1e-4, (
        "giving every head head-0's relative-position bias produced the same "
        "output as giving each head its own: the bias is not applied per head"
    )


# --------------------------------------------------------------------------
# HopfieldAttention -- the iterative retrieval must actually iterate
# --------------------------------------------------------------------------
def test_more_update_steps_change_the_result():
    """Hopfield attention exists to ITERATE toward a stored pattern.

    Disabling the loop-back -- so the layer repeats step 0 N times instead of
    feeding each step's result into the next -- passes all 76 tests. Nothing
    distinguished "iterates" from "repeats the first step".

    Why this can fail if the implementation is wrong: with the loop-back gone the
    step count stops mattering and these two outputs become identical.
    """
    tokens = _sequence()

    keras.utils.set_random_seed(0)
    one_step = HopfieldAttention(num_heads=4, key_dim=16, update_steps_max=1)
    first = np.asarray(one_step(tokens, training=False))

    keras.utils.set_random_seed(0)
    many_steps = HopfieldAttention(num_heads=4, key_dim=16, update_steps_max=4)
    many_steps(tokens, training=False)  # build before copying weights

    # Copy the weights across. Seeding alone is NOT enough: both layers build
    # lazily on first call, so constructing them back to back and calling them
    # afterwards draws two different weight sets from the stream, and the delta
    # would then measure weight difference rather than step count. That confound
    # made this test pass against the disabled-loop-back mutant.
    many_steps.set_weights(one_step.get_weights())
    second = np.asarray(many_steps(tokens, training=False))

    delta = np.abs(first - second).max()
    assert delta > 1e-4, (
        f"1 update step and 4 update steps differ by only {delta}: the iterative "
        "update is not feeding its result back, so the layer repeats step 0"
    )


# --------------------------------------------------------------------------
# PerceiverAttention -- the latents must read the DATA array
# --------------------------------------------------------------------------
def test_the_latents_depend_on_the_data_array():
    """Perceiver exists so a small latent set cross-attends to a long input.

    Replacing the key/value source with the query source severs the data array
    entirely -- the layer becomes latent self-attention -- and passes all 26
    tests, because the output shape is unchanged.
    """
    keras.utils.set_random_seed(0)
    layer = PerceiverAttention(dim=DIM, num_heads=4)

    latents = _sequence(tokens=8, seed=1)
    data = _sequence(tokens=TOKENS, seed=2)

    before = np.asarray(layer(latents, kv_input=data, training=False))

    perturbed = data.copy()
    perturbed[:, 0, :] += 10.0
    after = np.asarray(layer(latents, kv_input=perturbed, training=False))

    delta = np.abs(before - after).max()
    assert delta > 1e-4, (
        f"the latent output moved by only {delta} when the DATA array was "
        "perturbed: the latents are not reading the data, so this is latent "
        "self-attention rather than Perceiver cross-attention"
    )


# --------------------------------------------------------------------------
# NonLocalAttention -- distant positions must interact
# --------------------------------------------------------------------------
def test_distant_positions_interact():
    """The 'non-local' in the name is the whole claim.

    Forcing block-diagonal (self-only) attention passes all 16 tests. The
    property is that a perturbation at one corner of the feature map reaches the
    opposite corner.
    """
    keras.utils.set_random_seed(0)
    layer = NonLocalAttention(attention_channels=8)

    feature_map = (
        np.random.default_rng(0).normal(size=(1, 8, 8, 16)).astype("float32")
    )
    before = np.asarray(layer(feature_map, training=False))

    perturbed = feature_map.copy()
    perturbed[0, 0, 0, :] += 20.0
    after = np.asarray(layer(perturbed, training=False))

    far_corner_delta = np.abs(before[0, -1, -1, :] - after[0, -1, -1, :]).max()
    assert far_corner_delta > 1e-5, (
        f"the far corner moved by only {far_corner_delta} when the opposite "
        "corner was perturbed: attention is local, which contradicts the layer's "
        "name and purpose"
    )
