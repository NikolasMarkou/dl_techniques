"""Every one of ``DiTXABlock``'s 12 adaLN chunks reaches the sub-op it is named for.

``DiTXABlock``'s ``adaLN_modulation`` emits ``12 * hidden_size`` numbers that are
split into twelve ``(B, hidden)`` chunks, consumed in exactly this order::

    shift_msa, scale_msa, gate_msa,      # triple 1 -> self-attention on x
    shift_xa,  scale_xa,  gate_xa,       # triple 2 -> cross-attention QUERY (x)
    shift_cond, scale_cond, gate_cond,   # triple 3 -> cross-attention K/V (cond_tokens)
    shift_mlp, scale_mlp, gate_mlp       # triple 4 -> the MLP on x

A permutation of that order changes NOTHING observable by shape, by parameter
count, by ``get_config()`` or by a save/load round trip. It changes only which
learned scalar multiplies which sub-op, so it trains to a different -- wrong --
model under a fully green conventional suite. Hence this file.

**How the probe works, and why it is not a re-derivation of the block's own
arithmetic.** The chunk-producing ``Dense(12 * hidden)`` is zero-initialised in
both kernel AND bias, so at initialisation every chunk is exactly zero, every
``modulate`` is the identity and every gate is ``0`` -- the block is the exact
identity map on ``x`` (:func:`test_the_unperturbed_block_is_the_identity` pins
that premise). The probe then writes a one-hot-over-one-chunk-slice pattern into
that Dense's **bias** and asks only *which sub-op moved*. No expected value is
ever computed from the block's formula; every assertion is a
changed / bit-identical comparison between two runs of the block itself.

The four triples are made mutually distinguishable by four independent
mechanisms, each of which is its own arm:

1. ``gate_*`` chunks act with every other chunk at zero; ``shift_*`` / ``scale_*``
   chunks cannot act at all while their own gate is zero
   (:func:`test_only_the_three_residual_gates_act_when_every_other_chunk_is_zero`).
2. With exactly one gate open, a ``shift`` / ``scale`` chunk moves the output if
   and only if it belongs to that gate's sub-op
   (:func:`test_each_shift_and_scale_chunk_moves_only_its_own_sub_op`).
3. Triple 2 modulates ``norm_cross(x)`` and triple 3 modulates
   ``norm_cond(cond_tokens)``. Feeding an ``x`` whose rows are constant across
   the channel axis makes ``norm_cross(x)`` exactly zero, which kills
   ``scale_xa`` -- and only ``scale_xa`` -- while leaving ``scale_cond`` live
   (:func:`test_scale_xa_dies_on_a_channel_constant_x_while_scale_cond_lives`).
4. Triple 3 reads ``cond_tokens``, so the block output must depend on
   ``cond_tokens`` at all (:func:`test_the_block_output_depends_on_cond_tokens`).

The residual order is pinned by a composition identity that uses the block as
its own oracle: running the block three times with one gate open each, in the
order ``msa -> xa -> mlp``, must reproduce the single all-gates-open run
**bit-identically**, and the other five permutations must not
(:func:`test_the_residual_order_is_msa_then_xa_then_mlp`). The "must not" half
is the anti-vacuity arm: it is what proves the three sub-ops do not commute, so
the ordering claim is observable in the first place.

Success criterion: SC-3.
"""

import itertools
import math

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.blocks import (
    ADALN_CHUNK_NAMES,
    NUM_ADALN_CHUNKS,
    DiTXABlock,
)

# ---------------------------------------------------------------------
# Fixture geometry -- small enough to be instant, large enough to be generic
# ---------------------------------------------------------------------

BATCH = 2
SEQ = 4
HIDDEN = 8
NUM_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
SEED = 20260902

#: Chunk index of each residual gate, by the sub-op it gates.
GATE_INDEX = {"msa": 2, "xa": 5, "mlp": 11}

#: For every chunk index, the set of open-gate paths under which perturbing it
#: must move the block output. Derived from the chunk ORDER above, by hand --
#: never by running the implementation.
EXPECTED_SENSITIVITY = {
    0: {"msa"},                  # shift_msa
    1: {"msa"},                  # scale_msa
    2: {"msa", "xa", "mlp"},     # gate_msa -- opens its own path from anywhere
    3: {"xa"},                   # shift_xa
    4: {"xa"},                   # scale_xa
    5: {"msa", "xa", "mlp"},     # gate_xa
    6: {"xa"},                   # shift_cond  -- K/V side of the SAME residual
    7: {"xa"},                   # scale_cond
    8: set(),                    # gate_cond   -- gates no residual at all
    9: {"mlp"},                  # shift_mlp
    10: {"mlp"},                 # scale_mlp
    11: {"msa", "xa", "mlp"},    # gate_mlp
}


def _make_block(seed: int = SEED) -> DiTXABlock:
    """Return a built block with reproducible, non-degenerate sub-layer weights."""
    keras.utils.set_random_seed(seed)
    block = DiTXABlock(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        mlp_ratio=2.0,
        name="probe_block",
    )
    block.build(
        [
            (None, SEQ, HIDDEN),
            (None, HIDDEN),
            (None, SEQ, HIDDEN),
        ]
    )
    return block


def _inputs(seed: int = SEED + 1):
    """Return a generic ``(x, c, cond_tokens)`` triple as numpy arrays."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(BATCH, SEQ, HIDDEN)).astype("float32")
    c = rng.normal(size=(BATCH, HIDDEN)).astype("float32")
    cond = rng.normal(size=(BATCH, SEQ, HIDDEN)).astype("float32")
    return x, c, cond


def _chunk_slice(index: int) -> slice:
    return slice(index * HIDDEN, (index + 1) * HIDDEN)


def _bias_with(**chunks: float) -> np.ndarray:
    """Build a ``(12 * hidden,)`` bias vector with the named chunks set flat.

    :param chunks: ``chunk_name=value`` pairs; every other chunk stays zero.
    """
    vec = np.zeros((NUM_ADALN_CHUNKS * HIDDEN,), dtype="float32")
    for name, value in chunks.items():
        vec[_chunk_slice(ADALN_CHUNK_NAMES.index(name))] = value
    return vec


def _run(block: DiTXABlock, bias: np.ndarray, x, c, cond) -> np.ndarray:
    """Assign ``bias`` to the modulation Dense and run the block once."""
    block.adaln_dense.bias.assign(keras.ops.convert_to_tensor(bias))
    return np.array(block([x, c, cond], training=False))


# =====================================================================
# The probe's own premises
# =====================================================================


def test_the_modulation_dense_is_zero_initialised_in_kernel_and_bias():
    """The whole probe rests on both halves being exactly zero at init."""
    block = _make_block()
    kernel = np.array(block.adaln_dense.kernel)
    bias = np.array(block.adaln_dense.bias)

    assert kernel.shape == (HIDDEN, NUM_ADALN_CHUNKS * HIDDEN)
    assert bias.shape == (NUM_ADALN_CHUNKS * HIDDEN,)
    assert np.array_equal(kernel, np.zeros_like(kernel)), (
        "adaLN kernel must be zero-init (adaLN-Zero); a non-zero kernel makes "
        "every chunk depend on c and destroys this file's attribution probe"
    )
    assert np.array_equal(bias, np.zeros_like(bias))


def test_the_unperturbed_block_is_the_identity():
    """With every chunk zero, all three gates are zero, so the block returns x."""
    block = _make_block()
    x, c, cond = _inputs()
    y = np.array(block([x, c, cond], training=False))
    assert np.array_equal(y, x), (
        "at adaLN-Zero init the block must be the exact identity on x; it is "
        f"not (max|y-x| = {np.max(np.abs(y - x))})"
    )


def test_the_conditioning_vector_reaches_the_modulation_dense():
    """Anti-vacuity: ``c`` is inert only because the kernel is zero, not by wiring."""
    block = _make_block()
    x, c, cond = _inputs()
    rng = np.random.default_rng(7)
    block.adaln_dense.kernel.assign(
        keras.ops.convert_to_tensor(
            rng.normal(size=(HIDDEN, NUM_ADALN_CHUNKS * HIDDEN)).astype("float32")
        )
    )
    y_a = np.array(block([x, c, cond], training=False))
    y_b = np.array(block([x, c + 1.0, cond], training=False))
    assert not np.allclose(y_a, y_b), "c does not reach adaLN_modulation at all"


# =====================================================================
# Arm 1 -- only the three residual gates act on their own
# =====================================================================


@pytest.mark.parametrize("index", range(NUM_ADALN_CHUNKS))
def test_only_the_three_residual_gates_act_when_every_other_chunk_is_zero(index):
    """A shift/scale chunk is invisible while its gate is zero; a gate is not.

    This is the arm that separates the twelve chunks into
    ``{gate_msa, gate_xa, gate_mlp}`` and the other nine, purely positionally.
    Swapping triple 2 with triple 3 moves ``gate_xa`` from chunk 5 to chunk 8
    and puts the inert ``gate_cond`` at chunk 5, which fires here twice.
    """
    block = _make_block()
    x, c, cond = _inputs()
    name = ADALN_CHUNK_NAMES[index]

    bias = np.zeros((NUM_ADALN_CHUNKS * HIDDEN,), dtype="float32")
    bias[_chunk_slice(index)] = 0.9
    y = _run(block, bias, x, c, cond)

    is_residual_gate = name in ("gate_msa", "gate_xa", "gate_mlp")
    moved = not np.array_equal(y, x)

    if is_residual_gate:
        assert moved, (
            f"chunk {index} is named {name!r}, so opening it alone must add its "
            "sub-op's residual; the output is bit-identical to x instead"
        )
    else:
        assert not moved, (
            f"chunk {index} is named {name!r}; with every gate at zero no "
            "residual can be added, yet the output moved by "
            f"{np.max(np.abs(y - x))}"
        )


# =====================================================================
# Arm 2 -- each shift/scale chunk belongs to exactly one open gate
# =====================================================================


@pytest.mark.parametrize("index", range(NUM_ADALN_CHUNKS))
def test_each_shift_and_scale_chunk_moves_only_its_own_sub_op(index):
    """Open one gate at a time; perturb chunk ``index``; record which paths moved.

    The expected 12x3 pattern is :data:`EXPECTED_SENSITIVITY`, written by hand
    from the chunk order. ``gate_cond`` is the row that must be empty.
    """
    block = _make_block()
    x, c, cond = _inputs()

    observed = set()
    for path, gate_index in GATE_INDEX.items():
        base = np.zeros((NUM_ADALN_CHUNKS * HIDDEN,), dtype="float32")
        base[_chunk_slice(gate_index)] = 1.0
        y_base = _run(block, base, x, c, cond)

        perturbed = base.copy()
        perturbed[_chunk_slice(index)] = perturbed[_chunk_slice(index)] + 0.7
        y_pert = _run(block, perturbed, x, c, cond)

        if not np.array_equal(y_base, y_pert):
            observed.add(path)

    assert observed == EXPECTED_SENSITIVITY[index], (
        f"chunk {index} ({ADALN_CHUNK_NAMES[index]!r}) moved the output under "
        f"open gates {sorted(observed)}, expected exactly "
        f"{sorted(EXPECTED_SENSITIVITY[index])}"
    )


# =====================================================================
# Arm 3 -- gate_cond gates no residual, ever
# =====================================================================


@pytest.mark.parametrize("value", [-3.0, -0.25, 0.5, 2.0, 17.0])
def test_gate_cond_gates_no_residual_at_all(value):
    """From a fully populated modulation vector, moving ``gate_cond`` changes nothing.

    This holds the whole conditioning-stream modulation fixed (``shift_cond`` and
    ``scale_cond`` keep their values) and moves ONLY the ninth chunk, so a
    bit-identical result is evidence about that chunk alone. It is the arm that
    PROVES the unused-gate fact instead of assuming it.
    """
    block = _make_block()
    x, c, cond = _inputs()
    rng = np.random.default_rng(99)
    base = rng.normal(scale=0.6, size=(NUM_ADALN_CHUNKS * HIDDEN,)).astype("float32")

    y_base = _run(block, base, x, c, cond)
    perturbed = base.copy()
    perturbed[_chunk_slice(ADALN_CHUNK_NAMES.index("gate_cond"))] = value
    y_pert = _run(block, perturbed, x, c, cond)

    assert np.array_equal(y_base, y_pert), (
        "gate_cond is emitted by the 12-way split and consumed by NO residual "
        f"add; setting it to {value} moved the output by "
        f"{np.max(np.abs(y_pert - y_base))}"
    )


def test_every_other_chunk_of_a_populated_vector_does_move_the_output():
    """Anti-vacuity for the arm above: the other 11 chunks are all live here."""
    block = _make_block()
    x, c, cond = _inputs()
    rng = np.random.default_rng(99)
    base = rng.normal(scale=0.6, size=(NUM_ADALN_CHUNKS * HIDDEN,)).astype("float32")
    y_base = _run(block, base, x, c, cond)

    inert = []
    for index in range(NUM_ADALN_CHUNKS):
        perturbed = base.copy()
        perturbed[_chunk_slice(index)] = perturbed[_chunk_slice(index)] + 0.85
        if np.array_equal(_run(block, perturbed, x, c, cond), y_base):
            inert.append(ADALN_CHUNK_NAMES[index])

    assert inert == ["gate_cond"], (
        f"exactly one chunk may be inert from a populated vector, got {inert}"
    )


# =====================================================================
# Arm 4 -- triple 2 modulates x, triple 3 modulates cond_tokens
# =====================================================================


def test_scale_xa_dies_on_a_channel_constant_x_while_scale_cond_lives():
    """``norm_cross(x) == 0`` kills ``scale_xa`` and nothing else in the xa residual.

    A row of ``x`` that is constant across the channel axis normalises to exactly
    zero under ``LayerNormalization(center=False, scale=False)``, and
    ``modulate(0, shift, scale) = 0 * (1 + scale) + shift`` drops ``scale``
    entirely. That is a property of the QUERY stream only: ``cond_tokens`` is
    still generic, so ``scale_cond`` keeps acting. Swapping triples 2 and 3
    exchanges the two verdicts; applying triple 3's ``modulate`` to ``x`` instead
    of ``cond_tokens`` kills ``scale_cond`` too.
    """
    block = _make_block()
    _, c, cond = _inputs()
    rng = np.random.default_rng(4242)
    row = rng.normal(size=(BATCH, SEQ, 1)).astype("float32")
    x_flat = np.repeat(row, HIDDEN, axis=2)  # constant across channels

    base = _bias_with(gate_xa=1.0, shift_xa=0.3, shift_cond=0.2)
    y_base = _run(block, base, x_flat, c, cond)

    y_scale_xa = _run(
        block, _bias_with(gate_xa=1.0, shift_xa=0.3, shift_cond=0.2, scale_xa=1.5),
        x_flat, c, cond,
    )
    y_scale_cond = _run(
        block, _bias_with(gate_xa=1.0, shift_xa=0.3, shift_cond=0.2, scale_cond=1.5),
        x_flat, c, cond,
    )

    assert np.array_equal(y_base, y_scale_xa), (
        "triple 2 must modulate norm_cross(x); on a channel-constant x that "
        "normed stream is exactly zero, so scale_xa cannot act -- yet the "
        f"output moved by {np.max(np.abs(y_scale_xa - y_base))}"
    )
    assert not np.array_equal(y_base, y_scale_cond), (
        "triple 3 must modulate norm_cond(cond_tokens), which is generic here, "
        "so scale_cond must act -- yet the output is bit-identical"
    )


def test_the_block_output_depends_on_cond_tokens():
    """The conditioning stream must reach the cross-attention K/V at all."""
    block = _make_block()
    x, c, cond_a = _inputs()
    rng = np.random.default_rng(31337)
    cond_b = rng.normal(size=(BATCH, SEQ, HIDDEN)).astype("float32")
    assert not np.array_equal(cond_a, cond_b)

    bias = rng.normal(scale=0.6, size=(NUM_ADALN_CHUNKS * HIDDEN,)).astype("float32")
    y_a = _run(block, bias, x, c, cond_a)
    y_b = _run(block, bias, x, c, cond_b)

    assert not np.array_equal(y_a, y_b), (
        "cond_tokens is unused by the block -- triple 3 must modulate "
        "norm_cond(cond_tokens), not norm_cond(x)"
    )


# =====================================================================
# Arm 5 -- residual order: msa -> xa -> mlp
# =====================================================================


def _single_gate_bias(base: np.ndarray, path: str) -> np.ndarray:
    """Return ``base`` with the two gates other than ``path``'s zeroed."""
    vec = base.copy()
    for other, gate_index in GATE_INDEX.items():
        if other != path:
            vec[_chunk_slice(gate_index)] = 0.0
    return vec


def test_the_residual_order_is_msa_then_xa_then_mlp():
    """Chaining one-gate-open runs in the true order reproduces the combined run.

    Each residual reads the CURRENT ``x``, so running the block three times with
    a single gate open each is algebraically the same computation as one run with
    all three open -- but only when the three runs are chained in the block's own
    order. This uses the block as its own oracle: nothing here recomputes
    ``attn``, ``cross_attn`` or ``mlp``.
    """
    block = _make_block()
    x, c, cond = _inputs()
    rng = np.random.default_rng(555)
    base = rng.normal(scale=0.5, size=(NUM_ADALN_CHUNKS * HIDDEN,)).astype("float32")
    for gate_index in GATE_INDEX.values():
        base[_chunk_slice(gate_index)] = 0.75

    y_all = _run(block, base, x, c, cond)

    def chain(order):
        z = x
        for path in order:
            z = _run(block, _single_gate_bias(base, path), z, c, cond)
        return z

    true_order = ("msa", "xa", "mlp")
    assert np.array_equal(chain(true_order), y_all), (
        "composing the three single-gate runs in the order msa -> xa -> mlp must "
        "reproduce the all-gates run bit-identically; it does not, so the block's "
        "residual adds are not in that order"
    )

    mismatched = [
        perm
        for perm in itertools.permutations(true_order)
        if perm != true_order and not np.array_equal(chain(perm), y_all)
    ]
    assert len(mismatched) == 5, (
        "anti-vacuity: all five wrong orders must disagree with the combined run "
        f"(otherwise the sub-ops commute and this arm proves nothing); {5 - len(mismatched)} "
        "of them agreed"
    )


# =====================================================================
# The 1/sqrt(head_dim) scale is applied exactly once
# =====================================================================


def test_the_attention_scale_lives_only_in_the_reused_attention_layers():
    """Upstream's ``CrossAttention.scale`` is dead code; the port must not revive it.

    Both attention sub-layers already apply ``1 / sqrt(head_dim)`` internally. A
    second, block-level scale would make the effective temperature
    ``head_dim ** -1``.
    """
    block = _make_block()
    expected = 1.0 / math.sqrt(HEAD_DIM)

    assert block.attn.scale == pytest.approx(expected)
    assert block.cross_attn.scale == pytest.approx(expected)
    assert not hasattr(block, "scale"), (
        "DiTXABlock must not carry its own attention scale -- the reused "
        "MultiHeadCrossAttention layers apply it, exactly once each"
    )


# =====================================================================
# House-shape guards for the new module
# =====================================================================


def test_the_chunk_name_table_is_the_documented_order():
    assert ADALN_CHUNK_NAMES == (
        "shift_msa", "scale_msa", "gate_msa",
        "shift_xa", "scale_xa", "gate_xa",
        "shift_cond", "scale_cond", "gate_cond",
        "shift_mlp", "scale_mlp", "gate_mlp",
    )
    assert NUM_ADALN_CHUNKS == len(ADALN_CHUNK_NAMES) == 12


def test_every_normalization_epsilon_is_one_e_minus_six():
    """The bare-Keras 1e-3 default is a silent 1000x error with no shape symptom."""
    block = _make_block()
    for name in ("norm1", "norm_cross", "norm_cond", "norm2"):
        norm = getattr(block, name)
        assert norm.epsilon == 1e-6, f"{name}.epsilon == {norm.epsilon}"
        assert norm.center is False, f"{name} must be non-affine (center)"
        assert norm.scale is False, f"{name} must be non-affine (scale)"
    for attn_name in ("attn", "cross_attn"):
        attn = getattr(block, attn_name)
        assert attn.qk_norm_type == "rms_norm"
        assert attn.q_norm.use_scale is False
        assert attn.k_norm.use_scale is False
        assert attn.q_norm.epsilon == 1e-6
        assert attn.k_norm.epsilon == 1e-6


def test_no_initializer_instance_is_shared_between_the_two_zero_inits():
    block = _make_block()
    assert block.adaln_dense.kernel_initializer is not block.adaln_dense.bias_initializer


def test_compute_output_shape_returns_the_query_stream_shape():
    block = DiTXABlock(hidden_size=HIDDEN, num_heads=NUM_HEADS)
    shape = block.compute_output_shape(
        [(None, SEQ, HIDDEN), (None, HIDDEN), (None, SEQ, HIDDEN)]
    )
    assert shape == (None, SEQ, HIDDEN)


def test_get_config_round_trips_every_knob():
    block = DiTXABlock(
        hidden_size=16,
        num_heads=4,
        mlp_ratio=3.0,
        norm_epsilon=1e-6,
        qk_norm_epsilon=1e-6,
        use_bias=False,
        name="rt",
    )
    config = block.get_config()
    clone = DiTXABlock.from_config(config)
    assert clone.get_config() == config
    for key in (
        "hidden_size",
        "num_heads",
        "mlp_ratio",
        "norm_epsilon",
        "qk_norm_epsilon",
        "use_bias",
    ):
        assert getattr(clone, key) == getattr(block, key)


def test_build_rejects_an_input_shape_that_is_not_a_triple():
    block = DiTXABlock(hidden_size=HIDDEN, num_heads=NUM_HEADS)
    with pytest.raises(ValueError, match="triple"):
        block.build((None, SEQ, HIDDEN))


def test_a_forward_pass_builds_the_block_lazily_to_the_same_weight_signature():
    lazy = DiTXABlock(hidden_size=HIDDEN, num_heads=NUM_HEADS, mlp_ratio=2.0, name="lz")
    x, c, cond = _inputs()
    lazy([x, c, cond], training=False)
    explicit = _make_block()

    lazy_paths = sorted((w.path.split("/", 1)[1], tuple(w.shape)) for w in lazy.weights)
    explicit_paths = sorted(
        (w.path.split("/", 1)[1], tuple(w.shape)) for w in explicit.weights
    )
    assert lazy_paths == explicit_paths
