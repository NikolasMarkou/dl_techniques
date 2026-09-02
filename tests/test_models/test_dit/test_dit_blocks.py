"""``DiTBlock`` / ``DiTFinalLayer``: chunk attribution, non-causality, and the tanh GELU.

Three properties of these two layers are invisible to every conventional test
and each gets a dedicated arm here.

1. **The 6-way chunk order.** ``adaLN_modulation(c)`` emits ``6 * hidden_size``
   numbers split into six ``(B, hidden)`` chunks consumed in the order::

       shift_msa, scale_msa, gate_msa,   # triple 1 -> self-attention on x
       shift_mlp, scale_mlp, gate_mlp    # triple 2 -> the MLP on x

   A permutation of that order changes NOTHING observable by shape, by
   parameter count, by ``get_config()`` or by a save/load round trip -- a
   reversed permutation is still an exact bijection. It changes only which
   learned scalar multiplies which sub-op, so it trains to a different, wrong
   model under a fully green conventional suite. The probe here writes a
   one-chunk-at-a-time pattern into the modulation ``Dense``'s **bias** and asks
   only *which sub-op moved*. No expected value is ever computed from the
   block's own formula; every assertion is a changed / bit-identical comparison
   between two runs of the block itself.

2. **The attention is non-causal.** A causal mask changes no shape, no
   parameter count and no config; it only makes a later patch invisible to an
   earlier one. Pinned by a three-arm probe: perturbing a LATER token must move
   an EARLIER token's output (the claim), perturbing the later token must move
   its own output (the perturbation is live), and the same probe on an
   all-gates-closed block must move nothing (the probe reads the attention
   path, not the residual).

3. **The MLP is the tanh-approximate GELU**, not the exact one. Pinned by
   reconstructing the FFN from its own ``fc1`` / ``fc2`` weights under both
   activations and asserting the layer matches the approximate one and
   measurably differs from the exact one.

The premise all of this rests on: the modulation ``Dense`` is zero in both
kernel AND bias, so at initialisation every chunk is exactly zero, every
``modulate`` is the identity and both gates are ``0`` -- ``DiTBlock`` is the
exact identity map on ``x`` and ``DiTFinalLayer`` emits exactly ``0.0``. That
premise is itself asserted, at ``atol=0``.
"""

from typing import List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.dit.blocks import (
    DIT_ADALN_CHUNK_NAMES,
    DIT_FINAL_CHUNK_NAMES,
    NUM_DIT_ADALN_CHUNKS,
    NUM_DIT_FINAL_CHUNKS,
    DiTBlock,
    DiTFinalLayer,
)

# ---------------------------------------------------------------------
# Fixture geometry -- small enough to be instant, large enough to be generic
# ---------------------------------------------------------------------

BATCH = 2
SEQ = 5
HIDDEN = 8
NUM_HEADS = 2
PATCH = 2
OUT_CHANNELS = 3
SEED = 20260902

#: Chunk index of each residual gate, by the sub-op it gates. Read off
#: DIT_ADALN_CHUNK_NAMES by hand, never by running the implementation.
GATE_INDEX = {"msa": 2, "mlp": 5}

#: For every chunk index, the set of open-gate paths under which perturbing it
#: must move the block output. Derived from the chunk ORDER by hand.
EXPECTED_SENSITIVITY = {
    0: {"msa"},           # shift_msa
    1: {"msa"},           # scale_msa
    2: {"msa", "mlp"},    # gate_msa -- opens its own path, and moves the MLP's input
    3: {"mlp"},           # shift_mlp
    4: {"mlp"},           # scale_mlp
    5: {"msa", "mlp"},    # gate_mlp -- opens its own path from anywhere
}


def _make_block(seed: int = SEED, **kwargs) -> DiTBlock:
    """Return a built block with reproducible, non-degenerate sub-layer weights."""
    keras.utils.set_random_seed(seed)
    block = DiTBlock(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        mlp_ratio=2.0,
        name="probe_block",
        **kwargs,
    )
    block.build([(None, SEQ, HIDDEN), (None, HIDDEN)])
    return block


def _make_final(seed: int = SEED, **kwargs) -> DiTFinalLayer:
    """Return a built final layer with reproducible sub-layer weights."""
    keras.utils.set_random_seed(seed)
    final = DiTFinalLayer(
        hidden_size=HIDDEN,
        patch_size=PATCH,
        out_channels=OUT_CHANNELS,
        name="probe_final",
        **kwargs,
    )
    final.build([(None, SEQ, HIDDEN), (None, HIDDEN)])
    return final


def _inputs(seed: int = SEED + 1) -> Tuple[np.ndarray, np.ndarray]:
    """Return a generic ``(x, c)`` pair as numpy arrays."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(BATCH, SEQ, HIDDEN)).astype("float32")
    c = rng.normal(size=(BATCH, HIDDEN)).astype("float32")
    return x, c


def _chunk_slice(index: int) -> slice:
    return slice(index * HIDDEN, (index + 1) * HIDDEN)


def _bias_with(**chunks: float) -> np.ndarray:
    """Build a ``(6 * hidden,)`` bias vector with the named chunks set flat.

    :param chunks: ``chunk_name=value`` pairs; every other chunk stays zero.
    """
    vec = np.zeros((NUM_DIT_ADALN_CHUNKS * HIDDEN,), dtype="float32")
    for name, value in chunks.items():
        vec[_chunk_slice(DIT_ADALN_CHUNK_NAMES.index(name))] = value
    return vec


def _run(block: DiTBlock, bias: np.ndarray, x, c) -> np.ndarray:
    """Set the modulation bias, run the block once, return numpy."""
    block.adaln.linear.bias.assign(keras.ops.convert_to_tensor(bias))
    return np.asarray(keras.ops.convert_to_numpy(block([x, c], training=False)))


# =====================================================================
# The premise: identity at init
# =====================================================================


class TestTheIdentityAtInitPremise:
    """Both layers are exactly inert before training, at ``atol=0``."""

    def test_the_block_is_the_exact_identity_at_init(self):
        block = _make_block()
        x, c = _inputs()
        out = keras.ops.convert_to_numpy(block([x, c], training=False))
        # atol=0: the zero gates make this an EXACT statement, not a numerical one.
        np.testing.assert_array_equal(out, x)

    def test_the_final_layer_emits_exactly_zero_at_init(self):
        final = _make_final()
        x, c = _inputs()
        out = keras.ops.convert_to_numpy(final([x, c], training=False))
        assert out.shape == (BATCH, SEQ, PATCH * PATCH * OUT_CHANNELS)
        np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_the_modulation_dense_is_zero_in_kernel_and_bias(self):
        block = _make_block()
        final = _make_final()
        for dense in (block.adaln.linear, final.adaln.linear, final.linear):
            k = keras.ops.convert_to_numpy(dense.kernel)
            b = keras.ops.convert_to_numpy(dense.bias)
            np.testing.assert_array_equal(k, np.zeros_like(k))
            np.testing.assert_array_equal(b, np.zeros_like(b))


# =====================================================================
# 1. Per-chunk attribution
# =====================================================================


class TestTheSixWayModulationIsWired:
    """Every chunk reaches the sub-op its name claims, and no other."""

    def test_the_chunk_name_table_matches_the_projection_width(self):
        block = _make_block()
        assert NUM_DIT_ADALN_CHUNKS == 6
        assert len(DIT_ADALN_CHUNK_NAMES) == 6
        assert block.adaln.linear.units == NUM_DIT_ADALN_CHUNKS * HIDDEN

    def test_only_the_two_residual_gates_act_when_every_other_chunk_is_zero(self):
        """With all chunks at zero, exactly the gate chunks can move the output."""
        block = _make_block()
        x, c = _inputs()
        base = _run(block, _bias_with(), x, c)

        moved = set()
        for index, name in enumerate(DIT_ADALN_CHUNK_NAMES):
            out = _run(block, _bias_with(**{name: 1.0}), x, c)
            if not np.array_equal(out, base):
                moved.add(name)

        assert moved == {"gate_msa", "gate_mlp"}, (
            "with every gate closed, only a gate chunk can open a path; "
            f"these moved the output: {sorted(moved)}"
        )

    @pytest.mark.parametrize("chunk_index", range(NUM_DIT_ADALN_CHUNKS))
    @pytest.mark.parametrize("open_gate", sorted(GATE_INDEX))
    def test_each_chunk_moves_only_its_own_sub_op(self, chunk_index, open_gate):
        """With exactly one gate open, chunk ``k`` moves the output iff it belongs there."""
        block = _make_block()
        x, c = _inputs()
        gate_name = DIT_ADALN_CHUNK_NAMES[GATE_INDEX[open_gate]]
        chunk_name = DIT_ADALN_CHUNK_NAMES[chunk_index]

        base = _run(block, _bias_with(**{gate_name: 1.0}), x, c)
        perturbed = _run(
            block,
            _bias_with(**{gate_name: 1.0, chunk_name: 0.7})
            if chunk_name != gate_name
            else _bias_with(**{gate_name: 1.7}),
            x,
            c,
        )
        did_move = not np.array_equal(base, perturbed)
        should_move = open_gate in EXPECTED_SENSITIVITY[chunk_index]

        assert did_move == should_move, (
            f"chunk {chunk_index} ({chunk_name}) with the {open_gate} gate open: "
            f"moved={did_move}, expected={should_move}"
        )

    def test_the_residual_order_is_msa_then_mlp(self):
        """Running one gate at a time in msa->mlp order reproduces the joint run bit-exactly."""
        block = _make_block()
        x, c = _inputs()

        joint = _run(block, _bias_with(gate_msa=1.0, gate_mlp=1.0), x, c)

        def sequential(order: List[str]) -> np.ndarray:
            state = x
            for gate in order:
                name = DIT_ADALN_CHUNK_NAMES[GATE_INDEX[gate]]
                state = _run(block, _bias_with(**{name: 1.0}), state, c)
            return state

        np.testing.assert_array_equal(sequential(["msa", "mlp"]), joint)

        # Anti-vacuity: the two sub-ops must NOT commute, otherwise the claim
        # above is unobservable.
        assert not np.array_equal(sequential(["mlp", "msa"]), joint), (
            "the msa and mlp branches commuted, so the residual-order claim is "
            "vacuous -- this test would pass under either order"
        )


# =====================================================================
# 2. Non-causality
# =====================================================================


class TestTheAttentionIsNonCausal:
    """A later token must be able to influence an earlier one.

    The perturbation is deliberately NON-uniform across the channel axis. Adding
    a constant to every channel of one token is annihilated by ``norm1``:
    ``keras.layers.LayerNormalization`` subtracts the mean regardless of
    ``center=False``, so a uniform bump never reaches the attention at all. The
    first draft of this probe used ``+= 5.0`` and measured a delta of 1.2e-07 --
    a false RED. :func:`test_a_uniform_bump_is_annihilated_by_norm1` pins that.
    """

    #: A fixed non-uniform per-channel bump, large enough to be unmistakable.
    PERTURBATION = np.arange(1, HIDDEN + 1, dtype="float32")

    @staticmethod
    def _open_msa(block: DiTBlock) -> None:
        block.adaln.linear.bias.assign(
            keras.ops.convert_to_tensor(_bias_with(gate_msa=1.0))
        )

    def test_a_uniform_bump_is_annihilated_by_norm1(self):
        """Instrument check: why this probe cannot use a constant offset."""
        block = _make_block()
        self._open_msa(block)
        x, c = _inputs()
        x2 = x.copy()
        x2[:, -1, :] += 5.0

        out = keras.ops.convert_to_numpy(block([x, c], training=False))
        out2 = keras.ops.convert_to_numpy(block([x2, c], training=False))
        assert float(np.max(np.abs(out2[:, 0, :] - out[:, 0, :]))) < 1e-5

    def test_perturbing_a_later_token_moves_an_earlier_tokens_output(self):
        block = _make_block()
        self._open_msa(block)
        x, c = _inputs()

        x2 = x.copy()
        x2[:, -1, :] += self.PERTURBATION

        out = keras.ops.convert_to_numpy(block([x, c], training=False))
        out2 = keras.ops.convert_to_numpy(block([x2, c], training=False))

        delta_first = float(np.max(np.abs(out2[:, 0, :] - out[:, 0, :])))
        assert delta_first > 1e-5, (
            "token 0's output did not move when the LAST token changed: the "
            f"attention is causal (max |delta| = {delta_first})"
        )

    def test_the_perturbation_is_live_at_the_perturbed_token(self):
        """Control: the last token's own output must move (the probe is not dead)."""
        block = _make_block()
        self._open_msa(block)
        x, c = _inputs()
        x2 = x.copy()
        x2[:, -1, :] += self.PERTURBATION

        out = keras.ops.convert_to_numpy(block([x, c], training=False))
        out2 = keras.ops.convert_to_numpy(block([x2, c], training=False))
        assert float(np.max(np.abs(out2[:, -1, :] - out[:, -1, :]))) > 1e-5

    def test_the_probe_reads_the_attention_path_not_the_residual(self):
        """Anti-vacuity: with the msa gate CLOSED, token 0 must not move at all."""
        block = _make_block()  # gates closed -> exact identity
        x, c = _inputs()
        x2 = x.copy()
        x2[:, -1, :] += self.PERTURBATION

        out = keras.ops.convert_to_numpy(block([x, c], training=False))
        out2 = keras.ops.convert_to_numpy(block([x2, c], training=False))
        np.testing.assert_array_equal(out2[:, 0, :], out[:, 0, :])


# =====================================================================
# 3. The MLP activation
# =====================================================================


class TestTheMlpIsTanhApproximateGelu:
    """The block's FFN is ``gelu(approximate=True)``, not exact GELU.

    Verified by reconstructing the FFN from its OWN ``fc1`` / ``fc2`` weights
    under both activations: the layer must equal the approximate reconstruction
    and measurably differ from the exact one. The second half is the
    discriminator -- without it the test passes under either activation.
    """

    def test_the_ffn_matches_the_approximate_gelu_and_not_the_exact_one(self):
        block = _make_block()
        rng = np.random.default_rng(SEED + 7)
        # Scaled up so the two GELUs are separated well above float32 noise;
        # they agree to ~1e-3 near the origin.
        z = (rng.normal(size=(BATCH, SEQ, HIDDEN)) * 3.0).astype("float32")

        out = keras.ops.convert_to_numpy(block.mlp(z, training=False))
        h = block.mlp.fc1(z)
        approx = keras.ops.convert_to_numpy(
            block.mlp.fc2(keras.activations.gelu(h, approximate=True))
        )
        exact = keras.ops.convert_to_numpy(
            block.mlp.fc2(keras.activations.gelu(h, approximate=False))
        )

        np.testing.assert_allclose(out, approx, rtol=0, atol=1e-6)

        separation = float(np.max(np.abs(approx - exact)))
        assert separation > 1e-5, (
            "the two GELU variants are indistinguishable on this probe input, "
            f"so the assertion below is vacuous (max |delta| = {separation})"
        )
        assert float(np.max(np.abs(out - exact))) > 1e-5, (
            "the block's FFN matches EXACT gelu -- it is not the tanh "
            "approximation upstream uses (reference/models.py:104)"
        )


# =====================================================================
# The final layer's chunk order (D-011)
# =====================================================================


class TestTheFinalLayerChunkOrderIsScaleFirst:
    """``AdaLayerNormContinuous`` splits ``scale, shift`` -- upstream names it the other way.

    This is D-011: with a zero-init kernel and bias the two orders are the same
    function class under a permutation of the Dense's output units, so nothing
    observable differs except which slice holds which role. Pinned so the
    divergence is explicit rather than silent.
    """

    def test_chunk_zero_is_the_multiplicative_scale(self):
        final = _make_final()
        x, c = _inputs()

        bias = np.zeros((NUM_DIT_FINAL_CHUNKS * HIDDEN,), dtype="float32")
        bias[0:HIDDEN] = 1.0  # DIT_FINAL_CHUNK_NAMES[0]
        final.adaln.linear.bias.assign(keras.ops.convert_to_tensor(bias))

        # `linear` is zero-init, so give it an identity-ish kernel to observe.
        kernel = np.eye(HIDDEN, final.output_dim, dtype="float32")
        final.linear.kernel.assign(keras.ops.convert_to_tensor(kernel))

        with_chunk0 = keras.ops.convert_to_numpy(final([x, c], training=False))

        bias[:] = 0.0
        final.adaln.linear.bias.assign(keras.ops.convert_to_tensor(bias))
        plain = keras.ops.convert_to_numpy(final([x, c], training=False))

        # scale=1 doubles a modulated norm(x): out = norm(x)*(1+1) = 2*norm(x).
        np.testing.assert_allclose(
            with_chunk0[..., :HIDDEN], 2.0 * plain[..., :HIDDEN], rtol=0, atol=1e-5
        )
        assert DIT_FINAL_CHUNK_NAMES == ("scale", "shift")

    def test_chunk_one_is_the_additive_shift(self):
        final = _make_final()
        x, c = _inputs()

        bias = np.zeros((NUM_DIT_FINAL_CHUNKS * HIDDEN,), dtype="float32")
        bias[HIDDEN : 2 * HIDDEN] = 1.0  # DIT_FINAL_CHUNK_NAMES[1]
        final.adaln.linear.bias.assign(keras.ops.convert_to_tensor(bias))
        kernel = np.eye(HIDDEN, final.output_dim, dtype="float32")
        final.linear.kernel.assign(keras.ops.convert_to_tensor(kernel))
        with_chunk1 = keras.ops.convert_to_numpy(final([x, c], training=False))

        bias[:] = 0.0
        final.adaln.linear.bias.assign(keras.ops.convert_to_tensor(bias))
        plain = keras.ops.convert_to_numpy(final([x, c], training=False))

        # shift=1 adds exactly 1 per channel before the identity projection.
        np.testing.assert_allclose(
            with_chunk1[..., :HIDDEN] - plain[..., :HIDDEN],
            np.ones_like(plain[..., :HIDDEN]),
            rtol=0,
            atol=1e-5,
        )


# =====================================================================
# Standard v2 §16.3 layer coverage
# =====================================================================


class TestConstructionValidation:
    """Every documented ``ValueError`` fires."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"hidden_size": 0, "num_heads": 2},
            {"hidden_size": -8, "num_heads": 2},
            {"hidden_size": 8, "num_heads": 0},
            {"hidden_size": 8, "num_heads": 3},  # not divisible
            {"hidden_size": 8, "num_heads": 2, "mlp_ratio": 0.0},
            {"hidden_size": 8, "num_heads": 2, "norm_epsilon": 0.0},
            {"hidden_size": 8, "num_heads": 2, "dropout_rate": 1.0},
        ],
    )
    def test_the_block_rejects_an_illegal_config(self, kwargs):
        with pytest.raises(ValueError):
            DiTBlock(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"hidden_size": 0, "patch_size": 2, "out_channels": 3},
            {"hidden_size": 8, "patch_size": 0, "out_channels": 3},
            {"hidden_size": 8, "patch_size": 2, "out_channels": 0},
            {"hidden_size": 8, "patch_size": 2, "out_channels": 3, "norm_epsilon": -1.0},
        ],
    )
    def test_the_final_layer_rejects_an_illegal_config(self, kwargs):
        with pytest.raises(ValueError):
            DiTFinalLayer(**kwargs)

    @pytest.mark.parametrize("cls", [DiTBlock, DiTFinalLayer])
    def test_build_rejects_a_bare_single_shape(self, cls):
        layer = (
            DiTBlock(hidden_size=HIDDEN, num_heads=NUM_HEADS)
            if cls is DiTBlock
            else DiTFinalLayer(
                hidden_size=HIDDEN, patch_size=PATCH, out_channels=OUT_CHANNELS
            )
        )
        with pytest.raises(ValueError, match="pair"):
            layer.build((None, SEQ, HIDDEN))


class TestConfigAndShapes:
    """``get_config`` round-trips every ctor arg; shapes derive from the config."""

    def test_the_block_config_round_trips(self):
        block = DiTBlock(
            hidden_size=16,
            num_heads=4,
            mlp_ratio=3.0,
            norm_epsilon=1e-5,
            dropout_rate=0.1,
            use_bias=False,
            name="cfg_block",
        )
        config = block.get_config()
        for key, value in {
            "hidden_size": 16,
            "num_heads": 4,
            "mlp_ratio": 3.0,
            "norm_epsilon": 1e-5,
            "dropout_rate": 0.1,
            "use_bias": False,
        }.items():
            assert config[key] == value, key
        clone = DiTBlock.from_config(config)
        assert clone.get_config() == config

    def test_the_final_layer_config_round_trips(self):
        final = DiTFinalLayer(
            hidden_size=16,
            patch_size=4,
            out_channels=5,
            norm_epsilon=1e-5,
            use_bias=False,
            name="cfg_final",
        )
        config = final.get_config()
        for key, value in {
            "hidden_size": 16,
            "patch_size": 4,
            "out_channels": 5,
            "norm_epsilon": 1e-5,
            "use_bias": False,
        }.items():
            assert config[key] == value, key
        clone = DiTFinalLayer.from_config(config)
        assert clone.get_config() == config

    def test_compute_output_shape_on_an_unbuilt_block(self):
        block = DiTBlock(hidden_size=HIDDEN, num_heads=NUM_HEADS)
        assert not block.built
        assert block.compute_output_shape(
            [(None, SEQ, HIDDEN), (None, HIDDEN)]
        ) == (None, SEQ, HIDDEN)
        assert not block.built

    def test_compute_output_shape_on_an_unbuilt_final_layer(self):
        final = DiTFinalLayer(
            hidden_size=HIDDEN, patch_size=PATCH, out_channels=OUT_CHANNELS
        )
        assert not final.built
        assert final.compute_output_shape(
            [(None, SEQ, HIDDEN), (None, HIDDEN)]
        ) == (None, SEQ, PATCH * PATCH * OUT_CHANNELS)
        assert not final.built

    @pytest.mark.parametrize("seq", [1, 3, 17])
    @pytest.mark.parametrize("heads", [1, 2])
    def test_the_forward_shape_matches_compute_output_shape(self, seq, heads):
        keras.utils.set_random_seed(SEED)
        block = DiTBlock(hidden_size=HIDDEN, num_heads=heads, mlp_ratio=2.0)
        rng = np.random.default_rng(SEED)
        x = rng.normal(size=(BATCH, seq, HIDDEN)).astype("float32")
        c = rng.normal(size=(BATCH, HIDDEN)).astype("float32")
        out = block([x, c], training=False)
        assert tuple(out.shape) == (BATCH, seq, HIDDEN)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))


def _relative_paths(layer: keras.layers.Layer) -> set:
    """Weight paths relative to the owning layer's name."""
    prefix = layer.name + "/"
    out = set()
    for w in layer.weights:
        path = w.path
        out.add(path[len(prefix):] if path.startswith(prefix) else path)
    return out


class _BlockWithIncompleteBuild(DiTBlock):
    """Anti-vacuity sibling: a block whose ``build`` forgets the MLP."""

    def build(self, input_shape):
        if self.built:
            return
        x_shape, c_shape = input_shape[0], input_shape[1]
        self.adaln.build([tuple(x_shape), tuple(c_shape)])
        self.attn.build(query_shape=tuple(x_shape), value_shape=tuple(x_shape))
        self.norm2.build(tuple(x_shape))
        # self.mlp deliberately NOT built
        keras.layers.Layer.build(self, input_shape)


class TestBuildMaterializationParity:
    """``build()`` materializes exactly the sub-layer tree ``call()`` runs."""

    def test_the_block_build_matches_a_call_built_block(self):
        x, c = _inputs()

        explicit = _make_block()
        called = DiTBlock(
            hidden_size=HIDDEN, num_heads=NUM_HEADS, mlp_ratio=2.0,
            name="probe_block",
        )
        called([x, c], training=False)

        assert _relative_paths(explicit) == _relative_paths(called)
        assert len(explicit.weights) == len(called.weights)

    def test_the_final_layer_build_matches_a_call_built_layer(self):
        x, c = _inputs()

        explicit = _make_final()
        called = DiTFinalLayer(
            hidden_size=HIDDEN, patch_size=PATCH, out_channels=OUT_CHANNELS,
            name="probe_final",
        )
        called([x, c], training=False)

        assert _relative_paths(explicit) == _relative_paths(called)

    def test_the_parity_check_reddens_on_an_incomplete_build(self):
        """Anti-vacuity: the same comparison must FAIL for a build that skips a sub-layer."""
        x, c = _inputs()

        broken = _BlockWithIncompleteBuild(
            hidden_size=HIDDEN, num_heads=NUM_HEADS, mlp_ratio=2.0,
            name="probe_block",
        )
        broken.build([(None, SEQ, HIDDEN), (None, HIDDEN)])
        complete = _make_block()

        assert _relative_paths(broken) != _relative_paths(complete), (
            "the parity comparison could not tell a build that skips the MLP "
            "from a complete one -- it is vacuous"
        )


class TestSerializationRoundTrip:
    """A ``.keras`` round trip reproduces the forward output on VALUES."""

    @staticmethod
    def _model(layer: keras.layers.Layer, out_dim: int) -> keras.Model:
        x_in = keras.Input(shape=(SEQ, HIDDEN), name="x")
        c_in = keras.Input(shape=(HIDDEN,), name="c")
        return keras.Model(inputs=[x_in, c_in], outputs=layer([x_in, c_in]))

    def _round_trip(self, tmp_path, layer, out_dim, name):
        x, c = _inputs()
        model = self._model(layer, out_dim)

        # Move off the zero init so the round trip compares something.
        for w in model.weights:
            arr = keras.ops.convert_to_numpy(w)
            rng = np.random.default_rng(abs(hash(w.path)) % (2**32))
            w.assign(
                keras.ops.convert_to_tensor(
                    (arr + rng.normal(scale=0.1, size=arr.shape)).astype(arr.dtype)
                )
            )

        before = keras.ops.convert_to_numpy(model([x, c], training=False))
        path = tmp_path / f"{name}.keras"
        model.save(path)
        loaded = keras.models.load_model(path)

        # Weight VALUES before the loaded model's first call.
        for w0, w1 in zip(model.weights, loaded.weights):
            np.testing.assert_array_equal(
                keras.ops.convert_to_numpy(w0), keras.ops.convert_to_numpy(w1)
            )

        after = keras.ops.convert_to_numpy(loaded([x, c], training=False))
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-6)

    def test_the_block_round_trips(self, tmp_path):
        self._round_trip(tmp_path, _make_block(), HIDDEN, "block")

    def test_the_final_layer_round_trips(self, tmp_path):
        self._round_trip(
            tmp_path, _make_final(), PATCH * PATCH * OUT_CHANNELS, "final"
        )


@pytest.fixture
def restore_policy():
    original = keras.mixed_precision.global_policy()
    yield
    keras.mixed_precision.set_global_policy(original)


class TestPrecisionArms:
    """float32 control plus ``mixed_float16`` and ``float64`` arms."""

    @pytest.mark.parametrize(
        "policy,expected",
        [("float32", "float32"), ("mixed_float16", "float16"), ("float64", "float64")],
    )
    def test_the_block_runs_finite_under_each_policy(
        self, policy, expected, restore_policy
    ):
        keras.mixed_precision.set_global_policy(policy)
        block = _make_block()
        x, c = _inputs()
        out = block([x, c], training=False)
        assert keras.backend.standardize_dtype(out.dtype) == expected
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    @pytest.mark.parametrize(
        "policy,expected",
        [("float32", "float32"), ("mixed_float16", "float16"), ("float64", "float64")],
    )
    def test_the_final_layer_runs_finite_under_each_policy(
        self, policy, expected, restore_policy
    ):
        keras.mixed_precision.set_global_policy(policy)
        final = _make_final()
        x, c = _inputs()
        out = final([x, c], training=False)
        assert keras.backend.standardize_dtype(out.dtype) == expected
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))
