"""
Test suite for :class:`AdaLNZeroConditionalBlock` (``layers/transformers/adaln_zero.py``).

Templated on ``test_sd3_adaln.py`` (module-level constants, seeded fixture,
initialization / ctor-validation / forward-shape / ``compute_output_shape`` /
``get_config`` round trip / full ``.keras`` save+load / variable batch), adapted
to this block's ``inputs=[x, c]`` list-of-two-``(B,T,D)`` call convention.

**Why the SD3 template alone is not enough here.**
``adaLN_linear`` is zero-initialized in BOTH kernel and bias, so at init every
one of the six modulation streams is exactly ``0``: ``gate * anything == 0`` and
the block is the identity map in ``x``. A ``test_identity_at_init`` therefore
passes even if ``gate_msa``/``gate_mlp`` are swapped, if the 6-way ``ops.split``
is mis-ordered, or if ``mod`` is computed and thrown away. Identity-at-init is
kept below (it IS the documented contract) but is explicitly labelled vacuous
for wiring bugs; the wiring is pinned instead by four dead-component families:

1. :class:`TestModulationIsLive` — seeded NON-ZERO ``adaLN_linear`` weights must
   move the output off both ``x`` and a no-affine-LayerNorm reference.
2. :class:`TestSplitOrder` — with one sub-block's weights zeroed, each of the six
   modulation chunks must move the output IFF it belongs to the surviving half
   (exactly ``0.0`` movement otherwise), plus a numeric oracle on the modulated
   tensor actually handed to each sub-block.
3. :class:`TestFactoryAttentionDispatch` — a zeros-returning attention injected
   under an explicit ``attention_type`` must delete the attention residual and
   nothing else, proving ``_attn_via_factory`` selects real code.
4. :class:`TestCausalMaskContract` — the docstring's "``use_causal_mask`` is NOT
   forwarded to a factory-built attention layer" claim is MEASURED (identical
   outputs under the factory path, a live difference under the default path),
   not trusted.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.ffn.mlp import MLPBlock
from dl_techniques.layers.transformers.adaln_zero import AdaLNZeroConditionalBlock


DIM = 32
HEADS = 4
DIM_HEAD = 8
MLP_DIM = 64
N = 6
BATCH = 2
EPS = 1e-6

#: The order ``call()`` unpacks ``ops.split(mod, 6, axis=-1)`` into. Load-bearing:
#: chunks 0-2 drive the attention sub-block, chunks 3-5 the FFN sub-block.
CHUNK_ORDER = (
    "shift_msa", "scale_msa", "gate_msa",
    "shift_mlp", "scale_mlp", "gate_mlp",
)

#: The keys ``get_config()`` adds on top of ``keras.layers.Layer``'s own
#: (``name``, ``trainable``, ``dtype``). RE-DERIVED by reading the source: 15,
#: not the 14 the review notes claimed.
OWN_CONFIG_KEYS = frozenset({
    "dim", "num_heads", "dim_head", "mlp_dim", "dropout_rate", "use_causal_mask",
    "eps", "normalization_type", "normalization_args", "attention_type",
    "attention_args", "ffn_type", "ffn_args", "adaln_activation_type",
    "adaln_activation_args",
})

#: ``attention_args`` that satisfy ``create_attention_layer('multi_head', ...)``.
#: The block forwards ONLY ``attention_args`` to the factory (no implicit
#: ``dim``/``num_heads``), so these must be supplied by the caller.
FACTORY_ATTENTION_ARGS = {"dim": DIM, "num_heads": HEADS}


# ---------------------------------------------------------------------
# Helpers (test-local; not library abstractions)
# ---------------------------------------------------------------------


def _no_affine_layernorm(x_np: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Reference no-affine LayerNorm over the last axis (matches norm1/norm2)."""
    mean = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    return (x_np - mean) / np.sqrt(var + eps)


def _seed_all_weights(layer: keras.layers.Layer, seed: int) -> int:
    """Overwrite every weight of ``layer`` with a seeded NON-ZERO draw (I8).

    Default zero initializers make the gating sites structurally unobservable,
    and ``adaLN_linear`` here is zero-init by DESIGN, so every behavioural probe
    in this module must re-seed it first.

    NOTE (measured in this plan, step 4): ``keras.utils.set_random_seed`` does
    NOT make a subsequent ``keras.random.*`` draw reproducible on this backend.
    All randomness in this module therefore comes from ``np.random.RandomState``.

    :param layer: a BUILT layer.
    :param seed: numpy RNG seed.
    :return: the number of bias-like weights that received a non-zero value.
    """
    rng = np.random.RandomState(seed)
    values, non_zero_biases = [], 0
    for w in layer.weights:
        v = rng.normal(scale=0.25, size=tuple(w.shape)).astype("float32")
        name = getattr(w, "path", "") or w.name
        if "bias" in name and np.any(v != 0.0):
            non_zero_biases += 1
        values.append(v)
    layer.set_weights(values)
    assert non_zero_biases > 0, (
        "fixture is degenerate: no non-zero bias was seeded, so any bias-path "
        "defect would be invisible (invariant I8)"
    )
    return non_zero_biases


def _set_adaln_bias(block: AdaLNZeroConditionalBlock, groups: dict) -> None:
    """Drive the six modulation streams to EXACT constants.

    Zeroes ``adaLN_linear``'s kernel and writes ``groups`` into its bias, so
    ``mod`` no longer depends on ``c`` and each named chunk carries exactly the
    requested value. Chunks not named are exactly ``0.0``.

    :param block: a BUILT block.
    :param groups: mapping of a name in :data:`CHUNK_ORDER` to a float.
    """
    kernel, bias = block.adaLN_linear.get_weights()
    new_bias = np.zeros_like(bias)
    for name, value in groups.items():
        i = CHUNK_ORDER.index(name)
        new_bias[i * DIM:(i + 1) * DIM] = value
    block.adaLN_linear.set_weights([np.zeros_like(kernel), new_bias])


def _kill(layer: keras.layers.Layer) -> None:
    """Zero EVERY weight of ``layer``, making its output exactly ``0``.

    Both sub-blocks end in a Dense projection, so a fully-zeroed kernel AND bias
    forces an exact-zero output regardless of the input — which is what lets
    :class:`TestSplitOrder` attribute movement to one half at a time.
    """
    layer.set_weights([np.zeros_like(w) for w in layer.get_weights()])


class _SpyLayer(keras.layers.Layer):
    """Records the tensor (and ``use_causal_mask``) handed to it; returns zeros.

    Substituted for ``block.attn`` or ``block.mlp`` BEFORE the block is built
    (Keras forbids adding sub-layer state to an already-built layer). Tolerates
    both build conventions the block uses (``build(x_shape)`` for the factory
    attention path and the FFN, ``build(query_shape=..., value_shape=...,
    key_shape=...)`` for the default attention path) and both call conventions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen = {}

    def build(self, *args, **kwargs) -> None:
        self.built = True

    def call(self, inputs=None, query=None, value=None, key=None,
             use_causal_mask=None, training=None):
        h = query if query is not None else inputs
        self.seen["h"] = ops.convert_to_numpy(h)
        self.seen["use_causal_mask"] = use_causal_mask
        return ops.zeros_like(h)


def _make_block(seed=None, spy_on=None, **kwargs) -> AdaLNZeroConditionalBlock:
    """Construct, optionally spy-substitute a sub-layer, build, optionally seed.

    :param seed: if not None, every weight is re-seeded non-zero via
        :func:`_seed_all_weights` after the build.
    :param spy_on: ``'attn'`` or ``'mlp'`` — replace that sub-layer with a
        :class:`_SpyLayer` BEFORE the build.
    """
    block = AdaLNZeroConditionalBlock(
        dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM, **kwargs
    )
    if spy_on is not None:
        setattr(block, spy_on, _SpyLayer(name=f"spy_{spy_on}"))
    block.build([(BATCH, N, DIM), (BATCH, N, DIM)])
    if seed is not None:
        _seed_all_weights(block, seed)
    return block


@pytest.fixture
def sample():
    """Seeded ``(x, c)``, both ``(BATCH, N, DIM)`` — the block's call convention."""
    rng = np.random.RandomState(42)
    x = rng.normal(size=(BATCH, N, DIM)).astype("float32")
    c = rng.normal(size=(BATCH, N, DIM)).astype("float32")
    return x, c


# =====================================================================
# Construction, configuration, serialization
# =====================================================================


class TestConstructionAndConfig:

    def test_initialization(self):
        block = AdaLNZeroConditionalBlock(
            dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM
        )
        assert block.dim == DIM
        assert block.num_heads == HEADS
        assert block.dim_head == DIM_HEAD
        assert block.mlp_dim == MLP_DIM
        assert block.dropout_rate == 0.0
        assert block.use_causal_mask is True
        assert block.eps == 1e-6
        # The "Zero" of AdaLN-Zero: 6*dim units, zero kernel AND zero bias.
        assert block.adaLN_linear.units == 6 * DIM
        assert isinstance(
            block.adaLN_linear.kernel_initializer, keras.initializers.Zeros
        )
        assert isinstance(
            block.adaLN_linear.bias_initializer, keras.initializers.Zeros
        )
        block.build([(BATCH, N, DIM), (BATCH, N, DIM)])
        kernel, bias = block.adaLN_linear.get_weights()
        assert kernel.shape == (DIM, 6 * DIM)
        assert not np.any(kernel) and not np.any(bias)

    @pytest.mark.parametrize("bad", [
        {"dim": 0}, {"dim": -1}, {"num_heads": 0}, {"dropout_rate": 1.0},
        {"dropout_rate": -0.1},
    ])
    def test_ctor_raises_on_invalid_args(self, bad):
        kwargs = dict(dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD,
                      mlp_dim=MLP_DIM)
        kwargs.update(bad)
        with pytest.raises(ValueError):
            AdaLNZeroConditionalBlock(**kwargs)

    @pytest.mark.parametrize("shape_of_inputs", ["single", "three", "dict"])
    def test_call_rejects_anything_but_a_pair(self, sample, shape_of_inputs):
        x, c = sample
        block = _make_block()
        bad = {"single": x, "three": [x, c, x], "dict": {"x": x, "c": c}}[
            shape_of_inputs
        ]
        with pytest.raises(ValueError, match="inputs=\\[x, c\\]"):
            block(bad)

    def test_default_paths_build_the_bit_exact_original_components(self):
        """All four ``*_type=None`` defaults must reproduce DiT/LeWM exactly."""
        block = _make_block()
        # Normalization: Keras LayerNormalization with NO affine (AdaLN invariant).
        assert type(block.norm1) is keras.layers.LayerNormalization
        assert type(block.norm2) is keras.layers.LayerNormalization
        for norm in (block.norm1, block.norm2):
            assert norm.center is False and norm.scale is False
            assert norm.epsilon == EPS
        # Attention: keras.layers.MultiHeadAttention, NOT the dl_techniques
        # 'multi_head' factory entry (a different class with different defaults).
        assert type(block.attn) is keras.layers.MultiHeadAttention
        assert block.attn.num_heads == HEADS
        assert block.attn._key_dim == DIM_HEAD
        assert block._attn_via_factory is False
        # FFN: MLPBlock (Dense -> gelu -> Dropout -> Dense).
        assert type(block.mlp) is MLPBlock
        # AdaLN activation: plain keras Activation('silu').
        assert type(block.adaLN_act) is keras.layers.Activation

    @pytest.mark.parametrize("group,kwargs,attr,forbidden_type", [
        ("normalization",
         {"normalization_type": "rms_norm",
          "normalization_args": {"use_scale": False}},
         "norm1", keras.layers.LayerNormalization),
        ("attention",
         {"attention_type": "multi_head",
          "attention_args": dict(FACTORY_ATTENTION_ARGS)},
         "attn", keras.layers.MultiHeadAttention),
        ("ffn",
         {"ffn_type": "swiglu", "ffn_args": {"output_dim": DIM}},
         "mlp", MLPBlock),
        ("adaln_activation",
         {"adaln_activation_type": "mish"},
         "adaLN_act", keras.layers.Activation),
    ])
    def test_factory_path_replaces_exactly_the_requested_group(
        self, sample, group, kwargs, attr, forbidden_type
    ):
        """Each of the 4 dispatch pairs must swap ITS group and nothing else."""
        x, c = sample
        default = _make_block()
        block = _make_block(**kwargs)
        assert type(getattr(block, attr)) is not forbidden_type, (
            f"the {group} factory branch did not change which class is built"
        )
        # The three groups NOT selected must still be the default classes.
        for other in ("norm1", "attn", "mlp", "adaLN_act"):
            if other != attr:
                assert type(getattr(block, other)) is type(
                    getattr(default, other)
                ), f"selecting {group} also changed {other}"
        assert block._attn_via_factory is (attr == "attn")
        out = ops.convert_to_numpy(block([x, c]))
        assert out.shape == (BATCH, N, DIM)
        assert np.all(np.isfinite(out))

    def test_get_config_surface_and_round_trip(self):
        block = AdaLNZeroConditionalBlock(
            dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM,
            dropout_rate=0.1, use_causal_mask=False, eps=1e-5,
            normalization_type="rms_norm", normalization_args={"use_scale": False},
            attention_type="multi_head",
            attention_args=dict(FACTORY_ATTENTION_ARGS),
            ffn_type="swiglu", ffn_args={"output_dim": DIM},
            adaln_activation_type="mish", adaln_activation_args={},
        )
        cfg = block.get_config()
        assert OWN_CONFIG_KEYS <= set(cfg), (
            f"missing from get_config(): {sorted(OWN_CONFIG_KEYS - set(cfg))}"
        )
        assert set(cfg) - OWN_CONFIG_KEYS == {"name", "trainable", "dtype"}, (
            "unexpected extra get_config() keys beyond Layer's own three"
        )
        assert cfg["dim"] == DIM and cfg["dropout_rate"] == 0.1
        assert cfg["use_causal_mask"] is False and cfg["eps"] == 1e-5
        assert cfg["normalization_type"] == "rms_norm"
        assert cfg["attention_type"] == "multi_head"
        assert cfg["ffn_type"] == "swiglu"
        assert cfg["adaln_activation_type"] == "mish"
        rebuilt = AdaLNZeroConditionalBlock.from_config(cfg)
        assert rebuilt.get_config() == cfg

    def test_keras_serialization_round_trip(self, sample):
        """Full ``.keras`` save/load through a 2-input Functional model.

        The block's weights are re-seeded NON-ZERO first: at default zero-init
        the block is the identity map, so a round trip that dropped every weight
        would still compare equal.
        """
        x, c = sample
        x_in = keras.Input(shape=(N, DIM), name="x")
        c_in = keras.Input(shape=(N, DIM), name="c")
        block = AdaLNZeroConditionalBlock(
            dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM
        )
        model = keras.Model([x_in, c_in], block([x_in, c_in]))
        _seed_all_weights(block, seed=5)
        inputs = {"x": x, "c": c}
        before = model.predict(inputs, verbose=0)
        assert np.max(np.abs(before - x)) > 1e-2, (
            "the round-trip probe is vacuous: the model is still the identity"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adaln_zero_block.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            after = reloaded.predict(inputs, verbose=0)
        np.testing.assert_allclose(before, after, atol=1e-6)


# =====================================================================
# Forward contract
# =====================================================================


class TestForwardContract:

    @pytest.mark.parametrize("batch,seq", [(1, 1), (BATCH, N), (3, 17)])
    def test_forward_shape_and_finiteness(self, batch, seq):
        rng = np.random.RandomState(3)
        x = rng.normal(size=(batch, seq, DIM)).astype("float32")
        c = rng.normal(size=(batch, seq, DIM)).astype("float32")
        block = AdaLNZeroConditionalBlock(
            dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM
        )
        block.build([(batch, seq, DIM), (batch, seq, DIM)])
        _seed_all_weights(block, seed=batch * 100 + seq)
        out = ops.convert_to_numpy(block([x, c]))
        assert out.shape == (batch, seq, DIM)
        assert np.all(np.isfinite(out))

    def test_identity_at_init_is_exact_but_vacuous_for_wiring(self, sample):
        """Zero-init ``adaLN_linear`` => the block is EXACTLY ``x``.

        This is the documented AdaLN-Zero contract and is asserted here, but it
        is deliberately labelled vacuous: ``gate == 0`` makes ``gate * anything``
        zero, so a swapped gate, a mis-ordered ``ops.split`` or a discarded
        ``mod`` all still pass. The wiring is pinned by TestModulationIsLive /
        TestSplitOrder / TestFactoryAttentionDispatch instead.
        """
        x, c = sample
        block = _make_block()
        out = ops.convert_to_numpy(block([x, c]))
        np.testing.assert_array_equal(out, x)
        mod = ops.convert_to_numpy(
            block.adaLN_linear(block.adaLN_act(ops.convert_to_tensor(c)))
        )
        assert mod.shape == (BATCH, N, 6 * DIM)
        np.testing.assert_array_equal(mod, np.zeros_like(mod))

    @pytest.mark.parametrize("convention", ["pair", "single"])
    def test_both_build_input_shape_conventions(self, sample, convention):
        """``build``/``compute_output_shape`` accept ``[x_shape, c_shape]`` AND one shape."""
        x, c = sample
        shape = (BATCH, N, DIM)
        arg = [shape, shape] if convention == "pair" else shape
        block = AdaLNZeroConditionalBlock(
            dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM
        )
        assert block.compute_output_shape(arg) == shape  # before build
        block.build(arg)
        assert block.built
        assert block.adaLN_linear.built and block.norm1.built and block.mlp.built
        assert block.compute_output_shape(arg) == shape
        assert tuple(block([x, c]).shape) == shape


# =====================================================================
# Dead-component family 1 — the modulation path is LIVE
# =====================================================================


class TestModulationIsLive:

    def test_seeded_nonzero_modulation_moves_the_output(self, sample):
        """Non-zero ``adaLN_linear`` weights must reach the output.

        A block that computed ``mod`` and then never consumed it would pass
        every identity-at-init assertion. Here the same block, with only
        ``adaLN_linear`` re-seeded, must differ from BOTH ``x`` (the init-time
        output) and a no-affine-LayerNorm reference, and must differ per chunk
        group rather than by a single global scalar.
        """
        x, c = sample
        block = _make_block(seed=17)
        live = ops.convert_to_numpy(block([x, c]))
        # (i) live vs the identity the same block produces at zero-init.
        assert np.max(np.abs(live - x)) > 1e-2, (
            "seeded non-zero modulation left the block an identity map — the "
            "modulation path is computed and discarded"
        )
        # (ii) live vs a plain no-affine LayerNorm of x (the un-modulated shape).
        assert np.max(np.abs(live - _no_affine_layernorm(x))) > 1e-2
        # (iii) CONTROL: zeroing ONLY adaLN_linear restores the exact identity,
        # proving (i) is attributable to adaLN_linear and not to the other
        # seeded weights.
        _set_adaln_bias(block, {})
        np.testing.assert_array_equal(
            ops.convert_to_numpy(block([x, c])), x
        )


# =====================================================================
# Dead-component family 2 — the 6-way split ORDER
# =====================================================================


class TestSplitOrder:

    @pytest.mark.parametrize("chunk", CHUNK_ORDER)
    @pytest.mark.parametrize("dead", ["mlp", "attn"])
    def test_only_the_owning_half_responds_to_each_chunk(self, sample, chunk, dead):
        """Each modulation chunk must move ONLY its own sub-block.

        One sub-block is neutered (all weights zeroed => it outputs exactly 0),
        both gates are opened, and then one chunk at a time is raised off zero.
        A chunk belonging to the surviving half must move the output; a chunk
        belonging to the dead half must move it by EXACTLY ``0.0``. Any
        permutation of ``ops.split``'s unpack order across the msa/mlp halves —
        including the classic ``shift_msa``/``shift_mlp`` swap — breaks this.
        """
        x, c = sample
        block = _make_block(seed=23)
        _kill(block.mlp if dead == "mlp" else block.attn)
        surviving = "mlp" if dead == "attn" else "msa"

        _set_adaln_bias(block, {"gate_msa": 1.0, "gate_mlp": 1.0})
        base = ops.convert_to_numpy(block([x, c]))
        _set_adaln_bias(block, {"gate_msa": 1.0, "gate_mlp": 1.0, chunk: 0.6})
        moved = ops.convert_to_numpy(block([x, c]))
        delta = float(np.max(np.abs(moved - base)))

        if chunk.endswith(surviving):
            assert delta > 1e-3, (
                f"{chunk} belongs to the '{surviving}' half, which is alive, "
                f"yet moved the output by only {delta:.6e}"
            )
        else:
            np.testing.assert_array_equal(
                moved, base,
                err_msg=(
                    f"{chunk} belongs to the '{dead}' half, which is dead, yet "
                    f"moved the output by {delta:.6e} — ops.split's unpack "
                    f"order routes it to the wrong sub-block"
                ),
            )

    @pytest.mark.parametrize("half", ["msa", "mlp"])
    def test_modulated_subblock_input_matches_the_oracle(self, sample, half):
        """``_modulate(norm(x), shift, scale) == norm(x)*(1+scale) + shift``.

        Pins the split order WITHIN a half as well as across halves: the tensor
        actually handed to the sub-block is captured by a spy and compared to a
        numpy oracle built from the exact ``shift``/``scale`` written into
        ``adaLN_linear``'s bias. Distinct values (0.3 vs 0.7) are used so a
        shift/scale swap cannot pass.
        """
        x, c = sample
        shift, scale = (0.3, 0.7) if half == "msa" else (-0.4, 0.25)
        block = _make_block(seed=31, spy_on="attn" if half == "msa" else "mlp")
        # Only this half's shift/scale are non-zero => both gates are 0, so the
        # mlp half sees norm2(x + 0*attn) == norm2(x).
        _set_adaln_bias(block, {f"shift_{half}": shift, f"scale_{half}": scale})
        block([x, c])
        spy = block.attn if half == "msa" else block.mlp
        expected = _no_affine_layernorm(x) * (1.0 + scale) + shift
        np.testing.assert_allclose(spy.seen["h"], expected, atol=1e-5)
        # Non-vacuity: the oracle must not coincide with the un-modulated norm.
        assert np.max(np.abs(expected - _no_affine_layernorm(x))) > 1e-2


# =====================================================================
# Dead-component family 3 — the factory attention branch is real code
# =====================================================================


class TestFactoryAttentionDispatch:

    def test_zeroed_factory_attention_removes_only_the_attention_residual(
        self, sample
    ):
        """``_attn_via_factory`` must select code that actually runs.

        Under an explicit ``attention_type`` the block calls
        ``self.attn(h, training=...)``. Substituting a zeros-returning attention
        must collapse ``x + gate_msa * attn(...)`` to exactly ``x`` while leaving
        the FFN residual untouched; an intact factory attention must move it.
        """
        x, c = sample
        kwargs = {"attention_type": "multi_head",
                  "attention_args": dict(FACTORY_ATTENTION_ARGS)}
        intact = _make_block(seed=37, **kwargs)
        zeroed = _make_block(seed=37, spy_on="attn", **kwargs)
        assert intact._attn_via_factory and zeroed._attn_via_factory

        # Attention residual only (gate_mlp == 0).
        _set_adaln_bias(intact, {"gate_msa": 1.0})
        _set_adaln_bias(zeroed, {"gate_msa": 1.0})
        live = ops.convert_to_numpy(intact([x, c]))
        dead = ops.convert_to_numpy(zeroed([x, c]))
        assert np.max(np.abs(live - x)) > 1e-2, (
            "the intact factory attention contributes nothing — the probe "
            "cannot tell a dispatched layer from a dead branch"
        )
        np.testing.assert_array_equal(dead, x)

        # The FFN residual is untouched by the attention substitution.
        _set_adaln_bias(zeroed, {"gate_mlp": 1.0})
        ffn_only = ops.convert_to_numpy(zeroed([x, c]))
        assert np.max(np.abs(ffn_only - x)) > 1e-2


# =====================================================================
# Dead-component family 4 — the documented use_causal_mask contract
# =====================================================================


class TestCausalMaskContract:

    @pytest.mark.parametrize("path", ["default", "factory"])
    def test_use_causal_mask_is_live_by_default_and_ignored_under_the_factory(
        self, sample, path
    ):
        """MEASURE the docstring's "NOT forwarded to a factory-built attention".

        Two blocks differing ONLY in ``use_causal_mask``, with every other weight
        synchronised, must differ on the default ``keras.layers.MultiHeadAttention``
        path (the flag is genuinely live) and be BIT-IDENTICAL under a factory
        ``attention_type`` (the flag is genuinely dropped). The factory case also
        checks, via a spy, that the kwarg never arrives at the attention layer at
        all — so the equality cannot be explained by an attention type that
        merely ignores it.
        """
        x, c = sample
        kwargs = {} if path == "default" else {
            "attention_type": "multi_head",
            "attention_args": dict(FACTORY_ATTENTION_ARGS),
        }
        causal = _make_block(seed=41, use_causal_mask=True, **kwargs)
        acausal = _make_block(seed=41, use_causal_mask=False, **kwargs)
        # Same seed => same weights; assert it rather than trusting it.
        for a, b in zip(causal.weights, acausal.weights):
            np.testing.assert_array_equal(
                ops.convert_to_numpy(a), ops.convert_to_numpy(b)
            )
        _set_adaln_bias(causal, {"gate_msa": 1.0, "shift_msa": 0.2})
        _set_adaln_bias(acausal, {"gate_msa": 1.0, "shift_msa": 0.2})
        out_causal = ops.convert_to_numpy(causal([x, c]))
        out_acausal = ops.convert_to_numpy(acausal([x, c]))
        delta = float(np.max(np.abs(out_causal - out_acausal)))

        if path == "default":
            assert delta > 1e-3, (
                "use_causal_mask made NO difference on the default "
                "MultiHeadAttention path — the flag is dead everywhere, so the "
                "factory-path assertion below would be vacuous "
                f"(delta = {delta:.6e})"
            )
        else:
            np.testing.assert_array_equal(
                out_causal, out_acausal,
                err_msg=(
                    "use_causal_mask CHANGED the output under a factory "
                    "attention_type, contradicting the documented contract "
                    f"(delta = {delta:.6e}). If this fires, the SOURCE and its "
                    "docstring disagree — that is a source defect, not a test "
                    "failure."
                ),
            )
            spy_block = _make_block(seed=41, spy_on="attn",
                                    use_causal_mask=True, **kwargs)
            spy_block([x, c])
            assert spy_block.attn.seen["use_causal_mask"] is None, (
                "the factory attention branch DID receive use_causal_mask"
            )

    def test_causal_mask_does_not_leak_the_future(self):
        """Perturbing ``x`` at the LAST position must not move ANY earlier output.

        PORTED from the deleted ``tests/test_layers/test_adaln_zero.py`` (step 9
        of plan-2026-08-10-3649c19e), which was the only place in the repo that
        probed the mask's DIRECTION. The sibling test above proves only that the
        ``use_causal_mask`` flag is LIVE — that masking does *something*. A mask
        applied on the wrong triangle, or a per-tile rescue applied off the full
        softmax axis, changes the output while leaving every "flag is live"
        assertion green (see plans/LESSONS.md, "repair granularity"). This test
        pins the direction: only the future may be hidden.

        Non-vacuity is asserted two ways, so the equality below cannot be
        explained by an absent cross-token path: (1) the LAST position must
        itself move under the perturbation, and (2) the same probe run with
        ``use_causal_mask=False`` MUST leak.
        """
        rng = np.random.RandomState(11)
        x = rng.normal(size=(BATCH, N, DIM)).astype("float32")
        c = rng.normal(size=(BATCH, N, DIM)).astype("float32")
        x_perturbed = x.copy()
        x_perturbed[:, -1] += rng.normal(size=(BATCH, DIM)).astype("float32") * 3.0

        deltas = {}
        for causal in (True, False):
            # seed=13 re-seeds adaLN_linear NON-ZERO: at its zero-init default the
            # block is the identity in x and every position is trivially causal.
            block = _make_block(seed=13, use_causal_mask=causal)
            ref = ops.convert_to_numpy(block([x, c]))
            out = ops.convert_to_numpy(block([x_perturbed, c]))
            deltas[causal] = {
                "past": float(np.max(np.abs(ref[:, :-1] - out[:, :-1]))),
                "last": float(np.max(np.abs(ref[:, -1] - out[:, -1]))),
            }

        assert deltas[True]["last"] > 1e-3, (
            "the probe is vacuous: perturbing the last token did not even move "
            f"the last output (delta = {deltas[True]['last']:.6e})"
        )
        assert deltas[False]["past"] > 1e-3, (
            "the probe is vacuous: with use_causal_mask=False the earlier "
            "positions did NOT move either, so there is no cross-token path for "
            f"a mask to hide (delta = {deltas[False]['past']:.6e})"
        )
        assert deltas[True]["past"] < 1e-6, (
            "CAUSALITY VIOLATION: perturbing x at the last position moved the "
            f"outputs at positions 0..{N - 2} by {deltas[True]['past']:.6e} "
            f"(the acausal control moves them by {deltas[False]['past']:.6e}). "
            "The attention mask is not hiding the future."
        )
