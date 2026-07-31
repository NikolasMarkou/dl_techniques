"""Test suite for FreeTransformerLayer (encoder cross-attention) + BinaryMapper."""
import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.transformers.free_transformer import (
    FreeTransformerLayer, BinaryMapper,
)


class TestBinaryMapper:
    def test_forward_shape(self):
        mapper = BinaryMapper(num_bits=4)
        logits = keras.random.normal([2, 8, 4])
        out = mapper(logits, training=False)
        # one-hot over 2^num_bits categories
        assert tuple(out.shape) == (2, 8, 16)

    def test_invalid_num_bits(self):
        with pytest.raises(ValueError):
            BinaryMapper(num_bits=0)

    def test_get_config_round_trip(self):
        mapper = BinaryMapper(num_bits=6)
        rebuilt = BinaryMapper.from_config(mapper.get_config())
        assert rebuilt.num_bits == 6


class TestFreeTransformerLayer:
    HID = 32
    HEADS = 4
    INTER = 64
    T = 12

    def _layer(self, **kw):
        params = dict(hidden_size=self.HID, num_heads=self.HEADS,
                      intermediate_size=self.INTER)
        params.update(kw)
        return FreeTransformerLayer(**params)

    @pytest.fixture
    def x(self):
        return keras.random.normal([2, self.T, self.HID])

    def test_standard_path_forward(self, x):
        """use_free_transformer=False behaves like a standard transformer block."""
        out = self._layer(use_free_transformer=False)(x, training=False)
        assert tuple(out.shape) == (2, self.T, self.HID)

    def test_free_path_training_returns_tuple(self, x):
        """Training path returns (output, bit_logits)."""
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)
        result = layer(x, training=True)
        assert isinstance(result, (tuple, list)) and len(result) == 2
        output, bit_logits = result
        assert tuple(output.shape) == (2, self.T, self.HID)
        assert tuple(bit_logits.shape) == (2, self.T, 4)
        assert np.all(np.isfinite(np.array(output)))

    def test_free_path_inference(self, x):
        """Inference returns the SAME structure as training: (output, bit_logits).
        bit_logits is the uniform-prior zeros tensor."""
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)
        result = layer(x, training=False)
        assert isinstance(result, (tuple, list)) and len(result) == 2
        output, bit_logits = result
        assert tuple(output.shape) == (2, self.T, self.HID)
        assert tuple(bit_logits.shape) == (2, self.T, 4)
        assert np.allclose(np.array(bit_logits), 0.0)  # uniform prior

    def test_output_structure_matches_compute_output_shape(self, x):
        """call() output structure must match compute_output_shape in BOTH modes
        (the structure depends only on use_free_transformer, not on training)."""
        # use_free_transformer=True -> 2-tuple in both modes
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)
        cos = layer.compute_output_shape((2, self.T, self.HID))
        assert isinstance(cos, tuple) and len(cos) == 2 and isinstance(cos[0], tuple)
        for mode in (True, False):
            result = layer(x, training=mode)
            assert isinstance(result, (tuple, list)) and len(result) == 2
            assert tuple(result[0].shape) == tuple(cos[0])
            assert tuple(result[1].shape) == tuple(cos[1])
        # use_free_transformer=False -> single tensor matching single shape
        plain = self._layer(use_free_transformer=False)
        cos_plain = plain.compute_output_shape((2, self.T, self.HID))
        out = plain(x, training=False)
        assert not isinstance(out, (tuple, list))
        assert tuple(out.shape) == tuple(cos_plain)

    def test_encoder_is_cross_attention(self, x):
        """The redesign: the encoder must use cross-attention, so its output
        depends on the K/V (the sequence S), not only the query (zeta)."""
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)
        layer(x, training=True)  # build
        assert layer.encoder_attention_type == 'multi_head_cross'
        q = keras.random.normal([2, self.T, self.HID])
        kv1 = keras.random.normal([2, self.T, self.HID])
        kv2 = keras.random.normal([2, self.T, self.HID])
        o1 = np.array(layer.encoder_attention(q, kv_input=kv1, training=False))
        o2 = np.array(layer.encoder_attention(q, kv_input=kv2, training=False))
        assert not np.allclose(o1, o2, atol=1e-5)

    def test_graph_trace_training(self, x):
        layer = self._layer(use_free_transformer=True, num_latent_bits=4)

        @tf.function
        def traced(inp):
            output, _ = layer(inp, training=True)
            return output

        out = traced(tf.constant(np.array(x)))
        assert tuple(out.shape) == (2, self.T, self.HID)

    def test_get_config_round_trip(self):
        layer = self._layer(use_free_transformer=True, num_latent_bits=8)
        rebuilt = FreeTransformerLayer.from_config(layer.get_config())
        assert rebuilt.use_free_transformer is True
        assert rebuilt.num_latent_bits == 8
        assert rebuilt.encoder_attention_type == 'multi_head_cross'

    def test_model_save_load_round_trip(self, x):
        inp = keras.Input(shape=(self.T, self.HID))
        out = FreeTransformerLayer(hidden_size=self.HID, num_heads=self.HEADS,
                                   intermediate_size=self.INTER,
                                   use_free_transformer=False)(inp)
        model = keras.Model(inp, out)
        ref = model(x, training=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "free.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        out2 = loaded(x, training=False)
        np.testing.assert_allclose(np.array(ref), np.array(out2), atol=1e-5)


class TestEncoderHonoursPadding:
    """F-13: the encoder cross-attention must honour the caller's padding.

    At HEAD the encoder passed ``attention_mask=None`` unconditionally, so the
    posterior ``Q(Z|S)`` pooled over PAD keys/values. Because ``z_projected`` is
    added to ``attention_output`` *before* the FFN, PAD content reached the
    layer output at every REAL position too -- silently, with finite values.

    Two properties are asserted, and they pull in opposite directions:

    1. **Padding is excluded.** PAD content must not move ``bit_logits`` or the
       output at real positions -- at ``rtol=0, atol=0``.
    2. **Causality is NOT inherited.** The encoder is deliberately non-causal.
       A rank-3 causal mask must reduce to all-ones for the encoder, so an early
       query still sees a late key. Forwarding the caller's rank-3 mask verbatim
       (the naive fix) satisfies (1) and breaks (2).

    Fixture design follows D-008: ``use_bias=True`` plus every trainable weight
    assigned from a seeded RNG, i.e. the state a *trained* model is in. With the
    default zero biases several propagation paths are unobservable, so a probe
    on fresh weights can come back green while the defect is live.
    """

    B, T, R, HID, HEADS, INTER, BITS = 2, 10, 6, 32, 4, 64, 4

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _fresh_layer(self):
        """A built FreeTransformerLayer with every trainable weight non-trivial."""
        keras.utils.set_random_seed(7)
        layer = FreeTransformerLayer(
            hidden_size=self.HID, num_heads=self.HEADS,
            intermediate_size=self.INTER, use_free_transformer=True,
            num_latent_bits=self.BITS, dropout_rate=0.0,
            attention_dropout_rate=0.0, use_bias=True,
        )
        probe = keras.random.normal([self.B, self.T, self.HID])
        layer(probe, attention_mask=self._pad_mask(), training=True)  # build
        rng = np.random.default_rng(99)
        nonzero_bias = False
        for w in layer.weights:
            if not w.trainable:
                continue
            w.assign(rng.standard_normal(w.shape).astype("float32") * 0.15)
            if "bias" in w.path and float(np.max(np.abs(np.array(w)))) > 0.0:
                nonzero_bias = True
        assert nonzero_bias, (
            "no non-zero bias was assigned -- with zero biases a zeroed/masked "
            "position stays exactly zero through the block and the masking "
            "sites become unobservable (D-008)"
        )
        return layer

    def _pad_mask(self):
        """Rank-2 keep predicate: real at [0, R), PAD at [R, T)."""
        m = np.zeros((self.B, self.T), "float32")
        m[:, :self.R] = 1.0
        return keras.ops.convert_to_tensor(m)

    def _causal_mask(self, with_padding: bool):
        """Rank-3 ``(B, T, T)`` causal keep predicate, optionally AND-ed with padding."""
        causal = np.tril(np.ones((self.T, self.T), "float32"))
        m = np.broadcast_to(causal, (self.B, self.T, self.T)).copy()
        if with_padding:
            m = m * np.array(self._pad_mask())[:, None, :]
        return keras.ops.convert_to_tensor(m)

    def _inputs(self):
        rng = np.random.default_rng(1234)
        x = rng.standard_normal((self.B, self.T, self.HID)).astype("float32")
        # Non-uniform per-channel perturbation: a uniform shift would be
        # mean-centred away by the channel-axis LayerNorm (iteration-1 step-1
        # lesson) and the probe would measure reassociation noise.
        pert = rng.standard_normal((self.B, self.T, self.HID)).astype("float32") * 3.0
        x_pad_perturbed = x.copy()
        x_pad_perturbed[:, self.R:, :] += pert[:, self.R:, :]
        x_real_perturbed = x.copy()
        x_real_perturbed[:, :self.R, :] += pert[:, :self.R, :]
        return x, x_pad_perturbed, x_real_perturbed

    @staticmethod
    def _run(layer, x, mask):
        out, bit_logits = layer(
            keras.ops.convert_to_tensor(x), attention_mask=mask, training=True
        )
        return np.array(out), np.array(bit_logits)

    # ------------------------------------------------------------------
    # 1. padding is excluded
    # ------------------------------------------------------------------
    def test_bit_logits_at_real_positions_ignore_pad_content(self):
        """The posterior must not be conditioned on PAD content.

        RED at the pre-fix commit ``9ee28342``'s predecessor ``e38a9c3c``:
        perturbing PAD inputs moved ``bit_logits`` at REAL positions on 48/48
        elements, max ``3.606e-02``, where exactly ``0.0`` is required. Green
        since ``03980608`` (F-13).
        """
        layer = self._fresh_layer()
        x, x_pad, x_real = self._inputs()
        mask = self._pad_mask()

        _, base = self._run(layer, x, mask)
        _, moved = self._run(layer, x_pad, mask)
        _, control = self._run(layer, x_real, mask)

        # Live control FIRST: if perturbing REAL tokens does not move the
        # posterior either, the probe is vacuous and the bit-identity below
        # would pass for the wrong reason.
        control_delta = float(np.max(np.abs(base[:, :self.R] - control[:, :self.R])))
        assert control_delta > 1e-3, (
            f"vacuous probe: perturbing REAL tokens moved bit_logits by only "
            f"{control_delta:.3e}"
        )

        np.testing.assert_allclose(
            base[:, :self.R], moved[:, :self.R], rtol=0, atol=0,
            err_msg="PAD content reached bit_logits at real positions",
        )

    def test_output_at_real_positions_ignores_pad_content(self, monkeypatch):
        """PAD content must not reach the layer OUTPUT at real positions.

        Two instrument choices, both forced by measurement:

        * ``BinaryMapper`` draws ``keras.random.uniform`` per call and
          ``keras.utils.set_random_seed`` does NOT rewind that global draw
          (measured: two identical forwards differ by 0.397 on the output while
          ``bit_logits`` differ by exactly 0.0). The sampler is replaced with a
          constant 0.5 threshold for this test only, so the sampled bits stay a
          genuine deterministic function of ``bit_logits``
          (``sampled = 0.5 < sigmoid(bit_logits)``): the leak still propagates
          through sampling, only the unrelated RNG is removed.
        * The output path is **discrete**: ``z`` is a sampled one-hot, so a
          moved ``bit_logits`` only reaches the output when it flips a bit.
          Measured on pre-fix code at PAD-perturbation scales (3, 6, 10, 20):
          output deltas ``6.3e-08 / 0.0 / 0.0 / 0.265`` for ``bit_logits``
          deltas ``3.6e-02 / 4.1e-02 / 4.3e-02 / 4.4e-02``. A single-scale probe
          would therefore have been GREEN at scales 6 and 10 with the defect
          fully live. The whole sweep is asserted instead.
        """
        monkeypatch.setattr(
            keras.random, "uniform",
            lambda shape, **kw: keras.ops.full(shape, 0.5, dtype=kw.get("dtype", "float32")),
        )
        layer = self._fresh_layer()
        x, _, x_real = self._inputs()
        mask = self._pad_mask()
        rng = np.random.default_rng(1234)
        rng.standard_normal((self.B, self.T, self.HID))  # keep _inputs()' stream aligned
        pert = rng.standard_normal((self.B, self.T, self.HID)).astype("float32")

        base, _ = self._run(layer, x, mask)
        repeat, _ = self._run(layer, x, mask)
        assert float(np.max(np.abs(base - repeat))) == 0.0, (
            "the deterministic-sampler patch did not take effect; a bit-identity "
            "assertion on the output would be meaningless"
        )
        control, _ = self._run(layer, x_real, mask)
        control_delta = float(np.max(np.abs(base[:, :self.R] - control[:, :self.R])))
        assert control_delta > 1e-3, (
            f"vacuous probe: perturbing REAL tokens moved the output by only "
            f"{control_delta:.3e}"
        )

        deltas = {}
        for scale in (3.0, 6.0, 10.0, 20.0):
            x_pad = x.copy()
            x_pad[:, self.R:, :] += pert[:, self.R:, :] * scale
            moved, _ = self._run(layer, x_pad, mask)
            deltas[scale] = float(
                np.max(np.abs(base[:, :self.R] - moved[:, :self.R]))
            )
        assert all(d == 0.0 for d in deltas.values()), (
            f"PAD content reached the layer output at real positions: "
            f"per-scale max|delta| = {deltas}"
        )

    def test_rank3_causal_plus_padding_mask_also_excludes_pad(self):
        """A rank-3 causal+padding mask must reduce to the padding mask."""
        layer = self._fresh_layer()
        x, x_pad, x_real = self._inputs()
        mask = self._causal_mask(with_padding=True)

        _, base = self._run(layer, x, mask)
        _, moved = self._run(layer, x_pad, mask)
        _, control = self._run(layer, x_real, mask)

        control_delta = float(np.max(np.abs(base[:, :self.R] - control[:, :self.R])))
        assert control_delta > 1e-3, f"vacuous probe: {control_delta:.3e}"
        np.testing.assert_allclose(
            base[:, :self.R], moved[:, :self.R], rtol=0, atol=0,
            err_msg="PAD content reached bit_logits under a rank-3 causal+padding mask",
        )

    # ------------------------------------------------------------------
    # 2. causality is NOT inherited  (D-005)
    # ------------------------------------------------------------------
    def test_pure_causal_rank3_mask_does_not_make_the_encoder_causal(self):
        """``ops.max(mask, axis=-2)`` maps a pure causal mask to all-ones.

        The encoder is deliberately non-causal -- the posterior ``Q(Z|S)`` is
        meant to see the WHOLE sequence. This is the guard against the naive
        fix (forwarding the caller's rank-3 mask verbatim), which would silently
        impose causality on the encoder while still passing every padding test.
        """
        layer = self._fresh_layer()
        x, _, _ = self._inputs()
        mask = self._causal_mask(with_padding=False)

        _, base = self._run(layer, x, mask)
        x_last = x.copy()
        rng = np.random.default_rng(5150)
        x_last[:, -1, :] += rng.standard_normal((self.B, self.HID)).astype("float32") * 3.0
        _, moved = self._run(layer, x_last, mask)

        first_pos_delta = float(np.max(np.abs(base[:, 0] - moved[:, 0])))
        assert first_pos_delta > 1e-4, (
            f"the encoder inherited causality: perturbing the LAST token left "
            f"bit_logits at position 0 unchanged (delta {first_pos_delta:.3e}). "
            f"A rank-3 mask must reduce to key-validity, not be forwarded verbatim."
        )

    def test_no_mask_path_is_unchanged(self):
        """``attention_mask=None`` must still mean full attention everywhere."""
        layer = self._fresh_layer()
        x, _, _ = self._inputs()
        _, base = self._run(layer, x, None)
        x_last = x.copy()
        x_last[:, -1, :] += 4.0
        _, moved = self._run(layer, x_last, None)
        assert float(np.max(np.abs(base[:, 0] - moved[:, 0]))) > 1e-4

    def test_rank4_mask_reduces_over_every_query_side_axis(self):
        """A rank-4 ``(B, 1, T, T)`` mask is what a head-broadcast caller passes.

        ``MultiHeadCrossAttention`` forwards rank-4 masks verbatim, so the same
        union reduction must cover the head axis as well as the query axis.
        Both halves of the contract are asserted: PAD excluded, causality not
        inherited.
        """
        layer = self._fresh_layer()
        x, x_pad, x_real = self._inputs()
        m3 = np.array(self._causal_mask(with_padding=True))
        mask = keras.ops.convert_to_tensor(m3[:, None, :, :])

        _, base = self._run(layer, x, mask)
        _, moved = self._run(layer, x_pad, mask)
        _, control = self._run(layer, x_real, mask)
        assert float(np.max(np.abs(base[:, :self.R] - control[:, :self.R]))) > 1e-3
        np.testing.assert_allclose(
            base[:, :self.R], moved[:, :self.R], rtol=0, atol=0,
            err_msg="PAD content reached bit_logits under a rank-4 mask",
        )

    def test_unsupported_mask_rank_is_rejected_loudly(self):
        """A rank the derivation cannot interpret must raise, not mis-reduce."""
        layer = self._fresh_layer()
        x, _, _ = self._inputs()
        bad = keras.ops.convert_to_tensor(
            np.ones((self.B, 1, 1, self.T, self.T), "float32")
        )
        with pytest.raises(ValueError, match="rank"):
            self._run(layer, x, bad)

    # ------------------------------------------------------------------
    # 3. serialization (I3) on a config that exercises the new path
    # ------------------------------------------------------------------
    def test_masked_free_path_survives_save_load_by_value(self):
        """Round-trip a functional model over the masked free-transformer path.

        ``bit_logits`` is the deterministic half of the output (the layer output
        itself passes through the stochastic ``BinaryMapper``), so it is what the
        value comparison uses.
        """
        keras.utils.set_random_seed(11)
        x_in = keras.Input(shape=(self.T, self.HID))
        m_in = keras.Input(shape=(self.T,))
        out, bit_logits = FreeTransformerLayer(
            hidden_size=self.HID, num_heads=self.HEADS,
            intermediate_size=self.INTER, use_free_transformer=True,
            num_latent_bits=self.BITS, dropout_rate=0.0,
            attention_dropout_rate=0.0, use_bias=True,
        )(x_in, attention_mask=m_in)
        model = keras.Model([x_in, m_in], [out, bit_logits])

        x, _, _ = self._inputs()
        mask = self._pad_mask()
        ref = np.array(model([keras.ops.convert_to_tensor(x), mask], training=True)[1])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "free_masked.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        got = np.array(loaded([keras.ops.convert_to_tensor(x), mask], training=True)[1])
        assert float(np.max(np.abs(ref))) > 1e-3, "round-trip compared all-zero values"
        np.testing.assert_allclose(ref, got, rtol=0, atol=0)


class TestEncoderFFNKwargFiltering:
    """The encoder FFN bundle must not hand `create_ffn_layer` keys it rejects.

    `encoder_ffn_type` defaults to 'swiglu', which does not accept `activation`,
    yet the bundle passed `activation=self.activation` unconditionally. The
    factory filtered it out silently -- no warning at any log level -- so the
    layer's own default config was quietly discarding a parameter. This was one
    of only two (call site, ffn_type) pairs in the repo that armed that drop,
    measured over an instrumented 1634-test run.

    The layer now pre-filters its OWN generic defaults to what the chosen
    `encoder_ffn_type` accepts, as `gated_linear_attention_block.py` does.

    As of `plan-2026-07-30T140922-8af1028f`/D-023 the factory RAISES on a key
    it would have had to drop, so the pre-filter is no longer merely tidy: it
    is what keeps the default config CONSTRUCTIBLE. These tests assert the
    raise, not a log record -- there is no dropped-key warning left to capture.
    """

    @staticmethod
    def _build(caplog_level="WARNING", **kw):
        layer = FreeTransformerLayer(
            hidden_size=32, num_heads=4, intermediate_size=128,
            use_free_transformer=True, dropout_rate=0.0,
            attention_dropout_rate=0.0, **kw
        )
        layer(keras.random.normal([2, 8, 32]), training=False)
        return layer

    def test_default_config_drops_no_key(self):
        """swiglu default: the layer CONSTRUCTS, i.e. no key needed dropping.

        This is the defect, restated against the strict factory. Reverting the
        pre-filter makes `create_ffn_layer` raise on `activation`, so this
        assertion fails with that exact message rather than on a log record.
        The `_build` call is the assertion -- a strictness raise propagates.
        """
        layer = self._build()
        assert layer.encoder_ffn is not None, (
            "the default encoder FFN was never constructed, so this test "
            "measured nothing"
        )

    def test_misspelled_encoder_ffn_args_key_raises_naming_the_typo(self):
        """The caller's own `encoder_ffn_args` is deliberately NOT pre-filtered.

        The pre-filter covers only this layer's generic defaults; a caller key
        reaches `create_ffn_layer` verbatim so the factory can complain about
        it. As of D-023 that complaint is a `ValueError` naming the key, not a
        `logger.warning`. Softening the factory back to a warning fails here.
        """
        with pytest.raises(ValueError, match="activatoin"):
            self._build(encoder_ffn_args={"activatoin": "gelu"})

    def test_an_ffn_type_that_accepts_activation_still_receives_it(self):
        """CONTROL: the pre-filter must not strip a key the type DOES accept.

        Without this, a filter that dropped `activation` unconditionally would
        satisfy both tests above while silently changing every mlp-style encoder
        FFN. `mlp` lists `activation` as a real parameter, so it must arrive.
        """
        layer = self._build(encoder_ffn_type="mlp",
                            encoder_ffn_args={"activation": "relu"})
        # MLPBlock stores the raw value as `activation_name` and the resolved
        # callable as `activation_fn`; assert on the resolved one so a silently
        # ignored kwarg cannot pass by leaving the name set.
        resolved = layer.encoder_ffn.activation_fn
        assert resolved is not None
        activation_repr = getattr(resolved, "__name__", str(resolved))
        assert "relu" in activation_repr.lower(), (
            f"the pre-filter stripped an `activation` that mlp accepts: "
            f"{activation_repr}"
        )

    def test_unknown_encoder_ffn_type_raises_naming_alternatives(self):
        """An unknown type must fail loudly, matching the GLA site's behaviour."""
        with pytest.raises(ValueError, match="Unknown encoder_ffn_type"):
            FreeTransformerLayer(
                hidden_size=32, num_heads=4, intermediate_size=128,
                use_free_transformer=True, encoder_ffn_type="not_a_real_ffn",
            )


# ---------------------------------------------------------------------
# F-06 -- eq. 8's per-category G_{t,d}
# ---------------------------------------------------------------------


def _oracle_G(logits: np.ndarray, num_bits: int) -> np.ndarray:
    """float64 brute-force oracle for eq. 8's ``G_{t,d}``, all ``2^H`` categories.

    Deliberately NOT the shipped matmul contraction: this walks every
    ``(category, bit)`` pair and multiplies the per-bit probabilities directly,
    exactly as eq. 8 is written. ``1 - p`` is taken as ``sigmoid(-L)`` rather
    than by subtraction -- a measured requirement, not tidiness: the subtraction
    form loses catastrophically when ``p ~ 1`` (at ``|L| = 30`` its own relative
    error reaches 5.0e-05, which is 1e10 x worse than the form under test, and
    at ``|L| = 88`` it underflows to exactly 0.0). Verified against a 60-digit
    ``mpmath`` naive-product oracle: this float64 form and the shipped matmul
    form both agree with it to <= 8.2e-15 relative (37 x float64 eps).

    :param logits: bit logits ``(B, T, H)``.
    :param num_bits: ``H``.
    :return: ``G`` of shape ``(B, T, 2**H)``, float64.
    """
    logits = np.asarray(logits, dtype=np.float64)
    p1 = 1.0 / (1.0 + np.exp(-logits))          # P(bit = 1)
    p0 = 1.0 / (1.0 + np.exp(logits))           # P(bit = 0) == sigmoid(-L)
    out = np.ones(logits.shape[:-1] + (2 ** num_bits,), dtype=np.float64)
    for d in range(2 ** num_bits):
        for h in range(num_bits):
            # 0-indexed bit convention, matching the layer's `pow2 = 2**i`
            out[..., d] *= p1[..., h] if ((d >> h) & 1) else p0[..., h]
    return out


class TestBinaryMapperPerCategoryGradient:
    """F-06: eq. 8 defines ``G_{t,d}`` for EVERY one of the ``2^H`` categories.

    The layer used to compute the single SCALAR ``G`` of the sampled category
    and broadcast it across all ``2^H`` one-hot slots. The forward value is
    unaffected either way (``x + g - stop_gradient(g) == x`` for any ``g``), so
    nothing crashed and no pre-existing test could see it -- but every category
    received an IDENTICAL gradient instead of a per-category-routed one.

    These probes therefore live entirely in the BACKWARD pass. A forward-only
    assertion is structurally incapable of telling the two implementations
    apart, which is why this file carried no ``GradientTape`` at all before.
    """

    BITS = 3          # 8 categories -- small enough to brute-force in float64
    SEED = 20260731

    def _logits(self, dtype="float64", shape=(2, 3, 3)):
        """Seeded NON-ZERO logits, spanning both signs and both saturation tails.

        ``BinaryMapper`` is weightless (no kernel, no bias), so the plan's
        non-zero-bias fixture rule lands on the only live input there is: these
        logits. The in-fixture assertions below are what stop a silently
        all-zero probe -- at ``L = 0`` every ``G_d`` equals ``2^-H`` and the
        per-category rows would be indistinguishable for a REAL reason.
        """
        rng = np.random.default_rng(self.SEED)
        lg = rng.normal(0.0, 1.5, size=shape)
        lg.flat[0] = 4.0                 # deliberate near-saturation, P ~ 0.982
        lg.flat[1] = -3.5                # ... and the other tail
        assert np.min(np.abs(lg)) > 1e-3, "degenerate probe: a logit is ~0"
        assert np.max(np.abs(lg)) > 1.0, "degenerate probe: all logits are tiny"
        return np.asarray(lg, dtype=dtype)

    @staticmethod
    def _jacobian(mapper, logits_np):
        """d(z_one_hot) / d(bit_logits) at ``training=True``.

        :return: array ``(B, T, C, B, T, H)``.
        """
        x = tf.constant(logits_np)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = mapper(x, training=True)
        jac = tape.jacobian(y, x)
        assert jac is not None, "no gradient path reached bit_logits at all"
        return np.array(jac)

    def test_per_category_jacobian_rows_differ(self):
        """THE F-06 assertion: two distinct categories must route DIFFERENT
        gradients back to ``bit_logits``.

        Under the scalar broadcast every row of the Jacobian is the same
        ``d g_td / dL``, so ``jac[..., d1, ...]`` and ``jac[..., d2, ...]`` are
        bit-identical for every pair ``d1 != d2``.
        """
        mapper = BinaryMapper(num_bits=self.BITS, dtype="float64")
        logits = self._logits()
        jac = self._jacobian(mapper, logits)

        rows = [jac[0, 0, d] for d in range(2 ** self.BITS)]
        assert float(np.max(np.abs(rows[0]))) > 1e-6, (
            "the gradient itself is ~0, so 'rows differ' would be vacuous"
        )
        for d1 in range(2 ** self.BITS):
            for d2 in range(d1 + 1, 2 ** self.BITS):
                assert not np.allclose(rows[d1], rows[d2], rtol=0, atol=1e-12), (
                    f"categories {d1} and {d2} receive an IDENTICAL gradient "
                    f"(max|diff| = {np.max(np.abs(rows[d1] - rows[d2])):.6e}); "
                    f"eq. 8 routes a different G per category"
                )

    def test_jacobian_matches_the_float64_brute_force_oracle(self):
        """The per-category gradient must equal ``G_d * (U[d] - p)`` exactly.

        Differentiating ``log G_d = sum_h [U_dh log p_h + (1-U_dh) log(1-p_h)]``
        gives ``d G_d / d L_h = G_d * (U[d,h] - p_h)``. Comparing against the
        brute-force ``G`` (not the shipped contraction) is what makes this an
        oracle rather than a restatement of the implementation.
        """
        mapper = BinaryMapper(num_bits=self.BITS, dtype="float64")
        logits = self._logits()
        jac = self._jacobian(mapper, logits)

        G = _oracle_G(logits, self.BITS)                      # (B,T,C)
        p = 1.0 / (1.0 + np.exp(-logits))                     # (B,T,H)
        U = np.array([[(d >> h) & 1 for h in range(self.BITS)]
                      for d in range(2 ** self.BITS)], dtype=np.float64)
        # expected[b,t,d,h]
        expected = G[..., None] * (U[None, None, :, :] - p[:, :, None, :])

        B, T = logits.shape[0], logits.shape[1]
        got = np.stack([np.stack([np.stack([jac[b, t, d, b, t, :]
                                            for d in range(2 ** self.BITS)])
                                  for t in range(T)]) for b in range(B)])
        assert float(np.max(np.abs(expected))) > 1e-3, "oracle is all-zero"
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-13)

        # Cross-token gradients must stay exactly zero: G_{t,:} depends on
        # token t's logits only. A contraction over the wrong axis would show
        # up here and nowhere else.
        for b in range(B):
            for t in range(T):
                for t2 in range(T):
                    if t2 != t:
                        assert np.all(jac[b, t, :, b, t2, :] == 0.0)

    def test_forward_value_is_bit_identical_to_the_pass_through_free_path(
            self, monkeypatch):
        """I1: this is a BACKWARD-only correction.

        ``x + G - stop_gradient(G) == x`` exactly, for any ``G`` -- so the
        training-path output must equal the inference-path output BIT FOR BIT
        on the same draw, and must still be an exact one-hot.

        The draw has to be pinned by monkeypatching ``keras.random.uniform``:
        MEASURED here, ``keras.utils.set_random_seed`` does NOT reproduce a
        subsequent ``keras.random.uniform`` draw on this backend (three
        seed-reset pairs, three different draws), so a seed-based version of
        this test would compare two different samples and fail for a reason
        that has nothing to do with the pass-through.
        """
        mapper = BinaryMapper(num_bits=self.BITS, dtype="float64")
        logits = self._logits()

        rng = np.random.default_rng(self.SEED + 1)
        fixed = rng.uniform(size=logits.shape)
        monkeypatch.setattr(
            keras.random, "uniform",
            lambda shape, dtype=None, **kw: keras.ops.cast(fixed, dtype or "float32"),
        )

        with_pass_through = np.array(mapper(logits, training=True))
        without = np.array(mapper(logits, training=False))

        np.testing.assert_array_equal(with_pass_through, without)
        assert set(np.unique(with_pass_through).tolist()) <= {0.0, 1.0}, (
            "the pass-through leaked into the forward value"
        )
        np.testing.assert_array_equal(with_pass_through.sum(axis=-1),
                                      np.ones(logits.shape[:-1]))
        # control: the pinned draw must actually exercise several categories,
        # or "identical one-hots" would be trivially satisfiable
        assert len(np.unique(np.argmax(with_pass_through, axis=-1))) > 1

    def test_symbolic_trace_keeps_the_per_category_routing(self):
        """The new op path must survive a real symbolic trace, not just eager.

        The documented repo trap (``ops.tril``/``triu``) is an op that passes
        every eager test and then breaks ``fit``/``jit_compile``/save-load at
        once. So: trace with an UNKNOWN batch and sequence length and take the
        gradient inside the trace.
        """
        mapper = BinaryMapper(num_bits=self.BITS)
        C = 2 ** self.BITS

        @tf.function(input_signature=[
            tf.TensorSpec([None, None, self.BITS], tf.float32)
        ])
        def traced(lg):
            with tf.GradientTape() as tape:
                tape.watch(lg)
                y = mapper(lg, training=True)
                # contract each category against a distinct weight so a scalar
                # broadcast and a per-category G give different totals
                w = tf.reshape(tf.range(1.0, C + 1.0), [1, 1, C])
                s = tf.reduce_sum(y * w)
            return s, tape.gradient(s, lg)

        logits = self._logits(dtype="float32", shape=(2, 5, self.BITS))
        total, grad = traced(tf.constant(logits))
        grad = np.array(grad)
        assert np.all(np.isfinite(grad)), "symbolic trace produced non-finite grads"
        assert float(np.max(np.abs(grad))) > 1e-6, "traced gradient is ~0"

        # The weighted-sum gradient equals sum_d w_d * G_d * (U[d] - p).
        G = _oracle_G(logits, self.BITS)
        p = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64)))
        U = np.array([[(d >> h) & 1 for h in range(self.BITS)]
                      for d in range(C)], dtype=np.float64)
        w = np.arange(1.0, C + 1.0)
        expected = np.einsum('d,btd,btdh->bth', w, G,
                             U[None, None, :, :] - p[:, :, None, :])
        np.testing.assert_allclose(grad, expected, rtol=2e-4, atol=2e-6)


# =============================================================================
# F-21 + G-08: `if training is True:` -> `if training:` at BOTH sites
# =============================================================================

def _seeded_free_layer(bits=3, hidden=32, heads=4, inter=64, seq=6, batch=2):
    """A BUILT ``FreeTransformerLayer`` with seeded NON-ZERO weights.

    The default initializers put zeros in every bias; with a zero-init
    ``encoder_readout`` bias the training-path ``bit_logits`` could be all-zero
    for a reason that has nothing to do with which branch ran, which would make
    the "did the encoder run" assertion below vacuous. Dropout is disabled so
    the only stochastic op left is the sampler.
    """
    layer = FreeTransformerLayer(
        hidden_size=hidden, num_heads=heads, intermediate_size=inter,
        use_free_transformer=True, num_latent_bits=bits,
        dropout_rate=0.0, attention_dropout_rate=0.0,
    )
    x = np.asarray(np.random.default_rng(7).normal(size=(batch, seq, hidden)),
                   np.float32)
    layer(keras.ops.convert_to_tensor(x), training=False)  # build
    rng = np.random.default_rng(99)
    layer.set_weights([np.asarray(rng.normal(0.0, 0.25, size=w.shape), w.dtype)
                       for w in layer.get_weights()])
    assert all(np.any(np.abs(w) > 1e-6) for w in layer.get_weights()), (
        "degenerate fixture: some weight/bias is all-zero"
    )
    return layer, keras.ops.convert_to_tensor(x)


def _pin_uniform(monkeypatch):
    """Make ``keras.random.uniform`` reproducible across calls.

    MEASURED at step 4 and re-confirmed here: ``keras.utils.set_random_seed``
    does NOT reproduce a subsequent ``keras.random.*`` draw on this backend, so
    a seed-based version of the negative control would compare two different
    samples and fail for a reason unrelated to the branch.
    """
    counter = {"n": 0}

    def fake(shape, minval=0.0, maxval=1.0, dtype=None, seed=None, **kw):
        shp = tuple(int(s) for s in np.array(shape).reshape(-1))
        rng = np.random.default_rng(1234 + counter["n"])
        counter["n"] += 1
        return keras.ops.cast(
            rng.uniform(float(minval), float(maxval), size=shp),
            dtype or "float32")

    monkeypatch.setattr(keras.random, "uniform", fake)
    return counter


class TestTrainingFlagTruthiness:
    """F-21 / G-08: both sites used ``if training is True:``.

    ``tf.constant(True) is True`` is Python ``False``, so a tensor-valued
    ``training`` used to fall to the ``else`` branch: ``FreeTransformerLayer``
    ran the uniform-sampling INFERENCE path (no encoder sub-network, zero
    ``bit_logits``) and ``BinaryMapper`` dropped its gradient pass-through --
    silently, with no exception, while the caller believed it was training.

    ``if training:`` fixes the EAGER tensor case outright and converts the
    traced-symbolic case into a loud refusal. It cannot make a symbolic
    ``training`` work: the two branches run structurally different
    sub-networks (cross-attention + FFN + readout + sampling vs a bare
    ``keras.random.uniform``), so there is no ``ops.where`` blend that does not
    pay for both unconditionally.
    """

    BITS = 3

    def _mapper_logits(self):
        rng = np.random.default_rng(20260731)
        lg = rng.normal(0.0, 1.5, size=(2, 4, self.BITS))
        lg.flat[0], lg.flat[1] = 4.0, -3.5
        assert np.min(np.abs(lg)) > 1e-3, "degenerate probe: a logit is ~0"
        return np.asarray(lg, np.float64)

    # ---------------------------------------------------------------- EAGER
    @pytest.mark.parametrize("site", ["free_transformer_layer", "binary_mapper"])
    def test_eager_tensor_training_runs_the_training_path(self, site):
        """THE F-21/G-08 assertion, eager.

        RED before the fix: the layer returned all-zero ``bit_logits`` (the
        uniform-prior stand-in, i.e. the inference branch) and the mapper's
        pass-through never ran, so ``bit_logits`` received NO gradient at all.
        """
        if site == "free_transformer_layer":
            layer, x = _seeded_free_layer(bits=self.BITS)
            _, bl_true = layer(x, training=True)          # Python-bool control
            _, bl_tensor = layer(x, training=tf.constant(True))
            bl_true, bl_tensor = np.array(bl_true), np.array(bl_tensor)
            assert float(np.max(np.abs(bl_true))) > 1e-3, (
                "control is degenerate: even Python training=True gives ~0 "
                "bit_logits, so 'non-zero' would prove nothing"
            )
            assert not np.all(bl_tensor == 0.0), (
                "tensor-valued training=True ran the INFERENCE path: "
                f"bit_logits are the all-zero uniform prior (max|bl| = "
                f"{float(np.max(np.abs(bl_tensor))):.6e})"
            )
            # the encoder is deterministic given the weights, so the tensor
            # flag must reproduce the Python flag exactly
            np.testing.assert_array_equal(bl_tensor, bl_true)
        else:
            mapper = BinaryMapper(num_bits=self.BITS, dtype="float64")
            logits = tf.constant(self._mapper_logits())
            # Contract each category against a DISTINCT weight. An unweighted
            # reduction is zero by construction -- `sum_d G_d == 1` exactly, so
            # `d(sum_d G_d)/dL == 0` and the probe would be vacuous (MEASURED:
            # max|grad| = 1.11e-16, i.e. float64 round-off, with the fix in).
            w = np.arange(1.0, 2 ** self.BITS + 1.0).reshape(1, 1, -1)
            with tf.GradientTape() as tape:
                tape.watch(logits)
                y = mapper(logits, training=tf.constant(True))
                s = tf.reduce_sum(y * tf.constant(w))
            grad = tape.gradient(s, logits)
            assert grad is not None, (
                "tensor-valued training=True skipped the eq. 8 gradient "
                "pass-through entirely: bit_logits got NO gradient"
            )
            assert float(np.max(np.abs(np.array(grad)))) > 1e-6, (
                "the pass-through ran but its gradient is ~0"
            )

    # ---------------------------------------------------------------- GRAPH
    @pytest.mark.parametrize("site", ["free_transformer_layer", "binary_mapper"])
    def test_traced_symbolic_training_raises_instead_of_mis_routing(self, site):
        """A genuinely traced symbolic ``training`` must be REFUSED, loudly.

        MEASURED, not cited (the repo's ``energy_transformer.py`` D-003 comment
        was the only source for this claim): the exception is
        ``tensorflow.python.framework.errors_impl.OperatorNotAllowedInGraphError``,
        "Using a symbolic `tf.Tensor` as a Python `bool` is not allowed".

        Which cell was RED-proven:

        * ``binary_mapper`` -- RED before the fix (it traced CLEANLY and
          returned a one-hot with no pass-through and no error).
        * ``free_transformer_layer`` -- already loud before the fix, but NOT at
          this branch: Keras' own ``Dropout.call`` does ``if training and ...``
          and raises the identical error ~11 lines earlier, so a symbolic
          ``training`` never reached the F-21 site at all in graph mode. Pinned
          here so the layer cannot silently become permissive again.

        Note that AutoGraph rewrites a bare ``if <tensor>:`` in USER code into
        ``tf.cond`` (measured: it does NOT raise there). Keras ``call()``
        methods are ``do_not_convert``-allowlisted, which is why the same
        statement raises inside a layer.
        """
        if site == "free_transformer_layer":
            layer, x = _seeded_free_layer(bits=self.BITS)

            @tf.function(input_signature=[
                tf.TensorSpec([2, 6, 32], tf.float32),
                tf.TensorSpec([], tf.bool)])
            def traced(inp, trn):
                out, bl = layer(inp, training=trn)
                return out, bl

            args = (x, tf.constant(True))
        else:
            mapper = BinaryMapper(num_bits=self.BITS)

            @tf.function(input_signature=[
                tf.TensorSpec([None, None, self.BITS], tf.float32),
                tf.TensorSpec([], tf.bool)])
            def traced(lg, trn):
                return mapper(lg, training=trn)

            args = (tf.constant(np.asarray(self._mapper_logits(), np.float32)),
                    tf.constant(True))

        with pytest.raises(tf.errors.OperatorNotAllowedInGraphError) as exc:
            traced(*args)
        assert "as a Python `bool` is not allowed" in str(exc.value)

    # ------------------------------------------------------- NEGATIVE CONTROL
    def test_python_training_values_behave_exactly_as_before(self, monkeypatch):
        """The false-positive family: ``None`` / ``True`` / ``False``.

        All three already resolved identically under ``is True`` (``None`` and
        ``False`` both fell to ``else``), so this change must be a no-op for
        every reachable caller. Verified by EXECUTION rather than asserted.
        """
        results = {}
        for flag in (None, True, False):
            layer, x = _seeded_free_layer(bits=self.BITS)
            _pin_uniform(monkeypatch)
            out, bl = layer(x, training=flag)
            results[repr(flag)] = (np.array(out), np.array(bl))
            monkeypatch.undo()

            mapper = BinaryMapper(num_bits=self.BITS, dtype="float64")
            _pin_uniform(monkeypatch)
            z = np.array(mapper(self._mapper_logits(), training=flag))
            monkeypatch.undo()
            assert set(np.unique(z).tolist()) <= {0.0, 1.0}, (
                f"training={flag!r}: the mapper's forward is not a one-hot")
            np.testing.assert_array_equal(z.sum(axis=-1), np.ones(z.shape[:-1]))

        # None and False take the inference branch, bit-identically
        np.testing.assert_array_equal(results["None"][0], results["False"][0])
        np.testing.assert_array_equal(results["None"][1], results["False"][1])
        assert np.all(results["None"][1] == 0.0), (
            "training=None must still emit the all-zero uniform prior")
        # True takes the encoder branch and is therefore DIFFERENT
        assert not np.all(results["True"][1] == 0.0)
        assert not np.allclose(results["True"][0], results["None"][0])

    # ------------------------------------------------------- SHIPPED CAVEAT
    def test_the_symbolic_training_limitation_is_documented_in_the_source(self):
        """``plans/`` is gitignored, so the caveat has to live in the file."""
        import inspect
        for fn in (FreeTransformerLayer.call, BinaryMapper.call):
            src = inspect.getsource(fn)
            assert "symbolic" in src and "training" in src, (
                f"{fn.__qualname__} does not document the symbolic-`training` "
                f"limitation; a decisions.md-only caveat does not ship"
            )
            # Match the STATEMENT line-exactly, not the words: both the
            # docstrings and the D-020 anchors deliberately QUOTE
            # ``if training is True:`` while explaining why it is gone, so a
            # substring test would fire on the very prose it should protect.
            stmts = [ln.strip() for ln in src.splitlines()
                     if ln.strip().startswith("if training")]
            assert "if training is True:" not in stmts, (
                f"{fn.__qualname__} still branches on an `is True` identity "
                f"test: {stmts}"
            )
            assert "if training:" in stmts, (
                f"{fn.__qualname__} has no plain-truthiness `if training:` "
                f"branch at all: {stmts}"
            )
