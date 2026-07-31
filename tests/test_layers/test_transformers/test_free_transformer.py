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
        """The posterior must not be conditioned on PAD content. RED at HEAD."""
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
