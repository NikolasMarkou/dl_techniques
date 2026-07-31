"""Test suite for EomtTransformer (masked self-attention with object queries)."""
import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.transformers.eomt_transformer import EomtTransformer
from dl_techniques.layers.transformers.transformer import TransformerLayer


class TestEomtTransformer:
    """Tests for EomtTransformer, focusing on the masked-attention redesign."""

    NUM_PATCHES = 16
    NUM_QUERIES = 4
    H = W = 4  # H * W == NUM_PATCHES
    DIM = 32

    @pytest.fixture
    def x(self):
        return keras.random.normal([2, self.NUM_PATCHES + self.NUM_QUERIES, self.DIM])

    @pytest.fixture
    def seg_mask(self):
        # Segmentation mask in [0, 1], shape (B, num_queries, H, W)
        return keras.ops.cast(
            keras.random.uniform([2, self.NUM_QUERIES, self.H, self.W]) > 0.5, "float32"
        )

    def _layer(self, **kw):
        params = dict(hidden_size=self.DIM, num_heads=4, use_masked_attention=True,
                      mask_probability=1.0, mask_annealing_steps=0)
        params.update(kw)
        return EomtTransformer(**params)

    def test_forward_shape(self, x, seg_mask):
        out = self._layer()(x, mask=seg_mask, training=True)
        assert tuple(out.shape) == (2, self.NUM_PATCHES + self.NUM_QUERIES, self.DIM)

    def test_inference_no_mask(self, x):
        out = self._layer()(x, mask=None, training=False)
        assert out.shape == x.shape
        assert np.all(np.isfinite(np.array(out)))

    def test_masked_attention_changes_output(self, x, seg_mask):
        """The keep-mask must actually reach attention: masked output must differ
        from the unmasked output for the same input."""
        keras.utils.set_random_seed(42)
        layer = self._layer()
        out_masked = np.array(layer(x, mask=seg_mask, training=True))
        out_unmasked = np.array(layer(x, mask=None, training=True))
        assert not np.allclose(out_masked, out_unmasked, atol=1e-5)
        assert np.all(np.isfinite(out_masked))

    def test_empty_mask_no_nan(self, x):
        """An all-zero segmentation mask must not produce NaN (query->query block
        keeps every attention row non-empty)."""
        m0 = keras.ops.zeros([2, self.NUM_QUERIES, self.H, self.W])
        out = self._layer()(x, mask=m0, training=True)
        assert np.all(np.isfinite(np.array(out)))

    def test_graph_trace_training(self, x, seg_mask):
        layer = self._layer()

        @tf.function
        def traced(inp, m):
            return layer(inp, mask=m, training=True)

        out = traced(tf.constant(np.array(x)), tf.constant(np.array(seg_mask)))
        assert tuple(out.shape) == (2, self.NUM_PATCHES + self.NUM_QUERIES, self.DIM)

    def test_get_config_round_trip(self):
        layer = self._layer(mask_annealing_steps=10)
        cfg = layer.get_config()
        rebuilt = EomtTransformer.from_config(cfg)
        assert rebuilt.use_masked_attention is True
        assert rebuilt.mask_probability == 1.0
        assert rebuilt.mask_annealing_steps == 10

    def test_model_save_load_round_trip(self, x, seg_mask):
        inp = keras.Input(shape=(self.NUM_PATCHES + self.NUM_QUERIES, self.DIM))
        out = EomtTransformer(hidden_size=self.DIM, num_heads=4,
                              use_masked_attention=True)(inp)
        model = keras.Model(inp, out)
        ref = model(x, training=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "eomt.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        out2 = loaded(x, training=False)
        np.testing.assert_allclose(np.array(ref), np.array(out2), atol=1e-5)


class TestMaskedAttentionRequiresAMaskableAttentionType:
    """F-10: `use_masked_attention=True` + a maskless `attention_type` must RAISE.

    `TransformerLayer.call` dispatches on
    `attention_type in _MASKLESS_ATTENTION_TYPES` and calls
    `self.attention(x, training=training)` for those types -- the
    `attention_mask` argument is DROPPED on the floor. So before this guard,
    `EomtTransformer` happily built a real `(B, seq, seq)` keep-mask in
    `_compute_attention_keep_mask` and then threw it away, silently.

    RED capture at `c164f43c` (pre-guard): every one of the three maskless
    types CONSTRUCTED SILENTLY -- `pytest.raises(ValueError)` reported
    `DID NOT RAISE <class 'ValueError'>`, i.e. the raise assertion itself was
    the one that fired, not a setup assertion. The identity assertion fired
    separately with `AttributeError: type object 'EomtTransformer' has no
    attribute '_MASKLESS_ATTENTION_TYPES'`.

    The guard must reject ONLY that combination. Rejecting a maskless
    `attention_type` per se, or rejecting `use_masked_attention=True` per se,
    would be an over-refusal of exactly the same severity -- hence the three
    negative controls below.
    """

    DIM = 32
    HEADS = 4
    NUM_PATCHES = 16
    NUM_QUERIES = 4
    H = W = 4  # H * W == NUM_PATCHES

    @staticmethod
    def _seed_non_zero_weights(layer, seed=1234):
        """Assign seeded NON-ZERO values to every trainable weight and bias.

        Default `bias_initializer='zeros'` makes a masking site structurally
        unobservable, so a forward-pass control built on defaults proves
        nothing. `current_step` is the annealing counter, not a parameter, and
        is deliberately left at its initialized value.
        """
        rng = np.random.default_rng(seed)
        biases_seeded = 0
        for w in layer.weights:
            if "current_step" in w.path:
                continue
            w.assign(rng.normal(0.3, 0.5, size=w.shape).astype(w.dtype))
            if "bias" in w.path:
                biases_seeded += 1
        assert biases_seeded > 0, "fixture seeded no biases -- control is vacuous"
        return layer

    @pytest.mark.parametrize("attention_type", ['fnet', 'anchor', 'lighthouse'])
    def test_maskless_attention_type_with_masked_attention_raises(self, attention_type):
        with pytest.raises(ValueError, match="use_masked_attention"):
            EomtTransformer(hidden_size=self.DIM, num_heads=self.HEADS,
                            attention_type=attention_type,
                            use_masked_attention=True)

    @pytest.mark.parametrize("attention_type", ['fnet', 'anchor', 'lighthouse'])
    def test_the_message_names_both_flag_values(self, attention_type):
        """A guard that does not say WHICH two settings conflict is a puzzle."""
        with pytest.raises(ValueError) as exc:
            EomtTransformer(hidden_size=self.DIM, num_heads=self.HEADS,
                            attention_type=attention_type,
                            use_masked_attention=True)
        msg = str(exc.value)
        assert attention_type in msg
        assert "use_masked_attention" in msg
        assert "attention_type" in msg

    def test_maskless_set_is_the_SAME_frozenset_object_as_TransformerLayer(self):
        """Object identity, not equality (I4).

        A locally re-declared `frozenset({'fnet', 'anchor', 'lighthouse'})`
        compares EQUAL and then silently drifts the day a fourth maskless type
        is added to `TransformerLayer` only -- at which point this guard would
        stop firing for it and the silent no-op would come back.
        """
        assert (
            EomtTransformer._MASKLESS_ATTENTION_TYPES
            is TransformerLayer._MASKLESS_ATTENTION_TYPES
        ), "EomtTransformer re-declared the maskless set instead of reading it"

    def test_every_maskless_type_is_covered_by_this_suite(self):
        """If a fourth maskless type appears, the parametrizations must grow."""
        assert set(TransformerLayer._MASKLESS_ATTENTION_TYPES) == {
            'fnet', 'anchor', 'lighthouse'
        }

    # --- negative controls: the guard must reject NOTHING else -------------

    @pytest.mark.parametrize("attention_type", ['fnet', 'anchor', 'lighthouse'])
    def test_a_maskless_type_is_still_legal_without_masked_attention(self, attention_type):
        """`attention_type` alone is not the defect -- the COMBINATION is."""
        layer = EomtTransformer(hidden_size=self.DIM, num_heads=self.HEADS,
                                attention_type=attention_type,
                                use_masked_attention=False)
        assert layer.attention_type == attention_type
        assert layer.use_masked_attention is False

    def test_both_flags_at_their_defaults_construct(self):
        """Both flags are non-default opt-ins; the default path is untouched."""
        layer = EomtTransformer(hidden_size=self.DIM, num_heads=self.HEADS)
        assert layer.use_masked_attention is False
        assert layer.attention_type == 'multi_head'

    def test_a_maskable_type_with_masked_attention_still_constructs_and_runs(self):
        """The true-positive family's mirror image: `multi_head` must survive
        the guard AND still honour the mask it builds."""
        keras.utils.set_random_seed(7)
        layer = EomtTransformer(hidden_size=self.DIM, num_heads=self.HEADS,
                                attention_type='multi_head',
                                use_bias=True, use_masked_attention=True)
        seq = self.NUM_PATCHES + self.NUM_QUERIES
        x = keras.random.normal([2, seq, self.DIM])
        seg_mask = keras.ops.cast(
            keras.random.uniform([2, self.NUM_QUERIES, self.H, self.W]) > 0.5,
            "float32")
        layer(x, mask=seg_mask, training=True)  # build
        self._seed_non_zero_weights(layer)

        out_masked = np.array(layer(x, mask=seg_mask, training=True))
        out_unmasked = np.array(layer(x, mask=None, training=True))
        assert out_masked.shape == (2, seq, self.DIM)
        assert np.all(np.isfinite(out_masked))
        # Live control: the mask must actually reach attention for a maskable
        # type, otherwise "still constructs" would be worth nothing.
        assert not np.allclose(out_masked, out_unmasked, atol=1e-5)
