"""Tests for the BLT (Byte Latent Transformer) building blocks.

Covers all seven layers in ``blt_blocks.py``. Single-tensor-input layers
(``EntropyModel``, ``GlobalTransformer``) use a functional ``.keras`` round-trip;
the multi-argument-``call`` layers (``LocalEncoder``, ``LocalDecoder``) use a
``keras.Model`` wrapper round-trip (recreates the layer + restores weights),
which exercises the ``build()`` weight-structure consistency the H6 fix touches.
"""

import os
import math
import keras
import logging
import numpy as np
import pytest

from dl_techniques.layers.blt_blocks import (
    ByteTokenizer,
    EntropyModel,
    DynamicPatcher,
    PatchPooling,
    LocalEncoder,
    GlobalTransformer,
    LocalDecoder,
)

VOCAB, HID, SEQ, NP_, GDIM = 32, 16, 10, 8, 16
B = 2


def _tokens():
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).integers(0, VOCAB, size=(B, SEQ)).astype("int32")
    )


def _patch_ids():
    return keras.ops.convert_to_tensor(
        np.random.default_rng(1).integers(0, NP_, size=(B, SEQ)).astype("int32")
    )


# ---------------------------------------------------------------------
# ByteTokenizer (utility layer, no call)
# ---------------------------------------------------------------------

class TestByteTokenizer:

    def test_text_round_trip(self):
        tok = ByteTokenizer()
        ids = tok.text_to_bytes("Hello", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
        assert tok.tokens_to_text(ids) == "Hello"

    def test_compute_output_shape(self):
        assert ByteTokenizer().compute_output_shape((B, SEQ)) == (B, None)

    def test_get_config_round_trip(self):
        tok = ByteTokenizer(vocab_size=300, byte_offset=5)
        rebuilt = ByteTokenizer.from_config(tok.get_config())
        assert rebuilt.vocab_size == 300 and rebuilt.byte_offset == 5


# ---------------------------------------------------------------------
# DynamicPatcher (stateless tensor op)
# ---------------------------------------------------------------------

class TestDynamicPatcher:

    def test_forward_pass(self):
        patcher = DynamicPatcher(max_patches=NP_)
        entropy = keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((B, SEQ)).astype("float32")
        )
        out = patcher(entropy)
        assert tuple(out.shape) == (B, NP_)

    def test_get_config_round_trip(self):
        patcher = DynamicPatcher(entropy_threshold=2.0, max_patches=16)
        rebuilt = DynamicPatcher.from_config(patcher.get_config())
        assert rebuilt.max_patches == 16

    # -- entropy-driven patching -------------------------------------
    # `compute_patch_ids` does NOT validate that a row sums to `seq_len`;
    # it silently misassigns ids instead. The row sum is therefore pinned
    # here, at both degenerate ends, rather than being left to the
    # consumer to notice.

    @staticmethod
    def _lengths(rows, threshold=1.0, max_patches=4):
        patcher = DynamicPatcher(entropy_threshold=threshold,
                                 max_patches=max_patches)
        entropy = keras.ops.convert_to_tensor(np.asarray(rows, "float32"))
        return keras.ops.convert_to_numpy(patcher(entropy))

    @pytest.mark.parametrize(
        "name,row,expected",
        [
            # threshold 1.0, max_patches 4, seq_len 6
            ("zero_crossings", [0., 0., 0., 0., 0., 0.], [6, 0, 0, 0]),
            ("crossing_at_position_zero", [2., 0., 0., 0., 0., 0.], [0, 6, 0, 0]),
            ("exactly_max_patches_minus_one", [0., 2., 0., 2., 0., 2.], [1, 2, 2, 1]),
            ("more_crossings_than_max_patches", [2., 2., 2., 2., 2., 2.], [0, 1, 1, 4]),
        ],
    )
    def test_rows_sum_to_seq_len_in_every_degenerate_case(self, name, row, expected):
        got = self._lengths([row])[0]
        assert got.sum() == len(row), (
            f"{name}: row sums to {got.sum()}, not seq_len={len(row)} "
            f"(compute_patch_ids will silently misassign ids): {got.tolist()}"
        )
        assert (got >= 0).all(), f"{name}: negative length in {got.tolist()}"
        assert got.tolist() == expected, (
            f"{name}: expected {expected}, got {got.tolist()}"
        )

    def test_rows_sum_to_seq_len_when_seq_len_is_below_max_patches(self):
        got = self._lengths([[0., 2., 0.]], max_patches=8)[0]
        assert got.sum() == 3, f"row sums to {got.sum()}, not 3: {got.tolist()}"
        assert got.tolist() == [1, 2, 0, 0, 0, 0, 0, 0]

    def test_the_threshold_moves_the_patch_lengths_on_real_entropy(self):
        """`entropy_threshold` was stored and serialized at ~15 sites and read
        at NONE; nothing in the suite would have failed with it permanently
        dead. The thresholds swept here are derived from the fixture's OWN
        measured entropy, because a threshold outside the data's range changes
        nothing and would pass vacuously.

        MEASURED (this fixture, CPU, random init, vocab 32 so the uniform
        ceiling is ln(32) = 3.466 nats): entropy lands in roughly
        [2.9, 3.4] nats. Note that the SHIPPED default of 1.5 sits BELOW that
        whole band -- on an untrained model every position is a boundary.
        """
        model = EntropyModel(vocab_size=VOCAB, hidden_dim=HID, num_layers=1,
                             num_heads=2, max_seq_len=64)
        entropy = model.compute_entropy(model(_tokens(), training=False))
        values = keras.ops.convert_to_numpy(entropy)

        lo, hi = float(values.min()), float(values.max())
        assert hi - lo > 1e-3, (
            f"fixture entropy is degenerate ([{lo:.4f}, {hi:.4f}]); no "
            f"threshold could separate it and the sweep below would be vacuous"
        )
        low_t, high_t = float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))

        few = keras.ops.convert_to_numpy(
            DynamicPatcher(entropy_threshold=high_t, max_patches=NP_)(entropy))
        many = keras.ops.convert_to_numpy(
            DynamicPatcher(entropy_threshold=low_t, max_patches=NP_)(entropy))

        assert not np.array_equal(few, many), (
            f"entropy_threshold is inert: thresholds {high_t:.4f} and "
            f"{low_t:.4f}, both inside the fixture's measured entropy range "
            f"[{lo:.4f}, {hi:.4f}], produced identical patch lengths "
            f"{few.tolist()}"
        )
        assert few.sum(axis=1).tolist() == [SEQ] * B
        assert many.sum(axis=1).tolist() == [SEQ] * B
        # A higher threshold admits no more boundaries than a lower one, so it
        # can only occupy fewer or equal patch slots.
        assert (few > 0).sum() <= (many > 0).sum()

    def test_rows_are_derived_independently(self):
        """The pre-fix implementation broadcast ONE row to the whole batch, so
        two sequences with different entropy got identical boundaries."""
        got = self._lengths([
            [0., 0., 0., 0., 0., 0.],
            [0., 2., 0., 2., 0., 0.],
        ])
        assert not np.array_equal(got[0], got[1]), (
            f"both rows got {got[0].tolist()} -- the batch is still broadcast "
            f"from a single row instead of being segmented per sequence"
        )
        assert got[0].tolist() == [6, 0, 0, 0]
        assert got[1].tolist() == [1, 2, 3, 0]

    def test_patch_ids_depend_only_on_the_past(self):
        """Causality at the unit level: the id of byte t is the (saturated)
        count of boundaries at positions <= t, so perturbing entropy at a
        later position cannot move any earlier id. Asserted through the real
        `compute_patch_ids` round-trip, which is what the model consumes.
        """
        patcher = DynamicPatcher(entropy_threshold=1.0, max_patches=4)
        base = np.array([[0., 2., 0., 0., 2., 0., 0., 0.]], "float32")
        after = base.copy()
        after[0, 5:] = 2.0  # every change is at position >= 5

        def ids(rows):
            lengths = patcher(keras.ops.convert_to_tensor(rows))
            return keras.ops.convert_to_numpy(patcher.compute_patch_ids(lengths))

        a, b = ids(base), ids(after)
        assert np.array_equal(a[:, :5], b[:, :5]), (
            f"future leak: entropy changed only at positions >= 5 but the "
            f"patch ids before it moved from {a[0, :5].tolist()} to "
            f"{b[0, :5].tolist()}"
        )
        assert not np.array_equal(a[:, 5:], b[:, 5:]), "probe is inert"
        for row in (a, b):
            assert (np.diff(row[0]) >= 0).all(), (
                f"patch ids are not non-decreasing: {row[0].tolist()} "
                f"(LocalDecoder's preceding-patch gather requires this)"
            )

    def test_a_late_boundary_cannot_displace_an_earlier_one(self):
        """The deterministic gate on the inadmissible algorithm.

        A `top_k`-by-entropy-MAGNITUDE cap would keep the `max_patches - 1`
        HIGHEST-entropy crossings, so a late, very high entropy value evicts
        an earlier, lower one and an EARLIER byte's patch id changes. The
        position-ordered cap cannot do that: the row below is already
        saturated, so the extra late crossing is absorbed and NOTHING moves.

        "Nothing moves" is the correct answer here, which is why the sibling
        test below pins that the same construction DOES move when there is
        boundary budget left -- otherwise this assertion would be satisfied
        by a patcher that ignored entropy entirely.
        """
        patcher = DynamicPatcher(entropy_threshold=1.0, max_patches=4)
        # crossings at 1, 3, 5 -- exactly max_patches - 1, so the row is full.
        base = np.array([[0., 3.0, 0., 2.5, 0., 2.0, 0., 0.5]], "float32")
        after = base.copy()
        after[0, 7] = 9.0  # a LATE crossing that outranks every earlier one

        def ids(rows):
            lengths = patcher(keras.ops.convert_to_tensor(rows))
            return keras.ops.convert_to_numpy(patcher.compute_patch_ids(lengths))

        a, b = ids(base), ids(after)
        assert np.array_equal(a[:, :7], b[:, :7]), (
            f"a boundary at position 7 displaced an earlier one: patch ids "
            f"before it moved from {a[0, :7].tolist()} to {b[0, :7].tolist()} "
            f"-- this is the magnitude-ranked selection the D-012 anchor "
            f"forbids"
        )
        assert a[0].tolist() == [0, 1, 1, 2, 2, 3, 3, 3]

    def test_a_late_boundary_is_taken_when_there_is_budget_left(self):
        """Liveness companion for the test above: with only two crossings the
        row is not saturated, so raising the LAST position above the threshold
        must open a third patch."""
        patcher = DynamicPatcher(entropy_threshold=1.0, max_patches=4)
        base = np.array([[0., 3.0, 0., 2.5, 0., 0., 0., 0.5]], "float32")
        after = base.copy()
        after[0, 7] = 9.0

        def ids(rows):
            lengths = patcher(keras.ops.convert_to_tensor(rows))
            return keras.ops.convert_to_numpy(patcher.compute_patch_ids(lengths))

        a, b = ids(base), ids(after)
        assert a[0].tolist() == [0, 1, 1, 2, 2, 2, 2, 2]
        assert b[0].tolist() == [0, 1, 1, 2, 2, 2, 2, 3], (
            f"the patcher is inert: raising entropy at position 7 above the "
            f"threshold left the ids at {b[0].tolist()}"
        )

    def test_graph_mode_matches_eager(self):
        """`max_patches` must stay a static Python int and the body must stay
        raw-`tf`-free: either mistake compiles fine eagerly and breaks in the
        regime `fit()` actually uses."""
        import tensorflow as tf

        patcher = DynamicPatcher(entropy_threshold=1.0, max_patches=NP_)
        entropy = np.random.default_rng(3).standard_normal((B, SEQ)).astype("float32")

        eager = keras.ops.convert_to_numpy(patcher(keras.ops.convert_to_tensor(entropy)))

        @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
        def traced(x):
            return patcher(x)

        graph = traced(tf.constant(entropy)).numpy()
        assert np.array_equal(eager, graph), (
            f"graph mode disagrees with eager: {eager.tolist()} vs {graph.tolist()}"
        )
        assert graph.sum(axis=1).tolist() == [SEQ] * B


# ---------------------------------------------------------------------
# The SHIPPED DEFAULT threshold, pinned rather than left derivable
# ---------------------------------------------------------------------

class TestTheShippedDefaultThreshold:
    """Every other patcher test picks a threshold from its own fixture's
    entropy range — correctly, since that is what makes them non-vacuous. The
    consequence is that nothing exercises the value a caller who passes no
    argument actually gets. These tests pin it.
    """

    BYTE_VOCAB = 260  # ByteLatentTransformer's default vocab_size

    def _uniform_entropy(self, seq_len, vocab_size=None):
        """Entropy of a uniform next-byte distribution: the ceiling ln(V).

        This is what an UNTRAINED entropy model emits, and it is the regime
        every fresh model, every smoke test and the first epochs of every
        training run are in.
        """
        ceiling = math.log(float(vocab_size or self.BYTE_VOCAB))
        return keras.ops.convert_to_tensor(
            np.full((1, seq_len), ceiling, dtype="float32")
        )

    def test_default_threshold_makes_every_position_a_boundary(self):
        patcher = DynamicPatcher()  # entropy_threshold=1.5, max_patches=512
        assert patcher.entropy_threshold == 1.5, (
            "this test pins the behaviour of the SHIPPED default; the default moved"
        )

        lengths = keras.ops.convert_to_numpy(
            patcher(self._uniform_entropy(seq_len=12))
        )[0]

        # Patch 0 is EMPTY (position 0 is itself a boundary), then one byte per
        # patch, then nothing. Not adaptive patching: byte-level tokens.
        assert lengths[:13].tolist() == [0] + [1] * 12, (
            "the shipped default no longer degenerates to one byte per patch on "
            f"uniform ln({self.BYTE_VOCAB}) entropy: {lengths[:13].tolist()}"
        )
        assert lengths[13:].sum() == 0
        assert lengths.sum() == 12  # row-sum invariant holds regardless

    def test_default_threshold_collapses_the_tail_into_the_final_patch(self):
        # Same default threshold, a budget smaller than the sequence: the
        # segmentation is strictly WORSE than an equal-length split.
        patcher = DynamicPatcher(max_patches=8)
        lengths = keras.ops.convert_to_numpy(
            patcher(self._uniform_entropy(seq_len=20))
        )[0]

        assert lengths.tolist() == [0, 1, 1, 1, 1, 1, 1, 14], (
            f"the shipped default's tail collapse moved: {lengths.tolist()}"
        )
        assert lengths.sum() == 20


class TestObservedDegeneracyWarning:
    """The degeneracy check reads the ENTROPY, not the vocabulary size.

    Its predecessor compared ``entropy_threshold`` to ``0.5 * ln(vocab_size)``,
    which fired on every shipped configuration (1.5 and 1.3 against a 2.78-nat
    floor at ``vocab_size=260``) and could never fire on the regime it was
    aimed at -- a *trained* entropy model at 1.5 nats is exactly the case it
    called degenerate. These tests pin the two ends it now reports and, more
    importantly, the middle it must stay silent on.
    """

    def _entropy(self, values):
        return keras.ops.convert_to_tensor(np.array([values], dtype="float32"))

    def test_it_warns_when_every_position_is_a_boundary(self, caplog):
        patcher = DynamicPatcher(entropy_threshold=1.5, max_patches=8)
        with caplog.at_level(logging.WARNING):
            warned = patcher.warn_if_segmentation_is_degenerate(
                self._entropy([5.5] * 12)  # untrained model: near ln(260)=5.56
            )
        assert warned is True, (
            "entropy above the threshold at EVERY position gives an empty patch "
            "0, one byte per patch, and the whole tail merged into the last "
            "patch -- and it was reported as an ordinary segmentation"
        )
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "1.0" in text and "every position" in text.lower(), (
            f"the warning does not name the observed rate or consequence: {text}"
        )

    def test_it_warns_when_no_position_is_a_boundary(self, caplog):
        patcher = DynamicPatcher(entropy_threshold=1.5, max_patches=8)
        with caplog.at_level(logging.WARNING):
            warned = patcher.warn_if_segmentation_is_degenerate(
                self._entropy([0.2] * 12)
            )
        assert warned is True, (
            "no boundary at all means one patch for the whole sequence and an "
            "inert max_patches; that end is degenerate too"
        )
        assert "0.0" in " ".join(r.getMessage() for r in caplog.records)

    def test_it_is_silent_on_an_ordinary_segmentation(self, caplog):
        """The property the predecessor could not have: silence in the good case.

        A *trained* byte-level entropy model emits roughly 1-2 nats per byte, so
        the shipped default of 1.5 lands mid-sequence — the configuration the
        old check warned about unconditionally and this one must not.
        """
        patcher = DynamicPatcher(entropy_threshold=1.5, max_patches=8)
        with caplog.at_level(logging.WARNING):
            warned = patcher.warn_if_segmentation_is_degenerate(
                self._entropy([0.9, 2.1, 1.0, 0.8, 2.4, 1.1, 0.7, 1.9])
            )
        assert warned is False, (
            "a mixed-entropy batch at the shipped default is an ordinary "
            "segmentation and must not be reported"
        )
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_it_does_not_change_the_segmentation(self):
        # Diagnostic, not validation: calling it must leave `call` untouched.
        patcher = DynamicPatcher(entropy_threshold=1.5, max_patches=8)
        entropy = self._entropy([0.9, 2.1, 1.0, 0.8, 2.4, 1.1, 0.7, 1.9])
        before = keras.ops.convert_to_numpy(patcher(entropy))
        patcher.warn_if_segmentation_is_degenerate(entropy)
        after = keras.ops.convert_to_numpy(patcher(entropy))
        np.testing.assert_array_equal(before, after)


# ---------------------------------------------------------------------
# EntropyModel (single-tensor call -> functional .keras round-trip)
# ---------------------------------------------------------------------

class TestEntropyModel:

    def _make(self):
        return EntropyModel(vocab_size=VOCAB, hidden_dim=HID, num_layers=1,
                            num_heads=2, max_seq_len=64)

    def test_forward_pass(self):
        out = self._make()(_tokens())
        assert tuple(out.shape) == (B, SEQ, VOCAB)

    def test_compute_output_shape(self):
        assert self._make().compute_output_shape((B, SEQ)) == (B, SEQ, VOCAB)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(SEQ,), dtype="int32")
        out = self._make()(inp)
        model = keras.Model(inp, out)
        toks = _tokens()
        y0 = model(toks)
        path = os.path.join(tmp_path, "entropy.keras")
        model.save(path)
        loaded = keras.models.load_model(path, custom_objects={"EntropyModel": EntropyModel})
        y1 = loaded(toks)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------
# GlobalTransformer (single-tensor call -> functional .keras round-trip)
# ---------------------------------------------------------------------

class TestGlobalTransformer:

    def _make(self):
        return GlobalTransformer(global_dim=GDIM, num_global_layers=1,
                                num_heads_global=2, max_patches=NP_)

    def test_forward_pass(self):
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((B, NP_, GDIM)).astype("float32")
        )
        assert tuple(self._make()(x).shape) == (B, NP_, GDIM)

    def test_compute_output_shape(self):
        assert self._make().compute_output_shape((B, NP_, GDIM)) == (B, NP_, GDIM)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(NP_, GDIM))
        out = self._make()(inp)
        model = keras.Model(inp, out)
        x = np.random.default_rng(0).standard_normal((B, NP_, GDIM)).astype("float32")
        y0 = model(x)
        path = os.path.join(tmp_path, "global.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"GlobalTransformer": GlobalTransformer}
        )
        y1 = loaded(x)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------
# PatchPooling (multi-arg call)
# ---------------------------------------------------------------------

class TestPatchPooling:

    def test_forward_pass(self):
        pool = PatchPooling(output_dim=GDIM, num_queries=2, max_patches=NP_)
        byte_hiddens = keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((B, SEQ, HID)).astype("float32")
        )
        out = pool(byte_hiddens, _patch_ids())
        assert out.shape[0] == B and out.shape[-1] == GDIM

    def test_get_config_round_trip(self):
        pool = PatchPooling(output_dim=GDIM, num_queries=2, max_patches=NP_)
        rebuilt = PatchPooling.from_config(pool.get_config())
        assert rebuilt.output_dim == GDIM


# ---------------------------------------------------------------------
# LocalEncoder / LocalDecoder (multi-arg call -> Model-wrapper round-trip)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class _EncWrapper(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.enc = LocalEncoder(
            vocab_size=VOCAB, local_dim=HID, num_local_layers=1, num_heads_local=2,
            max_sequence_length=64, max_patches=NP_, global_dim=GDIM,
            cross_attention_queries=2,
        )

    def call(self, inputs, training=None):
        return self.enc(inputs[0], inputs[1], training=training)


@keras.saving.register_keras_serializable()
class _DecWrapper(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.dec = LocalDecoder(
            vocab_size=VOCAB, local_dim=HID, global_dim=GDIM, num_local_layers=1,
            num_heads_local=2,
        )

    def call(self, inputs, training=None):
        return self.dec(inputs[0], inputs[1], inputs[2], training=training)


class TestLocalEncoder:

    def test_forward_pass(self):
        enc = LocalEncoder(
            vocab_size=VOCAB, local_dim=HID, num_local_layers=1, num_heads_local=2,
            max_sequence_length=64, max_patches=NP_, global_dim=GDIM,
            cross_attention_queries=2,
        )
        out = enc(_tokens(), _patch_ids())
        assert tuple(out.shape) == (B, NP_, GDIM)

    def test_compute_output_shape(self):
        enc = LocalEncoder(
            vocab_size=VOCAB, local_dim=HID, max_patches=NP_, global_dim=GDIM,
        )
        assert enc.compute_output_shape((B, SEQ)) == (B, NP_, GDIM)

    def test_serialization_round_trip(self, tmp_path):
        model = _EncWrapper()
        inputs = [_tokens(), _patch_ids()]
        y0 = model(inputs)
        path = os.path.join(tmp_path, "enc.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y1 = loaded(inputs)
        assert tuple(y0.shape) == tuple(y1.shape) == (B, NP_, GDIM)


class TestLocalDecoder:

    def _gctx(self):
        return keras.ops.convert_to_tensor(
            np.random.default_rng(2).standard_normal((B, NP_, GDIM)).astype("float32")
        )

    def test_forward_pass(self):
        dec = LocalDecoder(vocab_size=VOCAB, local_dim=HID, global_dim=GDIM,
                          num_local_layers=1, num_heads_local=2)
        out = dec(_tokens(), self._gctx(), _patch_ids())
        assert tuple(out.shape) == (B, SEQ, VOCAB)

    def test_serialization_round_trip(self, tmp_path):
        model = _DecWrapper()
        inputs = [_tokens(), self._gctx(), _patch_ids()]
        y0 = model(inputs)
        path = os.path.join(tmp_path, "dec.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y1 = loaded(inputs)
        assert tuple(y0.shape) == tuple(y1.shape) == (B, SEQ, VOCAB)
