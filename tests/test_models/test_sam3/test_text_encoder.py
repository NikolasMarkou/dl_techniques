"""Tests for SAM 3's CLIP text tower wrapper (`models/sam3/text_encoder_ve.py`).

The load-bearing guard here is CAUSALITY, and it is asserted as an **exact zero**
rather than against a tolerance. The wrapped encoder is bidirectional by default,
so the failure mode is a silent value leak with no shape symptom and no
exception: perturbing the last token moves the position-0 output by ``5.5e-3`` at
this file's tiny width and by ``0.1404891`` at the settled 1024-wide tower (both
MEASURED here, the second one by
``TestSettledScale::test_settled_width_causality_and_leak_magnitude``). A
tolerance chosen to feel safe is exactly wide enough to hide the tiny-scale leak,
and the settled-scale number is 32x the toy figure the exploration phase
measured -- which is why this file re-measures instead of inheriting.

An absence assertion (``delta == 0.0``) is satisfied BY CONSTRUCTION by a dead
component, so the causality family ships with two positive arms beside it:
``test_the_last_position_does_see_position_zero`` (the mask must not be a
blanket) and ``test_the_causality_guard_can_see_a_leak`` (the comparator itself
is proven able to fire, by running the same tower with NO mask).
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.sam3.text_encoder_ve import Sam3TextEncoder

# ---------------------------------------------------------------------
# tiny variant
# ---------------------------------------------------------------------

TINY = dict(
    d_model=8, width=32, depth=2, num_heads=4, context_length=8,
    vocab_size=100, mlp_ratio=4.0,
)
TINY_SEQ = TINY["context_length"]

# The settled SAM 3 text tower, re-read from the pinned upstream clone's
# `_create_text_encoder()` (d_model=256, width=1024, heads=16, layers=24) plus
# `TextTransformer`'s defaults for the arguments that call does not pass
# (context_length=32, vocab_size=49408, mlp_ratio=4.0, use_ln_post=True).
SHIPPED = dict(
    d_model=256, width=1024, depth=24, num_heads=16, context_length=32,
    vocab_size=49408, mlp_ratio=4.0,
)


def _params(cfg: dict) -> int:
    """Closed-form parameter count for a `Sam3TextEncoder`.

    Written from the STRUCTURE, not read off a `count_params()` call, so that a
    silently dropped sub-layer changes the measurement and not the oracle.
    """
    width, depth = cfg["width"], cfg["depth"]
    hidden = int(cfg["width"] * cfg["mlp_ratio"])
    embeddings = cfg["vocab_size"] * width + cfg["context_length"] * width
    embed_norm = 2 * width                      # scale + offset
    attention = 4 * (width * width + width)     # q, k, v, o -- all biased
    block_norms = 2 * (2 * width)               # pre-attn + pre-ffn
    ffn = (width * hidden + hidden) + (hidden * width + width)
    final_norm = 2 * width                      # the 'pre' regime's terminal norm
    resizer = width * cfg["d_model"] + cfg["d_model"]
    return (
        embeddings + embed_norm
        + depth * (attention + block_norms + ffn)
        + final_norm + resizer
    )


def _ids(seq: int = TINY_SEQ, vocab: int = TINY["vocab_size"], seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, vocab, size=(2, seq)).astype("int32")


@pytest.fixture()
def tiny_encoder() -> Sam3TextEncoder:
    keras.utils.set_random_seed(7)
    encoder = Sam3TextEncoder(**TINY)
    encoder.build((None, TINY_SEQ))
    return encoder


def _forward(encoder: Sam3TextEncoder, ids) -> np.ndarray:
    return ops.convert_to_numpy(encoder(ids, training=False))


# ---------------------------------------------------------------------


class TestConstruction:

    def test_forward_shape_is_the_full_token_sequence(self, tiny_encoder):
        out = _forward(tiny_encoder, _ids())
        assert out.shape == (2, TINY_SEQ, TINY["d_model"])

    def test_output_rank_is_three_not_a_pooled_vector(self, tiny_encoder):
        # M4.3: routing through any pooled path collapses the sequence axis.
        assert _forward(tiny_encoder, _ids()).ndim == 3

    def test_the_sequence_axis_is_preserved_exactly(self, tiny_encoder):
        for seq in (1, 3, TINY_SEQ):
            out = _forward(tiny_encoder, _ids(seq=seq))
            assert out.shape[1] == seq

    def test_compute_output_shape_matches_the_forward_pass(self, tiny_encoder):
        declared = tiny_encoder.compute_output_shape((2, TINY_SEQ))
        assert declared == _forward(tiny_encoder, _ids()).shape

    @pytest.mark.parametrize("bad", [
        dict(d_model=0), dict(width=0), dict(depth=0), dict(num_heads=0),
        dict(context_length=0), dict(vocab_size=0), dict(mlp_ratio=0.0),
        dict(width=30),  # not divisible by num_heads=4
    ])
    def test_invalid_configuration_raises(self, bad):
        with pytest.raises(ValueError):
            Sam3TextEncoder(**{**TINY, **bad})

    def test_non_rank_two_input_raises(self):
        with pytest.raises(ValueError, match="batch, seq"):
            Sam3TextEncoder(**TINY).build((None, TINY_SEQ, 1))

    def test_sequence_longer_than_the_context_raises(self):
        with pytest.raises(ValueError, match="exceeds context_length"):
            Sam3TextEncoder(**TINY).build((None, TINY_SEQ + 1))


class TestCausality:
    """The one defect class that is silent. Exact zero, plus positive arms."""

    def test_position_zero_is_exactly_invariant_to_the_last_token(
            self, tiny_encoder
    ):
        ids = _ids()
        bumped = ids.copy()
        bumped[:, -1] = (bumped[:, -1] + 37) % TINY["vocab_size"]
        base = _forward(tiny_encoder, ids)
        moved = _forward(tiny_encoder, bumped)
        delta = np.abs(base[:, 0] - moved[:, 0]).max()
        assert delta == 0.0, f"future token leaked into position 0 by {delta}"

    def test_every_position_is_invariant_to_all_strictly_later_tokens(
            self, tiny_encoder
    ):
        ids = _ids()
        base = _forward(tiny_encoder, ids)
        for cut in range(1, TINY_SEQ):
            bumped = ids.copy()
            bumped[:, cut:] = (bumped[:, cut:] + 11) % TINY["vocab_size"]
            moved = _forward(tiny_encoder, bumped)
            delta = np.abs(base[:, :cut] - moved[:, :cut]).max()
            assert delta == 0.0, f"positions <{cut} moved by {delta}"

    def test_the_last_position_does_see_position_zero(self, tiny_encoder):
        # POSITIVE arm: a transposed (upper-triangular) mask, or a mask that
        # keeps nothing, would also satisfy the absence assertions above.
        ids = _ids()
        bumped = ids.copy()
        bumped[:, 0] = (bumped[:, 0] + 23) % TINY["vocab_size"]
        base = _forward(tiny_encoder, ids)
        moved = _forward(tiny_encoder, bumped)
        assert np.abs(base[:, -1] - moved[:, -1]).max() > 1e-4

    def test_the_causality_guard_can_see_a_leak(self, tiny_encoder):
        # The comparator is RED-proven in-suite: the SAME tower, called without
        # the mask this wrapper supplies, leaks measurably into position 0.
        ids = _ids()
        bumped = ids.copy()
        bumped[:, -1] = (bumped[:, -1] + 37) % TINY["vocab_size"]
        inner = tiny_encoder.encoder
        base = ops.convert_to_numpy(inner(ids, training=False))
        moved = ops.convert_to_numpy(inner(bumped, training=False))
        leak = np.abs(base[:, 0] - moved[:, 0]).max()
        assert leak > 1e-4, (
            f"the unmasked tower did NOT leak ({leak}); the exact-zero causality "
            f"assertions above are therefore vacuous"
        )


class TestCausalMask:

    def test_mask_is_lower_triangular_and_boolean(self, tiny_encoder):
        mask = ops.convert_to_numpy(
            tiny_encoder.causal_keep_mask(2, TINY_SEQ)
        )
        assert mask.dtype == np.bool_
        assert mask.shape == (2, TINY_SEQ, TINY_SEQ)
        expected = np.tril(np.ones((TINY_SEQ, TINY_SEQ), dtype=bool))
        np.testing.assert_array_equal(mask[0], expected)
        np.testing.assert_array_equal(mask[1], expected)

    def test_mask_keeps_the_diagonal(self, tiny_encoder):
        # `>=` not `>`: a token must attend to ITSELF. An off-by-one here makes
        # position 0 attend to nothing at all.
        mask = ops.convert_to_numpy(tiny_encoder.causal_keep_mask(1, TINY_SEQ))
        assert bool(mask[0, 0, 0]) is True

    def test_mask_is_built_at_the_supplied_length_not_the_context(
            self, tiny_encoder
    ):
        mask = ops.convert_to_numpy(tiny_encoder.causal_keep_mask(1, 3))
        assert mask.shape == (1, 3, 3)


class TestNoPooling:

    def test_the_wrapped_encoder_is_constructed_with_no_pooling(
            self, tiny_encoder
    ):
        assert tiny_encoder.encoder.output_mode == "none"

    def test_call_does_not_mutate_the_pooling_strategy(self, tiny_encoder):
        # `get_sequence_features()` reaches the same tensor by mutating this
        # attribute and re-invoking the layer. Nothing in `call()` may do that.
        before = list(tiny_encoder.encoder.pooling_layer.strategy)
        _forward(tiny_encoder, _ids())
        assert list(tiny_encoder.encoder.pooling_layer.strategy) == before

    def test_no_end_of_text_pooling_surface_exists(self, tiny_encoder):
        assert not hasattr(tiny_encoder, "pooled")
        assert not any(
            "argmax" in name or "eot" in name
            for name in dir(tiny_encoder)
        )


class TestUpstreamStructuralParity:
    """D-102: the wrapped layer's DEFAULTS are not the reference's structure."""

    def test_blocks_are_pre_normalized(self, tiny_encoder):
        assert tiny_encoder.encoder.normalization_position == "pre"

    def test_a_terminal_normalization_exists(self, tiny_encoder):
        assert tiny_encoder.encoder.final_norm is not None

    def test_dropout_is_disabled_everywhere(self, tiny_encoder):
        inner = tiny_encoder.encoder
        assert inner.dropout_rate == 0.0
        assert inner.attention_dropout_rate == 0.0
        assert inner.embed_dropout_rate == 0.0

    def test_training_true_is_deterministic(self, tiny_encoder):
        ids = _ids()
        first = ops.convert_to_numpy(tiny_encoder(ids, training=True))
        second = ops.convert_to_numpy(tiny_encoder(ids, training=True))
        assert np.abs(first - second).max() == 0.0

    def test_positional_embeddings_are_learned_absolute(self, tiny_encoder):
        assert tiny_encoder.encoder.positional_type == "learned"

    def test_feed_forward_width_truncates_rather_than_rounds(self):
        # int(32 * 2.55) == 81 while round(32 * 2.55) == 82 -- a probe point
        # where the two rules differ, unlike the settled 4.0.
        encoder = Sam3TextEncoder(**{**TINY, "mlp_ratio": 2.55})
        assert encoder.encoder.intermediate_size == 81


class TestParameterAudit:

    def test_tiny_count_matches_the_closed_form_exactly(self, tiny_encoder):
        assert tiny_encoder.count_params() == _params(TINY)

    def test_the_resizer_is_a_biased_projection_to_d_model(self, tiny_encoder):
        kernel, bias = tiny_encoder.resizer.weights
        assert tuple(kernel.shape) == (TINY["width"], TINY["d_model"])
        assert tuple(bias.shape) == (TINY["d_model"],)

    def test_the_resizer_sees_the_whole_sequence_not_a_pooled_vector(
            self, tiny_encoder
    ):
        # Perturbing ONE position must move that position's output. A resizer
        # applied to a pooled vector cannot satisfy this per-position.
        ids = _ids()
        bumped = ids.copy()
        bumped[:, 2] = (bumped[:, 2] + 5) % TINY["vocab_size"]
        base = _forward(tiny_encoder, ids)
        moved = _forward(tiny_encoder, bumped)
        assert np.abs(base[:, 2] - moved[:, 2]).max() > 1e-4


class TestLiveness:

    def test_every_position_emits_a_non_constant_vector(self, tiny_encoder):
        # A dead resizer satisfies every absence assertion in this file by
        # construction. A constant output has a fingerprint you COUNT.
        out = _forward(tiny_encoder, _ids())
        for position in range(TINY_SEQ):
            values = out[:, position, :]
            assert len(np.unique(values)) > 1
            assert float(values.std()) > 0.0


class TestSerialization:

    def test_config_round_trip_preserves_every_init_parameter(self):
        encoder = Sam3TextEncoder(**TINY)
        config = encoder.get_config()
        for key, value in TINY.items():
            assert config[key] == value
        rebuilt = Sam3TextEncoder.from_config(config)
        assert rebuilt.get_config() == config

    def test_full_keras_roundtrip_preserves_output_VALUES(self, tmp_path):
        # D-098: weight counts and weight PATHS are the instrument that FAILED.
        # Only an output-value comparison sees freshly-initialized kernels.
        keras.utils.set_random_seed(3)
        inputs = keras.Input(shape=(TINY_SEQ,), dtype="int32")
        model = keras.Model(inputs, Sam3TextEncoder(**TINY)(inputs))
        ids = _ids()
        before = ops.convert_to_numpy(model(ids, training=False))
        path = tmp_path / "sam3_text.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        after = ops.convert_to_numpy(restored(ids, training=False))
        assert np.abs(before - after).max() == 0.0

    def test_the_restored_layer_is_still_causal(self, tmp_path):
        keras.utils.set_random_seed(3)
        inputs = keras.Input(shape=(TINY_SEQ,), dtype="int32")
        model = keras.Model(inputs, Sam3TextEncoder(**TINY)(inputs))
        path = tmp_path / "sam3_text.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        ids = _ids()
        bumped = ids.copy()
        bumped[:, -1] = (bumped[:, -1] + 37) % TINY["vocab_size"]
        base = ops.convert_to_numpy(restored(ids, training=False))
        moved = ops.convert_to_numpy(restored(bumped, training=False))
        assert np.abs(base[:, 0] - moved[:, 0]).max() == 0.0


class TestSettledScale:
    """The 353M-parameter shipped tower, instantiated ONCE for this class."""

    @pytest.fixture(scope="class")
    def shipped(self) -> Sam3TextEncoder:
        keras.utils.set_random_seed(11)
        encoder = Sam3TextEncoder(**SHIPPED)
        encoder.build((None, SHIPPED["context_length"]))
        return encoder

    def test_shipped_parameter_count_is_exact(self, shipped):
        assert shipped.count_params() == _params(SHIPPED) == 353_202_432

    def test_settled_width_causality_and_leak_magnitude(self, shipped):
        seq = SHIPPED["context_length"]
        ids = _ids(seq=seq, vocab=SHIPPED["vocab_size"], seed=5)
        bumped = ids.copy()
        bumped[:, -1] = (bumped[:, -1] + 1234) % SHIPPED["vocab_size"]

        masked_base = ops.convert_to_numpy(shipped(ids, training=False))
        masked_moved = ops.convert_to_numpy(shipped(bumped, training=False))
        assert np.abs(masked_base[:, 0] - masked_moved[:, 0]).max() == 0.0

        # The same tower without the mask, at the SETTLED width: the leak is
        # ~0.14, two orders larger than the tiny-scale 5.5e-3 and 32x the toy
        # figure the exploration phase measured. Re-measured here, not inherited.
        inner = shipped.encoder
        leak = np.abs(
            ops.convert_to_numpy(inner(ids, training=False))[:, 0]
            - ops.convert_to_numpy(inner(bumped, training=False))[:, 0]
        ).max()
        assert leak > 1e-2

    def test_shipped_forward_shape(self, shipped):
        seq = SHIPPED["context_length"]
        out = ops.convert_to_numpy(
            shipped(_ids(seq=seq, vocab=SHIPPED["vocab_size"]), training=False)
        )
        assert out.shape == (2, seq, SHIPPED["d_model"])


class TestBuildIsReEntrant:
    """D-136 / D-126: the guard this class was recorded as missing.

    D-126 resolved the symptom in the CALLER (`Sam3Image._build_once`) and left
    the class itself without an `if self.built: return`, so the raise was
    invisible to the package gate. It is now guarded here and executed here.
    """

    def test_a_second_build_is_a_no_op(self, tiny_encoder):
        before = [np.asarray(w) for w in tiny_encoder.weights]
        tiny_encoder.build((1, TINY_SEQ))
        after = [np.asarray(w) for w in tiny_encoder.weights]
        assert len(after) == len(before)
        for old, new in zip(before, after):
            assert np.abs(old - new).max() == 0.0

    def test_the_guard_is_what_prevents_the_raise(self, tiny_encoder):
        """RED proof: clear the flag and the second build DIES as before."""
        assert tiny_encoder.built
        object.__setattr__(tiny_encoder, "built", False)
        with pytest.raises(ValueError, match="already built"):
            tiny_encoder.build((1, TINY_SEQ))

    def test_a_second_build_leaves_the_forward_pass_unchanged(self, tiny_encoder):
        ids = _ids()
        first = _forward(tiny_encoder, ids)
        tiny_encoder.build((1, TINY_SEQ))
        second = _forward(tiny_encoder, ids)
        assert np.abs(first - second).max() == 0.0
