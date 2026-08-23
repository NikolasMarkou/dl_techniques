"""Tests for SAM 3's CLIP text tower wrapper (`models/SAM/SAM3/text_encoder_ve.py`).

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

The second load-bearing guard is ``TestEmbeddingNormDivergence`` (D-142). The
parameter oracle here is derived from the REFERENCE (`_reference_params`) and
the port's count is reached from it by two SIGNED, named divergence terms --
because the previous version of ``_params`` transcribed the port's own extra
``embed_norm`` under a docstring claiming it was written from the structure, so
it agreed with the implementation by construction on exactly the quantity that
diverges.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.SAM.SAM3.text_encoder_ve import Sam3TextEncoder


# R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251.
# Keras `ops/nn.py:907` advises that a softmax over a size-1 axis always returns
# exactly 1.0. Every site in this module feeds that axis a size of 1 ON PURPOSE
# -- single class, single token, single head, single anchor, single cluster,
# minimum sequence length -- so the advisory describes the test's own input, not
# a defect. Suppressed HERE rather than in `pyproject.toml` so an ACCIDENTAL
# size-1 softmax anywhere else still fails under `error::UserWarning`.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:You are using a softmax over axis:UserWarning"),
]

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


#: `TextTransformer`'s `output_dim` DEFAULT. `_create_text_encoder` (pinned
#: clone, `sam3/model_builder.py:500-509`) passes only `d_model`, `width`,
#: `heads` and `layers`, so `text_projection` is `(width, 512)` -- NOT the
#: `(width, d_model)` a reader who assumes `output_dim == d_model` expects.
REFERENCE_OUTPUT_DIM = 512


def _reference_params(cfg: dict) -> int:
    """Closed-form parameter count of the UPSTREAM tower.

    Transcribed term by term from the pinned clone
    (`sam3/model/text_encoder_ve.py` at `96914d24`) and NOT from this port:

    * ``token_embedding`` + ``positional_embedding`` (`:200-201`), with
      **nothing between them and block 0** (`forward`, `:238-245`);
    * ``layers`` x ``ResidualAttentionBlock`` (`:15-88`) = ``ln_1`` +
      ``nn.MultiheadAttention`` (``in_proj`` 3d x d + 3d, ``out_proj`` d x d + d)
      + ``ln_2`` + ``mlp`` (``c_fc``, ``c_proj``); both ``LayerScale``s are
      ``nn.Identity`` because ``ls_init_value`` is ``None``;
    * ``ln_final`` (``use_ln_post=True``);
    * ``text_projection`` ``(width, REFERENCE_OUTPUT_DIM)`` (`:224`) -- allocated
      and checkpointed, never consumed by ``VETextEncoder``;
    * ``VETextEncoder.resizer`` ``Linear(width, d_model)`` (`:290`).

    This is the oracle the PORT is measured AGAINST. `_params` below is derived
    from it by two explicit, signed divergence terms, so neither can drift into
    agreeing with the implementation by construction (D-142).
    """
    width, depth = cfg["width"], cfg["depth"]
    hidden = int(cfg["width"] * cfg["mlp_ratio"])
    embeddings = cfg["vocab_size"] * width + cfg["context_length"] * width
    attention = 4 * (width * width + width)     # in_proj (3) + out_proj, biased
    block_norms = 2 * (2 * width)               # ln_1 + ln_2
    ffn = (width * hidden + hidden) + (hidden * width + width)
    ln_final = 2 * width
    text_projection = width * REFERENCE_OUTPUT_DIM
    resizer = width * cfg["d_model"] + cfg["d_model"]
    return (
        embeddings + depth * (attention + block_norms + ffn)
        + ln_final + text_projection + resizer
    )


def _port_only_embed_norm(cfg: dict) -> int:
    """D-142 divergence 1: the extra embedding norm the port CANNOT remove."""
    return 2 * cfg["width"]                     # scale + offset


def _reference_only_text_projection(cfg: dict) -> int:
    """D-142 divergence 2: the reference-only projection the port omits."""
    return cfg["width"] * REFERENCE_OUTPUT_DIM


def _params(cfg: dict) -> int:
    """Closed-form parameter count for a `Sam3TextEncoder`.

    Derived from `_reference_params` by the two SIGNED divergence terms of
    D-142, not written from the port's own structure -- so a silently dropped
    sub-layer changes the measurement and not the oracle, and a divergence from
    the reference has to be named here before the count can absorb it.
    """
    return (
        _reference_params(cfg)
        + _port_only_embed_norm(cfg)
        - _reference_only_text_projection(cfg)
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

    def test_the_port_carries_an_embedding_norm_the_reference_does_not(
            self, tiny_encoder
    ):
        """D-142: the default this class omitted, and the one that matters most.

        The reference goes token-embed -> +pos -> transformer with NOTHING
        between. The wrapped layer always builds an `embed_norm`. Asserted as
        PRESENT, because it is an accepted divergence -- if a future change
        removes it the port becomes MORE faithful and this arm must be updated
        deliberately rather than silently.
        """
        norm = tiny_encoder.encoder.embed_norm
        assert isinstance(norm, keras.layers.LayerNormalization)
        assert sum(int(np.prod(w.shape)) for w in norm.weights) == (
            _port_only_embed_norm(TINY)
        )

    def test_feed_forward_width_truncates_rather_than_rounds(self):
        # int(32 * 2.55) == 81 while round(32 * 2.55) == 82 -- a probe point
        # where the two rules differ, unlike the settled 4.0.
        encoder = Sam3TextEncoder(**{**TINY, "mlp_ratio": 2.55})
        assert encoder.encoder.intermediate_size == 81


class _Passthrough(keras.layers.Layer):
    """An inert stand-in for `embed_norm`, i.e. the REFERENCE's structure."""

    def call(self, inputs, training=None):
        return inputs


def _forward_with_embed_norm_replaced(encoder, ids, replacement=None):
    """Forward `encoder` with `embed_norm` swapped out, then restore it.

    `object.__setattr__` is required on a BUILT layer: Keras' tracker refuses a
    plain sub-layer assignment ("You cannot add new elements of state ... to a
    layer that is already built"). Nothing is left mutated; the restore is in a
    `finally` and is itself asserted by
    `test_the_probe_leaves_the_encoder_untouched`.
    """
    inner = encoder.encoder
    original = inner.embed_norm
    object.__setattr__(inner, "embed_norm", replacement or _Passthrough())
    try:
        return ops.convert_to_numpy(encoder(ids, training=False))
    finally:
        object.__setattr__(inner, "embed_norm", original)


# DECISION plan-2026-08-22T035419-a11304c8/D-032
# This helper used to take a 7-DIGIT MEASURED CONSTANT (`expected_delta`, pinned
# at `rel=2e-3`: 1.805329 at the tiny width, 5.917600 at the settled width) and
# both of its callers were RED at baseline, reporting 2.012500 and 5.832395.
# Do NOT re-pin a literal here -- re-pinning is what produced this failure mode.
#
# The pins were STALE SAMPLES, not a regression, and the discriminator is
# measured, not argued:
#   * `src/dl_techniques/models/SAM/SAM3/text_encoder_ve.py` has no behavioural
#     commit since the pinning commit `7eea17297` -- only a docstring pass
#     (`4f238a2fe`) and the `models/SAM/` move (`96c6a460b`).
#   * Every other test in this file passes, INCLUDING the reference-derived
#     parameter oracle (`count_params() == _params(SHIPPED) == 353_202_432`),
#     so the port's structure still agrees with the upstream term by term.
#   * `delta` is a SEED-DEPENDENT RANDOM VARIABLE. Measured over 30 model seeds
#     x 3 token-id seeds = 90 draws at the tiny width, `delta / amplitude`
#     spans 0.2799 .. 1.0306 (mean 0.5923, 5th pct 0.4196); raw `delta` at the
#     tiny width spans 1.596 .. 2.719 over 12 model seeds, and the pinned
#     1.805329 and the observed 2.012500 are both ordinary members of it. At the
#     settled width, over model seeds 11/12/13, `delta` reads 5.832395 /
#     5.792219 / 6.223118 -- the pinned 5.917600 sits inside that spread, and
#     the `rel=2e-3` band was 30x tighter than the spread it was pinning.
#   * A `set_random_seed(N)` fixture does not fix the WEIGHTS, only the draw
#     ORDER; any upstream change to weight-creation order re-deals the tower at
#     the same seed while changing nothing semantically.
# The old amplitude arm (`delta > 0.4 * amplitude`) was fragile for the same
# reason: 0.4 sits ABOVE the measured 5th percentile 0.4196's neighbourhood and
# BELOW the minimum only by luck (min 0.2799), so it too would fire on an
# innocent draw. The floor below is derived from that 90-draw population.
def _assert_carries_the_extra_embedding_norm(encoder, ids, min_ratio):
    """Assert this tower normalizes its embeddings where the reference does not.

    Two arms, both scale-free. (1) The output must MOVE AT ALL when `embed_norm`
    is replaced by a passthrough -- the reference's structure; a port that
    matched the reference reports EXACTLY 0.0 and fails this arm (proven by
    `test_this_guard_can_see_the_divergence_close`). (2) The move must be at
    least `min_ratio` of the output's own amplitude, so "it diverges" cannot
    degrade into "it diverges by a rounding error".

    :param min_ratio: floor on ``delta / amplitude``, derived per call site from
        the measured population -- never a hand-picked round number.
    """
    base = _forward(encoder, ids)
    reference_order = _forward_with_embed_norm_replaced(encoder, ids)
    delta = float(np.abs(base - reference_order).max())
    amplitude = float(np.abs(base).max())
    assert delta > 0.0, (
        "removing the extra embedding normalization changed NOTHING "
        f"(delta = {delta!r}); this port has stopped diverging from the "
        "reference at all, which is D-142's remedy landing silently"
    )
    assert delta > min_ratio * amplitude, (
        f"delta {delta:.6f} is only {delta / amplitude:.4f} of the output "
        f"amplitude {amplitude:.6f}, under the derived floor {min_ratio}"
    )


class TestEmbeddingNormDivergence:
    """D-142: the port normalizes its embeddings; the reference does not.

    `sam3/model/text_encoder_ve.py:238-245` at the pinned SHA is
    `token_embedding -> + positional_embedding -> transformer`, with nothing
    between. `layers/transformers/text_encoder.py:775` inserts a LayerNorm and
    offers no way to turn it off. MEASURED, not asserted from a docstring.
    """

    def test_the_reference_order_output_differs_at_the_tiny_width(
            self, tiny_encoder
    ):
        # min_ratio=0.15: the tiny-width `delta / amplitude` population is
        # 0.2799 .. 1.0306 over 90 draws (30 model seeds x 3 id seeds), so this
        # floor sits 1.87x below the worst observed draw and still convicts the
        # only thing it can be wrong about -- a reference-shaped port, which
        # reports exactly 0.0.
        _assert_carries_the_extra_embedding_norm(tiny_encoder, _ids(), 0.15)

    def test_the_substitution_mechanism_alone_changes_nothing(
            self, tiny_encoder
    ):
        """Control: an EQUIVALENT norm swapped in gives exactly 0.0.

        Without this, arm 1 above could be measuring the swap rather than the
        normalization. A freshly built `LayerNormalization` at the same epsilon
        is bit-identical to the shipped one at initialization (gamma 1, beta 0),
        so the harness itself must contribute nothing.
        """
        ids = _ids()
        equivalent = keras.layers.LayerNormalization(
            epsilon=tiny_encoder.encoder.embed_norm.epsilon
        )
        moved = _forward_with_embed_norm_replaced(
            tiny_encoder, ids, replacement=equivalent
        )
        assert np.abs(_forward(tiny_encoder, ids) - moved).max() == 0.0

    def test_the_probe_leaves_the_encoder_untouched(self, tiny_encoder):
        ids = _ids()
        before = _forward(tiny_encoder, ids)
        _forward_with_embed_norm_replaced(tiny_encoder, ids)
        assert isinstance(
            tiny_encoder.encoder.embed_norm, keras.layers.LayerNormalization
        )
        assert np.abs(_forward(tiny_encoder, ids) - before).max() == 0.0

    def test_this_guard_can_see_the_divergence_close(self):
        """RED proof: a REFERENCE-shaped port is required to FAIL the guard.

        The substitution is done BEFORE `build`, which is the one route that
        would actually remove the norm from this port (D-142 records it as the
        remedy for the weight-transfer phase and declines to apply it now). The
        resulting layer has 64 fewer parameters, reports a delta of exactly 0.0,
        and `_assert_carries_the_extra_embedding_norm` raises on it -- so the
        guard is not a fact about `LayerNormalization`, it is a fact about this
        encoder.
        """
        keras.utils.set_random_seed(7)
        reference_shaped = Sam3TextEncoder(**TINY)
        reference_shaped.encoder.embed_norm = _Passthrough()
        reference_shaped.build((None, TINY_SEQ))
        assert reference_shaped.count_params() == (
            _params(TINY) - _port_only_embed_norm(TINY)
        )
        # The message pinned here is the FIRST arm's ("delta > 0.0"), which is
        # the one a reference-shaped port trips; the same floor the live call
        # site uses is passed so this proof exercises the real configuration.
        with pytest.raises(AssertionError, match="changed NOTHING"):
            _assert_carries_the_extra_embedding_norm(
                reference_shaped, _ids(), 0.15
            )


class TestParameterAudit:

    def test_tiny_count_matches_the_closed_form_exactly(self, tiny_encoder):
        assert tiny_encoder.count_params() == _params(TINY)

    def test_the_port_and_the_reference_differ_by_two_named_terms(self):
        """D-142: the settled tower against the UPSTREAM closed form.

        `text_projection` is `(1024, 512)` because `_create_text_encoder` never
        passes `output_dim` -- 524,288 parameters, twice the 1024x256 that
        assuming `output_dim == d_model` would give.
        """
        assert _port_only_embed_norm(SHIPPED) == 2_048
        assert _reference_only_text_projection(SHIPPED) == 524_288
        assert _reference_params(SHIPPED) == 353_724_672
        assert _params(SHIPPED) == 353_202_432
        assert _reference_params(SHIPPED) - _params(SHIPPED) == 522_240

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

    def test_the_extra_embedding_norm_moves_the_settled_output(self, shipped):
        """D-142 asserted AT THE SETTLED WIDTH, where it is worst.

        The divergence EXCEEDS THE SIGNAL here: `delta / amplitude` measures
        1.1129 / 1.1791 / 1.1696 over model seeds 11 / 12 / 13 (2026-08-22),
        against 0.2799 .. 1.0306 at this file's tiny width. The tiny figure must
        not be quoted as the magnitude of this defect.

        `min_ratio=1.0` is therefore the claim, not a tolerance: it says the
        move is larger than the output itself. Its margin over the worst of the
        three settled draws is 11 %. The raw deltas (5.832395 / 5.792219 /
        6.223118, against the 5.917600 D-142 once pinned at `rel=2e-3`) are
        recorded here as a MEASUREMENT and deliberately not asserted -- see
        D-032 at `_assert_carries_the_extra_embedding_norm`.
        """
        seq = SHIPPED["context_length"]
        ids = _ids(seq=seq, vocab=SHIPPED["vocab_size"], seed=0)
        _assert_carries_the_extra_embedding_norm(shipped, ids, 1.0)

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
