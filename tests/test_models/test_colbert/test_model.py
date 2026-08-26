"""Guards for :class:`~dl_techniques.models.language.colbert.model.ColBERT`.

Everything here runs on a deliberately TINY geometry (1 backbone layer, hidden
16, dim 8, query 6, doc 10) so the whole module stays fast on CPU. The claims
being pinned are structural and value-level, none of them scale-dependent.

RED-proof record (guide v3 §14; injections applied to ``model.py``, restored
from a ``cp`` backup and verified with ``diff -q`` -- never ``git stash`` and
never ``git checkout --``):

======================================================  ========================================================
Injection                                               Named assertion that fired
======================================================  ========================================================
(a) delete ``materialize_sublayers(self, input_shape)``  ``test_the_explicit_build_matches_the_lazy_build``
    from ``ColBERT.build``                               -- "explicit build materialized 0 weights"
(b) give the document path its OWN second                ``test_the_query_and_document_share_one_projection``
    ``ColBERTProjection`` instead of the shared one      -- "the document path must reuse the query projection"
(c) drop the skiplist mask on the document path          ``test_a_skiplisted_document_position_is_zeroed``
    (``_participation`` returns ``attention_mask``)      -- "a skiplisted position must project to exactly zero"
(d) re-collapse the two document masks into one          ``test_a_kept_position_is_untouched_by_a_skiplist_elsewhere``
    (``_encode`` passes ``participation_mask`` to        -- "a kept document position moved because a DIFFERENT
    ``self.encoder`` as its ``attention_mask``)             position was skiplisted"
(e) neuter the backbone's padding mask                   ``test_a_padded_document_position_cannot_influence_a_real_one``
    (``_encode`` passes                                  -- "a real document position was UNCHANGED when the
    ``keras.ops.ones_like(attention_mask)`` to              trailing positions were marked as padding"
    ``self.encoder``)
(f) build the backbone with ``hidden_act="gelu"``        ``test_the_backbone_runs_the_tanh_gelu_approximation``
    (the exact erf form, not the tanh                    -- "the ColBERT backbone's activation drifted away from
    approximation README §9.4 documents)                    the tanh GELU approximation documented in README §9.4"
(g) narrow the D-007 reduction dtype in                  ``test_a_fully_masked_document_scores_the_exact_sentinel_under_xla``
    ``components.py`` (``MaxSimScorer._reduction_dtype`` -- "the XLA-compiled score is not float32 under
    returns the incoming dtype unchanged, dropping          mixed_float16: got <dtype: 'float16'>"
    the float32 promotion)
(h) rewrite the ``base`` row to hidden 1024 / layers    ``test_the_base_and_large_rows_keep_their_reference_backbone_geometry``
    10 / heads 16 / intermediate 4096 -- internally      -- "the 'base' row no longer matches its reference
    consistent, but no longer BERT-Base                     backbone: hidden_size=1024, but BERT-Base/ColBERT
                                                           declares hidden_size=768"
(i) ``DEFAULT_MAXSIM_MASK_VALUE = -2e4`` in             ``test_a_fully_masked_document_scores_the_exact_sentinel_under_xla``
    ``components.py``                                    -- "the MaxSim mask sentinel moved to -20000.0; this
                                                           guard's hand-derived reference is 32 * -1e4"
======================================================  ========================================================

Each injection was verified to redden its OWN named assertion -- the message
quoted above is the one pytest printed, not a paraphrase -- and each was
restored and re-verified green (66 passed / 66 collected for the directory).
Injection (a) additionally reddens six further tests, which is expected: a model
with zero materialized weights fails every claim that needs weights. Injections
(b) and (c) are narrow: (b) reddens two tests, (c) exactly one. Injection (d),
added with the D-029 mask split, reddens exactly one -- and, crucially, reddens
NEITHER of the two pre-existing skiplist guards, which is the whole reason it
had to be written. Injection (e) is D-029's OTHER half: before
``test_a_padded_document_position_cannot_influence_a_real_one`` existed it
reddened NOTHING at all in this directory, so a refactor dropping the padding
mask on the way to the backbone would have shipped green. Injections (e) and
(f) each reddened EXACTLY ONE test (1 failed / 90 passed for the directory), at
the assertion quoted above, and both were restored from the ``cp`` backup and
re-verified byte-identical with ``diff -q``.

Injection (g) is the only one applied to ``components.py`` rather than to
``model.py``. It is invisible under the default ``float32`` policy -- the
promotion it removes is a no-op there -- which is why the guard runs under
``mixed_float16``: with the promotion gone the compiled score came back
``float16`` and valued ``-inf``, so BOTH the dtype assertion quoted above and
the ``query_maxlen * mask_value`` value assertion below it are live. Restored
from the ``cp`` backup and re-verified byte-identical with ``diff -q``.

Injections (h) and (i) were added by the iter-1 completion fixes. (h) is the
adversarial reviewer's own injection: before the external
``REFERENCE_BACKBONE_GEOMETRY`` table existed it left BOTH parametrizations
GREEN, because every other arm in that test compares the built model against
the same row the injection edited. (i) is the same defect class one level down
-- the XLA guard read ``query_maxlen * mask_value`` off the object under test,
so moving the sentinel moved the expectation with it; with the two operands now
pinned it fails by name. Both restored by ``cp`` + ``diff -q``.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.language.colbert.components import ColBERTProjection
from dl_techniques.models.language.colbert.model import (
    DOC_ATTENTION_MASK_KEY,
    DOC_INPUT_IDS_KEY,
    DOC_SKIPLIST_MASK_KEY,
    QUERY_ATTENTION_MASK_KEY,
    QUERY_INPUT_IDS_KEY,
    ColBERT,
    create_colbert,
    create_colbert_v1,
    create_colbert_v2,
)

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing

# ---------------------------------------------------------------------
# Tiny geometry shared by every test in this module.
# ---------------------------------------------------------------------

TINY = dict(
    vocab_size=64,
    hidden_size=16,
    num_layers=1,
    num_heads=2,
    intermediate_size=32,
    dim=8,
    query_maxlen=6,
    doc_maxlen=10,
    max_position_embeddings=16,
)

BATCH = 2
QUERY_LEN = TINY["query_maxlen"]
DOC_LEN = TINY["doc_maxlen"]

BUILD_SHAPES = {
    QUERY_INPUT_IDS_KEY: (None, QUERY_LEN),
    DOC_INPUT_IDS_KEY: (None, DOC_LEN),
}


def make_model(**overrides):
    """Build a tiny ColBERT, unbuilt, with ``TINY`` overridden by ``overrides``."""
    config = dict(TINY)
    config.update(overrides)
    return ColBERT(**config)


def make_inputs(seed: int = 0):
    """Deterministic input batch. Called repeatedly; must return equal values."""
    rng = np.random.default_rng(seed)
    return {
        QUERY_INPUT_IDS_KEY: rng.integers(
            0, TINY["vocab_size"], (BATCH, QUERY_LEN)
        ).astype("int32"),
        QUERY_ATTENTION_MASK_KEY: np.ones((BATCH, QUERY_LEN), dtype="int32"),
        DOC_INPUT_IDS_KEY: rng.integers(
            0, TINY["vocab_size"], (BATCH, DOC_LEN)
        ).astype("int32"),
        DOC_ATTENTION_MASK_KEY: np.ones((BATCH, DOC_LEN), dtype="int32"),
        DOC_SKIPLIST_MASK_KEY: np.ones((BATCH, DOC_LEN), dtype="int32"),
    }


def weight_paths(model):
    """Weight paths with the model's own uniquified root segment stripped.

    Keras appends ``_1``, ``_2``, ... to the auto-generated name of the SECOND
    and later instances of a class in one session, so two models built from the
    same config carry ``col_bert/...`` and ``col_bert_1/...``. That difference
    is the instance counter, not the architecture. Only the leading segment is
    dropped -- the whole sub-layer tree below it is compared verbatim, so a
    missing encoder layer or a duplicated projection is still visible.
    """
    return {w.path.split("/", 1)[1] for w in model.weights}


# ---------------------------------------------------------------------
# 1. Explicit build vs lazy build
# ---------------------------------------------------------------------


def test_the_explicit_build_matches_the_lazy_build():
    """``build()`` must materialize exactly what a forward call materializes."""
    explicit = make_model()
    explicit.build(BUILD_SHAPES)

    lazy = make_model()
    lazy(make_inputs())

    assert len(explicit.weights) > 0, (
        "explicit build materialized 0 weights: ColBERT.build left the "
        "sub-layer tree unmaterialized while marking the model built"
    )
    assert explicit.count_params() > 0, (
        "explicit build reported count_params() == 0"
    )
    assert len(explicit.weights) == len(lazy.weights), (
        f"explicit build produced {len(explicit.weights)} weights against the "
        f"lazy build's {len(lazy.weights)}"
    )
    assert weight_paths(explicit) == weight_paths(lazy), (
        "explicit and lazy builds disagree on the weight PATH set; only in "
        f"explicit: {sorted(weight_paths(explicit) - weight_paths(lazy))}; only "
        f"in lazy: {sorted(weight_paths(lazy) - weight_paths(explicit))}"
    )
    assert explicit.count_params() == lazy.count_params(), (
        f"explicit count_params()={explicit.count_params()} against lazy "
        f"{lazy.count_params()}"
    )


def test_the_lazy_build_costs_nothing():
    """Shared oracle: a lazily-built model loses nothing across save/load."""
    assert_lazy_build_costs_nothing(
        build=make_model,
        make_inputs=make_inputs,
        input_shape=BUILD_SHAPES,
    )


# ---------------------------------------------------------------------
# 2. Value-level .keras round trip, TWICE
# ---------------------------------------------------------------------


def test_two_keras_round_trips_preserve_every_weight_value_and_the_output():
    """Save -> load -> save -> load, comparing values at ``atol=1e-6, rtol=0``.

    Done twice on purpose: a save-side check cannot see a load-side loss, and a
    model that reconstructs a sub-layer differently on reload only diverges on
    the SECOND write.
    """
    inputs = make_inputs()
    model = make_model()
    reference_output = model(inputs, training=False)

    reference_values = [
        np.asarray(keras.ops.convert_to_numpy(w)) for w in model.weights
    ]
    reference_paths = [w.path.split("/", 1)[1] for w in model.weights]

    current = model
    with tempfile.TemporaryDirectory() as directory:
        for cycle in (1, 2):
            path = os.path.join(directory, f"colbert_{cycle}.keras")
            current.save(path)
            current = keras.models.load_model(path, compile=False)

            restored_paths = [w.path.split("/", 1)[1] for w in current.weights]
            assert restored_paths == reference_paths, (
                f"cycle {cycle}: the reloaded weight path list changed; only "
                f"after reload: {sorted(set(restored_paths) - set(reference_paths))}"
            )

            for name, before, weight in zip(
                reference_paths, reference_values, current.weights
            ):
                after = np.asarray(keras.ops.convert_to_numpy(weight))
                np.testing.assert_allclose(
                    after,
                    before,
                    atol=1e-6,
                    rtol=0,
                    err_msg=(
                        f"cycle {cycle}: weight {name} changed value across the "
                        "round trip"
                    ),
                )

            restored_output = current(inputs, training=False)
            assert set(restored_output) == set(reference_output), (
                f"cycle {cycle}: the output key set changed across the round trip"
            )
            for key in reference_output:
                np.testing.assert_allclose(
                    np.asarray(keras.ops.convert_to_numpy(restored_output[key])),
                    np.asarray(keras.ops.convert_to_numpy(reference_output[key])),
                    atol=1e-6,
                    rtol=0,
                    err_msg=(
                        f"cycle {cycle}: output '{key}' changed across the "
                        "round trip"
                    ),
                )


# ---------------------------------------------------------------------
# 3. The shared projection
# ---------------------------------------------------------------------


def test_the_query_and_document_share_one_projection(monkeypatch):
    """The two towers must traverse the SAME projection object, not a twin.

    Asserted with ``is``, and backed by a weight-count claim: a second
    projection would add exactly one more kernel to the model. Comparing
    configurations would pass on two independently-initialized twins, which is
    the defect this guards.
    """
    model = make_model()
    model.build(BUILD_SHAPES)

    projection_kernels = [
        w.path for w in model.weights if "projection" in w.path
    ]
    assert len(projection_kernels) == 1, (
        "the document path must reuse the query projection: found "
        f"{len(projection_kernels)} projection weights ({projection_kernels}), "
        "so the two towers do not share one instance"
    )

    seen = []
    original_call = ColBERTProjection.call

    def recording_call(layer, *args, **kwargs):
        seen.append(id(layer))
        return original_call(layer, *args, **kwargs)

    monkeypatch.setattr(ColBERTProjection, "call", recording_call)
    model(make_inputs())

    assert len(seen) >= 2, (
        "the document path must reuse the query projection: the projection was "
        f"invoked {len(seen)} time(s) in one forward pass, expected at least 2 "
        "(query tower + document tower)"
    )
    assert set(seen) == {id(model.projection)}, (
        "the document path must reuse the query projection: the forward pass "
        f"invoked {len(set(seen))} distinct ColBERTProjection instances"
    )


def test_the_encoder_is_also_shared_by_both_towers():
    """One BERT, not two: the model must hold a single backbone."""
    model = make_model()
    model.build(BUILD_SHAPES)
    encoder_roots = {
        w.path.split("/", 1)[1].split("/", 1)[0] for w in model.weights
    }
    assert encoder_roots == {"encoder", "projection"}, (
        "expected exactly one 'encoder' subtree and one 'projection' subtree, "
        f"found top-level weight groups {sorted(encoder_roots)}"
    )


# ---------------------------------------------------------------------
# 4. from_variant contracts
# ---------------------------------------------------------------------


def test_from_variant_rejects_an_unknown_variant_and_lists_the_real_keys():
    with pytest.raises(ValueError) as excinfo:
        ColBERT.from_variant("nope")
    message = str(excinfo.value)
    assert "nope" in message
    for key in ColBERT.MODEL_VARIANTS:
        assert key in message, (
            f"the ValueError must list the available variants; '{key}' is "
            f"missing from: {message}"
        )


def test_from_variant_refuses_pretrained_true():
    with pytest.raises(NotImplementedError) as excinfo:
        ColBERT.from_variant("tiny", pretrained=True)
    message = str(excinfo.value)
    assert "tiny" in message, "the error must name the requested variant"
    assert "pretrained=False" in message, (
        "the error must name the supported route (pretrained=False plus "
        f"load_weights); got: {message}"
    )
    assert "load_weights" in message


@pytest.mark.parametrize("variant", sorted(ColBERT.MODEL_VARIANTS))
def test_every_variant_row_carries_the_reference_colbert_defaults(variant):
    """The ColBERT-side numbers are Class A and identical in every row."""
    row = ColBERT.MODEL_VARIANTS[variant]
    assert row["dim"] == 128
    assert row["query_maxlen"] == 32
    assert row["doc_maxlen"] == 220
    assert row["description"]


# ---------------------------------------------------------------------
# 5. v1 and v2 build the same architecture
# ---------------------------------------------------------------------


def test_the_v1_and_v2_factories_build_the_same_architecture():
    """The honest encoding of the shared-encoder ruling.

    v1 and v2 are the same network -- the reference has no v1-only code path --
    so the two factories must produce identical weight-path sets and identical
    parameter counts. If a future change makes them diverge structurally, this
    test is the place that says so out loud.
    """
    built = {}
    for name, factory in (
        ("v1", create_colbert_v1),
        ("v2", create_colbert_v2),
        ("neutral", create_colbert),
    ):
        model = factory("tiny", **TINY)
        model.build(BUILD_SHAPES)
        built[name] = model

    reference = weight_paths(built["v1"])
    assert reference, "the v1 factory produced a model with no weights"
    for name in ("v2", "neutral"):
        assert weight_paths(built[name]) == reference, (
            f"create_colbert_{name} produced a different weight-path set than "
            "create_colbert_v1; only in v1: "
            f"{sorted(reference - weight_paths(built[name]))}; only in {name}: "
            f"{sorted(weight_paths(built[name]) - reference)}"
        )
        assert built[name].count_params() == built["v1"].count_params()


# ---------------------------------------------------------------------
# 6. The punctuation skiplist reaches the document embeddings
# ---------------------------------------------------------------------


def test_a_skiplisted_document_position_is_zeroed():
    """A skiplisted position must project to exactly zero and lose the MaxSim.

    Two claims, because either alone is satisfiable by accident: the masked
    position's embedding is exactly the zero vector (so it cannot contribute a
    positive similarity), AND giving that position the query's own embedding --
    the strongest possible match -- does not raise the score.
    """
    # SEEDED, and the seed is load-bearing. After step 3.1 split the two document
    # masks (D-029) the skiplist no longer reaches the backbone, so removing a
    # position changes the score ONLY when that position was winning some query
    # term's max. On an unseeded draw it often is not, and the liveness arm below
    # then fails a CORRECT implementation -- measured 2 of 5 identical runs before
    # this pin. `plans/LESSONS.md`: pin the draw, never widen the bar.
    keras.utils.set_random_seed(3)
    model = make_model()
    inputs = make_inputs()

    skiplist = np.ones((BATCH, DOC_LEN), dtype="int32")
    skiplist[:, 3] = 0
    masked_inputs = dict(inputs)
    masked_inputs[DOC_SKIPLIST_MASK_KEY] = skiplist

    unmasked = model(inputs, training=False)
    masked = model(masked_inputs, training=False)

    doc_embeddings = np.asarray(
        keras.ops.convert_to_numpy(masked["doc_embeddings"])
    )
    np.testing.assert_allclose(
        doc_embeddings[:, 3, :],
        np.zeros((BATCH, TINY["dim"])),
        atol=0.0,
        rtol=0,
        err_msg=(
            "a skiplisted position must project to exactly zero; the skiplist "
            "mask is not reaching the projection's mask multiply"
        ),
    )

    unmasked_position = np.asarray(
        keras.ops.convert_to_numpy(unmasked["doc_embeddings"])
    )[:, 3, :]
    assert np.max(np.abs(unmasked_position)) > 0.0, (
        "control failed: position 3 is the zero vector even WITHOUT the "
        "skiplist, so the zero above proves nothing"
    )

    # NOT a monotonicity claim, and NOT an unconditional liveness claim.
    #
    # [CORRECTED 2026-08-25, step 3.2] The comment that stood here described the
    # PRE-SPLIT model: it said masking a position also moves every other
    # position's contextual representation "because the participation mask is
    # also the backbone's attention mask". Step 3.1 (D-029) ended exactly that,
    # so the sentence became false in the commit that fixed the defect it
    # described. After the split, removing a document position changes the score
    # if and only if that position was winning some query term's max -- the
    # backbone's own output is now untouched by the skiplist.
    #
    # The liveness arm is therefore guarded by its own PRECONDITION rather than
    # asserted unconditionally: at the pinned seed, position 3 wins at least one
    # query term, so the score MUST move. Without the precondition the arm is a
    # coin flip on the draw (measured 2 failures in 5 identical runs).
    masked_score = np.asarray(keras.ops.convert_to_numpy(masked["score"]))
    unmasked_score = np.asarray(keras.ops.convert_to_numpy(unmasked["score"]))
    assert np.all(np.isfinite(masked_score))

    q = np.asarray(keras.ops.convert_to_numpy(unmasked["query_embeddings"]))
    d = np.asarray(keras.ops.convert_to_numpy(unmasked["doc_embeddings"]))
    winners = np.argmax(np.einsum("bqd,bsd->bqs", q, d), axis=-1)
    assert np.any(winners == 3), (
        "PRECONDITION failed at the pinned seed: position 3 wins no query "
        "term's max, so the liveness arm below would be vacuous. Re-pin the "
        f"seed rather than deleting the arm (winners={winners.tolist()})"
    )
    assert np.max(np.abs(masked_score - unmasked_score)) > 0.0, (
        "position 3 wins at least one query term's max, so skiplisting it MUST "
        "change the score; it did not, so the skiplist is inert on the MaxSim "
        f"path (masked={masked_score}, unmasked={unmasked_score})"
    )


def test_a_skiplisted_position_cannot_win_the_max_even_when_it_is_the_best_match():
    """The adversarial arm: plant a perfect match at a skiplisted position."""
    model = make_model()
    model.build(BUILD_SHAPES)

    rng = np.random.default_rng(7)
    query = rng.normal(size=(1, QUERY_LEN, TINY["dim"])).astype("float32")
    query /= np.linalg.norm(query, axis=-1, keepdims=True)

    docs = np.zeros((1, DOC_LEN, TINY["dim"]), dtype="float32")
    docs[0, 0, 0] = 1.0
    # Position 5 is a copy of every query term's ideal partner AND is masked.
    docs[0, 5, :] = query[0, 0, :] * 50.0

    doc_mask = np.ones((1, DOC_LEN), dtype="int32")
    doc_mask[0, 5] = 0

    score_masked = float(
        keras.ops.convert_to_numpy(
            model.score(query, docs, doc_mask=doc_mask)
        )[0]
    )
    score_unmasked = float(
        keras.ops.convert_to_numpy(
            model.score(query, docs, doc_mask=np.ones((1, DOC_LEN), dtype="int32"))
        )[0]
    )

    assert np.isfinite(score_masked)
    assert score_masked < score_unmasked, (
        "the masked position won the max anyway: masked score "
        f"{score_masked} is not below the unmasked {score_unmasked}"
    )


def test_a_kept_position_is_untouched_by_a_skiplist_elsewhere():
    """The skiplist must NOT reach the backbone's attention mask (D-029).

    This is the ordering guard the two tests above cannot be. Both of them
    assert only that the FILTERED position is zero, and both pass identically
    whether the skiplist is fed to ``self.encoder`` as its attention mask (the
    collapsed ordering this port shipped at step 3) or applied only after
    encoding (the reference ordering). The axis that separates the two is a
    KEPT, non-punctuation position: under the collapsed ordering, skiplisting
    positions 3 and 7 hides them from every other token's self-attention and
    therefore MOVES the contextual embedding of positions 0-2, 4-6, 8-9.

    Property, not sample. MEASURED 2026-08-25 on this geometry with
    ``num_layers=2`` over 40 seeded inits (``keras.utils.set_random_seed(0..39)``):
    the collapsed ordering gives a kept-position ``max |delta|`` of min
    0.0003569424 / median 0.0009457758 / max 0.0024135113 -- never zero -- while
    the split ordering gives exactly ``0.0`` at all 40. This test sets no seed
    on purpose: its assertion is exact equality against a bit-identical
    baseline, which no initialization can perturb. Do not loosen it to
    ``atol > 0``; that would make it seed-sensitive over a ~7x magnitude spread.
    The delta grows with depth and with a trained backbone.
    """
    model = make_model(num_layers=2)
    inputs = make_inputs()

    doc_inputs = {
        "input_ids": inputs[DOC_INPUT_IDS_KEY],
        "attention_mask": inputs[DOC_ATTENTION_MASK_KEY],
    }

    skiplist = np.ones((BATCH, DOC_LEN), dtype="int32")
    skiplist[:, 3] = 0
    skiplist[:, 7] = 0
    skiplisted_inputs = dict(doc_inputs)
    skiplisted_inputs["skiplist_mask"] = skiplist

    without = np.asarray(
        keras.ops.convert_to_numpy(
            model.encode_document(doc_inputs, training=False)
        )
    )
    with_skiplist = np.asarray(
        keras.ops.convert_to_numpy(
            model.encode_document(skiplisted_inputs, training=False)
        )
    )

    kept = [i for i in range(DOC_LEN) if i not in (3, 7)]

    # Control: without the skiplist those positions are alive, so the zero
    # below is a consequence of the mask and not of a dead forward pass.
    assert np.max(np.abs(without[:, [3, 7], :])) > 0.0, (
        "control failed: positions 3 and 7 are already zero WITHOUT the "
        "skiplist, so this test would pass on a dead model"
    )
    assert np.max(np.abs(without[:, kept, :])) > 0.0, (
        "control failed: the kept positions are all zero, so an equality "
        "against them proves nothing"
    )

    np.testing.assert_allclose(
        with_skiplist[:, kept, :],
        without[:, kept, :],
        atol=0.0,
        rtol=0,
        err_msg=(
            "a kept document position moved because a DIFFERENT position was "
            "skiplisted: the punctuation skiplist is reaching the backbone's "
            "attention mask, which the reference never does -- it passes the "
            "plain padding mask to BERT and applies the skiplist only to the "
            "projected embeddings. See D-029."
        ),
    )

    # And the filtered positions are still exactly zero, i.e. the split did not
    # silently disable the skiplist it was meant to relocate.
    np.testing.assert_allclose(
        with_skiplist[:, [3, 7], :],
        np.zeros((BATCH, 2, TINY["dim"])),
        atol=0.0,
        rtol=0,
        err_msg=(
            "the skiplist stopped reaching the projection's mask multiply: a "
            "skiplisted position must still project to exactly zero"
        ),
    )


def test_a_padded_document_position_cannot_influence_a_real_one():
    """The OTHER half of D-029: the padding mask MUST reach the backbone.

    ``test_a_kept_position_is_untouched_by_a_skiplist_elsewhere`` pins one
    direction (the skiplist must NOT reach ``self.encoder``) and is blind to
    the other: it passes unchanged on a model whose backbone receives
    ``ones_like(attention_mask)``, i.e. one that attends to padding as if it
    were content. MEASURED as the RED-proof for this test -- with that
    injection at the ``self.encoder`` call the whole colbert directory stayed
    green. This test is the missing control: zeroing the TRAILING positions of
    the padding mask must MOVE the contextual embedding of the real prefix
    positions, because they can no longer attend to the padding.

    Property, not sample: the assertion is strict non-equality. The magnitude
    is a property of one random initialization -- MEASURED over 40 seeded
    inits on this geometry (``keras.utils.set_random_seed``, ``num_layers=2``,
    8 real positions of 10, trailing 2 padded): min 0.0003021657, median
    0.0009489805, max 0.0022316426, and never once exactly 0.0.
    """
    model = make_model(num_layers=2)
    input_ids = make_inputs()[DOC_INPUT_IDS_KEY]

    real = list(range(DOC_LEN - 2))
    full = np.ones((BATCH, DOC_LEN), dtype="int32")
    trimmed = full.copy()
    trimmed[:, DOC_LEN - 2:] = 0

    def encode(attention_mask):
        return np.asarray(
            keras.ops.convert_to_numpy(
                model.encode_document(
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    training=False,
                )
            )
        )

    attended = encode(full)
    masked = encode(trimmed)

    # Control 1: the compared positions are alive under BOTH masks, so a
    # non-equality between them is not a comparison of two zero blocks.
    assert np.max(np.abs(attended[:, real, :])) > 0.0, (
        "control failed: the real positions are all zero with an all-ones "
        "mask, so this test would pass on a dead model"
    )
    assert np.max(np.abs(masked[:, real, :])) > 0.0, (
        "control failed: the real positions are all zero once the trailing "
        "positions are padded, so the forward pass died rather than changed"
    )
    # Control 2: the padded positions themselves ARE zeroed, i.e. the padding
    # mask reached the projection's mask multiply as well.
    assert np.max(np.abs(masked[:, DOC_LEN - 2:, :])) == 0.0, (
        "a padded position did not project to exactly zero: the padding mask "
        "is not reaching the projection's mask multiply"
    )

    delta = float(np.max(np.abs(masked[:, real, :] - attended[:, real, :])))
    assert delta > 0.0, (
        "a real document position was UNCHANGED when the trailing positions "
        "were marked as padding: the padding attention_mask is not reaching "
        "the BERT backbone, so the model attends to padding as if it were "
        "content. This is the second half of the D-029 mask split -- the "
        "skiplist must not reach self.encoder, and the padding mask must. "
        f"max |delta| over the {len(real)} real positions was {delta}."
    )


def test_the_backbone_runs_the_tanh_gelu_approximation():
    """Pins README §9.4, which was previously a documentation-only deviation.

    This library's ``BERT`` defaults to ``gelu_tanh`` (the tanh approximation)
    rather than the exact ``erf`` form, and ``ColBERT`` keeps that default.
    Nothing in the suite could see a silent flip to ``"gelu"``; §9's claim
    about its own deviation list is only worth what this assertion is worth.
    """
    model = make_model()

    assert model.encoder.hidden_act == "gelu_tanh", (
        "the ColBERT backbone's activation drifted away from the tanh GELU "
        f"approximation documented in README §9.4: got "
        f"{model.encoder.hidden_act!r}"
    )
    assert model.encoder.get_config()["hidden_act"] == "gelu_tanh", (
        "the backbone's serialized config disagrees with its live attribute "
        "about the activation, so a reloaded ColBERT would run a different "
        f"non-linearity: {model.encoder.get_config()['hidden_act']!r}"
    )


# ---------------------------------------------------------------------
# 7. Gradient flow, AFTER one real optimizer step
# ---------------------------------------------------------------------


def test_every_trainable_weight_receives_a_gradient_after_one_optimizer_step():
    """Adopted after a real step, never at init.

    A weight can look dead at initialization for reasons that have nothing to do
    with wiring (an all-zero bias multiplying into a zero activation), so the
    oracle is applied to a model that has already taken one gradient step.
    """
    model = make_model()
    inputs = make_inputs()

    def loss_fn(outputs):
        return keras.ops.mean(keras.ops.square(outputs["score"]))

    optimizer = keras.optimizers.SGD(learning_rate=0.1)
    import tensorflow as tf

    with tf.GradientTape() as tape:
        loss = loss_fn(model(inputs, training=True))
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(
        [
            (g, w)
            for g, w in zip(gradients, model.trainable_weights)
            if g is not None
        ]
    )

    assert_gradients_reach_every_trainable_weight(
        model,
        inputs,
        loss_fn=loss_fn,
    )


# ---------------------------------------------------------------------
# 8. get_config round trip
# ---------------------------------------------------------------------


def test_get_config_round_trip_reconstructs_an_equivalent_model():
    model = make_model()
    config = model.get_config()

    for key, value in TINY.items():
        assert config[key] == value, (
            f"get_config lost constructor argument '{key}': expected {value}, "
            f"got {config.get(key)}"
        )
    assert "mask_value" in config

    rebuilt = ColBERT.from_config(config)
    rebuilt.build(BUILD_SHAPES)
    model.build(BUILD_SHAPES)

    assert weight_paths(rebuilt) == weight_paths(model)
    assert rebuilt.count_params() == model.count_params()
    assert rebuilt.get_config()["dim"] == config["dim"]


def test_mask_punctuation_is_rejected_loudly_rather_than_silently_inert():
    """``mask_punctuation=`` must RAISE, not serialize an intent nothing honors.

    Until 2026-08-25 ``ColBERT`` took a ``mask_punctuation`` constructor
    argument, stored it and emitted it from ``get_config()`` -- and nothing
    anywhere read it. The model applies whatever ``doc_skiplist_mask`` it is
    handed, so ``create_colbert_v2(..., mask_punctuation=False)`` returned a
    model that still applied the punctuation mask in full while reporting the
    opposite. The live flag is ``ColBERTTokenizer.mask_punctuation``, a
    different class.

    The field was deleted, so Keras' unknown-keyword check now fires. MEASURED:
    the exception is ``ValueError`` with the message ``Unrecognized keyword
    arguments passed to ColBERT: {'mask_punctuation': False}`` -- on the class
    and through both factories. This test also pins that the key is gone from
    ``get_config()``, so a re-added-but-still-inert field cannot pass it.
    """
    for construct in (
        lambda: make_model(mask_punctuation=False),
        lambda: ColBERT.from_variant("tiny", mask_punctuation=False),
        lambda: create_colbert_v1("tiny", mask_punctuation=False),
        lambda: create_colbert_v2("tiny", mask_punctuation=False),
    ):
        with pytest.raises(ValueError, match="mask_punctuation"):
            construct()

    assert "mask_punctuation" not in make_model().get_config(), (
        "mask_punctuation reappeared in get_config(): the model has no reader "
        "for it, so serializing it advertises an intent nothing honors"
    )


def test_a_structural_knob_changes_the_weight_signature():
    """Shared oracle: ``num_layers`` and ``dim`` must reach the parameterisation."""
    assert_structural_knob_changes_weights(
        {
            1: lambda: _built(make_model(num_layers=1)),
            2: lambda: _built(make_model(num_layers=2)),
            3: lambda: _built(make_model(num_layers=3)),
        },
        knob="num_layers",
    )
    assert_structural_knob_changes_weights(
        {
            4: lambda: _built(make_model(dim=4)),
            8: lambda: _built(make_model(dim=8)),
            16: lambda: _built(make_model(dim=16)),
        },
        knob="dim",
    )


def _built(model):
    """Build ``model`` on the shared shapes and return it."""
    model.build(BUILD_SHAPES)
    return model


# ---------------------------------------------------------------------
# 9. .predict() on a dict input
# ---------------------------------------------------------------------


def test_predict_works_on_a_dict_input_with_and_without_the_optional_masks():
    """The D-032 structural constraint, inherited from BERT.

    ``model(inputs)`` works whatever the output structure is; ``.predict()``
    concatenates per-batch outputs and breaks the moment a slot's presence
    depends on the input. Both arms are exercised because the optional-mask arm
    is the one that regresses.
    """
    model = make_model()
    full = make_inputs()
    minimal = {
        QUERY_INPUT_IDS_KEY: full[QUERY_INPUT_IDS_KEY],
        DOC_INPUT_IDS_KEY: full[DOC_INPUT_IDS_KEY],
    }

    for name, inputs in (("full", full), ("minimal", minimal)):
        predicted = model.predict(inputs, verbose=0)
        assert set(predicted) == {"score", "query_embeddings", "doc_embeddings"}, (
            f"{name}: .predict() returned keys {sorted(predicted)}; the output "
            "structure must not depend on which optional inputs were supplied"
        )
        assert np.asarray(predicted["score"]).shape == (BATCH,)
        assert np.asarray(predicted["query_embeddings"]).shape == (
            BATCH,
            QUERY_LEN,
            TINY["dim"],
        )
        assert np.all(np.isfinite(np.asarray(predicted["score"])))


# ---------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,needle",
    [
        ({"dim": 0}, "dim"),
        ({"query_maxlen": 3}, "query_maxlen"),
        ({"doc_maxlen": 2}, "doc_maxlen"),
        ({"doc_maxlen": 999}, "max_position_embeddings"),
    ],
)
def test_an_unusable_configuration_raises_naming_the_bad_value(overrides, needle):
    with pytest.raises(ValueError) as excinfo:
        make_model(**overrides)
    assert needle in str(excinfo.value)


def test_call_rejects_a_missing_input_ids_entry():
    model = make_model()
    inputs = make_inputs()
    del inputs[DOC_INPUT_IDS_KEY]
    with pytest.raises(ValueError, match=DOC_INPUT_IDS_KEY):
        model(inputs)


def test_encode_query_and_encode_document_return_normalized_embeddings():
    """Both public encoders emit unit-norm vectors at kept positions."""
    model = make_model()
    inputs = make_inputs()

    query = np.asarray(
        keras.ops.convert_to_numpy(
            model.encode_query(
                {
                    "input_ids": inputs[QUERY_INPUT_IDS_KEY],
                    "attention_mask": inputs[QUERY_ATTENTION_MASK_KEY],
                },
                training=False,
            )
        )
    )
    document = np.asarray(
        keras.ops.convert_to_numpy(
            model.encode_document(
                {
                    "input_ids": inputs[DOC_INPUT_IDS_KEY],
                    "attention_mask": inputs[DOC_ATTENTION_MASK_KEY],
                    "skiplist_mask": inputs[DOC_SKIPLIST_MASK_KEY],
                },
                training=False,
            )
        )
    )

    assert query.shape == (BATCH, QUERY_LEN, TINY["dim"])
    assert document.shape == (BATCH, DOC_LEN, TINY["dim"])
    for name, array in (("query", query), ("document", document)):
        norms = np.linalg.norm(array, axis=-1)
        np.testing.assert_allclose(
            norms,
            np.ones_like(norms),
            atol=1e-6,
            rtol=0,
            err_msg=f"{name} embeddings are not unit-norm at kept positions",
        )


# ---------------------------------------------------------------------
# 10. XLA-compiled forward pass
# ---------------------------------------------------------------------


def test_a_fully_masked_document_scores_the_exact_sentinel_under_xla():
    """The D-006/D-007 arithmetic survives ``jit_compile=True``.

    This is the module's only XLA-compiled guard, and it runs the whole
    ``from_variant("tiny")`` forward pass -- backbone, projection and
    ``MaxSimScorer`` -- inside ``tf.function(..., jit_compile=True)`` under
    ``mixed_float16``, the policy the promotion exists for.

    The reference is HAND-DERIVED, not a second run of the same source. A
    document whose ``doc_attention_mask`` is all zeros puts every document
    position behind the sentinel, so every one of the ``query_maxlen`` query
    terms maxes to ``mask_value`` and the sum is exactly
    ``query_maxlen * mask_value`` -- ``32 * -1e4 = -320000.0`` for every
    variant row. That number is arithmetic, so an injection can only move the
    measured side; an eager-vs-XLA comparison of the same source would move
    both sides and could never fail. Same construction and same claim as
    ``test_components.py``'s
    ``test_an_all_masked_document_yields_a_finite_score``, lifted to the full
    model and to XLA.

    The scorer's return dtype is asserted too: it is ``float32`` under
    ``mixed_float16`` by D-007's deliberately accepted consequence, and a
    float16 sum of 32 sentinels overflows binary16's 65504 to ``-inf``.
    """
    import tensorflow as tf

    previous = keras.mixed_precision.global_policy()
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")

        model = ColBERT.from_variant("tiny")
        query_len = model.query_maxlen
        # Pin the two operands the docstring's "32 * -1e4 = -320000.0" is made
        # of. Without these the expectation below is read off the object under
        # test, so a change to DEFAULT_MAXSIM_MASK_VALUE or to the row's
        # query_maxlen would move oracle and measurement together and this
        # guard would keep passing while the documented arithmetic was gone.
        assert query_len == 32, (
            f"the 'tiny' row's query_maxlen moved to {query_len}; this guard's "
            "hand-derived reference is 32 * -1e4 = -320000.0"
        )
        assert model.scorer.mask_value == -1e4, (
            f"the MaxSim mask sentinel moved to {model.scorer.mask_value}; "
            "this guard's hand-derived reference is 32 * -1e4 = -320000.0"
        )
        doc_len = 16
        rng = np.random.default_rng(17)
        inputs = {
            QUERY_INPUT_IDS_KEY: rng.integers(
                0, model.vocab_size, (BATCH, query_len)
            ).astype("int32"),
            QUERY_ATTENTION_MASK_KEY: np.ones((BATCH, query_len), dtype="int32"),
            DOC_INPUT_IDS_KEY: rng.integers(
                0, model.vocab_size, (BATCH, doc_len)
            ).astype("int32"),
            # Every document position masked out.
            DOC_ATTENTION_MASK_KEY: np.zeros((BATCH, doc_len), dtype="int32"),
            DOC_SKIPLIST_MASK_KEY: np.ones((BATCH, doc_len), dtype="int32"),
        }

        compiled = tf.function(
            lambda batch: model(batch, training=False), jit_compile=True
        )
        outputs = compiled(inputs)

        assert outputs["score"].dtype == tf.float32, (
            "the XLA-compiled score is not float32 under mixed_float16: got "
            f"{outputs['score'].dtype}; the D-007 promotion did not survive "
            "XLA lowering, and a float16 sum of "
            f"{query_len} sentinels overflows to -inf"
        )

        score = np.asarray(
            keras.ops.convert_to_numpy(outputs["score"]), dtype=np.float64
        )
        assert np.all(np.isfinite(score)), (
            f"an all-masked document produced a non-finite score under XLA: "
            f"{score}"
        )
        np.testing.assert_allclose(
            score,
            np.full((BATCH,), query_len * model.scorer.mask_value),
            atol=1e-6,
            rtol=0,
            err_msg=(
                "the XLA-compiled all-masked score is not "
                "query_maxlen * mask_value"
            ),
        )
    finally:
        keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# 11. base/large variant geometry
# ---------------------------------------------------------------------

# EXTERNAL oracle. These are the published BERT-Base and BERT-Large backbone
# dimensions (Devlin et al. 2018, table at the end of section 3: L=12 H=768
# A=12 and L=24 H=1024 A=16, feed-forward 4H in both), plus ColBERT's own
# retrieval dimension dim=128 (Khattab & Zaharia 2020, section 3.1). They are
# facts about the literature, NOT about this implementation, and that is the
# whole point: every other assertion in the test below compares the built
# model against the SAME MODEL_VARIANTS row, so a row rewritten to a different
# but internally consistent backbone moves both sides together and stays green
# (measured: rewriting `base` to hidden 1024 / layers 10 / heads 16 /
# intermediate 4096 passed both parametrizations before this table existed).
REFERENCE_BACKBONE_GEOMETRY = {
    "base": {
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "dim": 128,
    },
    "large": {
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "intermediate_size": 4096,
        "dim": 128,
    },
}


@pytest.mark.parametrize("variant", ("base", "large"))
def test_the_base_and_large_rows_keep_their_reference_backbone_geometry(variant):
    """Structural regression guard over the two full-size ``MODEL_VARIANTS`` rows.

    Both rows are internally consistent today -- ``intermediate_size /
    hidden_size`` is exactly 4.0x and ``hidden_size % num_heads == 0`` for
    every row in the table -- so this test finds nothing at the commit that
    adds it. Its subject is a FUTURE edit to those rows: ``base`` is
    BERT-Base's backbone and ``large`` is BERT-Large's, and a hand-tweaked
    number that breaks the head divisor or the 4x feed-forward ratio would
    otherwise only surface as a construction error in somebody's training run.

    The FIRST arm compares the row against ``REFERENCE_BACKBONE_GEOMETRY``, an
    external oracle taken from the BERT and ColBERT papers. Everything after it
    compares the built model against the row, which is self-referential: a row
    rewritten to a different-but-consistent backbone moves both sides together.
    Those arms are kept because they catch a different defect class -- a row the
    constructor silently fails to honour -- but only the external table can
    catch the row itself drifting off BERT-Base/BERT-Large.

    The build shape is fixed and stated: ``{query_input_ids: (None, 32),
    doc_input_ids: (None, 64)}``. It is written here rather than taken from
    the row's ``doc_maxlen`` because these are the two largest variants and
    construction alone is what is being guarded -- there is no forward pass.

    Construction runs inside ``tf.device("/CPU:0")``. ``large`` is ~334M
    parameters, new peak memory for this module, and on a GPU 1 shared with a
    training job it raised ``ResourceExhaustedError ... [Op:AddV2]`` -- a red
    that depends on what else is running, in the gate this plan uses as its
    instrument. The test needs no device (it never runs a forward pass) and
    costs 4.45s on CPU against 2.8s on an idle GPU.

    No exact ``count_params()`` is asserted, deliberately. That figure tracks
    the build shapes the caller happens to pass, not the variant row: the same
    ``base`` model measures 108,989,952 parameters at document length 64 and
    162,561,792 at a longer one, because the backbone's position-embedding
    table is sized by the build. Pinning either number would pin whatever the
    implementation produced on the day this was written. The geometry
    invariants below are build-shape independent, and they are what a bad row
    edit actually violates.
    """
    import tensorflow as tf

    row = ColBERT.MODEL_VARIANTS[variant]
    reference = REFERENCE_BACKBONE_GEOMETRY[variant]

    # Asserted from the row BEFORE construction, so a bad row is reported as a
    # geometry violation naming the offending numbers rather than as whatever
    # the backbone happens to raise first.
    assert row["hidden_size"] % row["num_heads"] == 0, (
        f"the '{variant}' row cannot split its hidden size across its heads: "
        f"hidden_size={row['hidden_size']} is not divisible by "
        f"num_heads={row['num_heads']}"
    )
    assert row["intermediate_size"] == 4 * row["hidden_size"], (
        f"the '{variant}' row broke the 4x feed-forward ratio every BERT-family "
        f"backbone in this table uses: intermediate_size="
        f"{row['intermediate_size']}, hidden_size={row['hidden_size']}"
    )

    # The external arm, run AFTER the two intra-row arms so an internally
    # INCONSISTENT row still reports as the specific violation it is (a head
    # divisor or ratio break) rather than as a reference mismatch. Nothing
    # here reads the built model, so an edit to the row cannot move this
    # expectation with it.
    for key, expected in reference.items():
        assert row[key] == expected, (
            f"the '{variant}' row no longer matches its reference backbone: "
            f"{key}={row[key]}, but BERT-"
            f"{'Base' if variant == 'base' else 'Large'}/ColBERT declares "
            f"{key}={expected}"
        )

    with tf.device("/CPU:0"):
        model = ColBERT.from_variant(variant)
    try:
        with tf.device("/CPU:0"):
            model.build(
                {
                    QUERY_INPUT_IDS_KEY: (None, 32),
                    DOC_INPUT_IDS_KEY: (None, 64),
                }
            )

        assert len(model.encoder.encoder_layers) == row["num_layers"], (
            f"the '{variant}' backbone materialized "
            f"{len(model.encoder.encoder_layers)} transformer layers, but the "
            f"row declares num_layers={row['num_layers']}"
        )
        assert model.projection.dense.units == row["dim"], (
            f"the '{variant}' projection emits "
            f"{model.projection.dense.units} dimensions, not the row's "
            f"dim={row['dim']}"
        )
        assert tuple(model.projection.dense.kernel.shape) == (
            row["hidden_size"],
            row["dim"],
        ), (
            "the projection kernel does not map the backbone's hidden size to "
            f"the retrieval dimension: got "
            f"{tuple(model.projection.dense.kernel.shape)}, expected "
            f"({row['hidden_size']}, {row['dim']})"
        )
    finally:
        del model
