"""Guards for ColBERT's two scoring components.

The MaxSim oracle in this module is written from the *formula*, not from
``components.py``. Its source is the ColBERT paper's definition, restated in
``plans/.../findings/colbert-architecture-reference.md`` §1::

    S_{q,d} = sum_i  max_j  E_qi . E_dj^T

Everything the oracle does beyond that sum-of-max is an explicitly named
divergence term, so a reader can see exactly where the port stops being the
paper:

    D1 (reference behaviour): a masked document position does not have its
        column deleted; its score is overwritten with a large negative
        sentinel before the max-reduce. Source: ``colbert_score_reduce``
        (``scores_padded[D_padding] = -9999``).
    D2 (this port's addition): a masked *query* position contributes exactly
        0 to the sum. The reference pads queries with ``[MASK]``, whose
        embedding is real and non-zero, so without this term a padding query
        term would contribute its own best match to the score.

No constant in the oracle comes from reading the implementation. The sentinel
is passed in by the caller rather than hard-coded, so the oracle never encodes
the implementation's default.

RED-PROOF RESULTS (four injections, each restored by ``cp`` + ``diff -q``;
never ``git stash`` / ``git checkout --``):

    (a) delete the sentinel ``where`` block in ``MaxSimScorer.call``
        -> RED: test_a_padded_doc_position_with_a_huge_embedding_cannot_win_the_max
           (assertion "the huge masked doc position won the max")
    (b) swap the max and sum axes (max over query axis, sum over doc axis)
        -> RED: test_the_maxsim_matches_the_reference_derived_oracle
           (assertion "MaxSim disagrees with the reference-derived oracle")
    (c) move ``keras.ops.normalize`` BEFORE the mask multiply
        -> RED: test_the_mask_is_applied_before_the_normalize
           (assertion "kept rows are not unit-norm ...")
    (d) dead-component arm: ``ColBERTProjection.call`` returns its input
        unchanged
        -> RED: test_projection_rows_are_unit_norm_at_unmasked_positions
           (assertion "projection did not return the requested dim")

MEASURED CAVEAT for (c): with a strictly BINARY mask the two orderings are
mathematically identical -- ``normalize(x * m) == normalize(x) * m`` for
``m in {0, 1}`` -- so no binary-mask test can distinguish them, and a guard
written with a binary mask would be vacuous. The ordering is therefore pinned
with a fractional mask, which is the smallest input on which the two orders
actually differ. See that test's own docstring.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.language.colbert.components import (
    ColBERTProjection,
    MaxSimScorer,
)


# ---------------------------------------------------------------------
# Independent oracle -- derived from the paper formula, not from the code
# ---------------------------------------------------------------------


def maxsim_oracle(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_mask: np.ndarray = None,
    query_mask: np.ndarray = None,
    sentinel: float = -1.0e4,
) -> np.ndarray:
    """``S_{q,d} = sum_i max_j E_qi . E_dj`` in explicit Python loops.

    Written from the formula. D1 and D2 (see the module docstring) are the only
    departures from the bare sum-of-max, and both are applied here in the open.

    :param query_embeddings: ``(batch, query_len, dim)``.
    :param doc_embeddings: ``(batch, doc_len, dim)``.
    :param doc_mask: ``(batch, doc_len)``, 1 = keep. D1.
    :param query_mask: ``(batch, query_len)``, 1 = keep. D2.
    :param sentinel: the value a masked document position scores (D1).
    :return: ``(batch,)`` scores as float64.
    """
    query_embeddings = np.asarray(query_embeddings, dtype=np.float64)
    doc_embeddings = np.asarray(doc_embeddings, dtype=np.float64)

    batch, query_len, _ = query_embeddings.shape
    doc_len = doc_embeddings.shape[1]

    scores = np.zeros((batch,), dtype=np.float64)
    for b in range(batch):
        total = 0.0
        for i in range(query_len):
            per_doc_token = []
            for j in range(doc_len):
                if doc_mask is not None and doc_mask[b, j] == 0:
                    per_doc_token.append(sentinel)  # D1
                else:
                    per_doc_token.append(
                        float(
                            np.dot(query_embeddings[b, i], doc_embeddings[b, j])
                        )
                    )
            best = max(per_doc_token)
            if query_mask is not None and query_mask[b, i] == 0:
                best = 0.0  # D2
            total += best
        scores[b] = total
    return scores


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _unit(rng, shape):
    x = rng.normal(size=shape).astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ---------------------------------------------------------------------
# 1. MaxSim vs the oracle
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch,query_len,doc_len,dim",
    [
        (1, 1, 1, 4),
        (2, 3, 7, 8),  # non-square: query_len != doc_len
        (3, 8, 2, 16),  # non-square the other way round
        (2, 5, 5, 32),
    ],
)
def test_the_maxsim_matches_the_reference_derived_oracle(
    batch, query_len, doc_len, dim
):
    """MaxSim reproduces ``sum_i max_j E_qi . E_dj`` at atol=1e-6, rtol=0."""
    rng = np.random.default_rng(1234 + batch * 31 + doc_len)
    q = _unit(rng, (batch, query_len, dim))
    d = _unit(rng, (batch, doc_len, dim))
    doc_mask = (rng.random((batch, doc_len)) > 0.3).astype(np.float32)
    doc_mask[:, 0] = 1.0  # at least one kept position per document
    query_mask = (rng.random((batch, query_len)) > 0.3).astype(np.float32)
    query_mask[:, 0] = 1.0

    scorer = MaxSimScorer()
    got = np.asarray(
        scorer(q, d, doc_mask=doc_mask, query_mask=query_mask), dtype=np.float64
    )
    expected = maxsim_oracle(
        q, d, doc_mask=doc_mask, query_mask=query_mask, sentinel=-1.0e4
    )

    assert got.shape == (batch,), (
        f"MaxSim output shape {got.shape} is not the expected ({batch},)"
    )
    np.testing.assert_allclose(
        got,
        expected,
        atol=1e-6,
        rtol=0,
        err_msg="MaxSim disagrees with the reference-derived oracle",
    )


def test_the_maxsim_matches_the_oracle_without_any_mask():
    """The unmasked path is the bare sum-of-max, with no divergence terms."""
    rng = np.random.default_rng(7)
    q = _unit(rng, (2, 4, 8))
    d = _unit(rng, (2, 6, 8))

    got = np.asarray(MaxSimScorer()(q, d), dtype=np.float64)
    expected = maxsim_oracle(q, d)

    np.testing.assert_allclose(
        got,
        expected,
        atol=1e-6,
        rtol=0,
        err_msg="MaxSim disagrees with the reference-derived oracle",
    )


# ---------------------------------------------------------------------
# 2. A padded document position cannot win the max
# ---------------------------------------------------------------------


def test_a_padded_doc_position_with_a_huge_embedding_cannot_win_the_max():
    """A masked doc token given a deliberately enormous, query-aligned
    embedding must be excluded by the sentinel, not merely out-competed.

    The trap this avoids: with ordinary unit-norm embeddings a padded position
    loses the max by construction, so the guard would pass with the sentinel
    deleted. Here the padded position is 1000x the query direction, so it wins
    unless something actively removes it.
    """
    dim = 8
    q = np.zeros((1, 2, dim), dtype=np.float32)
    q[0, 0, 0] = 1.0
    q[0, 1, 1] = 1.0

    d = np.zeros((1, 4, dim), dtype=np.float32)
    d[0, 0, 0] = 1.0  # kept, matches query term 0 with score 1.0
    d[0, 1, 1] = 1.0  # kept, matches query term 1 with score 1.0
    d[0, 2, 0] = 1000.0  # MASKED, huge, aligned with query term 0
    d[0, 3, 1] = 1000.0  # MASKED, huge, aligned with query term 1
    doc_mask = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    got = float(np.asarray(MaxSimScorer()(q, d, doc_mask=doc_mask))[0])
    expected = float(maxsim_oracle(q, d, doc_mask=doc_mask, sentinel=-1.0e4)[0])

    assert got < 10.0, (
        f"the huge masked doc position won the max (score {got}, expected "
        f"about {expected})"
    )
    np.testing.assert_allclose(
        got,
        expected,
        atol=1e-6,
        rtol=0,
        err_msg="the huge masked doc position won the max",
    )


# ---------------------------------------------------------------------
# 3. An all-masked document scores finite, in both dtype policies
# ---------------------------------------------------------------------


@pytest.mark.parametrize("policy_name", ["float32", "mixed_float16"])
def test_an_all_masked_document_yields_a_finite_score(policy_name):
    """Every document position masked -> the max reduces over an all-sentinel
    row. The score must be finite: never NaN, never -inf."""
    previous = keras.mixed_precision.global_policy()
    try:
        keras.mixed_precision.set_global_policy(policy_name)

        rng = np.random.default_rng(11)
        q = _unit(rng, (2, 3, 8))
        d = _unit(rng, (2, 5, 8))
        doc_mask = np.zeros((2, 5), dtype=np.float32)

        got = np.asarray(
            MaxSimScorer()(q, d, doc_mask=doc_mask), dtype=np.float64
        )

        assert np.all(np.isfinite(got)), (
            f"an all-masked document produced a non-finite score under "
            f"{policy_name}: {got}"
        )
        # 3 query terms, each reducing to the sentinel.
        np.testing.assert_allclose(
            got,
            np.full((2,), 3 * -1.0e4),
            atol=1e-6,
            rtol=0,
            err_msg=(
                "an all-masked document did not reduce to query_len * sentinel"
            ),
        )
    finally:
        keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# 4 + 5. Projection norms
# ---------------------------------------------------------------------


def test_projection_rows_are_unit_norm_at_unmasked_positions():
    """Every kept position leaves the projection with L2 norm 1."""
    rng = np.random.default_rng(3)
    hidden = rng.normal(size=(2, 6, 32)).astype(np.float32)
    mask = np.array(
        [[1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0]], dtype=np.float32
    )

    projection = ColBERTProjection(dim=12)
    out = np.asarray(projection(hidden, mask=mask), dtype=np.float64)

    assert out.shape == (2, 6, 12), (
        f"projection did not return the requested dim: got shape {out.shape}, "
        "expected (2, 6, 12)"
    )
    norms = np.linalg.norm(out, axis=-1)
    kept = mask.astype(bool)
    np.testing.assert_allclose(
        norms[kept],
        np.ones(int(kept.sum())),
        atol=1e-6,
        rtol=0,
        err_msg="an unmasked projection row is not unit-norm",
    )


def test_a_fully_masked_projection_row_is_exactly_zero():
    """MEASURED behaviour, asserted as such: because the mask multiply happens
    BEFORE ``keras.ops.normalize``, a masked row normalizes a zero vector.
    ``keras.ops.normalize`` divides by ``max(norm, epsilon)``, so the result is
    **exactly zero** -- not NaN, and not a unit vector. This test asserts the
    exact-zero branch of the "zero OR unit-norm-safe" alternative.
    """
    rng = np.random.default_rng(5)
    hidden = rng.normal(size=(2, 4, 16)).astype(np.float32)
    mask = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32)

    out = np.asarray(ColBERTProjection(dim=8)(hidden, mask=mask))

    assert np.all(np.isfinite(out)), (
        "the projection produced a non-finite value at a masked position"
    )
    masked_rows = out[~mask.astype(bool)]
    np.testing.assert_array_equal(
        masked_rows,
        np.zeros_like(masked_rows),
        err_msg="a masked projection row is not exactly zero",
    )


def test_the_mask_is_applied_before_the_normalize():
    """Pins the mask/normalize ORDER (H-3).

    Measured first: with a strictly binary mask the two orderings are
    identical, since ``normalize(x * m) == normalize(x) * m`` for
    ``m in {0, 1}``. A binary-mask guard would therefore pass both ways -- the
    vacuous-guard failure mode. The smallest input that separates the two
    orders is a FRACTIONAL mask: masking first then normalizing divides the
    scale back out and yields a unit-norm row, whereas normalizing first and
    masking second leaves a row of norm equal to the mask value.
    """
    rng = np.random.default_rng(17)
    hidden = rng.normal(size=(1, 3, 16)).astype(np.float32)
    mask = np.full((1, 3), 0.25, dtype=np.float32)

    out = np.asarray(ColBERTProjection(dim=8)(hidden, mask=mask), dtype=np.float64)
    norms = np.linalg.norm(out, axis=-1)

    np.testing.assert_allclose(
        norms,
        np.ones((1, 3)),
        atol=1e-6,
        rtol=0,
        err_msg=(
            "kept rows are not unit-norm under a fractional mask -- the "
            "normalize ran BEFORE the mask multiply, not after"
        ),
    )


# ---------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("bad_dim", [0, -1, -128])
def test_the_projection_rejects_a_non_positive_dim(bad_dim):
    with pytest.raises(ValueError, match="strictly positive"):
        ColBERTProjection(dim=bad_dim)


@pytest.mark.parametrize("bad_value", [0.0, 1.0, float("-inf"), float("inf")])
def test_the_scorer_rejects_a_non_finite_or_non_negative_sentinel(bad_value):
    with pytest.raises(ValueError):
        MaxSimScorer(mask_value=bad_value)


@pytest.mark.parametrize("mask_value", [-1.0e4, -1.0e9])
def test_the_sentinel_sum_stays_finite_under_mixed_float16(mask_value):
    """The overflow lives in the SUM, not in the sentinel.

    A fully-masked document reduces to ``query_len * sentinel``. At ColBERT's
    own default ``query_maxlen = 32`` that is ``32 * -1e4 = -3.2e5``, which is
    not representable in binary16 (max 65504). Clamping the sentinel alone does
    not save it; the reduction itself must be promoted. This arm uses the real
    default query length, and a second, deliberately out-of-range sentinel.
    """
    previous = keras.mixed_precision.global_policy()
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
        rng = np.random.default_rng(23)
        q = _unit(rng, (1, 32, 8))  # ColBERT's default query_maxlen
        d = _unit(rng, (1, 12, 8))
        doc_mask = np.zeros((1, 12), dtype=np.float32)

        got = np.asarray(
            MaxSimScorer(mask_value=mask_value)(q, d, doc_mask=doc_mask),
            dtype=np.float64,
        )
        assert np.all(np.isfinite(got)), (
            f"the sentinel sum overflowed to a non-finite score: {got}"
        )
        np.testing.assert_allclose(
            got,
            np.full((1,), 32 * mask_value),
            rtol=1e-6,
            atol=0,
            err_msg="the all-masked score is not query_len * sentinel",
        )
    finally:
        keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# 6. get_config round trips
# ---------------------------------------------------------------------


def test_the_projection_config_round_trips():
    layer = ColBERTProjection(dim=64, name="proj")
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(layer)
    )
    assert isinstance(restored, ColBERTProjection), (
        "the projection did not deserialize back to ColBERTProjection"
    )
    assert restored.dim == 64, (
        f"dim did not survive the config round trip: {restored.dim} != 64"
    )
    assert restored.get_config()["dim"] == layer.get_config()["dim"]


def test_the_scorer_config_round_trips_and_preserves_behaviour():
    layer = MaxSimScorer(mask_value=-5000.0, name="scorer")
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(layer)
    )
    assert isinstance(restored, MaxSimScorer), (
        "the scorer did not deserialize back to MaxSimScorer"
    )
    assert restored.mask_value == -5000.0, (
        f"mask_value did not survive the config round trip: {restored.mask_value}"
    )

    rng = np.random.default_rng(41)
    q = _unit(rng, (2, 3, 8))
    d = _unit(rng, (2, 4, 8))
    doc_mask = np.array(
        [[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32
    )
    np.testing.assert_allclose(
        np.asarray(restored(q, d, doc_mask=doc_mask), dtype=np.float64),
        np.asarray(layer(q, d, doc_mask=doc_mask), dtype=np.float64),
        atol=1e-6,
        rtol=0,
        err_msg="the restored scorer does not reproduce the original's scores",
    )


# ---------------------------------------------------------------------
# 7. dtype-policy arm
# ---------------------------------------------------------------------


def test_both_layers_run_under_mixed_float16_with_finite_outputs():
    """Process-global dtype policy is restored in a finally -- policy leakage
    across tests has bitten this repository repeatedly."""
    previous = keras.mixed_precision.global_policy()
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")

        rng = np.random.default_rng(97)
        hidden = rng.normal(size=(2, 6, 32)).astype(np.float32)
        doc_mask = np.array(
            [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=np.float32
        )
        query_mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.float32)

        projection = ColBERTProjection(dim=8)
        doc = projection(hidden, mask=doc_mask)
        query = projection(hidden[:, :3, :], mask=query_mask)

        assert keras.backend.standardize_dtype(doc.dtype) == "float16", (
            f"the projection did not compute in float16 under mixed_float16: "
            f"{doc.dtype}"
        )

        scores = np.asarray(
            MaxSimScorer()(query, doc, doc_mask=doc_mask, query_mask=query_mask),
            dtype=np.float64,
        )
        assert np.all(np.isfinite(scores)), (
            f"MaxSim produced a non-finite score under mixed_float16: {scores}"
        )
        assert np.all(np.isfinite(np.asarray(doc, dtype=np.float64))), (
            "the projection produced a non-finite value under mixed_float16"
        )

        # Half precision only; the oracle is the same formula at float64.
        expected = maxsim_oracle(
            np.asarray(query, dtype=np.float64),
            np.asarray(doc, dtype=np.float64),
            doc_mask=doc_mask,
            query_mask=query_mask,
            sentinel=-1.0e4,
        )
        np.testing.assert_allclose(
            scores,
            expected,
            atol=1e-2,
            rtol=0,
            err_msg="MaxSim disagrees with the oracle under mixed_float16",
        )
    finally:
        keras.mixed_precision.set_global_policy(previous)
