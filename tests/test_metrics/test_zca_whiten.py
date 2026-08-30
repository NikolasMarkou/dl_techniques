"""`zca_whiten` must equalize variances without inventing signal.

The transform exists because cosine similarity weights every dimension equally,
so content surviving only in low-variance directions is invisible to it while
remaining fully available to a linear probe. That gap is not hypothetical: on
this repository's character-level encoders, SST-2 probe accuracy stayed within
0.03 while SQuAD recall@1 fell up to 10x, and whitening recovered ~3.3x of it.

The risk with any such transform is that it manufactures apparent signal. These
tests pin both directions -- it must recover genuinely drowned content, and it
must NOT improve retrieval when there is nothing to recover.
"""

import numpy as np
import pytest

from dl_techniques.metrics.embedding_quality import (
    anisotropy,
    l2_normalize,
    ranking_metrics,
    zca_whiten,
)


def test_the_output_is_white() -> None:
    """Zero mean, and a covariance that is the identity."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8)) @ rng.normal(size=(8, 8)) + 3.0
    (white,) = zca_whiten(x)
    assert np.allclose(white.mean(axis=0), 0.0, atol=1e-8)
    cov = (white.T @ white) / (len(white) - 1)
    assert np.allclose(cov, np.eye(8), atol=1e-4)


def test_applied_arrays_use_the_fit_statistics() -> None:
    """A second array is transformed by `fit`'s mean and covariance, not its own.

    This is what makes an out-of-domain fit meaningful: the queries must not be
    re-centred on themselves, or query and context land in different frames.
    """
    rng = np.random.default_rng(0)
    fit = rng.normal(size=(300, 6)) * 2.0 + 1.0
    other = rng.normal(size=(50, 6)) * 2.0 + 1.0
    _, moved = zca_whiten(fit, other)
    # transforming `other` alone would centre it on its own mean instead
    (alone,) = zca_whiten(other)
    assert not np.allclose(moved, alone, atol=1e-6)


def test_it_recovers_content_drowned_by_a_high_variance_direction() -> None:
    """The case the transform exists for."""
    rng = np.random.default_rng(0)
    n, d = 400, 16
    axis = np.array([[5.0] + [0.0] * (d - 1)])
    content = rng.normal(size=(n, d)) * 0.05
    # The loud direction is drawn INDEPENDENTLY for contexts and queries. Give
    # both the same draw and it becomes a perfect matching key, raw recall hits
    # 1.0000, and there is nothing left for whitening to recover -- which is
    # how the first version of this test managed to fail against a working
    # transform.
    ctx = content + rng.normal(size=(n, 1)) * axis
    queries = (
        content
        + rng.normal(size=(n, d)) * 0.01
        + rng.normal(size=(n, 1)) * axis
    )
    truth = np.arange(n)

    raw = ranking_metrics(
        l2_normalize(queries) @ l2_normalize(ctx).T, truth
    )["recall_at_1"]
    ctx_w, q_w = zca_whiten(ctx, queries)
    whitened = ranking_metrics(
        l2_normalize(q_w) @ l2_normalize(ctx_w).T, truth
    )["recall_at_1"]
    assert whitened > raw * 2.0, (
        f"raw {raw:.4f} -> whitened {whitened:.4f}: the transform failed to "
        f"recover content that is present but low-variance."
    )


def test_it_does_not_invent_signal_where_there_is_none() -> None:
    """Anti-vacuity, and the more important half.

    With queries independent of the contexts, retrieval is at chance and must
    stay there. A transform that raised recall here would be manufacturing
    structure, and every gain measured with it would be an artifact.
    """
    rng = np.random.default_rng(0)
    n, d = 400, 16
    ctx = rng.normal(size=(n, d)) @ rng.normal(size=(d, d))
    queries = rng.normal(size=(n, d)) @ rng.normal(size=(d, d))
    truth = np.arange(n)

    ctx_w, q_w = zca_whiten(ctx, queries)
    whitened = ranking_metrics(
        l2_normalize(q_w) @ l2_normalize(ctx_w).T, truth
    )["recall_at_1"]
    chance = 1.0 / n
    assert whitened < 20 * chance, (
        f"whitened recall@1 is {whitened:.4f} against chance {chance:.4f} on "
        f"INDEPENDENT queries and contexts. The transform is inventing signal."
    )


def test_anisotropy_is_removed() -> None:
    """A shared direction is exactly what whitening should flatten."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 12)) * 0.1 + 4.0
    before = anisotropy(l2_normalize(x))
    (white,) = zca_whiten(x)
    after = anisotropy(l2_normalize(white))
    assert before > 0.5 and abs(after) < 0.05, (
        f"anisotropy {before:.4f} -> {after:.4f}"
    )


@pytest.mark.parametrize(
    "fit_shape,other_shape",
    [((1, 4), (2, 4)), ((10, 4), (2, 5))],
)
def test_bad_shapes_raise(fit_shape, other_shape) -> None:
    """A silent broadcast here would corrupt every downstream metric."""
    with pytest.raises(ValueError):
        zca_whiten(np.zeros(fit_shape), np.zeros(other_shape))
