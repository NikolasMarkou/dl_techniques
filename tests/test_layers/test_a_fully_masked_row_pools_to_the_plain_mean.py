"""A fully-masked row pools to the plain mean in both learnable poolers.

Guard for plan-2026-10-14T062616-81938e81 step 5.4 (D-203). ``WeightedPooling``
documented this degeneration and ``AttentionPooling.call`` did not; both now
say it, and this pins what they say: every logit becomes the same sentinel, so
the softmax is uniform and the output is the mean over the sequence, finite.
"""

import numpy as np
import pytest

from dl_techniques.layers.sequence_pooling import AttentionPooling, WeightedPooling

# The pooled sum is an einsum: TF32 on a GPU moves it by ~1e-3 relative.
pytestmark = pytest.mark.usefixtures("tf32_disabled")


@pytest.mark.parametrize(
    "layer",
    [lambda: AttentionPooling(hidden_dim=4, num_heads=2), lambda: WeightedPooling(max_seq_len=8)],
    ids=["attention", "weighted"],
)
def test_a_fully_masked_row_is_the_plain_mean(layer):
    x = np.random.default_rng(0).normal(size=(2, 5, 3)).astype("float32")
    mask = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], "float32")
    pooler = layer()
    pooler.build((None, 5, 3))
    for w in pooler.weights:
        w.assign(np.random.default_rng(1).normal(size=tuple(w.shape)).astype("float32"))
    out = np.asarray(pooler(x, mask=mask))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0], x[0].mean(axis=0), rtol=1e-5, atol=1e-6)
