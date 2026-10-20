"""A masked position gets no weight even when every kept logit is below -1e4.

Guard for plan-2026-10-14T062616-81938e81 step 5.4 (D-203). ``AttentionPooling``
and ``WeightedPooling`` masked their logits with a literal ``-1e4`` in every
dtype. Measured at 69b68fe5a in float32 with a small temperature (kept logits
near -1e6): the masked position won the whole softmax, and both poolers returned
the masked token itself instead of the mean of the kept ones. They now select
``utils.dtype_policy.mask_sentinel(compute_dtype)`` (-1e9 in float32, -1e4 in
float16, unchanged there), the same sentinel as the ``max``, ``min`` and
``top_k_*`` strategies.
"""

import numpy as np
import pytest

from dl_techniques.layers.sequence_pooling import (
    AttentionPooling,
    SequencePooling,
    WeightedPooling,
)
from dl_techniques.utils.dtype_policy import mask_sentinel

# The pooled sum is an einsum: TF32 on a GPU moves it by ~1e-3 relative.
pytestmark = pytest.mark.usefixtures("tf32_disabled")

X = np.random.default_rng(0).normal(size=(1, 4, 3)).astype("float32")
MASK = np.array([[1, 1, 1, 0]], "float32")
KEPT_MEAN = X[0, :3].mean(axis=0)


def _attention():
    layer = AttentionPooling(hidden_dim=4, temperature=1e-6)
    layer.build((None, 4, 3))
    layer.attention_dense.kernel.assign(np.zeros((3, 4), "float32"))
    layer.attention_dense.bias.assign(np.full((4,), -1.0, "float32"))
    layer.context_vector.assign(np.ones((1, 4), "float32"))
    return layer  # every logit is 4 * tanh(-1) / 1e-6, about -3e6


def _weighted():
    layer = WeightedPooling(max_seq_len=4, temperature=1e-6)
    layer.build((None, 4, 3))
    layer.position_weights.assign(np.full((4,), -1.0, "float32"))
    return layer  # every logit is -1e6


@pytest.mark.parametrize("build", [_attention, _weighted], ids=["attention", "weighted"])
def test_the_masked_token_gets_no_weight(build):
    out = np.asarray(build()(X, mask=MASK))[0]
    np.testing.assert_allclose(out, KEPT_MEAN, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("build", [_attention, _weighted], ids=["attention", "weighted"])
def test_without_a_mask_the_masked_token_counts(build):
    """Liveness: the probe's masked token moves the unmasked output."""
    out = np.asarray(build()(X))[0]
    assert np.abs(out - KEPT_MEAN).max() > 1e-2


def test_a_fully_masked_top_k_max_row_returns_the_dtype_sentinel():
    out = np.asarray(SequencePooling(strategy="top_k_max", top_k=2)(X, mask=np.zeros((1, 4))))
    np.testing.assert_array_equal(out, np.full_like(out, mask_sentinel("float32")))
