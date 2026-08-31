"""`TextSimilarityHead`'s cosine branch must survive a float16 compute dtype.

Guard for `plan-2026-08-31T134711-6271592d` step 8, defect class I-3. The site is

    emb1 / ops.maximum(ops.norm(emb1, axis=-1, keepdims=True), 1e-8)

`float16(1e-8)` is exactly `0.0`, so under `mixed_float16` the floor was
`ops.maximum(norm, 0.0)` -- no floor at all -- and a zero embedding produced
`0 / 0 = NaN`. MEASURED at HEAD under `mixed_float16` with an all-zero pair:
`similarity_score == [nan nan]`.

A zero embedding is the ordinary all-padding case: the head's `_process_sequence`
chain is a norm, a dropout, and a zero-initialised-bias `Dense`, so an all-zero
(fully padded) sequence maps to an exactly zero embedding. `float32` returns
`[0. 0.]` either way, which is why 318 tests in this directory never saw it.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.heads.nlp.factory import create_nlp_head
from dl_techniques.layers.heads.nlp.task_types import NLPTaskConfig, NLPTaskType
from dl_techniques.utils.dtype_policy import stability_floor

BATCH = 2
SEQ = 4
WIDTH = 8


def _cosine_similarity_head():
    config = NLPTaskConfig(name="sim", task_type=NLPTaskType.TEXT_SIMILARITY)
    return create_nlp_head(
        config,
        input_dim=WIDTH,
        pooling_type="mean",
        similarity_function="cosine",
    )


class TestTheCosineFloorHazardIsReal:
    """Anti-vacuity: prove the literal is inert in float16 before defending."""

    def test_the_original_literal_is_exactly_zero_in_float16(self):
        assert np.float16(1e-8) == np.float16(0.0)
        assert np.float32(1e-8) > np.float32(0.0)

    def test_the_policy_floor_is_strictly_positive_in_float16(self):
        assert np.float16(stability_floor("float16", 1e-8)) > np.float16(0.0)
        assert stability_floor("float32", 1e-8) == 1e-8


class TestTheCosineBranchSurvivesAZeroEmbedding:
    """The all-padding pair: both embeddings are exactly zero."""

    def test_the_similarity_score_is_finite(self, dtype_policy):
        head = _cosine_similarity_head()
        zeros = keras.ops.zeros((BATCH, SEQ, WIDTH), dtype=head.compute_dtype)

        score = head((zeros, zeros))["similarity_score"]

        # Anti-vacuity: a head that silently ran in float32 proves nothing.
        assert keras.backend.standardize_dtype(score.dtype) == head.compute_dtype
        if dtype_policy == "mixed_float16":
            assert head.compute_dtype == "float16"

        values = keras.ops.convert_to_numpy(score)
        assert np.all(np.isfinite(values)), (
            f"emb / max(norm, eps) went non-finite under {dtype_policy}: "
            f"{values}"
        )

    def test_the_cosine_branch_is_differentiable_at_a_zero_embedding(
        self, dtype_policy
    ):
        head = _cosine_similarity_head()
        source = tf.Variable(np.zeros((BATCH, SEQ, WIDTH), np.float32))

        with tf.GradientTape() as tape:
            tape.watch(source)
            sequence = keras.ops.cast(source, head.compute_dtype)
            score = head((sequence, sequence))["similarity_score"]
            loss = keras.ops.sum(keras.ops.cast(score, "float32"))
        grad = tape.gradient(loss, source)

        assert grad is not None
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(grad))), (
            f"d(similarity)/d(inputs) went non-finite under {dtype_policy}"
        )

    def test_a_nonzero_pair_still_scores_near_one(self, dtype_policy):
        """The floor must not move the ordinary answer.

        Two identical NON-zero sequences must still score ~1.0 in every dtype.
        Without this arm the guard would pass on a head that returned a
        constant.
        """
        head = _cosine_similarity_head()
        rng = np.random.default_rng(0)
        sequence = keras.ops.cast(
            rng.normal(size=(BATCH, SEQ, WIDTH)).astype(np.float32),
            head.compute_dtype,
        )

        score = keras.ops.convert_to_numpy(
            head((sequence, sequence))["similarity_score"]
        )

        assert np.all(np.isfinite(score))
        np.testing.assert_allclose(score, np.ones_like(score), atol=2e-2)
